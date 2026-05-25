"""
LOCAL BUNDLE ADJUSTMENT — sliding-window, GTSAM GenericProjectionFactorCal3_S2.

CRITICAL RULE: BA must NEVER modify trajectory[0] through trajectory[-2].
It only returns a correction for the LAST keyframe (current frame).
The caller applies it to self.current_pose and self.trajectory[-1] only.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class LocalBundleAdjuster:
    """
    Sliding-window local bundle adjustment using GTSAM projection factors.
    Window = last `window_size` keyframes.
    Only the correction for the most recent keyframe is returned and applied.
    """

    def __init__(self, window_size: int = 8, min_keyframes: int = 3,
                 max_trans_correction: float = 0.3,
                 max_rot_correction_deg: float = 3.0):
        self.window_size = window_size
        self.min_keyframes = min_keyframes
        self.max_trans_correction = max_trans_correction
        self.max_rot_correction_deg = max_rot_correction_deg

        self.keyframes: List[Dict] = []
        self._last_kf_pose: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def is_keyframe(self, current_pose: np.ndarray) -> bool:
        """
        Returns True if this pose is far enough from the last keyframe
        to be selected as a new keyframe.
        Updates _last_kf_pose when True.
        """
        if self._last_kf_pose is None:
            self._last_kf_pose = current_pose.copy()
            return True

        # Translation distance (metres)
        trans = np.linalg.norm(current_pose[:3, 3] - self._last_kf_pose[:3, 3])

        # Rotation angle (degrees)
        rel_R = current_pose[:3, :3] @ self._last_kf_pose[:3, :3].T
        rot_deg = float(np.arccos(
            np.clip((np.trace(rel_R) - 1.0) / 2.0, -1.0, 1.0)
        )) * 180.0 / np.pi

        if trans > 0.15 or rot_deg > 1.0:
            self._last_kf_pose = current_pose.copy()
            return True

        return False

    # ------------------------------------------------------------------
    def add_keyframe(self, frame_idx: int, pose_4x4: np.ndarray,
                     points_3d_m: np.ndarray, points_2d: np.ndarray,
                     camera_matrix: np.ndarray) -> None:
        """Add keyframe to sliding window; evict oldest if full."""
        self.keyframes.append({
            'frame_idx':     frame_idx,
            'pose_4x4':      pose_4x4.copy(),
            'points_3d_m':   np.array(points_3d_m, dtype=np.float64),
            'points_2d':     np.array(points_2d,   dtype=np.float64),
            'camera_matrix': camera_matrix.copy(),
        })
        if len(self.keyframes) > self.window_size:
            self.keyframes.pop(0)

    # ------------------------------------------------------------------
    def get_pose_correction(self) -> Tuple[np.ndarray, bool]:
        """
        Run BA on the current window.
        Returns (correction_4x4, success).
        correction_4x4 is the delta to apply to the LAST keyframe pose only.
        Returns (np.eye(4), False) on any failure or rejected correction.
        """
        if len(self.keyframes) < self.min_keyframes:
            return np.eye(4), False

        try:
            import gtsam

            graph   = gtsam.NonlinearFactorGraph()
            initial = gtsam.Values()

            # Camera calibration from last keyframe
            K = self.keyframes[-1]['camera_matrix']
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            cal = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.5)
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
            )

            # Insert pose variables; anchor first keyframe with prior
            for i, kf in enumerate(self.keyframes):
                R = kf['pose_4x4'][:3, :3]
                t = kf['pose_4x4'][:3, 3]
                gtsam_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*t))
                initial.insert(gtsam.symbol_shorthand.X(i), gtsam_pose)
                if i == 0:
                    graph.push_back(gtsam.PriorFactorPose3(
                        gtsam.symbol_shorthand.X(0), gtsam_pose, prior_noise
                    ))

            # Add projection factors for each keyframe's observations
            for i, kf in enumerate(self.keyframes):
                pts3 = kf['points_3d_m']   # Nx3, metres
                pts2 = kf['points_2d']     # Nx2, pixels
                n = min(len(pts3), len(pts2), 200)  # cap at 200 pts per keyframe
                for j in range(n):
                    lm_key = gtsam.symbol_shorthand.L(i * 1000 + j)
                    pt3 = gtsam.Point3(float(pts3[j, 0]),
                                       float(pts3[j, 1]),
                                       float(pts3[j, 2]))
                    initial.insert(lm_key, pt3)
                    measured = gtsam.Point2(float(pts2[j, 0]), float(pts2[j, 1]))
                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        measured, measurement_noise,
                        gtsam.symbol_shorthand.X(i), lm_key, cal
                    )
                    graph.push_back(factor)

            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(30)
            params.setVerbosity('SILENT')
            result = gtsam.LevenbergMarquardtOptimizer(
                graph, initial, params
            ).optimize()

            # Extract correction for LAST keyframe only
            last_idx = len(self.keyframes) - 1
            opt_pose = result.atPose3(gtsam.symbol_shorthand.X(last_idx))
            opt_mat = np.eye(4)
            opt_mat[:3, :3] = opt_pose.rotation().matrix()
            opt_mat[:3, 3]  = opt_pose.translation()

            orig_mat   = self.keyframes[-1]['pose_4x4']
            correction = opt_mat @ np.linalg.inv(orig_mat)

            # Sanity check — reject large corrections
            trans_c = float(np.linalg.norm(correction[:3, 3]))
            rot_c   = float(np.arccos(np.clip(
                (np.trace(correction[:3, :3]) - 1.0) / 2.0, -1.0, 1.0
            ))) * 180.0 / np.pi

            if trans_c > self.max_trans_correction or rot_c > self.max_rot_correction_deg:
                print(f"   [BA] Correction rejected: t={trans_c:.3f}m r={rot_c:.2f}°")
                return np.eye(4), False

            print(f"   [BA] Correction applied: t={trans_c*1000:.1f}mm r={rot_c:.3f}°")
            return correction, True

        except Exception as e:
            print(f"   [BA] Failed: {e}")
            return np.eye(4), False
