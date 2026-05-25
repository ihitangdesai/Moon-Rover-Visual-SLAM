"""
MAIN SLAM SYSTEM WITH PNP POSE ESTIMATION - ImprovedVisualSLAM class.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Tuple, List, Dict, Optional

from visual_slam.data_structures import CAHVCamera
from visual_slam.config import DatasetAdaptiveParameters
from visual_slam.triangulation import RobustTriangulator
from visual_slam.correspondence_validation import Enhanced3DCorrespondenceFinder
from visual_slam.temporal_matching import SmartTemporalMatcher
from visual_slam.feature_extraction import OptimizedASIFTMatcher
from visual_slam.visualisation import SLAMVisualizationSystem


class ImprovedVisualSLAM:
    """Improved Visual SLAM System with working parameters and PnP comparison"""

    def __init__(self, left_cahv: Dict, right_cahv: Dict, show_lines: bool = False,
                 output_dir=None, initial_pose: np.ndarray = None):
        """Initialize improved Visual SLAM system."""

        self.left_camera = CAHVCamera(
            C=left_cahv['C'], A=left_cahv['A'],
            H=left_cahv['H'], V=left_cahv['V']
        )
        self.right_camera = CAHVCamera(
            C=right_cahv['C'], A=right_cahv['A'],
            H=right_cahv['H'], V=right_cahv['V']
        )

        self.adaptive_params = DatasetAdaptiveParameters(left_cahv, right_cahv)

        self.triangulator = RobustTriangulator(self.left_camera, self.right_camera, self.adaptive_params)
        self.correspondence_finder = Enhanced3DCorrespondenceFinder(self.adaptive_params)
        self.temporal_matcher = SmartTemporalMatcher(self.adaptive_params)
        self.asift = OptimizedASIFTMatcher(self.adaptive_params)

        # Custom visualization system
        self.visualization_system = SLAMVisualizationSystem(show_lines, output_dir=output_dir)

        self.show_lines = show_lines

        # Initialize pose — use provided initial_pose (world frame) if given,
        # otherwise start at identity (body frame = world frame at t=0).
        if initial_pose is not None:
            self.current_pose = np.array(initial_pose, dtype=np.float64)
        else:
            self.current_pose = np.eye(4)
            self.current_pose[:3, 3] = np.array(left_cahv['C']) / 1000.0
        self.trajectory = [self.current_pose.copy()]
        self.frame_results = []
        self._motion_started = False

        # Camera intrinsics for PnP
        self.camera_matrix = self.left_camera.intrinsics_from_cahv()
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

        print("Working Visual SLAM Initialized")
        print(f"   Working parameters: Ready")
        print(f"   Enhanced 3D correspondence: Ready")
        print(f"   Smart temporal matching: Ready")
        print(f"   Robust triangulation: Ready")
        print(f"   Custom visualization: Ready")
        print(f"   PnP pose estimation: Ready")
        print(f"   Show connection lines: {show_lines}")
        print(f"   Initial pose (meters): [{self.current_pose[0,3]:.3f}, {self.current_pose[1,3]:.3f}, {self.current_pose[2,3]:.3f}]")

    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract optimized ASIFT features."""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.asift.affine_detect(gray)
        return keypoints, descriptors

    def match_stereo_features(self, left_kp: List, left_desc: np.ndarray,
                             right_kp: List, right_desc: np.ndarray) -> List[Tuple[int, int]]:
        """Match stereo features with proper epipolar constraint."""
        matches = self.asift.match_features(left_desc, right_desc)

        print(f"     Raw stereo matches: {len(matches)}")

        epipolar_threshold = self.adaptive_params.get_param('stereo_epipolar_threshold')
        good_matches = []
        epipolar_violations = 0

        for match in matches:
            left_pt = left_kp[match.queryIdx].pt
            right_pt = right_kp[match.trainIdx].pt

            y_diff = abs(left_pt[1] - right_pt[1])
            x_diff = left_pt[0] - right_pt[0]  # Should be positive for proper stereo

            if y_diff < epipolar_threshold and x_diff > 0:
                good_matches.append((match.queryIdx, match.trainIdx))
            else:
                epipolar_violations += 1

        print(f"     Epipolar violations: {epipolar_violations}")
        print(f"     Good stereo matches: {len(good_matches)}")

        return good_matches

    def triangulate_stereo_points(self, left_kp: List, right_kp: List,
                                 matches: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Triangulate 3D points using robust triangulation."""
        if not matches:
            return [], []

        left_points = np.array([left_kp[left_idx].pt for left_idx, right_idx in matches])
        right_points = np.array([right_kp[right_idx].pt for left_idx, right_idx in matches])

        points_3d, valid_indices, errors = self.triangulator.triangulate_multiple_points(
            left_points, right_points
        )

        valid_matches = [matches[i] for i in valid_indices]

        print(f"     Robust triangulation: {len(valid_indices)}/{len(matches)} successful")
        if len(errors) > 0:
            print(f"     Average reprojection error: {np.mean(errors):.2f} pixels")

        return points_3d, valid_matches

    def estimate_pose_change_robust(self, points_3d_t: List[np.ndarray],
                                   points_3d_t1: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Robust pose estimation with correct transformation direction."""
        if len(points_3d_t) < 3 or len(points_3d_t1) < 3:
            return np.zeros(3), np.eye(3), {'status': 'insufficient_points'}

        points_src = np.array(points_3d_t, dtype=np.float64) / 1000.0  # Convert to meters
        points_tgt = np.array(points_3d_t1, dtype=np.float64) / 1000.0  # Convert to meters

        threshold = float(self.adaptive_params.get_param('ransac_threshold_mm') / 1000.0)  # Convert to meters
        max_iterations = int(self.adaptive_params.get_param('ransac_iterations'))

        best_inliers = np.array([], dtype=bool)
        best_transform = np.eye(4)

        for iteration in range(max_iterations):
            try:
                if len(points_src) > 3:
                    indices = np.random.choice(len(points_src), 3, replace=False)
                else:
                    indices = np.arange(len(points_src))

                sample_src = points_src[indices]
                sample_tgt = points_tgt[indices]

                # Estimate transformation from t to t+1
                transform = self._estimate_transformation_kabsch(sample_src, sample_tgt)

                R = transform[:3, :3]
                t = transform[:3, 3]

                transformed_points = (R @ points_src.T).T + t
                distances = np.linalg.norm(transformed_points - points_tgt, axis=1)

                inliers = distances < threshold
                n_inliers = int(np.sum(inliers))

                if n_inliers > len(best_inliers) or (len(best_inliers) == 0):
                    best_inliers = inliers.copy()
                    best_transform = transform.copy()

            except Exception as e:
                continue

        if np.sum(best_inliers) >= 3:
            try:
                inlier_src = points_src[best_inliers]
                inlier_tgt = points_tgt[best_inliers]
                best_transform = self._estimate_transformation_kabsch(inlier_src, inlier_tgt)
            except Exception as e:
                pass

        R = best_transform[:3, :3]
        t = best_transform[:3, 3]

        num_inliers = int(np.sum(best_inliers))
        inlier_ratio = float(num_inliers / len(points_src))

        # Calculate mean error for inliers
        if np.sum(best_inliers) > 0:
            inlier_src = points_src[best_inliers]
            inlier_tgt = points_tgt[best_inliers]
            transformed_inliers = (R @ inlier_src.T).T + t
            inlier_errors = np.linalg.norm(transformed_inliers - inlier_tgt, axis=1)
            mean_error = float(np.mean(inlier_errors))
        else:
            mean_error = float('inf')

        summary = {
            'status': 'success',
            'method': 'kabsch_ransac',
            'num_inliers': num_inliers,
            'inlier_ratio': inlier_ratio,
            'mean_error': mean_error
        }

        return t, R, summary

    def estimate_pose_change_pnp(self, points_3d_t: List[np.ndarray],
                                correspondence_info: List[Dict],
                                left_kp_t1: List) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """PnP-based pose estimation for comparison."""

        if len(correspondence_info) < 4:
            return np.zeros(3), np.eye(3), {'status': 'insufficient_points_pnp'}

        try:
            # Build correctly-paired 3D-to-2D correspondences for PnP:
            # object_points[i] is the 3D position of feature i in the t frame (in metres),
            # image_points[i] is where that same feature appears in the t1 left image.
            object_points = np.array(points_3d_t, dtype=np.float64) / 1000.0
            image_points = np.array([info['kp_t1_pt'] for info in correspondence_info],
                                    dtype=np.float64)

            if len(object_points) < 4:
                return np.zeros(3), np.eye(3), {'status': 'insufficient_points_pnp'}

            # Solve PnP
            success, rvec, tvec, inliers = cv.solvePnPRansac(
                object_points.reshape(-1, 1, 3),
                image_points.reshape(-1, 1, 2),
                self.camera_matrix,
                self.dist_coeffs,
                iterationsCount=1000,
                reprojectionError=2.0,
                confidence=0.99
            )

            if not success:
                return np.zeros(3), np.eye(3), {'status': 'pnp_failed'}

            # Convert rotation vector to matrix
            R, _ = cv.Rodrigues(rvec)
            t = tvec.flatten()

            # Calculate relative transformation (camera motion from t to t+1)
            # PnP gives camera pose, we need the relative transformation
            current_cam_pose = np.eye(4)
            current_cam_pose[:3, :3] = R
            current_cam_pose[:3, 3] = t

            # For relative motion, we need to invert this
            relative_transform = np.linalg.inv(current_cam_pose)
            rel_R = relative_transform[:3, :3]
            rel_t = relative_transform[:3, 3]

            num_inliers = len(inliers) if inliers is not None else 0
            inlier_ratio = num_inliers / len(object_points) if len(object_points) > 0 else 0.0

            # Calculate reprojection error
            projected_points, _ = cv.projectPoints(
                object_points.reshape(-1, 1, 3), rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            reprojection_errors = np.linalg.norm(image_points - projected_points, axis=1)
            mean_error = float(np.mean(reprojection_errors))

            summary = {
                'status': 'success',
                'method': 'pnp_ransac',
                'num_inliers': int(num_inliers),
                'inlier_ratio': float(inlier_ratio),
                'mean_error': mean_error,
                'reprojection_error': mean_error
            }

            return rel_t, rel_R, summary

        except Exception as e:
            return np.zeros(3), np.eye(3), {'status': f'pnp_error: {str(e)}'}

    def _estimate_transformation_kabsch(self, points_src, points_tgt):
        """Kabsch algorithm for optimal 3D transformation."""
        try:
            points_src = np.array(points_src, dtype=np.float64)
            points_tgt = np.array(points_tgt, dtype=np.float64)

            centroid_src = np.mean(points_src, axis=0)
            centroid_tgt = np.mean(points_tgt, axis=0)

            src_centered = points_src - centroid_src
            tgt_centered = points_tgt - centroid_tgt

            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Handle reflection case
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = centroid_tgt - R @ centroid_src

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            return T

        except Exception as e:
            return np.eye(4)

    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles."""
        try:
            sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            else:
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0

            return np.array([x, y, z])
        except Exception as e:
            return np.zeros(3)

    def process_frame_pair(self, left_t: np.ndarray, right_t: np.ndarray,
                          left_t1: np.ndarray, right_t1: np.ndarray,
                          frame_info: Dict = None) -> Dict:
        """Process frame pair with working parameters and PnP comparison."""
        frame_desc = f"frames {frame_info.get('frame_t_number', '?')} -> {frame_info.get('frame_t1_number', '?')}" if frame_info else "frame pair"
        print(f"Processing {frame_desc} with Working Visual SLAM...")
        start_time = time.time()

        try:
            print("1. Extracting ASIFT features...")
            left_kp_t, left_desc_t = self.extract_features(left_t)
            right_kp_t, right_desc_t = self.extract_features(right_t)
            left_kp_t1, left_desc_t1 = self.extract_features(left_t1)
            right_kp_t1, right_desc_t1 = self.extract_features(right_t1)

            print(f"   Features: L_t={len(left_kp_t)}, R_t={len(right_kp_t)}, L_t1={len(left_kp_t1)}, R_t1={len(right_kp_t1)}")

            print("2. Matching stereo features...")
            stereo_matches_t = self.match_stereo_features(left_kp_t, left_desc_t, right_kp_t, right_desc_t)
            stereo_matches_t1 = self.match_stereo_features(left_kp_t1, left_desc_t1, right_kp_t1, right_desc_t1)

            # Save stereo matches visualization
            if frame_info:
                self.visualization_system.save_stereo_matches(
                    left_t, right_t, left_kp_t, right_kp_t, stereo_matches_t,
                    frame_info.get('frame_t_number', 0)
                )
                self.visualization_system.save_stereo_matches(
                    left_t1, right_t1, left_kp_t1, right_kp_t1, stereo_matches_t1,
                    frame_info.get('frame_t1_number', 1)
                )

            print("3. Robust triangulation...")
            points_3d_t, valid_matches_t = self.triangulate_stereo_points(left_kp_t, right_kp_t, stereo_matches_t)
            points_3d_t1, valid_matches_t1 = self.triangulate_stereo_points(left_kp_t1, right_kp_t1, stereo_matches_t1)

            print("4. Smart temporal matching...")
            temporal_matches = self.temporal_matcher.match_temporal_features_smart(
                left_kp_t, left_desc_t, left_kp_t1, left_desc_t1
            )

            # Save temporal matches visualization
            if frame_info:
                self.visualization_system.save_temporal_matches(
                    left_t, left_t1, left_kp_t, left_kp_t1, temporal_matches,
                    frame_info.get('frame_t_number', 0),
                    frame_info.get('frame_t1_number', 1)
                )

            print("5. 3D correspondence finding...")
            try:
                correspondence_result = self.correspondence_finder.find_3d_correspondences(
                    temporal_matches, left_kp_t, left_desc_t, left_kp_t1, left_desc_t1,
                    points_3d_t, valid_matches_t, points_3d_t1, valid_matches_t1
                )

                if correspondence_result is None or len(correspondence_result) != 3:
                    corresponding_3d_t, corresponding_3d_t1, correspondence_info = [], [], []
                else:
                    corresponding_3d_t, corresponding_3d_t1, correspondence_info = correspondence_result

            except Exception as e:
                corresponding_3d_t, corresponding_3d_t1, correspondence_info = [], [], []

            if len(corresponding_3d_t) >= 3:
                print("6. Pose estimation comparison...")

                # Method 1: Kabsch + RANSAC
                try:
                    translation_kabsch, rotation_kabsch, pose_summary_kabsch = self.estimate_pose_change_robust(
                        corresponding_3d_t, corresponding_3d_t1
                    )
                except Exception as e:
                    translation_kabsch, rotation_kabsch = np.zeros(3), np.eye(3)
                    pose_summary_kabsch = {'status': f'kabsch_error: {str(e)}'}

                # Method 2: PnP
                try:
                    translation_pnp, rotation_pnp, pose_summary_pnp = self.estimate_pose_change_pnp(
                        corresponding_3d_t, correspondence_info, left_kp_t1
                    )
                except Exception as e:
                    translation_pnp, rotation_pnp = np.zeros(3), np.eye(3)
                    pose_summary_pnp = {'status': f'pnp_error: {str(e)}'}

                # Print comparison
                print("   POSE ESTIMATION COMPARISON:")
                print(f"   Kabsch+RANSAC:")
                print(f"      Translation: [{translation_kabsch[0]:.4f}, {translation_kabsch[1]:.4f}, {translation_kabsch[2]:.4f}] m")
                print(f"      Translation magnitude: {np.linalg.norm(translation_kabsch):.4f} m")
                print(f"      Rotation angle: {np.arccos(np.clip((np.trace(rotation_kabsch) - 1) / 2, -1, 1)) * 180 / np.pi:.2f} ")
                print(f"      Status: {pose_summary_kabsch.get('status', 'unknown')}")
                print(f"      Inliers: {pose_summary_kabsch.get('num_inliers', 0)}")

                print(f"   PnP+RANSAC:")
                print(f"      Translation: [{translation_pnp[0]:.4f}, {translation_pnp[1]:.4f}, {translation_pnp[2]:.4f}] m")
                print(f"      Translation magnitude: {np.linalg.norm(translation_pnp):.4f} m")
                print(f"      Rotation angle: {np.arccos(np.clip((np.trace(rotation_pnp) - 1) / 2, -1, 1)) * 180 / np.pi:.2f} ")
                print(f"      Status: {pose_summary_pnp.get('status', 'unknown')}")
                print(f"      Inliers: {pose_summary_pnp.get('num_inliers', 0)}")

                # Choose the better method based on inlier count and status
                use_pnp = (pose_summary_pnp.get('status') == 'success' and
                          pose_summary_pnp.get('num_inliers', 0) > pose_summary_kabsch.get('num_inliers', 0) * 0.8)

                if use_pnp:
                    print("   ? Using PnP result (better inlier count)")
                    translation, rotation, pose_summary = translation_pnp, rotation_pnp, pose_summary_pnp
                    chosen_method = "PnP+RANSAC"
                else:
                    print("   ? Using Kabsch result (more reliable)")
                    translation, rotation, pose_summary = translation_kabsch, rotation_kabsch, pose_summary_kabsch
                    chosen_method = "Kabsch+RANSAC"

                # ── CAHV-frame → rover body frame ─────────────────────────────
                # CAHV optical axis A=[0,0,1] is body +Z; rover forward is body +X.
                # Kabsch gives scene motion → invert to get camera motion.
                # PnP already gives camera motion (inverted internally).
                _R_cahv_to_body = np.array([[0., 0., 1.],
                                            [0., 1., 0.],
                                            [-1., 0., 0.]])
                _T_cahv = np.eye(4)
                _T_cahv[:3, :3] = rotation
                _T_cahv[:3, 3] = translation
                if chosen_method == "Kabsch+RANSAC":
                    _T_cahv = np.linalg.inv(_T_cahv)
                transform = np.eye(4)
                transform[:3, :3] = _R_cahv_to_body @ _T_cahv[:3, :3] @ _R_cahv_to_body.T
                transform[:3, 3] = _R_cahv_to_body @ _T_cahv[:3, 3]

                translation_norm = float(np.linalg.norm(translation))
                rotation_angle = float(
                    np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)) * 180 / np.pi
                )

                # ── Fix 1: stationary startup detection ───────────────────────
                if not self._motion_started:
                    if translation_norm > 0.05 or rotation_angle > 1.0:
                        self._motion_started = True
                    if translation_norm < 0.01 and rotation_angle < 0.5:
                        chosen_method = "skipped_stationary"
                        print(f"   [INFO] Stationary frame skipped "
                              f"(t={translation_norm*1000:.1f}mm, r={rotation_angle:.3f}deg)")
                    else:
                        self.current_pose = self.current_pose @ transform
                else:
                    self.current_pose = self.current_pose @ transform

                self.trajectory.append(self.current_pose.copy())
                is_keyframe = False  # placeholder — keyframe manager removed

                try:
                    rotation_vector, _ = cv.Rodrigues(rotation.astype(np.float64))
                    rotation_vector = rotation_vector.flatten()
                    euler_angles = self._rotation_matrix_to_euler_angles(rotation)
                except:
                    rotation_vector = np.zeros(3)
                    euler_angles = np.zeros(3)

                processing_time = time.time() - start_time

                results = {
                    'success': True,
                    'frame_info': frame_info,
                    'chosen_method': chosen_method,
                    'is_keyframe': is_keyframe,
                    'pose_comparison': {
                        'kabsch': {
                            'translation': translation_kabsch,
                            'rotation': rotation_kabsch,
                            'summary': pose_summary_kabsch
                        },
                        'pnp': {
                            'translation': translation_pnp,
                            'rotation': rotation_pnp,
                            'summary': pose_summary_pnp
                        }
                    },
                    'translation': translation,
                    'translation_components_m': {
                        'x': float(translation[0]),
                        'y': float(translation[1]),
                        'z': float(translation[2])
                    },
                    'rotation': rotation,
                    'rotation_vector': rotation_vector,
                    'rotation_vector_deg': rotation_vector * 180 / np.pi,
                    'euler_angles_deg': {
                        'roll': float(euler_angles[0] * 180 / np.pi),
                        'pitch': float(euler_angles[1] * 180 / np.pi),
                        'yaw': float(euler_angles[2] * 180 / np.pi)
                    },
                    'transformation_matrix': transform,
                    'current_pose': self.current_pose,
                    'translation_magnitude_m': translation_norm,
                    'rotation_angle_deg': rotation_angle,
                    'num_3d_correspondences': len(corresponding_3d_t),
                    'pose_estimation_summary': pose_summary,
                    'processing_time': processing_time,
                    'num_features': len(left_kp_t1),
                    'points_3d': points_3d_t1,
                    'left_keypoints': left_kp_t1,
                    'right_keypoints': right_kp_t1,
                    'left_image': left_t1,
                    'right_image': right_t1,
                    'trajectory': self.trajectory.copy(),
                    'all_poses': {i: pose for i, pose in enumerate(self.trajectory)},
                    'current_pose_id': len(self.trajectory) - 1,
                    'edges': [],
                    'statistics': {
                        'num_features': len(left_kp_t1),
                        'num_3d_correspondences': len(corresponding_3d_t),
                        'translation_magnitude_m': translation_norm,
                        'rotation_angle_deg': rotation_angle
                    }
                }

                print(f"Pose estimation successful using {chosen_method}!")
                print(f"   Translation magnitude: {translation_norm:.3f} m")
                print(f"   Rotation angle: {rotation_angle:.2f} degrees")
                print(f"   3D correspondences: {len(corresponding_3d_t)}")
                print(f"   Current position (m): [{self.current_pose[0,3]:.3f}, {self.current_pose[1,3]:.3f}, {self.current_pose[2,3]:.3f}]")
                print(f"   Processing time: {processing_time:.2f} seconds")

            else:
                results = {
                    'success': False,
                    'frame_info': frame_info,
                    'error': f'Insufficient 3D correspondences: {len(corresponding_3d_t)} < 3',
                    'num_3d_correspondences': len(corresponding_3d_t)
                }
                print(f"Pose estimation failed: {results['error']}")

            return results

        except Exception as e:
            print(f"   Unexpected error: {e}")
            return {
                'success': False,
                'frame_info': frame_info,
                'error': f'Unexpected error: {str(e)}'
            }

    def get_trajectory_summary(self) -> Dict:
        """Get comprehensive trajectory summary."""
        try:
            if len(self.trajectory) < 1:
                return {
                    'error': 'No trajectory data available',
                    'num_poses': 0,
                    'total_distance_m': 0.0
                }

            trajectory = np.array([pose[:3, 3] for pose in self.trajectory])

            if len(trajectory) < 2:
                return {
                    'num_poses': len(self.trajectory),
                    'total_distance_m': 0.0,
                    'start_position': trajectory[0] if len(trajectory) > 0 else np.zeros(3),
                    'end_position': trajectory[-1] if len(trajectory) > 0 else np.zeros(3),
                    'trajectory_points': trajectory
                }

            try:
                distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                total_distance = float(np.sum(distances))
            except Exception as e:
                total_distance = 0.0

            displacement = trajectory[-1] - trajectory[0]
            displacement_magnitude = float(np.linalg.norm(displacement))

            summary = {
                'num_poses': len(self.trajectory),
                'total_distance_m': total_distance,
                'start_position': trajectory[0],
                'end_position': trajectory[-1],
                'displacement': displacement,
                'displacement_magnitude_m': displacement_magnitude,
                'trajectory_points': trajectory
            }

            return summary

        except Exception as e:
            return {
                'error': f'Trajectory summary failed: {str(e)}',
                'num_poses': len(self.trajectory) if hasattr(self, 'trajectory') else 0,
                'total_distance_m': 0.0
            }

    def create_interactive_3d_trajectory_plot(self, show_plot: bool = True, figsize: Tuple[int, int] = (12, 9)) -> Optional[plt.Figure]:
        """Create interactive 3D trajectory plot using matplotlib."""
        print("Creating interactive 3D trajectory plot...")

        if len(self.trajectory) == 0:
            print("No trajectory data available for 3D visualization")
            return None

        if len(self.trajectory) < 2:
            print(f"Insufficient trajectory data: {len(self.trajectory)} poses (need at least 2)")
            return None

        # Positions are already in meters
        positions = np.array([pose[:3, 3] for pose in self.trajectory])

        print(f"Position range:")
        print(f"   X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] meters")
        print(f"   Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] meters")
        print(f"   Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] meters")

        # Create figure with interactive 3D subplot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory line
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
               'b-', linewidth=2, alpha=0.8, label='Rover Path')

        # Plot poses as points with different colors for start/end
        colors = ['green'] + ['blue'] * (len(positions) - 2) + ['red'] if len(positions) > 1 else ['green']
        sizes = [100] + [30] * (len(positions) - 2) + [100] if len(positions) > 1 else [100]

        for i, (pos, color, size) in enumerate(zip(positions, colors, sizes)):
            if i == 0:
                ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker='^',
                          label='Start', edgecolors='black', linewidth=1, alpha=0.9)
            elif i == len(positions) - 1:
                ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker='v',
                          label='End', edgecolors='black', linewidth=1, alpha=0.9)
            else:
                ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=0.6)

        # Add loop closure points if any
        if hasattr(self.visualization_system, 'loop_closure_points') and self.visualization_system.loop_closure_points:
            lc_positions = np.array([lc['position'] for lc in self.visualization_system.loop_closure_points])
            ax.scatter(lc_positions[:, 0], lc_positions[:, 1], lc_positions[:, 2],
                      c='orange', s=150, marker='*', label='Loop Closures',
                      edgecolors='black', linewidth=1, alpha=0.9)

        # Calculate trajectory statistics
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
            max_distance_step = np.max(distances)
            avg_distance_step = np.mean(distances)
        else:
            total_distance = 0
            max_distance_step = 0
            avg_distance_step = 0

        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'Working Visual SLAM 3D Trajectory\n'
                    f'Total Distance: {total_distance:.2f}m | '
                    f'Poses: {len(positions)} | '
                    f'Max Step: {max_distance_step:.3f}m')

        # Add legend and grid
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio if possible
        try:
            max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                                positions[:, 1].max()-positions[:, 1].min(),
                                positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0
            mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
            mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
            mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        except:
            pass

        # Add instructions for interaction
        fig.text(0.02, 0.02, "Interactive controls available - move the plot!\n"
                            "Mouse: Left=Rotate, Right=Zoom, Middle=Pan",
                fontsize=9, alpha=0.7)

        # Show plot with interactive controls
        if show_plot:
            print("Attempting to display interactive 3D plot...")
            print("   Mouse controls:")
            print("   Left click + drag: Rotate view")
            print("   Right click + drag: Zoom in/out")
            print("   Middle click + drag: Pan view")

            try:
                plt.ion()  # Turn on interactive mode
                plt.show()
                print("Interactive plot displayed - check for new window!")

                # Keep the plot alive
                input("Press Enter after viewing the 3D plot to continue...")

            except Exception as e:
                print(f"Failed to display interactive plot: {e}")

        return fig
