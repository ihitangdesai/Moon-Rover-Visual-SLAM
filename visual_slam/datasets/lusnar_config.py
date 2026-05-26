"""
LuSNAR dataset configuration.
Source: LuSNAR paper (arXiv:2407.06512), Fig. 5 and Table IV.

Coordinate frames (Fig. 5):
  Rover/world frame : X=forward, Y=right,   Z=down
  Camera frame      : Z=forward, X=right,   Y=down
  Camera pitch      : -20 degrees (looking downward)
  Left camera mount : (1.0, -0.155, -1.5) in rover frame
"""

import numpy as np


def get_left_camera_extrinsic() -> np.ndarray:
    """
    Returns T_cam_to_rover (4x4) for the LuSNAR left stereo camera.
    Use this as T_sensor_to_world when constructing the SLAM system
    on any LuSNAR sequence.

    Derivation (Fig. 5):
      Step 1: axis permutation — camera(Z=fwd, X=right, Y=down)
                                  → rover(X=fwd, Y=right, Z=down)
              rover_X = cam_Z
              rover_Y = cam_X
              rover_Z = cam_Y

      Step 2: -20° pitch around rover Y axis (camera looks downward)
    """
    # Step 1: axis permutation camera → rover
    R_axes = np.array([[0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.]], dtype=np.float64)

    # Step 2: -20° pitch around rover Y axis
    pitch = np.radians(-20.0)
    R_pitch = np.array([[ np.cos(pitch), 0., np.sin(pitch)],
                        [ 0.,            1., 0.            ],
                        [-np.sin(pitch), 0., np.cos(pitch)]], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_pitch @ R_axes
    T[:3, 3]  = np.array([1.0, -0.155, -1.5])  # left camera mount in rover frame
    return T


# ── CAHV camera parameters from Table IV ──────────────────────────────
FOCAL_LENGTH_PX = 610.17784
IMAGE_WIDTH     = 1024
IMAGE_HEIGHT    = 1024
BASELINE_MM     = 310.0
FOV_DEG         = 80.0

LEFT_CAHV = {
    'C': [0.0,        0.0, 0.0],
    'A': [0.0,        0.0, 1.0],
    'H': [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH  / 2.0],
    'V': [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0],
}

RIGHT_CAHV = {
    'C': [BASELINE_MM, 0.0, 0.0],
    'A': [0.0,         0.0, 1.0],
    'H': [FOCAL_LENGTH_PX, 0.0, IMAGE_WIDTH  / 2.0],
    'V': [0.0, FOCAL_LENGTH_PX, IMAGE_HEIGHT / 2.0],
}


# ══════════════════════════════════════════════════════════════════════
# OFFLINE EXTRINSIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════

def calibrate_extrinsic(est_trajectory: list, gt_trajectory: list,
                        motion_frames: int = 50) -> np.ndarray:
    """
    Solve for the optimal rotation R that minimises alignment error between
    estimated and GT motion directions over the first `motion_frames` moving
    frames.

    Strategy:
      - Collect per-frame translation deltas from both EST and GT.
      - Skip stationary frames (step < 0.01m in GT).
      - Stack the direction vectors and solve R = argmin ||R @ est_dirs - gt_dirs||
        using SVD (same as Kabsch/Umeyama for direction alignment).
      - Decompose the resulting R into roll/pitch/yaw and print them.
      - Return the full 4x4 T_cam_to_rover with mount translation from Fig. 5.

    Usage (run once offline, then hardcode the printed angles):
        from visual_slam.datasets.lusnar_config import calibrate_extrinsic
        T = calibrate_extrinsic(slam.trajectory, gt_poses)
    """
    from scipy.spatial.transform import Rotation as _Rot

    est_pos = np.array([p[:3, 3] for p in est_trajectory])
    gt_pos  = np.array([p[:3, 3] for p in gt_trajectory[:len(est_trajectory)]])

    # Collect motion direction pairs (skip stationary GT frames)
    est_dirs, gt_dirs = [], []
    collected = 0
    for i in range(1, len(est_pos)):
        gt_step  = gt_pos[i]  - gt_pos[i - 1]
        est_step = est_pos[i] - est_pos[i - 1]
        gt_norm  = np.linalg.norm(gt_step)
        est_norm = np.linalg.norm(est_step)
        if gt_norm < 0.01 or est_norm < 0.01:
            continue
        gt_dirs.append(gt_step   / gt_norm)
        est_dirs.append(est_step / est_norm)
        collected += 1
        if collected >= motion_frames:
            break

    if collected < 5:
        print(f"[calibrate_extrinsic] Only {collected} motion frames — not enough. "
              f"Returning paper-spec extrinsic.")
        return get_left_camera_extrinsic()

    est_dirs = np.array(est_dirs)   # N x 3
    gt_dirs  = np.array(gt_dirs)    # N x 3

    # Kabsch SVD: find R such that R @ est_dirs[i] ≈ gt_dirs[i]
    H   = est_dirs.T @ gt_dirs      # 3x3
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # Decompose into Euler angles for inspection
    euler = _Rot.from_matrix(R_opt).as_euler('zyx', degrees=True)
    yaw, pitch, roll = euler[0], euler[1], euler[2]

    print("=" * 60)
    print("[calibrate_extrinsic] Optimal rotation found:")
    print(f"  Roll  (X): {roll:.4f} deg  → np.radians({roll:.4f})")
    print(f"  Pitch (Y): {pitch:.4f} deg  → np.radians({pitch:.4f})")
    print(f"  Yaw   (Z): {yaw:.4f} deg  → np.radians({yaw:.4f})")
    print(f"  Rotation matrix:\n{np.round(R_opt, 6)}")
    print(f"  Used {collected} motion frames for calibration.")
    print("  → Hardcode these angles in get_left_camera_extrinsic()")
    print("=" * 60)

    # Build full 4x4 T with paper-spec mount translation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_opt
    T[:3, 3]  = np.array([1.0, -0.155, -1.5])
    return T
