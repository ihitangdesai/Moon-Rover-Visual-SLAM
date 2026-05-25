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
