"""
Data structures (@dataclass types) used throughout the Visual SLAM system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlaceDescriptor:
    """Container for place descriptor data."""
    place_id: int
    descriptors: np.ndarray
    pose: np.ndarray
    timestamp: float
    feature_count: int


@dataclass
class LoopClosureCandidate:
    """Container for loop closure candidate data."""
    current_frame_id: int
    matched_frame_id: int
    similarity_score: float
    spatial_distance: float
    temporal_distance: int
    relative_pose: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class PoseGraphEdge:
    """Container for pose graph edge data."""
    from_id: int
    to_id: int
    relative_pose: np.ndarray
    information: np.ndarray
    edge_type: str


@dataclass
class CAHVCamera:
    """CAHV Camera Model with proper coordinate handling."""
    C: np.ndarray
    A: np.ndarray
    H: np.ndarray
    V: np.ndarray

    def intrinsics_from_cahv(self) -> np.ndarray:
        """Extract pinhole camera intrinsics from CAHV model."""
        fx = np.linalg.norm(np.cross(self.A, self.H)) / np.linalg.norm(self.A)
        fy = np.linalg.norm(np.cross(self.A, self.V)) / np.linalg.norm(self.A)
        cx = np.dot(self.H, self.A) / np.dot(self.A, self.A)
        cy = np.dot(self.V, self.A) / np.dot(self.A, self.A)
        return np.array([[fx, 0,  cx],
                        [0,  fy, cy],
                        [0,  0,   1]], dtype=float)

    def cahv_to_gtsam_pose(self) -> 'gtsam.Pose3':
        """Convert CAHV parameters to GTSAM Pose3."""
        from visual_slam import GTSAM_AVAILABLE
        import gtsam
        if not GTSAM_AVAILABLE:
            raise ImportError("GTSAM not available for pose conversion")

        # Convert to meters for GTSAM
        position = gtsam.Point3(self.C[0]/1000.0, self.C[1]/1000.0, self.C[2]/1000.0)
        A_norm = self.A / np.linalg.norm(self.A)

        if abs(np.dot(A_norm, [0, 0, 1])) > 0.9:
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = np.cross([0, 0, 1], A_norm)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(A_norm, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = A_norm

        R_world_to_camera = np.column_stack([x_axis, y_axis, z_axis])
        rotation = gtsam.Rot3(R_world_to_camera)

        return gtsam.Pose3(rotation, position)

    def to_gtsam_camera(self) -> 'gtsam.PinholeCameraCal3_S2':
        """Convert CAHV to GTSAM PinholeCamera."""
        from visual_slam import GTSAM_AVAILABLE
        import gtsam
        if not GTSAM_AVAILABLE:
            raise ImportError("GTSAM not available for camera conversion")

        K = self.intrinsics_from_cahv()
        cal = gtsam.Cal3_S2(K[0,0], K[1,1], 0.0, K[0,2], K[1,2])
        pose = self.cahv_to_gtsam_pose()

        return gtsam.PinholeCameraCal3_S2(pose, cal)

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D point to image coordinates using CAHV model."""
        w = point_3d - self.C
        depth = np.dot(w, self.A)

        if depth <= 1e-6:
            return np.array([np.nan, np.nan])

        u = np.dot(w, self.H) / depth
        v = np.dot(w, self.V) / depth

        return np.array([u, v])

    def unproject_ray(self, image_point: np.ndarray):
        """Unproject image point to 3D ray using CAHV model."""
        u, v = image_point

        H_term = self.H - u * self.A
        V_term = self.V - v * self.A

        ray_dir = np.cross(H_term, V_term)
        ray_dir_norm = np.linalg.norm(ray_dir)

        if ray_dir_norm < 1e-10:
            ray_dir = self.A
        else:
            ray_dir = ray_dir / ray_dir_norm

        if np.dot(ray_dir, self.A) < 0:
            ray_dir = -ray_dir

        return self.C, ray_dir
