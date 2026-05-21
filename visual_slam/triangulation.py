"""
ROBUST TRIANGULATOR - 3D point triangulation with proper validation.
"""

import numpy as np
from typing import List, Tuple, Optional

from visual_slam.config import DatasetAdaptiveParameters
from visual_slam.data_structures import CAHVCamera


class RobustTriangulator:
    """Robust triangulation with proper validation and scaling."""

    def __init__(self, left_cahv: CAHVCamera, right_cahv: CAHVCamera,
                 adaptive_params: DatasetAdaptiveParameters):
        self.left_cahv = left_cahv
        self.right_cahv = right_cahv
        self.params = adaptive_params

        # Calculate baseline in millimeters
        baseline_vec = (np.array(right_cahv.C) - np.array(left_cahv.C))
        self.baseline_mm = np.linalg.norm(baseline_vec)

        from visual_slam import GTSAM_AVAILABLE
        if GTSAM_AVAILABLE:
            self.left_camera = left_cahv.to_gtsam_camera()
            self.right_camera = right_cahv.to_gtsam_camera()
            print("GTSAM triangulation initialized")
        else:
            print("Using manual triangulation")

    def triangulate_point_robust(self, left_point: np.ndarray, right_point: np.ndarray) -> Optional[np.ndarray]:
        """Robust triangulation with proper validation."""
        from visual_slam import GTSAM_AVAILABLE

        if GTSAM_AVAILABLE:
            point_3d = self._triangulate_gtsam(left_point, right_point)
            if point_3d is not None:
                return point_3d  # Keep in mm

        point_3d = self._triangulate_manual(left_point, right_point)
        if point_3d is not None:
            return point_3d  # Keep in mm

        return None

    def _triangulate_gtsam(self, left_point: np.ndarray, right_point: np.ndarray) -> Optional[np.ndarray]:
        """GTSAM triangulation with robust validation."""
        try:
            import gtsam
            measurement1 = gtsam.Point2(float(left_point[0]), float(left_point[1]))
            measurement2 = gtsam.Point2(float(right_point[0]), float(right_point[1]))
            measurements = gtsam.Point2Vector([measurement1, measurement2])

            cameras = gtsam.CameraSetCal3_S2([self.left_camera, self.right_camera])
            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

            point_3d_gtsam = gtsam.triangulatePoint3(
                cameras, measurements,
                rank_tol=1e-9,
                optimize=True,
                model=measurement_noise,
                useLOST=True
            )

            # Convert back to mm
            point_coords = np.array([point_3d_gtsam[0]*1000.0, point_3d_gtsam[1]*1000.0, point_3d_gtsam[2]*1000.0])

            if self._validate_triangulated_point(point_coords, left_point, right_point):
                return point_coords

        except Exception as e:
            pass

        return None

    def _triangulate_manual(self, left_point: np.ndarray, right_point: np.ndarray) -> Optional[np.ndarray]:
        """Manual triangulation with improved robustness."""
        try:
            left_origin, left_dir = self.left_cahv.unproject_ray(left_point)
            right_origin, right_dir = self.right_cahv.unproject_ray(right_point)

            # Ensure normalized direction vectors
            left_dir = left_dir / np.linalg.norm(left_dir)
            right_dir = right_dir / np.linalg.norm(right_dir)

            w0 = left_origin - right_origin
            a = np.dot(left_dir, left_dir)
            b = np.dot(left_dir, right_dir)
            c = np.dot(right_dir, right_dir)
            d = np.dot(left_dir, w0)
            e = np.dot(right_dir, w0)

            denom = a * c - b * b

            if abs(denom) < 1e-10:
                return None

            t1 = (b * e - c * d) / denom
            t2 = (a * e - b * d) / denom

            if t1 < 0 or t2 < 0:
                return None

            point1 = left_origin + t1 * left_dir
            point2 = right_origin + t2 * right_dir
            point_3d = (point1 + point2) / 2

            if self._validate_triangulated_point(point_3d, left_point, right_point):
                return point_3d

        except Exception as e:
            pass

        return None

    def _validate_triangulated_point(self, point_3d: np.ndarray, left_point: np.ndarray,
                                   right_point: np.ndarray) -> bool:
        """Comprehensive validation of triangulated point."""

        try:
            point_3d = np.array(point_3d, dtype=np.float64)
            left_point = np.array(left_point, dtype=np.float64)
            right_point = np.array(right_point, dtype=np.float64)

            # Depth validation in millimeters
            min_depth = float(self.params.get_param('triangulation_min_depth_mm'))
            max_depth = float(self.params.get_param('triangulation_max_depth_mm'))

            left_depth = float(np.dot(point_3d - self.left_cahv.C, self.left_cahv.A))
            right_depth = float(np.dot(point_3d - self.right_cahv.C, self.right_cahv.A))

            if left_depth < min_depth or left_depth > max_depth:
                return False
            if right_depth < min_depth or right_depth > max_depth:
                return False

            # Reprojection error validation
            max_error = float(self.params.get_param('triangulation_max_reprojection_error'))

            try:
                left_reproj = self.left_cahv.project_point(point_3d)
                right_reproj = self.right_cahv.project_point(point_3d)

                if np.isnan(left_reproj).any() or np.isnan(right_reproj).any():
                    return False

                left_error = float(np.linalg.norm(left_point - left_reproj))
                right_error = float(np.linalg.norm(right_point - right_reproj))

                if left_error > max_error or right_error > max_error:
                    return False

            except Exception as e:
                return False

            # Parallax validation
            min_parallax_deg = float(self.params.get_param('triangulation_min_parallax_deg'))

            ray1 = point_3d - self.left_cahv.C
            ray2 = point_3d - self.right_cahv.C

            ray1_norm = ray1 / np.linalg.norm(ray1)
            ray2_norm = ray2 / np.linalg.norm(ray2)

            cos_parallax = float(np.clip(np.dot(ray1_norm, ray2_norm), -1, 1))
            parallax_deg = float(np.arccos(cos_parallax) * 180 / np.pi)

            if parallax_deg < min_parallax_deg:
                return False

            # Reasonable coordinate bounds
            if (abs(float(point_3d[0])) > 1e6 or
                abs(float(point_3d[1])) > 1e6 or
                abs(float(point_3d[2])) > 1e6):
                return False

            return True

        except Exception as e:
            return False

    def triangulate_multiple_points(self, left_points: np.ndarray, right_points: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """Batch triangulation with validation."""
        points_3d = []
        valid_indices = []
        reprojection_errors = []

        for i, (left_pt, right_pt) in enumerate(zip(left_points, right_points)):
            point_3d = self.triangulate_point_robust(left_pt, right_pt)

            if point_3d is not None:
                try:
                    left_reproj = self.left_cahv.project_point(point_3d)
                    right_reproj = self.right_cahv.project_point(point_3d)

                    if not (np.isnan(left_reproj).any() or np.isnan(right_reproj).any()):
                        left_error = np.linalg.norm(left_pt - left_reproj)
                        right_error = np.linalg.norm(right_pt - right_reproj)
                        avg_error = (left_error + right_error) / 2.0

                        points_3d.append(point_3d)  # Keep in mm
                        valid_indices.append(i)
                        reprojection_errors.append(avg_error)
                except:
                    continue

        print(f"     Robust triangulation: {len(valid_indices)}/{len(left_points)} successful")
        if reprojection_errors:
            print(f"     Average reprojection error: {np.mean(reprojection_errors):.2f} pixels")

        return points_3d, valid_indices, reprojection_errors
