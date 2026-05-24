"""
All constants, thresholds, and default parameters for the Visual SLAM system.
"""

from typing import Dict, Any
import numpy as np


# =====================================================================
# DATASET ADAPTIVE PARAMETERS WITH WORKING VALUES
# =====================================================================

class DatasetAdaptiveParameters:
    """
    Dataset parameters using the working values that produce good matches.
    """

    def __init__(self, left_cahv: Dict, right_cahv: Dict):
        self.left_cahv = left_cahv
        self.right_cahv = right_cahv

        # Calculate baseline in millimeters (original coordinate system)
        baseline_vec = np.array(right_cahv['C']) - np.array(left_cahv['C'])
        self.baseline = np.linalg.norm(baseline_vec)  # in mm

        self.is_rectified = self._detect_rectified_geometry()
        self.params = self._compute_working_parameters()

        print(f"Dataset Analysis:")
        print(f"   Baseline: {self.baseline:.3f} mm")
        print(f"   Rectified geometry: {'Yes' if self.is_rectified else 'No'}")
        print(f"   Using proven working parameters")

    def _detect_rectified_geometry(self) -> bool:
        """Detect if cameras are in rectified stereo configuration."""
        left_A = np.array(self.left_cahv['A'])
        right_A = np.array(self.right_cahv['A'])

        axis_similarity = np.dot(left_A, right_A) / (np.linalg.norm(left_A) * np.linalg.norm(right_A))

        baseline_vec = np.array(self.right_cahv['C']) - np.array(self.left_cahv['C'])
        baseline_vec_norm = baseline_vec / np.linalg.norm(baseline_vec)

        is_parallel = axis_similarity > 0.95
        is_horizontal_baseline = abs(baseline_vec_norm[1]) < 0.3

        return is_parallel and is_horizontal_baseline

    def _compute_working_parameters(self) -> Dict:
        """Compute working parameters that produce good matches."""

        # Base working parameters that produce good matches
        params = {
            'asift_max_tilts': 6,
            'asift_rotations_per_tilt': 4,
            'asift_ratio_threshold': 0.8,

            'stereo_epipolar_threshold': 50.0,
            'temporal_fundamental_threshold': 3.0,
            'temporal_min_inliers': 10,

            # Triangulation parameters in mm
            'triangulation_min_parallax_deg': 0.1,
            'triangulation_max_reprojection_error': 20.0,
            'triangulation_min_depth_mm': 10.0,
            'triangulation_max_depth_mm': 500000.0,

            # Correspondence parameters in mm
            'correspondence_spatial_threshold_mm': 2000.0,
            'correspondence_descriptor_threshold': 300.0,

            # RANSAC parameters in mm
            'ransac_iterations': 1000,
            'ransac_threshold_mm': 500.0,
        }

        # Adapt based on baseline
        if self.baseline < 500:
            params.update({
                'stereo_epipolar_threshold': 100.0,
                'triangulation_min_parallax_deg': 0.05,
                'triangulation_max_reprojection_error': 30.0,
                'triangulation_min_depth_mm': 1.0,
                'correspondence_spatial_threshold_mm': 5000.0,
                'correspondence_descriptor_threshold': 500.0,
                'ransac_threshold_mm': 20.0,   # Phase 1 Fix 4: strict Kabsch fallback
                'temporal_fundamental_threshold': 5.0,
            })
        elif self.baseline > 1000:
            params.update({
                'stereo_epipolar_threshold': 200.0,
                'triangulation_min_parallax_deg': 0.5,
                'triangulation_max_reprojection_error': 15.0,
                'triangulation_min_depth_mm': 100.0,
                'correspondence_spatial_threshold_mm': 3000.0,
                'correspondence_descriptor_threshold': 400.0,
                'ransac_threshold_mm': 800.0,
            })

        # Adapt for non-rectified geometry
        if not self.is_rectified:
            params.update({
                'stereo_epipolar_threshold': params['stereo_epipolar_threshold'] * 2.0,
                'temporal_fundamental_threshold': params['temporal_fundamental_threshold'] * 1.5,
                'asift_max_tilts': 8,
                'correspondence_spatial_threshold_mm': params['correspondence_spatial_threshold_mm'] * 2.0,
                'correspondence_descriptor_threshold': params['correspondence_descriptor_threshold'] * 1.5,
            })

        return params

    def get_param(self, key: str) -> Any:
        """Get adaptive parameter value."""
        return self.params.get(key)
