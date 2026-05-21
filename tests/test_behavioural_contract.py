"""
Behavioural contract tests: verify the refactored visual_slam package
is behaviourally equivalent to the original monolithic_code.py.

Run with:
    conda run -n slam python -m unittest discover -s tests -v
"""

import subprocess
import sys
import os
import unittest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEFT_CAHV = {
    'C': np.array([-14.75594, 124.24118, 326.07168]),
    'A': np.array([1.000000, 0.004296, -0.165492]),
    'H': np.array([498.239481, -1416.743211, -77.511692]),
    'V': np.array([241.480345, 6.423157, -1480.974653]),
}
RIGHT_CAHV = {
    'C': np.array([-6.40296, -117.67110, 324.87617]),
    'A': np.array([1.000000, 0.014939, -0.180854]),
    'H': np.array([517.434318, -1416.421622, -104.037138]),
    'V': np.array([236.239929, 12.943905, -1489.032846]),
}


def _import_output(code: str) -> str:
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, cwd=ROOT
    )
    return result.stdout + result.stderr


# =====================================================================
# Module-level side effects
# =====================================================================

class TestModuleLevelSideEffects(unittest.TestCase):

    def test_gtsam_import_print_fires(self):
        out = _import_output('import visual_slam')
        self.assertTrue(
            'GTSAM successfully imported' in out or 'GTSAM not available' in out,
            f"GTSAM status print missing. Got:\n{out}"
        )

    def test_numpy_version_print_fires(self):
        out = _import_output('import visual_slam')
        self.assertIn('NumPy version:', out,
                      f"NumPy version print missing. Got:\n{out}")

    def test_side_effects_order(self):
        out = _import_output('import visual_slam')
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        gtsam_idx = next((i for i, l in enumerate(lines)
                          if 'GTSAM' in l and ('successfully imported' in l or 'not available' in l)), -1)
        numpy_idx = next((i for i, l in enumerate(lines) if 'NumPy version:' in l), -1)
        self.assertGreaterEqual(gtsam_idx, 0, "GTSAM status line not found")
        self.assertGreaterEqual(numpy_idx, 0, "NumPy version line not found")
        self.assertLess(gtsam_idx, numpy_idx,
                        "GTSAM status must print before NumPy version")


# =====================================================================
# GTSAM_AVAILABLE flag
# =====================================================================

class TestGTSAMAvailableFlag(unittest.TestCase):

    def test_flag_accessible(self):
        import visual_slam
        self.assertTrue(hasattr(visual_slam, 'GTSAM_AVAILABLE'))

    def test_flag_is_bool(self):
        import visual_slam
        self.assertIsInstance(visual_slam.GTSAM_AVAILABLE, bool)

    def test_flag_matches_gtsam_import(self):
        import visual_slam
        try:
            import gtsam
            expected = True
        except ImportError:
            expected = False
        self.assertEqual(visual_slam.GTSAM_AVAILABLE, expected)


# =====================================================================
# Public API surface
# =====================================================================

class TestPublicAPI(unittest.TestCase):
    EXPECTED_NAMES = [
        'GTSAM_AVAILABLE',
        'PlaceRecognitionInterface', 'LoopClosureInterface', 'PoseGraphOptimizerInterface',
        'PlaceDescriptor', 'LoopClosureCandidate', 'PoseGraphEdge', 'CAHVCamera',
        'DatasetAdaptiveParameters',
        'DescriptorBasedPlaceRecognition',
        'SpatialTemporalLoopClosureDetector', 'LoopClosureManager',
        'LeastSquaresPoseGraphOptimizer', 'GTSAMPoseGraphOptimizer',
        'SLAMVisualizationSystem',
        'OptimizedASIFTMatcher', 'StereoImageLoader', 'SmartTemporalMatcher',
        'RobustTriangulator', 'Enhanced3DCorrespondenceFinder',
        'ImprovedVisualSLAM', 'EnhancedVisualSLAMWithLoopClosure',
        'main_working_slam',
    ]

    def test_all_names_present(self):
        import visual_slam
        for name in self.EXPECTED_NAMES:
            self.assertTrue(hasattr(visual_slam, name),
                            f"Missing from visual_slam namespace: {name}")


# =====================================================================
# DatasetAdaptiveParameters
# =====================================================================

class TestDatasetAdaptiveParameters(unittest.TestCase):

    def _make_params(self):
        from visual_slam import DatasetAdaptiveParameters
        return DatasetAdaptiveParameters(LEFT_CAHV, RIGHT_CAHV)

    def test_baseline_value(self):
        params = self._make_params()
        self.assertAlmostEqual(params.baseline, 242.059, places=1)

    def test_asift_max_tilts_non_rectified(self):
        params = self._make_params()
        self.assertEqual(params.get_param('asift_max_tilts'), 8)

    def test_stereo_epipolar_threshold(self):
        params = self._make_params()
        self.assertEqual(params.get_param('stereo_epipolar_threshold'), 200.0)

    def test_ransac_iterations(self):
        params = self._make_params()
        self.assertEqual(params.get_param('ransac_iterations'), 1000)


# =====================================================================
# CAHVCamera
# =====================================================================

class TestCAHVCamera(unittest.TestCase):

    def _make_cam(self):
        from visual_slam import CAHVCamera
        return CAHVCamera(
            C=LEFT_CAHV['C'], A=LEFT_CAHV['A'],
            H=LEFT_CAHV['H'], V=LEFT_CAHV['V'],
        )

    def test_intrinsics_shape(self):
        K = self._make_cam().intrinsics_from_cahv()
        self.assertEqual(K.shape, (3, 3))

    def test_intrinsics_bottom_row(self):
        K = self._make_cam().intrinsics_from_cahv()
        np.testing.assert_array_equal(K[2], [0., 0., 1.])

    def test_intrinsics_skew_zero(self):
        K = self._make_cam().intrinsics_from_cahv()
        self.assertEqual(K[0, 1], 0.0)


# =====================================================================
# Feature extraction
# =====================================================================

class TestFeatureExtraction(unittest.TestCase):

    def _make_slam(self):
        from visual_slam import ImprovedVisualSLAM
        return ImprovedVisualSLAM(LEFT_CAHV, RIGHT_CAHV)

    def _make_image(self):
        import cv2 as cv
        img = np.zeros((240, 320), dtype=np.uint8)
        for i in range(10):
            cv.circle(img, (30 + i*25, 80 + (i % 3)*50), 15, 255, -1)
        cv.rectangle(img, (50, 50), (270, 190), 200, 2)
        return img

    def test_extract_returns_nonempty(self):
        slam = self._make_slam()
        kp, desc = slam.extract_features(self._make_image())
        self.assertGreater(len(kp), 0)
        self.assertGreater(desc.size, 0)

    def test_descriptor_dimension(self):
        slam = self._make_slam()
        kp, desc = slam.extract_features(self._make_image())
        if len(kp) > 0:
            self.assertEqual(desc.shape[1], 128)


# =====================================================================
# Loop closure statistics keys
# =====================================================================

class TestLoopClosureStatistics(unittest.TestCase):

    def test_detector_stats_keys_no_closures(self):
        from visual_slam import SpatialTemporalLoopClosureDetector
        det = SpatialTemporalLoopClosureDetector(debug=False)
        stats = det.get_closure_statistics()
        self.assertIn('total_closures', stats)

    def test_enhanced_slam_stats_keys(self):
        from visual_slam import EnhancedVisualSLAMWithLoopClosure
        slam = EnhancedVisualSLAMWithLoopClosure(LEFT_CAHV, RIGHT_CAHV)
        stats = slam.get_loop_closure_statistics()
        for k in ['loop_closure_enabled', 'frames_processed',
                  'loop_closures_detected', 'optimizer_type']:
            self.assertIn(k, stats, f"Missing key: {k}")


# =====================================================================
# Trajectory
# =====================================================================

class TestTrajectory(unittest.TestCase):

    def _make_slam(self):
        from visual_slam import ImprovedVisualSLAM
        return ImprovedVisualSLAM(LEFT_CAHV, RIGHT_CAHV)

    def test_initial_trajectory_length(self):
        slam = self._make_slam()
        self.assertEqual(len(slam.trajectory), 1)

    def test_initial_pose_shape(self):
        slam = self._make_slam()
        self.assertEqual(slam.current_pose.shape, (4, 4))

    def test_initial_position_in_meters(self):
        slam = self._make_slam()
        pos = slam.current_pose[:3, 3]
        expected = np.array(LEFT_CAHV['C']) / 1000.0
        np.testing.assert_array_almost_equal(pos, expected, decimal=6)


# =====================================================================
# Output paths
# =====================================================================

class TestOutputPaths(unittest.TestCase):

    def test_visualization_dirs_created(self):
        from visual_slam import SLAMVisualizationSystem
        viz = SLAMVisualizationSystem(show_lines=False)
        self.assertTrue(os.path.isdir(str(viz.stereo_matches_dir)))
        self.assertTrue(os.path.isdir(str(viz.temporal_matches_dir)))


# =====================================================================
# Kabsch algorithm
# =====================================================================

class TestKabsch(unittest.TestCase):

    def _make_slam(self):
        from visual_slam import ImprovedVisualSLAM
        return ImprovedVisualSLAM(LEFT_CAHV, RIGHT_CAHV)

    def _known_R(self, angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

    def test_kabsch_rotation_is_valid(self):
        slam = self._make_slam()
        np.random.seed(42)
        pts_src = np.random.randn(10, 3)
        R_k = self._known_R(np.pi / 4)
        t_k = np.array([0.5, -0.3, 0.1])
        pts_tgt = (R_k @ pts_src.T).T + t_k

        T = slam._estimate_transformation_kabsch(pts_src, pts_tgt)
        R_est = T[:3, :3]
        np.testing.assert_array_almost_equal(R_est @ R_est.T, np.eye(3), decimal=5)
        self.assertAlmostEqual(np.linalg.det(R_est), 1.0, places=5)

    def test_kabsch_recovers_known_transform(self):
        slam = self._make_slam()
        np.random.seed(7)
        pts_src = np.random.randn(20, 3)
        R_k = self._known_R(np.pi / 6)
        t_k = np.array([1.0, 2.0, -0.5])
        pts_tgt = (R_k @ pts_src.T).T + t_k

        T = slam._estimate_transformation_kabsch(pts_src, pts_tgt)
        np.testing.assert_array_almost_equal(T[:3, :3], R_k, decimal=5)
        np.testing.assert_array_almost_equal(T[:3, 3], t_k, decimal=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
