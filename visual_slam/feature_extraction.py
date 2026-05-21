"""
OPTIMIZED ASIFT MATCHER - feature extraction and stereo/temporal matching.
"""

import numpy as np
import cv2 as cv
from typing import List, Tuple

from visual_slam.config import DatasetAdaptiveParameters


# =====================================================================
# OPTIMIZED ASIFT MATCHER
# =====================================================================

class OptimizedASIFTMatcher:
    """Optimized ASIFT with better performance/accuracy balance."""

    def __init__(self, adaptive_params: DatasetAdaptiveParameters):
        self.params = adaptive_params
        # Use working SIFT parameters
        self.detector = cv.SIFT_create(
            nfeatures=1200,  # More features for better matches
            contrastThreshold=0.04,  # Less restrictive
            edgeThreshold=10,  # Less restrictive
            sigma=1.6
        )
        self.matcher = self._init_matcher()

    def _init_matcher(self):
        """Initialize optimized matcher."""
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        return cv.FlannBasedMatcher(index_params, search_params)

    def affine_detect(self, img: np.ndarray) -> Tuple[List, np.ndarray]:
        """Optimized ASIFT with working parameters."""

        views = self._generate_working_views(img)
        results = [self._detect_features_in_view(view) for view in views]

        all_keypoints = []
        all_descriptors = []

        for keypoints, descriptors in results:
            if len(keypoints) > 0 and descriptors.size > 0:
                all_keypoints.extend(keypoints)
                all_descriptors.append(descriptors)

        if all_descriptors:
            combined_descriptors = np.vstack(all_descriptors)
        else:
            combined_descriptors = np.array([])

        return all_keypoints, combined_descriptors

    def _generate_working_views(self, img: np.ndarray) -> List:
        """Generate working affine views that produce good matches."""

        views = []
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)
        identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        views.append((1.0, 0.0, img.copy(), mask, identity_matrix))

        max_tilts = self.params.get_param('asift_max_tilts')
        rotations_per_tilt = self.params.get_param('asift_rotations_per_tilt')

        for tilt in np.sqrt(2) ** np.arange(1, max_tilts):
            for phi in np.arange(0, 180, 180.0 / max(rotations_per_tilt, 1)):
                warped_img, warped_mask, A_inv = self._affine_skew(tilt, phi, img)
                views.append((tilt, phi, warped_img, warped_mask, A_inv))

        return views

    def _affine_skew(self, tilt: float, phi: float, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply affine transformation."""
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)

        A_rot = cv.getRotationMatrix2D((w/2, h/2), phi, 1.0)
        A_tilt = np.array([[1.0, 0.0, 0.0], [0.0, 1.0/tilt, 0.0]], dtype=np.float32)
        A_combined = np.dot(A_tilt, np.vstack([A_rot, [0, 0, 1]]))

        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        transformed_corners = np.dot(A_combined, corners)

        x_min, x_max = transformed_corners[0].min(), transformed_corners[0].max()
        y_min, y_max = transformed_corners[1].min(), transformed_corners[1].max()

        tx = -x_min if x_min < 0 else 0
        ty = -y_min if y_min < 0 else 0

        A_combined[0, 2] += tx
        A_combined[1, 2] += ty

        out_w = int(np.ceil(x_max - x_min))
        out_h = int(np.ceil(y_max - y_min))

        warped_img = cv.warpAffine(img, A_combined, (out_w, out_h),
                                  flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        warped_mask = cv.warpAffine(mask, A_combined, (out_w, out_h), flags=cv.INTER_NEAREST)

        A_combined_3x3 = np.vstack([A_combined, [0, 0, 1]])
        A_inv = np.linalg.inv(A_combined_3x3)[:2, :]

        return warped_img, warped_mask, A_inv

    def _detect_features_in_view(self, view_data: Tuple) -> Tuple[List, np.ndarray]:
        """Detect features in a single view."""
        tilt, phi, img, mask, A_inv = view_data

        keypoints, descriptors = self.detector.detectAndCompute(img, mask)

        if descriptors is None:
            return [], np.array([])

        transformed_keypoints = []
        for kp in keypoints:
            x, y = kp.pt
            original_pt = np.dot(A_inv, np.array([x, y, 1]))

            new_kp = cv.KeyPoint(
                x=float(original_pt[0]), y=float(original_pt[1]),
                size=kp.size, angle=kp.angle, response=kp.response,
                octave=kp.octave, class_id=kp.class_id
            )
            transformed_keypoints.append(new_kp)

        return transformed_keypoints, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features with working parameters."""
        if desc1.size == 0 or desc2.size == 0:
            return []

        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        ratio_threshold = self.params.get_param('asift_ratio_threshold')
        good_matches = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches
