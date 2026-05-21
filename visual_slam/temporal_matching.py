"""
SMART TEMPORAL MATCHER - temporal feature tracking between consecutive frames.
"""

import numpy as np
import cv2 as cv
from typing import List

from visual_slam.config import DatasetAdaptiveParameters


class SmartTemporalMatcher:
    """Smart temporal matching with improved validation."""

    def __init__(self, adaptive_params: DatasetAdaptiveParameters):
        self.params = adaptive_params

    def match_temporal_features_smart(self, left_kp_t: List, left_desc_t: np.ndarray,
                                    left_kp_t1: List, left_desc_t1: np.ndarray) -> List:
        """Smart temporal matching with multi-stage validation."""

        initial_matches = self._initial_temporal_matching(left_desc_t, left_desc_t1)
        print(f"     Stage 1 - Initial matches: {len(initial_matches)}")

        if len(initial_matches) < 8:
            return initial_matches

        try:
            fundamental_validated = self._fundamental_matrix_validation(
                left_kp_t, left_kp_t1, initial_matches
            )
            print(f"     Stage 2 - Fundamental validated: {len(fundamental_validated)}")
        except Exception as e:
            fundamental_validated = self._distance_based_filtering(initial_matches)
            print(f"     Stage 2 - Distance filtered fallback: {len(fundamental_validated)}")

        min_inliers = self.params.get_param('temporal_min_inliers')
        if len(fundamental_validated) < min_inliers:
            return self._distance_based_filtering(initial_matches)

        return fundamental_validated

    def _initial_temporal_matching(self, left_desc_t: np.ndarray, left_desc_t1: np.ndarray) -> List:
        """Initial temporal matching with better parameters."""

        if left_desc_t.size == 0 or left_desc_t1.size == 0:
            return []

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        desc1 = left_desc_t.astype(np.float32)
        desc2 = left_desc_t1.astype(np.float32)

        try:
            matches = flann.knnMatch(desc1, desc2, k=2)
        except Exception as e:
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(desc1, desc2, k=2)

        ratio_threshold = self.params.get_param('asift_ratio_threshold')
        good_matches = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        return good_matches

    def _fundamental_matrix_validation(self, left_kp_t: List, left_kp_t1: List,
                                     matches: List) -> List:
        """Fundamental matrix validation with proper thresholds."""

        if len(matches) < 8:
            return matches

        try:
            pts1 = np.float32([left_kp_t[m.queryIdx].pt for m in matches])
            pts2 = np.float32([left_kp_t1[m.trainIdx].pt for m in matches])
        except (IndexError, AttributeError) as e:
            return matches

        threshold = self.params.get_param('temporal_fundamental_threshold')

        try:
            F, mask = cv.findFundamentalMat(
                pts1, pts2,
                cv.FM_RANSAC,
                ransacReprojThreshold=threshold,
                confidence=0.99,
                maxIters=self.params.get_param('ransac_iterations')
            )
        except Exception as e:
            return self._distance_based_filtering(matches)

        if F is None or mask is None:
            return self._distance_based_filtering(matches)

        validated_matches = []
        for i, m in enumerate(matches):
            try:
                if i < len(mask):
                    mask_value = mask[i]
                    if hasattr(mask_value, '__len__') and len(mask_value) > 0:
                        is_inlier = bool(mask_value[0])
                    else:
                        is_inlier = bool(mask_value)

                    if is_inlier:
                        validated_matches.append(m)
            except (IndexError, TypeError) as e:
                continue

        min_inliers = self.params.get_param('temporal_min_inliers')
        if len(validated_matches) < min_inliers:
            return self._distance_based_filtering(matches)

        return validated_matches

    def _distance_based_filtering(self, matches: List) -> List:
        """Better distance-based filtering."""
        if not matches:
            return []

        try:
            distances = [float(m.distance) for m in matches]

            if len(distances) == 0:
                return matches

            distances_array = np.array(distances, dtype=np.float64)
            mean_dist = float(np.mean(distances_array))
            std_dist = float(np.std(distances_array))

            # Use working threshold that's not too restrictive
            threshold = mean_dist + 2.0 * std_dist  # Less restrictive than before

            filtered_matches = []
            for m, dist in zip(matches, distances):
                if dist <= threshold:
                    filtered_matches.append(m)

            if len(filtered_matches) == 0 and len(matches) > 0:
                sorted_matches = sorted(matches, key=lambda x: float(x.distance))
                filtered_matches = sorted_matches[:max(1, len(matches)//2)]  # Keep top 50%

            return filtered_matches

        except Exception as e:
            if matches:
                sorted_matches = sorted(matches, key=lambda x: float(x.distance))
                return sorted_matches[:max(1, len(matches)//2)]
            return []
