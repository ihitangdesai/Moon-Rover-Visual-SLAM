"""
ENHANCED 3D CORRESPONDENCE FINDER WITH COMPLETE STAGE 3 - 3-stage validator.
"""

import numpy as np
from typing import List, Tuple, Dict

from visual_slam.config import DatasetAdaptiveParameters


class Enhanced3DCorrespondenceFinder:
    """Enhanced 3D correspondence finder with complete 3-stage validation."""

    def __init__(self, adaptive_params: DatasetAdaptiveParameters):
        self.params = adaptive_params

    def find_3d_correspondences(self,
                              temporal_matches: List,
                              left_kp_t: List, left_desc_t: np.ndarray,
                              left_kp_t1: List, left_desc_t1: np.ndarray,
                              points_3d_t: List[np.ndarray], stereo_matches_t: List[Tuple[int, int]],
                              points_3d_t1: List[np.ndarray], stereo_matches_t1: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Enhanced 3D correspondence finding with complete 3-stage validation."""

        print(f"   Finding 3D correspondences...")
        print(f"      Temporal matches: {len(temporal_matches)}")
        print(f"      3D points t: {len(points_3d_t)}, t+1: {len(points_3d_t1)}")

        kp_to_3d_t = self._build_keypoint_to_3d_mapping(stereo_matches_t, points_3d_t)
        kp_to_3d_t1 = self._build_keypoint_to_3d_mapping(stereo_matches_t1, points_3d_t1)

        print(f"      3D mappings: t={len(kp_to_3d_t)}, t+1={len(kp_to_3d_t1)}")

        initial_correspondences = []
        correspondence_info = []

        for match_idx, match in enumerate(temporal_matches):
            left_idx_t = match.queryIdx
            left_idx_t1 = match.trainIdx

            if left_idx_t in kp_to_3d_t and left_idx_t1 in kp_to_3d_t1:
                point_3d_t = kp_to_3d_t[left_idx_t]
                point_3d_t1 = kp_to_3d_t1[left_idx_t1]

                if left_idx_t < len(left_kp_t) and left_idx_t1 < len(left_kp_t1):
                    kp_t = left_kp_t[left_idx_t]
                    kp_t1 = left_kp_t1[left_idx_t1]

                    corr_info = {
                        'match_idx': match_idx,
                        'match_distance': float(match.distance),
                        'kp_t_pt': kp_t.pt,
                        'kp_t1_pt': kp_t1.pt,
                        'left_idx_t': left_idx_t,
                        'left_idx_t1': left_idx_t1,
                        'descriptor_distance': float(match.distance)
                    }

                    initial_correspondences.append((point_3d_t, point_3d_t1))
                    correspondence_info.append(corr_info)

        print(f"      Initial correspondences: {len(initial_correspondences)}")

        if len(initial_correspondences) == 0:
            print("      No initial correspondences found, trying fallback methods...")
            fallback_result = self._descriptor_based_correspondence_fallback(
                temporal_matches, left_kp_t, left_desc_t, left_kp_t1, left_desc_t1,
                kp_to_3d_t, kp_to_3d_t1, points_3d_t, points_3d_t1
            )
            if fallback_result[0]:
                return fallback_result

            return self._spatial_proximity_correspondence_fallback(points_3d_t, points_3d_t1)

        validated_correspondences, validated_info = self._validate_correspondences_complete_3stage(
            initial_correspondences, correspondence_info, left_desc_t, left_desc_t1
        )

        if len(validated_correspondences) == 0:
            print("      No correspondences passed validation")
            return self._relaxed_correspondence_finding(
                temporal_matches, kp_to_3d_t, kp_to_3d_t1, left_kp_t, left_kp_t1
            )

        corresponding_3d_t = [corr[0] for corr in validated_correspondences]
        corresponding_3d_t1 = [corr[1] for corr in validated_correspondences]

        print(f"      Final correspondences: {len(corresponding_3d_t)}")

        return corresponding_3d_t, corresponding_3d_t1, validated_info

    def _build_keypoint_to_3d_mapping(self, stereo_matches: List[Tuple[int, int]],
                                     points_3d: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """Build mapping from keypoint index to 3D point."""
        mapping = {}

        for i, (left_idx, right_idx) in enumerate(stereo_matches):
            if i < len(points_3d):
                mapping[left_idx] = points_3d[i]

        return mapping

    def _validate_correspondences_complete_3stage(self, correspondences: List[Tuple[np.ndarray, np.ndarray]],
                                                 correspondence_info: List[Dict],
                                                 left_desc_t: np.ndarray, left_desc_t1: np.ndarray) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Dict]]:
        """COMPLETE 3-stage validation of 3D correspondences."""

        if len(correspondences) == 0:
            return [], []

        print(f"         Starting 3-stage correspondence validation...")

        # ===== STAGE 1: SPATIAL VALIDATION =====
        spatial_threshold = float(self.params.get_param('correspondence_spatial_threshold_mm'))
        spatial_valid_indices = []
        spatial_distances = []

        for i in range(len(correspondences)):
            try:
                point_3d_t, point_3d_t1 = correspondences[i]
                pt1 = np.array(point_3d_t, dtype=np.float64)
                pt2 = np.array(point_3d_t1, dtype=np.float64)
                spatial_distance = float(np.linalg.norm(pt1 - pt2))
                spatial_distances.append(spatial_distance)

                if spatial_distance < spatial_threshold:
                    spatial_valid_indices.append(i)

            except Exception as e:
                spatial_distances.append(float('inf'))
                continue

        print(f"         Stage 1 (Spatial): {len(spatial_valid_indices)}/{len(correspondences)} passed")
        if spatial_distances:
            valid_distances = [d for d in spatial_distances if d != float('inf')]
            if valid_distances:
                print(f"         Spatial distances: min={min(valid_distances):.1f}, max={max(valid_distances):.1f}, mean={np.mean(valid_distances):.1f} mm")
                print(f"         Spatial threshold: {spatial_threshold:.1f} mm")

        if len(spatial_valid_indices) == 0:
            print(f"         ? All correspondences failed spatial validation - threshold too strict!")
            return [], []

        # ===== STAGE 2: DESCRIPTOR VALIDATION =====
        descriptor_threshold = self.params.get_param('correspondence_descriptor_threshold')
        descriptor_valid_indices = []
        descriptor_distances = []

        for i in spatial_valid_indices:
            info = correspondence_info[i]
            desc_distance = info.get('descriptor_distance', 0.5)
            descriptor_distances.append(desc_distance)

            try:
                desc_dist_scalar = float(desc_distance)
                if desc_dist_scalar < descriptor_threshold:
                    descriptor_valid_indices.append(i)
            except (TypeError, ValueError):
                print(f"         ?? Invalid descriptor distance at index {i}: {desc_distance}")
                continue

        print(f"         Stage 2 (Descriptor): {len(descriptor_valid_indices)}/{len(spatial_valid_indices)} passed")
        if descriptor_distances:
            print(f"         Descriptor distances: min={min(descriptor_distances):.3f}, max={max(descriptor_distances):.3f}, mean={np.mean(descriptor_distances):.3f}")
            print(f"         Descriptor threshold: {descriptor_threshold:.3f}")

        if len(descriptor_valid_indices) == 0:
            print(f"         ?? All correspondences failed descriptor validation - using spatial-only validation")
            spatial_correspondences = [correspondences[i] for i in spatial_valid_indices]
            spatial_info = [correspondence_info[i] for i in spatial_valid_indices]
            return spatial_correspondences, spatial_info

        # ===== STAGE 3: STATISTICAL OUTLIER REMOVAL =====
        if len(descriptor_valid_indices) > 3:
            try:
                statistical_valid_indices = self._remove_statistical_outliers_by_indices(
                    correspondences, descriptor_valid_indices
                )
                print(f"         Stage 3 (Statistical): {len(statistical_valid_indices)}/{len(descriptor_valid_indices)} passed")

                if len(statistical_valid_indices) > 0:
                    final_correspondences = [correspondences[i] for i in statistical_valid_indices]
                    final_info = [correspondence_info[i] for i in statistical_valid_indices]
                    return final_correspondences, final_info
                else:
                    # Fall back to descriptor results
                    descriptor_correspondences = [correspondences[i] for i in descriptor_valid_indices]
                    descriptor_info = [correspondence_info[i] for i in descriptor_valid_indices]
                    return descriptor_correspondences, descriptor_info

            except Exception as e:
                print(f"         ?? Statistical outlier removal failed: {e}")
                # Fall back to descriptor results
                descriptor_correspondences = [correspondences[i] for i in descriptor_valid_indices]
                descriptor_info = [correspondence_info[i] for i in descriptor_valid_indices]
                return descriptor_correspondences, descriptor_info
        else:
            print(f"         Stage 3 (Statistical): Skipped - insufficient correspondences ({len(descriptor_valid_indices)} < 4)")

        # Return descriptor validation results
        descriptor_correspondences = [correspondences[i] for i in descriptor_valid_indices]
        descriptor_info = [correspondence_info[i] for i in descriptor_valid_indices]
        return descriptor_correspondences, descriptor_info

    def _remove_statistical_outliers_by_indices(self, correspondences: List[Tuple[np.ndarray, np.ndarray]],
                                           valid_indices: List[int]) -> List[int]:
        """OPTIMIZED Stage 3: Statistical outlier removal using vectorized operations."""

        if len(valid_indices) < 4:
            return valid_indices

        try:
            # Vectorized motion calculation
            valid_correspondences = [correspondences[i] for i in valid_indices]

            # Convert to numpy arrays for vectorization
            points_t = np.array([corr[0] for corr in valid_correspondences])  # Shape: (n, 3)
            points_t1 = np.array([corr[1] for corr in valid_correspondences])  # Shape: (n, 3)

            # Calculate motion vectors for all correspondences at once
            motions = points_t1 - points_t  # Shape: (n, 3)

            # Calculate motion magnitudes
            motion_magnitudes = np.linalg.norm(motions, axis=1)

            # Use median absolute deviation for outlier detection
            median_magnitude = np.median(motion_magnitudes)
            mad = np.median(np.abs(motion_magnitudes - median_magnitude))

            # Define outlier threshold (2.5 sigma equivalent)
            threshold = median_magnitude + 2.5 * mad

            # Filter based on motion magnitude consistency
            inlier_mask = motion_magnitudes < threshold

            # Additional direction consistency check (vectorized)
            if np.sum(inlier_mask) > 3:
                inlier_motions = motions[inlier_mask]
                # Calculate mean motion direction
                mean_motion = np.mean(inlier_motions, axis=0)

                # Calculate angle between each motion and mean motion
                motion_norms = np.linalg.norm(inlier_motions, axis=1)
                mean_norm = np.linalg.norm(mean_motion)

                # Avoid division by zero
                valid_motion_mask = (motion_norms > 1e-6) & (mean_norm > 1e-6)

                if np.sum(valid_motion_mask) > 0:
                    valid_inlier_motions = inlier_motions[valid_motion_mask]
                    valid_norms = motion_norms[valid_motion_mask]

                    # Vectorized dot product
                    dots = np.dot(valid_inlier_motions, mean_motion)
                    cos_angles = dots / (valid_norms * mean_norm)

                    # Filter based on direction consistency (within 45 degrees)
                    direction_threshold = np.cos(np.pi / 4)  # 45 degrees
                    direction_consistent = cos_angles > direction_threshold

                    # Update inlier mask
                    temp_inlier_indices = np.where(inlier_mask)[0]
                    temp_valid_indices = temp_inlier_indices[valid_motion_mask]
                    final_valid_indices = temp_valid_indices[direction_consistent]

                    final_inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
                    final_inlier_mask[final_valid_indices] = True
                    inlier_mask = final_inlier_mask

            # Convert back to original indices
            inlier_indices = [valid_indices[i] for i in range(len(valid_indices)) if inlier_mask[i]]

            print(f"         Vectorized outlier removal: {len(inlier_indices)}/{len(valid_indices)} kept")

            # Ensure minimum number of correspondences
            if len(inlier_indices) < 3 and len(valid_indices) >= 3:
                print(f"         Keeping top 70% by motion magnitude")
                # Sort by motion magnitude and keep middle 70%
                sorted_indices = sorted(range(len(valid_indices)),
                                    key=lambda i: motion_magnitudes[i])
                start_idx = len(sorted_indices) // 6
                end_idx = len(sorted_indices) - len(sorted_indices) // 6
                inlier_indices = [valid_indices[i] for i in sorted_indices[start_idx:end_idx]]

            return inlier_indices if len(inlier_indices) > 0 else valid_indices

        except Exception as e:
            print(f"         Optimized outlier removal error: {e}")
            return valid_indices

    def _descriptor_based_correspondence_fallback(self, temporal_matches: List,
                                                left_kp_t: List, left_desc_t: np.ndarray,
                                                left_kp_t1: List, left_desc_t1: np.ndarray,
                                                kp_to_3d_t: Dict, kp_to_3d_t1: Dict,
                                                points_3d_t: List, points_3d_t1: List) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Enhanced fallback using descriptor similarity."""
        print("      Trying descriptor-based correspondence fallback...")

        try:
            corresponding_3d_t = []
            corresponding_3d_t1 = []
            info_list = []

            valid_kp_indices_t = list(kp_to_3d_t.keys())
            valid_kp_indices_t1 = list(kp_to_3d_t1.keys())

            if len(valid_kp_indices_t) == 0 or len(valid_kp_indices_t1) == 0:
                return [], [], []

            if (max(valid_kp_indices_t) >= len(left_desc_t) or
                max(valid_kp_indices_t1) >= len(left_desc_t1)):
                return [], [], []

            # Use working spatial threshold
            max_spatial_distance = self.params.get_param('correspondence_spatial_threshold_mm') * 2.0

            for match in temporal_matches[:50]:
                try:
                    left_idx_t = match.queryIdx
                    left_idx_t1 = match.trainIdx

                    if left_idx_t in kp_to_3d_t and left_idx_t1 in kp_to_3d_t1:
                        point_3d_t = kp_to_3d_t[left_idx_t]
                        point_3d_t1 = kp_to_3d_t1[left_idx_t1]

                        spatial_dist = float(np.linalg.norm(point_3d_t - point_3d_t1))
                        if spatial_dist < max_spatial_distance:
                            corresponding_3d_t.append(point_3d_t)
                            corresponding_3d_t1.append(point_3d_t1)
                            info_list.append({
                                'match_distance': float(match.distance),
                                'spatial_distance': spatial_dist,
                                'descriptor_fallback': True
                            })

                except Exception as e:
                    continue

            print(f"      Descriptor fallback found: {len(corresponding_3d_t)} correspondences")
            return corresponding_3d_t, corresponding_3d_t1, info_list

        except Exception as e:
            print(f"      Descriptor fallback failed: {e}")
            return [], [], []

    def _spatial_proximity_correspondence_fallback(self, points_3d_t: List, points_3d_t1: List) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Final fallback using pure spatial proximity."""
        print("      Trying spatial proximity correspondence fallback...")

        try:
            if len(points_3d_t) == 0 or len(points_3d_t1) == 0:
                return [], [], []

            corresponding_3d_t = []
            corresponding_3d_t1 = []
            info_list = []

            max_points = min(50, len(points_3d_t), len(points_3d_t1))
            max_distance = self.params.get_param('correspondence_spatial_threshold_mm') * 3.0

            for i in range(max_points):
                try:
                    point_t = points_3d_t[i]
                    min_distance = float('inf')
                    best_match_idx = -1

                    for j in range(max_points):
                        point_t1 = points_3d_t1[j]
                        distance = float(np.linalg.norm(point_t - point_t1))

                        if distance < min_distance:
                            min_distance = distance
                            best_match_idx = j

                    if best_match_idx >= 0 and min_distance < max_distance:
                        corresponding_3d_t.append(point_t)
                        corresponding_3d_t1.append(points_3d_t1[best_match_idx])
                        info_list.append({
                            'spatial_distance': min_distance,
                            'spatial_fallback': True
                        })

                        if len(corresponding_3d_t) >= 10:
                            break

                except Exception as e:
                    continue

            print(f"      Spatial fallback found: {len(corresponding_3d_t)} correspondences")
            return corresponding_3d_t, corresponding_3d_t1, info_list

        except Exception as e:
            print(f"      Spatial fallback failed: {e}")
            return [], [], []

    def _relaxed_correspondence_finding(self, temporal_matches: List,
                                      kp_to_3d_t: Dict, kp_to_3d_t1: Dict,
                                      left_kp_t: List, left_kp_t1: List) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Super relaxed correspondence finding as fallback."""
        print("      Applying super-relaxed correspondence finding...")

        super_relaxed_spatial_threshold = self.params.get_param('correspondence_spatial_threshold_mm') * 5.0
        super_relaxed_descriptor_threshold = self.params.get_param('correspondence_descriptor_threshold') * 2.0

        corresponding_3d_t = []
        corresponding_3d_t1 = []
        info_list = []

        for match in temporal_matches:
            left_idx_t = match.queryIdx
            left_idx_t1 = match.trainIdx

            if left_idx_t in kp_to_3d_t and left_idx_t1 in kp_to_3d_t1:
                point_3d_t = kp_to_3d_t[left_idx_t]
                point_3d_t1 = kp_to_3d_t1[left_idx_t1]

                spatial_distance = np.linalg.norm(point_3d_t - point_3d_t1)
                spatial_ok = spatial_distance < super_relaxed_spatial_threshold
                descriptor_ok = match.distance < super_relaxed_descriptor_threshold

                if spatial_ok and descriptor_ok:
                    corresponding_3d_t.append(point_3d_t)
                    corresponding_3d_t1.append(point_3d_t1)
                    info_list.append({
                        'match_distance': match.distance,
                        'spatial_distance': spatial_distance,
                        'relaxed': True
                    })

        print(f"      Super-relaxed correspondences: {len(corresponding_3d_t)}")

        return corresponding_3d_t, corresponding_3d_t1, info_list
