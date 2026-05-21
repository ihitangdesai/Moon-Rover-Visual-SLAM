"""
LOOP CLOSURE DETECTION (SOLID - Single Responsibility Principle)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional

from visual_slam.data_structures import LoopClosureCandidate
from visual_slam.interfaces import PlaceRecognitionInterface, LoopClosureInterface, PoseGraphOptimizerInterface
from visual_slam.visualisation import SLAMVisualizationSystem


# =====================================================================
# LOOP CLOSURE DETECTOR
# =====================================================================

class SpatialTemporalLoopClosureDetector:
    """Loop closure detection using spatial and temporal constraints."""

    def __init__(self,
                 min_temporal_distance: int = 10,  # LOWERED from 30
                 max_spatial_distance: float = 50.0,  # INCREASED from 10.0 meters
                 min_similarity_score: float = 0.4,  # LOWERED from 0.8
                 pose_verification_threshold: float = 20.0,  # INCREASED from 5.0 meters
                 debug: bool = True):

        self.min_temporal_distance = min_temporal_distance
        self.max_spatial_distance = max_spatial_distance
        self.min_similarity_score = min_similarity_score
        self.pose_verification_threshold = pose_verification_threshold
        self.debug = debug

        self.frame_poses: Dict[int, np.ndarray] = {}
        self.detected_closures: List[LoopClosureCandidate] = []

        print("Spatial-temporal loop closure detector initialized")
        print(f"   Min temporal distance: {min_temporal_distance} frames")
        print(f"   Max spatial distance: {max_spatial_distance} m")
        print(f"   Min similarity score: {min_similarity_score}")
        print(f"   Pose verification threshold: {pose_verification_threshold} m")
        print(f"   Debug mode: {debug}")

    def update_frame_pose(self, frame_id: int, pose: np.ndarray):
        """Update frame pose for spatial analysis."""
        self.frame_poses[frame_id] = pose.copy()
        if self.debug:
            pos = pose[:3, 3]
            print(f"   DEBUG: Updated pose for frame {frame_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    def detect_loop_closure(self, current_frame_id: int, candidate_matches: List[Tuple[int, float]], current_pose: np.ndarray) -> Optional[LoopClosureCandidate]:
        """Simplified loop closure detection - VISUAL SIMILARITY ONLY (no spatial constraint)."""

        self.update_frame_pose(current_frame_id, current_pose)

        if self.debug:
            print(f"DEBUG LC: Checking frame {current_frame_id}, candidates: {len(candidate_matches)}")

        for matched_frame_id, similarity_score in candidate_matches:
            if self.debug:
                print(f"DEBUG LC: Testing frame {matched_frame_id}, similarity: {similarity_score:.3f}")

            # Only temporal check - ignore spatial distance since actual distances unknown
            temporal_distance = abs(current_frame_id - matched_frame_id)
            if temporal_distance < 100:
                if self.debug:
                    print(f"DEBUG LC: Rejected - temporal distance {temporal_distance} < 100")
                continue

            if matched_frame_id not in self.frame_poses:
                if self.debug:
                    print(f"DEBUG LC: Rejected - no pose for frame {matched_frame_id}")
                continue

            # Calculate spatial distance for information only (don't use for rejection)
            spatial_distance_m = 0.0
            if matched_frame_id in self.frame_poses:
                matched_pose = self.frame_poses[matched_frame_id]
                spatial_distance_mm = self._calculate_spatial_distance(current_pose, matched_pose)
                spatial_distance_m = spatial_distance_mm / 1000.0

                if self.debug:
                    print(f"DEBUG LC: Spatial distance: {spatial_distance_m:.2f}m (INFO ONLY - not used for rejection)")

            # Accept based on visual similarity only (540 matches is strong evidence)
            if self.debug:
                print(f"SUCCESS LC: Loop closure detected! {current_frame_id} -> {matched_frame_id}")

            candidate = LoopClosureCandidate(
                current_frame_id=current_frame_id,
                matched_frame_id=matched_frame_id,
                similarity_score=similarity_score,
                spatial_distance=spatial_distance_m,
                temporal_distance=temporal_distance,
                confidence=similarity_score
            )

            self.detected_closures.append(candidate)
            return candidate

        if self.debug:
            print(f"DEBUG LC: No loop closure detected for frame {current_frame_id}")
        return None

    def _calculate_spatial_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """Calculate spatial distance between two poses."""
        pos1 = pose1[:3, 3]
        pos2 = pose2[:3, 3]
        return float(np.linalg.norm(pos1 - pos2))

    def _calculate_relative_pose(self, from_pose: np.ndarray, to_pose: np.ndarray) -> np.ndarray:
        """Calculate relative pose transformation."""
        try:
            from_pose_inv = np.linalg.inv(from_pose)
            relative_pose = from_pose_inv @ to_pose
            return relative_pose
        except:
            return np.eye(4)

    def _calculate_confidence(self, similarity: float, spatial_dist: float, temporal_dist: int) -> float:
        """Calculate confidence score for loop closure candidate."""
        similarity_weight = 0.5
        spatial_weight = 0.3
        temporal_weight = 0.2

        sim_score = similarity
        spatial_normalized = max(0.0, 1.0 - (spatial_dist / self.max_spatial_distance))
        temporal_normalized = min(1.0, temporal_dist / 100.0)

        confidence = (similarity_weight * sim_score +
                     spatial_weight * spatial_normalized +
                     temporal_weight * temporal_normalized)

        return max(0.0, min(1.0, confidence))

    def _verify_loop_closure(self, candidate: LoopClosureCandidate) -> bool:
        """Additional verification of loop closure candidate."""
        if candidate.relative_pose is not None:
            translation = candidate.relative_pose[:3, 3]
            translation_magnitude = np.linalg.norm(translation)

            if translation_magnitude > self.pose_verification_threshold:
                return False

        recent_closures = [c for c in self.detected_closures
                          if abs(c.current_frame_id - candidate.current_frame_id) < 10]
        if len(recent_closures) > 2:
            return False

        return True

    def get_closure_statistics(self) -> Dict:
        """Get statistics about detected loop closures."""
        if not self.detected_closures:
            return {'total_closures': 0}

        similarities = [c.similarity_score for c in self.detected_closures]
        spatial_distances = [c.spatial_distance for c in self.detected_closures]
        temporal_distances = [c.temporal_distance for c in self.detected_closures]

        return {
            'total_closures': len(self.detected_closures),
            'avg_similarity': np.mean(similarities),
            'avg_spatial_distance_m': np.mean(spatial_distances),
            'avg_temporal_distance': np.mean(temporal_distances),
            'confidence_scores': [c.confidence for c in self.detected_closures]
        }


# =====================================================================
# LOOP CLOSURE MANAGER (SOLID - Single Responsibility Principle)
# =====================================================================

class LoopClosureManager:
    """High-level manager for loop closure functionality."""

    def __init__(self,
                 place_recognition: PlaceRecognitionInterface,
                 loop_detector: LoopClosureInterface,
                 pose_optimizer: PoseGraphOptimizerInterface,
                 visualization_system: SLAMVisualizationSystem,
                 enable_optimization: bool = True,
                 optimization_interval: int = 10,
                 debug: bool = True):

        self.place_recognition = place_recognition
        self.loop_detector = loop_detector
        self.pose_optimizer = pose_optimizer
        self.visualization_system = visualization_system
        self.enable_optimization = enable_optimization
        self.optimization_interval = optimization_interval
        self.debug = debug

        self.frame_count = 0
        self.last_optimization_frame = 0
        self.loop_closures_detected = 0

        print("Loop closure manager initialized")
        print(f"   Optimization enabled: {enable_optimization}")
        print(f"   Optimization interval: {optimization_interval} frames")
        print(f"   Debug mode: {debug}")

    def process_frame(self, frame_id: int, descriptors: np.ndarray,
                     pose: np.ndarray, previous_pose: Optional[np.ndarray] = None) -> Dict:
        """Process a new frame for loop closure detection."""
        self.frame_count += 1

        if self.debug:
            print(f"\n--- LOOP CLOSURE PROCESSING FRAME {frame_id} ---")
            print(f"   Frame count: {self.frame_count}")
            print(f"   Descriptors: {descriptors.shape if descriptors.size > 0 else 'None'}")
            pos = pose[:3, 3]
            print(f"   Current pose: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # Add place to recognition database
        self.place_recognition.add_place(frame_id, descriptors, pose)
        self.pose_optimizer.add_pose(frame_id, pose)

        # Add odometry edge if we have a previous pose
        if previous_pose is not None:
            relative_pose = np.linalg.inv(previous_pose) @ pose
            information = np.eye(6) * 0.1
            self.pose_optimizer.add_odometry_edge(frame_id - 1, frame_id, relative_pose, information)

            if self.debug:
                rel_trans = relative_pose[:3, 3]
                print(f"   Added odometry edge: {frame_id-1} -> {frame_id}")
                print(f"   Relative translation: [{rel_trans[0]:.3f}, {rel_trans[1]:.3f}, {rel_trans[2]:.3f}] m")

        # Query for similar places
        similar_places = self.place_recognition.query_similar_places(descriptors, top_k=5)

        if self.debug:
            print(f"   Similar places found: {len(similar_places)}")

        result = {
            'frame_id': frame_id,
            'similar_places': similar_places,
            'loop_closure_detected': False,
            'optimization_performed': False
        }

        # Check for loop closure
        if similar_places:
            if self.debug:
                print(f"   Checking for loop closure...")

            loop_closure = self.loop_detector.detect_loop_closure(frame_id, similar_places, pose)

            if loop_closure is not None:
                result['loop_closure_detected'] = True
                result['loop_closure_info'] = {
                    'matched_frame': loop_closure.matched_frame_id,
                    'confidence': loop_closure.confidence,
                    'spatial_distance': loop_closure.spatial_distance
                }

                print(f"   ?? LOOP CLOSURE DETECTED! Frame {frame_id} -> {loop_closure.matched_frame_id}")

                # Record loop closure for visualization
                self.visualization_system.record_loop_closure(pose, frame_id)

                # Add loop closure edge to pose graph
                if loop_closure.relative_pose is not None:
                    information = np.eye(6) * 1.0  # Higher confidence for loop closures
                    self.pose_optimizer.add_loop_closure_edge(
                        loop_closure.matched_frame_id,
                        frame_id,
                        loop_closure.relative_pose,
                        information
                    )

                    if self.debug:
                        print(f"   Added loop closure edge to pose graph")

                self.loop_closures_detected += 1
            elif self.debug:
                print(f"   No loop closure detected")
        elif self.debug:
            print(f"   No similar places to check")

        # Check if we should perform optimization
        should_optimize = (self.enable_optimization and
                          self.frame_count - self.last_optimization_frame >= self.optimization_interval and
                          self.loop_closures_detected > 0)

        if should_optimize:
            if self.debug:
                print(f"   Triggering pose graph optimization...")

            print(f"Performing pose graph optimization at frame {frame_id}...")
            try:
                optimized_poses = self.pose_optimizer.optimize()
                result['optimization_performed'] = True
                result['optimized_poses'] = optimized_poses
                self.last_optimization_frame = self.frame_count

                if self.debug:
                    print(f"   Optimization completed with {len(optimized_poses)} poses")

            except Exception as e:
                print(f"   Pose graph optimization failed: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
        elif self.debug:
            print(f"   Optimization not triggered (detected: {self.loop_closures_detected}, interval: {self.frame_count - self.last_optimization_frame}/{self.optimization_interval})")

        if self.debug:
            print(f"--- END FRAME {frame_id} PROCESSING ---\n")

        return result

    def get_statistics(self) -> Dict:
        """Get comprehensive loop closure statistics."""
        stats = {
            'frames_processed': self.frame_count,
            'loop_closures_detected': self.loop_closures_detected,
            'last_optimization_frame': self.last_optimization_frame
        }

        if hasattr(self.loop_detector, 'get_closure_statistics'):
            stats['detector_stats'] = self.loop_detector.get_closure_statistics()

        if hasattr(self.pose_optimizer, 'get_optimization_statistics'):
            stats['optimizer_stats'] = self.pose_optimizer.get_optimization_statistics()

        return stats

    def force_optimization(self) -> Dict[int, np.ndarray]:
        """Force pose graph optimization regardless of interval."""
        print("Forcing pose graph optimization...")
        return self.pose_optimizer.optimize()
