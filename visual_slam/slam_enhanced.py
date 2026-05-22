"""
ENHANCED SLAM WITH LOOP CLOSURE - EnhancedVisualSLAMWithLoopClosure wrapper.
"""

import numpy as np
import time
from typing import List, Dict, Optional

from visual_slam.slam_core import ImprovedVisualSLAM
from visual_slam.stereo_matching import StereoImageLoader
from visual_slam.place_recognition import DescriptorBasedPlaceRecognition
from visual_slam.loop_closure import SpatialTemporalLoopClosureDetector, LoopClosureManager
from visual_slam.pose_graph.custom_optimizer import LeastSquaresPoseGraphOptimizer
from visual_slam.pose_graph.gtsam_optimizer import GTSAMPoseGraphOptimizer


class EnhancedVisualSLAMWithLoopClosure:
    """
    Enhanced Visual SLAM system with loop closure detection and pose graph optimization.
    """

    def __init__(self, left_cahv: Dict, right_cahv: Dict,
                 use_gtsam: bool = True,
                 enable_loop_closure: bool = True,
                 show_lines: bool = False,
                 output_dir=None):

        from visual_slam import GTSAM_AVAILABLE

        # Initialize the working SLAM system
        self.original_slam = ImprovedVisualSLAM(left_cahv, right_cahv, show_lines,
                                                output_dir=output_dir)

        # Store configuration
        self.left_cahv = left_cahv
        self.right_cahv = right_cahv
        self.use_gtsam = use_gtsam and GTSAM_AVAILABLE
        self.enable_loop_closure = enable_loop_closure

        # Frame tracking
        self.current_frame_id = 0

        # Loop closure components
        if enable_loop_closure:
            place_recognition = DescriptorBasedPlaceRecognition(
                similarity_threshold=0.4,  # LOWERED for debugging
                min_matches=8,  # LOWERED for debugging
                use_clustering=True,
                debug=True  # ENABLE DEBUG
            )

            loop_detector = SpatialTemporalLoopClosureDetector(
                min_temporal_distance=10,  # LOWERED from 30
                max_spatial_distance=50.0,  # INCREASED for debugging
                min_similarity_score=0.3,  # LOWERED for debugging
                debug=True  # ENABLE DEBUG
            )

            # Choose optimizer based on configuration
            if self.use_gtsam:
                try:
                    pose_optimizer = GTSAMPoseGraphOptimizer(
                        max_iterations=50,
                        convergence_threshold=1e-6
                    )
                    print("Using GTSAM pose graph optimizer")
                except Exception as e:
                    print(f"GTSAM optimizer failed to initialize: {e}")
                    pose_optimizer = LeastSquaresPoseGraphOptimizer(
                        max_iterations=50,
                        convergence_threshold=1e-6
                    )
                    self.use_gtsam = False
                    print("Falling back to custom least squares optimizer")
            else:
                pose_optimizer = LeastSquaresPoseGraphOptimizer(
                    max_iterations=50,
                    convergence_threshold=1e-6
                )
                print("Using custom least squares optimizer")

            self.loop_closure_manager = LoopClosureManager(
                place_recognition=place_recognition,
                loop_detector=loop_detector,
                pose_optimizer=pose_optimizer,
                visualization_system=self.original_slam.visualization_system,
                enable_optimization=True,
                optimization_interval=5,  # REDUCED for faster debugging
                debug=True  # ENABLE DEBUG
            )

        # Storage for comparison data
        self.optimization_history = []

        print("Working Enhanced Visual SLAM with Loop Closure Initialized")
        print(f"   Optimizer: {'GTSAM Factor Graph' if self.use_gtsam else 'Custom Least Squares'}")
        print(f"   Loop closure enabled: {'Yes' if enable_loop_closure else 'No'}")
        print(f"   Show connection lines: {show_lines}")
        print(f"   Camera baseline: {self.original_slam.adaptive_params.baseline:.3f} mm")
        print(f"   Working parameters: Applied")
        print(f"   PnP pose estimation: Available for comparison")

    def process_frame_pair_with_loop_closure(self, left_t: np.ndarray, right_t: np.ndarray,
                                            left_t1: np.ndarray, right_t1: np.ndarray,
                                            frame_info: Dict = None) -> Dict:
        """
        Process frame pair with complete SLAM pipeline including loop closure.
        """

        frame_desc = f"frames {frame_info.get('frame_t_number', '?')} -> {frame_info.get('frame_t1_number', '?')}" if frame_info else "frame pair"
        print(f"Processing {frame_desc} with Working Enhanced Loop Closure SLAM...")

        start_time = time.time()
        self.current_frame_id += 1

        # 1. Process with the complete working SLAM system first
        print("Running complete Working Visual SLAM pipeline...")
        original_result = self.original_slam.process_frame_pair(
            left_t, right_t, left_t1, right_t1, frame_info
        )

        # 2. Add optimizer type information
        optimizer_type = "GTSAM Factor Graph" if self.use_gtsam else "Custom Least Squares"
        original_result['optimizer_type'] = optimizer_type

        # 3. Add loop closure processing if enabled and successful
        loop_closure_result = {}
        if self.enable_loop_closure and original_result.get('success', False):
            try:
                print("Adding loop closure processing...")

                # Extract features for loop closure
                left_kp_t1, left_desc_t1 = self.original_slam.extract_features(left_t1)

                if left_desc_t1.size > 0:
                    current_pose = self.original_slam.current_pose
                    previous_pose = self.original_slam.trajectory[-2] if len(self.original_slam.trajectory) > 1 else None

                    # Store poses before optimization
                    poses_before_optimization = {}
                    if hasattr(self.loop_closure_manager.pose_optimizer, 'poses'):
                        poses_before_optimization = self.loop_closure_manager.pose_optimizer.poses.copy()

                    loop_closure_result = self.loop_closure_manager.process_frame(
                        frame_id=self.current_frame_id,
                        descriptors=left_desc_t1,
                        pose=current_pose,
                        previous_pose=previous_pose
                    )

                    # Check for optimization
                    if loop_closure_result.get('optimization_performed', False):
                        optimized_poses = loop_closure_result.get('optimized_poses', {})

                        if optimized_poses and poses_before_optimization:
                            print("Pose graph optimization performed!")

                            # Update trajectory with optimized poses
                            for i, pose_id in enumerate(sorted(optimized_poses.keys())):
                                if i < len(self.original_slam.trajectory):
                                    self.original_slam.trajectory[i] = optimized_poses[pose_id]

                            # Update current pose
                            if self.current_frame_id in optimized_poses:
                                self.original_slam.current_pose = optimized_poses[self.current_frame_id]

                            # Store optimization data
                            self.optimization_history.append({
                                'frame_id': self.current_frame_id,
                                'optimizer_type': optimizer_type,
                                'poses_before': poses_before_optimization.copy(),
                                'poses_after': optimized_poses.copy(),
                                'timestamp': time.time()
                            })

            except Exception as e:
                print(f"   Loop closure processing failed: {e}")
                loop_closure_result = {'error': str(e)}

        # 4. Combine results
        processing_time = time.time() - start_time

        enhanced_result = original_result.copy()
        enhanced_result.update({
            'loop_closure_enabled': self.enable_loop_closure,
            'loop_closure_result': loop_closure_result,
            'enhanced_processing_time': processing_time,
            'original_processing_time': original_result.get('processing_time', 0),
            'system_type': 'working_enhanced_with_loop_closure',
            'optimizer_type': optimizer_type
        })

        # Print summary
        if loop_closure_result.get('loop_closure_detected', False):
            print(f"   Loop closure detected with {optimizer_type}!")
            info = loop_closure_result.get('loop_closure_info', {})
            print(f"      Matched frame: {info.get('matched_frame', 'N/A')}")
            print(f"      Confidence: {info.get('confidence', 0):.3f}")

        if loop_closure_result.get('optimization_performed', False):
            print(f"   Pose graph optimization performed with {optimizer_type}!")

        return enhanced_result

    def process_image_sequence(self, image_loader: StereoImageLoader,
                              max_frames: Optional[int] = None,
                              start_frame: int = 0) -> List[Dict]:
        """
        Process image sequence with loop closure.
        """
        sequence_info = image_loader.get_info()
        print(f"\nProcessing stereo sequence with Working Enhanced Loop Closure SLAM:")
        print(f"   Folder: {sequence_info['folder_path']}")
        print(f"   Total pairs: {sequence_info['num_pairs']}")
        print(f"   Frame range: {sequence_info['frame_range']}")
        print(f"   Optimizer: {'GTSAM Factor Graph' if self.use_gtsam else 'Custom Least Squares'}")
        print(f"   Loop closure: {'Yes' if self.enable_loop_closure else 'No'}")
        print(f"   Working parameters: Active")
        print(f"   PnP comparison: Active")

        consecutive_pairs = image_loader.get_consecutive_pairs(
            start_index=start_frame,
            count=max_frames
        )

        if not consecutive_pairs:
            print("No consecutive pairs available for processing")
            return []

        print(f"   Processing {len(consecutive_pairs)} consecutive frame pairs...")
        print()

        sequence_results = []
        total_start_time = time.time()

        for i, pair_data in enumerate(consecutive_pairs):
            pair_num = i + 1
            print(f"={'='*80}")
            print(f"Processing pair {pair_num}/{len(consecutive_pairs)}: "
                  f"Frame {pair_data['frame_t_number']} -> {pair_data['frame_t1_number']}")
            print(f"={'='*80}")

            result = self.process_frame_pair_with_loop_closure(
                pair_data['left_t'], pair_data['right_t'],
                pair_data['left_t1'], pair_data['right_t1'],
                frame_info=pair_data
            )

            result['pair_number'] = pair_num
            result['total_pairs'] = len(consecutive_pairs)

            sequence_results.append(result)

            # Progress update
            elapsed_time = time.time() - total_start_time
            avg_time_per_pair = elapsed_time / pair_num
            remaining_pairs = len(consecutive_pairs) - pair_num
            estimated_remaining_time = remaining_pairs * avg_time_per_pair

            print(f"\nProgress: {pair_num}/{len(consecutive_pairs)} pairs completed")
            print(f"   Elapsed: {elapsed_time:.1f}s, Estimated remaining: {estimated_remaining_time:.1f}s")
            print()

        total_time = time.time() - total_start_time
        successful_pairs = sum(1 for r in sequence_results if r['success'])

        print(f"={'='*80}")
        print(f"Working Enhanced Loop Closure SLAM sequence processing complete!")
        print(f"   Successful pairs: {successful_pairs}/{len(consecutive_pairs)}")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Average time per pair: {total_time/len(consecutive_pairs):.1f} seconds")
        print(f"   Optimizer used: {'GTSAM Factor Graph' if self.use_gtsam else 'Custom Least Squares'}")

        # Loop closure and optimization statistics
        if self.enable_loop_closure:
            lc_stats = self.get_loop_closure_statistics()
            print(f"   Loop closures detected: {lc_stats.get('loop_closures_detected', 0)}")
            print(f"   Optimizations performed: {len(self.optimization_history)}")

            if 'optimizer_stats' in lc_stats:
                opt_stats = lc_stats['optimizer_stats']
                print(f"   Pose graph edges: {opt_stats.get('total_edges', 0)}")
                print(f"   Loop closure edges: {opt_stats.get('loop_closure_edges', 0)}")

        # Pose estimation method statistics
        kabsch_count = sum(1 for r in sequence_results if r.get('chosen_method') == 'Kabsch+RANSAC')
        pnp_count = sum(1 for r in sequence_results if r.get('chosen_method') == 'PnP+RANSAC')
        print(f"   Pose estimation methods:")
        print(f"      Kabsch+RANSAC: {kabsch_count} frames")
        print(f"      PnP+RANSAC: {pnp_count} frames")

        # Create final 2D trajectory map with loop closure points
        print("\nCreating final 2D trajectory map...")
        map_path = self.original_slam.visualization_system.create_2d_trajectory_map(
            self.original_slam.trajectory, "working_trajectory_2d_map.png"
        )

        # Show interactive 3D plot
        print("\nDisplaying interactive 3D trajectory...")
        try:
            self.original_slam.create_interactive_3d_trajectory_plot(show_plot=True)
        except Exception as e:
            print(f"Failed to show 3D plot: {e}")

        print(f"\n   Working Enhancements applied:")
        print(f"      Working parameters that produce good matches: ?")
        print(f"      Complete 3-stage correspondence validation: ?")
        print(f"      {'GTSAM' if self.use_gtsam else 'Custom'} pose graph optimization: ?")
        print(f"      Loop closure detection: {'?' if self.enable_loop_closure else 'Disabled'}")
        print(f"      PnP vs Kabsch pose estimation comparison: ?")
        print(f"      Stereo matches visualization: ?")
        print(f"      Temporal matches visualization: ?")
        print(f"      Interactive 3D trajectory: ?")
        print(f"      2D trajectory map: ?")
        print(f"      Statistical outlier removal (Stage 3): ?")
        print(f"={'='*80}")

        return sequence_results

    def get_loop_closure_statistics(self) -> Dict:
        """Get comprehensive loop closure statistics."""
        if not self.enable_loop_closure:
            return {'loop_closure_enabled': False}

        stats = self.loop_closure_manager.get_statistics()
        stats['loop_closure_enabled'] = True
        stats['optimizer_type'] = "GTSAM Factor Graph" if self.use_gtsam else "Custom Least Squares"

        return stats

    def get_trajectory_summary(self) -> Dict:
        """Get trajectory summary from working SLAM system."""
        summary = self.original_slam.get_trajectory_summary()
        summary['loop_closure_enabled'] = self.enable_loop_closure
        summary['optimizer_type'] = "GTSAM Factor Graph" if self.use_gtsam else "Custom Least Squares"
        summary['system_type'] = 'Working Parameters'
        return summary

    @property
    def current_pose(self) -> np.ndarray:
        """Access current pose from working SLAM system."""
        return self.original_slam.current_pose

    @property
    def trajectory(self) -> List[np.ndarray]:
        """Access trajectory from working SLAM system."""
        return self.original_slam.trajectory
