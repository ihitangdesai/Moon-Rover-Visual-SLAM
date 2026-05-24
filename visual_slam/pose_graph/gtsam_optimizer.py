"""
GTSAM POSE GRAPH OPTIMIZATION (SOLID - Single Responsibility)

GTSAM import is guarded exactly as in the original — GTSAM_AVAILABLE is
resolved at call time from the top-level visual_slam namespace.
"""

import numpy as np
from typing import Dict, List


class GTSAMPoseGraphOptimizer:
    """GTSAM-based pose graph optimization using factor graph framework."""

    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 1e-6):
        from visual_slam import GTSAM_AVAILABLE
        if not GTSAM_AVAILABLE:
            raise ImportError("GTSAM is required for GTSAMPoseGraphOptimizer")

        import gtsam

        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.optimizer_type = "GTSAM Factor Graph"

        # GTSAM components
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.poses: Dict[int, np.ndarray] = {}
        self.pose_symbols = {}  # Map pose_id to GTSAM symbol

        # Noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # translation, rotation
        )
        self.loop_closure_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])  # tighter for loop closures
        )
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # very tight for anchor
        )

        # Edge tracking
        self.edges = []
        self.first_pose_anchored = False

        print("GTSAM factor graph pose optimizer initialized")

    def add_pose(self, pose_id: int, pose: np.ndarray) -> None:
        """Add a pose to the GTSAM graph."""
        import gtsam

        self.poses[pose_id] = pose.copy()

        # Create GTSAM symbol for this pose
        symbol = gtsam.symbol_shorthand.X(pose_id)
        self.pose_symbols[pose_id] = symbol

        # Convert numpy pose to GTSAM Pose3
        gtsam_pose = self._numpy_to_gtsam_pose(pose)
        self.initial_estimates.insert(symbol, gtsam_pose)

        # Add prior factor to anchor the first pose
        if not self.first_pose_anchored:
            prior_factor = gtsam.PriorFactorPose3(symbol, gtsam_pose, self.prior_noise)
            self.graph.add(prior_factor)
            self.first_pose_anchored = True
            print(f"GTSAM: Anchored first pose {pose_id} with prior factor")

    def add_odometry_edge(self, from_id: int, to_id: int,
                         relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add odometry edge using GTSAM BetweenFactorPose3."""
        import gtsam

        if from_id not in self.pose_symbols or to_id not in self.pose_symbols:
            # Auto-add missing poses with identity poses
            if from_id not in self.pose_symbols:
                self.add_pose(from_id, np.eye(4))
            if to_id not in self.pose_symbols:
                self.add_pose(to_id, np.eye(4))
            print(f"Auto-added missing poses for odometry edge {from_id}->{to_id}")

        from_symbol = self.pose_symbols[from_id]
        to_symbol = self.pose_symbols[to_id]

        # Convert relative pose to GTSAM Pose3
        gtsam_relative_pose = self._numpy_to_gtsam_pose(relative_pose)

        edge_noise = self.odometry_noise

        # Create between factor
        between_factor = gtsam.BetweenFactorPose3(
            from_symbol, to_symbol, gtsam_relative_pose, edge_noise
        )

        self.graph.add(between_factor)

        # Store edge info
        edge_info = {
            'from_id': from_id,
            'to_id': to_id,
            'relative_pose': relative_pose.copy(),
            'information': information.copy(),
            'edge_type': 'odometry'
        }
        self.edges.append(edge_info)

    def add_loop_closure_edge(self, from_id: int, to_id: int,
                            relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add loop closure edge using GTSAM BetweenFactorPose3 with tighter noise model."""
        import gtsam

        if from_id not in self.pose_symbols or to_id not in self.pose_symbols:
            print(f"Cannot add loop closure edge {from_id}->{to_id}: poses not in graph")
            return

        from_symbol = self.pose_symbols[from_id]
        to_symbol = self.pose_symbols[to_id]

        # Convert relative pose to GTSAM Pose3
        gtsam_relative_pose = self._numpy_to_gtsam_pose(relative_pose)

        # Create between factor with tighter noise model for loop closures
        between_factor = gtsam.BetweenFactorPose3(
            from_symbol, to_symbol, gtsam_relative_pose, self.loop_closure_noise
        )

        self.graph.add(between_factor)

        # Store edge info
        edge_info = {
            'from_id': from_id,
            'to_id': to_id,
            'relative_pose': relative_pose.copy(),
            'information': information.copy(),
            'edge_type': 'loop_closure'
        }
        self.edges.append(edge_info)

        print(f"GTSAM optimizer: Added loop closure edge: {from_id} -> {to_id}")

    def optimize(self) -> Dict[int, np.ndarray]:
        """Optimize the pose graph using GTSAM's LevenbergMarquardtOptimizer."""
        import gtsam

        if len(self.poses) < 2 or self.graph.size() == 0:
            return self.poses.copy()

        print(f"Starting GTSAM pose graph optimization...")
        print(f"   Poses: {len(self.poses)}, Factors: {self.graph.size()}")

        try:
            # Calculate initial error
            initial_error = self.graph.error(self.initial_estimates)

            # Set up optimizer parameters
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(self.max_iterations)
            params.setRelativeErrorTol(self.convergence_threshold)
            params.setAbsoluteErrorTol(self.convergence_threshold)

            # Optimize
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
            result = optimizer.optimize()

            # Calculate final error
            final_error = self.graph.error(result)
            improvement = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0

            # Convert results back to numpy poses
            optimized_poses = {}
            for pose_id, symbol in self.pose_symbols.items():
                if result.exists(symbol):
                    gtsam_pose = result.atPose3(symbol)
                    numpy_pose = self._gtsam_to_numpy_pose(gtsam_pose)
                    optimized_poses[pose_id] = numpy_pose

            print(f"GTSAM pose graph optimization completed")
            print(f"   Initial error: {initial_error:.6f}")
            print(f"   Final error: {final_error:.6f}")
            print(f"   Cost reduction: {improvement:.2f}%")
            print(f"   Iterations: {optimizer.iterations()}")

            return optimized_poses

        except Exception as e:
            print(f"GTSAM pose graph optimization failed: {e}")
            return self.poses.copy()

    def _numpy_to_gtsam_pose(self, pose: np.ndarray) -> 'gtsam.Pose3':
        """Convert numpy 4x4 pose matrix to GTSAM Pose3."""
        import gtsam
        translation = gtsam.Point3(pose[0, 3], pose[1, 3], pose[2, 3])
        rotation = gtsam.Rot3(pose[:3, :3])
        return gtsam.Pose3(rotation, translation)

    def _gtsam_to_numpy_pose(self, gtsam_pose: 'gtsam.Pose3') -> np.ndarray:
        """Convert GTSAM Pose3 to numpy 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, :3] = gtsam_pose.rotation().matrix()
        pose[:3, 3] = gtsam_pose.translation()
        return pose

    def get_optimization_statistics(self) -> Dict:
        """Get statistics about the GTSAM pose graph."""
        odometry_edges = [e for e in self.edges if e['edge_type'] == 'odometry']
        loop_closure_edges = [e for e in self.edges if e['edge_type'] == 'loop_closure']

        return {
            'optimizer_type': self.optimizer_type,
            'total_poses': len(self.poses),
            'total_factors': self.graph.size(),
            'total_edges': len(self.edges),
            'odometry_edges': len(odometry_edges),
            'loop_closure_edges': len(loop_closure_edges),
            'edge_ratio': len(self.edges) / max(1, len(self.poses)),
            'loop_closure_ratio': len(loop_closure_edges) / max(1, len(self.edges))
        }
