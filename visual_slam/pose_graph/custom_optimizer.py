"""
CUSTOM POSE GRAPH OPTIMIZATION (SOLID - Single Responsibility)
"""

import numpy as np
import cv2 as cv
from typing import List, Dict
from scipy.optimize import least_squares

from visual_slam.data_structures import PoseGraphEdge


class LeastSquaresPoseGraphOptimizer:
    """Custom pose graph optimization using least squares method."""

    def __init__(self, max_iterations: int = 50, convergence_threshold: float = 1e-6):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.poses: Dict[int, np.ndarray] = {}
        self.edges: List[PoseGraphEdge] = []
        self.optimizer_type = "Custom Least Squares"

        print("Custom least squares pose graph optimizer initialized")

    def add_pose(self, pose_id: int, pose: np.ndarray) -> None:
        """Add a pose to the graph."""
        self.poses[pose_id] = pose.copy()

    def add_odometry_edge(self, from_id: int, to_id: int,
                         relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add odometry edge to the graph."""
        edge = PoseGraphEdge(
            from_id=from_id,
            to_id=to_id,
            relative_pose=relative_pose.copy(),
            information=information.copy(),
            edge_type='odometry'
        )
        self.edges.append(edge)

    def add_loop_closure_edge(self, from_id: int, to_id: int,
                            relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add loop closure edge to the graph."""
        edge = PoseGraphEdge(
            from_id=from_id,
            to_id=to_id,
            relative_pose=relative_pose.copy(),
            information=information.copy(),
            edge_type='loop_closure'
        )
        self.edges.append(edge)
        print(f"Custom optimizer: Added loop closure edge: {from_id} -> {to_id}")

    def optimize(self) -> Dict[int, np.ndarray]:
        """Optimize the pose graph using least squares."""
        if len(self.poses) < 2 or len(self.edges) == 0:
            return self.poses.copy()

        print(f"Starting custom pose graph optimization...")
        print(f"   Poses: {len(self.poses)}, Edges: {len(self.edges)}")

        try:
            pose_ids = sorted(self.poses.keys())
            initial_variables = self._poses_to_variables(pose_ids)

            result = least_squares(
                self._residual_function,
                initial_variables,
                args=(pose_ids,),
                max_nfev=self.max_iterations * len(initial_variables),
                ftol=self.convergence_threshold,
                xtol=self.convergence_threshold,
                method='lm'
            )

            optimized_poses = self._variables_to_poses(result.x, pose_ids)

            initial_cost = np.sum(self._residual_function(initial_variables, pose_ids) ** 2)
            final_cost = np.sum(result.fun ** 2)
            improvement = (initial_cost - final_cost) / initial_cost * 100 if initial_cost > 0 else 0

            print(f"Custom pose graph optimization completed")
            print(f"   Cost reduction: {improvement:.2f}%")
            print(f"   Iterations: {result.nfev}")
            print(f"   Success: {result.success}")

            return optimized_poses

        except Exception as e:
            print(f"Custom pose graph optimization failed: {e}")
            return self.poses.copy()

    def _poses_to_variables(self, pose_ids: List[int]) -> np.ndarray:
        """Convert poses to optimization variables."""
        variables = []

        for pose_id in pose_ids:
            pose = self.poses[pose_id]

            translation = pose[:3, 3]
            variables.extend(translation.flatten())

            rotation_matrix = pose[:3, :3]
            rotation_vector, _ = cv.Rodrigues(rotation_matrix.astype(np.float64))
            variables.extend(rotation_vector.flatten())

        return np.array(variables, dtype=np.float64)

    def _variables_to_poses(self, variables: np.ndarray, pose_ids: List[int]) -> Dict[int, np.ndarray]:
        """Convert optimization variables back to poses."""
        poses = {}

        for i, pose_id in enumerate(pose_ids):
            start_idx = i * 6

            translation = variables[start_idx:start_idx+3]
            rotation_vector = variables[start_idx+3:start_idx+6]

            rotation_matrix, _ = cv.Rodrigues(rotation_vector.astype(np.float64))

            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix
            pose[:3, 3] = translation

            poses[pose_id] = pose

        return poses

    def _residual_function(self, variables: np.ndarray, pose_ids: List[int]) -> np.ndarray:
        """Calculate residuals for all edges."""
        poses = self._variables_to_poses(variables, pose_ids)
        residuals = []

        for edge in self.edges:
            if edge.from_id in poses and edge.to_id in poses:
                residual = self._calculate_edge_residual(edge, poses[edge.from_id], poses[edge.to_id])
                residuals.extend(residual)

        return np.array(residuals, dtype=np.float64)

    def _calculate_edge_residual(self, edge: PoseGraphEdge,
                               from_pose: np.ndarray, to_pose: np.ndarray) -> List[float]:
        """Calculate residual for a single edge."""
        try:
            from_pose_inv = np.linalg.inv(from_pose)
            predicted_relative = from_pose_inv @ to_pose

            error_pose = np.linalg.inv(edge.relative_pose) @ predicted_relative

            translation_error = error_pose[:3, 3]

            rotation_error_matrix = error_pose[:3, :3]
            rotation_error_vector, _ = cv.Rodrigues(rotation_error_matrix.astype(np.float64))
            rotation_error_vector = rotation_error_vector.flatten()

            error_vector = np.concatenate([translation_error, rotation_error_vector])

            if edge.edge_type == 'loop_closure':
                error_vector *= 2.0

            return error_vector.tolist()

        except Exception as e:
            return [0.0] * 6

    def get_optimization_statistics(self) -> Dict:
        """Get statistics about the pose graph."""
        odometry_edges = [e for e in self.edges if e.edge_type == 'odometry']
        loop_closure_edges = [e for e in self.edges if e.edge_type == 'loop_closure']

        return {
            'optimizer_type': self.optimizer_type,
            'total_poses': len(self.poses),
            'total_edges': len(self.edges),
            'odometry_edges': len(odometry_edges),
            'loop_closure_edges': len(loop_closure_edges),
            'edge_ratio': len(self.edges) / max(1, len(self.poses)),
            'loop_closure_ratio': len(loop_closure_edges) / max(1, len(self.edges))
        }
