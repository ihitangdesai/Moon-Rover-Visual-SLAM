"""
INTERFACES (SOLID - Interface Segregation Principle)
"""

from typing import Tuple, List, Dict, Optional, Any, Protocol
import numpy as np


class PlaceRecognitionInterface(Protocol):
    """Interface for place recognition systems."""

    def add_place(self, place_id: int, descriptors: np.ndarray, pose: np.ndarray) -> None:
        """Add a new place to the database."""
        ...

    def query_similar_places(self, descriptors: np.ndarray,
                           top_k: int = 5) -> List[Tuple[int, float]]:
        """Query for similar places."""
        ...


class LoopClosureInterface(Protocol):
    """Interface for loop closure detection."""

    def detect_loop_closure(self, current_frame_id: int,
                          candidate_matches: List[Tuple[int, float]],
                          current_pose: np.ndarray) -> Optional[Any]:
        """Detect loop closure and return match info."""
        ...


class PoseGraphOptimizerInterface(Protocol):
    """Interface for pose graph optimization."""

    def add_pose(self, pose_id: int, pose: np.ndarray) -> None:
        """Add a pose to the graph."""
        ...

    def add_odometry_edge(self, from_id: int, to_id: int,
                         relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add odometry edge."""
        ...

    def add_loop_closure_edge(self, from_id: int, to_id: int,
                            relative_pose: np.ndarray, information: np.ndarray) -> None:
        """Add loop closure edge."""
        ...

    def optimize(self) -> Dict[int, np.ndarray]:
        """Optimize the pose graph and return optimized poses."""
        ...
