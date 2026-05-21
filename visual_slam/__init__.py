"""
visual_slam package — re-exports public API and fires module-level side effects
in the same order as the original monolithic_code.py.

Side-effect order preserved:
  1. GTSAM try/except import (prints status, sets GTSAM_AVAILABLE)
  2. NumPy version print
"""

import numpy as np

# =====================================================================
# GTSAM optional import — must be module-level, same as original
# =====================================================================
try:
    import gtsam
    from gtsam import (
        symbol_shorthand, Pose3, Rot3, Point3, Point2, Cal3_S2,
        PinholeCameraCal3_S2, NonlinearFactorGraph,
        LevenbergMarquardtOptimizer, Values, noiseModel,
        BetweenFactorPose3, PriorFactorPose3
    )
    GTSAM_AVAILABLE = True
    print("GTSAM successfully imported - Advanced optimization available")
except ImportError:
    GTSAM_AVAILABLE = False
    print("GTSAM not available. Install with: pip install gtsam")
    print("Using custom optimization only...")

print(f"NumPy version: {np.__version__}")

# =====================================================================
# Public API re-exports
# =====================================================================
from visual_slam.interfaces import (
    PlaceRecognitionInterface,
    LoopClosureInterface,
    PoseGraphOptimizerInterface,
)
from visual_slam.data_structures import (
    PlaceDescriptor,
    LoopClosureCandidate,
    PoseGraphEdge,
    CAHVCamera,
)
from visual_slam.config import DatasetAdaptiveParameters
from visual_slam.place_recognition import DescriptorBasedPlaceRecognition
from visual_slam.loop_closure import (
    SpatialTemporalLoopClosureDetector,
    LoopClosureManager,
)
from visual_slam.pose_graph.custom_optimizer import LeastSquaresPoseGraphOptimizer
from visual_slam.pose_graph.gtsam_optimizer import GTSAMPoseGraphOptimizer
from visual_slam.visualisation import SLAMVisualizationSystem
from visual_slam.feature_extraction import OptimizedASIFTMatcher
from visual_slam.stereo_matching import StereoImageLoader
from visual_slam.temporal_matching import SmartTemporalMatcher
from visual_slam.triangulation import RobustTriangulator
from visual_slam.correspondence_validation import Enhanced3DCorrespondenceFinder
from visual_slam.slam_core import ImprovedVisualSLAM
from visual_slam.slam_enhanced import EnhancedVisualSLAMWithLoopClosure
from visual_slam.main import main_working_slam

__all__ = [
    # Flag
    'GTSAM_AVAILABLE',
    # Interfaces
    'PlaceRecognitionInterface',
    'LoopClosureInterface',
    'PoseGraphOptimizerInterface',
    # Data structures
    'PlaceDescriptor',
    'LoopClosureCandidate',
    'PoseGraphEdge',
    'CAHVCamera',
    # Config
    'DatasetAdaptiveParameters',
    # Place recognition
    'DescriptorBasedPlaceRecognition',
    # Loop closure
    'SpatialTemporalLoopClosureDetector',
    'LoopClosureManager',
    # Pose graph optimizers
    'LeastSquaresPoseGraphOptimizer',
    'GTSAMPoseGraphOptimizer',
    # Visualization
    'SLAMVisualizationSystem',
    # Feature extraction
    'OptimizedASIFTMatcher',
    # Stereo/image loading
    'StereoImageLoader',
    # Temporal matching
    'SmartTemporalMatcher',
    # Triangulation
    'RobustTriangulator',
    # Correspondence validation
    'Enhanced3DCorrespondenceFinder',
    # SLAM systems
    'ImprovedVisualSLAM',
    'EnhancedVisualSLAMWithLoopClosure',
    # Entry point
    'main_working_slam',
]
