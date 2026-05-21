# Visual SLAM — Refactored Package

Complete Visual SLAM implementation with loop closure, GTSAM pose graph optimization,
ASIFT feature extraction, and PnP/Kabsch pose estimation comparison.

## Install

```bash
conda activate slam
pip install -e .
```

## Run

```bash
python -m visual_slam.main
```

Update `image_folder` in `visual_slam/main.py` to point to your stereo image dataset.

## Module Layout

```
visual_slam/
├── __init__.py                 # Re-exports public API; fires module-level side effects
├── interfaces.py               # Protocol classes (PlaceRecognitionInterface, etc.)
├── data_structures.py          # @dataclass types (PlaceDescriptor, SLAMFrame, CAHVCamera…)
├── config.py                   # DatasetAdaptiveParameters — all thresholds and defaults
├── place_recognition.py        # DescriptorBasedPlaceRecognition
├── loop_closure.py             # SpatialTemporalLoopClosureDetector + LoopClosureManager
├── pose_graph/
│   ├── __init__.py
│   ├── custom_optimizer.py     # LeastSquaresPoseGraphOptimizer
│   └── gtsam_optimizer.py      # GTSAMPoseGraphOptimizer (guarded import)
├── feature_extraction.py       # OptimizedASIFTMatcher
├── stereo_matching.py          # StereoImageLoader
├── temporal_matching.py        # SmartTemporalMatcher
├── triangulation.py            # RobustTriangulator
├── correspondence_validation.py# Enhanced3DCorrespondenceFinder (3-stage)
├── slam_core.py                # ImprovedVisualSLAM
├── slam_enhanced.py            # EnhancedVisualSLAMWithLoopClosure
├── visualisation.py            # SLAMVisualizationSystem
└── main.py                     # main_working_slam() + __main__ block
```

## Tests

```bash
conda activate slam
python -m unittest discover -s tests -v
```

## Key Parameters

| Parameter | Value |
|---|---|
| ASIFT max_tilts | 6 (8 for non-rectified) |
| ASIFT ratio_threshold | 0.8 |
| Stereo epipolar threshold | 50–200 px (adaptive) |
| Temporal fundamental threshold | 3–5 px (adaptive) |
| Triangulation min/max depth | 10 mm / 500 000 mm |
| RANSAC iterations | 1000 |
| RANSAC threshold | 500 mm |
| Correspondence spatial threshold | 2000 mm |
| Loop closure temporal guard | 100 frames |

## Dependencies

- numpy ≥ 1.22
- opencv-python ≥ 4.5
- scikit-learn ≥ 1.0
- scipy ≥ 1.7
- matplotlib ≥ 3.5
- gtsam ≥ 4.0 (optional — falls back to custom least-squares optimizer)
