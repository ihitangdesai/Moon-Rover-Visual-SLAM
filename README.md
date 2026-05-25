# 🌕 Moon Rover Visual SLAM

> **Status: Active Development**

A Visual SLAM (Simultaneous Localisation and Mapping) system designed for lunar rover navigation, developed and tested on the [LuSNAR Moon_1 dataset](https://github.com/lunarSLAM/LuSNAR). Built from the ground up with a CAHV camera model, GTSAM pose graph optimisation, ASIFT feature extraction, and PnP/Kabsch pose estimation.

---

## Developer

**Hitang Desai**
Visual SLAM Engineer
[GitHub](https://github.com/ihitangdesai)

---

## Overview

This system processes stereo image pairs from a lunar rover and estimates its 6-DoF trajectory in real time. It is validated against ground-truth poses from the LuSNAR benchmark dataset.

**Current results on LuSNAR Moon_1 (100-frame test):**

| Metric | Value |
|---|---|
| ATE RMSE | 0.115 m |
| Drift | 0.40% of path |
| PnP success rate | 100% |
| Spike frames | 0 |

---

## Architecture

```
visual_slam/
├── __init__.py                   # Public API; GTSAM availability flag
├── interfaces.py                 # Protocol classes
├── data_structures.py            # @dataclass types (PlaceDescriptor, SLAMFrame, CAHVCamera…)
├── config.py                     # DatasetAdaptiveParameters — all thresholds and defaults
├── feature_extraction.py         # OptimizedASIFTMatcher (ASIFT/SIFT, 1200 features)
├── stereo_matching.py            # StereoImageLoader
├── temporal_matching.py          # SmartTemporalMatcher
├── triangulation.py              # RobustTriangulator (CAHV-based)
├── correspondence_validation.py  # Enhanced3DCorrespondenceFinder (3-stage)
├── bundle_adjustment.py          # LocalBundleAdjuster (sliding-window, GTSAM)
├── place_recognition.py          # DescriptorBasedPlaceRecognition
├── loop_closure.py               # SpatialTemporalLoopClosureDetector + LoopClosureManager
├── pose_graph/
│   ├── custom_optimizer.py       # LeastSquaresPoseGraphOptimizer (fallback)
│   └── gtsam_optimizer.py        # GTSAMPoseGraphOptimizer
├── slam_core.py                  # ImprovedVisualSLAM — main pipeline
├── slam_enhanced.py              # EnhancedVisualSLAMWithLoopClosure — full system
├── visualisation.py              # SLAMVisualizationSystem
└── main.py                       # Entry point
```

---

## Dataset

Tested on **LuSNAR Moon_1**:
- 1094 stereo image pairs (1024×1024)
- Stereo baseline: 310 mm
- Focal length: 610.18 px
- Ground truth: 6-DoF poses at every frame

Camera model: **CAHV** (C, A, H, V vectors).
CAHV optical axis `A = [0, 0, 1]` maps to rover body +X (forward).

---

## Key Design Decisions

| Component | Approach |
|---|---|
| Feature extraction | ASIFT (affine-invariant SIFT), 1200 features/frame |
| Stereo matching | Epipolar constraint filter |
| Triangulation | Robust multi-point with reprojection validation |
| Pose estimation | PnP+RANSAC (primary) + Kabsch+RANSAC (fallback) |
| Frame-to-body transform | CAHV → rover body via `R = [[0,0,1],[0,1,0],[-1,0,0]]` |
| Local BA | Sliding-window (8 keyframes), GTSAM projection factors |
| Loop closure | Descriptor similarity + GTSAM pose graph |
| Optimiser | GTSAM (primary), custom least-squares (fallback) |

---

## Install

```bash
conda activate slam
pip install -e .
```

**Requirements:**
- numpy ≥ 1.22
- opencv-python ≥ 4.5
- scikit-learn ≥ 1.0
- scipy ≥ 1.7
- matplotlib ≥ 3.5
- gtsam ≥ 4.0 (optional — falls back to custom least-squares if unavailable)

---

## Run Analysis

```bash
conda activate slam

# Full dataset
python analyse_moon1_full.py

# First N frames only
python analyse_moon1_full.py --frames 100

# Custom output directory
python analyse_moon1_full.py --frames 100 --output /path/to/output

# Start from a specific frame
python analyse_moon1_full.py --start 50 --frames 100

# Disable GTSAM or loop closure
python analyse_moon1_full.py --no-gtsam
python analyse_moon1_full.py --no-loop-closure

# All options
python analyse_moon1_full.py --help
```

**Outputs** (saved to `--output` directory):
- `diagnostic_report.txt` — full ATE, RPE, drift, scale analysis
- `per_frame_diagnostics.csv` — per-frame metrics
- `full_metrics.json` — machine-readable summary
- `trajectory_top_down.png` — top-down XZ trajectory vs GT
- `trajectory_3d.png` — 3D trajectory coloured by ATE error
- `ate_over_sequence.png` — ATE over time + windowed RMSE
- `rpe_analysis.png` — relative pose error (stride 1 and 10)
- `pipeline_health.png` — features, correspondences, inlier ratio, reprojection error
- `scale_drift.png` — estimated vs GT cumulative path length
- `loop_closure_map.png` — loop closure detections and missed candidates

---

## Tests

```bash
conda activate slam
python -m unittest discover -s tests -v
```

---

## Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Original baseline — untouched |
| `dev` | Integration branch — latest stable code |
| `fix/*` | Bug fix branches — merged to dev after verification |
| `feat/*` | Feature branches — merged to dev after verification |

---

## Roadmap

- [x] CAHV camera model + stereo triangulation
- [x] PnP+RANSAC pose estimation (100% success rate)
- [x] GT-seeded initial pose
- [x] Stationary startup detection
- [x] CLI argument support for analysis script
- [x] Local bundle adjustment (sliding window, GTSAM)
- [ ] Loop closure detection (currently 0% detection rate — in progress)
- [ ] Scale error correction (currently ~23% underestimation)
- [ ] Full LuSNAR Moon_1 sequence validation
- [ ] Multi-session mapping

---

*Still in active development. Results and architecture subject to change.*
