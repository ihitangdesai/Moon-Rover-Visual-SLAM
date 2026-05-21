# Refactoring Log

## Overview

`monolithic_code.py` (4 007 lines) split into the `visual_slam/` package.
Behavioural equivalence is the hard constraint — no algorithm changes, no threshold
changes, no reordering of logic.

---

## Module boundary decisions

### `interfaces.py`
Contains only the three `Protocol` classes (`PlaceRecognitionInterface`,
`LoopClosureInterface`, `PoseGraphOptimizerInterface`). No logic.
Moved first because they have zero dependencies.

### `data_structures.py`
Contains all `@dataclass` types and `CAHVCamera`.  
`CAHVCamera` is included here (not in a separate camera module) because it owns both
data (C, A, H, V) and the projection/unprojection methods that every other class uses.
Splitting it would require circular workarounds.  
`CAHVCamera.cahv_to_gtsam_pose` and `to_gtsam_camera` do a late `import gtsam` and
consult `visual_slam.GTSAM_AVAILABLE` at call time so the module can be imported even
without GTSAM installed.

### `config.py`
Contains `DatasetAdaptiveParameters` — the single source of truth for all thresholds.
All numerical constants live here (asift_max_tilts, stereo_epipolar_threshold,
ransac_iterations, etc.).

### `place_recognition.py`
`DescriptorBasedPlaceRecognition` is self-contained (depends on sklearn and cv2).
`PlaceDescriptor` dataclass imported from `data_structures`.

### `loop_closure.py`
Contains both `SpatialTemporalLoopClosureDetector` and `LoopClosureManager`.
Decision: keep them together because `LoopClosureManager` references
`SLAMVisualizationSystem` and the two loop closure classes are tightly coupled by design
(the manager calls the detector). Splitting would add no value.

### `pose_graph/`
Sub-package chosen because the two optimizers (`GTSAMPoseGraphOptimizer`,
`LeastSquaresPoseGraphOptimizer`) are alternative implementations of the same interface.
`gtsam_optimizer.py` does all GTSAM imports lazily inside methods to avoid hard
import failures when GTSAM is absent.

### `feature_extraction.py`
`OptimizedASIFTMatcher` — pure cv2/numpy, no SLAM-specific dependencies.

### `stereo_matching.py`
`StereoImageLoader` — despite the name this file handles image I/O, not stereo matching
geometry. The name was kept to match the planned layout specification.

### `temporal_matching.py`
`SmartTemporalMatcher` — depends only on cv2 and `config.py`.

### `triangulation.py`
`RobustTriangulator` — depends on `CAHVCamera`, `DatasetAdaptiveParameters`, and
optionally GTSAM. Imports GTSAM lazily.

### `correspondence_validation.py`
`Enhanced3DCorrespondenceFinder` with the complete 3-stage pipeline. Heavy but
self-contained.

### `slam_core.py`
`ImprovedVisualSLAM` — the main SLAM class.  Imports all pipeline components.

### `slam_enhanced.py`
`EnhancedVisualSLAMWithLoopClosure` — thin wrapper; holds loop closure manager and
delegates to `ImprovedVisualSLAM`.

### `visualisation.py`
`SLAMVisualizationSystem` — all matplotlib and OpenCV visualisation code.
Moved before `slam_core.py` because slam_core imports it.

### `main.py`
Contains `main_working_slam()` and the `if __name__ == "__main__"` block, verbatim from
the original.

---

## Module-level side effects — preservation strategy

The original fires two side effects at import time, in this order:

1. GTSAM `try/except` import → prints status, sets `GTSAM_AVAILABLE`
2. `print(f"NumPy version: {np.__version__}")`

These are reproduced verbatim at the top of `visual_slam/__init__.py` so they fire
exactly once, in the same order, whenever the package is first imported.

The `GTSAM_AVAILABLE` flag is set in `__init__.py` and accessed by all modules via
`from visual_slam import GTSAM_AVAILABLE` (lazy lookup inside methods where necessary
to avoid circular import at module load time).

---

## Circular import avoidance

The main risk was `data_structures.CAHVCamera` calling GTSAM functions, while
`__init__.py` sets `GTSAM_AVAILABLE` and imports everything else.

Resolution: `CAHVCamera.cahv_to_gtsam_pose` and `to_gtsam_camera` do
`from visual_slam import GTSAM_AVAILABLE` and `import gtsam` **inside the method body**,
not at module level. This breaks the circular dependency because by the time any method
is called, `__init__.py` has already finished executing.

Similarly `triangulation.py` and `pose_graph/gtsam_optimizer.py` use late imports of
`gtsam`.

---

## Bugs preserved intentionally

1. **`SpatialTemporalLoopClosureDetector.detect_loop_closure`** — the temporal distance
   guard is hardcoded to `100` rather than using `self.min_temporal_distance`. This is in
   the original and is preserved exactly. The comment in the original says
   "VISUAL SIMILARITY ONLY".

2. **`LoopClosureManager.process_frame`** adds the odometry edge as
   `frame_id - 1 → frame_id` regardless of whether `frame_id - 1` is the actual
   previous frame. This matches the original.

3. **`estimate_pose_change_pnp`** uses keypoints from `left_kp_t1[:len(points_3d_t1)]`
   without validating that those keypoints correspond to the 3D points. Preserved.

---

## `__main__` RuntimeWarning

Running `python -m visual_slam.main` produces:

```
RuntimeWarning: 'visual_slam.main' found in sys.modules after import of package
'visual_slam', but prior to execution of 'visual_slam.main'
```

This is a standard Python 3.10 warning when a `__main__` submodule is imported as part
of the package before being executed. It is benign and does not affect output or
behaviour. The original monolith does not have this warning because it is a single file.
