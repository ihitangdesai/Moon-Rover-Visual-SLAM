#!/usr/bin/env python3
"""
SLAM Behavioural Verification Script
Monolith vs Refactored Package on LuSNAR dataset (first 20 frames)

Dataset notes (discovered in Phase 0):
- Sequence: Moon_1
- Images: image0/color/<timestamp>.png (left), image1/color/<timestamp>.png (right)
  Names are nanosecond timestamps (integers), NOT sequential frame numbers 0,1,2,...
- GT: gt.txt — custom format with header line, columns:
    timestamp, x, y, z, qw, qx, qy, qz, vx, vy, vz, bwx, bwy, bwz, bax, bay, baz
  (17 columns, first row is a comment header starting with #)
- GT timestamps match image timestamps exactly.
- Adapter: LuSNARLoader subclasses StereoImageLoader logic directly (no folder rename).

Camera parameters: Derived from LuSNAR stereo camera spec (README):
  Resolution 1024x1024, FOV 80°x80°, Baseline 310 mm, Focal length 610.17784 px.
  Converted to CAHV: A=[0,0,1], H=[610.17784,0,512], V=[0,610.17784,512].
  Left C=[0,0,0] mm, Right C=[310,0,0] mm. Same params used for both SLAM runs.
"""

import matplotlib
matplotlib.use('Agg')  # must be before any pyplot import

import os
import sys
import json
import time
import traceback
import importlib.util
from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "/media/hitang-desai/T7 Shield/LuSNAR"
SEQUENCE_PATH = DATASET_ROOT + "/Moon_1"
GT_FILE       = SEQUENCE_PATH + "/gt.txt"
NUM_FRAMES    = 20
OUTPUT_DIR    = "./verification_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── CAHV camera parameters derived from LuSNAR stereo camera spec:
#    Resolution: 1024x1024, FOV: 80°x80°, Baseline: 310 mm, Focal length: 610.17784 px
#    Principal point assumed at image centre (512, 512) — standard for symmetric sim cameras.
#    CAHV conversion: A=[0,0,1], H=[fx,0,cx], V=[0,fy,cy], C=optical centre in mm.
#    Left camera at world origin; right camera offset +310 mm along X (baseline).
_F  = 610.17784   # focal length in pixels
_CX = 512.0       # principal point x
_CY = 512.0       # principal point y
LEFT_CAHV = {
    'C': np.array([0.0, 0.0, 0.0]),
    'A': np.array([0.0, 0.0, 1.0]),
    'H': np.array([_F,  0.0, _CX]),
    'V': np.array([0.0, _F,  _CY]),
    'width':  1024,
    'height': 1024,
}
RIGHT_CAHV = {
    'C': np.array([310.0, 0.0, 0.0]),   # 310 mm baseline along X
    'A': np.array([0.0, 0.0, 1.0]),
    'H': np.array([_F,  0.0, _CX]),
    'V': np.array([0.0, _F,  _CY]),
    'width':  1024,
    'height': 1024,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTER: LuSNAR image loader
# LuSNAR uses image0/color/<timestamp>.png — not image_0/<frame>.png.
# We replicate StereoImageLoader's interface without subclassing (avoids import
# order issues when loading the monolith module separately).
# ═══════════════════════════════════════════════════════════════════════════════

class LuSNARLoader:
    """
    Drop-in replacement for StereoImageLoader for the LuSNAR dataset.
    Maps Moon_1/image0/color/ → left, Moon_1/image1/color/ → right.
    Images are sorted by timestamp (filename stem as integer).
    """

    def __init__(self, sequence_path: str, num_frames: int):
        self.sequence_path = Path(sequence_path)
        left_dir  = self.sequence_path / "image0" / "color"
        right_dir = self.sequence_path / "image1" / "color"

        if not left_dir.exists():
            raise FileNotFoundError(f"Left dir not found: {left_dir}")
        if not right_dir.exists():
            raise FileNotFoundError(f"Right dir not found: {right_dir}")

        left_files  = sorted(left_dir.glob("*.png"),  key=lambda p: int(p.stem))
        right_files = sorted(right_dir.glob("*.png"), key=lambda p: int(p.stem))

        # Match by timestamp stem
        left_dict  = {int(p.stem): p for p in left_files}
        right_dict = {int(p.stem): p for p in right_files}
        common     = sorted(set(left_dict) & set(right_dict))

        # We need num_frames+1 images to form num_frames consecutive pairs
        need = num_frames + 1
        if len(common) < need:
            raise RuntimeError(f"Need {need} images, found {len(common)}")
        common = common[:need]

        self.stereo_pairs = [
            {
                'frame_number': ts,
                'left_path':    str(left_dict[ts]),
                'right_path':   str(right_dict[ts]),
                'left_name':    left_dict[ts].name,
                'right_name':   right_dict[ts].name,
            }
            for ts in common
        ]
        print(f"LuSNARLoader: {len(self.stereo_pairs)} frames loaded from {sequence_path}")

    def get_stereo_pair(self, index: int):
        pair = self.stereo_pairs[index]
        left  = cv.imread(pair['left_path'],  cv.IMREAD_GRAYSCALE)
        right = cv.imread(pair['right_path'], cv.IMREAD_GRAYSCALE)
        if left  is None: raise ValueError(f"Cannot load left:  {pair['left_path']}")
        if right is None: raise ValueError(f"Cannot load right: {pair['right_path']}")
        return left, right

    def get_consecutive_pairs(self, start_index: int = 0, count=None):
        if count is None:
            count = len(self.stereo_pairs) - start_index - 1
        pairs = []
        for i in range(start_index, min(start_index + count, len(self.stereo_pairs) - 1)):
            left_t,  right_t  = self.get_stereo_pair(i)
            left_t1, right_t1 = self.get_stereo_pair(i + 1)
            pairs.append({
                'frame_t_index':   i,
                'frame_t1_index':  i + 1,
                'frame_t_number':  self.stereo_pairs[i]['frame_number'],
                'frame_t1_number': self.stereo_pairs[i + 1]['frame_number'],
                'left_t':   left_t,
                'right_t':  right_t,
                'left_t1':  left_t1,
                'right_t1': right_t1,
                'pair_t_info':  self.stereo_pairs[i],
                'pair_t1_info': self.stereo_pairs[i + 1],
            })
        return pairs

    def get_info(self):
        frames = [p['frame_number'] for p in self.stereo_pairs]
        return {
            'num_pairs':   len(self.stereo_pairs),
            'frame_range': (min(frames), max(frames)),
            'folder_path': str(self.sequence_path),
            'left_folder': str(self.sequence_path / "image0" / "color"),
            'right_folder': str(self.sequence_path / "image1" / "color"),
            'first_pair': self.stereo_pairs[0],
            'last_pair':  self.stereo_pairs[-1],
        }

    def __len__(self):
        return len(self.stereo_pairs)


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH LOADER
# gt.txt format (custom — not standard TUM):
#   Header line starting with #
#   Columns: timestamp, x, y, z, qw, qx, qy, qz, ... (17 cols total)
# We match GT rows to image timestamps.
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt_lusnar(gt_path: str, image_timestamps: list) -> list:
    """
    Load GT poses for the given image timestamps.
    gt.txt has a header row (#...) then rows:
        timestamp_ns, x, y, z, qw, qx, qy, qz, ...
    Returns list of 4x4 np.float64 arrays, one per image_timestamp.
    """
    gt_by_ts = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            ts = int(float(parts[0]))  # nanosecond timestamp
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3,  3] = [x, y, z]
            gt_by_ts[ts] = T

    poses = []
    for ts in image_timestamps:
        if ts in gt_by_ts:
            poses.append(gt_by_ts[ts])
        else:
            # nearest-neighbour fallback (timestamps may differ by a few ns)
            nearest = min(gt_by_ts.keys(), key=lambda k: abs(k - ts))
            if abs(nearest - ts) < 1_000_000:  # within 1 ms
                poses.append(gt_by_ts[nearest])
            else:
                raise RuntimeError(f"No GT found within 1ms of image timestamp {ts}")
    return poses


# ═══════════════════════════════════════════════════════════════════════════════
# RUN BOTH SLAM SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════

def run_slam_system(slam_instance, loader, num_frames: int, label: str):
    """
    Run a SLAM system (monolith or refactored) on the loader, collecting
    per-frame results. Returns (trajectory, results, success_count).
    trajectory: list of 4x4 np arrays (one per frame including initial pose).
    """
    print(f"\n{'#'*70}")
    print(f"# RUNNING: {label}")
    print(f"{'#'*70}\n")

    sequence_results = []
    success_count = 0

    consecutive_pairs = loader.get_consecutive_pairs(start_index=0, count=num_frames)
    if not consecutive_pairs:
        print(f"[{label}] ERROR: No consecutive pairs from loader.")
        return [], [], 0

    for i, pair_data in enumerate(consecutive_pairs):
        pair_num = i + 1
        print(f"\n[{label}] Pair {pair_num}/{len(consecutive_pairs)}: "
              f"frames {pair_data['frame_t_number']} → {pair_data['frame_t1_number']}")
        try:
            result = slam_instance.process_frame_pair_with_loop_closure(
                pair_data['left_t'],  pair_data['right_t'],
                pair_data['left_t1'], pair_data['right_t1'],
                frame_info=pair_data,
            )
            result['pair_number'] = pair_num
            result['success']     = True
            sequence_results.append(result)
            success_count += 1
        except Exception as e:
            print(f"[{label}] FRAME {pair_num} FAILED: {e}")
            traceback.print_exc()
            sequence_results.append({'pair_number': pair_num, 'success': False})

    trajectory = [p.copy() for p in slam_instance.trajectory]
    return trajectory, sequence_results, success_count


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_trajectories(traj_a, traj_b, label_a="Monolith", label_b="Refactored"):
    assert len(traj_a) == len(traj_b), \
        f"Trajectory length mismatch: {len(traj_a)} vs {len(traj_b)}"

    max_t_diff = 0.0
    max_r_diff = 0.0
    per_frame_t = []

    for Pa, Pb in zip(traj_a, traj_b):
        t_diff = float(np.linalg.norm(Pa[:3, 3] - Pb[:3, 3]))
        R_diff = Pa[:3, :3] @ Pb[:3, :3].T
        trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
        r_diff = float(np.degrees(np.arccos(trace_val)))
        max_t_diff = max(max_t_diff, t_diff)
        max_r_diff = max(max_r_diff, r_diff)
        per_frame_t.append(t_diff)

    passed = max_t_diff < 1e-6 and max_r_diff < 1e-4

    print(f"\n{'='*60}")
    print(f"EQUIVALENCE CHECK: {label_a} vs {label_b}")
    print(f"  Max translation difference: {max_t_diff:.8f} m")
    print(f"  Max rotation difference:    {max_r_diff:.8f} deg")
    print(f"  Result: {'PASS — Systems are numerically equivalent' if passed else 'FAIL — Trajectories differ!'}")
    print(f"{'='*60}\n")

    return max_t_diff, max_r_diff, passed, per_frame_t


def compute_ate(estimated_poses, gt_poses):
    n = min(len(estimated_poses), len(gt_poses))
    est_t = np.array([p[:3, 3] for p in estimated_poses[:n]])
    gt_t  = np.array([p[:3, 3] for p in gt_poses[:n]])

    est_mean = est_t.mean(axis=0)
    gt_mean  = gt_t.mean(axis=0)
    est_c = est_t - est_mean
    gt_c  = gt_t  - gt_mean

    H = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R_align = Vt.T @ D @ U.T
    t_align = gt_mean - R_align @ est_mean

    aligned_t = (R_align @ est_c.T).T + gt_mean
    errors = np.linalg.norm(aligned_t - gt_t, axis=1)

    return {
        'rmse':      float(np.sqrt(np.mean(errors**2))),
        'mean':      float(np.mean(errors)),
        'max':       float(np.max(errors)),
        'std':       float(np.std(errors)),
        'per_frame': errors.tolist(),
        'R_align':   R_align,
        't_align':   t_align,
    }


def compute_rpe(estimated_poses, gt_poses, delta=1):
    n = min(len(estimated_poses), len(gt_poses)) - delta
    trans_errors = []
    rot_errors   = []

    for i in range(n):
        Q_est = np.linalg.inv(estimated_poses[i]) @ estimated_poses[i + delta]
        Q_gt  = np.linalg.inv(gt_poses[i])        @ gt_poses[i + delta]
        Q_err = np.linalg.inv(Q_gt) @ Q_est

        trans_errors.append(float(np.linalg.norm(Q_err[:3, 3])))
        trace_val = np.clip((np.trace(Q_err[:3, :3]) - 1) / 2, -1.0, 1.0)
        rot_errors.append(float(np.degrees(np.arccos(trace_val))))

    if not trans_errors:
        return {k: 0.0 for k in ('translation_rmse','translation_mean','translation_max',
                                  'rotation_rmse_deg','rotation_mean_deg','rotation_max_deg',
                                  'per_frame_trans','per_frame_rot_deg')}
    return {
        'translation_rmse':  float(np.sqrt(np.mean(np.array(trans_errors)**2))),
        'translation_mean':  float(np.mean(trans_errors)),
        'translation_max':   float(np.max(trans_errors)),
        'rotation_rmse_deg': float(np.sqrt(np.mean(np.array(rot_errors)**2))),
        'rotation_mean_deg': float(np.mean(rot_errors)),
        'rotation_max_deg':  float(np.max(rot_errors)),
        'per_frame_trans':   trans_errors,
        'per_frame_rot_deg': rot_errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE TRAJECTORY CSV
# ═══════════════════════════════════════════════════════════════════════════════

def save_trajectory_csv(poses, path):
    with open(path, 'w') as f:
        f.write("frame_idx,x,y,z,r00,r01,r02,r10,r11,r12,r20,r21,r22\n")
        for i, P in enumerate(poses):
            r = P[:3, :3]
            t = P[:3,  3]
            f.write(f"{i},{t[0]:.8f},{t[1]:.8f},{t[2]:.8f},"
                    f"{r[0,0]:.8f},{r[0,1]:.8f},{r[0,2]:.8f},"
                    f"{r[1,0]:.8f},{r[1,1]:.8f},{r[1,2]:.8f},"
                    f"{r[2,0]:.8f},{r[2,1]:.8f},{r[2,2]:.8f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _traj_xyz(poses):
    xs = [p[0, 3] for p in poses]
    ys = [p[1, 3] for p in poses]
    zs = [p[2, 3] for p in poses]
    return np.array(xs), np.array(ys), np.array(zs)


def plot_trajectory_2d(gt_poses, mono_traj, ref_traj, mono_ate, ref_ate, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    gx, gy, gz = _traj_xyz(gt_poses)
    mx, my, mz = _traj_xyz(mono_traj)
    rx, ry, rz = _traj_xyz(ref_traj)

    ax.plot(gx, gz, 'k--', linewidth=2,   label='Ground Truth', zorder=3)
    ax.plot(mx, mz, 'b-',  linewidth=1.5, label=f'Monolith (ATE RMSE={mono_ate["rmse"]:.4f}m)', zorder=2)
    ax.plot(rx, rz, 'r:',  linewidth=2,   label=f'Refactored (ATE RMSE={ref_ate["rmse"]:.4f}m)', zorder=2)

    # Start/end markers
    for xs, zs, color in [(gx, gz, 'k'), (mx, mz, 'b'), (rx, rz, 'r')]:
        ax.scatter(xs[0],  zs[0],  marker='^', s=100, c=color, zorder=5)
        ax.scatter(xs[-1], zs[-1], marker='s', s=100, c=color, zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Trajectory Comparison — Top-Down (XZ plane)')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_ate_per_frame(mono_ate, ref_ate, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    frames_m = range(len(mono_ate['per_frame']))
    frames_r = range(len(ref_ate['per_frame']))

    ax.plot(frames_m, mono_ate['per_frame'], 'b-o', markersize=4, label='Monolith ATE')
    ax.plot(frames_r, ref_ate['per_frame'],  'r-s', markersize=4, label='Refactored ATE')
    ax.axhline(mono_ate['rmse'], color='b', linestyle='--', alpha=0.6, label=f"Mono RMSE={mono_ate['rmse']:.4f}m")
    ax.axhline(ref_ate['rmse'],  color='r', linestyle='--', alpha=0.6, label=f"Ref RMSE={ref_ate['rmse']:.4f}m")

    ax.set_xlabel('Frame index')
    ax.set_ylabel('ATE error (m)')
    ax.set_title('Per-frame Absolute Trajectory Error')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rpe_per_frame(mono_rpe, ref_rpe, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(mono_rpe['per_frame_trans'], 'b-o', markersize=4, label='Monolith')
    ax1.plot(ref_rpe['per_frame_trans'],  'r-s', markersize=4, label='Refactored')
    ax1.set_ylabel('RPE translation (m)')
    ax1.set_title('Per-frame Relative Pose Error — Translation')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(mono_rpe['per_frame_rot_deg'], 'b-o', markersize=4, label='Monolith')
    ax2.plot(ref_rpe['per_frame_rot_deg'],  'r-s', markersize=4, label='Refactored')
    ax2.set_xlabel('Frame pair index')
    ax2.set_ylabel('RPE rotation (deg)')
    ax2.set_title('Per-frame Relative Pose Error — Rotation')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectory_3d(gt_poses, mono_traj, ref_traj, out_path):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    gx, gy, gz = _traj_xyz(gt_poses)
    mx, my, mz = _traj_xyz(mono_traj)
    rx, ry, rz = _traj_xyz(ref_traj)

    ax.plot(gx, gy, gz, 'k--', linewidth=2,   label='Ground Truth')
    ax.plot(mx, my, mz, 'b-',  linewidth=1.5, label='Monolith')
    ax.plot(rx, ry, rz, 'r:',  linewidth=2,   label='Refactored')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_equivalence_diff(per_frame_t_diff, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    frames = range(len(per_frame_t_diff))
    ax.plot(frames, per_frame_t_diff, 'g-o', markersize=5, label='Translation diff')

    threshold = 1e-4
    bad_frames = [i for i, d in enumerate(per_frame_t_diff) if d > threshold]
    if bad_frames:
        bad_vals = [per_frame_t_diff[i] for i in bad_frames]
        ax.scatter(bad_frames, bad_vals, c='red', zorder=5,
                   label=f'Diff > 1e-4 m ({len(bad_frames)} frames)')
        print(f"WARNING: {len(bad_frames)} frames exceed 1e-4 m equivalence threshold: {bad_frames}")

    ax.axhline(threshold, color='r', linestyle='--', alpha=0.5, label='1e-4 m threshold')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Translation difference (m)')
    ax.set_title('Per-frame Monolith vs Refactored Translation Difference')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def build_report(seq_name, num_frames, gt_format,
                 eq_max_t, eq_max_r, eq_pass,
                 mono_ate, mono_rpe, ref_ate, ref_rpe,
                 mono_success, ref_success):
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║         SLAM VERIFICATION REPORT — LuSNAR Dataset           ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Dataset:   LuSNAR / {seq_name:<40}║",
        f"║  Frames:    {num_frames} consecutive pairs (frames 0 → {num_frames})           ║",
        f"║  GT format: {gt_format:<49}║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  EQUIVALENCE CHECK (Monolith vs Refactored)                  ║",
        f"║  Max translation diff:  {eq_max_t:.8f} m                  ║",
        f"║  Max rotation diff:     {eq_max_r:.8f} deg               ║",
        f"║  Result:  {'✓ PASS' if eq_pass else '✗ FAIL':<56}║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  ACCURACY vs GROUND TRUTH                                    ║",
        "║                       Monolith      Refactored               ║",
        f"║  ATE RMSE (m):        {mono_ate['rmse']:<13.4f} {ref_ate['rmse']:<13.4f}       ║",
        f"║  ATE Mean (m):        {mono_ate['mean']:<13.4f} {ref_ate['mean']:<13.4f}       ║",
        f"║  ATE Max  (m):        {mono_ate['max']:<13.4f} {ref_ate['max']:<13.4f}       ║",
        f"║  RPE Trans RMSE (m):  {mono_rpe['translation_rmse']:<13.4f} {ref_rpe['translation_rmse']:<13.4f}       ║",
        f"║  RPE Rot  RMSE (deg): {mono_rpe['rotation_rmse_deg']:<13.4f} {ref_rpe['rotation_rmse_deg']:<13.4f}       ║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  PROCESSING SUCCESS RATE                                      ║",
        f"║  Monolith:    {mono_success}/{num_frames} frames succeeded                      ║",
        f"║  Refactored:  {ref_success}/{num_frames} frames succeeded                      ║",
        "╠══════════════════════════════════════════════════════════════╣",
        "║  OUTPUT FILES                                                 ║",
        "║  verification_results/trajectory_comparison_2d.png            ║",
        "║  verification_results/ate_per_frame.png                       ║",
        "║  verification_results/rpe_per_frame.png                       ║",
        "║  verification_results/trajectory_3d.png                       ║",
        "║  verification_results/equivalence_diff.png                    ║",
        "║  verification_results/trajectory_monolith.csv                 ║",
        "║  verification_results/trajectory_refactored.csv               ║",
        "║  verification_results/trajectory_gt.csv                       ║",
        "║  verification_results/metrics_summary.json                    ║",
        "╚══════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SLAM VERIFICATION: Monolith vs Refactored on LuSNAR Moon_1")
    print("=" * 70)

    # ── Build loader (shared between both runs) ──────────────────────────────
    loader_mono = LuSNARLoader(SEQUENCE_PATH, NUM_FRAMES)
    loader_ref  = LuSNARLoader(SEQUENCE_PATH, NUM_FRAMES)

    image_timestamps = [p['frame_number'] for p in loader_mono.stereo_pairs]

    # ── Load ground truth ────────────────────────────────────────────────────
    print("\nLoading ground truth...")
    gt_poses = load_gt_lusnar(GT_FILE, image_timestamps)
    print(f"  Loaded {len(gt_poses)} GT poses")
    print(f"  GT[0] translation: {gt_poses[0][:3,3]}")
    print(f"  GT[-1] translation: {gt_poses[-1][:3,3]}")

    # ── Load monolith module ─────────────────────────────────────────────────
    print("\nLoading monolith module...")
    mono_module = None
    mono_traj   = []
    mono_success = 0

    try:
        spec = importlib.util.spec_from_file_location(
            "monolith",
            os.path.join(os.path.dirname(__file__), "monolithic_code.py")
        )
        mono_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mono_module)
        print("  Monolith loaded successfully")
    except Exception as e:
        print(f"  ERROR loading monolith: {e}")
        traceback.print_exc()

    if mono_module is not None:
        try:
            np.random.seed(42)   # fix numpy RANSAC non-determinism
            cv.setRNGSeed(42)    # fix OpenCV solvePnPRansac non-determinism
            mono_slam = mono_module.EnhancedVisualSLAMWithLoopClosure(
                LEFT_CAHV, RIGHT_CAHV,
                use_gtsam=True,
                enable_loop_closure=True,
                show_lines=False,
            )
            mono_traj, _, mono_success = run_slam_system(
                mono_slam, loader_mono, NUM_FRAMES, "MONOLITH"
            )
        except Exception as e:
            print(f"  ERROR running monolith SLAM: {e}")
            traceback.print_exc()

    # ── Load refactored package ──────────────────────────────────────────────
    print("\nLoading refactored package...")
    ref_traj    = []
    ref_success = 0

    try:
        from visual_slam.slam_enhanced import EnhancedVisualSLAMWithLoopClosure as RefSLAM
        print("  visual_slam package loaded successfully")

        np.random.seed(42)   # same seed as monolith run
        cv.setRNGSeed(42)    # same OpenCV seed
        ref_slam = RefSLAM(
            LEFT_CAHV, RIGHT_CAHV,
            use_gtsam=True,
            enable_loop_closure=True,
            show_lines=False,
        )
        ref_traj, _, ref_success = run_slam_system(
            ref_slam, loader_ref, NUM_FRAMES, "REFACTORED"
        )
    except ImportError as e:
        print(f"  ERROR: could not import visual_slam package: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"  ERROR running refactored SLAM: {e}")
        traceback.print_exc()

    # ── Trajectory alignment: both should start at identity-like pose ────────
    # SLAM trajectory has one pose per frame (not per pair), starting from frame 0.
    # GT has one pose per frame. We align counts.
    n_gt   = len(gt_poses)
    n_mono = len(mono_traj)
    n_ref  = len(ref_traj)

    print(f"\nTrajectory lengths: GT={n_gt}, Monolith={n_mono}, Refactored={n_ref}")

    # Use the minimum available length for ATE/RPE
    n_common = min(n_gt, n_mono, n_ref) if (n_mono > 0 and n_ref > 0) else 0

    # ── Equivalence check ────────────────────────────────────────────────────
    eq_max_t = eq_max_r = 0.0
    eq_pass  = False
    per_frame_t_diff = []

    if n_mono > 0 and n_ref > 0:
        n_eq = min(n_mono, n_ref)
        eq_max_t, eq_max_r, eq_pass, per_frame_t_diff = compare_trajectories(
            mono_traj[:n_eq], ref_traj[:n_eq]
        )
    else:
        print("WARNING: Cannot compare — one or both trajectories are empty.")

    # ── ATE / RPE ────────────────────────────────────────────────────────────
    empty_ate = {'rmse': 0.0, 'mean': 0.0, 'max': 0.0, 'std': 0.0, 'per_frame': []}
    empty_rpe = {
        'translation_rmse': 0.0, 'translation_mean': 0.0, 'translation_max': 0.0,
        'rotation_rmse_deg': 0.0, 'rotation_mean_deg': 0.0, 'rotation_max_deg': 0.0,
        'per_frame_trans': [], 'per_frame_rot_deg': [],
    }

    if n_mono > 1 and n_gt > 1:
        mono_ate = compute_ate(mono_traj, gt_poses)
        mono_rpe = compute_rpe(mono_traj, gt_poses)
    else:
        mono_ate, mono_rpe = empty_ate.copy(), empty_rpe.copy()

    if n_ref > 1 and n_gt > 1:
        ref_ate = compute_ate(ref_traj, gt_poses)
        ref_rpe = compute_rpe(ref_traj, gt_poses)
    else:
        ref_ate, ref_rpe = empty_ate.copy(), empty_rpe.copy()

    # ── Save trajectory CSVs ─────────────────────────────────────────────────
    save_trajectory_csv(mono_traj, f"{OUTPUT_DIR}/trajectory_monolith.csv")
    save_trajectory_csv(ref_traj,  f"{OUTPUT_DIR}/trajectory_refactored.csv")
    save_trajectory_csv(gt_poses,  f"{OUTPUT_DIR}/trajectory_gt.csv")
    print(f"\nTrajectory CSVs saved to {OUTPUT_DIR}/")

    # ── Plots ────────────────────────────────────────────────────────────────
    use_mono_traj = mono_traj if n_mono > 0 else [np.eye(4)]
    use_ref_traj  = ref_traj  if n_ref  > 0 else [np.eye(4)]
    use_gt_poses  = gt_poses  if n_gt   > 0 else [np.eye(4)]

    plot_trajectory_2d(use_gt_poses, use_mono_traj, use_ref_traj,
                       mono_ate, ref_ate,
                       f"{OUTPUT_DIR}/trajectory_comparison_2d.png")

    plot_ate_per_frame(mono_ate, ref_ate, f"{OUTPUT_DIR}/ate_per_frame.png")

    plot_rpe_per_frame(mono_rpe, ref_rpe, f"{OUTPUT_DIR}/rpe_per_frame.png")

    plot_trajectory_3d(use_gt_poses, use_mono_traj, use_ref_traj,
                       f"{OUTPUT_DIR}/trajectory_3d.png")

    plot_equivalence_diff(per_frame_t_diff if per_frame_t_diff else [0.0],
                          f"{OUTPUT_DIR}/equivalence_diff.png")

    print(f"All plots saved to {OUTPUT_DIR}/")

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    def _rpe_summary(rpe):
        return {
            'translation_rmse': rpe['translation_rmse'],
            'translation_mean': rpe['translation_mean'],
            'translation_max':  rpe['translation_max'],
            'rotation_rmse_deg': rpe['rotation_rmse_deg'],
            'rotation_mean_deg': rpe['rotation_mean_deg'],
            'rotation_max_deg':  rpe['rotation_max_deg'],
        }

    def _ate_summary(ate):
        return {k: ate[k] for k in ('rmse', 'mean', 'max', 'std')}

    metrics = {
        "dataset":             "LuSNAR",
        "sequence":            "Moon_1",
        "num_frames_processed": NUM_FRAMES,
        "gt_format":           "Custom (timestamp,x,y,z,qw,qx,qy,qz,...)",
        "equivalence": {
            "max_translation_diff_m":  eq_max_t,
            "max_rotation_diff_deg":   eq_max_r,
            "pass":                    eq_pass,
        },
        "monolith_vs_gt": {
            "ate": _ate_summary(mono_ate),
            "rpe": _rpe_summary(mono_rpe),
        },
        "refactored_vs_gt": {
            "ate": _ate_summary(ref_ate),
            "rpe": _rpe_summary(ref_rpe),
        },
        "success_rate": {
            "monolith":   f"{mono_success}/{NUM_FRAMES}",
            "refactored": f"{ref_success}/{NUM_FRAMES}",
        }
    }

    json_path = f"{OUTPUT_DIR}/metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Validate JSON round-trip
    with open(json_path) as f:
        json.loads(f.read())
    print(f"metrics_summary.json written and validated.")

    # ── Final report ─────────────────────────────────────────────────────────
    report = build_report(
        "Moon_1", NUM_FRAMES,
        "Custom (ts,x,y,z,qw,qx,qy,qz,...)",
        eq_max_t, eq_max_r, eq_pass,
        mono_ate, mono_rpe,
        ref_ate,  ref_rpe,
        mono_success, ref_success,
    )
    print("\n" + report)

    report_path = f"{OUTPUT_DIR}/verification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report + "\n")
    print(f"\nReport saved to {report_path}")

    sys.exit(0 if eq_pass else 1)


if __name__ == "__main__":
    main()
