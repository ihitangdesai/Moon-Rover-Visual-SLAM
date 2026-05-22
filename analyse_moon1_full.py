#!/usr/bin/env python3
"""
Full Moon_1 Scene — SLAM Diagnostic Analysis
Runs the complete refactored Visual SLAM pipeline on LuSNAR Moon_1,
compares against GT, and produces a detailed weakness report.

Dataset facts (confirmed in Phase 0):
  Images:     1094 per camera (image0/color/, image1/color/)
              Named as nanosecond timestamps — NOT sequential integers.
  GT:         gt.txt, 1095 lines (1 header + 1094 data rows)
              Format: timestamp_ns,x,y,z,qw,qx,qy,qz,...  (17 cols)
              GT timestamps match image timestamps (nearest-ns lookup).
  Other:      imu.txt, LiDAR/ — not used by visual SLAM.
  Pairs:      1093 consecutive frame pairs (images-1).

Camera (from LuSNAR README):
  Resolution 1024x1024, FOV 80x80 deg, Baseline 310 mm, f=610.17784 px.
  Converted to CAHV: A=[0,0,1], H=[f,0,512], V=[0,f,512].
"""

import matplotlib
matplotlib.use('Agg')   # MUST be first — headless, no display

import numpy as np
import cv2
import json
import csv
import os
import sys
import time
import traceback
import datetime
from pathlib import Path
from collections import Counter
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Output / dataset paths ─────────────────────────────────────────────────────
OUTPUT_DIR   = Path("/media/hitang-desai/T7 Shield/slam_analysis/dev_run")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = "/media/hitang-desai/T7 Shield/LuSNAR/Moon_1"
GT_FILE      = f"{DATASET_PATH}/gt.txt"

# ── Camera — LuSNAR stereo spec → CAHV ────────────────────────────────────────
_F  = 610.17784
_CX = 512.0
_CY = 512.0
LEFT_CAHV = {
    'C': np.array([0.0,   0.0, 0.0]),
    'A': np.array([0.0,   0.0, 1.0]),
    'H': np.array([_F,    0.0, _CX]),
    'V': np.array([0.0,   _F,  _CY]),
    'width': 1024, 'height': 1024,
}
RIGHT_CAHV = {
    'C': np.array([310.0, 0.0, 0.0]),
    'A': np.array([0.0,   0.0, 1.0]),
    'H': np.array([_F,    0.0, _CX]),
    'V': np.array([0.0,   _F,  _CY]),
    'width': 1024, 'height': 1024,
}

WINDOW = 50   # frames per windowed-ATE segment

# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTER — LuSNAR image loader (mirrors verify_slam_lusnar.py)
# StereoImageLoader expects image_0/ and image_1/ with integer-stem filenames.
# LuSNAR has image0/color/ with nanosecond-timestamp filenames.
# ═══════════════════════════════════════════════════════════════════════════════

class LuSNARLoader:
    def __init__(self, sequence_path: str, num_frames=None):
        self.sequence_path = Path(sequence_path)
        left_dir  = self.sequence_path / "image0" / "color"
        right_dir = self.sequence_path / "image1" / "color"
        if not left_dir.exists():
            raise FileNotFoundError(f"Left dir not found: {left_dir}")
        if not right_dir.exists():
            raise FileNotFoundError(f"Right dir not found: {right_dir}")

        left_files  = sorted(left_dir.glob("*.png"),  key=lambda p: int(p.stem))
        right_files = sorted(right_dir.glob("*.png"), key=lambda p: int(p.stem))
        left_dict   = {int(p.stem): p for p in left_files}
        right_dict  = {int(p.stem): p for p in right_files}
        common      = sorted(set(left_dict) & set(right_dict))

        if num_frames is not None:
            common = common[:num_frames + 1]

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
        print(f"LuSNARLoader: {len(self.stereo_pairs)} frames from {sequence_path}")

    def get_stereo_pair(self, index):
        pair  = self.stereo_pairs[index]
        left  = cv2.imread(pair['left_path'],  cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(pair['right_path'], cv2.IMREAD_GRAYSCALE)
        if left  is None: raise ValueError(f"Cannot load: {pair['left_path']}")
        if right is None: raise ValueError(f"Cannot load: {pair['right_path']}")
        return left, right

    def get_consecutive_pairs(self, start_index=0, count=None):
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
                'left_t':   left_t,  'right_t':  right_t,
                'left_t1':  left_t1, 'right_t1': right_t1,
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
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt(path):
    """Load GT — format: timestamp_ns,x,y,z,qw,qx,qy,qz,..."""
    poses    = []
    timestamps = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith('#'):
                continue
            try:
                ts = int(float(row[0]))
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3,  3] = [x, y, z]
                poses.append(T)
                timestamps.append(ts)
            except (ValueError, IndexError):
                continue
    return poses, timestamps


def match_gt_to_images(gt_poses, gt_timestamps, image_timestamps):
    """For each image timestamp, find nearest GT pose (within 1ms)."""
    gt_ts = np.array(gt_timestamps)
    matched = []
    for img_ts in image_timestamps:
        idx = np.argmin(np.abs(gt_ts - img_ts))
        if abs(int(gt_ts[idx]) - img_ts) > 1_000_000:
            raise RuntimeError(f"No GT within 1ms of image ts {img_ts}")
        matched.append(gt_poses[idx])
    return matched


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE STAGE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_failure_stage(result):
    if result.get('success', False):
        return 'success'
    err = result.get('error', '').lower()
    if 'feature' in err or 'extract' in err:
        return 'feature_extraction'
    if 'stereo' in err or 'match_stereo' in err:
        return 'stereo_matching'
    if 'triangulat' in err:
        return 'triangulation'
    if 'temporal' in err:
        return 'temporal_matching'
    if 'insufficient' in err or '3d correspondences' in err or 'correspondences' in err:
        return 'insufficient_correspondences'
    if 'pnp' in err or 'kabsch' in err or 'pose' in err:
        return 'pose_estimation'
    if 'loop' in err:
        return 'loop_closure'
    return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ate_umeyama(est_poses, gt_poses):
    est_t = np.array([p[:3, 3] for p in est_poses])
    gt_t  = np.array([p[:3, 3] for p in gt_poses])
    est_c = est_t - est_t.mean(0)
    gt_c  = gt_t  - gt_t.mean(0)
    H     = est_c.T @ gt_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_align = Vt.T @ np.diag([1, 1, d]) @ U.T
    aligned = (R_align @ est_c.T).T + gt_t.mean(0)
    errs = np.linalg.norm(aligned - gt_t, axis=1)
    return errs, R_align, gt_t.mean(0) - R_align @ est_t.mean(0)


def compute_rpe(est_poses, gt_poses, delta=1):
    n = min(len(est_poses), len(gt_poses)) - delta
    t_errs, r_errs = [], []
    for i in range(n):
        Q_e = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        Q_g = np.linalg.inv(gt_poses[i])  @ gt_poses[i + delta]
        Qe  = np.linalg.inv(Q_g) @ Q_e
        t_errs.append(float(np.linalg.norm(Qe[:3, 3])))
        ang = np.arccos(np.clip((np.trace(Qe[:3, :3]) - 1) / 2, -1.0, 1.0))
        r_errs.append(float(np.degrees(ang)))
    return np.array(t_errs), np.array(r_errs)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def pipeline_breakdown(records):
    total   = len(records)
    success = [r for r in records if r['success']]
    failed  = [r for r in records if not r['success']]
    failure_stages = Counter(r['failure_stage'] for r in failed)

    feat_counts   = [r['num_features'] for r in success if r['num_features'] > 0]
    corr_counts   = [r['num_3d_corr']  for r in success if r['num_3d_corr']  > 0]
    inlier_ratios = [r['inlier_ratio'] for r in success if r['inlier_ratio'] > 0]
    reproj_errs   = [r['reprojection_err'] for r in success
                     if r['chosen_method'] == 'PnP+RANSAC'
                     and not np.isnan(r['reprojection_err'])]
    methods = Counter(r['chosen_method'] for r in success)
    lc_det  = sum(1 for r in records if r['loop_closure_detected'])
    times   = [r['processing_time_s'] for r in records
                if not np.isnan(r.get('processing_time_s', float('nan')))]

    def _stats(arr):
        if not arr:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        return {
            'mean': float(np.mean(arr)),
            'min':  float(np.min(arr)),
            'max':  float(np.max(arr)),
            'std':  float(np.std(arr)),
        }

    return {
        'total_pairs':      total,
        'successful':       len(success),
        'failed':           len(failed),
        'success_rate_pct': len(success) / total * 100 if total else 0.0,
        'failure_stages':   dict(failure_stages),
        'features': {
            **_stats(feat_counts),
            'frames_below_50':  sum(1 for c in feat_counts if c < 50),
            'frames_below_150': sum(1 for c in feat_counts if c < 150),
        },
        'correspondences': {
            **_stats(corr_counts),
            'frames_below_5':  sum(1 for c in corr_counts if c < 5),
            'frames_below_15': sum(1 for c in corr_counts if c < 15),
        },
        'inlier_ratio': {
            **_stats(inlier_ratios),
            'frames_below_0_2': sum(1 for r in inlier_ratios if r < 0.2),
            'frames_below_0_4': sum(1 for r in inlier_ratios if r < 0.4),
        },
        'reprojection_error_px': _stats(reproj_errs),
        'pose_method_counts':    dict(methods),
        'loop_closures_detected': lc_det,
        'processing_time_s': {
            'total':          float(np.sum(times)) if times else 0.0,
            'mean_per_frame': float(np.mean(times)) if times else 0.0,
            'max':            float(np.max(times))  if times else 0.0,
            **({'per_frame': times} if times else {}),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=120, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {name}")


def plot_trajectory_top_down(gt_poses, est_traj, ate_errs, lc_frames, spike_frames):
    fig, ax = plt.subplots(figsize=(12, 9))
    gx = [p[0, 3] for p in gt_poses]
    gz = [p[2, 3] for p in gt_poses]
    ex = [p[0, 3] for p in est_traj]
    ez = [p[2, 3] for p in est_traj]

    ax.plot(gx, gz, 'k--', linewidth=2, label='Ground Truth', zorder=2)
    ax.plot(ex, ez, 'b-',  linewidth=1.2, label='Estimated',  zorder=2, alpha=0.8)

    # Frame number labels every 50 frames
    for i in range(0, len(ex), 50):
        ax.annotate(str(i), (ex[i], ez[i]), fontsize=7, color='blue', alpha=0.7)

    # ATE spike frames
    for f in spike_frames:
        if f < len(ex):
            ax.scatter(ex[f], ez[f], c='red', s=60, zorder=5, marker='o')

    # Loop closure detections
    for f in lc_frames:
        if f < len(ex):
            ax.scatter(ex[f], ez[f], c='orange', s=80, zorder=5, marker='*')

    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k',      linestyle='--', label='Ground Truth'),
        Line2D([0], [0], color='b',      linestyle='-',  label='Estimated'),
        Line2D([0], [0], marker='o',     color='red',    linestyle='None', label='ATE spike'),
        Line2D([0], [0], marker='*',     color='orange', linestyle='None', label='Loop closure'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title('Full Moon_1 Trajectory (top-down XZ)')
    ax.grid(True)
    _savefig(fig, 'trajectory_top_down.png')


def plot_trajectory_3d(gt_poses, est_traj, ate_errs):
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    gx = [p[0,3] for p in gt_poses]; gy = [p[1,3] for p in gt_poses]; gz = [p[2,3] for p in gt_poses]
    ex = [p[0,3] for p in est_traj]; ey = [p[1,3] for p in est_traj]; ez = [p[2,3] for p in est_traj]

    ax.plot(gx, gy, gz, 'k--', linewidth=1.5, label='Ground Truth')

    # Colour estimated by ATE (hot colormap)
    n = len(ex)
    norm_ate = ate_errs[:n] / (ate_errs.max() + 1e-9)
    cmap = plt.cm.hot
    for i in range(n - 1):
        ax.plot(ex[i:i+2], ey[i:i+2], ez[i:i+2],
                color=cmap(norm_ate[i]), linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, ate_errs.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='ATE error (m)', shrink=0.5)

    ax.scatter(*[ex[0]], *[ey[0]], *[ez[0]], c='green', s=80, marker='^', zorder=5, label='Start')
    ax.scatter(*[ex[-1]], *[ey[-1]], *[ez[-1]], c='red', s=80, marker='s', zorder=5, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory (coloured by ATE error)')
    ax.legend(fontsize=8)
    _savefig(fig, 'trajectory_3d.png')


def plot_ate_over_sequence(ate_errs, window_ate, lc_frames, spike_frames, failed_frames):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    frames = np.arange(len(ate_errs))

    # Running mean
    rm = np.convolve(ate_errs, np.ones(20)/20, mode='valid')
    rm_x = np.arange(19, len(ate_errs))

    ax1.plot(frames, ate_errs, 'b-', linewidth=0.8, alpha=0.7, label='ATE per frame')
    ax1.plot(rm_x, rm, 'orange', linewidth=1.5, linestyle='--', label='Running mean (20)')

    for f in spike_frames:
        ax1.axvline(f, color='red', alpha=0.5, linewidth=0.8, linestyle='--')
    for f in lc_frames:
        ax1.axvline(f, color='green', alpha=0.4, linewidth=0.8)
    for f in failed_frames:
        ax1.scatter(f, 0, c='grey', marker='|', s=20, alpha=0.5)

    ax1.set_ylabel('ATE (m)'); ax1.set_title('ATE over sequence')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

    # Windowed ATE bar chart
    labels = list(window_ate.keys())
    vals   = [window_ate[k]['rmse'] for k in labels]
    colors = ['green' if v < 0.05 else 'orange' if v < 0.2 else 'red' for v in vals]
    ax2.bar(range(len(labels)), vals, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax2.set_ylabel('ATE RMSE (m)'); ax2.set_title(f'Windowed ATE RMSE (window={WINDOW} frames)')
    ax2.grid(True, alpha=0.4, axis='y')

    fig.tight_layout()
    _savefig(fig, 'ate_over_sequence.png')


def plot_rpe_analysis(rpe1_t, rpe1_r, rpe10_t, rpe10_r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rpe1_t,  'b-',  linewidth=0.8, alpha=0.8, label='Stride-1')
    ax1.plot(rpe10_t, 'orange', linewidth=0.8, alpha=0.8, label='Stride-10')
    for arr, col in [(rpe1_t, 'b'), (rpe10_t, 'orange')]:
        th = arr.mean() + 2 * arr.std()
        ax1.axhline(th, color=col, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Frame pair'); ax1.set_ylabel('RPE translation (m)')
    ax1.set_title('RPE — Translation'); ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.plot(rpe1_r,  'b-',  linewidth=0.8, alpha=0.8, label='Stride-1')
    ax2.plot(rpe10_r, 'orange', linewidth=0.8, alpha=0.8, label='Stride-10')
    for arr, col in [(rpe1_r, 'b'), (rpe10_r, 'orange')]:
        th = arr.mean() + 2 * arr.std()
        ax2.axhline(th, color=col, linestyle='--', alpha=0.5)
    bad = [i for i, v in enumerate(rpe1_r) if v > 5.0]
    if bad:
        ax2.scatter(bad, rpe1_r[bad], c='red', s=20, zorder=5, label='>5° frames')
    ax2.set_xlabel('Frame pair'); ax2.set_ylabel('RPE rotation (deg)')
    ax2.set_title('RPE — Rotation'); ax2.legend(); ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    _savefig(fig, 'rpe_analysis.png')


def plot_pipeline_health(records):
    success = [r for r in records if r['success']]
    n = len(records)
    idxs = [r['pair_idx'] for r in success]

    feat  = [r['num_features'] for r in success]
    corr  = [r['num_3d_corr']  for r in success]
    inlr  = [r['inlier_ratio'] for r in success]
    reprj = [r['reprojection_err'] for r in success
             if r['chosen_method'] == 'PnP+RANSAC' and not np.isnan(r['reprojection_err'])]
    reprj_idx = [r['pair_idx'] for r in success
                 if r['chosen_method'] == 'PnP+RANSAC' and not np.isnan(r['reprojection_err'])]

    def _colours(vals, low, med):
        return ['red' if v < low else 'gold' if v < med else 'green' for v in vals]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax = axes[0, 0]
    ax.bar(idxs, feat, color=_colours(feat, 50, 150), width=1.0, alpha=0.8)
    ax.axhline(50,  color='red',  linestyle='--', alpha=0.6, label='50 kp (critical)')
    ax.axhline(150, color='gold', linestyle='--', alpha=0.6, label='150 kp (weak)')
    ax.set_xlabel('Frame'); ax.set_ylabel('Keypoints'); ax.set_title('Feature count per frame')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    ax.bar(idxs, corr, color=_colours(corr, 5, 15), width=1.0, alpha=0.8)
    ax.axhline(5,  color='red',  linestyle='--', alpha=0.6, label='5 corr (critical)')
    ax.axhline(15, color='gold', linestyle='--', alpha=0.6, label='15 corr (weak)')
    ax.set_xlabel('Frame'); ax.set_ylabel('3D correspondences')
    ax.set_title('3D correspondences per frame'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 0]
    ax.bar(idxs, inlr, color=_colours(inlr, 0.2, 0.4), width=1.0, alpha=0.8)
    ax.axhline(0.2, color='red',  linestyle='--', alpha=0.6, label='0.2 (critical)')
    ax.axhline(0.4, color='gold', linestyle='--', alpha=0.6, label='0.4 (weak)')
    ax.set_ylim(0, 1); ax.set_xlabel('Frame'); ax.set_ylabel('Inlier ratio')
    ax.set_title('Inlier ratio per frame'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1, 1]
    if reprj:
        ax.bar(reprj_idx, reprj,
               color=_colours(reprj, 5.0, 10.0)[::-1],  # reversed: low=good
               width=1.0, alpha=0.8)
        # re-colour properly
        cols = ['green' if v < 5 else 'gold' if v < 10 else 'red' for v in reprj]
        ax.bar(reprj_idx, reprj, color=cols, width=1.0, alpha=0.8)
        ax.axhline(5,  color='gold', linestyle='--', alpha=0.6, label='5px (weak)')
        ax.axhline(10, color='red',  linestyle='--', alpha=0.6, label='10px (critical)')
        ax.legend(fontsize=8)
    ax.set_xlabel('Frame'); ax.set_ylabel('Reprojection error (px)')
    ax.set_title('Reprojection error (PnP frames only)')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _savefig(fig, 'pipeline_health.png')


def plot_failure_analysis(records):
    failed = [r for r in records if not r['success']]
    stage_counts = Counter(r['failure_stage'] for r in failed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if stage_counts:
        ax1.pie(stage_counts.values(), labels=stage_counts.keys(),
                autopct='%1.1f%%', startangle=140)
        ax1.set_title('Failure stage distribution')
    else:
        ax1.text(0.5, 0.5, 'No failures!', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=16, color='green')
        ax1.set_title('Failure stage distribution')

    stage_palette = {
        'insufficient_correspondences': 'red',
        'feature_extraction':           'darkred',
        'stereo_matching':              'coral',
        'triangulation':                'purple',
        'temporal_matching':            'blue',
        'pose_estimation':              'orange',
        'loop_closure':                 'brown',
        'unknown':                      'grey',
    }
    all_stages = list(stage_palette.keys())
    y_map = {s: i for i, s in enumerate(all_stages)}

    for r in records:
        color = 'green' if r['success'] else stage_palette.get(r['failure_stage'], 'grey')
        y = 0 if r['success'] else y_map.get(r['failure_stage'], len(all_stages))
        ax2.scatter(r['pair_idx'], y, c=color, s=6, marker='|', alpha=0.8)

    yticks = [0] + list(range(1, len(all_stages) + 1))
    ylabels = ['success'] + all_stages
    ax2.set_yticks(yticks[:len(all_stages)+1])
    ax2.set_yticklabels(ylabels[:len(all_stages)+1], fontsize=8)
    ax2.set_xlabel('Frame index')
    ax2.set_title('Frame-by-frame success/failure timeline')
    ax2.grid(True, alpha=0.3, axis='x')

    fig.tight_layout()
    _savefig(fig, 'failure_analysis.png')


def plot_scale_drift(est_traj, gt_aligned):
    pos_e = np.array([p[:3, 3] for p in est_traj])
    pos_g = np.array([p[:3, 3] for p in gt_aligned[:len(est_traj)]])

    est_steps = np.linalg.norm(np.diff(pos_e, axis=0), axis=1)
    gt_steps  = np.linalg.norm(np.diff(pos_g, axis=0), axis=1)
    scale     = est_steps / (gt_steps + 1e-9)

    path_e = np.concatenate([[0], np.cumsum(est_steps)])
    path_g = np.concatenate([[0], np.cumsum(gt_steps)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(scale, 'b-', linewidth=0.8, alpha=0.8, label='Scale ratio')
    ax1.axhline(1.0, color='black', linewidth=1.5, label='Ideal (1.0)')
    ax1.fill_between(range(len(scale)), 0.9, 1.1, alpha=0.1, color='green',
                     label='±10% band')
    ax1.set_ylabel('Est/GT step magnitude'); ax1.set_xlabel('Frame pair')
    ax1.set_title('Scale consistency (estimated step / GT step)')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)
    ax1.set_ylim(max(0, scale.min() - 0.1), scale.max() + 0.1)

    ax2.plot(path_e, 'b-',  linewidth=1.2, label='Estimated path length')
    ax2.plot(path_g, 'k--', linewidth=1.5, label='GT path length')
    ax2.set_xlabel('Frame'); ax2.set_ylabel('Cumulative path length (m)')
    ax2.set_title('Cumulative path length — Estimated vs GT')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    _savefig(fig, 'scale_drift.png')
    return scale


def plot_loop_closure_map(est_traj, lc_records, missed_lc):
    pos = np.array([p[:3, 3] for p in est_traj])
    fig, ax = plt.subplots(figsize=(12, 9))

    ax.plot(pos[:, 0], pos[:, 2], '-', color='lightgrey', linewidth=0.8, zorder=1)

    for r in lc_records:
        f = r['pair_idx']
        if f < len(pos):
            conf = r['loop_closure_confidence']
            ax.scatter(pos[f, 0], pos[f, 2], c='orange',
                       s=max(20, conf * 200), marker='*', zorder=4,
                       alpha=0.8)

    top5_missed = sorted(missed_lc, key=lambda m: m['distance_m'])[:5]
    all_missed  = missed_lc
    for m in all_missed:
        f = m['frame']
        if f < len(pos):
            ax.scatter(pos[f, 0], pos[f, 2], c='red', s=20, marker='x', zorder=3, alpha=0.6)
    for m in top5_missed:
        f = m['frame']
        if f < len(pos):
            ax.annotate(f"f{f}→f{m['closest_past_frame']}\n{m['distance_m']:.2f}m",
                        (pos[f, 0], pos[f, 2]), fontsize=6, color='darkred')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='lightgrey', label='Trajectory'),
        Line2D([0], [0], marker='*', color='orange', linestyle='None', label='LC detected'),
        Line2D([0], [0], marker='x', color='red',    linestyle='None', label='Missed LC candidate'),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
    ax.set_title('Loop Closure Map')
    ax.grid(True, alpha=0.4)
    _savefig(fig, 'loop_closure_map.png')


def plot_processing_time(records):
    times  = [(r['pair_idx'], r['processing_time_s']) for r in records
               if not np.isnan(r.get('processing_time_s', float('nan')))]
    if not times:
        return
    idxs, vals = zip(*times)
    mean_t = np.mean(vals)
    colors = ['red' if v > mean_t * 2 else 'steelblue' for v in vals]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(idxs, vals, color=colors, width=1.0, alpha=0.8)
    ax.axhline(mean_t, color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean {mean_t:.1f}s')

    # Annotate 5 slowest
    top5 = sorted(times, key=lambda x: x[1], reverse=True)[:5]
    for f, t in top5:
        ax.annotate(f'{t:.0f}s', (f, t), fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Frame pair'); ax.set_ylabel('Processing time (s)')
    ax.set_title('Per-frame processing time')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    _savefig(fig, 'processing_time.png')


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _assess(val, good, weak):
    """Classify metric as GOOD / WEAK / CRITICAL (higher=better thresholds)."""
    if val >= good: return 'GOOD'
    if val >= weak: return 'WEAK'
    return 'CRITICAL'

def _assess_lo(val, good, weak):
    """Lower-is-better version."""
    if val <= good: return 'GOOD'
    if val <= weak: return 'WEAK'
    return 'CRITICAL'

def _drift_interpretation(drift_pct, ate_rmse, path_len):
    if drift_pct < 1.0:
        level = "excellent"
    elif drift_pct < 3.0:
        level = "acceptable"
    else:
        level = "significant — drift accumulation is a concern"
    return (
        f"ATE RMSE of {ate_rmse:.4f}m over a {path_len:.2f}m path = "
        f"{drift_pct:.2f}% drift. For lunar rover navigation, <1% is excellent, "
        f"1–3% is acceptable, >3% indicates significant drift accumulation. "
        f"Assessment: {level}."
    )


def build_report(
    run_ts, total_pairs, succ_pairs,
    wall_time, ate_errs, window_ate,
    rpe1_t, rpe1_r, rpe10_t, rpe10_r,
    path_len, drift_pct,
    pb,  # pipeline breakdown dict
    scale,  # np array of scale ratios
    worst_ate, spike_frames, low_feat, missed_lc,
    n_failed_pairs,
):
    n = len(ate_errs)
    ate_rmse = float(np.sqrt(np.mean(ate_errs**2)))
    ate_mean = float(np.mean(ate_errs))
    ate_max  = float(np.max(ate_errs))
    ate_std  = float(np.std(ate_errs))
    worst_frame = int(np.argmax(ate_errs))

    scale_mean = float(np.mean(scale))
    scale_std  = float(np.std(scale))
    n_under    = int(np.sum(scale < 0.9))
    n_over     = int(np.sum(scale > 1.1))

    lc_det    = pb['loop_closures_detected']
    n_missed  = len(missed_lc)
    lc_total  = lc_det + n_missed
    lc_miss_pct = (n_missed / lc_total * 100) if lc_total > 0 else 0.0

    feat_mean       = pb['features']['mean']
    feat_below50    = pb['features']['frames_below_50']
    feat_below150   = pb['features']['frames_below_150']
    corr_mean       = pb['correspondences']['mean']
    corr_below5     = pb['correspondences']['frames_below_5']
    corr_below15    = pb['correspondences']['frames_below_15']
    inlr_mean       = pb['inlier_ratio']['mean']
    inlr_below02    = pb['inlier_ratio']['frames_below_0_2']
    inlr_below04    = pb['inlier_ratio']['frames_below_0_4']
    reproj_mean     = pb['reprojection_error_px']['mean']
    reproj_max      = pb['reprojection_error_px']['max']
    t_total         = pb['processing_time_s']['total']
    t_mean          = pb['processing_time_s']['mean_per_frame']
    methods         = pb['pose_method_counts']
    kabsch_n        = methods.get('Kabsch+RANSAC', 0)
    pnp_n           = methods.get('PnP+RANSAC', 0)

    RULE = '─' * 68
    DBLE = '═' * 68

    lines = [
        DBLE,
        "VISUAL SLAM DIAGNOSTIC REPORT — LuSNAR Moon_1 Full Sequence",
        DBLE,
        "",
        "RUN SUMMARY",
        f"  Dataset:             LuSNAR Moon_1 (full sequence)",
        f"  Total frame pairs:   {total_pairs}",
        f"  Successful:          {succ_pairs} / {total_pairs}  ({succ_pairs/total_pairs*100:.1f}%)",
        f"  Failed:              {n_failed_pairs} / {total_pairs}  ({n_failed_pairs/total_pairs*100:.1f}%)",
        f"  Total wall time:     {wall_time:.1f}s  ({wall_time/total_pairs:.1f}s/frame avg)",
        f"  Report generated:    {run_ts}",
        "",
        RULE,
        "1. OVERALL ACCURACY",
        RULE,
        f"  ATE RMSE:            {ate_rmse:.4f} m",
        f"  ATE Mean:            {ate_mean:.4f} m",
        f"  ATE Max:             {ate_max:.4f} m  (frame {worst_frame})",
        f"  ATE Std:             {ate_std:.4f} m",
        f"  RPE Trans RMSE:      {float(np.sqrt(np.mean(rpe1_t**2))):.4f} m  (stride-1)",
        f"  RPE Rot   RMSE:      {float(np.sqrt(np.mean(rpe1_r**2))):.4f} °  (stride-1)",
        f"  RPE Trans RMSE:      {float(np.sqrt(np.mean(rpe10_t**2))):.4f} m  (stride-10)",
        f"  RPE Rot   RMSE:      {float(np.sqrt(np.mean(rpe10_r**2))):.4f} °  (stride-10)",
        f"  Path length:         {path_len:.2f} m",
        f"  Final displacement:  {ate_errs[-1]:.4f} m  (last-frame ATE after alignment)",
        f"  Drift:               {drift_pct:.2f}% of path length",
        "",
        "  Interpretation:",
        f"  {_drift_interpretation(drift_pct, ate_rmse, path_len)}",
        f"  (Baseline from 20-frame verification: ATE RMSE = 0.0104m.)",
        "",
        RULE,
        "2. WHERE DOES DRIFT ACCUMULATE?",
        RULE,
        "",
        f"  {'Segment':<15} {'ATE RMSE (m)':<15} Assessment",
    ]

    for seg, stats in window_ate.items():
        rmse = stats['rmse']
        if   rmse < 0.05: assess = 'Good'
        elif rmse < 0.20: assess = 'Moderate'
        else:             assess = 'Poor — investigate'
        lines.append(f"  {seg:<15} {rmse:<15.4f} {assess}")

    if spike_frames:
        lines.append("")
        lines.append(f"  ATE spike events (sudden jump > mean+3σ): {len(spike_frames)} frames")
        for f in spike_frames[:10]:
            lines.append(f"    Frame {f}")
    else:
        lines.append("")
        lines.append("  No ATE spike events detected.")

    lines += [
        "",
        RULE,
        "3. PIPELINE COMPONENT HEALTH",
        RULE,
        "",
        "  Feature Extraction:",
        f"    Mean keypoints/frame:     {feat_mean:.0f}",
        f"    Frames with <50  kp:      {feat_below50} / {succ_pairs}  ({feat_below50/succ_pairs*100:.1f}%)",
        f"    Frames with <150 kp:      {feat_below150} / {succ_pairs}  ({feat_below150/succ_pairs*100:.1f}%)",
        f"    Assessment: {_assess(feat_mean, 150, 50)}",
    ]
    if feat_mean < 150:
        lines.append(f"    Note: Mean of {feat_mean:.0f} kp/frame is below the 150-kp 'good' threshold.")
        lines.append( "          Consider reducing ASIFT contrastThreshold or increasing nfeatures.")
    else:
        lines.append( "          Feature extraction is healthy across the sequence.")

    lines += [
        "",
        "  Stereo Matching & Triangulation:",
        f"    Mean 3D correspondences:  {corr_mean:.1f}",
        f"    Frames with <5  corr:     {corr_below5} / {succ_pairs}  ({corr_below5/succ_pairs*100:.1f}%)",
        f"    Frames with <15 corr:     {corr_below15} / {succ_pairs}  ({corr_below15/succ_pairs*100:.1f}%)",
        f"    Assessment: {_assess(corr_mean, 15, 5)}",
    ]
    if corr_below15 > succ_pairs * 0.1:
        lines.append(f"    Note: {corr_below15/succ_pairs*100:.0f}% of frames have <15 3D correspondences.")
        lines.append( "          Investigate stereo epipolar threshold or triangulation depth bounds.")

    lines += [
        "",
        "  Pose Estimation:",
        f"    Kabsch+RANSAC used:       {kabsch_n} frames  ({kabsch_n/succ_pairs*100:.1f}%)" if succ_pairs else "    Kabsch+RANSAC: N/A",
        f"    PnP+RANSAC used:          {pnp_n} frames  ({pnp_n/succ_pairs*100:.1f}%)" if succ_pairs else "    PnP+RANSAC: N/A",
        f"    Mean inlier ratio:        {inlr_mean:.3f}",
        f"    Frames with ratio <0.2:   {inlr_below02} ({inlr_below02/succ_pairs*100:.1f}%)" if succ_pairs else "",
        f"    Frames with ratio <0.4:   {inlr_below04} ({inlr_below04/succ_pairs*100:.1f}%)" if succ_pairs else "",
        f"    Mean reprojection error:  {reproj_mean:.2f} px  (PnP frames)",
        f"    Max  reprojection error:  {reproj_max:.2f} px  (PnP frames)",
        f"    Assessment: {_assess(inlr_mean, 0.4, 0.2)} (inlier ratio) / "
        f"{_assess_lo(reproj_mean, 5.0, 10.0)} (reprojection error)",
        "",
        "  Loop Closure:",
        f"    Detections:               {lc_det}",
        f"    Missed candidates:        {n_missed}",
        f"    Miss rate:                {lc_miss_pct:.1f}%",
        f"    Assessment: {'GOOD' if lc_miss_pct < 30 else 'WEAK' if lc_miss_pct < 70 else 'CRITICAL'}",
    ]
    if lc_miss_pct > 30:
        lines.append(f"    Note: {lc_miss_pct:.0f}% missed rate suggests similarity_threshold may be")
        lines.append( "          too strict. Consider lowering from 0.4 → 0.3.")

    # Failure breakdown
    fail_stages = pb['failure_stages']
    lines += [
        "",
        RULE,
        "4. FAILURE ANALYSIS",
        RULE,
        "",
        f"  Total failures: {n_failed_pairs} / {total_pairs}",
        "",
        "  Failure stage breakdown:",
    ]
    if fail_stages:
        for stage, count in sorted(fail_stages.items(), key=lambda x: -x[1]):
            lines.append(f"    {stage:<35} {count}  ({count/n_failed_pairs*100:.1f}%)")
    else:
        lines.append("    None — all frames succeeded.")

    # Scale analysis
    lines += [
        "",
        RULE,
        "5. SCALE ANALYSIS",
        RULE,
        "",
        f"  Mean scale ratio:    {scale_mean:.4f}  (1.0 = perfect)",
        f"  Std  scale ratio:    {scale_std:.4f}",
        f"  Frames >10% under:   {n_under} ({n_under/len(scale)*100:.1f}%)",
        f"  Frames >10% over:    {n_over}  ({n_over/len(scale)*100:.1f}%)",
        f"  Assessment: {'stable' if scale_std < 0.1 else 'moderate variance' if scale_std < 0.3 else 'high variance'}",
    ]
    if abs(scale_mean - 1.0) > 0.05:
        direction = "underestimation" if scale_mean < 1.0 else "overestimation"
        lines.append(f"  Note: Systematic {direction} of {abs(1-scale_mean)*100:.1f}% detected.")
        lines.append( "        Check triangulation depth bounds or baseline calibration.")

    # Weaknesses
    lines += [
        "",
        RULE,
        "6. SPECIFIC WEAKNESSES IDENTIFIED",
        RULE,
        "",
    ]

    weaknesses = []

    if feat_below50 / max(succ_pairs, 1) > 0.05:
        weaknesses.append((
            "Feature sparsity",
            f"{feat_below50/succ_pairs*100:.1f}% of frames have <50 keypoints (mean={feat_mean:.0f})",
            "pipeline_health.png top-left",
            f"Reduce ASIFT contrastThreshold (currently 0.04) or increase nfeatures (currently 1200). "
            f"Consider an ORB fallback when ASIFT returns <30 keypoints.",
        ))

    if corr_below5 / max(succ_pairs, 1) > 0.05:
        weaknesses.append((
            "Low 3D correspondence count",
            f"{corr_below5/succ_pairs*100:.1f}% of frames have <5 3D correspondences (mean={corr_mean:.1f})",
            "pipeline_health.png top-right",
            "Widen stereo epipolar threshold or relax triangulation depth bounds. "
            "Check stereo_epipolar_threshold in DatasetAdaptiveParameters.",
        ))

    if inlr_below02 / max(succ_pairs, 1) > 0.05:
        weaknesses.append((
            "Low pose estimation inlier ratio",
            f"{inlr_below02/succ_pairs*100:.1f}% of frames have inlier ratio <0.2 (mean={inlr_mean:.3f})",
            "pipeline_health.png bottom-left",
            "Investigate whether low inlier count is caused by poor correspondences upstream. "
            "Consider tightening the correspondence spatial threshold.",
        ))

    if lc_miss_pct > 30:
        weaknesses.append((
            "Loop closure miss rate",
            f"{n_missed} missed candidates vs {lc_det} detections ({lc_miss_pct:.0f}% miss rate)",
            "loop_closure_map.png",
            "Lower similarity_threshold from 0.4 → 0.3 in EnhancedVisualSLAMWithLoopClosure "
            "and/or lower min_matches from 8 → 5.",
        ))

    if drift_pct > 3.0:
        worst_seg = max(window_ate.items(), key=lambda x: x[1]['rmse'])
        weaknesses.append((
            "Drift accumulation",
            f"{drift_pct:.2f}% drift over path; worst segment: {worst_seg[0]} "
            f"(ATE RMSE={worst_seg[1]['rmse']:.4f}m)",
            "ate_over_sequence.png",
            f"Inspect stereo_matches/ and temporal_matches/ images for frames in "
            f"segment {worst_seg[0]} for visual failure clues. "
            "Loop closure may not be firing frequently enough to correct accumulated drift.",
        ))

    if abs(scale_mean - 1.0) > 0.1:
        weaknesses.append((
            "Systematic scale error",
            f"Mean scale ratio = {scale_mean:.4f} (deviation {abs(1-scale_mean)*100:.1f}% from ideal)",
            "scale_drift.png",
            "Verify CAHV baseline value (currently 310mm). A 10% scale error can arise "
            "from a 10% baseline miscalibration. Also check triangulation min/max depth bounds.",
        ))

    worst_windows = [k for k, v in window_ate.items() if v['rmse'] > 0.2]
    if worst_windows:
        weaknesses.append((
            "Localised drift hotspots",
            f"Segments {worst_windows} have ATE RMSE > 0.2m",
            "ate_over_sequence.png",
            "These segments may contain difficult terrain (uniform texture, lighting changes). "
            "Inspect images in these frame ranges.",
        ))

    if not weaknesses:
        lines.append("  No significant weaknesses detected. All metrics are within healthy thresholds.")
    else:
        for i, (title, symptom, evidence, suggestion) in enumerate(weaknesses, 1):
            lines += [
                f"  WEAKNESS {i}: {title}",
                f"    Symptom:    {symptom}",
                f"    Evidence:   moon1_analysis/{evidence}",
                f"    Suggestion: {suggestion}",
                "",
            ]

    # What is working well
    lines += [
        RULE,
        "7. WHAT IS WORKING WELL",
        RULE,
        "",
    ]
    strengths = []
    if succ_pairs / total_pairs > 0.95:
        strengths.append(f"High success rate: {succ_pairs/total_pairs*100:.1f}% of frames processed successfully.")
    if drift_pct < 3.0:
        strengths.append(f"Drift is within acceptable bounds: {drift_pct:.2f}% over the full path.")
    if inlr_mean > 0.4:
        strengths.append(f"Pose estimation inlier ratio is healthy: mean={inlr_mean:.3f}.")
    if reproj_mean < 5.0 and reproj_mean > 0:
        strengths.append(f"PnP reprojection error is low: mean={reproj_mean:.2f}px.")
    if feat_mean >= 150:
        strengths.append(f"Feature extraction is robust: mean={feat_mean:.0f} keypoints/frame.")
    if not strengths:
        strengths.append("Run the script and review metrics to identify strengths.")
    for s in strengths:
        lines.append(f"  • {s}")

    lines += [
        "",
        RULE,
        "8. OUTPUT FILES",
        RULE,
        "",
        "  moon1_analysis/trajectory_top_down.png",
        "  moon1_analysis/trajectory_3d.png",
        "  moon1_analysis/ate_over_sequence.png",
        "  moon1_analysis/rpe_analysis.png",
        "  moon1_analysis/pipeline_health.png",
        "  moon1_analysis/failure_analysis.png",
        "  moon1_analysis/scale_drift.png",
        "  moon1_analysis/loop_closure_map.png",
        "  moon1_analysis/processing_time.png",
        "  moon1_analysis/per_frame_diagnostics.csv",
        "  moon1_analysis/full_metrics.json",
        "  moon1_analysis/diagnostic_report.txt",
        "",
        DBLE,
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("SLAM DIAGNOSTIC ANALYSIS — LuSNAR Moon_1 Full Sequence")
    print("=" * 70)

    # ── Load SLAM package ────────────────────────────────────────────────────
    from visual_slam.slam_enhanced import EnhancedVisualSLAMWithLoopClosure

    # ── Build loader ─────────────────────────────────────────────────────────
    loader = LuSNARLoader(DATASET_PATH)   # all 1094 frames
    image_timestamps = [p['frame_number'] for p in loader.stereo_pairs]
    total_pairs = len(loader) - 1  # 1093

    # ── Load GT ──────────────────────────────────────────────────────────────
    print("\nLoading ground truth...")
    gt_poses_full, gt_timestamps = load_gt(GT_FILE)
    gt_per_image = match_gt_to_images(gt_poses_full, gt_timestamps, image_timestamps)
    print(f"  {len(gt_per_image)} GT poses matched to {len(image_timestamps)} images")

    # ── Initialise SLAM ──────────────────────────────────────────────────────
    print("\nInitialising SLAM...")
    np.random.seed(42)
    cv2.setRNGSeed(42)
    slam = EnhancedVisualSLAMWithLoopClosure(
        LEFT_CAHV, RIGHT_CAHV,
        use_gtsam=True,
        enable_loop_closure=True,
        show_lines=False,
    )

    # ── Run full sequence ────────────────────────────────────────────────────
    print(f"\nRunning full sequence: {total_pairs} frame pairs...")
    start_wall = time.time()

    consecutive_pairs = loader.get_consecutive_pairs(start_index=0, count=None)
    sequence_results  = []

    for i, pair_data in enumerate(consecutive_pairs):
        pair_num = i + 1
        if pair_num % 50 == 0 or pair_num == 1:
            elapsed = time.time() - start_wall
            eta = (elapsed / pair_num) * (total_pairs - pair_num)
            print(f"  Pair {pair_num}/{total_pairs}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        try:
            result = slam.process_frame_pair_with_loop_closure(
                pair_data['left_t'],  pair_data['right_t'],
                pair_data['left_t1'], pair_data['right_t1'],
                frame_info=pair_data,
            )
            result['pair_number'] = pair_num
            result['success']     = result.get('success', False)
        except Exception as e:
            result = {
                'success': False,
                'pair_number': pair_num,
                'frame_info': pair_data,
                'error': f'Unexpected: {str(e)}',
            }
            print(f"  FRAME {pair_num} FAILED: {e}")
        sequence_results.append(result)

    wall_time = time.time() - start_wall
    est_trajectory = [p.copy() for p in slam.trajectory]
    print(f"\nRun complete in {wall_time:.1f}s. Trajectory length: {len(est_trajectory)} poses.")

    # ── Build per-frame records ──────────────────────────────────────────────
    per_frame_records = []
    for i, result in enumerate(sequence_results):
        psum  = result.get('pose_estimation_summary', {}) or {}
        lc    = result.get('loop_closure_result',    {}) or {}
        lc_info = lc.get('loop_closure_info', {}) or {}
        rec = {
            'pair_idx':              i,
            'frame_t':               result.get('frame_info', {}).get('frame_t_number',  i)   if result.get('frame_info') else i,
            'frame_t1':              result.get('frame_info', {}).get('frame_t1_number', i+1) if result.get('frame_info') else i+1,
            'success':               result.get('success', False),
            'processing_time_s':     result.get('enhanced_processing_time', result.get('processing_time', float('nan'))),
            'error_msg':             result.get('error', ''),
            'failure_stage':         infer_failure_stage(result),
            'num_features':          result.get('num_features', 0),
            'num_3d_corr':           result.get('num_3d_correspondences', 0),
            'chosen_method':         result.get('chosen_method', 'none'),
            'num_inliers':           psum.get('num_inliers', 0),
            'inlier_ratio':          psum.get('inlier_ratio', 0.0),
            'reprojection_err':      psum.get('mean_error', float('nan')),
            'translation_m':         result.get('translation_magnitude_m', 0.0),
            'rotation_deg':          result.get('rotation_angle_deg', 0.0),
            'loop_closure_detected': lc.get('loop_closure_detected', False),
            'loop_closure_confidence': lc_info.get('confidence', 0.0),
        }
        per_frame_records.append(rec)

    # ── Align GT to trajectory ───────────────────────────────────────────────
    # trajectory[i] = pose after processing pair i (= frame i+1)
    # So gt_per_image[i+1] aligns to trajectory[i]
    N = len(est_trajectory)
    gt_aligned = gt_per_image[1 : N + 1]
    if len(gt_aligned) < N:
        print(f"WARNING: GT shorter than trajectory ({len(gt_aligned)} < {N}). Truncating.")
        est_trajectory = est_trajectory[:len(gt_aligned)]
        N = len(est_trajectory)

    # ── Metrics ──────────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    ate_errors, R_align, t_align = compute_ate_umeyama(est_trajectory, gt_aligned)

    # Windowed ATE
    window_ate = {}
    for start in range(0, N, WINDOW):
        end = min(start + WINDOW, N)
        if end - start < 5:
            continue
        w_errs, _, _ = compute_ate_umeyama(est_trajectory[start:end], gt_aligned[start:end])
        window_ate[f"{start}-{end}"] = {
            'rmse':  float(np.sqrt(np.mean(w_errs**2))),
            'mean':  float(np.mean(w_errs)),
            'start': start, 'end': end,
        }

    rpe1_t,  rpe1_r  = compute_rpe(est_trajectory, gt_aligned, delta=1)
    rpe10_t, rpe10_r = compute_rpe(est_trajectory, gt_aligned, delta=10)

    positions   = np.array([p[:3, 3] for p in est_trajectory])
    path_lens   = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    total_path  = float(path_lens[-1])
    drift_pct   = (float(ate_errors[-1]) / total_path * 100) if total_path > 0 else 0.0

    gt_pos      = np.array([p[:3, 3] for p in gt_aligned])
    est_steps   = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    gt_steps    = np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)
    scale       = est_steps / (gt_steps + 1e-9)

    # Problematic frames
    worst_ate_top10 = sorted(
        [(float(ate_errors[i]), i) for i in range(N)], reverse=True
    )[:10]

    ate_diffs = np.diff(ate_errors)
    spike_th  = np.mean(np.abs(ate_diffs)) + 3 * np.std(ate_diffs)
    spike_frames = [i + 1 for i, d in enumerate(ate_diffs) if abs(d) > spike_th]

    low_feat_frames = [r['pair_idx'] for r in per_frame_records
                       if r['success'] and r['num_features'] < 50]

    missed_lc = []
    for i in range(30, N):
        dists = np.linalg.norm(positions[:max(0, i-20)] - positions[i], axis=1) \
                if i > 20 else np.array([float('inf')])
        if len(dists) == 0:
            continue
        min_d = float(np.min(dists))
        if min_d < 2.0 and not per_frame_records[i]['loop_closure_detected']:
            missed_lc.append({
                'frame': i,
                'closest_past_frame': int(np.argmin(dists)),
                'distance_m': min_d,
            })

    lc_frames   = [r['pair_idx'] for r in per_frame_records if r['loop_closure_detected']]
    failed_fr   = [r['pair_idx'] for r in per_frame_records if not r['success']]
    pb          = pipeline_breakdown(per_frame_records)
    n_success   = pb['successful']
    n_failed    = pb['failed']

    # Fill ATE into per-frame records (only for successful frames with trajectory entries)
    for r in per_frame_records:
        idx = r['pair_idx']
        r['ate_error_m']   = float(ate_errors[idx])   if idx < len(ate_errors) else float('nan')
        r['rpe1_trans_m']  = float(rpe1_t[idx])       if idx < len(rpe1_t)    else float('nan')
        r['rpe1_rot_deg']  = float(rpe1_r[idx])       if idx < len(rpe1_r)    else float('nan')
        r['scale_ratio']   = float(scale[idx])         if idx < len(scale)     else float('nan')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / 'per_frame_diagnostics.csv'
    fieldnames = [
        'pair_idx', 'frame_t', 'frame_t1', 'success', 'failure_stage',
        'processing_time_s', 'num_features', 'num_3d_corr', 'chosen_method',
        'num_inliers', 'inlier_ratio', 'reprojection_err', 'translation_m',
        'rotation_deg', 'ate_error_m', 'rpe1_trans_m', 'rpe1_rot_deg',
        'scale_ratio', 'loop_closure_detected', 'loop_closure_confidence',
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(per_frame_records)
    print(f"  CSV saved: {csv_path}")

    # ── Save full metrics JSON ────────────────────────────────────────────────
    def _rpe_dict(t, r):
        return {
            'trans_rmse': float(np.sqrt(np.mean(t**2))),
            'trans_mean': float(np.mean(t)),
            'trans_max':  float(np.max(t)),
            'rot_rmse_deg': float(np.sqrt(np.mean(r**2))),
            'rot_mean_deg': float(np.mean(r)),
        }

    pb_json = {k: v for k, v in pb.items() if k != 'processing_time_s'}
    pt      = pb['processing_time_s']
    pb_json['processing_time_s'] = {k: v for k, v in pt.items() if k != 'per_frame'}

    metrics = {
        "run_info": {
            "dataset":                "LuSNAR Moon_1",
            "total_frames":           len(loader),
            "total_pairs_attempted":  total_pairs,
            "wall_time_seconds":      round(wall_time, 2),
            "timestamp":              run_ts,
        },
        "overall_accuracy": {
            "ate": {
                "rmse": float(np.sqrt(np.mean(ate_errors**2))),
                "mean": float(np.mean(ate_errors)),
                "max":  float(np.max(ate_errors)),
                "std":  float(np.std(ate_errors)),
                "worst_frame": int(np.argmax(ate_errors)),
            },
            "rpe_stride1":          _rpe_dict(rpe1_t, rpe1_r),
            "rpe_stride10":         _rpe_dict(rpe10_t, rpe10_r),
            "drift_percent_of_path": round(drift_pct, 4),
            "total_path_length_m":   round(total_path, 4),
            "final_ate_m":           round(float(ate_errors[-1]), 6),
        },
        "windowed_ate":    window_ate,
        "scale_analysis": {
            "mean_scale_ratio":       float(np.mean(scale)),
            "std_scale_ratio":        float(np.std(scale)),
            "frames_underestimating": int(np.sum(scale < 0.9)),
            "frames_overestimating":  int(np.sum(scale > 1.1)),
        },
        "pipeline_breakdown": pb_json,
        "problematic_frames": {
            "worst_ate_top10":          [{'ate': a, 'frame': f} for a, f in worst_ate_top10],
            "ate_spike_frames":          spike_frames[:20],
            "low_feature_frames":        low_feat_frames[:20],
            "missed_lc_candidates_top10": missed_lc[:10],
        }
    }
    json_str = json.dumps(metrics, indent=2)
    json.loads(json_str)  # validate
    with open(OUTPUT_DIR / 'full_metrics.json', 'w') as f:
        f.write(json_str)
    print(f"  JSON saved: {OUTPUT_DIR / 'full_metrics.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_trajectory_top_down(gt_aligned, est_trajectory, ate_errors, lc_frames, spike_frames)
    plot_trajectory_3d(gt_aligned, est_trajectory, ate_errors)
    plot_ate_over_sequence(ate_errors, window_ate, lc_frames, spike_frames, failed_fr)
    plot_rpe_analysis(rpe1_t, rpe1_r, rpe10_t, rpe10_r)
    plot_pipeline_health(per_frame_records)
    plot_failure_analysis(per_frame_records)
    scale_arr = plot_scale_drift(est_trajectory, gt_aligned)
    plot_loop_closure_map(est_trajectory, [r for r in per_frame_records if r['loop_closure_detected']], missed_lc)
    plot_processing_time(per_frame_records)

    # ── Diagnostic report ─────────────────────────────────────────────────────
    print("\nBuilding diagnostic report...")
    report = build_report(
        run_ts, total_pairs, n_success,
        wall_time, ate_errors, window_ate,
        rpe1_t, rpe1_r, rpe10_t, rpe10_r,
        total_path, drift_pct,
        pb, scale_arr,
        worst_ate_top10, spike_frames, low_feat_frames, missed_lc,
        n_failed,
    )

    report_path = OUTPUT_DIR / 'diagnostic_report.txt'
    with open(report_path, 'w') as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
