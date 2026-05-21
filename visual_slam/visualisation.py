"""
CUSTOM VISUALIZATION SYSTEM (SOLID - Single Responsibility)
All matplotlib / trajectory plotting and OpenCV match visualisation.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class SLAMVisualizationSystem:
    """Custom visualization system for stereo and temporal matches."""

    def __init__(self, show_lines: bool = False):
        self.show_lines = show_lines

        # Create output directories
        self.stereo_matches_dir = Path("stereo_matches")
        self.temporal_matches_dir = Path("temporal_matches")
        self.stereo_matches_dir.mkdir(exist_ok=True)
        self.temporal_matches_dir.mkdir(exist_ok=True)

        self.frame_count = 0
        self.loop_closure_points = []  # Store loop closure points for final map

        print(f"SLAM visualization system initialized")
        print(f"   Stereo matches directory: {self.stereo_matches_dir}")
        print(f"   Temporal matches directory: {self.temporal_matches_dir}")
        print(f"   Show connection lines: {show_lines}")

    def save_stereo_matches(self, left_img: np.ndarray, right_img: np.ndarray,
                           left_kp: List, right_kp: List, matches: List[Tuple[int, int]],
                           frame_number: int) -> str:
        """Save stereo matches visualization."""
        if len(left_img.shape) == 2:
            left_img = cv.cvtColor(left_img, cv.COLOR_GRAY2BGR)
        if len(right_img.shape) == 2:
            right_img = cv.cvtColor(right_img, cv.COLOR_GRAY2BGR)

        h1, w1 = left_img.shape[:2]
        h2, w2 = right_img.shape[:2]
        combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = left_img
        combined_img[:h2, w1:w1+w2] = right_img

        # Draw matches
        for left_idx, right_idx in matches:
            left_pt = tuple(map(int, left_kp[left_idx].pt))
            right_pt = tuple(map(int, (right_kp[right_idx].pt[0] + w1, right_kp[right_idx].pt[1])))

            # Draw connection line if enabled
            if self.show_lines:
                cv.line(combined_img, left_pt, right_pt, (0, 255, 0), 1)

            # Draw green dots
            cv.circle(combined_img, left_pt, 3, (0, 255, 0), -1)
            cv.circle(combined_img, right_pt, 3, (0, 255, 0), -1)

        # Add title
        title = f"Stereo Matches Frame {frame_number}: {len(matches)} matches"
        cv.putText(combined_img, title, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save image
        filename = f"stereo_matches_frame_{frame_number:04d}.png"
        filepath = self.stereo_matches_dir / filename
        cv.imwrite(str(filepath), combined_img)

        print(f"   Saved stereo matches: {filename}")
        return str(filepath)

    def save_temporal_matches(self, left_t: np.ndarray, left_t1: np.ndarray,
                             left_kp_t: List, left_kp_t1: List, matches: List,
                             frame_t: int, frame_t1: int) -> str:
        """Save temporal matches visualization with left(t) at bottom and left(t+1) at top."""
        if len(left_t.shape) == 2:
            left_t = cv.cvtColor(left_t, cv.COLOR_GRAY2BGR)
        if len(left_t1.shape) == 2:
            left_t1 = cv.cvtColor(left_t1, cv.COLOR_GRAY2BGR)

        h_t, w_t = left_t.shape[:2]
        h_t1, w_t1 = left_t1.shape[:2]

        # Create combined image with left(t+1) on top and left(t) on bottom
        total_width = max(w_t, w_t1)
        total_height = h_t + h_t1
        combined_img = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Place left(t+1) at top
        combined_img[:h_t1, :w_t1] = left_t1
        # Place left(t) at bottom
        combined_img[h_t1:h_t1+h_t, :w_t] = left_t

        # Draw matches
        for match in matches:
            try:
                pt_t = tuple(map(int, left_kp_t[match.queryIdx].pt))
                pt_t1 = tuple(map(int, left_kp_t1[match.trainIdx].pt))

                # Adjust coordinates for combined image
                pt_t_adjusted = (pt_t[0], pt_t[1] + h_t1)  # left(t) is at bottom
                pt_t1_adjusted = pt_t1  # left(t+1) is at top

                # Draw connection line if enabled
                if self.show_lines:
                    cv.line(combined_img, pt_t1_adjusted, pt_t_adjusted, (0, 255, 0), 1)

                # Draw green dots
                cv.circle(combined_img, pt_t1_adjusted, 3, (0, 255, 0), -1)
                cv.circle(combined_img, pt_t_adjusted, 3, (0, 255, 0), -1)

            except (IndexError, AttributeError):
                continue

        # Add labels
        cv.putText(combined_img, f"Left Frame {frame_t1} (t+1)", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.putText(combined_img, f"Left Frame {frame_t} (t)", (10, h_t1 + 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Add match count
        match_count_text = f"Temporal Matches: {len(matches)}"
        cv.putText(combined_img, match_count_text, (10, total_height - 20),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save image
        filename = f"temporal_matches_frame_{frame_t:04d}_{frame_t1:04d}.png"
        filepath = self.temporal_matches_dir / filename
        cv.imwrite(str(filepath), combined_img)

        print(f"   Saved temporal matches: {filename}")
        return str(filepath)

    def record_loop_closure(self, pose: np.ndarray, frame_id: int):
        """Record loop closure point for final map."""
        position = pose[:3, 3]  # Keep in meters
        self.loop_closure_points.append({
            'position': position,
            'frame_id': frame_id
        })

    def create_2d_trajectory_map(self, trajectory: List[np.ndarray],
                                save_path: str = "trajectory_2d_map.png") -> str:
        """Create 2D trajectory map with start, end, and loop closure points."""
        if len(trajectory) == 0:
            print("No trajectory data for 2D map")
            return ""

        # Positions are already in meters
        positions = np.array([pose[:3, 3] for pose in trajectory])

        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot trajectory path
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory Path')

        # Plot trajectory points
        ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=20, alpha=0.6, label='Poses')

        # Mark start point
        ax.scatter([positions[0, 0]], [positions[0, 1]],
                  c='green', s=200, marker='^', label='Start Point',
                  edgecolors='black', linewidth=2, zorder=5)

        # Mark end point
        if len(positions) > 1:
            ax.scatter([positions[-1, 0]], [positions[-1, 1]],
                      c='red', s=200, marker='v', label='End Point',
                      edgecolors='black', linewidth=2, zorder=5)

        # Mark loop closure points
        if self.loop_closure_points:
            lc_positions = np.array([lc['position'] for lc in self.loop_closure_points])
            ax.scatter(lc_positions[:, 0], lc_positions[:, 1],
                      c='orange', s=150, marker='*', label='Loop Closures',
                      edgecolors='black', linewidth=1, zorder=4)

            # Annotate loop closure points
            for lc in self.loop_closure_points:
                ax.annotate(f'LC{lc["frame_id"]}',
                           (lc['position'][0], lc['position'][1]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', fc='orange', alpha=0.8),
                           fontsize=8, fontweight='bold')

        # Calculate trajectory statistics
        if len(positions) > 1:
            distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            total_distance = np.sum(distances)
        else:
            total_distance = 0

        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'SLAM 2D Trajectory Map\n'
                    f'Total Distance: {total_distance:.2f}m | '
                    f'Poses: {len(positions)} | '
                    f'Loop Closures: {len(self.loop_closure_points)}')

        # Add legend and grid
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Save plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"2D trajectory map saved: {save_path}")
        return save_path
