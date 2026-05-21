"""
STEREO IMAGE LOADER - loads and manages stereo image sequences.
"""

import numpy as np
import cv2 as cv
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class StereoImageLoader:
    """Utility class to load and manage stereo image sequences."""

    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)
        self.left_folder = self.folder_path / "image_0"
        self.right_folder = self.folder_path / "image_1"

        if not self.folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")

        if not self.left_folder.exists():
            raise ValueError(f"Left image folder does not exist: {self.left_folder}")

        if not self.right_folder.exists():
            raise ValueError(f"Right image folder does not exist: {self.right_folder}")

        self.stereo_pairs = self._discover_stereo_pairs()

    def _discover_stereo_pairs(self) -> List[Dict[str, str]]:
        """Discover stereo image pairs."""
        left_files = list(self.left_folder.glob("*.png")) + list(self.left_folder.glob("*.jpg")) + list(self.left_folder.glob("*.jpeg"))
        right_files = list(self.right_folder.glob("*.png")) + list(self.right_folder.glob("*.jpg")) + list(self.right_folder.glob("*.jpeg"))

        print(f"Discovering stereo pairs in: {self.folder_path}")
        print(f"   Found {len(left_files)} left images and {len(right_files)} right images")

        left_dict = {}
        right_dict = {}

        for left_file in left_files:
            try:
                frame_num = int(left_file.stem)
                left_dict[frame_num] = left_file
            except ValueError:
                continue

        for right_file in right_files:
            try:
                frame_num = int(right_file.stem)
                right_dict[frame_num] = right_file
            except ValueError:
                continue

        stereo_pairs = []
        common_frames = set(left_dict.keys()) & set(right_dict.keys())

        for frame_num in sorted(common_frames):
            stereo_pairs.append({
                'frame_number': frame_num,
                'left_path': str(left_dict[frame_num]),
                'right_path': str(right_dict[frame_num]),
                'left_name': left_dict[frame_num].name,
                'right_name': right_dict[frame_num].name
            })

        print(f"Found {len(stereo_pairs)} stereo pairs")
        return stereo_pairs

    def get_stereo_pair(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a stereo pair by index."""
        if index < 0 or index >= len(self.stereo_pairs):
            raise IndexError(f"Index {index} out of range [0, {len(self.stereo_pairs)-1}]")

        pair = self.stereo_pairs[index]

        left_img = cv.imread(pair['left_path'], cv.IMREAD_GRAYSCALE)
        right_img = cv.imread(pair['right_path'], cv.IMREAD_GRAYSCALE)

        if left_img is None:
            raise ValueError(f"Failed to load left image: {pair['left_path']}")
        if right_img is None:
            raise ValueError(f"Failed to load right image: {pair['right_path']}")

        return left_img, right_img

    def get_consecutive_pairs(self, start_index: int = 0, count: Optional[int] = None) -> List[Dict]:
        """Get consecutive stereo pairs for temporal processing."""
        if count is None:
            count = len(self.stereo_pairs) - start_index - 1

        consecutive_pairs = []

        for i in range(start_index, min(start_index + count, len(self.stereo_pairs) - 1)):
            left_t, right_t = self.get_stereo_pair(i)
            left_t1, right_t1 = self.get_stereo_pair(i + 1)

            consecutive_pairs.append({
                'frame_t_index': i,
                'frame_t1_index': i + 1,
                'frame_t_number': self.stereo_pairs[i]['frame_number'],
                'frame_t1_number': self.stereo_pairs[i + 1]['frame_number'],
                'left_t': left_t,
                'right_t': right_t,
                'left_t1': left_t1,
                'right_t1': right_t1,
                'pair_t_info': self.stereo_pairs[i],
                'pair_t1_info': self.stereo_pairs[i + 1]
            })

        return consecutive_pairs

    def __len__(self) -> int:
        return len(self.stereo_pairs)

    def get_info(self) -> Dict:
        """Get information about the image sequence."""
        if not self.stereo_pairs:
            return {'num_pairs': 0, 'frame_range': None}

        frame_numbers = [pair['frame_number'] for pair in self.stereo_pairs]

        return {
            'num_pairs': len(self.stereo_pairs),
            'frame_range': (min(frame_numbers), max(frame_numbers)),
            'folder_path': str(self.folder_path),
            'left_folder': str(self.left_folder),
            'right_folder': str(self.right_folder),
            'first_pair': self.stereo_pairs[0] if self.stereo_pairs else None,
            'last_pair': self.stereo_pairs[-1] if self.stereo_pairs else None
        }
