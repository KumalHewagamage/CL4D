import json
import logging
import random
import math
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class StitchedDynAction4D(torch.utils.data.Dataset):
    """
    4D Temporal Grounding Dataset utilizing flattened indices for deterministic evaluation,
    X-Z root translation for stitching, and chunked normalization.
    """
    def __init__(self, config: dict):
        self.data_root = Path(config["data_root"])
        self.json_path = Path(config["stitched_json_path"])
        self.split = config["split"].lower()
        self.target_raw_frames = config["target_raw_frames"]
        self.sampled_frames = config["sampled_frames"]
        self.stride = config["stride"]
        self.num_points = config["num_points"]
        self.window_size = config["window_size"]
        self.no_normalize = config.get("no_normalize", False)
        self.augment = config.get("augment", False)
        self.seed = config.get("seed", 42)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.samples = self._build_flattened_index()
        logger.info(f"Dataset initialized with {len(self.samples)} query-video pairs.")

    def _build_flattened_index(self) -> List[Dict]:
        """
        Flattens the dataset so each specific action query is an independent sample.
        This prevents center-bias and ensures deterministic evaluation.
        """
        with open(self.json_path, 'r') as f:
            stitched_data = json.load(f)

        flattened_samples = []
        for stitched_id, seq_meta in stitched_data.items():
            for action_idx, action_meta in enumerate(seq_meta["actions"]):
                # Skip actions with unknown captions or pure backgrounds if desired
                if action_meta["caption"] == "unknown action":
                    continue
                    
                flattened_samples.append({
                    "stitched_id": stitched_id,
                    "seq_meta": seq_meta,
                    "target_action_idx": action_idx
                })
        return flattened_samples

    def _load_pcd(self, pcd_path: str) -> np.ndarray:
        points, colors, normals = [], [], []
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA ascii'):
                    header_end = i + 1
                    break
                    
            default_color, default_normal = [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]
            for line in lines[header_end:]:
                parts = line.strip().split()
                if len(parts) < 3: continue
                
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                
                n = [float(parts[3]), float(parts[4]), float(parts[5])] if len(parts) >= 6 else default_normal
                normals.append(n)
                
                c = default_color
                if len(parts) >= 7:
                    try:
                        packed_rgb = int(float(parts[6]))
                        c = [((packed_rgb >> 16) & 255) / 255.0, 
                             ((packed_rgb >> 8) & 255) / 255.0, 
                             (packed_rgb & 255) / 255.0]
                    except: pass
                colors.append(c)

            pts = np.array(points, dtype=np.float32)
            cols = np.array(colors, dtype=np.float32)
            nrms = np.array(normals, dtype=np.float32)
            if len(pts) == 0: return np.zeros((0, 9), dtype=np.float32)
            return np.concatenate([pts, cols, nrms], axis=1)
        except Exception:
            return np.zeros((0, 9), dtype=np.float32)

    def _process_points(self, point_data: np.ndarray) -> np.ndarray:
        N, T = point_data.shape[0], self.num_points
        if N == 0: return np.zeros((T, 9), dtype=np.float32)
        if N >= T:
            indices = np.random.choice(N, T, replace=False)
            return point_data[indices]
        return np.concatenate([point_data, np.zeros((T - N, 9), dtype=np.float32)], axis=0)

    def _load_and_stitch_frames(self, actions: List[Dict]) -> np.ndarray:
        """Loads physical frames and aligns their roots (X, Z) sequentially."""
        stitched_frames = []
        prev_end_frame_xz = None

        for action in actions:
            seq_dir = Path(action["sequence_path"])
            # Load frames based on the original length allocated in the JSON
            frame_files = sorted(list(seq_dir.glob('frame_*.pcd')))[:action["original_length"]]
            
            action_frames = []
            for f_path in frame_files:
                raw_data = self._load_pcd(str(f_path))
                processed = self._process_points(raw_data)
                action_frames.append(processed)
                
            if not action_frames:
                continue
                
            action_tensor = np.stack(action_frames, axis=0)  # [L, N, 9]

            # X-Z Root Translation to match the end of the previous sequence
            if prev_end_frame_xz is not None:
                curr_start_xz = np.mean(action_tensor[0, :, [0, 2]], axis=0)
                offset = prev_end_frame_xz - curr_start_xz
                action_tensor[:, :, 0] += offset[0]  # Shift X
                action_tensor[:, :, 2] += offset[1]  # Shift Z

            stitched_frames.append(action_tensor)
            
            # Store the final frame's XZ mean for the next sequence's alignment
            prev_end_frame_xz = np.mean(action_tensor[-1, :, [0, 2]], axis=0)

        # Concatenate all shifted sequences
        if stitched_frames:
            full_sequence = np.concatenate(stitched_frames, axis=0)
        else:
            full_sequence = np.zeros((1, self.num_points, 9), dtype=np.float32)
            
        # Pad by repeating the last frame if it falls short of the target raw frames
        current_len = full_sequence.shape[0]
        if current_len < self.target_raw_frames:
            padding = np.repeat(full_sequence[-1:], self.target_raw_frames - current_len, axis=0)
            full_sequence = np.concatenate([full_sequence, padding], axis=0)
            
        return full_sequence[:self.target_raw_frames]

    def _chunked_normalization(self, seq_data: np.ndarray) -> np.ndarray:
        """Normalizes spatial coordinates strictly within 32-frame blocks."""
        T = seq_data.shape[0]
        out_data = np.copy(seq_data)
        
        for start_idx in range(0, T, self.window_size):
            end_idx = min(start_idx + self.window_size, T)
            window_xyz = seq_data[start_idx:end_idx, :, 0:3]
            
            min_xyz = np.min(window_xyz, axis=(0, 1), keepdims=True)
            max_xyz = np.max(window_xyz, axis=(0, 1), keepdims=True)
            range_xyz = max_xyz - min_xyz
            range_xyz[range_xyz == 0] = 1.0 
            
            out_data[start_idx:end_idx, :, 0:3] = (window_xyz - min_xyz) / range_xyz
            
        return out_data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict, torch.Tensor]:
        sample = self.samples[idx]
        seq_meta = sample["seq_meta"]
        target_action = seq_meta["actions"][sample["target_action_idx"]]
        caption = target_action["caption"]
        
        # 1. Load, stitch, and pad raw frames [320, N, 9]
        raw_sequence = self._load_and_stitch_frames(seq_meta["actions"])
        
        # 2. Strided temporal sampling [160, N, 9]
        sampled_sequence = raw_sequence[::self.stride][:self.sampled_frames]
        
        # 3. Apply 32-frame chunked normalization
        if not self.no_normalize:
            sampled_sequence = self._chunked_normalization(sampled_sequence)
            
        seq_tensor = torch.from_numpy(sampled_sequence)
        
        # 4. Generate Ground Truth targets (Normalized Center/Width)
        # Convert raw start/end to strided bounds
        strided_start = target_action["start_frame"] / self.stride
        strided_end = target_action["end_frame"] / self.stride
        
        # Normalize bounds relative to the final temporal size (160)
        center_x = (strided_start + strided_end) / 2.0 / self.sampled_frames
        width = (strided_end - strided_start) / self.sampled_frames
        
        # Ensure values are safely clamped between 0 and 1
        center_x = max(0.0, min(1.0, float(center_x)))
        width = max(0.0, min(1.0, float(width)))

        # Format exactly as QD-DETR SetCriterion expects
        target_dict = {
            "span_labels": [{
                "spans": torch.tensor([[center_x, width]], dtype=torch.float32)
            }]
        }
        
        # Video padding mask (False means it IS valid data, True means padding)
        # For simplicity, if we guarantee it's always padded to exactly sampled_frames, 
        # it is all valid.
        vid_mask = torch.zeros(self.sampled_frames, dtype=torch.bool)
        
        return seq_tensor, caption, target_dict, vid_mask

    def __len__(self):
        return len(self.samples)


def grounding_collate_fn(batch):
    """
    Collates the data specifically for the QD-DETR architecture input.
    """
    sequences = torch.stack([item[0] for item in batch], dim=0)  # [B, T, N, 9]
    captions = [item[1] for item in batch]                       # List[str]
    targets = [item[2] for item in batch]                        # List[Dict]
    vid_masks = torch.stack([item[3] for item in batch], dim=0)  # [B, T]
    
    return sequences, captions, targets, vid_masks