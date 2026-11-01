import os
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class StitchedDynAction4D(torch.utils.data.Dataset):
    """
    4D Temporal Grounding Dataset with Fully Precomputed Disk Caching.
    All processing (stitching, striding, normalization, masking, and target formatting)
    is done upfront. `__getitem__` only performs a fast torch.load().
    """
    def __init__(self, config: dict):
        self.data_root = Path(config["data_root"])
        self.json_path = Path(config["stitched_json_path"])
        self.split = config["split"].lower()
        self.target_raw_frames = config["target_raw_frames"]
        self.sampled_frames = config["sampled_frames"]
        self.stride = config["stride"]
        self.num_points = config["num_points"]
        self.num_features = config.get("num_features", 3)
        self.window_size = config["window_size"]
        self.no_normalize = config.get("no_normalize", False)
        self.augment = config.get("augment", False)
        self.seed = config.get("seed", 42)
        
        # Disk Caching Setup
        self.cache_dir = Path(str(self.data_root) + f"_{self.split}_cache_v2")
        self._verify_or_init_disk_cache(config)
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Parse JSON and build the unique sequences and flattened index
        with open(self.json_path, 'r') as f:
            self.stitched_data = json.load(f)
            
        # Debugging: Limit dataset size for faster iteration (remove/comment out in production)
            
        # self.stitched_data = dict(list(self.stitched_data.items())[:256])
        self.samples = self._build_flattened_index()
        logger.info(f"Dataset initialized with {len(self.samples)} query-video pairs.")
        
        # FORCE PARALLEL UPFRONT CACHING BEFORE TRAINING STARTS
        self._precompute_cache_parallel()

    def _verify_or_init_disk_cache(self, current_config: dict):
        cache_meta_path = self.cache_dir / "cache_config.json"
        
        # Now including all parameters that affect the final tensor and targets
        critical_keys = [
            "stitched_json_path", "target_raw_frames", "sampled_frames", 
            "stride", "num_points", "num_features", "window_size", "no_normalize"
        ]
        cache_config_subset = {k: current_config.get(k) for k in critical_keys}

        if self.cache_dir.exists() and cache_meta_path.exists():
            with open(cache_meta_path, 'r') as f:
                saved_config = json.load(f)
            
            if saved_config != cache_config_subset:
                raise ValueError(
                    f"\n[CACHE CONFIG MISMATCH DETECTED]\n"
                    f"Saved Config: {saved_config}\n"
                    f"Current Config: {cache_config_subset}\n"
                    f"Action Required: Delete the cache directory at '{self.cache_dir}' and rerun."
                )
            logger.info(f"Valid fully-processed disk cache found at {self.cache_dir}")
        else:
            logger.info(f"Initializing new fully-processed disk cache at {self.cache_dir}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_meta_path, 'w') as f:
                json.dump(cache_config_subset, f, indent=4)

    def _build_flattened_index(self) -> List[Dict]:
        flattened_samples = []
        for stitched_id, seq_meta in self.stitched_data.items():
            for action_idx, action_meta in enumerate(seq_meta["actions"]):
                if action_meta["caption"] == "unknown action":
                    continue
                flattened_samples.append({
                    "stitched_id": stitched_id,
                    "target_action_idx": action_idx
                })
        return flattened_samples

    # =========================================================================
    # WORKER FUNCTIONS (Used only during initialization)
    # =========================================================================

    def _load_pcd(self, pcd_path: str) -> np.ndarray:
        points = []
        try:
            with open(pcd_path, 'r') as f:
                lines = f.readlines()
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('DATA ascii'):
                    header_end = i + 1
                    break
            for line in lines[header_end:]:
                parts = line.strip().split()
                if len(parts) < 3: continue
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            pts = np.array(points, dtype=np.float32)
            if len(pts) == 0:
                return np.zeros((0, self.num_features), dtype=np.float32)
            return pts
        except Exception:
            return np.zeros((0, self.num_features), dtype=np.float32)

    def _process_points(self, point_data: np.ndarray) -> np.ndarray:
        N, T = point_data.shape[0], self.num_points
        if N == 0: return np.zeros((T, self.num_features), dtype=np.float32)
        if N >= T:
            indices = np.random.choice(N, T, replace=False)
            return point_data[indices, :self.num_features]
        return np.concatenate([
            point_data[:, :self.num_features],
            np.zeros((T - N, self.num_features), dtype=np.float32)
        ], axis=0)

    def _chunked_normalization(self, seq_data: np.ndarray) -> np.ndarray:
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

    def _cache_single_stitched_video(self, stitched_id: str, seq_meta: dict) -> str:
        """Fully processes a stitched video, computes all targets, and saves a .pt dictionary."""
        disk_cache_path = self.cache_dir / f"{stitched_id}.pt"
        if disk_cache_path.exists():
            return stitched_id

        # 1. Load and Stitch Raw Physical Frames
        stitched_frames = []
        prev_end_frame_xz = None
        for action in seq_meta["actions"]:
            seq_dir = Path(action["sequence_path"])
            frame_files = sorted(list(seq_dir.glob('frame_*.pcd')))[:action["original_length"]]
            
            action_frames = []
            for f_path in frame_files:
                raw_data = self._load_pcd(str(f_path))
                processed = self._process_points(raw_data)
                action_frames.append(processed)
                
            if not action_frames:
                continue
                
            action_tensor = np.stack(action_frames, axis=0)
            if prev_end_frame_xz is not None:
                curr_start_xz = np.mean(action_tensor[0, :, [0, 2]], axis=0)
                offset = prev_end_frame_xz - curr_start_xz
                action_tensor[:, :, 0] += offset[0]
                action_tensor[:, :, 2] += offset[1]

            stitched_frames.append(action_tensor)
            prev_end_frame_xz = np.mean(action_tensor[-1, :, [0, 2]], axis=0)

        if stitched_frames:
            full_sequence = np.concatenate(stitched_frames, axis=0)
        else:
            full_sequence = np.zeros((1, self.num_points, self.num_features), dtype=np.float32)
            
        current_len = full_sequence.shape[0]
        if current_len < self.target_raw_frames:
            padding = np.repeat(full_sequence[-1:], self.target_raw_frames - current_len, axis=0)
            full_sequence = np.concatenate([full_sequence, padding], axis=0)
        
        raw_sequence = full_sequence[:self.target_raw_frames]

        # 2. Stride and Normalize
        sampled_sequence = raw_sequence[::self.stride][:self.sampled_frames]
        if not self.no_normalize:
            sampled_sequence = self._chunked_normalization(sampled_sequence)
        seq_tensor = torch.from_numpy(sampled_sequence)
        vid_mask = torch.zeros(self.sampled_frames, dtype=torch.bool)

        # 3. Precompute all Target Dicts for this sequence
        processed_actions = []
        for action in seq_meta["actions"]:
            strided_start = action["start_frame"] / self.stride
            strided_end = action["end_frame"] / self.stride
            
            center_x = (strided_start + strided_end) / 2.0 / self.sampled_frames
            width = (strided_end - strided_start) / self.sampled_frames
            
            center_x = max(0.0, min(1.0, float(center_x)))
            width = max(0.0, min(1.0, float(width)))

            target_dict = {
                "span_labels": [{
                    "spans": torch.tensor([[center_x, width]], dtype=torch.float32)
                }]
            }
            processed_actions.append({
                "caption": action["caption"],
                "target_dict": target_dict
            })

        # 4. Save Final Payload
        payload = {
            "seq_tensor": seq_tensor,
            "vid_mask": vid_mask,
            "actions": processed_actions
        }
        
        temp_path = disk_cache_path.with_suffix('.tmp.pt')
        torch.save(payload, temp_path)
        temp_path.rename(disk_cache_path)
        return stitched_id

    def _precompute_cache_parallel(self):
        missing_ids = [
            s_id for s_id in self.stitched_data.keys() 
            if not (self.cache_dir / f"{s_id}.pt").exists()
        ]
        
        if not missing_ids:
            logger.info(f"All {len(self.stitched_data)} sequences already cached for '{self.split}'.")
            return
            
        num_workers = 24
        logger.info(f"Pre-computing {len(missing_ids)} missing sequences using {num_workers} workers.")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self._cache_single_stitched_video, s_id, self.stitched_data[s_id]): s_id 
                for s_id in missing_ids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Caching {self.split}"):
                try:
                    future.result()
                except Exception as e:
                    failed_id = futures[future]
                    logger.error(f"Failed to cache sequence {failed_id}: {e}")

    # =========================================================================
    # FAST DATA LOADER EXECUTION
    # =========================================================================

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict, torch.Tensor]:
        sample = self.samples[idx]
        stitched_id = sample["stitched_id"]
        action_idx = sample["target_action_idx"]
        
        # Load the fully processed payload
        disk_cache_path = self.cache_dir / f"{stitched_id}.pt"
        payload = torch.load(disk_cache_path, map_location="cpu", weights_only=True)
        
        seq_tensor = payload["seq_tensor"]
        vid_mask = payload["vid_mask"]
        target_action = payload["actions"][action_idx]
        
        return seq_tensor, target_action["caption"], target_action["target_dict"], vid_mask

    def __len__(self):
        return len(self.samples)


def grounding_collate_fn(batch):
    sequences = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    vid_masks = torch.stack([item[3] for item in batch], dim=0)
    return sequences, captions, targets, vid_masks