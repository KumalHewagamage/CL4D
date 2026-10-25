import torch

from src.models.pipelines.cl4d_module.cl4d_pipeline import CL4D


class CL4DGrounding(CL4D):
    """Run CL4D windows and retain the complete ViT token sequence."""

    def __init__(self, *args, grounding_window_size: int = 32, **kwargs):
        kwargs["num_frames"] = grounding_window_size
        kwargs["feature_mode"] = "all"
        super().__init__(*args, **kwargs)
        self.grounding_window_size = grounding_window_size
        grid_size = getattr(self.motion_encoder.model.patch_embed, "grid_size", None)
        if grid_size is None:
            raise ValueError("The configured motion encoder does not expose a patch grid.")
        self.window_grid_size = tuple(int(size) for size in grid_size)
        self.tokens_per_window = self.window_grid_size[0] * self.window_grid_size[1] + 1

    def output_tokens_for_frames(self, frames: int) -> int:
        if frames % self.grounding_window_size != 0:
            raise ValueError(
                f"{frames} frames cannot be split into "
                f"{self.grounding_window_size}-frame windows."
            )
        return frames // self.grounding_window_size * self.tokens_per_window

    def expand_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """Expand a frame mask to the flattened ViT patch-token layout."""
        batch_size, frames = padding_mask.shape
        if frames % self.grounding_window_size != 0:
            raise ValueError(
                f"Padding mask has {frames} frames, which is not divisible by "
                f"grounding_window_size={self.grounding_window_size}."
            )

        num_windows = frames // self.grounding_window_size
        temporal_patches, spatial_patches = self.window_grid_size
        if self.grounding_window_size % temporal_patches != 0:
            raise ValueError(
                f"Motion encoder temporal patch grid {temporal_patches} does not "
                f"divide the {self.grounding_window_size}-frame window."
            )
        frames_per_patch = self.grounding_window_size // temporal_patches
        window_mask = padding_mask.reshape(
            batch_size, num_windows, self.grounding_window_size
        )
        window_mask = window_mask.reshape(
            batch_size, num_windows, temporal_patches, frames_per_patch
        )
        patch_mask = window_mask.all(dim=-1).unsqueeze(-1).expand(
            -1, -1, -1, spatial_patches
        )
        patch_mask = patch_mask.reshape(batch_size, num_windows, -1)
        cls_mask = window_mask.all(dim=(2, 3)).unsqueeze(-1)
        return torch.cat([cls_mask, patch_mask], dim=-1).reshape(
            batch_size, num_windows * self.tokens_per_window
        )

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        batch_size, frames, points, features = motion.shape
        if frames % self.grounding_window_size != 0:
            raise ValueError(
                f"Grounding input has {frames} frames, which is not divisible by "
                f"grounding_window_size={self.grounding_window_size}."
            )

        num_windows = frames // self.grounding_window_size
        windows = motion.reshape(
            batch_size * num_windows,
            self.grounding_window_size,
            points,
            features,
        )
        window_tokens = super().encode_motion(windows)
        if window_tokens.ndim != 3:
            raise ValueError(
                "CL4D grounding requires feature_mode='all' to return "
                "[batch, tokens, dimension] features."
            )
        if window_tokens.shape[1] != self.tokens_per_window:
            raise ValueError(
                f"Expected {self.tokens_per_window} tokens including CLS "
                f"from each window, got {window_tokens.shape[1]}."
            )
        return window_tokens.reshape(batch_size, num_windows * self.tokens_per_window, -1)
