import torch

from src.models.pipelines.cl4d_module.cl4d_pipeline import CL4D


class CL4DGrounding(CL4D):
    """Run the 32-frame CL4D encoder over longer sequences in temporal chunks."""

    def __init__(self, *args, grounding_window_size: int = 32, **kwargs):
        kwargs["num_frames"] = grounding_window_size
        kwargs["feature_mode"] = "all"
        super().__init__(*args, **kwargs)
        self.grounding_window_size = grounding_window_size

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
        tokens_per_window = window_tokens.shape[1]
        if tokens_per_window != self.grounding_window_size:
            raise ValueError(
                f"Expected {self.grounding_window_size} tokens from each "
                f"{self.grounding_window_size}-frame window, got {tokens_per_window}. "
                "Adjust the CL4D patch configuration or grounding window."
            )
        return window_tokens.reshape(batch_size, num_windows * tokens_per_window, -1)
