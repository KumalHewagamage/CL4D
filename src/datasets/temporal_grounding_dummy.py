import torch
from torch.utils.data import Dataset


class DummyTemporalGroundingDataset(Dataset):
    """Shape-compatible placeholder until the temporal grounding dataset is wired."""

    def __init__(
        self,
        size: int = 16,
        num_frames: int = 32,
        num_points: int = 2048,
        sequence_features: int = 9,
        seed: int = 42,
        **_,
    ):
        self.size = size
        self.num_frames = num_frames
        self.num_points = num_points
        self.sequence_features = sequence_features
        generator = torch.Generator().manual_seed(seed)
        self.motions = torch.randn(
            size, num_frames, num_points, sequence_features, generator=generator
        )
        self.spans = torch.tensor([[0.5, 0.25]], dtype=torch.float32).repeat(size, 1)
        self.queries = ["a person performs an action"] * size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "motion": self.motions[index],
            "query": self.queries[index],
            "span": self.spans[index],
        }


def collate_fn(batch):
    return {
        "motion": torch.stack([item["motion"] for item in batch]),
        "queries": [item["query"] for item in batch],
        "spans": [item["span"] for item in batch],
    }
