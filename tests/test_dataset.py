"""
Tests for the GestureDataset.

Uses synthetic .npz files and a temporary labels.csv to test
data loading, temporal sampling, normalization, augmentation,
and class weight computation.
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from training.dataset import GestureDataset


def _create_test_data(
    data_dir: Path,
    num_clips: int = 6,
    frames_per_clip: int = 60,
    labels: list[int] | None = None,
) -> None:
    """Create synthetic test data (clips + labels.csv).

    Args:
        data_dir:        Root data directory.
        num_clips:       Number of clips to create.
        frames_per_clip: Frames per clip.
        labels:          Per-clip labels. Defaults to alternating 0/1.
    """
    clips_dir = data_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = [i % 2 for i in range(num_clips)]

    # Create .npz files with synthetic landmarks.
    for i in range(num_clips):
        T = frames_per_clip
        timestamps = np.arange(T, dtype=np.int64) * 33  # ~30fps
        landmarks = np.random.randn(T, 33, 4).astype(np.float32)
        world_landmarks = np.random.randn(T, 33, 3).astype(np.float32)
        metadata = json.dumps({
            "source": f"clip_{i:03d}.mp4",
            "fps": 30.0,
            "frame_count": T,
        })

        np.savez_compressed(
            clips_dir / f"clip_{i:03d}.npz",
            timestamps=timestamps,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            metadata=metadata,
        )

    # Create labels.csv.
    csv_path = data_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "facing", "arms_crossed", "tense"])
        for i in range(num_clips):
            writer.writerow([f"clip_{i:03d}.mp4", labels[i], 0, 0])


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory with synthetic clips."""
    _create_test_data(tmp_path, num_clips=6)
    return tmp_path


class TestGestureDataset:
    """Tests for GestureDataset."""

    def test_load_samples(self, test_data_dir):
        """Should load all 6 samples from the manifest."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        assert len(ds) == 6

    def test_item_shape(self, test_data_dir):
        """Each item should have shape (3, clip_length, 33, 1)."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        x, label = ds[0]
        assert x.shape == (3, 64, 33, 1)
        assert isinstance(label, int)

    def test_label_values(self, test_data_dir):
        """Labels should be 0 or 1."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label in (0, 1)

    def test_temporal_padding(self, tmp_path):
        """Short clips should be zero-padded to clip_length."""
        _create_test_data(tmp_path, num_clips=2, frames_per_clip=30)
        ds = GestureDataset(str(tmp_path), "facing", clip_length=128)
        x, _ = ds[0]
        assert x.shape == (3, 128, 33, 1)

        # Last frames should be zeros (padding).
        padding_region = x[:, 30:, :, :]
        assert torch.all(padding_region == 0), "Padding region should be zeros"

    def test_temporal_subsampling(self, tmp_path):
        """Long clips should be subsampled to clip_length."""
        _create_test_data(tmp_path, num_clips=2, frames_per_clip=300)
        ds = GestureDataset(str(tmp_path), "facing", clip_length=128)
        x, _ = ds[0]
        assert x.shape == (3, 128, 33, 1)

    def test_hip_centering(self, test_data_dir):
        """Normalized data should be centered on hip midpoint."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        x, _ = ds[0]

        # After centering, the hip midpoint (avg of joints 23, 24)
        # should be near zero for non-padded frames.
        hip_mid = (x[:, :, 23, 0] + x[:, :, 24, 0]) / 2.0
        # The first ~60 frames should have near-zero hip midpoint.
        hip_mid_real = hip_mid[:, :60]
        assert torch.allclose(hip_mid_real, torch.zeros_like(hip_mid_real), atol=1e-5)

    def test_augmentation_changes_data(self, test_data_dir):
        """Augmented dataset should produce different data each time."""
        ds = GestureDataset(
            str(test_data_dir), "facing", clip_length=64, augment=True,
        )
        x1, _ = ds[0]
        x2, _ = ds[0]
        # With augmentation, repeated loads should differ (probabilistic).
        # This could technically fail with very low probability.
        assert not torch.allclose(x1, x2), (
            "Augmented data should vary between loads"
        )

    def test_no_augmentation_is_deterministic(self, test_data_dir):
        """Non-augmented dataset should produce identical data."""
        ds = GestureDataset(
            str(test_data_dir), "facing", clip_length=64, augment=False,
        )
        x1, l1 = ds[0]
        x2, l2 = ds[0]
        assert torch.allclose(x1, x2)
        assert l1 == l2

    def test_class_weights(self, test_data_dir):
        """Class weights should be computed and have shape (2,)."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        weights = ds.get_class_weights()
        assert weights.shape == (2,)
        assert torch.all(weights > 0)

    def test_split_indices(self, test_data_dir):
        """Train/val split should cover all indices without overlap."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        train_idx, val_idx = ds.get_split_indices(val_fraction=0.3)

        all_idx = set(train_idx) | set(val_idx)
        assert len(all_idx) == len(ds), "Split doesn't cover all samples"
        assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap"

    def test_invalid_gesture_type(self, test_data_dir):
        """Should raise ValueError for unknown gesture type."""
        with pytest.raises(ValueError, match="Unknown gesture type"):
            GestureDataset(str(test_data_dir), "nonexistent", clip_length=64)

    def test_missing_csv(self, tmp_path):
        """Should raise FileNotFoundError if labels.csv is missing."""
        (tmp_path / "clips").mkdir()
        with pytest.raises(FileNotFoundError, match="labels.csv"):
            GestureDataset(str(tmp_path), "facing", clip_length=64)

    def test_output_dtype(self, test_data_dir):
        """Tensor output should be float32."""
        ds = GestureDataset(str(test_data_dir), "facing", clip_length=64)
        x, _ = ds[0]
        assert x.dtype == torch.float32
