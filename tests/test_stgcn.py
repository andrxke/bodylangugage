"""
Tests for the lightweight ST-GCN model.

Verifies forward pass shapes, parameter count, and gradient flow
using synthetic data. Does NOT require a GPU — all tests run on CPU.
"""

import numpy as np
import pytest
import torch

from classification.graph import MediaPipeGraph
from classification.stgcn import STGCN, SpatialGraphConv, STGCNBlock


@pytest.fixture
def graph():
    """Create a MediaPipeGraph instance."""
    return MediaPipeGraph()


@pytest.fixture
def adjacency(graph):
    """Get the partitioned adjacency matrix."""
    return graph.get_adjacency_matrix()


class TestSpatialGraphConv:
    """Tests for the SpatialGraphConv layer."""

    def test_output_shape(self, adjacency):
        """Output should have the correct channel dimension."""
        layer = SpatialGraphConv(
            in_channels=3, out_channels=32, adjacency=adjacency,
        )
        x = torch.randn(2, 3, 64, 33)  # (N, C, T, V)
        out = layer(x)
        assert out.shape == (2, 32, 64, 33)

    def test_different_batch_sizes(self, adjacency):
        """Should work with various batch sizes."""
        layer = SpatialGraphConv(
            in_channels=3, out_channels=16, adjacency=adjacency,
        )
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 33)
            out = layer(x)
            assert out.shape[0] == batch_size


class TestSTGCNBlock:
    """Tests for a single ST-GCN block."""

    def test_output_shape_no_stride(self, adjacency):
        """Without stride, temporal dimension should be preserved."""
        block = STGCNBlock(
            in_channels=3, out_channels=32,
            adjacency=adjacency, stride=1,
        )
        x = torch.randn(2, 3, 64, 33)
        out = block(x)
        assert out.shape == (2, 32, 64, 33)

    def test_output_shape_with_stride(self, adjacency):
        """With stride=2, temporal dimension should halve."""
        block = STGCNBlock(
            in_channels=32, out_channels=64,
            adjacency=adjacency, stride=2,
        )
        x = torch.randn(2, 32, 64, 33)
        out = block(x)
        assert out.shape == (2, 64, 32, 33)

    def test_residual_connection(self, adjacency):
        """Block with matching dims should use identity residual."""
        block = STGCNBlock(
            in_channels=32, out_channels=32,
            adjacency=adjacency, stride=1,
        )
        assert isinstance(block.residual, torch.nn.Identity)

    def test_residual_with_projection(self, adjacency):
        """Block with mismatched dims should use conv residual."""
        block = STGCNBlock(
            in_channels=3, out_channels=32,
            adjacency=adjacency, stride=1,
        )
        assert not isinstance(block.residual, torch.nn.Identity)


class TestSTGCN:
    """Tests for the full ST-GCN model."""

    def test_forward_shape(self, graph):
        """Output should be (N, num_classes)."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        x = torch.randn(4, 3, 128, 33, 1)  # (N, C, T, V, M)
        out = model(x)
        assert out.shape == (4, 2)

    def test_forward_single_sample(self, graph):
        """Should work with batch size 1."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        x = torch.randn(1, 3, 64, 33, 1)
        out = model(x)
        assert out.shape == (1, 2)

    def test_multiclass(self, graph):
        """Should work with more than 2 classes."""
        model = STGCN(num_classes=5, graph=graph, in_channels=3)
        x = torch.randn(2, 3, 64, 33, 1)
        out = model(x)
        assert out.shape == (2, 5)

    def test_different_temporal_lengths(self, graph):
        """Should handle various temporal dimensions."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        for T in [32, 64, 128, 256]:
            x = torch.randn(2, 3, T, 33, 1)
            out = model(x)
            assert out.shape == (2, 2), f"Failed for T={T}"

    def test_parameter_count(self, graph):
        """Model should be lightweight (~50K parameters)."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        num_params = sum(p.numel() for p in model.parameters())
        # Should be roughly 30K-80K parameters.
        assert 10_000 < num_params < 200_000, (
            f"Parameter count {num_params} outside expected range"
        )

    def test_gradient_flow(self, graph):
        """Gradients should flow through the entire model."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        x = torch.randn(2, 3, 64, 33, 1)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check that all trainable parameters received gradients.
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient for parameter: {name}"
                )
                assert not torch.all(param.grad == 0), (
                    f"Zero gradient for parameter: {name}"
                )

    def test_eval_mode(self, graph):
        """Model should produce deterministic output in eval mode."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        model.eval()

        x = torch.randn(2, 3, 64, 33, 1)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2), (
            "Model output is not deterministic in eval mode"
        )

    def test_output_not_nan(self, graph):
        """Output should not contain NaN values."""
        model = STGCN(num_classes=2, graph=graph, in_channels=3)
        x = torch.randn(2, 3, 128, 33, 1)
        out = model(x)
        assert not torch.any(torch.isnan(out)), "Output contains NaN values"
