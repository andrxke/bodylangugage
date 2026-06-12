"""
Lightweight ST-GCN (Spatial-Temporal Graph Convolutional Network).

A compact implementation of ST-GCN (Yan et al., 2018) designed for
small-dataset body language classification. Uses PyTorch standard
operations — no external GNN libraries required.

Architecture (4 blocks, ~50K parameters):
    Input:  (N, 3, T, 33, 1)
    Block1:  3 → 32 channels, stride=1
    Block2: 32 → 32 channels, stride=1
    Block3: 32 → 64 channels, stride=2  (temporal downsampling)
    Block4: 64 → 64 channels, stride=1
    Pool:   Global average pooling over time and joints
    FC:     64 → num_classes

Reference:
    Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph
    Convolutional Networks for Skeleton-Based Action Recognition.
    AAAI 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialGraphConv(nn.Module):
    """Spatial graph convolution layer.

    Performs graph convolution using partitioned adjacency matrices:
        X' = Σ_k (A_k · X · W_k)

    where A_k are the K partition matrices and W_k are learnable
    1×1 convolution weights.

    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        adjacency:    Partitioned adjacency matrices, shape (K, V, V).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
    ) -> None:
        super().__init__()

        num_partitions = adjacency.shape[0]

        # Register adjacency as a non-trainable buffer.
        self.register_buffer(
            "A",
            torch.tensor(adjacency, dtype=torch.float32),
        )

        # One 1×1 conv per partition (weight sharing across partitions
        # would lose the spatial partitioning benefit).
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_partitions,
            kernel_size=1,
        )

        self.num_partitions = num_partitions
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, T, V).

        Returns:
            Output tensor of shape (N, C_out, T, V).
        """
        N, C, T, V = x.shape
        K = self.num_partitions

        # Apply all partition convolutions at once.
        # (N, C_in, T, V) → (N, C_out * K, T, V)
        x_conv = self.conv(x)

        # Split into K partitions: (N, K, C_out, T, V)
        x_conv = x_conv.view(N, K, self.out_channels, T, V)

        # Apply each partition's adjacency matrix and sum.
        # For each partition k: multiply features by A[k] along the V dimension.
        # A shape: (K, V, V), x_conv[:, k] shape: (N, C_out, T, V)
        out = torch.zeros(N, self.out_channels, T, V, device=x.device, dtype=x.dtype)
        for k in range(K):
            # (N, C_out, T, V) @ (V, V) → (N, C_out, T, V)
            out = out + torch.einsum("nctv,vw->nctw", x_conv[:, k], self.A[k])

        return out


class STGCNBlock(nn.Module):
    """Single ST-GCN block: Spatial GCN → Temporal Conv → Residual.

    Each block consists of:
      1. Batch norm on input
      2. Spatial graph convolution (learns inter-joint relationships)
      3. Batch norm + ReLU
      4. Temporal convolution (learns temporal dynamics)
      5. Batch norm + ReLU + Dropout
      6. Residual connection (skip from input to output)

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        adjacency:    Partitioned adjacency matrices (K, V, V).
        stride:       Temporal stride (>1 for temporal downsampling).
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: np.ndarray,
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Spatial graph convolution.
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.bn_gcn = nn.BatchNorm2d(out_channels)

        # Temporal convolution (1D conv along time axis, kernel=9).
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection — must match dimensions.
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, C_in, T, V).

        Returns:
            Output tensor of shape (N, C_out, T', V),
            where T' = T // stride.
        """
        res = self.residual(x)

        # Spatial graph convolution.
        x = self.gcn(x)
        x = self.bn_gcn(x)
        x = self.relu(x)

        # Temporal convolution.
        x = self.tcn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection.
        x = x + res

        return x


class STGCN(nn.Module):
    """Lightweight ST-GCN for body language gesture classification.

    A compact variant of the original ST-GCN with 4 blocks instead
    of 10, and narrower channels (32/64 instead of 64/128/256).
    Designed for small datasets with ~50K trainable parameters.

    Args:
        num_classes: Number of output classes (2 for binary).
        graph:       Skeleton graph object with get_adjacency_matrix().
        in_channels: Number of input channels per joint (3 for x,y,z).
        dropout:     Dropout probability in each block.

    Input shape:  (N, C, T, V, M) — batch, channels, time, joints, persons
    Output shape: (N, num_classes) — logits
    """

    def __init__(
        self,
        num_classes: int,
        graph,
        in_channels: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        adjacency = graph.get_adjacency_matrix()

        # Input batch normalization.
        self.bn_input = nn.BatchNorm1d(in_channels * graph.num_nodes)

        # ST-GCN blocks.
        self.blocks = nn.Sequential(
            STGCNBlock(in_channels, 32, adjacency, stride=1, dropout=dropout),
            STGCNBlock(32, 32, adjacency, stride=1, dropout=dropout),
            STGCNBlock(32, 64, adjacency, stride=2, dropout=dropout),
            STGCNBlock(64, 64, adjacency, stride=1, dropout=dropout),
        )

        # Classification head.
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Skeleton sequence tensor of shape (N, C, T, V, M).
               N = batch size
               C = input channels (3 for x, y, z)
               T = number of frames
               V = number of joints (33)
               M = number of persons (1)

        Returns:
            Logits of shape (N, num_classes).
        """
        N, C, T, V, M = x.shape

        # Collapse person dimension (single presenter: M=1).
        x = x[:, :, :, :, 0]  # (N, C, T, V)

        # Input normalization: reshape to (N, C*V, T), apply BN, reshape back.
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)
        x = x.view(N, C * V, T)                 # (N, C*V, T)
        x = self.bn_input(x)
        x = x.view(N, C, V, T)                  # (N, C, V, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, T, V)

        # ST-GCN blocks.
        x = self.blocks(x)  # (N, 64, T', V)

        # Global average pooling over time and joints.
        x = x.mean(dim=(2, 3))  # (N, 64)

        # Classification.
        x = self.fc(x)  # (N, num_classes)

        return x
