"""
MediaPipe 33-joint skeleton graph for ST-GCN.

Defines the spatial graph structure that ST-GCN uses for graph
convolution. The edges are derived from the same skeleton connections
used in the visualization module (skeleton_renderer.py), ensuring
consistency across the pipeline.

The adjacency matrix uses the "spatial configuration" partitioning
strategy from the original ST-GCN paper (Yan et al., 2018), which
splits each node's neighbours into three subsets:
  1. The node itself (identity / self-loop)
  2. Centripetal neighbours (closer to the body center)
  3. Centrifugal neighbours (farther from the body center)

This partitioning gives the network separate learnable weights for
each spatial relationship, which improves performance over a single
adjacency matrix.
"""

from __future__ import annotations

import numpy as np

from config import LANDMARK_INDICES


class MediaPipeGraph:
    """Skeleton graph for MediaPipe's 33-landmark pose model.

    Attributes:
        num_nodes: Number of joints (33).
        center_joint: Index of the body center joint (hip midpoint
                      approximated by left hip, index 23).
        edges: List of (start, end) joint index pairs.
        num_partitions: Number of adjacency matrix partitions (3).
    """

    num_nodes: int = 33
    num_partitions: int = 3

    # Use left hip as the center joint for distance computation.
    # (The true center is the midpoint of hips 23 & 24, but using
    # an actual joint simplifies the graph distance calculation.)
    center_joint: int = LANDMARK_INDICES["left_hip"]  # 23

    # Skeleton edges — same connections drawn by skeleton_renderer.py.
    # fmt: off
    edges: list[tuple[int, int]] = [
        # Torso
        (11, 12),  # Left shoulder → Right shoulder
        (11, 23),  # Left shoulder → Left hip
        (12, 24),  # Right shoulder → Right hip
        (23, 24),  # Left hip → Right hip

        # Left arm
        (11, 13),  # Left shoulder → Left elbow
        (13, 15),  # Left elbow → Left wrist
        (15, 17),  # Left wrist → Left pinky
        (15, 19),  # Left wrist → Left index
        (15, 21),  # Left wrist → Left thumb
        (17, 19),  # Left pinky → Left index (palm)

        # Right arm
        (12, 14),  # Right shoulder → Right elbow
        (14, 16),  # Right elbow → Right wrist
        (16, 18),  # Right wrist → Right pinky
        (16, 20),  # Right wrist → Right index
        (16, 22),  # Right wrist → Right thumb
        (18, 20),  # Right pinky → Right index (palm)

        # Left leg
        (23, 25),  # Left hip → Left knee
        (25, 27),  # Left knee → Left ankle
        (27, 29),  # Left ankle → Left heel
        (27, 31),  # Left ankle → Left foot index
        (29, 31),  # Left heel → Left foot index

        # Right leg
        (24, 26),  # Right hip → Right knee
        (26, 28),  # Right knee → Right ankle
        (28, 30),  # Right ankle → Right heel
        (28, 32),  # Right ankle → Right foot index
        (30, 32),  # Right heel → Right foot index

        # Head
        (0, 1),    # Nose → Left eye inner
        (0, 4),    # Nose → Right eye inner
        (1, 2),    # Left eye inner → Left eye
        (2, 3),    # Left eye → Left eye outer
        (3, 7),    # Left eye outer → Left ear
        (4, 5),    # Right eye inner → Right eye
        (5, 6),    # Right eye → Right eye outer
        (6, 8),    # Right eye outer → Right ear
        (9, 10),   # Mouth left → Mouth right
        (0, 9),    # Nose → Mouth left
        (0, 10),   # Nose → Mouth right

        # Head-to-torso (nose to shoulder midpoint approximated
        # by connecting nose to both shoulders).
        (0, 11),   # Nose → Left shoulder
        (0, 12),   # Nose → Right shoulder
    ]
    # fmt: on

    def __init__(self) -> None:
        """Precompute the adjacency matrices."""
        self._adjacency = self._build_partitioned_adjacency()

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the partitioned adjacency matrices.

        Returns:
            Array of shape (3, 33, 33) containing the three normalized
            partition matrices: [identity, centripetal, centrifugal].
        """
        return self._adjacency.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> np.ndarray:
        """Build the raw symmetric adjacency matrix (with self-loops).

        Returns:
            Binary adjacency matrix of shape (33, 33).
        """
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0

        # Add self-loops.
        np.fill_diagonal(A, 1.0)

        return A

    def _compute_hop_distances(self) -> np.ndarray:
        """Compute shortest-path distance from each joint to center_joint.

        Uses BFS on the skeleton graph.

        Returns:
            Array of shape (33,) with hop distances.
        """
        # Build adjacency list.
        adj_list: dict[int, list[int]] = {i: [] for i in range(self.num_nodes)}
        for i, j in self.edges:
            adj_list[i].append(j)
            adj_list[j].append(i)

        # BFS from center_joint.
        distances = np.full(self.num_nodes, -1, dtype=np.int32)
        distances[self.center_joint] = 0
        queue = [self.center_joint]
        head = 0

        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbour in adj_list[node]:
                if distances[neighbour] == -1:
                    distances[neighbour] = distances[node] + 1
                    queue.append(neighbour)

        return distances

    def _build_partitioned_adjacency(self) -> np.ndarray:
        """Build the spatial-configuration partitioned adjacency matrices.

        Partitions each node's neighbourhood into three subsets:
          0. Self-loop (identity matrix)
          1. Centripetal: neighbour is closer to the center than the node
          2. Centrifugal: neighbour is farther from (or same distance to)
             the center than the node

        Each partition is separately normalized (symmetric normalization:
        D^{-1/2} A D^{-1/2}).

        Returns:
            Array of shape (3, 33, 33).
        """
        hop_dist = self._compute_hop_distances()
        A_full = self._build_adjacency()

        partitions = np.zeros(
            (self.num_partitions, self.num_nodes, self.num_nodes),
            dtype=np.float32,
        )

        # Partition 0: self-loops (identity).
        partitions[0] = np.eye(self.num_nodes, dtype=np.float32)

        # Partition 1 & 2: centripetal and centrifugal.
        for i, j in self.edges:
            if hop_dist[j] < hop_dist[i]:
                # j is closer to center → centripetal for node i.
                partitions[1][i, j] = 1.0
                partitions[2][j, i] = 1.0
            elif hop_dist[j] > hop_dist[i]:
                # j is farther from center → centrifugal for node i.
                partitions[2][i, j] = 1.0
                partitions[1][j, i] = 1.0
            else:
                # Same distance — assign to centrifugal by convention.
                partitions[2][i, j] = 1.0
                partitions[2][j, i] = 1.0

        # Normalize each partition with symmetric normalization.
        for k in range(self.num_partitions):
            partitions[k] = self._normalize(partitions[k])

        return partitions

    @staticmethod
    def _normalize(A: np.ndarray) -> np.ndarray:
        """Symmetric normalization: D^{-1/2} A D^{-1/2}.

        Args:
            A: Raw adjacency matrix (N, N).

        Returns:
            Normalized adjacency matrix (N, N).
        """
        d = np.sum(A, axis=1)

        # Avoid division by zero for isolated nodes.
        d_inv_sqrt = np.zeros_like(d)
        nonzero = d > 0
        d_inv_sqrt[nonzero] = np.power(d[nonzero], -0.5)

        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt
