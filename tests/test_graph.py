"""
Tests for the MediaPipe skeleton graph definition.

Verifies adjacency matrix shape, symmetry, normalization, and
graph connectivity properties.
"""

import numpy as np
import pytest

from classification.graph import MediaPipeGraph


@pytest.fixture
def graph():
    """Create a MediaPipeGraph instance."""
    return MediaPipeGraph()


class TestMediaPipeGraph:
    """Tests for MediaPipeGraph."""

    def test_num_nodes(self, graph):
        """Graph should have 33 nodes (MediaPipe landmarks)."""
        assert graph.num_nodes == 33

    def test_adjacency_shape(self, graph):
        """Adjacency matrix should be (3, 33, 33) — 3 partitions."""
        A = graph.get_adjacency_matrix()
        assert A.shape == (3, 33, 33)

    def test_adjacency_dtype(self, graph):
        """Adjacency matrices should be float32."""
        A = graph.get_adjacency_matrix()
        assert A.dtype == np.float32

    def test_adjacency_non_negative(self, graph):
        """All values in the adjacency matrices should be non-negative."""
        A = graph.get_adjacency_matrix()
        assert np.all(A >= 0)

    def test_identity_partition(self, graph):
        """Partition 0 should be derived from the identity matrix.

        After normalization, diagonal entries should be 1.0 (since
        each node's self-loop has degree 1, so D^{-1/2} * 1 * D^{-1/2} = 1).
        """
        A = graph.get_adjacency_matrix()
        identity_partition = A[0]

        # All off-diagonal entries should be 0.
        off_diag = identity_partition - np.diag(np.diag(identity_partition))
        assert np.allclose(off_diag, 0.0, atol=1e-6)

        # Diagonal entries should be 1.0 (normalized self-loop).
        assert np.allclose(np.diag(identity_partition), 1.0, atol=1e-6)

    def test_centripetal_centrifugal_coverage(self, graph):
        """Partitions 1 and 2 together should cover all edges."""
        A = graph.get_adjacency_matrix()

        # Combined non-zero entries in partitions 1 and 2.
        combined = A[1] + A[2]

        # Every edge should appear somewhere in combined.
        for i, j in graph.edges:
            assert combined[i, j] > 0 or combined[j, i] > 0, (
                f"Edge ({i}, {j}) not covered by centripetal/centrifugal partitions"
            )

    def test_adjacency_returns_copy(self, graph):
        """get_adjacency_matrix should return a copy, not a reference."""
        A1 = graph.get_adjacency_matrix()
        A2 = graph.get_adjacency_matrix()

        A1[0, 0, 0] = 999.0
        assert A2[0, 0, 0] != 999.0

    def test_edges_valid_indices(self, graph):
        """All edge indices should be in [0, 32]."""
        for i, j in graph.edges:
            assert 0 <= i < 33, f"Invalid edge start index: {i}"
            assert 0 <= j < 33, f"Invalid edge end index: {j}"

    def test_graph_connected(self, graph):
        """The skeleton graph should be connected (all nodes reachable)."""
        # BFS from node 0.
        visited = set()
        queue = [0]
        adj = {i: [] for i in range(33)}
        for i, j in graph.edges:
            adj[i].append(j)
            adj[j].append(i)

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbour in adj[node]:
                if neighbour not in visited:
                    queue.append(neighbour)

        assert len(visited) == 33, (
            f"Graph is not connected: only {len(visited)}/33 nodes reachable"
        )

    def test_hop_distances_all_computed(self, graph):
        """BFS hop distances should be computed for all 33 nodes."""
        distances = graph._compute_hop_distances()
        assert distances.shape == (33,)
        assert np.all(distances >= 0), "Some nodes have negative distance"
