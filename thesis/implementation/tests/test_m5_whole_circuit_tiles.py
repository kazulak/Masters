from __future__ import annotations

import numpy as np
import pytest

from quantum_bench.core.records import TensorSpec
from quantum_bench.tn.graph import ContractNode, TensorView
from quantum_bench.targets.upmem.m5_whole_circuit_tiles import (
    M5TileLimits,
    TileLoweringError,
    assemble_output_tiles,
    canonical_label_geometry,
    lower_binary_contraction,
)


def _task(
    left_labels: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_labels: tuple[int, ...],
    right_shape: tuple[int, ...],
    output_labels: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> ContractNode:
    return ContractNode(
        node_id="task",
        left=TensorView(tensor_id="left", labels=left_labels, shape=left_shape),
        right=TensorView(tensor_id="right", labels=right_labels, shape=right_shape),
        output=TensorSpec(
            id="output",
            labels=output_labels,
            shape=output_shape,
            structure="dense",
        ),
        contracted_labels=tuple(
            label
            for label in left_labels
            if label in right_labels and label not in output_labels
        ),
        output_labels=output_labels,
    )


def _contract(node: ContractNode, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum(
        left,
        list(node.left.labels),
        right,
        list(node.right.labels),
        list(node.output_labels),
        optimize=True,
    )


def _tile_partials(lowering):
    partials = {}
    for tile in lowering.tiles:
        left, right = lowering.extract_tile_operands(tile)
        partials[tile.id] = left @ right
    return partials


def test_label_lowering_matches_contract_with_one_sided_reductions():
    task = _task(
        (10, 20, 30),
        (2, 3, 4),
        (40, 20, 50),
        (5, 3, 7),
        (10,),
        (2,),
    )
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(105, dtype=np.float32).reshape(5, 3, 7)
    lowering = lower_binary_contraction(task, left, right)

    expected = _contract(task, left, right)
    actual = assemble_output_tiles(lowering, _tile_partials(lowering))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert lowering.canonical.left.shape == (1, 2, 3)
    assert lowering.canonical.right.shape == (1, 3, 1)
    assert lowering.canonical.canonical_output_labels == (10,)
    assert canonical_label_geometry(
        task.left.labels,
        task.left.shape,
        task.right.labels,
        task.right.shape,
        task.output_labels,
    ) == (
        lowering.canonical.b,
        lowering.canonical.m,
        lowering.canonical.k,
        lowering.canonical.n,
    )


def test_label_lowering_restores_permuted_output_labels():
    task = _task((10, 20), (2, 3), (20, 30), (3, 4), (30, 10), (4, 2))
    left = np.arange(6, dtype=np.float32).reshape(2, 3)
    right = np.arange(12, dtype=np.float32).reshape(3, 4)
    lowering = lower_binary_contraction(task, left, right)
    expected = _contract(task, left, right)
    actual = assemble_output_tiles(lowering, _tile_partials(lowering))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert lowering.canonical.canonical_output_labels == (10, 30)


def test_shared_output_label_is_lowered_as_batch_and_permuted_back():
    task = _task(
        (5, 0, 1),
        (2, 3, 4),
        (5, 1, 2),
        (2, 4, 5),
        (2, 5, 0),
        (5, 2, 3),
    )
    left = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    right = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    lowering = lower_binary_contraction(
        task, left, right, limits=M5TileLimits.float32(max_tile_dim=2)
    )

    assert lowering.canonical.batch_labels == (5,)
    assert lowering.canonical.left.shape == (2, 3, 4)
    assert lowering.canonical.right.shape == (2, 4, 5)
    assert {tile.batch_index for tile in lowering.tiles} == {0, 1}
    expected = _contract(task, left, right)
    actual = assemble_output_tiles(lowering, _tile_partials(lowering))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_remainder_tiles_are_contiguous_and_cover_output_without_overlap():
    task = _task((0, 1), (5, 7), (1, 2), (7, 6), (0, 2), (5, 6))
    left = np.arange(35, dtype=np.float32).reshape(5, 7)
    right = np.arange(42, dtype=np.float32).reshape(7, 6)
    limits = M5TileLimits(max_tile_dim=3)
    lowering = lower_binary_contraction(task, left, right, limits=limits)

    coverage = np.zeros((5, 6), dtype=np.int32)
    for output_tile in lowering.output_tiles:
        coverage[
            output_tile.m_start : output_tile.m_start + output_tile.m_size,
            output_tile.n_start : output_tile.n_start + output_tile.n_size,
        ] += 1
    assert np.all(coverage == 1)
    assert [tile.id for tile in lowering.tiles] == [
        tile.id
        for tile in lower_binary_contraction(task, left, right, limits=limits).tiles
    ]
    for tile in lowering.tiles:
        tile_left, tile_right = lowering.extract_tile_operands(tile)
        assert tile_left.flags.c_contiguous
        assert tile_right.flags.c_contiguous
        assert tile_left.shape == (tile.m_size, tile.k_size)
        assert tile_right.shape == (tile.k_size, tile.n_size)
        assert tile.left_element_count <= 65_536
        assert tile.right_element_count <= 65_536
        assert tile.output_element_count <= 65_536
        assert tile.k_size <= 65_536
        assert tile.aligned_mram_bytes <= 512 * 1024

    expected = _contract(task, left, right)
    np.testing.assert_allclose(
        assemble_output_tiles(lowering, _tile_partials(lowering)),
        expected,
        rtol=0,
        atol=0,
    )


def test_zero_dimensions_are_rejected_before_tile_lowering():
    task = _task((0, 1), (2, 0), (1, 2), (0, 3), (0, 2), (2, 3))
    left = np.empty((2, 0), dtype=np.float32)
    right = np.empty((0, 3), dtype=np.float32)
    with pytest.raises(TileLoweringError, match="label_dimension_is_not_positive"):
        lower_binary_contraction(task, left, right)


def test_preflight_is_conservative_and_reports_chunking():
    task = _task((0, 1), (2, 300), (1, 2), (300, 3), (0, 2), (2, 3))
    lowering = lower_binary_contraction(
        task,
        np.ones((2, 300), dtype=np.float32),
        np.ones((300, 3), dtype=np.float32),
    )
    summary = lowering.preflight
    assert summary.k_chunk_count == 2
    assert summary.max_k_chunk <= 65_536
    assert summary.max_tile_mram_bytes <= 512 * 1024
    assert summary.supported
    assert summary.as_dict()["limits"]["max_elements"] == 65_536


def test_numeric_modes_use_true_mram_widths_and_int8_only_safety_fields():
    task = _task((0, 1), (2, 2), (1, 2), (2, 2), (0, 2), (2, 2))
    left = np.ones((2, 2), dtype=np.float32)
    right = np.ones((2, 2), dtype=np.float32)
    packed = lower_binary_contraction(
        task,
        left,
        right,
        limits=M5TileLimits.host_packed_int8(max_tile_dim=2, max_mram_bytes=40),
    )
    float32 = lower_binary_contraction(
        task,
        left,
        right,
        limits=M5TileLimits.float32(max_tile_dim=2, max_mram_bytes=40),
    )

    assert len(packed.tiles) == 1
    assert packed.tiles[0].left_bytes == 4
    assert packed.tiles[0].right_bytes == 4
    assert packed.tiles[0].aligned_mram_bytes == 32
    assert len(float32.tiles) == 2
    assert all(tile.left_bytes == 8 for tile in float32.tiles)
    assert all(tile.right_bytes == 8 for tile in float32.tiles)
    assert all(tile.aligned_mram_bytes == 32 for tile in float32.tiles)
    assert packed.preflight.packed_int8_k_safe is True
    assert packed.preflight.int32_full_k_safe is True
    assert float32.preflight.packed_int8_k_safe is None
    assert float32.preflight.int32_full_k_safe is None


def test_numeric_mode_is_validated():
    with pytest.raises(ValueError, match="unsupported numeric_mode"):
        M5TileLimits(numeric_mode="implicit")


@pytest.mark.parametrize(
    "bad_left", [np.array([[np.nan]], dtype=np.float32), np.array([[1.0 + 1.0j]])]
)
def test_nonfinite_and_nonzero_complex_inputs_are_rejected(bad_left):
    task = _task((0, 1), (1, 1), (1, 2), (1, 1), (0, 2), (1, 1))
    with pytest.raises(TileLoweringError):
        lower_binary_contraction(task, bad_left, np.ones((1, 1), dtype=np.float32))


def test_zero_imaginary_complex_inputs_are_accepted():
    task = _task((0, 1), (1, 1), (1, 2), (1, 1), (0, 2), (1, 1))
    lowering = lower_binary_contraction(
        task,
        np.array([[2.0 + 0.0j]]),
        np.array([[3.0]], dtype=np.float32),
    )
    actual = assemble_output_tiles(lowering, _tile_partials(lowering))
    np.testing.assert_array_equal(actual, np.array([[6.0]], dtype=np.float32))
