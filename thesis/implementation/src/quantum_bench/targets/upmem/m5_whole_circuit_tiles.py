"""Deterministic bounded lowering for one real binary contraction.

This module is deliberately independent of the physical M5 runner.  It turns
the label semantics used by :func:`contract_binary_task` into small dense
matrix tiles that a future engine can execute on a DPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from quantum_bench.core.records import ContractionTask


DEFAULT_MAX_ELEMENTS = 65_536
DEFAULT_MAX_PACKED_K = 65_536
DEFAULT_MRAM_BYTES = 512 * 1024
DEFAULT_ALIGNMENT_BYTES = 8
DEFAULT_MAX_TILE_DIM = 256
NUMERIC_MODE_FLOAT32 = "float32"
NUMERIC_MODE_HOST_PACKED_INT8 = "host_packed_int8"
_NUMERIC_MODES = {NUMERIC_MODE_FLOAT32, NUMERIC_MODE_HOST_PACKED_INT8}
_INT32_MAX = (1 << 31) - 1
_MAX_INT8_PRODUCT = 128 * 128


class TileLoweringError(ValueError):
    """Raised when a task cannot be represented by the real M5 tile policy."""


@dataclass(frozen=True)
class M5TileLimits:
    """Conservative per-tile limits for one explicit M5 numeric mode."""

    numeric_mode: str = NUMERIC_MODE_HOST_PACKED_INT8
    max_elements: int = DEFAULT_MAX_ELEMENTS
    max_packed_k: int = DEFAULT_MAX_PACKED_K
    max_mram_bytes: int = DEFAULT_MRAM_BYTES
    alignment_bytes: int = DEFAULT_ALIGNMENT_BYTES
    max_tile_dim: int = DEFAULT_MAX_TILE_DIM

    def __post_init__(self) -> None:
        if self.numeric_mode not in _NUMERIC_MODES:
            raise ValueError(f"unsupported numeric_mode: {self.numeric_mode}")
        if self.max_elements < 1:
            raise ValueError("max_elements must be positive")
        if self.max_packed_k < 1:
            raise ValueError("max_packed_k must be positive")
        if self.max_mram_bytes < 1:
            raise ValueError("max_mram_bytes must be positive")
        if self.alignment_bytes < 1:
            raise ValueError("alignment_bytes must be positive")
        if self.max_tile_dim < 1:
            raise ValueError("max_tile_dim must be positive")

    @classmethod
    def float32(cls, **kwargs: int) -> M5TileLimits:
        return cls(numeric_mode=NUMERIC_MODE_FLOAT32, **kwargs)

    @classmethod
    def host_packed_int8(cls, **kwargs: int) -> M5TileLimits:
        return cls(numeric_mode=NUMERIC_MODE_HOST_PACKED_INT8, **kwargs)

    @property
    def input_element_bytes(self) -> int:
        return 1 if self.numeric_mode == NUMERIC_MODE_HOST_PACKED_INT8 else 4

    @property
    def output_element_bytes(self) -> int:
        return 4

    @property
    def packed_int8(self) -> bool:
        return self.numeric_mode == NUMERIC_MODE_HOST_PACKED_INT8


@dataclass(frozen=True)
class CanonicalContraction:
    """Arrays and labels for canonical batched ``(B,M,K) @ (B,K,N)``."""

    task: ContractionTask
    left: np.ndarray
    right: np.ndarray
    b: int
    m: int
    k: int
    n: int
    batch_labels: tuple[int, ...]
    free_left_labels: tuple[int, ...]
    contracted_labels: tuple[int, ...]
    free_right_labels: tuple[int, ...]
    canonical_output_labels: tuple[int, ...]
    label_dimensions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class M5OutputTile:
    id: str
    batch_index: int
    index_m: int
    index_n: int
    m_start: int
    m_size: int
    n_start: int
    n_size: int

    @property
    def element_count(self) -> int:
        return self.m_size * self.n_size


@dataclass(frozen=True)
class M5KChunk:
    id: str
    index: int
    k_start: int
    k_size: int


@dataclass(frozen=True)
class M5Tile:
    """One output-tile and K-chunk pair sent to an execution engine."""

    id: str
    batch_index: int
    output_tile_id: str
    k_chunk_id: str
    m_start: int
    m_size: int
    n_start: int
    n_size: int
    k_start: int
    k_size: int
    left_element_count: int
    right_element_count: int
    output_element_count: int
    left_bytes: int
    right_bytes: int
    output_bytes: int
    aligned_mram_bytes: int


@dataclass(frozen=True)
class M5PreflightSummary:
    supported: bool
    reason: str | None
    numeric_mode: str
    matrix_shape: tuple[int, int, int, int]
    output_shape: tuple[int, ...]
    output_elements: int
    output_bytes: int
    output_tile_count: int
    k_chunk_count: int
    tile_count: int
    max_tile_mram_bytes: int
    max_left_elements: int
    max_right_elements: int
    max_output_elements: int
    max_k_chunk: int
    packed_int8_k_safe: bool | None
    int32_full_k_safe: bool | None
    limits: M5TileLimits

    def as_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "numeric_mode": self.numeric_mode,
            "matrix_shape": self.matrix_shape,
            "output_shape": self.output_shape,
            "output_elements": self.output_elements,
            "output_bytes": self.output_bytes,
            "output_tile_count": self.output_tile_count,
            "k_chunk_count": self.k_chunk_count,
            "tile_count": self.tile_count,
            "max_tile_mram_bytes": self.max_tile_mram_bytes,
            "max_left_elements": self.max_left_elements,
            "max_right_elements": self.max_right_elements,
            "max_output_elements": self.max_output_elements,
            "max_k_chunk": self.max_k_chunk,
            "packed_int8_k_safe": self.packed_int8_k_safe,
            "int32_full_k_safe": self.int32_full_k_safe,
            "limits": {
                "max_elements": self.limits.max_elements,
                "max_packed_k": self.limits.max_packed_k,
                "max_mram_bytes": self.limits.max_mram_bytes,
                "alignment_bytes": self.limits.alignment_bytes,
                "max_tile_dim": self.limits.max_tile_dim,
                "input_element_bytes": self.limits.input_element_bytes,
                "output_element_bytes": self.limits.output_element_bytes,
            },
        }


@dataclass(frozen=True)
class M5TileLowering:
    canonical: CanonicalContraction
    output_tiles: tuple[M5OutputTile, ...]
    k_chunks: tuple[M5KChunk, ...]
    tiles: tuple[M5Tile, ...]
    preflight: M5PreflightSummary

    def extract_tile_operands(self, tile: M5Tile) -> tuple[np.ndarray, np.ndarray]:
        return extract_tile_operands(self, tile)

    def assemble(
        self, partials: Mapping[str, np.ndarray], *, dtype: np.dtype | type = np.float32
    ) -> np.ndarray:
        return assemble_output_tiles(self, partials, dtype=dtype)


def lower_binary_contraction(
    task: ContractionTask,
    left: np.ndarray,
    right: np.ndarray,
    *,
    limits: M5TileLimits = M5TileLimits(),
) -> M5TileLowering:
    """Lower a real binary task into deterministic bounded matrix tiles."""

    canonical = canonicalize_binary_contraction(task, left, right)
    tile_m, tile_k, tile_n = _choose_tile_shape(
        canonical.m, canonical.k, canonical.n, limits
    )
    output_tiles = _output_tiles(canonical.b, canonical.m, canonical.n, tile_m, tile_n)
    k_chunks = _k_chunks(canonical.k, tile_k)
    tiles = tuple(
        _make_tile(output_tile, k_chunk, limits)
        for output_tile in output_tiles
        for k_chunk in k_chunks
    )
    preflight = _preflight(canonical, output_tiles, k_chunks, tiles, limits)
    return M5TileLowering(canonical, output_tiles, k_chunks, tiles, preflight)


def canonicalize_binary_contraction(
    task: ContractionTask,
    left: np.ndarray,
    right: np.ndarray,
) -> CanonicalContraction:
    """Apply the same label semantics as ``contract_binary_task`` before GEMM."""

    left_array = _real_float32(left, "left")
    right_array = _real_float32(right, "right")
    if tuple(left_array.shape) != tuple(task.input_shapes[0]):
        raise TileLoweringError(
            f"left_shape_mismatch:{left_array.shape}!={task.input_shapes[0]}"
        )
    if tuple(right_array.shape) != tuple(task.input_shapes[1]):
        raise TileLoweringError(
            f"right_shape_mismatch:{right_array.shape}!={task.input_shapes[1]}"
        )

    left_labels = tuple(task.left_labels)
    right_labels = tuple(task.right_labels)
    output_labels = tuple(task.output_labels)
    _validate_labels(left_labels, right_labels, output_labels)
    dimensions = _label_dimensions(left_labels, right_labels, left_array, right_array)

    right_set = set(right_labels)
    output_set = set(output_labels)
    shared = set(left_labels) & set(right_labels)
    batch = tuple(
        label for label in left_labels if label in shared and label in output_set
    )
    contracted = tuple(
        label for label in left_labels if label in right_set and label not in output_set
    )
    left_reduced = tuple(
        label
        for label in left_labels
        if label not in output_set and label not in contracted
    )
    right_reduced = tuple(
        label
        for label in right_labels
        if label not in output_set and label not in contracted
    )
    free_left = tuple(
        label
        for label in left_labels
        if label not in left_reduced and label not in contracted and label not in batch
    )
    free_right = tuple(
        label
        for label in right_labels
        if label not in right_reduced and label not in contracted and label not in batch
    )
    canonical_output_labels = batch + free_left + free_right

    left_reduced_array = _sum_labels(left_array, left_labels, left_reduced)
    right_reduced_array = _sum_labels(right_array, right_labels, right_reduced)
    left_order = batch + free_left + contracted
    right_order = batch + contracted + free_right
    b = _product(tuple(dimensions[label] for label in batch))
    m = _product(tuple(dimensions[label] for label in free_left))
    k = _product(tuple(dimensions[label] for label in contracted))
    n = _product(tuple(dimensions[label] for label in free_right))
    left_matrix = _as_batched_matrix(
        left_reduced_array,
        _remaining_labels(left_labels, left_reduced),
        left_order,
        dimensions,
        first_split=len(batch),
        second_split=len(batch) + len(free_left),
    )
    right_matrix = _as_batched_matrix(
        right_reduced_array,
        _remaining_labels(right_labels, right_reduced),
        right_order,
        dimensions,
        first_split=len(batch),
        second_split=len(batch) + len(contracted),
    )
    expected_output_shape = tuple(dimensions[label] for label in output_labels)
    if tuple(task.output_shape) != expected_output_shape:
        raise TileLoweringError(
            f"output_shape_mismatch:{task.output_shape}!={expected_output_shape}"
        )
    if tuple(left_matrix.shape) != (b, m, k) or tuple(right_matrix.shape) != (
        b,
        k,
        n,
    ):
        raise TileLoweringError("canonical_matrix_shape_mismatch")
    return CanonicalContraction(
        task=task,
        left=left_matrix,
        right=right_matrix,
        b=b,
        m=m,
        k=k,
        n=n,
        batch_labels=batch,
        free_left_labels=free_left,
        contracted_labels=contracted,
        free_right_labels=free_right,
        canonical_output_labels=canonical_output_labels,
        label_dimensions=tuple(sorted(dimensions.items())),
    )


def extract_tile_operands(
    lowering: M5TileLowering, tile: M5Tile
) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous ``(m,k)`` and ``(k,n)`` operands for one tile."""

    if tile not in lowering.tiles:
        raise KeyError(f"Unknown tile: {tile.id}")
    left = lowering.canonical.left[
        tile.batch_index,
        tile.m_start : tile.m_start + tile.m_size,
        tile.k_start : tile.k_start + tile.k_size,
    ]
    right = lowering.canonical.right[
        tile.batch_index,
        tile.k_start : tile.k_start + tile.k_size,
        tile.n_start : tile.n_start + tile.n_size,
    ]
    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def assemble_output_tiles(
    lowering: M5TileLowering,
    partials: Mapping[str, np.ndarray],
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Reduce K chunks, place disjoint output tiles, and restore task labels."""

    expected_ids = {tile.id for tile in lowering.tiles}
    if set(partials) != expected_ids:
        missing = sorted(expected_ids - set(partials))
        extra = sorted(set(partials) - expected_ids)
        raise TileLoweringError(
            f"tile_result_set_mismatch:missing={missing}:extra={extra}"
        )
    canonical_shape = tuple(
        _shape_for_labels(
            lowering.canonical, lowering.canonical.canonical_output_labels
        )
    )
    canonical_output = np.zeros(
        (lowering.canonical.b, lowering.canonical.m, lowering.canonical.n),
        dtype=dtype,
    )
    by_output: dict[str, list[tuple[M5KChunk, np.ndarray]]] = {}
    chunks_by_id = {chunk.id: chunk for chunk in lowering.k_chunks}
    for tile in lowering.tiles:
        value = np.asarray(partials[tile.id], dtype=dtype)
        expected_shape = (tile.m_size, tile.n_size)
        if tuple(value.shape) != expected_shape:
            raise TileLoweringError(
                f"tile_output_shape_mismatch:{tile.id}:{value.shape}!={expected_shape}"
            )
        by_output.setdefault(tile.output_tile_id, []).append(
            (chunks_by_id[tile.k_chunk_id], value)
        )
    for output_tile in lowering.output_tiles:
        pieces = sorted(
            by_output.get(output_tile.id, ()), key=lambda item: item[0].index
        )
        if not pieces:
            raise TileLoweringError(f"missing_output_tile:{output_tile.id}")
        reduced = np.zeros((output_tile.m_size, output_tile.n_size), dtype=dtype)
        for _, piece in pieces:
            reduced = reduced + piece
        canonical_output[
            output_tile.batch_index,
            output_tile.m_start : output_tile.m_start + output_tile.m_size,
            output_tile.n_start : output_tile.n_start + output_tile.n_size,
        ] = reduced
    canonical_output = canonical_output.reshape(canonical_shape or ())
    if (
        lowering.canonical.canonical_output_labels
        != lowering.canonical.task.output_labels
    ):
        axes = tuple(
            lowering.canonical.canonical_output_labels.index(label)
            for label in lowering.canonical.task.output_labels
        )
        canonical_output = np.transpose(canonical_output, axes)
    return np.asarray(canonical_output, dtype=dtype).reshape(
        lowering.canonical.task.output_shape
    )


def _validate_labels(
    left: tuple[int, ...], right: tuple[int, ...], output: tuple[int, ...]
) -> None:
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise TileLoweringError("duplicate_input_labels_are_not_supported")
    if len(set(output)) != len(output):
        raise TileLoweringError("duplicate_output_labels_are_not_supported")
    if not set(output) <= set(left) | set(right):
        raise TileLoweringError("output_label_missing_from_inputs")


def _label_dimensions(
    left_labels: tuple[int, ...],
    right_labels: tuple[int, ...],
    left: np.ndarray,
    right: np.ndarray,
) -> dict[int, int]:
    dimensions: dict[int, int] = {}
    for labels, array in ((left_labels, left), (right_labels, right)):
        for axis, label in enumerate(labels):
            dimension = int(array.shape[axis])
            previous = dimensions.setdefault(label, dimension)
            if previous != dimension:
                raise TileLoweringError(
                    f"label_dimension_mismatch:{label}:{previous}!={dimension}"
                )
    return dimensions


def _sum_labels(
    array: np.ndarray, labels: tuple[int, ...], reduced: tuple[int, ...]
) -> np.ndarray:
    if not reduced:
        return array
    axes = tuple(labels.index(label) for label in reduced)
    return np.asarray(np.sum(array, axis=axes), dtype=np.float32)


def _remaining_labels(
    labels: tuple[int, ...], reduced: tuple[int, ...]
) -> tuple[int, ...]:
    reduced_set = set(reduced)
    return tuple(label for label in labels if label not in reduced_set)


def _as_batched_matrix(
    array: np.ndarray,
    current_labels: tuple[int, ...],
    target_labels: tuple[int, ...],
    dimensions: Mapping[int, int],
    *,
    first_split: int,
    second_split: int,
) -> np.ndarray:
    if current_labels != target_labels:
        axes = tuple(current_labels.index(label) for label in target_labels)
        array = np.transpose(array, axes)
    shape = tuple(dimensions[label] for label in target_labels)
    matrix_shape = (
        _product(shape[:first_split]),
        _product(shape[first_split:second_split]),
        _product(shape[second_split:]),
    )
    return np.ascontiguousarray(array).reshape(matrix_shape)


def _choose_tile_shape(
    m: int, k: int, n: int, limits: M5TileLimits
) -> tuple[int, int, int]:
    tile_m = min(m, limits.max_tile_dim) if m else 0
    k_cap = (
        min(limits.max_tile_dim, limits.max_packed_k)
        if limits.packed_int8
        else limits.max_tile_dim
    )
    tile_k = min(k, k_cap) if k else 0
    tile_n = min(n, limits.max_tile_dim) if n else 0
    if m and n and not tile_k:
        tile_k = 0
    while not _fits(tile_m, tile_k, tile_n, limits):
        candidates = [(tile_m, "m"), (tile_n, "n"), (tile_k, "k")]
        candidates = [(value, axis) for value, axis in candidates if value > 1]
        if not candidates:
            raise TileLoweringError("no_tile_shape_fits_m5_limits")
        _, axis = max(
            candidates, key=lambda item: (item[0], {"m": 0, "n": 1, "k": 2}[item[1]])
        )
        if axis == "m":
            tile_m = max(1, (tile_m + 1) // 2)
        elif axis == "n":
            tile_n = max(1, (tile_n + 1) // 2)
        else:
            tile_k = max(1, (tile_k + 1) // 2)
    return tile_m, tile_k, tile_n


def _fits(m: int, k: int, n: int, limits: M5TileLimits) -> bool:
    if (
        m * k > limits.max_elements
        or k * n > limits.max_elements
        or m * n > limits.max_elements
    ):
        return False
    if limits.packed_int8 and k > limits.max_packed_k:
        return False
    return (
        _aligned(m * k, limits.input_element_bytes, limits)
        + _aligned(k * n, limits.input_element_bytes, limits)
        + _aligned(n * m, limits.output_element_bytes, limits)
        <= limits.max_mram_bytes
    )


def _output_tiles(
    b: int, m: int, n: int, tile_m: int, tile_n: int
) -> tuple[M5OutputTile, ...]:
    if not b or not m or not n:
        return ()
    return tuple(
        M5OutputTile(
            id=f"b_{batch_index}:out_{m_index}_{n_index}",
            batch_index=batch_index,
            index_m=m_index,
            index_n=n_index,
            m_start=m_start,
            m_size=min(tile_m, m - m_start),
            n_start=n_start,
            n_size=min(tile_n, n - n_start),
        )
        for batch_index in range(b)
        for m_index, m_start in enumerate(range(0, m, tile_m))
        for n_index, n_start in enumerate(range(0, n, tile_n))
    )


def _k_chunks(k: int, tile_k: int) -> tuple[M5KChunk, ...]:
    if not k:
        return (M5KChunk("k_0", 0, 0, 0),)
    return tuple(
        M5KChunk(
            id=f"k_{index}", index=index, k_start=start, k_size=min(tile_k, k - start)
        )
        for index, start in enumerate(range(0, k, tile_k))
    )


def _make_tile(
    output_tile: M5OutputTile, k_chunk: M5KChunk, limits: M5TileLimits
) -> M5Tile:
    left_elements = output_tile.m_size * k_chunk.k_size
    right_elements = k_chunk.k_size * output_tile.n_size
    output_elements = output_tile.element_count
    left_bytes = left_elements * limits.input_element_bytes
    right_bytes = right_elements * limits.input_element_bytes
    output_bytes = output_elements * limits.output_element_bytes
    aligned_mram_bytes = (
        _aligned(left_elements, limits.input_element_bytes, limits)
        + _aligned(right_elements, limits.input_element_bytes, limits)
        + _aligned(output_elements, limits.output_element_bytes, limits)
    )
    if (
        left_elements > limits.max_elements
        or right_elements > limits.max_elements
        or output_elements > limits.max_elements
        or (limits.packed_int8 and k_chunk.k_size > limits.max_packed_k)
        or aligned_mram_bytes > limits.max_mram_bytes
    ):
        raise TileLoweringError(f"tile_exceeds_m5_limits:{output_tile.id}:{k_chunk.id}")
    return M5Tile(
        id=f"{output_tile.id}:{k_chunk.id}",
        batch_index=output_tile.batch_index,
        output_tile_id=output_tile.id,
        k_chunk_id=k_chunk.id,
        m_start=output_tile.m_start,
        m_size=output_tile.m_size,
        n_start=output_tile.n_start,
        n_size=output_tile.n_size,
        k_start=k_chunk.k_start,
        k_size=k_chunk.k_size,
        left_element_count=left_elements,
        right_element_count=right_elements,
        output_element_count=output_elements,
        left_bytes=left_bytes,
        right_bytes=right_bytes,
        output_bytes=output_bytes,
        aligned_mram_bytes=aligned_mram_bytes,
    )


def _preflight(
    canonical: CanonicalContraction,
    output_tiles: tuple[M5OutputTile, ...],
    k_chunks: tuple[M5KChunk, ...],
    tiles: tuple[M5Tile, ...],
    limits: M5TileLimits,
) -> M5PreflightSummary:
    output_elements = canonical.b * canonical.m * canonical.n
    output_shape = tuple(_shape_for_labels(canonical, canonical.task.output_labels))
    return M5PreflightSummary(
        supported=True,
        reason=None,
        numeric_mode=limits.numeric_mode,
        matrix_shape=(canonical.b, canonical.m, canonical.k, canonical.n),
        output_shape=output_shape,
        output_elements=output_elements,
        output_bytes=output_elements * limits.output_element_bytes,
        output_tile_count=len(output_tiles),
        k_chunk_count=len(k_chunks),
        tile_count=len(tiles),
        max_tile_mram_bytes=max((tile.aligned_mram_bytes for tile in tiles), default=0),
        max_left_elements=max((tile.left_element_count for tile in tiles), default=0),
        max_right_elements=max((tile.right_element_count for tile in tiles), default=0),
        max_output_elements=max(
            (tile.output_element_count for tile in tiles), default=0
        ),
        max_k_chunk=max((chunk.k_size for chunk in k_chunks), default=0),
        packed_int8_k_safe=(
            max((chunk.k_size for chunk in k_chunks), default=0) <= limits.max_packed_k
            if limits.packed_int8
            else None
        ),
        int32_full_k_safe=(
            canonical.k * _MAX_INT8_PRODUCT <= _INT32_MAX
            if limits.packed_int8
            else None
        ),
        limits=limits,
    )


def _shape_for_labels(
    canonical: CanonicalContraction, labels: tuple[int, ...]
) -> tuple[int, ...]:
    dimensions = dict(canonical.label_dimensions)
    return tuple(dimensions[label] for label in labels)


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _aligned(elements: int, element_bytes: int, limits: M5TileLimits) -> int:
    value = elements * element_bytes
    alignment = limits.alignment_bytes
    return ((value + alignment - 1) // alignment) * alignment


def _real_float32(value: np.ndarray, side: str) -> np.ndarray:
    array = np.asarray(value)
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise TileLoweringError(f"{side}_dtype_not_numeric") from exc
    if not np.all(finite):
        raise TileLoweringError(f"{side}_contains_nonfinite_values")
    if np.iscomplexobj(array):
        if np.any(np.imag(array) != 0):
            raise TileLoweringError(f"{side}_contains_nonzero_imaginary_values")
        array = np.real(array)
    result = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise TileLoweringError(f"{side}_overflows_float32")
    return result
