"""Pure lowering of one scheduled UPMEM cohort into prepared wave records."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from quantum_bench.model import ContractNode
from quantum_bench.numerics import EncodedComplexTensor
from quantum_bench.upmem.packed_wave import MAX_DPUS, MAX_TASKLETS, WaveTile
from quantum_bench.upmem.plan import UpmemStage, UpmemWorkUnit
from quantum_bench.upmem.tiling import M5Tile, M5TileLowering
from quantum_bench.upmem.wave_protocol import (
    FOUR_PRODUCT_PANEL,
    FOUR_PRODUCT_OUTER,
    FOUR_PRODUCT_KERNELS,
    IDLE,
    NO_OPERATION,
    REAL_PANEL,
    REAL_OUTER,
    REAL_KERNELS,
    WaveControl,
    product_layout,
)


_UINT64_MAX = (1 << 64) - 1
_MRAM_ADMISSION_ERROR = "kernel working set exceeds the MRAM arena"
_NUMERIC_MODE_NAMES = ("float32", "host_packed_int8")


def build_cohort_waves(
    stage: UpmemStage,
    lowerings: Mapping[str, M5TileLowering],
    operands: Mapping[
        str, tuple[EncodedComplexTensor, EncodedComplexTensor]
    ],
    *,
    dpu_count: int,
    tasklets: int,
    numeric_mode: int,
    request_start: int,
    fuse: bool,
    geometry_policy: str = "panel_only_v1",
) -> tuple[tuple[tuple[WaveTile, ...], ...], tuple[int, ...]]:
    """Build dense prepared waves for one already-scheduled contract cohort.

    Operands are encoded in each lowering's canonical ``(B, M, K)`` and
    ``(B, K, N)`` layouts.  The scheduled work units are only validated and
    rearranged into dense DPU slots; their geometry and ownership are never
    changed.

    ``generic_lanes`` is parallel to the returned waves.  Its value is the
    real-product lane represented by a ``REAL_PANEL`` tile in that micro-wave,
    so values repeat ``0, 1, 2, 3`` for each original wave requiring generic
    work.  A fused-only micro-wave has lane ``0``; its fused controls still
    produce all four products.
    """

    if geometry_policy not in ("panel_only_v1", "outer_k1_v1"):
        raise ValueError("unsupported geometry kernel policy")
    _validate_call_arguments(
        stage,
        lowerings,
        operands,
        dpu_count=dpu_count,
        tasklets=tasklets,
        numeric_mode=numeric_mode,
        request_start=request_start,
        fuse=fuse,
    )

    node_index = {node_id: index for index, node_id in enumerate(stage.node_ids)}
    units_by_node: dict[str, list[tuple[int, UpmemWorkUnit]]] = {
        node_id: [] for node_id in stage.node_ids
    }
    unit_by_index: dict[int, UpmemWorkUnit] = {}
    unit_index_by_id: dict[str, int] = {}
    for index, unit in enumerate(stage.work_units):
        if not isinstance(unit, UpmemWorkUnit):
            raise TypeError("stage work_units must contain UpmemWorkUnit records")
        if unit.node_id not in units_by_node:
            raise ValueError(
                f"stage work unit {unit.stable_tile_id!r} references an unknown node"
            )
        if unit.logical_rank != 0:
            raise ValueError("cohort wave encoding requires rank-zero work units")
        if not 0 <= unit.logical_dpu < dpu_count:
            raise ValueError(
                f"stage work unit {unit.stable_tile_id!r} has an invalid DPU slot"
            )
        if unit.stable_tile_id in unit_index_by_id:
            raise ValueError("stage work units contain duplicate stable tile IDs")
        unit_by_index[index] = unit
        unit_index_by_id[unit.stable_tile_id] = index
        units_by_node[unit.node_id].append((index, unit))

    tile_by_index: dict[int, M5Tile] = {}
    operands_by_node: dict[
        str, tuple[EncodedComplexTensor, EncodedComplexTensor]
    ] = {}
    for node_id in stage.node_ids:
        lowering = lowerings[node_id]
        _validate_lowering_and_operands(
            node_id,
            lowering,
            operands[node_id],
            units_by_node[node_id],
            numeric_mode=numeric_mode,
        )
        tiles_by_id: dict[str, M5Tile] = {}
        for tile in lowering.tiles:
            stable_tile_id = f"{node_id}:{tile.id}"
            if stable_tile_id in tiles_by_id:
                raise ValueError(
                    f"lowering for node {node_id!r} contains duplicate tile IDs"
                )
            tiles_by_id[stable_tile_id] = tile

        units = units_by_node[node_id]
        if len(units) != len(tiles_by_id) or {
            unit.stable_tile_id for _, unit in units
        } != set(tiles_by_id):
            raise ValueError(
                f"stage work-unit set differs from lowering for node {node_id!r}"
            )
        for index, unit in units:
            tile = tiles_by_id.get(unit.stable_tile_id)
            if tile is None:
                raise ValueError(
                    f"stage work unit {unit.stable_tile_id!r} references an unknown tile"
                )
            expected = (
                tile.batch_index,
                1,
                tile.m_start,
                tile.m_size,
                tile.n_start,
                tile.n_size,
                tile.k_start,
                tile.k_size,
                tile.left_bytes + tile.right_bytes,
                tile.output_bytes,
                tile.aligned_mram_bytes,
                tile.m_size * tile.n_size * tile.k_size,
            )
            actual = (
                unit.batch_start,
                unit.batch_size,
                unit.m_start,
                unit.m_size,
                unit.n_start,
                unit.n_size,
                unit.k_start,
                unit.k_size,
                unit.estimated_input_bytes,
                unit.estimated_output_bytes,
                unit.aligned_mram_bytes,
                unit.estimated_arithmetic_work,
            )
            if actual != expected:
                raise ValueError(
                    f"stage work unit {unit.stable_tile_id!r} extents differ "
                    "from lowering"
                )
            tile_by_index[index] = tile
        operands_by_node[node_id] = operands[node_id]

    wave_numbers = sorted({unit.wave for unit in stage.work_units})
    if wave_numbers != list(range(len(wave_numbers))):
        raise ValueError("stage work-unit waves must be dense starting at zero")
    for node_id, units in units_by_node.items():
        node_wave_numbers = sorted({unit.wave for _, unit in units})
        if node_wave_numbers != list(range(len(node_wave_numbers))):
            raise ValueError(
                f"stage work-unit waves for node {node_id!r} must be dense starting at zero"
            )

    units_by_wave: dict[int, list[tuple[int, UpmemWorkUnit]]] = {
        wave: [] for wave in wave_numbers
    }
    owners: list[int | None] = [None] * dpu_count
    for index, unit in unit_by_index.items():
        slot = unit.logical_dpu
        operation = node_index[unit.node_id]
        previous_owner = owners[slot]
        if previous_owner is not None and previous_owner != operation:
            raise ValueError(
                f"DPU {slot} changes cohort operation ownership across waves"
            )
        owners[slot] = operation
        units_by_wave[unit.wave].append((index, unit))

    for wave, units in units_by_wave.items():
        seen_slots: set[int] = set()
        for _, unit in units:
            if unit.logical_dpu in seen_slots:
                raise ValueError(f"cohort wave {wave} reuses a DPU slot")
            seen_slots.add(unit.logical_dpu)
        if not seen_slots:
            raise ValueError(f"cohort wave {wave} has no active work unit")

    layouts: dict[int, tuple[tuple[int, int], ...]] = {}
    fused_by_index: dict[int, bool] = {}
    for index, tile in tile_by_index.items():
        real_layout = product_layout(
            tile.m_size,
            tile.n_size,
            tile.k_size,
            numeric_mode=numeric_mode,
            kernel=REAL_PANEL,
        )
        if fuse:
            try:
                fused_layout = product_layout(
                    tile.m_size,
                    tile.n_size,
                    tile.k_size,
                    numeric_mode=numeric_mode,
                    kernel=FOUR_PRODUCT_PANEL,
                )
            except ValueError as exc:
                if str(exc) != _MRAM_ADMISSION_ERROR:
                    raise
                layouts[index] = real_layout
                fused_by_index[index] = False
                continue
            layouts[index] = fused_layout
            fused_by_index[index] = True
        else:
            layouts[index] = real_layout
            fused_by_index[index] = False

    waves: list[tuple[WaveTile, ...]] = []
    generic_lanes: list[int] = []
    for original_wave in wave_numbers:
        active_by_dpu = {
            unit.logical_dpu: (index, unit)
            for index, unit in units_by_wave[original_wave]
        }
        has_generic = any(
            not fused_by_index[index] for index, _ in active_by_dpu.values()
        )
        micro_lanes = range(4) if has_generic else range(1)
        for lane in micro_lanes:
            wave_id = len(waves)
            request_sequence = request_start + wave_id
            if request_sequence > _UINT64_MAX:
                raise ValueError("wave request sequence exceeds uint64")
            dense_slots: list[WaveTile] = []
            for dpu_id in range(dpu_count):
                active = active_by_dpu.get(dpu_id)
                if active is None:
                    dense_slots.append(
                        _idle_tile(
                            dpu_id,
                            tasklets,
                            numeric_mode,
                            wave_id,
                            request_sequence,
                        )
                    )
                    continue

                index, unit = active
                fused = fused_by_index[index]
                if fused and lane != 0:
                    dense_slots.append(
                        _idle_tile(
                            dpu_id,
                            tasklets,
                            numeric_mode,
                            wave_id,
                            request_sequence,
                        )
                    )
                    continue

                node_id = unit.node_id
                left, right = operands_by_node[node_id]
                kernel = FOUR_PRODUCT_PANEL if fused else REAL_PANEL
                if geometry_policy == "outer_k1_v1" and unit.k_size == 1:
                    kernel = FOUR_PRODUCT_OUTER if fused else REAL_OUTER
                inputs = _tile_inputs(
                    left,
                    right,
                    unit,
                    layouts[index],
                    numeric_mode=numeric_mode,
                    kernel=kernel,
                    lane=lane,
                )
                control = WaveControl(
                    dpu_id=dpu_id,
                    tasklets=tasklets,
                    flags=0,
                    numeric_mode=numeric_mode,
                    kernel=kernel,
                    operation_index=node_index[node_id],
                    wave_id=wave_id,
                    request_sequence=request_sequence,
                    tile_id=index,
                    batch_index=unit.batch_start,
                    m=unit.m_size,
                    n=unit.n_size,
                    k=unit.k_size,
                    k_offset=unit.k_start,
                    planes=layouts[index],
                )
                control.validate()
                dense_slots.append(
                    WaveTile(control, unit.m_start, unit.n_start, inputs)
                )
            waves.append(tuple(dense_slots))
            generic_lanes.append(lane)

    return tuple(waves), tuple(generic_lanes)


def _validate_call_arguments(
    stage: UpmemStage,
    lowerings: Mapping[str, M5TileLowering],
    operands: Mapping[str, tuple[EncodedComplexTensor, EncodedComplexTensor]],
    *,
    dpu_count: int,
    tasklets: int,
    numeric_mode: int,
    request_start: int,
    fuse: bool,
) -> None:
    if not isinstance(stage, UpmemStage):
        raise TypeError("stage must be an UpmemStage")
    if stage.kind != "contract_batch":
        raise ValueError("cohort wave encoding requires a contract_batch stage")
    if not isinstance(lowerings, Mapping) or not isinstance(operands, Mapping):
        raise TypeError("lowerings and operands must be mappings")
    if set(lowerings) != set(stage.node_ids):
        raise ValueError("lowering keys must exactly match stage node_ids")
    if set(operands) != set(stage.node_ids):
        raise ValueError("operand keys must exactly match stage node_ids")
    if type(dpu_count) is not int or not 1 <= dpu_count <= MAX_DPUS:
        raise ValueError(f"dpu_count must be an integer in [1, {MAX_DPUS}]")
    if len(stage.node_ids) > dpu_count:
        raise ValueError("cohort operation count exceeds the DPU count")
    if type(tasklets) is not int or not 1 <= tasklets <= MAX_TASKLETS:
        raise ValueError(f"tasklets must be an integer in [1, {MAX_TASKLETS}]")
    if type(numeric_mode) is not int or numeric_mode not in (0, 1):
        raise ValueError("numeric_mode must be 0 or 1")
    if type(request_start) is not int or not 0 <= request_start <= _UINT64_MAX:
        raise ValueError("request_start must be a uint64")
    if type(fuse) is not bool:
        raise TypeError("fuse must be a bool")


def _validate_lowering_and_operands(
    node_id: str,
    lowering: M5TileLowering,
    operand_pair: tuple[EncodedComplexTensor, EncodedComplexTensor],
    units: list[tuple[int, UpmemWorkUnit]],
    *,
    numeric_mode: int,
) -> None:
    if not isinstance(lowering, M5TileLowering):
        raise TypeError(f"lowering for node {node_id!r} must be an M5TileLowering")
    canonical = lowering.canonical
    if not isinstance(canonical.node, ContractNode) or canonical.node.node_id != node_id:
        raise ValueError(f"lowering node identity does not match {node_id!r}")
    expected_mode_name = _NUMERIC_MODE_NAMES[numeric_mode]
    if lowering.preflight.numeric_mode != expected_mode_name:
        raise ValueError(
            f"lowering numeric policy for node {node_id!r} does not match numeric_mode"
        )
    dimensions = (canonical.b, canonical.m, canonical.k, canonical.n)
    if any(type(value) is not int or value < 1 for value in dimensions):
        raise ValueError(f"lowering canonical geometry for node {node_id!r} is invalid")
    if tuple(canonical.left.shape) != (canonical.b, canonical.m, canonical.k):
        raise ValueError(f"lowering left canonical shape for node {node_id!r} is invalid")
    if tuple(canonical.right.shape) != (canonical.b, canonical.k, canonical.n):
        raise ValueError(f"lowering right canonical shape for node {node_id!r} is invalid")
    if tuple(lowering.preflight.matrix_shape) != dimensions:
        raise ValueError(
            f"lowering preflight geometry for node {node_id!r} is inconsistent"
        )
    if not isinstance(lowering.tiles, tuple) or not lowering.tiles:
        raise ValueError(f"lowering for node {node_id!r} has no tiles")
    if type(operand_pair) is not tuple or len(operand_pair) != 2:
        raise TypeError(
            f"operands for node {node_id!r} must be a pair of encoded tensors"
        )
    expected_dtype = np.dtype("<f4") if numeric_mode == 0 else np.dtype("<i1")
    for side, (encoded, expected_shape) in enumerate(
        zip(
            operand_pair,
            (
                tuple(canonical.left.shape),
                tuple(canonical.right.shape),
            ),
            strict=True,
        )
    ):
        if not isinstance(encoded, EncodedComplexTensor):
            raise TypeError(
                f"operand {side} for node {node_id!r} must be an EncodedComplexTensor"
            )
        if tuple(encoded.real.shape) != expected_shape or tuple(encoded.imag.shape) != expected_shape:
            raise ValueError(
                f"operand {side} for node {node_id!r} is not in canonical shape"
            )
        if encoded.real.dtype != expected_dtype or encoded.imag.dtype != expected_dtype:
            raise ValueError(
                f"operand {side} for node {node_id!r} has the wrong numeric policy"
            )
        if not encoded.real.flags.c_contiguous or not encoded.imag.flags.c_contiguous:
            raise ValueError(f"operand {side} for node {node_id!r} is not contiguous")
        if numeric_mode == 0 and (
            encoded.scale != 1.0
            or encoded.saturation_real != 0
            or encoded.saturation_imag != 0
        ):
            raise ValueError(
                f"float32 operand {side} for node {node_id!r} has quantized metadata"
            )
    for tile in lowering.tiles:
        if not isinstance(tile, M5Tile):
            raise TypeError(f"lowering for node {node_id!r} contains a non-M5 tile")
    if not units:
        raise ValueError(f"stage has no work units for node {node_id!r}")


def _tile_inputs(
    left: EncodedComplexTensor,
    right: EncodedComplexTensor,
    unit: UpmemWorkUnit,
    layout: tuple[tuple[int, int], ...],
    *,
    numeric_mode: int,
    kernel: int,
    lane: int,
) -> tuple[bytes, bytes, bytes, bytes]:
    dtype = np.dtype("<f4") if numeric_mode == 0 else np.dtype("<i1")
    if kernel in FOUR_PRODUCT_KERNELS:
        return (
            _encoded_slice(
                left.real,
                unit.batch_start,
                unit.m_start,
                unit.m_size,
                unit.k_start,
                unit.k_size,
                layout[0][1],
                dtype,
            ),
            _encoded_slice(
                left.imag,
                unit.batch_start,
                unit.m_start,
                unit.m_size,
                unit.k_start,
                unit.k_size,
                layout[1][1],
                dtype,
            ),
            _encoded_slice(
                right.real,
                unit.batch_start,
                unit.k_start,
                unit.k_size,
                unit.n_start,
                unit.n_size,
                layout[2][1],
                dtype,
            ),
            _encoded_slice(
                right.imag,
                unit.batch_start,
                unit.k_start,
                unit.k_size,
                unit.n_start,
                unit.n_size,
                layout[3][1],
                dtype,
            ),
        )
    if kernel not in REAL_KERNELS or lane not in range(4):
        raise ValueError("invalid real-panel lane")
    lane_operands = (
        (left.real, right.real),
        (left.imag, right.imag),
        (left.real, right.imag),
        (left.imag, right.real),
    )
    selected_left, selected_right = lane_operands[lane]
    return (
        _encoded_slice(
            selected_left,
            unit.batch_start,
            unit.m_start,
            unit.m_size,
            unit.k_start,
            unit.k_size,
            layout[0][1],
            dtype,
        ),
        b"",
        _encoded_slice(
            selected_right,
            unit.batch_start,
            unit.k_start,
            unit.k_size,
            unit.n_start,
            unit.n_size,
            layout[2][1],
            dtype,
        ),
        b"",
    )


def _encoded_slice(
    plane: np.ndarray,
    batch: int,
    row_start: int,
    rows: int,
    column_start: int,
    columns: int,
    aligned_length: int,
    dtype: np.dtype,
) -> bytes:
    try:
        view = plane[
            batch,
            row_start : row_start + rows,
            column_start : column_start + columns,
        ]
    except (IndexError, TypeError) as exc:
        raise ValueError("encoded operand slice is outside canonical geometry") from exc
    if tuple(view.shape) != (rows, columns):
        raise ValueError("encoded operand slice is outside canonical geometry")
    contiguous = np.ascontiguousarray(view, dtype=dtype)
    payload = contiguous.tobytes(order="C")
    if len(payload) > aligned_length:
        raise ValueError("encoded operand slice exceeds product layout")
    return payload + bytes(aligned_length - len(payload))


def _idle_tile(
    dpu_id: int,
    tasklets: int,
    numeric_mode: int,
    wave_id: int,
    request_sequence: int,
) -> WaveTile:
    control = WaveControl(
        dpu_id=dpu_id,
        tasklets=tasklets,
        flags=IDLE,
        numeric_mode=numeric_mode,
        kernel=0,
        operation_index=NO_OPERATION,
        wave_id=wave_id,
        request_sequence=request_sequence,
        tile_id=0,
        batch_index=0,
        m=0,
        n=0,
        k=0,
        k_offset=0,
        planes=((0, 0),) * 8,
    )
    control.validate()
    return WaveTile(control, 0, 0, (b"", b"", b"", b""))


__all__ = ["build_cohort_waves"]
