from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from quantum_bench.model import ContractNode, TensorSpec, TensorView
from quantum_bench.numerics import EncodedComplexTensor
from quantum_bench.upmem.packed_wave import (
    WaveOperation,
    pack_wave_envelope,
    unpack_wave_envelope,
)
from quantum_bench.upmem.plan import UpmemStage, UpmemWorkUnit
from quantum_bench.upmem.tiling import (
    M5TileLimits,
    lower_binary_contraction,
)
from quantum_bench.upmem.wave_protocol import (
    FOUR_PRODUCT_PANEL,
    FOUR_PRODUCT_OUTER,
    IDLE,
    REAL_PANEL,
    REAL_OUTER,
    product_layout,
)
from quantum_bench.upmem.wave_work import build_cohort_waves


FLOAT_POLICY = "split_complex_float32_v1"
INT8_POLICY = "complex_int8_shared_scale_v1"


@pytest.mark.parametrize("numeric_mode", [0, 1])
@pytest.mark.parametrize("fuse", [False, True])
def test_outer_dispatch_changes_only_selector_for_existing_k1_work(numeric_mode, fuse):
    lowering = _lower(_node("outer", 3, 5, 1), numeric_mode)
    stage = UpmemStage(stage_id="outer", kind="contract_batch", node_ids=("outer",),
                       work_units=(_unit("outer", lowering.tiles[0], wave=0, dpu=0),))
    args = dict(dpu_count=3, tasklets=7, numeric_mode=numeric_mode, request_start=17, fuse=fuse)
    panels, panel_lanes = build_cohort_waves(stage, {"outer": lowering},
                                           {"outer": _operands(lowering, numeric_mode)}, **args)
    outers, outer_lanes = build_cohort_waves(stage, {"outer": lowering},
                                           {"outer": _operands(lowering, numeric_mode)},
                                           geometry_policy="outer_k1_v1", **args)
    assert panel_lanes == outer_lanes
    for panel_wave, outer_wave in zip(panels, outers, strict=True):
        assert panel_wave[1:] == outer_wave[1:]
        panel, outer = panel_wave[0], outer_wave[0]
        assert outer.control.kernel == (FOUR_PRODUCT_OUTER if fuse else REAL_OUTER)
        assert replace(outer.control, kernel=panel.control.kernel) == panel.control
        assert outer.inputs == panel.inputs
    with pytest.raises(ValueError, match="geometry kernel policy"):
        build_cohort_waves(stage, {"outer": lowering}, {"outer": _operands(lowering, numeric_mode)},
                           geometry_policy="unknown", **args)


def _node(node_id: str, m: int, n: int, k: int) -> ContractNode:
    return ContractNode(
        node_id=node_id,
        left=TensorView(
            tensor_id=f"{node_id}:left",
            labels=(0, 1),
            shape=(m, k),
        ),
        right=TensorView(
            tensor_id=f"{node_id}:right",
            labels=(1, 2),
            shape=(k, n),
        ),
        output=TensorSpec(
            id=f"{node_id}:output",
            labels=(0, 2),
            shape=(m, n),
            structure="dense",
        ),
        contracted_labels=(1,),
        output_labels=(0, 2),
    )


def _lower(node: ContractNode, numeric_mode: int):
    limits = (
        M5TileLimits.float32()
        if numeric_mode == 0
        else M5TileLimits.host_packed_int8()
    )
    left = np.ones(node.left.shape, dtype=np.float32)
    right = np.ones(node.right.shape, dtype=np.float32)
    return lower_binary_contraction(node, left, right, limits=limits)


def _encoded(shape: tuple[int, ...], numeric_mode: int, offset: int) -> EncodedComplexTensor:
    size = int(np.prod(shape))
    if numeric_mode == 0:
        values = np.arange(offset, offset + size, dtype=np.float32).reshape(shape)
        return EncodedComplexTensor(
            real=values,
            imag=values + np.float32(100.0),
            scale=1.0,
            saturation_real=0,
            saturation_imag=0,
        )
    values = (
        np.arange(offset, offset + size, dtype=np.int64) % 17 - 8
    ).astype(np.int8).reshape(shape)
    return EncodedComplexTensor(
        real=values,
        imag=np.asarray(values + 1, dtype=np.int8),
        scale=1.5,
        saturation_real=0,
        saturation_imag=0,
    )


def _operands(lowering, numeric_mode: int, offset: int = 0):
    return (
        _encoded(tuple(lowering.canonical.left.shape), numeric_mode, offset),
        _encoded(tuple(lowering.canonical.right.shape), numeric_mode, offset + 17),
    )


def _unit(
    node_id: str,
    tile,
    *,
    wave: int,
    dpu: int,
    rank: int = 0,
) -> UpmemWorkUnit:
    return UpmemWorkUnit(
        node_id=node_id,
        stable_tile_id=f"{node_id}:{tile.id}",
        wave=wave,
        logical_rank=rank,
        logical_dpu=dpu,
        batch_start=tile.batch_index,
        batch_size=1,
        m_start=tile.m_start,
        m_size=tile.m_size,
        n_start=tile.n_start,
        n_size=tile.n_size,
        k_start=tile.k_start,
        k_size=tile.k_size,
        estimated_input_bytes=tile.left_bytes + tile.right_bytes,
        estimated_output_bytes=tile.output_bytes,
        aligned_mram_bytes=tile.aligned_mram_bytes,
        estimated_arithmetic_work=tile.m_size * tile.n_size * tile.k_size,
    )


def _stage(lowerings, placements, node_ids=None) -> UpmemStage:
    node_ids = tuple(lowerings) if node_ids is None else tuple(node_ids)
    units = tuple(
        _unit(node_id, lowerings[node_id].tiles[tile_index], wave=wave, dpu=dpu)
        for node_id, tile_index, wave, dpu in placements
    )
    return UpmemStage(
        stage_id="dag_cohort:0:contract_batch",
        kind="contract_batch",
        node_ids=node_ids,
        work_units=units,
    )


def _operations(stage: UpmemStage, lowerings, operands, numeric_mode: int):
    records = []
    for node_id in stage.node_ids:
        canonical = lowerings[node_id].canonical
        left, right = operands[node_id]
        records.append(
            WaveOperation(
                hashlib.sha256((node_id + ":node").encode()).digest(),
                hashlib.sha256((node_id + ":contract").encode()).digest(),
                canonical.b,
                canonical.m,
                canonical.n,
                canonical.k,
                left.scale,
                right.scale,
            )
        )
    return tuple(records)


def _pack(stage, lowerings, operands, numeric_mode, waves):
    return pack_wave_envelope(
        plan_sha256=b"p" * 32,
        dpu_binary_sha256=b"b" * 32,
        sequence=7,
        operations=_operations(stage, lowerings, operands, numeric_mode),
        waves=waves,
        numeric_mode=numeric_mode,
    )


@pytest.mark.parametrize("numeric_mode", [0, 1])
def test_mixed_fusion_and_nonfit_preserves_geometry_and_codec(numeric_mode: int) -> None:
    fit_node = _node("fit", 3, 5, 7)
    wide_dim = 160 if numeric_mode == 0 else 256
    wide_node = _node("wide", wide_dim, wide_dim, wide_dim)
    lowerings = {
        "fit": _lower(fit_node, numeric_mode),
        "wide": _lower(wide_node, numeric_mode),
    }
    operands = {
        "fit": _operands(lowerings["fit"], numeric_mode),
        "wide": _operands(lowerings["wide"], numeric_mode, 31),
    }
    stage = _stage(
        lowerings,
        (("fit", 0, 0, 0), ("wide", 0, 0, 1)),
        node_ids=("fit", "wide"),
    )

    waves, generic_lanes = build_cohort_waves(
        stage,
        lowerings,
        operands,
        dpu_count=2,
        tasklets=8,
        numeric_mode=numeric_mode,
        request_start=100,
        fuse=True,
    )

    assert len(waves) == 4
    assert generic_lanes == (0, 1, 2, 3)
    for wave_index, wave in enumerate(waves):
        assert tuple(tile.control.dpu_id for tile in wave) == (0, 1)
        assert tuple(tile.control.wave_id for tile in wave) == (wave_index, wave_index)
        assert tuple(
            tile.control.request_sequence for tile in wave
        ) == (100 + wave_index, 100 + wave_index)
    assert waves[0][0].control.kernel == FOUR_PRODUCT_PANEL
    assert waves[0][0].control.tile_id == 0
    assert waves[0][1].control.kernel == REAL_PANEL
    assert waves[0][1].control.tile_id == 1
    for wave in waves[1:]:
        assert wave[0].control.flags == IDLE
        assert wave[1].control.kernel == REAL_PANEL
        assert wave[1].control.tile_id == 1
        assert wave[1].control.operation_index == 1

    fit_layout = product_layout(3, 5, 7, numeric_mode=numeric_mode, kernel=FOUR_PRODUCT_PANEL)
    assert waves[0][0].control.planes == fit_layout
    assert all(
        len(payload) == length
        for payload, (_, length) in zip(waves[0][0].inputs, fit_layout[:4], strict=True)
    )
    for lane, wave in enumerate(waves):
        wide_tile = wave[1]
        wide_layout = product_layout(
            wide_dim,
            wide_dim,
            wide_dim,
            numeric_mode=numeric_mode,
            kernel=REAL_PANEL,
        )
        assert wide_tile.control.planes == wide_layout
        assert wide_tile.inputs[1] == b""
        assert wide_tile.inputs[3] == b""
        left, right = operands["wide"]
        left_plane, right_plane = (
            (left.real, right.real),
            (left.imag, right.imag),
            (left.real, right.imag),
            (left.imag, right.real),
        )[lane]
        assert wide_tile.inputs[0] == left_plane[0].tobytes(order="C")
        assert wide_tile.inputs[2] == right_plane[0].tobytes(order="C")

    packed = _pack(stage, lowerings, operands, numeric_mode, waves)
    operations, decoded_waves = unpack_wave_envelope(packed)
    assert len(operations) == 2
    assert decoded_waves == waves


@pytest.mark.parametrize("numeric_mode", [0, 1])
def test_fused_tail_has_aligned_zero_padding_and_idle_slots(numeric_mode: int) -> None:
    node = _node("tail", 3, 5, 7)
    lowering = _lower(node, numeric_mode)
    operands = {"tail": _operands(lowering, numeric_mode)}
    stage = _stage({"tail": lowering}, (("tail", 0, 0, 1),))

    waves, generic_lanes = build_cohort_waves(
        stage,
        {"tail": lowering},
        operands,
        dpu_count=2,
        tasklets=3,
        numeric_mode=numeric_mode,
        request_start=11,
        fuse=True,
    )

    assert len(waves) == 1
    assert generic_lanes == (0,)
    idle, active = waves[0]
    assert idle.control.flags == IDLE
    assert active.control.kernel == FOUR_PRODUCT_PANEL
    layout = product_layout(3, 5, 7, numeric_mode=numeric_mode, kernel=FOUR_PRODUCT_PANEL)
    left, right = operands["tail"]
    expected = (
        left.real[0].tobytes(order="C"),
        left.imag[0].tobytes(order="C"),
        right.real[0].tobytes(order="C"),
        right.imag[0].tobytes(order="C"),
    )
    for payload, raw, (_, length) in zip(active.inputs, expected, layout[:4], strict=True):
        assert payload == raw + b"\0" * (length - len(raw))
        assert payload[-(length - len(raw)) :] == b"\0" * (length - len(raw))
    assert all(length for _, length in active.control.planes[4:])
    _pack(stage, {"tail": lowering}, operands, numeric_mode, waves)


def test_real_panel_generic_lane_ordering_is_explicit() -> None:
    node = _node("generic", 3, 5, 7)
    lowering = _lower(node, 0)
    operands = {"generic": _operands(lowering, 0)}
    stage = _stage({"generic": lowering}, (("generic", 0, 0, 0),))

    waves, generic_lanes = build_cohort_waves(
        stage,
        {"generic": lowering},
        operands,
        dpu_count=1,
        tasklets=1,
        numeric_mode=0,
        request_start=20,
        fuse=False,
    )

    assert len(waves) == 4
    assert generic_lanes == (0, 1, 2, 3)
    left, right = operands["generic"]
    expected_pairs = (
        (left.real, right.real),
        (left.imag, right.imag),
        (left.real, right.imag),
        (left.imag, right.real),
    )
    for lane, wave in enumerate(waves):
        tile = wave[0]
        assert tile.control.kernel == REAL_PANEL
        assert tile.control.wave_id == lane
        assert tile.control.request_sequence == 20 + lane
        expected_left, expected_right = expected_pairs[lane]
        assert tile.inputs == (
            expected_left[0].tobytes(order="C") + b"\0" * 4,
            b"",
            expected_right[0].tobytes(order="C") + b"\0" * 4,
            b"",
        )


def test_rejects_coverage_estimate_and_geometry_corruption() -> None:
    node = _node("split", 257, 2, 2)
    lowering = _lower(node, 0)
    operands = {"split": _operands(lowering, 0)}
    stage = _stage(
        {"split": lowering},
        (("split", 0, 0, 0), ("split", 1, 1, 0)),
    )
    kwargs = dict(
        lowerings={"split": lowering},
        operands=operands,
        dpu_count=1,
        tasklets=1,
        numeric_mode=0,
        request_start=0,
        fuse=True,
    )

    with pytest.raises(ValueError, match="set differs"):
        build_cohort_waves(replace(stage, work_units=stage.work_units[:1]), **kwargs)
    with pytest.raises(ValueError, match="extents differ"):
        build_cohort_waves(
            replace(
                stage,
                work_units=(
                    replace(
                        stage.work_units[0],
                        estimated_output_bytes=stage.work_units[0].estimated_output_bytes + 1,
                    ),
                    stage.work_units[1],
                ),
            ),
            **kwargs,
        )
    with pytest.raises(ValueError, match="extents differ"):
        build_cohort_waves(
            replace(
                stage,
                work_units=(
                    replace(stage.work_units[0], m_start=1),
                    stage.work_units[1],
                ),
            ),
            **kwargs,
        )


def test_rejects_dense_wave_gaps_duplicate_slots_and_nonzero_rank() -> None:
    node = _node("invalid", 257, 2, 2)
    lowering = _lower(node, 0)
    operands = {"invalid": _operands(lowering, 0)}
    base = _stage(
        {"invalid": lowering},
        (("invalid", 0, 0, 0), ("invalid", 1, 1, 0)),
    )
    kwargs = dict(
        lowerings={"invalid": lowering},
        operands=operands,
        dpu_count=1,
        tasklets=1,
        numeric_mode=0,
        request_start=0,
        fuse=True,
    )
    with pytest.raises(ValueError, match="dense"):
        build_cohort_waves(
            replace(base, work_units=(base.work_units[0], replace(base.work_units[1], wave=2))),
            **kwargs,
        )
    with pytest.raises(ValueError, match="reuses a DPU"):
        build_cohort_waves(
            replace(base, work_units=(base.work_units[0], replace(base.work_units[1], wave=0))),
            **kwargs,
        )
    with pytest.raises(ValueError, match="rank-zero"):
        build_cohort_waves(
            replace(base, work_units=(replace(base.work_units[0], logical_rank=1), base.work_units[1])),
            **kwargs,
        )


def test_freezes_dpu_ownership_across_mixed_node_waves() -> None:
    first = _node("first", 257, 2, 2)
    second = _node("second", 257, 2, 2)
    lowerings = {"first": _lower(first, 0), "second": _lower(second, 0)}
    operands = {
        node_id: _operands(lowering, 0)
        for node_id, lowering in lowerings.items()
    }
    stage = _stage(
        lowerings,
        (
            ("first", 0, 0, 0),
            ("second", 0, 0, 1),
            ("first", 1, 1, 2),
            ("second", 1, 1, 0),
        ),
        node_ids=("first", "second"),
    )
    with pytest.raises(ValueError, match="ownership"):
        build_cohort_waves(
            stage,
            lowerings,
            operands,
            dpu_count=3,
            tasklets=1,
            numeric_mode=0,
            request_start=0,
            fuse=True,
        )


def test_rejects_operand_shape_dtype_and_lowering_policy_mismatch() -> None:
    node = _node("metadata", 3, 5, 7)
    float_lowering = _lower(node, 0)
    int8_lowering = _lower(node, 1)
    stage = _stage({"metadata": float_lowering}, (("metadata", 0, 0, 0),))
    good_float = _operands(float_lowering, 0)
    wrong_shape = (
        _encoded(node.left.shape, 0, 0),
        good_float[1],
    )
    with pytest.raises(ValueError, match="canonical shape"):
        build_cohort_waves(
            stage,
            {"metadata": float_lowering},
            {"metadata": wrong_shape},
            dpu_count=1,
            tasklets=1,
            numeric_mode=0,
            request_start=0,
            fuse=True,
        )
    int8_operands = _operands(int8_lowering, 1)
    with pytest.raises(ValueError, match="numeric policy"):
        build_cohort_waves(
            stage,
            {"metadata": float_lowering},
            {"metadata": int8_operands},
            dpu_count=1,
            tasklets=1,
            numeric_mode=0,
            request_start=0,
            fuse=True,
        )
    with pytest.raises(ValueError, match="numeric policy"):
        build_cohort_waves(
            _stage({"metadata": int8_lowering}, (("metadata", 0, 0, 0),)),
            {"metadata": int8_lowering},
            {"metadata": int8_operands},
            dpu_count=1,
            tasklets=1,
            numeric_mode=0,
            request_start=0,
            fuse=True,
        )
