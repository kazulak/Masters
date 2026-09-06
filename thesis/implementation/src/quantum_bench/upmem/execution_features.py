"""Schedule-aware accounting for one-rank prepared UPMEM waves.

The records in this module are plan facts, not measurements.  Control expansion
is delegated to ``wave_work.build_cohort_controls`` so fused admission, generic
lanes, idle slots, and kernel selection have one implementation shared with the
prepared-wave runtime.  No tensors, operand payloads, or execution sessions are
created here.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import prod

import numpy as np

from quantum_bench.model import ContractNode, ContractionDAG
from quantum_bench.upmem.packed_wave import wave_snapshot_sizes
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemStage,
    UpmemWorkUnit,
    physical_plan_id,
    validate_upmem_plan,
)
from quantum_bench.upmem.protocol import (
    WRAM_PANEL_DMA_BYTES,
    WRAM_PANEL_KC,
    WRAM_PANEL_NC,
    WRAM_PANEL_UNALIGNED_SCRATCH_BYTES,
)
from quantum_bench.upmem.wave_protocol import (
    COMPLETION,
    CONTROL,
    FOUR_PRODUCT_KERNELS,
    IDLE,
    OUTER_KERNELS,
    REAL_KERNELS,
    WaveControl,
)
from quantum_bench.upmem.wave_work import build_cohort_controls
from quantum_bench.upmem.tiling import canonical_label_geometry


SCHEMA_VERSION = "upmem_execution_features_v1"
FACTS_SCOPE = "schedule_aware_prepared_wave_one_rank_v1"
_GEOMETRY_POLICIES = frozenset(("panel_only_v1", "outer_k1_v1"))
_NUMERIC_MODES = {
    "split_complex_float32_v1": 0,
    "complex_int8_shared_scale_v1": 1,
}
_KERNEL_NAMES = {
    0: "idle",
    1: "real_panel",
    2: "four_product_panel",
    3: "real_outer",
    4: "four_product_outer",
}
_LOCAL_FIELDS = (
    "mram_read_helper_calls_exact",
    "mram_write_helper_calls_exact",
    "mram_unaligned_read_helper_calls_exact",
    "mram_unaligned_write_helper_calls_exact",
    "mram_requested_payload_bytes_exact",
    "mram_aligned_transfer_bytes_estimate",
)
_DISPATCH_SOURCE = "native/upmem/runtime/dpu_wave.c"
_PANEL_SOURCE = "native/upmem/runtime/panel_compute.h"
_OUTER_SOURCE = "native/upmem/runtime/outer_compute.h"


def extract_execution_features(
    dag: ContractionDAG,
    plan: UpmemPlan,
    *,
    fuse_complex: bool = False,
    geometry_policy: str = "panel_only_v1",
) -> dict[str, object]:
    """Extract deterministic boundary, work, and source-level wave facts.

    ``real_mac_count`` is the four-real-product arithmetic count.  Generic
    controls contribute one product in each of four lane waves; fused controls
    contribute four products in lane zero.  ``wave_critical_real_mac_sum`` is
    the sum of the maximum slot work in each emitted micro-wave, preserving the
    schedule instead of summing independent node timings.
    """

    validate_upmem_plan(dag, plan)
    _validate_options(plan, fuse_complex=fuse_complex, geometry_policy=geometry_policy)
    topology = plan.topology
    numeric_mode = _NUMERIC_MODES[plan.numeric_policy]

    waves: list[dict[str, object]] = []
    totals = _new_totals()
    static_peak_mram = 0
    cohort_count = 0
    cohort_snapshots: list[dict[str, object]] = []

    for declared_stage in plan.stages:
        if declared_stage.kind == "host_reduce":
            totals["host_reduce_count"] += 1
            continue
        for cohort_stage, declared_stage_id, cohort_kind in _cohort_stages(
            declared_stage, plan.schedule_policy
        ):
            cohort_count += 1
            controls, generic_lanes = build_cohort_controls(
                cohort_stage,
                dpu_count=topology.dpu_count,
                tasklets=topology.tasklets_per_dpu,
                numeric_mode=numeric_mode,
                request_start=0,
                fuse=fuse_complex,
                geometry_policy=geometry_policy,
            )
            envelope_bytes, result_bytes = wave_snapshot_sizes(len(cohort_stage.node_ids), controls)
            cohort_snapshots.append({
                "cohort_id": cohort_stage.stage_id,
                "node_ids": cohort_stage.node_ids,
                "input_envelope_bytes": envelope_bytes,
                "response_snapshot_bytes": result_bytes,
                "control_count": len(controls) * topology.dpu_count,
            })
            for cohort_wave_index, (control_wave, lane) in enumerate(
                zip(controls, generic_lanes, strict=True)
            ):
                wave = _make_wave(
                    cohort_stage,
                    declared_stage_id=declared_stage_id,
                    cohort_kind=cohort_kind,
                    cohort_wave_index=cohort_wave_index,
                    global_wave_id=len(waves),
                    control_wave=control_wave,
                    lane=lane,
                    numeric_mode=numeric_mode,
                    tasklets=topology.tasklets_per_dpu,
                )
                waves.append(wave)
                static_peak_mram = max(static_peak_mram, int(wave["mram_bytes"]))
                _add_wave_totals(totals, wave)

    totals["cohort_count"] = cohort_count
    static_memory = _static_memory(topology.tasklets_per_dpu, static_peak_mram)

    return {
        "schema_version": SCHEMA_VERSION,
        "facts_scope": FACTS_SCOPE,
        "claims": {
            "physical_timing": False,
            "physical_profile_adoption": False,
            "full_host_memory_bound": False,
            "numeric_overhead": "not_estimated",
            "local_traffic": "source_level_geometric_estimate_only",
            "barrier_events": "source_level_barrier_wait_call_count",
            "barrier_formula": {
                "dpu_wave_common_per_slot": 3,
                "panel_compute_per_product": "2 * ceil(n/32) * ceil(k/64)",
                "outer_compute_per_product": 2,
                "idle_slot": 3,
                "all_tasklets_participate": True,
            },
        },
        "plan": {
            "logical_plan_id": plan.logical_plan_id,
            "physical_plan_id": physical_plan_id(plan),
            "schedule_policy": plan.schedule_policy,
            "numeric_policy": plan.numeric_policy,
            "numeric_mode": numeric_mode,
            "geometry_policy": geometry_policy,
            "fuse_complex": fuse_complex,
            "request_transport": "packed_wave_v1",
            "kernel_identity": "dpu_panel_dispatch_v5_v1",
            "rank_count": topology.rank_count,
            "dpu_count": topology.dpu_count,
            "tasklets_per_dpu": topology.tasklets_per_dpu,
        },
        "waves": waves,
        "totals": totals,
        "static_memory": static_memory,
        "host_buffers": _retained_host_buffers(dag, numeric_mode, cohort_snapshots),
    }


def _retained_host_buffers(dag, numeric_mode, cohorts):
    """Count bulk allocations retained until post-steady evidence construction.

    Runtime keeps every produced tensor, encoded operand and raw lane view.
    Each raw view pins its entire immutable cohort response, including idle
    completion records. These are not peak RSS or temporary workspace bounds.
    """
    input_bytes = sum(prod(t.shape) * np.dtype(t.dtype).itemsize for t in dag.tensors)
    output_bytes = sum(prod(node.output.shape) * 8 for node in dag.nodes)
    encoded_bytes = 0
    for node in dag.nodes:
        if isinstance(node, ContractNode):
            b, m, k, n = canonical_label_geometry(
                node.left.labels, node.left.shape, node.right.labels,
                node.right.shape, node.output_labels)
            encoded_bytes += 2 * (1 if numeric_mode else 4) * b * k * (m + n)
    response_bytes = sum(row["response_snapshot_bytes"] for row in cohorts)
    final_bytes = prod(dag.output.shape) * 8
    retained = output_bytes + encoded_bytes + response_bytes + final_bytes
    return {
        "scope": "post_steady_retained_bulk_allocations_v1",
        "full_host_memory_bound": False,
        "peak_rss_measured": False,
        "caller_input_declared_bytes": input_bytes,
        "caller_input_alias_storage_deduplicated": False,
        "graph_output_array_bytes": output_bytes,
        "encoded_operand_plane_bytes": encoded_bytes,
        "retained_response_snapshot_bytes": response_bytes,
        "final_output_copy_bytes": final_bytes,
        "retained_executor_bulk_bytes": retained,
        "max_input_envelope_bytes": max((row["input_envelope_bytes"] for row in cohorts), default=0),
        "max_response_snapshot_bytes": max((row["response_snapshot_bytes"] for row in cohorts), default=0),
        "cohorts": cohorts,
        "excluded_from_retained_sum": [
            "caller input backing allocations",
            "materialization/canonicalization/quantization workspace",
            "live cohort input payloads and envelope assembly copies",
            "lane assembly, reduction and decode workspace",
            "native envelope snapshot and output scratch",
            "evidence hashing and serialization temporaries",
            "Python objects, allocator overhead, SDK and process storage",
        ],
        "lifetime_sources": [
            "runtime.py:UpmemSession._run_once_unlocked",
            "runtime.py:UpmemV4Session._execute_cohort",
            "native_session.py:V4Session._read_wave_snapshot",
        ],
    }


def _validate_options(
    plan: UpmemPlan, *, fuse_complex: bool, geometry_policy: str
) -> None:
    if type(fuse_complex) is not bool:
        raise TypeError("fuse_complex must be a bool")
    if geometry_policy not in _GEOMETRY_POLICIES:
        raise ValueError("unsupported geometry kernel policy")
    if plan.topology.rank_count != 1:
        raise ValueError(
            "schedule-aware prepared-wave accounting currently requires exactly one rank"
        )
    if any(
        unit.logical_rank != 0
        for stage in plan.stages
        for unit in stage.work_units
    ):
        raise ValueError("prepared-wave accounting requires rank-zero work units")


def _cohort_stages(
    stage: UpmemStage, schedule_policy: str
) -> Iterable[tuple[UpmemStage, str, str]]:
    """Yield effective runtime cohorts without changing the physical plan."""

    if schedule_policy == "static_dag_waves_v1":
        yield stage, stage.stage_id, "static_stage"
        return

    # This mirrors runtime._session_stage_nodes for serial prepared waves.
    for node_id in stage.node_ids:
        cohort_id = stage.stage_id
        if len(stage.node_ids) > 1:
            cohort_id = f"{stage.stage_id}:node:{node_id}"
        yield (
            UpmemStage(
                stage_id=cohort_id,
                kind="contract_batch",
                node_ids=(node_id,),
                work_units=tuple(
                    unit for unit in stage.work_units if unit.node_id == node_id
                ),
            ),
            stage.stage_id,
            "serial_node",
        )


def _make_wave(
    stage: UpmemStage,
    *,
    declared_stage_id: str,
    cohort_kind: str,
    cohort_wave_index: int,
    global_wave_id: int,
    control_wave: tuple[WaveControl, ...],
    lane: int,
    numeric_mode: int,
    tasklets: int,
) -> dict[str, object]:
    units = stage.work_units
    active_controls = [control for control in control_wave if control.flags != IDLE]
    original_waves = {units[control.tile_id].wave for control in active_controls}
    if len(original_waves) != 1:
        raise ValueError("control wave contains work from multiple original waves")
    original_wave = next(iter(original_waves))

    slots: list[dict[str, object]] = []
    for control in control_wave:
        if control.flags == IDLE:
            slots.append(_idle_slot(control.dpu_id, tasklets))
            continue
        unit = units[control.tile_id]
        if unit.logical_dpu != control.dpu_id:
            raise ValueError("control DPU does not match the scheduled work unit")
        slots.append(
            _active_slot(
                control,
                unit,
                numeric_mode=numeric_mode,
                tasklets=tasklets,
                lane=lane,
            )
        )

    kernel_counts = {name: 0 for name in _KERNEL_NAMES.values()}
    for slot in slots:
        kernel_counts[slot["kernel_name"]] += 1
    local = _aggregate_local(slot["local_traffic"] for slot in slots)
    dpu_work = {
        int(slot["dpu_id"]): int(slot["real_mac_count"]) for slot in slots
    }
    wave_critical = max(dpu_work.values(), default=0)
    return {
        "cohort_id": stage.stage_id,
        "declared_stage_id": declared_stage_id,
        "cohort_kind": cohort_kind,
        "node_ids": list(stage.node_ids),
        "wave_id": cohort_wave_index,
        "global_wave_id": global_wave_id,
        "original_wave": original_wave,
        "lane": lane,
        "slots": slots,
        "kernel_counts": kernel_counts,
        "active_slot_launch_count": sum(bool(slot["active"]) for slot in slots),
        "idle_slot_launch_count": sum(not slot["active"] for slot in slots),
        "product_count": sum(int(slot["product_count"]) for slot in slots),
        "real_mac_count": sum(int(slot["real_mac_count"]) for slot in slots),
        "wave_critical_real_mac_count": wave_critical,
        "control_bytes": sum(int(slot["control_bytes"]) for slot in slots),
        "completion_bytes": sum(int(slot["completion_bytes"]) for slot in slots),
        "input_payload_bytes": sum(int(slot["input_payload_bytes"]) for slot in slots),
        "output_payload_bytes": sum(int(slot["output_payload_bytes"]) for slot in slots),
        "h2d_bytes": sum(int(slot["h2d_bytes"]) for slot in slots),
        "d2h_bytes": sum(int(slot["d2h_bytes"]) for slot in slots),
        "barrier_events": sum(int(slot["barrier_events"]) for slot in slots),
        "barrier_tasklet_calls": sum(
            int(slot["barrier_tasklet_calls"]) for slot in slots
        ),
        "mram_bytes": max(
            (int(slot["mram_bytes"]) for slot in slots),
            default=0,
        ),
        "local_traffic": local,
    }


def _idle_slot(dpu_id: int, tasklets: int) -> dict[str, object]:
    return {
        "dpu_id": dpu_id,
        "active": False,
        "node_id": None,
        "stable_tile_id": None,
        "tile_id": None,
        "kernel": 0,
        "kernel_name": "idle",
        "m": 0,
        "n": 0,
        "k": 0,
        "product_count": 0,
        "real_mac_count": 0,
        "planes": [[0, 0] for _ in range(8)],
        "mram_bytes": 0,
        "control_bytes": CONTROL.size,
        "completion_bytes": COMPLETION.size,
        "input_payload_bytes": 0,
        "output_payload_bytes": 0,
        "h2d_bytes": CONTROL.size,
        "d2h_bytes": COMPLETION.size,
        "barrier_events": 3,
        "barrier_tasklet_calls": 3 * tasklets,
        "local_traffic": _empty_local("not_applicable"),
    }


def _active_slot(
    control: WaveControl,
    unit: UpmemWorkUnit,
    *,
    numeric_mode: int,
    tasklets: int,
    lane: int,
) -> dict[str, object]:
    product_count = 4 if control.kernel in FOUR_PRODUCT_KERNELS else 1
    input_payload = sum(length for _, length in control.planes[:4])
    output_payload = sum(length for _, length in control.planes[4:])
    local = _local_traffic(
        unit,
        numeric_mode=numeric_mode,
        kernel=control.kernel,
        product_count=product_count,
    )
    barriers = _barrier_events(
        control.kernel,
        product_count,
        n=control.n,
        k=control.k,
    )
    return {
        "dpu_id": control.dpu_id,
        "active": True,
        "node_id": unit.node_id,
        "stable_tile_id": unit.stable_tile_id,
        "tile_id": control.tile_id,
        "kernel": control.kernel,
        "kernel_name": _KERNEL_NAMES[control.kernel],
        "m": control.m,
        "n": control.n,
        "k": control.k,
        "batch_index": control.batch_index,
        "m_start": unit.m_start,
        "n_start": unit.n_start,
        "k_start": unit.k_start,
        "k_offset": control.k_offset,
        "product_count": product_count,
        "real_mac_count": product_count * control.m * control.n * control.k,
        "planes": [list(span) for span in control.planes],
        "mram_bytes": max(
            (offset + length for offset, length in control.planes), default=0
        ),
        "control_bytes": CONTROL.size,
        "completion_bytes": COMPLETION.size,
        "input_payload_bytes": input_payload,
        "output_payload_bytes": output_payload,
        # H2D/D2H include control/completion and aligned product spans, exactly
        # as submit_waves reports them.
        "h2d_bytes": CONTROL.size + input_payload,
        "d2h_bytes": COMPLETION.size + output_payload,
        "barrier_events": barriers,
        "barrier_tasklet_calls": barriers * tasklets,
        "lane": lane,
        "local_traffic": local,
    }


def _new_totals() -> dict[str, object]:
    return {
        "cohort_count": 0,
        "host_reduce_count": 0,
        "original_wave_count": 0,
        "launch_count": 0,
        "active_slot_launch_count": 0,
        "idle_slot_launch_count": 0,
        "product_count": 0,
        "real_mac_count": 0,
        "wave_critical_real_mac_sum": 0,
        "control_bytes": 0,
        "completion_bytes": 0,
        "input_payload_bytes": 0,
        "output_payload_bytes": 0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "barrier_events": 0,
        "barrier_tasklet_calls": 0,
        "kernel_counts": {name: 0 for name in _KERNEL_NAMES.values()},
        "local_traffic": _empty_local("not_applicable"),
    }


def _add_wave_totals(total: dict[str, object], wave: dict[str, object]) -> None:
    total["original_wave_count"] += int(wave["lane"] == 0)
    total["launch_count"] += 1
    for field in (
        "active_slot_launch_count",
        "idle_slot_launch_count",
        "product_count",
        "real_mac_count",
        "control_bytes",
        "completion_bytes",
        "input_payload_bytes",
        "output_payload_bytes",
        "h2d_bytes",
        "d2h_bytes",
        "barrier_events",
        "barrier_tasklet_calls",
    ):
        total[field] += int(wave[field])
    total["wave_critical_real_mac_sum"] += int(
        wave["wave_critical_real_mac_count"]
    )
    for name, count in wave["kernel_counts"].items():
        total["kernel_counts"][name] += int(count)
    total["local_traffic"] = _aggregate_local(
        (total["local_traffic"], wave["local_traffic"])
    )


def _static_memory(tasklets: int, peak_mram: int) -> dict[str, object]:
    shared = WRAM_PANEL_KC * WRAM_PANEL_NC * 4
    private = WRAM_PANEL_KC * 4 + WRAM_PANEL_NC * 4 + WRAM_PANEL_UNALIGNED_SCRATCH_BYTES
    return {
        "static_peak_mram_bytes": peak_mram,
        "known_wram_shared_bytes": shared,
        "known_wram_private_bytes_per_tasklet": private,
        "known_wram_tasklet_count": tasklets,
        "known_wram_buffers_bytes": shared + tasklets * private,
        "scope": "per_dpu_static_kernel_buffers_and_per_dpu_mram_arena_v1",
        "full_host_memory_bound": False,
        "linked_max_footprint": False,
    }


def _barrier_events(kernel: int, product_count: int, *, n: int, k: int) -> int:
    common = 3  # dpu_wave.c barriers, including idle controls.
    if kernel in OUTER_KERNELS:
        return common + 2 * product_count
    if kernel in (*REAL_KERNELS, *FOUR_PRODUCT_KERNELS):
        n_panels = (n + WRAM_PANEL_NC - 1) // WRAM_PANEL_NC
        k_panels = (k + WRAM_PANEL_KC - 1) // WRAM_PANEL_KC
        return common + 2 * product_count * n_panels * k_panels
    if kernel == 0:
        return common
    raise ValueError(f"unknown wave kernel: {kernel}")


def _local_traffic(
    unit: UpmemWorkUnit,
    *,
    numeric_mode: int,
    kernel: int,
    product_count: int,
) -> dict[str, object]:
    if kernel in OUTER_KERNELS:
        base = _outer_traffic(unit, numeric_mode=numeric_mode)
        algorithm = "outer_compute_v1"
        source = _OUTER_SOURCE
    else:
        base = _panel_traffic(unit, numeric_mode=numeric_mode)
        algorithm = "panel_compute_v1"
        source = _PANEL_SOURCE
    return {
        "status": "estimated",
        "algorithms": [algorithm],
        "source_files": [_DISPATCH_SOURCE, source],
        "scope": "source_level_helper_calls_and_geometric_aligned_spans",
        **{
            field: base[field] * product_count
            for field in _LOCAL_FIELDS
        },
    }


def _panel_traffic(unit: UpmemWorkUnit, *, numeric_mode: int) -> dict[str, int]:
    element_bytes = 1 if numeric_mode == 1 else 4
    result = _zero_local()
    a_bytes = _align8(unit.m_size * unit.k_size * element_bytes)
    b_offset = a_bytes
    b_bytes = _align8(unit.k_size * unit.n_size * element_bytes)
    c_offset = b_offset + b_bytes

    for n_start in range(0, unit.n_size, WRAM_PANEL_NC):
        actual_n = min(WRAM_PANEL_NC, unit.n_size - n_start)
        for k_start in range(0, unit.k_size, WRAM_PANEL_KC):
            actual_k = min(WRAM_PANEL_KC, unit.k_size - k_start)
            b_panel_bytes = actual_k * actual_n * element_bytes
            full_b_panel = (
                actual_k == WRAM_PANEL_KC
                and actual_n == WRAM_PANEL_NC
                and unit.n_size == WRAM_PANEL_NC
            )
            if full_b_panel:
                calls = (b_panel_bytes + WRAM_PANEL_DMA_BYTES - 1) // WRAM_PANEL_DMA_BYTES
                result["mram_read_helper_calls_exact"] += calls
                result["mram_requested_payload_bytes_exact"] += b_panel_bytes
                result["mram_aligned_transfer_bytes_estimate"] += b_panel_bytes
            else:
                for k_index in range(actual_k):
                    offset = b_offset + (
                        (k_start + k_index) * unit.n_size + n_start
                    ) * element_bytes
                    _record_read(result, offset, actual_n * element_bytes)

            for row in range(unit.m_size):
                a_offset = (row * unit.k_size + k_start) * element_bytes
                _record_read(result, a_offset, actual_k * element_bytes)
                c_offset_row = c_offset + (row * unit.n_size + n_start) * 4
                c_payload = actual_n * 4
                if k_start:
                    _record_read(result, c_offset_row, c_payload)
                _record_write(result, c_offset_row, c_payload)
    return result


def _outer_traffic(unit: UpmemWorkUnit, *, numeric_mode: int) -> dict[str, int]:
    element_bytes = 1 if numeric_mode == 1 else 4
    result = _zero_local()
    a_bytes = _align8(unit.m_size * element_bytes)
    b_bytes = _align8(unit.n_size * element_bytes)
    result["mram_read_helper_calls_exact"] = (a_bytes + 255) // 256 + (b_bytes + 255) // 256
    result["mram_requested_payload_bytes_exact"] = a_bytes + b_bytes
    result["mram_aligned_transfer_bytes_estimate"] = a_bytes + b_bytes

    c_offset = a_bytes + b_bytes
    output_bytes = unit.m_size * unit.n_size * 4
    for block_start in range(0, unit.m_size * unit.n_size, WRAM_PANEL_NC):
        block_count = min(WRAM_PANEL_NC, unit.m_size * unit.n_size - block_start)
        block_bytes = block_count * 4
        aligned = block_bytes & ~7
        if aligned:
            result["mram_write_helper_calls_exact"] += 1
            result["mram_requested_payload_bytes_exact"] += aligned
            result["mram_aligned_transfer_bytes_estimate"] += aligned
        tail = block_bytes - aligned
        if tail:
            result["mram_write_helper_calls_exact"] += 1
            result["mram_unaligned_write_helper_calls_exact"] += 1
            result["mram_requested_payload_bytes_exact"] += tail
            result["mram_aligned_transfer_bytes_estimate"] += _aligned_span(
                c_offset + block_start * 4 + aligned, tail
            )
    # Keep this assertion local and structural; it guards accidental omission
    # of the final output stream while retaining one public traffic total.
    if result["mram_requested_payload_bytes_exact"] != a_bytes + b_bytes + output_bytes:
        raise AssertionError("outer source traffic accounting lost output bytes")
    return result


def _record_read(result: dict[str, int], offset: int, payload: int) -> None:
    result["mram_read_helper_calls_exact"] += 1
    if not _dma_aligned(offset, payload):
        result["mram_unaligned_read_helper_calls_exact"] += 1
    result["mram_requested_payload_bytes_exact"] += payload
    result["mram_aligned_transfer_bytes_estimate"] += _aligned_span(offset, payload)


def _record_write(result: dict[str, int], offset: int, payload: int) -> None:
    result["mram_write_helper_calls_exact"] += 1
    if not _dma_aligned(offset, payload):
        result["mram_unaligned_write_helper_calls_exact"] += 1
    result["mram_requested_payload_bytes_exact"] += payload
    result["mram_aligned_transfer_bytes_estimate"] += _aligned_span(offset, payload)


def _aggregate_local(records: Iterable[object]) -> dict[str, object]:
    result = _zero_local()
    active = False
    sources: list[str] = []
    algorithms: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("local traffic records must be mappings")
        active |= record["status"] == "estimated"
        for field in _LOCAL_FIELDS:
            result[field] += int(record[field])
        for source in record["source_files"]:
            if source not in sources:
                sources.append(source)
        for algorithm in record["algorithms"]:
            if algorithm not in algorithms:
                algorithms.append(algorithm)
    return {
        "status": "estimated" if active else "not_applicable",
        "algorithms": algorithms,
        "source_files": sources,
        "scope": "source_level_helper_calls_and_geometric_aligned_spans",
        **result,
    }


def _empty_local(status: str) -> dict[str, object]:
    return {
        "status": status,
        "algorithms": [],
        "source_files": [],
        "scope": "source_level_helper_calls_and_geometric_aligned_spans",
        **_zero_local(),
    }


def _zero_local() -> dict[str, int]:
    return {field: 0 for field in _LOCAL_FIELDS}


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _aligned_span(offset: int, payload: int) -> int:
    return _align8((offset & 7) + payload)


def _dma_aligned(offset: int, payload: int) -> bool:
    return offset % 8 == 0 and payload % 8 == 0 and 8 <= payload <= 2048


__all__ = ["extract_execution_features"]
