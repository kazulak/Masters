from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import struct

import numpy as np
import pytest

from quantum_bench.circuits import builtin_circuit
from quantum_bench.targets.upmem.distributed_plan_v3 import (
    CONTRACTED_PARTIAL_SUM,
    DEFAULT_MAX_DPUS_PER_RANK,
    DEFAULT_TASKLETS_PER_DPU,
    DISTRIBUTED_PLAN_V3_SCHEMA_VERSION,
    NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE,
    NUMERIC_MODE_FLOAT32_REAL,
    OUTPUT_TILE,
    UPXDPV3_HEADER_BYTES,
    UPXDPV3_HEADER_FORMAT,
    UPXDPV3_MAGIC,
    UPXDPV3_NUMERIC_INT8_REQUANTIZE,
    UPXDPV3_RECORD_BYTES,
    UPXDPV3_RECORD_FORMAT,
    UPXDPV3_VERSION,
    build_contracted_partial_sum_plan_v3,
    build_output_tile_plan_v3,
    load_upxdpv3,
    parse_distributed_plan_v3,
    serialize_distributed_plan_v3,
    serialize_upxdpv3,
    UnsupportedPartitionError,
    validate_distributed_plan_v3,
)
from quantum_bench.targets.upmem.m5_task_selection import select_highest_work_supported_task
from quantum_bench.tn.network import build_tensor_network
from quantum_bench.tn.task_graph import plan_task_graph


PACKAGE = b"package-v3"
OPERATION = b"operation-v3"
PACKAGE_SHA256 = hashlib.sha256(PACKAGE).hexdigest()
OPERATION_SHA256 = hashlib.sha256(OPERATION).hexdigest()


def _output_plan(*, dpu_count: int = 3, **kwargs: object):
    options = {
        "logical_operation_id": "operation",
        "logical_task_id": "task",
        "total_output_elements": 120,
        "total_contracted_elements": 17,
        "package_sha256": PACKAGE_SHA256,
        "operation_sha256": OPERATION_SHA256,
        "dpu_count": dpu_count,
        **kwargs,
    }
    return build_output_tile_plan_v3(**options)


def _contracted_plan(*, dpu_count: int = 3, **kwargs: object):
    return build_contracted_partial_sum_plan_v3(
        logical_operation_id="operation",
        logical_task_id="task",
        total_output_elements=11,
        total_contracted_elements=29,
        package_sha256=PACKAGE_SHA256,
        operation_sha256=OPERATION_SHA256,
        dpu_count=dpu_count,
        **kwargs,
    )


@pytest.mark.parametrize("dpu_count", (3, 5, 12))
def test_arbitrary_dpu_counts_have_exact_remainder_coverage(dpu_count: int) -> None:
    plan = _output_plan(dpu_count=dpu_count, tasklets_per_dpu=3)
    ranges = [(unit.output_offset, unit.output_elements) for unit in plan.work_units]

    assert plan.schema_version == DISTRIBUTED_PLAN_V3_SCHEMA_VERSION
    assert plan.max_dpus_per_rank == DEFAULT_MAX_DPUS_PER_RANK
    assert plan.tasklets_per_dpu == 3
    assert [unit.dpu_id for unit in plan.work_units] == list(range(dpu_count))
    assert ranges[0][0] == 0
    assert all(left[0] + left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert ranges[-1][0] + ranges[-1][1] == 120
    assert all(offset % 2 == 0 for offset, _ in ranges)
    assert all(size % 2 == 0 for _, size in ranges[:-1])
    assert max(size for _, size in ranges) - min(size for _, size in ranges) <= 2


def test_default_tasklets_and_both_partition_modes_are_explicit() -> None:
    output = _output_plan(dpu_count=5)
    contracted = _contracted_plan(dpu_count=5, tasklets_per_dpu=3)

    assert output.tasklets_per_dpu == DEFAULT_TASKLETS_PER_DPU
    assert output.partition_kind == OUTPUT_TILE
    assert output.numeric_mode == NUMERIC_MODE_FLOAT32_REAL
    assert contracted.partition_kind == CONTRACTED_PARTIAL_SUM
    contracted_ranges = [(unit.contracted_offset, unit.contracted_elements) for unit in contracted.work_units]
    assert contracted_ranges == [(0, 6), (6, 6), (12, 6), (18, 6), (24, 5)]
    assert all(unit.output_elements == 11 for unit in contracted.work_units)


def test_json_and_native_serialization_are_deterministic_and_bound() -> None:
    plan = _contracted_plan(dpu_count=3, tasklets_per_dpu=3, output_slot=4)
    encoded_json = serialize_distributed_plan_v3(plan)
    assert encoded_json == serialize_distributed_plan_v3(parse_distributed_plan_v3(encoded_json))

    sidecar = serialize_upxdpv3(plan, package_bytes=PACKAGE, operation_bytes=OPERATION)
    assert sidecar == serialize_upxdpv3(plan, package_bytes=PACKAGE, operation_bytes=OPERATION)
    assert len(sidecar) == UPXDPV3_HEADER_BYTES + 3 * UPXDPV3_RECORD_BYTES
    header = struct.unpack_from(UPXDPV3_HEADER_FORMAT, sidecar)
    assert header[0] == UPXDPV3_MAGIC
    assert header[1:3] == (UPXDPV3_VERSION, UPXDPV3_HEADER_BYTES)
    assert header[3:7] == (3, 3, 3, 1)
    assert header[7:14] == (2, 0, 0, 0, 11, 29, 4)
    assert header[14:17] == (UPXDPV3_RECORD_BYTES, 0, 0)
    loaded = load_upxdpv3(sidecar, package_bytes=PACKAGE, operation_bytes=OPERATION, logical_operation_id="operation", logical_task_id="task")
    assert loaded == plan
    assert struct.calcsize(UPXDPV3_RECORD_FORMAT) == 8 * 4


@pytest.mark.parametrize(
    ("numeric_mode", "numeric_code"),
    ((NUMERIC_MODE_FLOAT32_REAL, 0), (NUMERIC_MODE_PER_TASK_RESIDENT_REQUANTIZE, UPXDPV3_NUMERIC_INT8_REQUANTIZE)),
)
def test_native_numeric_modes_use_stable_codes(numeric_mode: str, numeric_code: int) -> None:
    plan = _output_plan(numeric_mode=numeric_mode)
    sidecar = serialize_upxdpv3(plan, package_bytes=PACKAGE, operation_bytes=OPERATION)
    header = struct.unpack_from(UPXDPV3_HEADER_FORMAT, sidecar)

    assert header[8] == numeric_code
    assert load_upxdpv3(sidecar, package_bytes=PACKAGE, operation_bytes=OPERATION).numeric_mode == numeric_mode


def test_invalid_caps_tasklets_hashes_and_numeric_modes_are_rejected() -> None:
    with pytest.raises(ValueError, match="dpu_count"):
        _output_plan(dpu_count=3, max_dpus_per_rank=2)
    with pytest.raises(ValueError, match="tasklets"):
        _output_plan(tasklets_per_dpu=0)
    with pytest.raises(ValueError, match="tasklets"):
        _output_plan(tasklets_per_dpu=25)
    with pytest.raises(ValueError, match="SHA256"):
        _output_plan(package_sha256="bad")
    with pytest.raises(ValueError, match="numeric_mode"):
        replace(_output_plan(), numeric_mode="complex128")

    plan = _output_plan()
    with pytest.raises(ValueError, match="binding mismatch"):
        serialize_upxdpv3(plan, package_bytes=b"wrong", operation_bytes=OPERATION)


@pytest.mark.parametrize("builder", (build_output_tile_plan_v3, build_contracted_partial_sum_plan_v3))
@pytest.mark.parametrize("field", ("total_output_elements", "total_contracted_elements"))
def test_native_element_profile_cap_is_rejected(builder, field: str) -> None:
    kwargs = {
        "logical_operation_id": "operation",
        "logical_task_id": "task",
        "total_output_elements": 1,
        "total_contracted_elements": 1,
        "package_sha256": PACKAGE_SHA256,
        "operation_sha256": OPERATION_SHA256,
        "dpu_count": 1,
        field: 65537,
    }

    with pytest.raises(ValueError, match="65536"):
        builder(**kwargs)


@pytest.mark.parametrize(
    "builder",
    (build_output_tile_plan_v3, build_contracted_partial_sum_plan_v3),
)
def test_overpartitioning_is_an_explicit_unsupported_preparation(builder) -> None:
    kwargs = {
        "logical_operation_id": "operation",
        "logical_task_id": "task",
        "total_output_elements": 2,
        "total_contracted_elements": 2,
        "package_sha256": PACKAGE_SHA256,
        "operation_sha256": OPERATION_SHA256,
        "dpu_count": 3,
    }

    with pytest.raises(UnsupportedPartitionError, match="unsupported preparation"):
        builder(**kwargs)


def test_native_loader_rejects_wrong_hash_and_corrupt_record() -> None:
    plan = _output_plan(dpu_count=3)
    sidecar = serialize_upxdpv3(plan, package_bytes=PACKAGE, operation_bytes=OPERATION)
    with pytest.raises(ValueError, match="package SHA256"):
        load_upxdpv3(sidecar, package_bytes=b"wrong", operation_bytes=OPERATION)

    corrupted = bytearray(sidecar)
    record_dpu_id_offset = UPXDPV3_HEADER_BYTES + 3 * 4
    corrupted[record_dpu_id_offset:record_dpu_id_offset + 4] = (99).to_bytes(4, "little")
    with pytest.raises(ValueError):
        load_upxdpv3(bytes(corrupted), package_bytes=PACKAGE, operation_bytes=OPERATION)


def test_validation_rejects_gaps_and_duplicate_logical_work() -> None:
    plan = _output_plan(dpu_count=3)
    candidate = object.__new__(type(plan))
    for item in fields(plan):
        value = getattr(plan, item.name)
        if item.name == "work_units":
            value = (replace(plan.work_units[0], output_elements=38), *plan.work_units[1:])
        object.__setattr__(candidate, item.name, value)
    with pytest.raises(ValueError, match="coverage"):
        validate_distributed_plan_v3(candidate)


def test_real_circuit_task_selection_reuses_graph_prefix_materializer() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)
    selection = select_highest_work_supported_task(graph, network)

    assert selection.circuit_id == "bell_2q"
    assert selection.task_id in {task.id for task in graph.tasks}
    assert selection.task_index == next(index for index, task in enumerate(graph.tasks) if task.id == selection.task_id)
    assert selection.circuit_semantics_hash == graph.circuit_semantics_hash
    assert selection.tensor_network_hash == graph.tensor_network_hash
    assert selection.contraction_plan_hash == graph.contraction_plan_hash
    assert selection.contraction_path_structure_hash
    assert selection.task_hash
    assert selection.selected_task is selection.identified_graph.tasks[selection.task_index]
    assert selection.identified_graph.circuit_semantics_hash == graph.circuit_semantics_hash
    assert selection.materialization_status in {"initial_inputs_available", "materialized"}
    assert selection.left_operand.dtype == np.dtype(np.float32)
    assert selection.right_operand.dtype == np.dtype(np.float32)
    assert np.isrealobj(selection.left_operand)
    assert np.isrealobj(selection.right_operand)


def test_task_selection_rejects_nonzero_imaginary_operands_explicitly() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)
    network.tensors[1].array = np.asarray(network.tensors[1].array) + 1.0j

    with pytest.raises(ValueError, match="nonzero imaginary"):
        select_highest_work_supported_task(graph, network)


def test_task_selection_rejects_nonfinite_and_float32_overflow_operands() -> None:
    network = build_tensor_network(builtin_circuit("bell_2q"))
    graph = plan_task_graph(network)
    network.tensors[1].array = np.full(network.tensors[1].spec.shape, np.nan, dtype=np.complex128)

    with pytest.raises(ValueError, match="nonfinite"):
        select_highest_work_supported_task(graph, network)

    network.tensors[1].array = np.full(network.tensors[1].spec.shape, 1.0e100, dtype=np.complex128)
    with pytest.raises(ValueError, match="overflows float32"):
        select_highest_work_supported_task(graph, network)
