from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from quantum_bench.circuits import load_circuit
from quantum_bench.core.records import (
    CircuitSpec,
    ContractionTask,
    PathSummary,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
)
from quantum_bench.tn import (
    build_tensor_network,
    execute_task_sequence_np_einsum,
    plan_task_graph_with_config,
    with_execution_identity,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_MAX_COMPONENT_OPS,
    RESIDENT_MAX_LOGICAL_TASKS,
    RESIDENT_MAX_RANK,
    RESIDENT_MAX_SLOT_DESCRIPTORS,
    RESIDENT_MRAM_POOL_BYTES,
    RESIDENT_OPERATION_BYTES,
    RESIDENT_PACKAGE_HEADER_BYTES,
    RESIDENT_ROUTE_ID,
    ResidentCapacityError,
    build_resident_graph_package,
    build_resident_policy_reference,
    build_resident_slot_lifetime_map,
    load_hardware_taskgraph_resident_suite,
    resident_requantize,
    resident_round_nearest_even,
    resident_tile_ranges,
    validate_resident_graph_package_bytes,
    validate_hardware_taskgraph_resident_execution_request,
)
from quantum_bench.bench.upmem_hardware_taskgraph_resident import (
    prepare_upmem_hardware_taskgraph_resident,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    ResidentGraphSessionExecution,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_taskgraph_resident_path_quantization.yml"


def _one_task_complex_graph() -> tuple[TaskGraph, object]:
    left = np.array([[1.0 + 1.0j, 0.0], [0.0, 1.0 - 1.0j]], dtype=np.complex128)
    right = np.array([[1.0, 0.0 + 1.0j], [1.0 - 1.0j, 1.0]], dtype=np.complex128)
    circuit = CircuitSpec("resident_complex", 0, (), {"kind": "test"})
    left_spec = TensorSpec("left", (0, 1), left.shape, "dense", dtype="complex128")
    right_spec = TensorSpec("right", (1, 2), right.shape, "dense", dtype="complex128")
    network_spec = TensorNetworkSpec(circuit, (left_spec, right_spec), (0, 2), "ab,bc->ac")
    task = ContractionTask(
        id="task_complex",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(left.shape, right.shape),
        output_shape=(2, 2),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=2,
        gemm_n=2,
        structure="dense",
        estimated_flops=16,
        estimated_bytes=0,
    )
    graph = TaskGraph(
        network_spec,
        (task,),
        ((0, 1),),
        PathSummary("test", "greedy", 1, 1, None, None, "test"),
        0.0,
    )
    from quantum_bench.tn.network import TensorNetworkValue

    return graph, TensorNetworkValue(
        network_spec,
        [TensorValue(left_spec, left), TensorValue(right_spec, right)],
    )


def _real_graph() -> tuple[TaskGraph, object]:
    suite = load_hardware_taskgraph_resident_suite(SUITE_PATH)
    case = next(item for item in suite.suite["cases"] if item["case_id"] == "bv_3q_one_dpu")
    network = build_tensor_network(load_circuit(dict(case), ROOT))
    graph = with_execution_identity(
        plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    )
    return graph, network


def test_resident_suite_identity_and_exact_matrix() -> None:
    suite = load_hardware_taskgraph_resident_suite(SUITE_PATH)
    assert len(suite.suite["cases"]) == 13
    assert len(suite.variants) == 2
    assert suite.suite["warmups"] == 2
    assert suite.suite["repeats"] == 7
    assert suite.profile.route_id == RESIDENT_ROUTE_ID
    assert suite.profile.backend_id == "upmem_sdk_hardware_taskgraph_resident"
    assert suite.profile.session_protocol == "generic_loop_resident_graph_session_v1"
    assert suite.profile.timing_scope == "one_dpu_mram_resident_full_taskgraph_v1"
    assert suite.profile.numeric_modes == ("none", "per_task_resident_requantize")
    assert suite.profile.requested_dpu_count == 1
    assert suite.profile.tasklets_per_dpu == 1
    assert suite.profile.max_rank == RESIDENT_MAX_RANK == 16
    assert suite.profile.max_logical_tasks == RESIDENT_MAX_LOGICAL_TASKS == 32
    assert suite.profile.max_component_ops == RESIDENT_MAX_COMPONENT_OPS == 128
    assert suite.profile.max_slot_descriptors == RESIDENT_MAX_SLOT_DESCRIPTORS == 128
    assert suite.profile.mram_pool_bytes == RESIDENT_MRAM_POOL_BYTES == 512 * 1024


def test_allocator_is_deterministic_and_lifetimes_do_not_overlap() -> None:
    graph, network = _real_graph()
    first = build_resident_slot_lifetime_map(graph, network)
    second = build_resident_slot_lifetime_map(graph, network)
    assert first.to_json_dict() == second.to_json_dict()
    assert all(slot.offset_bytes % 8 == 0 for slot in first.slots)
    assert first.mram_used_bytes <= first.mram_pool_bytes
    for slot in first.slots:
        for left_index, left in enumerate(slot.lifetimes):
            for right in slot.lifetimes[left_index + 1 :]:
                assert left.end_task < right.start_task or right.end_task < left.start_task
    ordered = sorted(first.slots, key=lambda slot: slot.offset_bytes)
    for left, right in zip(ordered, ordered[1:]):
        assert left.offset_bytes + left.capacity_bytes <= right.offset_bytes


def test_capacity_failure_is_structured_and_has_no_spill() -> None:
    graph, network = _real_graph()
    profile = replace(load_hardware_taskgraph_resident_suite(SUITE_PATH).profile, max_slot_descriptors=0)
    with pytest.raises(ResidentCapacityError, match="slot_descriptor_cap_exceeded"):
        build_resident_slot_lifetime_map(graph, network, profile=profile)


def test_split_complex_has_four_intermediates_and_dpu_combine() -> None:
    graph, network = _one_task_complex_graph()
    package = build_resident_graph_package(
        graph,
        network,
        case_id="complex",
        suite_id="resident",
        quantization_mode="none",
    )
    assert [operation.component for operation in package.operations] == [
        "ar_br",
        "ai_bi",
        "ar_bi",
        "ai_br",
        "complex_combine",
    ]
    assert all(operation.to_json_dict()["intermediate_output_path"] is None for operation in package.operations)
    reference = build_resident_policy_reference(graph, network, quantization_mode="none")
    expected = np.einsum("ab,bc->ac", network.tensors[0].array, network.tensors[1].array)
    np.testing.assert_allclose(reference["output"], expected)


def test_tile_boundaries_and_explicit_nearest_even_ties() -> None:
    assert resident_tile_ranges(255) == ((0, 254),)
    assert resident_tile_ranges(256) == ((0, 255),)
    assert resident_tile_ranges(257) == ((0, 255), (256, 256))
    values = np.array([0.5, 1.5, -0.5, -1.5, 2.5, -2.5], dtype=np.float32)
    np.testing.assert_array_equal(
        resident_round_nearest_even(values),
        np.array([0, 2, 0, -2, 2, -2], dtype=np.float32),
    )
    quantized, scale, saturation = resident_requantize(np.zeros(4, dtype=np.float32))
    assert scale == 1.0
    assert saturation == 0
    np.testing.assert_array_equal(quantized, np.zeros(4, dtype=np.int8))


def test_binary_package_rejects_magic_length_and_slot_overlap() -> None:
    graph, network = _real_graph()
    package = build_resident_graph_package(
        graph, network, case_id="real", suite_id="resident", quantization_mode="none"
    )
    import quantum_bench.targets.upmem.hardware_taskgraph_resident as resident

    payload = bytearray(resident._encode_package(package.allocation.slots, package.operations))
    assert len(payload) >= RESIDENT_PACKAGE_HEADER_BYTES + 2 * 16 + RESIDENT_OPERATION_BYTES
    bad_magic = bytearray(payload)
    bad_magic[0] ^= 0x01
    with pytest.raises(ValueError, match="bad_magic"):
        validate_resident_graph_package_bytes(bytes(bad_magic))
    bad_length = bytearray(payload)
    struct.pack_into("<Q", bad_length, 24, len(payload) + 8)
    with pytest.raises(ValueError, match="file_length_mismatch"):
        validate_resident_graph_package_bytes(bytes(bad_length))
    bad_overlap = bytearray(payload)
    first_offset = struct.unpack_from("<I", payload, RESIDENT_PACKAGE_HEADER_BYTES + 4)[0]
    struct.pack_into("<I", bad_overlap, RESIDENT_PACKAGE_HEADER_BYTES + 16 + 4, first_offset)
    with pytest.raises(ValueError, match="slot_overlap"):
        validate_resident_graph_package_bytes(bytes(bad_overlap))


def test_policy_reference_float_and_resident_int8_mismatch_are_separate() -> None:
    graph, network = _real_graph()
    float_reference = build_resident_policy_reference(graph, network, quantization_mode="none")
    int8_reference = build_resident_policy_reference(
        graph, network, quantization_mode="per_task_resident_requantize"
    )
    assert float_reference["reference_kind"] == "cpu_resident_policy_reference"
    assert int8_reference["dpu_local_requantization"] is True
    assert int8_reference["scale_formula"] == "max_abs/127_or_1_for_all_zero"
    assert not np.allclose(
        int8_reference["output"], np.asarray(int8_reference["output"]) + 1.0, atol=1.0e-5
    )


def test_written_request_has_one_graph_no_intermediate_files_and_aligned_bytes(tmp_path: Path) -> None:
    graph, network = _real_graph()
    package = build_resident_graph_package(
        graph, network, case_id="real", suite_id="resident", quantization_mode="none"
    )
    dpu_binary = tmp_path / "bin" / "dpu_resident"
    dpu_binary.parent.mkdir()
    dpu_binary.write_bytes(b"resident-dpu")
    written = package.write(tmp_path, dpu_binary=dpu_binary, request_id="request-0")
    assert written.manifest_path is not None
    manifest = json.loads(written.manifest_path.read_text(encoding="utf-8"))
    assert manifest["graph_request_count"] == 1
    assert manifest["no_host_intermediate_output_files"] is True
    assert manifest["intermediate_output_paths"] == []
    assert manifest["intermediate_h2d_bytes"] == 0
    assert manifest["intermediate_d2h_bytes"] == 0
    assert manifest["control_h2d_bytes_per_launch"] == 8
    assert manifest["control_h2d_bytes"] == 16 + manifest["component_operation_count"] * 8
    for key in (
        "initial_h2d_bytes",
        "descriptor_h2d_bytes",
        "descriptor_control_bytes",
        "control_h2d_bytes_per_launch",
        "final_d2h_bytes",
    ):
        assert manifest[key] % 8 == 0
    assert not list(written.package_path.parent.glob("*operation*"))
    assert set(written.final_output_paths) == {"real"}


def test_physical_guard_and_resident_build_macros_are_strict() -> None:
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        validate_hardware_taskgraph_resident_execution_request(execute=True, environment={})
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        validate_hardware_taskgraph_resident_execution_request(
            execute=True,
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "DPU_BACKEND": "simulator"},
        )
    makefile = (
        ROOT / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop_resident" / "Makefile"
    ).read_text(encoding="utf-8")
    assert "NR_TASKLETS ?= 1" in makefile
    assert "RESIDENT_MAX_LOGICAL_TASKS" in makefile
    assert "RESIDENT_MAX_COMPONENT_OPS" in makefile
    assert "RESIDENT_MAX_SLOT_DESCRIPTORS" in makefile
    assert "RESIDENT_MRAM_POOL_BYTES" in makefile


def test_runner_validates_fake_resident_response_and_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import quantum_bench.bench.upmem_hardware_taskgraph_resident as resident_bench

    graph, network = _real_graph()
    suite = load_hardware_taskgraph_resident_suite(SUITE_PATH)
    reference = np.asarray(execute_task_sequence_np_einsum(graph, network)[0])
    build_root = tmp_path / "session"
    dpu_binary = build_root / "bin" / "dpu_resident"
    dpu_binary.parent.mkdir(parents=True)
    dpu_binary.write_bytes(b"resident-dpu")
    build = HardwareSessionBuild(
        session_root=build_root,
        source_snapshot=build_root,
        build_dir=build_root,
        host_binary=build_root / "bin" / "host",
        dpu_binary=dpu_binary,
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.0,
        build_command=("make", "NR_TASKLETS=1"),
        sdk_tools={},
    )
    perturb = {"value": False}

    def fake_execute(build, *, manifest_path, response_path, profile, environment):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        package_output = np.asarray(reference.real, dtype=np.float32) + (
            1.0 if perturb["value"] else 0.0
        )
        for item in manifest["final_outputs"]:
            path = build.session_root / item["output_path"]
            np.asarray(package_output, dtype=np.float32).tofile(path)
        response = {
            "schema_version": "generic_loop_resident_graph_session_v1",
            "manifest_kind": "resident_graph_response",
            "route_id": "upmem_tn_hardware_taskgraph_resident",
            "backend_id": "upmem_sdk_hardware_taskgraph_resident",
            "hardware_profile_version": "hardware_taskgraph_single_dpu_mram_resident_v1",
            "target_requested": "hardware",
            "target_observed": "hardware",
            "sdk_allocation_profile": "backend=hw",
            "sdk_allocation_profile_verified": True,
            "session_protocol": "generic_loop_resident_graph_session_v1",
            "quantization_mode": manifest["quantization_mode"],
            "status": "completed",
            "failure_stage": None,
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "tasklets": 1,
            "graph_request_count": 1,
            "native_launch_count": manifest["component_operation_count"],
            "native_task_count": manifest["component_operation_count"],
            "allocation_count": 1,
            "hardware_allocation_verified": True,
            "hardware_execution": True,
            "hardware_kernel_executed": True,
            "native_execution": True,
            "native_hardware_backend": True,
            "hardware_backend_verified": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "hardware_release_verified": True,
            "release_confirmed": True,
            "physical_dependency_chain_verified": True,
            "hardware_timing_available": True,
            "persistent_session_reused": False,
            "resident_slots_persist_for_graph": True,
            "final_output_only_d2h": True,
            "physical_bus_bytes_available": False,
            "intermediate_h2d_bytes": 0,
            "intermediate_d2h_bytes": 0,
            "initial_h2d_bytes": manifest["initial_h2d_bytes"],
            "descriptor_h2d_bytes": manifest["descriptor_h2d_bytes"],
            "control_h2d_bytes": manifest["control_h2d_bytes"],
            "final_d2h_bytes": manifest["final_d2h_bytes"],
            "actual_h2d_bytes": manifest["initial_h2d_bytes"] + manifest["descriptor_h2d_bytes"] + manifest["control_h2d_bytes"],
            "actual_d2h_bytes": manifest["final_d2h_bytes"],
            "actual_transfer_bytes": manifest["initial_h2d_bytes"] + manifest["descriptor_h2d_bytes"] + manifest["control_h2d_bytes"] + manifest["final_d2h_bytes"],
            "final_outputs": [
                {"component": item["component"], "slot_id": item["slot_id"], "status": "completed", "output_path": item["output_path"], "elements": item["elements"], "raw_bytes": item["raw_bytes"], "transfer_bytes": item["transfer_bytes"]}
                for item in manifest["final_outputs"]
            ],
        }
        for key in (
            "package_parse_time_s", "allocation_time_s", "binary_load_time_s", "initial_h2d_time_s",
            "descriptor_h2d_time_s", "control_h2d_time_s", "kernel_time_s", "final_d2h_time_s",
            "output_write_time_s", "release_time_s", "steady_state_graph_execution_s",
        ):
            response[key] = 0.0
        response_path.write_text(json.dumps(response), encoding="utf-8")
        return ResidentGraphSessionExecution(
            "completed", None, response_path, response, 0.001, ("fake",), "", ""
        )

    monkeypatch.setattr(resident_bench, "execute_resident_graph_session", fake_execute)
    result = resident_bench.execute_resident_variant(
        root_dir=ROOT,
        run_dir=tmp_path,
        native_build=build,
        profile=suite.profile,
        suite_id=str(suite.suite["suite_id"]),
        case_id="real",
        variant_id="opt_einsum_greedy",
        graph=graph,
        network=network,
        reference_output=reference,
        quantization_mode="none",
        request_id="pass",
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    assert result.status == "completed"
    assert result.summary["graph_request_count"] == 1
    assert result.summary["bytes_invariant_status"] == "passed"
    assert result.summary["policy_reference_validation"]["passed"] is True
    perturb["value"] = True
    mismatch = resident_bench.execute_resident_variant(
        root_dir=ROOT,
        run_dir=tmp_path,
        native_build=build,
        profile=suite.profile,
        suite_id=str(suite.suite["suite_id"]),
        case_id="real",
        variant_id="opt_einsum_greedy",
        graph=graph,
        network=network,
        reference_output=reference,
        quantization_mode="none",
        request_id="mismatch",
        environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    assert mismatch.status == "failed"
    assert mismatch.summary["policy_reference_validation"]["passed"] is False


def test_prepare_only_runner_creates_plan_without_hardware(tmp_path: Path) -> None:
    result = prepare_upmem_hardware_taskgraph_resident(
        tmp_path, suite_path=SUITE_PATH, build=False, environment={}
    )
    assert result.status == "prepared"
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False
    assert len(summary["prepared_cases"]) == 13
