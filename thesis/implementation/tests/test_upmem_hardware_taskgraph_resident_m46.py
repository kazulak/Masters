from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from quantum_bench.bench.reporting import report_run
from quantum_bench.bench.upmem_hardware_taskgraph_resident import (
    prepare_upmem_hardware_taskgraph_resident,
)
from quantum_bench.circuits import load_circuit
from quantum_bench.core.jsonio import write_jsonl
from quantum_bench.tn import build_tensor_network, plan_task_graph_with_config, with_execution_identity
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    RESIDENT_M46_OUTPUT_TILE_ELEMENTS,
    RESIDENT_M46_PROFILE_VERSION,
    RESIDENT_SUPPORTED_TASKLETS,
    RESIDENT_V3_PROFILE_VERSION,
    _canonical_profile,
    _parse_profile,
    build_resident_graph_package,
    load_hardware_taskgraph_resident_suite,
    resident_tile_ranges,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs/suites/upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling.yml"


def test_m46_profile_is_versioned_and_restricts_tasklets() -> None:
    suite = load_hardware_taskgraph_resident_suite(SUITE)

    assert suite.profile.version == RESIDENT_M46_PROFILE_VERSION
    assert suite.profile.output_tile_elements == RESIDENT_M46_OUTPUT_TILE_ELEMENTS
    assert suite.profile.requested_dpu_count == 1
    assert suite.profile.tasklets_per_dpu == 1

    for tasklets in RESIDENT_SUPPORTED_TASKLETS:
        profile = dict(suite.profile.to_json_dict())
        profile["tasklets_per_dpu"] = tasklets
        assert _parse_profile(profile).tasklets_per_dpu == tasklets

    invalid = dict(suite.profile.to_json_dict())
    invalid["tasklets_per_dpu"] = 3
    with pytest.raises(ValueError, match="one of 1, 2, 4, 8, 16"):
        _parse_profile(invalid)

    invalid_dpus = dict(suite.profile.to_json_dict())
    invalid_dpus["requested_dpu_count"] = 2
    with pytest.raises(ValueError, match="requested_dpu_count=1"):
        _parse_profile(invalid_dpus)

    v3 = _canonical_profile(
        3, version=RESIDENT_V3_PROFILE_VERSION, requested_dpu_count=3
    ).to_json_dict()
    parsed_v3 = _parse_profile(v3)
    assert parsed_v3.requested_dpu_count == 3
    assert parsed_v3.tasklets_per_dpu == 3


def test_m46_package_request_carries_tasklets_and_aligned_slots(tmp_path: Path) -> None:
    circuit = load_circuit(
        {"circuit": {"kind": "qasm_file", "path": "configs/circuits/upmem_m2/one_qubit_ry_h_ry_a.qasm"}},
        ROOT,
    )
    network = build_tensor_network(circuit)
    graph = with_execution_identity(
        plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    )
    suite = load_hardware_taskgraph_resident_suite(SUITE)
    profile = _parse_profile({**suite.profile.to_json_dict(), "tasklets_per_dpu": 8})
    package = build_resident_graph_package(
        graph,
        network,
        case_id="m46-contract",
        suite_id=str(suite.suite["suite_id"]),
        quantization_mode="none",
        profile=profile,
    )
    binary = tmp_path / "dpu_resident"
    binary.write_bytes(b"placeholder")
    artifact = package.write(tmp_path, dpu_binary=binary, request_id="m46-contract")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))

    assert manifest["hardware_profile_version"] == RESIDENT_M46_PROFILE_VERSION
    assert manifest["requested_dpus"] == 1
    assert manifest["tasklets"] == 8
    assert all(slot.offset_bytes % 8 == 0 for slot in package.allocation.slots)
    ranges = sorted(
        (slot.offset_bytes, slot.offset_bytes + slot.capacity_bytes)
        for slot in package.allocation.slots
    )
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
    assert resident_tile_ranges(5, RESIDENT_M46_OUTPUT_TILE_ELEMENTS) == ((0, 1), (2, 3), (4, 4))


def test_m46_prepare_records_effective_cli_tasklets(tmp_path: Path) -> None:
    result = prepare_upmem_hardware_taskgraph_resident(
        tmp_path,
        suite_path=SUITE,
        tasklets_per_dpu=8,
        environment={},
    )
    resolved = yaml.safe_load(
        (result.plan_dir / "config" / "resolved_suite.yml").read_text(encoding="utf-8")
    )

    assert resolved["metadata"]["hardware_profile"]["tasklets_per_dpu"] == 8
    assert resolved["metadata"]["hardware_profile"]["effective_cli_tasklets_override"] == 8
    assert resolved["metadata"]["effective_cli_overrides"]["tasklets_per_dpu"] == 8


def test_m46_report_aggregates_explicit_sweep_parent(tmp_path: Path) -> None:
    parent = tmp_path / "runs" / "evidence" / "m46" / "upmem_hw_taskgraph_resident"
    for tasklets in (1, 8):
        child = parent / f"tasklets_{tasklets}"
        child.mkdir(parents=True)
        write_jsonl(
            child / "normalized_records.jsonl",
            [
                {
                    "run_id": child.name,
                    "suite_id": "upmem_hardware_taskgraph_resident_m4_6_tasklet_scaling",
                    "case_id": "fixture",
                    "repeat_id": 0,
                    "path_variant_id": "opt_einsum_greedy",
                    "quantization_mode": "none",
                    "status": "completed",
                    "operation_timing": [
                        {
                            "operation_id": 0,
                            "task_id": "task",
                            "component": "real",
                            "dpu_cycles": 11,
                            "tasklet_processed_elements": [2] * tasklets,
                            "active_tasklet_count": tasklets,
                            "idle_tasklet_count": 0,
                            "tasklet_utilization": 1.0,
                            "work_imbalance": 0.0,
                        }
                    ],
                    "tasklets_per_dpu": tasklets,
                    "component_operation_count": 1,
                    "real_pass_count": 1,
                    "imaginary_pass_count": 0,
                    "complex_contract_pass_count": 0,
                    "complex_combine_pass_count": 0,
                    "complex_pass_count": 0,
                    "completion_abi_version": 2,
                    "dpu_run_time_cycles": 11,
                    "dpu_graph_cycle_sum": 11,
                }
            ],
        )

    output = tmp_path / "runs" / "comparisons" / "m46"
    report_run(parent, output, output_plots=False, root_dir=tmp_path)

    rows = (output / "metrics" / "resident_observability.csv").read_text(encoding="utf-8")
    metrics = json.loads(
        (output / "metrics" / "resident_operation_tasklet_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert rows.count("fixture") == 2
    assert "tasklets_per_dpu" in rows
    assert len(metrics["records"]) == 2
