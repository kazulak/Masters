from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from quantum_bench.bench.m5_circuit_study import (
    DEFAULT_TOLERANCES,
    _engine_metadata,
    _estimate_resources,
    _executor_config_hash,
    _first_byte_count,
    _is_quantized_policy,
    _physical_timing_complete,
    _policy,
    _validation,
    load_study_config,
    plan_study,
    run_study,
)
from quantum_bench.core.records import TensorSpec
from quantum_bench.execution.contracts import NumericMode, UpmemTopology
from quantum_bench.execution.numeric import contract_node
from quantum_bench.tn.graph import ContractionDAG, ContractNode, TensorView


def _study(
    path: Path,
    *,
    max_live_bytes: int = 1_000_000,
    include_int8: bool = False,
    physical: bool = False,
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "study_id": "m5_test",
                "cases": [
                    {
                        "id": "bell",
                        "family": "bell",
                        "circuit": {"kind": "builtin", "name": "bell_2q"},
                    },
                ],
                "planner_variants": [
                    {
                        "id": "greedy",
                        "planner": {"engine": "opt_einsum", "optimize": "greedy"},
                    },
                    {
                        "id": "auto",
                        "planner": {"engine": "opt_einsum", "optimize": "auto"},
                    },
                ],
                "numeric_policies": [
                    {"id": "float", "policy": "float32_real"},
                    *(
                        [{"id": "int8", "policy": "host_packed_int8"}]
                        if include_int8
                        else []
                    ),
                ],
                "engine_variants": [
                    {
                        "id": "cpu",
                        "engine": "numpy_cpu",
                        "topology": {
                            "backend": "cpu",
                            "device_ids": ["cpu"],
                            "tasklets_per_device": 1,
                        },
                    },
                    *(
                        [
                            {
                                "id": "injected",
                                "engine": "fake",
                                "timeout_enforcement": "engine_subprocess",
                                "topology": {
                                    "backend": "upmem",
                                    "device_ids": ["dpu:0", "dpu:1"],
                                    "rank_paths": ["/dev/dpu_rank0"],
                                    "tasklets_per_device": 1,
                                },
                                "executor_config": {
                                    "profile": "m5_whole_circuit_v4_v1",
                                    "abi": "execution_plan_v4",
                                    "session_protocol": "persistent_rank_session_v1",
                                    "dispatch_mode": "bulk_set_synchronous_v1",
                                    "kernel_identity": "dpu_gemm_tile_v4",
                                    "execution_class": "physical_v4_output_tile",
                                },
                            }
                        ]
                        if physical
                        else []
                    ),
                ],
                "warmups": 1,
                "repeats": 2,
                "timeout_s": 30,
                "resource_limits": {
                    "max_live_bytes": max_live_bytes,
                    "element_bytes": 4,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class _FakeEngine:
    name = "fake"


class _VerifiedPhysicalEngine:
    name = "verified_physical"
    target_observed = "physical_hardware"
    observed_rank_count = 1
    allocated_dpu_count = 2
    observed_tasklets_per_dpu = 1
    session_root = Path("/tmp/m5-functional-study-fake")
    host_binary = Path(sys.executable)
    dpu_binary = Path(__file__)
    initialization_binary = Path(__file__)
    rank_paths = ("/dev/dpu_rank0",)

    def open_session(self, policy: NumericMode, topology: UpmemTopology):
        class Session:
            def execute(
                self,
                task: ContractNode,
                left,
                right,
                *,
                node_plan: object | None = None,
            ):
                mode = (
                    NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                    if policy is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                    else NumericMode.FLOAT32_REAL
                )
                output = contract_node(task, left, right, mode)
                return output, {
                    "engine": "verified_physical",
                    "device": "physical-test-dpu",
                    "input_dtype": "int8"
                    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                    else "float32",
                    "native_kernel_executed": True,
                    "hardware_kernel_executed": True,
                    "hardware_allocation_verified": True,
                    "hardware_release_verified": True,
                    "target_observed": "physical-test-dpu",
                    "physical_profile": "m5_whole_circuit_v4_v1",
                    "profile": "m5_whole_circuit_v4_v1",
                    "abi": "execution_plan_v4",
                    "abi_version": "execution_plan_v4",
                    "numeric_transport": policy.value,
                    "session_protocol": "persistent_rank_session_v1",
                    "dispatch_mode": "bulk_set_synchronous_v1",
                    "kernel_identity": "dpu_gemm_tile_v4",
                    "execution_class": "physical_v4_output_tile",
                    "graph_intermediate_placement": "host_managed",
                    "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
                    "native_identity_verified": True,
                    "physical_plan_consumed": node_plan is not None,
                    "application_visible_h2d_bytes": 2,
                    "application_visible_d2h_bytes": 3,
                    "application_visible_transfer_bytes": 5,
                    "timing": {
                        "preparation_time_s": 0.04
                        if mode is NumericMode.FLOAT32_REAL
                        else 0.0,
                        "h2d_time_s": 0.01,
                        "kernel_time_s": 0.02,
                        "d2h_time_s": 0.03,
                        "host_quantization_time_s": 0.04
                        if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                        else 0.0,
                        "host_dequantization_time_s": 0.05,
                    },
                    # The physical engine currently exposes host conversion
                    # times both directly and in its nested timing object.
                    # Aggregation must count each stage once.
                    "host_quantization_time_s": 0.04
                    if mode is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
                    else 0.0,
                    "host_dequantization_time_s": 0.05,
                    "request_level_speedup_applicable": False,
                    "request_timing_is_bringup_only": True,
                }

            def close(self):
                return {
                    "backend_id": "upmem_sdk_hardware_v4_tile_session",
                    "target_observed": self_owner.target_observed,
                    "observed_rank_count": self_owner.observed_rank_count,
                    "requested_dpu_count": self_owner.allocated_dpu_count,
                    "allocated_dpu_count": self_owner.allocated_dpu_count,
                    "observed_tasklets_per_dpu": self_owner.observed_tasklets_per_dpu,
                    "tasklets_per_dpu": self_owner.observed_tasklets_per_dpu,
                    "hardware_allocation_verified": True,
                    "native_kernel_executed": True,
                    "hardware_kernel_executed": True,
                    "simulator_kernel_executed": False,
                    "cpu_fallback_used": False,
                    "hardware_release_verified": True,
                    "hardware_release_confirmed": True,
                    "profile": "m5_whole_circuit_v4_v1",
                    "abi": "execution_plan_v4",
                    "session_protocol": "persistent_rank_session_v1",
                    "dispatch_mode": "bulk_set_synchronous_v1",
                    "kernel_identity": "dpu_gemm_tile_v4",
                    "execution_class": "physical_v4_output_tile",
                    "graph_intermediate_placement": "host_managed",
                    "graph_intermediate_placement_origin": "m5_host_coordinator_v1",
                    "native_identity_verified": True,
                }

        self_owner = self
        return Session()


class _FailingEngine:
    name = "failing"

    def open_session(self, policy: NumericMode, topology: UpmemTopology):
        raise RuntimeError("deliberate engine failure")


def _records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "normalized_records.jsonl").read_text().splitlines()
    ]


def test_config_plans_all_case_planner_combinations_without_engine_calls(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    config = load_study_config(study)
    assert config["study_id"] == "m5_test"
    assert config["schema_version"] == "m5_circuit_study_v2"

    manifest = plan_study(tmp_path, study)
    payload = json.loads(manifest.read_text())
    assert len(payload["plans"]) == 2
    assert payload["hardware_opened"] is False
    assert all(item["dag_node_count"] > 0 for item in payload["plans"])
    assert all(
        item["contraction_plan_hash"]
        == item["contraction_path_structure_hash"]
        == item["contraction_dag_hash"]
        for item in payload["plans"]
    )
    assert all(
        item["task_count"] == item["dag_node_count"] for item in payload["plans"]
    )


def test_route_selection_filters_rows_and_rejects_unknown_or_empty_ids(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    route_id = "greedy__float__cpu"

    plan = json.loads(
        plan_study(tmp_path, study, route_ids=[route_id]).read_text(encoding="utf-8")
    )
    assert plan["selected_route_ids"] == [route_id]
    assert len(plan["pipeline_routes"]) == 1
    assert plan["pipeline_comparisons"] == []

    run_dir = run_study(tmp_path, study, route_ids=[route_id])
    rows = _records(run_dir)
    assert rows and {row["route_id"] for row in rows} == {route_id}
    assert all(
        row["route_label"] and row["route_config_hash"] and row["route_modules"]
        for row in rows
    )
    assert all(row["comparison_ids"] == [] for row in rows)
    summary = json.loads((run_dir / "m5_circuit_study_summary.json").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert summary["selected_route_ids"] == [route_id]
    assert manifest["selected_route_ids"] == [route_id]
    assert len(manifest["pipeline_routes"]) == 1
    assert manifest["pipeline_comparisons"] == []

    with pytest.raises(ValueError, match="unknown route ids"):
        plan_study(tmp_path, study, route_ids=["not-a-route"])
    with pytest.raises(ValueError, match="selection cannot be empty"):
        plan_study(tmp_path, study, route_ids=[])
    with pytest.raises(ValueError, match="duplicate route ids"):
        plan_study(tmp_path, study, route_ids=[route_id, route_id])


def test_numeric_policy_rejects_unknown_configuration_keys(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    value = yaml.safe_load(study.read_text())
    value["numeric_policies"][0]["rounding"] = "nearest"
    study.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported keys: rounding"):
        load_study_config(study)


def test_rank_override_regenerates_route_contract_without_changing_plan_hash(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    route_id = "greedy__float__injected"
    base_run = run_study(
        tmp_path,
        study,
        engine_factories={"fake": _VerifiedPhysicalEngine},
        route_ids=[route_id],
    )
    overridden_run = run_study(
        tmp_path,
        study,
        engine_factories={"fake": _VerifiedPhysicalEngine},
        rank_paths=["/dev/dpu_rank1"],
        route_ids=[route_id],
    )
    assert {row["contraction_plan_hash"] for row in _records(base_run)} == {
        row["contraction_plan_hash"] for row in _records(overridden_run)
    }


def test_physical_engine_rank_paths_must_match_suite_binding(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    value = yaml.safe_load(study.read_text())
    value["engine_variants"][1]["topology"]["rank_paths"] = ["/dev/dpu_rank1"]
    study.write_text(yaml.safe_dump(value), encoding="utf-8")

    run_dir = run_study(
        tmp_path, study, engine_factories={"fake": _VerifiedPhysicalEngine}
    )
    rows = [row for row in _records(run_dir) if row["engine_id"] == "injected"]
    assert rows and all(row["status"] == "failed" for row in rows)
    assert all(
        "rank_paths do not match suite-resolved topology" in row["error"]
        for row in rows
    )


def test_cpu_route_rejects_non_numpy_engine_before_execution(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    value = yaml.safe_load(study.read_text())
    value["engine_variants"][0]["engine"] = "not_numpy"
    study.write_text(yaml.safe_dump(value), encoding="utf-8")

    run_dir = run_study(tmp_path, study)
    rows = _records(run_dir)
    assert rows and all(row["status"] == "failed" for row in rows)
    assert all(
        row["failure_stage"] == "execution_plan_compilation_failed" for row in rows
    )
    assert all("numpy_cpu" in row["error"] for row in rows)


def test_run_records_warmups_repeats_and_same_plan_hashes(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    run_dir = run_study(tmp_path, study, engine_factories={"fake": _FakeEngine})
    records = _records(run_dir)

    assert len(records) == 4  # 2 planners x CPU x 2 measured repeats
    assert all(row["repeat_id"] in {0, 1} for row in records)
    assert all(row["status"] == "completed" for row in records)
    assert all(row["dag_node_count"] > 0 and row["exact_once"] for row in records)
    assert all(row["complete_task_count"] == row["dag_node_count"] for row in records)
    assert all(row["planner_hash"] == row["planner_config_hash"] for row in records)
    assert all(
        row["contraction_plan_hash"]
        == row["contraction_path_structure_hash"]
        == row["contraction_dag_hash"]
        for row in records
    )
    assert all(
        row["exact_once_scope"] == "host_dag_node_completion_per_route"
        and row["host_dag_node_completion_coverage"] is True
        for row in records
    )
    for planner_id in {"greedy", "auto"}:
        selected = [row for row in records if row["planner_id"] == planner_id]
        assert len({row["contraction_plan_hash"] for row in selected}) == 1
        assert {row["contraction_plan_hash"] for row in selected} == {
            row["contraction_dag_hash"] for row in selected
        }
        assert len({row["circuit_semantics_hash"] for row in selected}) == 1
        assert len({row["executor_config_hash"] for row in selected}) == 1
    assert (run_dir / "m5_circuit_study_summary.json").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_resource_estimate_keeps_initial_inputs_live_and_labels_width() -> None:
    left = TensorSpec("left", (0,), (2,), "dense", dtype="float64")
    right = TensorSpec("right", (0,), (2,), "dense", dtype="float64")
    scale = TensorSpec("scale", (), (), "dense", dtype="float64")
    partial = TensorSpec(
        "partial", (), (), "dense", dtype="float64", produced_by="contract_0"
    )
    output = TensorSpec(
        "output", (), (), "dense", dtype="float64", produced_by="contract_1"
    )
    first = ContractNode(
        node_id="contract_0",
        left=TensorView(tensor_id="left", labels=(0,), shape=(2,)),
        right=TensorView(tensor_id="right", labels=(0,), shape=(2,)),
        output=partial,
        contracted_labels=(0,),
        output_labels=(),
    )
    second = ContractNode(
        node_id="contract_1",
        left=TensorView(tensor_id="partial", labels=(), shape=()),
        right=TensorView(tensor_id="scale", labels=(), shape=()),
        output=output,
        contracted_labels=(),
        output_labels=(),
        dependencies=("contract_0",),
    )
    dag = ContractionDAG(
        tensors=(left, right, scale),
        nodes=(first, second),
        output=TensorView(tensor_id="output", labels=(), shape=()),
    )

    resources = _estimate_resources(dag, {"element_bytes": 4})

    assert resources["initial_tensor_bytes"] == 20
    assert resources["peak_live_bytes"] == 28
    assert resources["initial_host_input_bytes"] == 40
    assert resources["byte_accounting_basis"] == "configured_execution_element_width"
    assert resources["configured_execution_element_bytes"] == 4
    assert resources["element_bytes"] == 4


def test_route_compilation_is_once_per_route_outside_warmups_and_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quantum_bench.bench.m5_circuit_study as module

    study = tmp_path / "study.yml"
    _study(study)
    original = module.compile_execution
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "compile_execution", counted)
    run_dir = run_study(tmp_path, study)
    rows = _records(run_dir)
    # Two CPU anchors plus two planned CPU routes, independent of 1 warmup and
    # two measured repetitions for each route.
    assert calls == 4
    assert all(row["compilation_time_s"] >= 0.0 for row in rows)


def test_timing_admission_requires_warmup_and_three_repeats(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    short_run = run_study(tmp_path, study)
    short_rows = _records(short_run)
    assert all(row["measurement_repetitions_sufficient"] is False for row in short_rows)
    assert all(row["timing_is_bringup_only"] is True for row in short_rows)

    value = yaml.safe_load(study.read_text())
    value["repeats"] = 3
    study.write_text(yaml.safe_dump(value), encoding="utf-8")
    measured_run = run_study(tmp_path, study)
    measured_rows = _records(measured_run)
    assert all(
        row["measurement_repetitions_sufficient"] is True for row in measured_rows
    )
    assert all(row["timing_is_bringup_only"] is False for row in measured_rows)
    assert all(row["hardware_speedup_applicable"] is False for row in measured_rows)


def test_preflight_unsupported_is_recorded_without_engine_factory(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, max_live_bytes=1)
    run_dir = run_study(tmp_path, study, engine_factories={"injected": _FakeEngine})
    records = _records(run_dir)
    assert len(records) == 2  # one row per planner and CPU policy combination
    assert all(row["status"] == "unsupported" for row in records)
    assert all(row["failure_stage"] == "preflight_resource_limit" for row in records)
    assert all(row["error"] for row in records)


def test_engine_failure_is_preserved_without_fallback(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    run_dir = run_study(tmp_path, study, engine_factories={"fake": _FailingEngine})
    records = _records(run_dir)
    failed = [row for row in records if row["engine_id"] == "injected"]
    assert failed
    assert all(row["status"] == "failed" for row in failed)
    assert all(row["no_fallback_used"] for row in failed)
    assert all(row["failure_stage"] == "engine_execution_failed" for row in failed)
    assert all(row["simulator_kernel_executed"] is False for row in failed)


def test_quantized_row_keeps_policy_and_full_precision_validation_separate(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, include_int8=True)
    run_dir = run_study(tmp_path, study)
    rows = [
        row
        for row in _records(run_dir)
        if row["numeric_policy_id"] == "int8" and row["engine_id"] == "cpu"
    ]
    assert rows
    assert all("policy_reference_validation" in row for row in rows)
    assert all("full_precision_accuracy" in row for row in rows)
    assert all(row["policy_reference_validation"]["status"] == "passed" for row in rows)
    assert all(
        row["full_precision_accuracy"]["status"] in {"passed", "failed"} for row in rows
    )
    float_rows = [
        row
        for row in _records(run_dir)
        if row["numeric_policy_id"] == "float" and row["engine_id"] == "cpu"
    ]
    assert {row["executor_config_hash"] for row in rows} == {
        row["executor_config_hash"] for row in float_rows
    }


def test_host_packed_runtime_name_uses_quantized_validation_tolerances() -> None:
    policy = _policy("host_packed_int8")
    assert policy is NumericMode.HOST_PACKED_INT8_PER_TASK_V1
    assert _is_quantized_policy(policy) is True
    actual = np.array([1.1], dtype=np.float32)
    expected = np.array([1.0], dtype=np.float32)
    assert (
        _validation(actual, expected, DEFAULT_TOLERANCES, quantized=False)["status"]
        == "failed"
    )
    assert (
        _validation(
            actual, expected, DEFAULT_TOLERANCES, quantized=_is_quantized_policy(policy)
        )["status"]
        == "passed"
    )


def test_physical_success_requires_native_metadata_and_sums_task_bytes(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    run_dir = run_study(
        tmp_path, study, engine_factories={"fake": _VerifiedPhysicalEngine}
    )
    rows = [row for row in _records(run_dir) if row["engine_id"] == "injected"]
    assert rows and all(row["status"] == "completed" for row in rows)
    assert all(row["hardware_execution_verified"] for row in rows)
    assert all(row["target_observed"] == "physical_hardware" for row in rows)
    assert all(row["observed_rank_count"] == 1 for row in rows)
    assert all(row["allocated_dpu_count"] == 2 for row in rows)
    assert all(row["observed_tasklets_per_dpu"] == 1 for row in rows)
    assert all(row["rank_binding_sha256"] for row in rows)
    assert all(row["native_identity_verified"] is True for row in rows)
    assert all(row["physical_plan_consumed"] is True for row in rows)
    assert all(
        row["graph_intermediate_placement_origin"] == "m5_host_coordinator_v1"
        for row in rows
    )
    assert len({row["executor_config_hash"] for row in rows}) == 1
    assert rows[0]["engine_metadata"]["application_visible_h2d_bytes"] == 6
    assert rows[0]["engine_metadata"]["application_visible_d2h_bytes"] == 9
    assert rows[0]["engine_metadata"]["application_visible_transfer_bytes"] == 15
    assert rows[0]["application_visible_h2d_bytes"] == 6
    assert rows[0]["application_visible_d2h_bytes"] == 9
    assert rows[0]["application_visible_transfer_bytes"] == 15
    assert rows[0]["transfer"] == {
        "application_visible_h2d_bytes": 6,
        "h2d_bytes": 6,
        "application_visible_d2h_bytes": 9,
        "d2h_bytes": 9,
        "application_visible_transfer_bytes": 15,
        "transfer_bytes": 15,
    }
    assert rows[0]["transfer_accounting_verified"] is True
    assert rows[0]["h2d_time_s"] == pytest.approx(0.03)
    assert rows[0]["kernel_time_s"] == pytest.approx(0.06)
    assert rows[0]["d2h_time_s"] == pytest.approx(0.09)
    assert rows[0]["preparation_time_s"] == pytest.approx(0.12)
    assert rows[0]["host_quantization_time_s"] is None
    assert rows[0]["host_dequantization_time_s"] == pytest.approx(0.15)
    assert rows[0]["timing_breakdown"]["session_open_s"] >= 0.0
    assert rows[0]["timing_breakdown"]["graph_execution_s"] >= 0.0
    assert rows[0]["timing_breakdown"]["session_close_s"] >= 0.0
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["hardware_opened"] is True


def test_physical_route_rejects_observed_kernel_identity_mismatch(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    value = yaml.safe_load(study.read_text())
    physical = next(
        item for item in value["engine_variants"] if item["id"] == "injected"
    )
    physical["executor_config"]["kernel_identity"] = "different-kernel-v1"
    study.write_text(yaml.safe_dump(value), encoding="utf-8")

    run_dir = run_study(
        tmp_path, study, engine_factories={"fake": _VerifiedPhysicalEngine}
    )
    rows = [row for row in _records(run_dir) if row["engine_id"] == "injected"]
    assert rows and all(row["status"] == "failed" for row in rows)
    assert all(row["failure_stage"] == "route_observation_mismatch" for row in rows)
    assert all(row["route_adapter_validation"]["status"] == "passed" for row in rows)
    assert all(row["route_observation_admission"]["status"] == "failed" for row in rows)
    assert all("kernel_identity" in row["error"] for row in rows)


def test_repeated_verified_physical_route_is_admitted_only_at_study_level(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    value = yaml.safe_load(study.read_text())
    value["repeats"] = 3
    study.write_text(yaml.safe_dump(value), encoding="utf-8")

    run_dir = run_study(
        tmp_path, study, engine_factories={"fake": _VerifiedPhysicalEngine}
    )
    rows = [row for row in _records(run_dir) if row["engine_id"] == "injected"]
    assert rows
    assert all(row["measurement_repetitions_sufficient"] for row in rows)
    assert all(row["timing_is_bringup_only"] is False for row in rows)
    assert all(row["hardware_speedup_applicable"] is True for row in rows)
    assert all(
        all(
            metric["request_level_speedup_applicable"] is False
            and metric["request_timing_is_bringup_only"] is True
            for metric in row["engine_metadata"]["task_metrics"]
        )
        for row in rows
    )


def test_physical_evidence_helpers_reject_unsafe_values() -> None:
    assert _first_byte_count({"h2d_bytes": 8}, keys=("h2d_bytes",)) == 8
    assert _first_byte_count({"h2d_bytes": 1.5}, keys=("h2d_bytes",)) is None
    assert _first_byte_count({"h2d_bytes": -1}, keys=("h2d_bytes",)) is None
    assert _first_byte_count({"h2d_bytes": True}, keys=("h2d_bytes",)) is None

    valid = {"h2d_time_s": 0.1, "kernel_time_s": 0.2, "d2h_time_s": 0.3}
    assert _physical_timing_complete(valid)
    for invalid in (-1.0, float("nan"), float("inf"), True):
        assert not _physical_timing_complete({**valid, "kernel_time_s": invalid})


def test_engine_metadata_preserves_session_level_fallback_flags() -> None:
    result = _engine_metadata(
        {
            "cpu_fallback_used": True,
            "simulator_kernel_executed": True,
            "task_metrics": (
                {
                    "cpu_fallback_used": False,
                    "simulator_kernel_executed": False,
                },
            ),
        }
    )
    assert result["cpu_fallback_used"] is True
    assert result["simulator_kernel_executed"] is True


def test_physical_topology_without_rank_paths_is_rejected(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    value = yaml.safe_load(study.read_text())
    value["engine_variants"][1]["topology"]["rank_paths"] = []
    study.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="rank_paths"):
        load_study_config(study)


def test_physical_manifest_stays_closed_when_no_hardware_factory_runs(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    run_dir = run_study(tmp_path, study)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["hardware_opened"] is False


def test_anchor_failure_is_recorded_and_never_emits_completed_evidence(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    value = yaml.safe_load(study.read_text())
    value["cases"][0]["circuit"] = {
        "kind": "builtin",
        "name": "quantization_stress",
        "n_qubits": 2,
        "repeat_layers": 1,
    }
    study.write_text(yaml.safe_dump(value), encoding="utf-8")
    run_dir = run_study(tmp_path, study, engine_factories={"fake": _FakeEngine})
    rows = _records(run_dir)
    assert rows and all(row["status"] == "failed" for row in rows)
    assert all(row["failure_stage"] == "anchor_generation_failed" for row in rows)
    assert all(row["validation_status"] == "not_run" for row in rows)


def test_physical_timeout_must_be_engine_enforced(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)
    value = yaml.safe_load(study.read_text())
    value["engine_variants"][1]["timeout_enforcement"] = "posthoc_observation"
    study.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="subprocess"):
        load_study_config(study)


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_observed": None},
        {"allocated_dpu_count": 1},
        {"observed_tasklets_per_dpu": 2},
    ],
)
def test_physical_admission_rejects_missing_or_mismatched_proof(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    study = tmp_path / "study.yml"
    _study(study, physical=True)

    class InvalidPhysicalEngine(_VerifiedPhysicalEngine):
        pass

    for key, value in overrides.items():
        setattr(InvalidPhysicalEngine, key, value)
    run_dir = run_study(
        tmp_path, study, engine_factories={"fake": InvalidPhysicalEngine}
    )
    rows = [row for row in _records(run_dir) if row["engine_id"] == "injected"]
    assert rows
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["failure_stage"] == "hardware_execution_unverified" for row in rows)
    assert all(not row["hardware_execution_verified"] for row in rows)


def test_executor_hash_matches_numeric_ablations_but_separates_fixed_identity() -> None:
    variant = {
        "id": "upmem",
        "engine": "upmem_resident",
        "topology": {
            "rank_paths": ["/dev/dpu_rank0"],
        },
        "executor_config": {
            "physical_profile": "m5-v1",
            "abi_version": "abi-v1",
            "numeric_transport": "float32_mram",
            "kernel_identity": "contract-v1",
        },
    }
    topology = UpmemTopology(dpu_count=2, tasklets_per_dpu=1, rank_count=1)
    float_hash = _executor_config_hash(
        variant,
        _policy("float32_real"),
        topology,
        {"numeric_transport": "float32_mram"},
    )
    int8_hash = _executor_config_hash(
        {
            **variant,
            "executor_config": {
                **variant["executor_config"],
                "numeric_transport": "packed_int8_mram",
            },
        },
        _policy("host_packed_int8"),
        topology,
        {"numeric_transport": "packed_int8_mram"},
    )
    assert float_hash == int8_hash

    profile_hash = _executor_config_hash(
        {
            **variant,
            "executor_config": {
                **variant["executor_config"],
                "physical_profile": "m5-v2",
            },
        },
        _policy("float32_real"),
        topology,
        {},
    )
    kernel_hash = _executor_config_hash(
        {
            **variant,
            "executor_config": {
                **variant["executor_config"],
                "kernel_identity": "contract-v2",
            },
        },
        _policy("float32_real"),
        topology,
        {},
    )
    topology_hash = _executor_config_hash(
        {
            **variant,
            "id": "same-implementation-different-label",
            "topology": {"rank_paths": ["/dev/dpu_rank1", "/dev/dpu_rank2"]},
        },
        _policy("float32_real"),
        UpmemTopology(dpu_count=1, tasklets_per_dpu=1, rank_count=1),
        {},
    )
    engine_hash = _executor_config_hash(
        {**variant, "engine": "upmem_resident_v4"},
        _policy("float32_real"),
        topology,
        {},
    )
    assert float_hash == topology_hash
    assert float_hash not in {profile_hash, kernel_hash, engine_hash}
