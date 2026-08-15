from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from quantum_bench.bench.m5_circuit_study import (
    DEFAULT_TOLERANCES,
    _engine_metadata,
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
from quantum_bench.whole_circuit import DeviceTopology, EngineTaskResult, NumpyCpuEngine


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
                    {
                        "id": "injected",
                        "engine": "fake",
                        "timeout_enforcement": "engine_subprocess"
                        if physical
                        else "posthoc_observation",
                        "topology": {
                            "backend": "upmem" if physical else "cpu",
                            "device_ids": ["dpu:0", "dpu:1"],
                            "rank_paths": ["/dev/dpu_rank0"] if physical else [],
                            "tasklets_per_device": 1,
                        },
                        **(
                            {
                                "executor_config": {
                                    "profile": "test-profile-v1",
                                    "abi": "test-abi-v1",
                                    "session_protocol": "test-session-v1",
                                    "dispatch_mode": "bulk-synchronous",
                                    "kernel_identity": "test-kernel-v1",
                                    "execution_class": "physical_test",
                                }
                            }
                            if physical
                            else {}
                        ),
                    },
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


class _FakeEngine(NumpyCpuEngine):
    name = "fake"

    def open_session(self, policy: object, topology: DeviceTopology):
        return super().open_session(policy, DeviceTopology())


class _VerifiedPhysicalEngine(NumpyCpuEngine):
    name = "verified_physical"
    target_observed = "physical_hardware"
    observed_rank_count = 1
    allocated_dpu_count = 2
    observed_tasklets_per_dpu = 1

    def open_session(self, policy: object, topology: DeviceTopology):
        inner = super().open_session(policy, DeviceTopology())

        class Session:
            def execute(self, task, left, right):
                result = inner.execute(task, left, right)
                return EngineTaskResult(
                    result.output,
                    {
                        **result.metadata,
                        "native_kernel_executed": True,
                        "hardware_kernel_executed": True,
                        "hardware_allocation_verified": True,
                        "hardware_release_verified": True,
                        "target_observed": "physical-test-dpu",
                        "physical_profile": "test-profile-v1",
                        "profile": "test-profile-v1",
                        "abi": "test-abi-v1",
                        "abi_version": "test-abi-v1",
                        "numeric_transport": policy.name,
                        "session_protocol": "test-session-v1",
                        "dispatch_mode": "bulk-synchronous",
                        "kernel_identity": "test-kernel-v1",
                        "execution_class": "physical_test",
                        "graph_intermediate_placement": "host_managed",
                        "application_visible_h2d_bytes": 2,
                        "application_visible_d2h_bytes": 3,
                        "application_visible_transfer_bytes": 5,
                        "timing": {
                            "h2d_time_s": 0.01,
                            "kernel_time_s": 0.02,
                            "d2h_time_s": 0.03,
                            "host_quantization_time_s": 0.04,
                            "host_dequantization_time_s": 0.05,
                        },
                        # The physical engine currently exposes host conversion
                        # times both directly and in its nested timing object.
                        # Aggregation must count each stage once.
                        "host_quantization_time_s": 0.04,
                        "host_dequantization_time_s": 0.05,
                        "request_level_speedup_applicable": False,
                        "request_timing_is_bringup_only": True,
                    },
                )

            def close(self):
                inner.close()
                return {
                    "target_observed": self_owner.target_observed,
                    "observed_rank_count": self_owner.observed_rank_count,
                    "allocated_dpu_count": self_owner.allocated_dpu_count,
                    "observed_tasklets_per_dpu": self_owner.observed_tasklets_per_dpu,
                }

        self_owner = self
        return Session()


class _FailingEngine:
    name = "failing"

    def open_session(self, policy: object, topology: DeviceTopology):
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

    manifest = plan_study(tmp_path, study)
    payload = json.loads(manifest.read_text())
    assert len(payload["plans"]) == 2
    assert payload["hardware_opened"] is False
    assert all(item["task_count"] > 0 for item in payload["plans"])


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
        engine_factories={"fake": _FakeEngine},
        route_ids=[route_id],
    )
    overridden_run = run_study(
        tmp_path,
        study,
        engine_factories={"fake": _FakeEngine},
        rank_paths=["/dev/dpu_rank1"],
        route_ids=[route_id],
    )
    assert {row["contraction_plan_hash"] for row in _records(base_run)} == {
        row["contraction_plan_hash"] for row in _records(overridden_run)
    }


def test_run_records_warmups_repeats_and_same_plan_hashes(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
    run_dir = run_study(tmp_path, study, engine_factories={"fake": _FakeEngine})
    records = _records(run_dir)

    assert len(records) == 8  # 2 planners x 2 engines x 2 measured repeats
    assert all(row["repeat_id"] in {0, 1} for row in records)
    assert all(row["status"] == "completed" for row in records)
    assert all(row["complete_task_count"] > 0 and row["exact_once"] for row in records)
    for planner_id in {"greedy", "auto"}:
        selected = [row for row in records if row["planner_id"] == planner_id]
        assert len({row["contraction_plan_hash"] for row in selected}) == 1
        assert len({row["circuit_semantics_hash"] for row in selected}) == 1
        assert len({row["executor_config_hash"] for row in selected}) == 2
    assert (run_dir / "m5_circuit_study_summary.json").exists()
    assert (run_dir / "run_manifest.json").exists()


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
    assert len(records) == 4  # one row per planner and engine/policy combination
    assert all(row["status"] == "unsupported" for row in records)
    assert all(row["failure_stage"] == "preflight_resource_limit" for row in records)
    assert all(row["error"] for row in records)


def test_engine_failure_is_preserved_without_fallback(tmp_path: Path) -> None:
    study = tmp_path / "study.yml"
    _study(study)
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
    assert policy.name == "host_packed_int8_per_task_v1"
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
    assert rows[0]["host_quantization_time_s"] == pytest.approx(0.12)
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
        {"observed_rank_count": 2},
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
    topology = DeviceTopology(
        backend="upmem",
        device_ids=("dpu:0", "dpu:1"),
        tasklets_per_device=1,
    )
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
        DeviceTopology(
            backend="upmem",
            device_ids=("dpu:0",),
            tasklets_per_device=1,
        ),
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
