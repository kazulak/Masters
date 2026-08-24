from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from uuid import uuid4

import pytest

from quantum_bench.evidence import (
    append_sample,
    append_session,
    canonical_json,
    environment_id,
    finalize_artifacts,
    load_artifacts,
    sample_id,
    validation_policy_id,
    write_manifest,
)
from quantum_bench.report import (
    _assert_unique_plot_points,
    _point,
    _plot_grouped_bars,
    report_artifacts,
    verify_artifacts,
)
from quantum_bench.results import Measurement


def _measurement(scope_id: str, total_wall_s: float) -> dict[str, object]:
    value = {field.name: None for field in fields(Measurement)}
    value.update({"scope_id": scope_id, "total_wall_s": total_wall_s})
    return value


def _validation() -> dict[str, object]:
    return {
        "policy_reference_applicable": True,
        "policy_reference_passed": True,
        "full_precision_threshold_applicable": True,
        "full_precision_passed": True,
        "scientific_validation_passed": True,
        "max_abs_error": 0.01,
        "relative_l2_error": 0.02,
    }


def _sample(
    *,
    run_id: str,
    experiment_id: str,
    environment_id: str,
    validation_policy_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None = None,
    index: int,
    total_wall_s: float | None,
    scope_id: str = "steady_execution_v1",
    facts: dict[str, object] | None = None,
    session_instance_id: str | None = None,
    status: str = "success",
) -> dict[str, object]:
    sample: dict[str, object] = {
        "schema_version": "evidence_sample_v1",
        "sample_id": sample_id(
            run_id,
            case_id,
            route_id,
            "measurement",
            index,
            plan_id=plan_id,
        ),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "sample_kind": "measurement",
        "sample_index": index,
        "session_instance_id": session_instance_id,
        "status": status,
        "identities": {
            "problem_id": "1" * 64,
            "tensor_network_structure_id": "2" * 64,
            "logical_plan_id": "3" * 64,
            "physical_plan_id": "4" * 64 if session_instance_id else None,
            "executable_id": "5" * 64 if session_instance_id else None,
            "environment_id": environment_id,
            "validation_policy_id": validation_policy_id,
        },
        "measurement": None,
        "backend_facts": facts or {"backend_id": "numpy_cpu_v1"},
        "numeric_facts": {"numeric_policy": "split_complex_float32_v1"},
        "output_sha256": None,
        "validation": None,
        "failure": None,
    }
    if status == "success":
        assert total_wall_s is not None
        sample["measurement"] = _measurement(scope_id, total_wall_s)
        sample["output_sha256"] = "b" * 64
        sample["validation"] = _validation()
    else:
        sample["failure"] = {"stage": "kernel", "reason": "failed"}
    return sample


def _session(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    instance: str,
    plan_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "evidence_session_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "session_instance_id": instance,
        "session_protocol_id": "upmem_real_tile_v4",
        "open_s": 0.1,
        "session_close_s": 0.1,
        "status": "success",
        "terminal_backend_facts": {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "active_dpus": 2,
        },
        "release_attempted": True,
        "release_succeeded": True,
        "release_verified": True,
        "failure": None,
    }


_ENVIRONMENT = {"host": "test-host", "os": "test-os"}
_VALIDATION_POLICY = {"reference_dtype": "complex128", "atol": 1.0e-5}


def _artifact(
    directory: Path,
    samples: list[dict[str, object]],
    sessions: list[dict[str, object]] | None = None,
    *,
    status: str = "completed",
    source_worktree_dirty: bool = False,
) -> Path:
    sessions = sessions or []
    run_id = str(samples[0]["run_id"])
    experiment_id = str(samples[0]["experiment_id"])
    environment_id = str(samples[0]["identities"]["environment_id"])
    policy_id = str(samples[0]["identities"]["validation_policy_id"])
    manifest = {
        "schema_version": "evidence_manifest_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "environment_id": environment_id,
        "validation_policy_id": policy_id,
        "created_at_utc": "2026-08-24T12:00:00Z",
        "source_commit": "a" * 40,
        "source_worktree_dirty": source_worktree_dirty,
        "configuration": {
            "experiment": {"experiment_id": experiment_id},
            "environment": _ENVIRONMENT,
            "validation_policy": _VALIDATION_POLICY,
        },
        "expected_counts": {
            "warmup": 0,
            "measurement": len(samples),
            "sessions": len(sessions),
        },
        "files": {
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
            "sessions": "sessions.jsonl",
        },
        "status": "running",
    }
    write_manifest(directory / "manifest.json", manifest)
    for sample in samples:
        append_sample(directory / "samples.jsonl", sample)
    for session in sessions:
        append_session(directory / "sessions.jsonl", session)
    if not samples:
        (directory / "samples.jsonl").write_text("", encoding="utf-8")
    if not sessions:
        (directory / "sessions.jsonl").write_text("", encoding="utf-8")
    finalize_artifacts(directory, status=status)
    return directory


def _ids() -> tuple[str, str, str, str]:
    return (
        str(uuid4()),
        "e" * 64,
        environment_id(_ENVIRONMENT),
        validation_policy_id(_VALIDATION_POLICY),
    )


def test_load_rejects_noncanonical_reencoding(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id,
                validation_policy_id=policy_id,
                case_id="case",
                route_id="cpu",
                index=0,
                total_wall_s=1.0,
            )
        ],
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_artifacts(artifact)


def test_load_requires_all_primary_files(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()

    with pytest.raises(ValueError, match="missing required evidence file"):
        load_artifacts(directory)


def test_load_binds_manifest_identity_payloads(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id_value,
                validation_policy_id=policy_id,
                case_id="case",
                route_id="cpu",
                index=0,
                total_wall_s=1.0,
            )
        ],
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["environment"]["host"] = "other-host"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment_id"):
        load_artifacts(artifact)


def test_verify_and_report_aggregate_duplicate_measurements_once(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id,
                validation_policy_id=policy_id,
                case_id="bell",
                route_id="cpu",
                index=index,
                total_wall_s=float(index + 1),
            )
            for index in range(2)
        ],
    )

    assert verify_artifacts(artifact)["success_count"] == 2
    report = report_artifacts(artifact, tmp_path / "report")

    assert report["status"] == "completed"
    assert report["aggregate_count"] == 1
    rows = (
        (tmp_path / "report" / "aggregate.csv").read_text(encoding="utf-8").splitlines()
    )
    assert len(rows) == 2
    assert (tmp_path / "report" / "plots" / "runtime_steady_execution_v1.png").is_file()


def test_report_excludes_sdk_simulator_speedup(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="simulator",
            index=index,
            total_wall_s=0.5,
            facts={
                "backend_id": "upmem_sdk_simulator_v4",
                "target_observed": "sdk_simulator",
                "simulator_kernel_executed": True,
            },
        )
        for index in range(2)
    )
    report = report_artifacts(
        _artifact(tmp_path / "evidence", samples), tmp_path / "report"
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["simulator_execution"] == 1


def test_report_admits_physical_speedup_from_terminal_session_facts(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    session_instance_id = "physical-session"
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=3.0,
        )
        for index in range(2)
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            index=index,
            total_wall_s=1.0,
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=session_instance_id,
        )
        for index in range(2)
    )
    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=session_instance_id,
                )
            ],
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 1
    speedup = (tmp_path / "report" / "speedups.csv").read_text(encoding="utf-8")
    assert ",3.0\n" in speedup
    assert (tmp_path / "report" / "plots" / "physical_speedup_by_case.png").is_file()


def test_report_rejects_scope_mismatch(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    session_instance_id = "physical-session"
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            index=index,
            total_wall_s=1.0,
            scope_id="simulation_end_to_end_v1",
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=session_instance_id,
        )
        for index in range(2)
    )
    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=session_instance_id,
                )
            ],
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["timing_scope_mismatch"] == 1


def test_report_rejects_speedup_without_full_precision_threshold(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    instance = "physical-session"
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id=route,
            index=index,
            total_wall_s=2.0 if route == "cpu" else 1.0,
            facts=(
                {"backend_id": "numpy_cpu_v1"}
                if route == "cpu"
                else {"backend_id": "upmem_sdk_hardware_v4"}
            ),
            session_instance_id=instance if route == "upmem" else None,
        )
        for route in ("cpu", "upmem")
        for index in range(2)
    ]
    for sample in samples:
        sample["numeric_facts"] = {
            "numeric_policy": "split_complex_int8_shared_scale_v1"
        }
        sample["validation"] = {
            **_validation(),
            "full_precision_threshold_applicable": False,
            "full_precision_passed": None,
        }

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=instance,
                )
            ],
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["full_precision_threshold_not_passed"] == 1


def test_report_rejects_claims_from_failed_or_dirty_artifacts(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]

    failed = report_artifacts(
        _artifact(tmp_path / "failed", samples, status="failed"),
        tmp_path / "failed-report",
    )
    dirty = report_artifacts(
        _artifact(
            tmp_path / "dirty",
            samples,
            source_worktree_dirty=True,
        ),
        tmp_path / "dirty-report",
    )

    assert failed["speedup_rejections"] == {"artifact_not_completed": 1}
    assert dirty["speedup_rejections"] == {"source_worktree_dirty": 1}


def test_report_rejects_conflicting_terminal_physical_facts(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    instance = "physical-session"
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            index=index,
            total_wall_s=1.0,
            facts={
                "backend_id": "upmem_sdk_hardware_v4",
                "target_observed": "physical_hardware",
            },
            session_instance_id=instance,
        )
        for index in range(2)
    )
    session = _session(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id="bell",
        route_id="upmem",
        instance=instance,
    )
    session["terminal_backend_facts"]["target_observed"] = "sdk_simulator"

    report = report_artifacts(
        _artifact(tmp_path / "evidence", samples, [session]),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["terminal_fact_conflict"] == 1


def test_failed_artifact_generates_diagnostic_report(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id,
                validation_policy_id=policy_id,
                case_id="bell",
                route_id="cpu",
                index=0,
                total_wall_s=None,
                status="failed",
            )
        ],
        status="failed",
    )

    report = report_artifacts(artifact, tmp_path / "report")

    assert report["failed_count"] == 1
    assert report["aggregate_count"] == 0
    assert (tmp_path / "report" / "plots").is_dir()


def test_plot_key_uniqueness_and_png_smoke(tmp_path: Path) -> None:
    point = {
        "figure_id": "figure",
        "facet_id": "facet",
        "series_id": "series",
        "series_label": "Series",
        "x_value": "case",
        "x_label": "Case",
        "value": 1.0,
    }
    with pytest.raises(ValueError, match="duplicate plot point"):
        _assert_unique_plot_points([point, dict(point)])

    _plot_grouped_bars(
        tmp_path / "plot.png",
        [point],
        title="Smoke",
        ylabel="Value",
    )
    assert (tmp_path / "plot.png").is_file()


def test_plot_series_uses_readable_plan_dimension() -> None:
    base = {
        "route_id": "upmem",
        "numeric_policy": "split_complex_float32_v1",
        "case_id": "circuit",
    }
    first = _point(
        figure_id="runtime",
        facet_id="steady",
        row={**base, "plan_id": "greedy"},
        value=1.0,
    )
    second = _point(
        figure_id="runtime",
        facet_id="steady",
        row={**base, "plan_id": "cotengra"},
        value=2.0,
    )

    _assert_unique_plot_points([first, second])
    assert first["series_id"] != second["series_id"]
    assert first["series_label"].startswith("Greedy |")
