from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tarfile

import numpy as np
import pytest
import yaml

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_m7b.py"
PHYSICAL_SCRIPT = ROOT / "scripts" / "qualify_m7c_physical.py"
SELECTION_SCRIPT = ROOT / "scripts" / "select_m7c_workload.py"
SCALING_SCRIPT = ROOT / "scripts" / "run_m7c_scaling_campaign.py"
PARALLEL_SCALING_SCRIPT = ROOT / "scripts" / "inspect_parallel_scaling.py"
M7C_QUALIFIER_SCRIPT = ROOT / "scripts" / "qualify_m7c.py"
ATTRIBUTION_SCRIPT = ROOT / "scripts" / "analyze_m7d_attribution.py"
CONFORMANCE_SCRIPT = ROOT / "scripts" / "check_sequential_conformance.py"
SEQUENTIAL_BASELINE_SCRIPT = ROOT / "scripts" / "qualify_sequential_baseline.py"
BOOTSTRAP_SCRIPT = ROOT / "scripts" / "bootstrap_env.py"


def _load_script(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualifier():
    return _load_script(SCRIPT, "qualify_m7b")


def _physical_qualifier():
    return _load_script(PHYSICAL_SCRIPT, "qualify_m7c_physical")


def _selector():
    return _load_script(SELECTION_SCRIPT, "select_m7c_workload")


def _scaling_campaign():
    return _load_script(SCALING_SCRIPT, "run_m7c_scaling_campaign")


def _parallel_scaling():
    return _load_script(PARALLEL_SCALING_SCRIPT, "inspect_parallel_scaling")


def _m7c_qualifier():
    return _load_script(M7C_QUALIFIER_SCRIPT, "qualify_m7c")


def _attribution():
    return _load_script(ATTRIBUTION_SCRIPT, "analyze_m7d_attribution")


def _conformance():
    return _load_script(CONFORMANCE_SCRIPT, "check_sequential_conformance")


def _sequential_baseline():
    return _load_script(SEQUENTIAL_BASELINE_SCRIPT, "qualify_sequential_baseline")


def _bootstrap():
    return _load_script(BOOTSTRAP_SCRIPT, "bootstrap_env")


def test_sequential_conformance_covers_exact_fixtures_and_oracle_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conformance = _conformance()
    commit = conformance._git_output("rev-parse", "HEAD")
    monkeypatch.setattr(conformance, "_source_state", lambda: (commit, False))
    artifact = conformance.run_conformance()

    assert artifact["passed"] is True
    assert [fixture["fixture_id"] for fixture in artifact["fixtures"]] == [
        "basis_order_2q",
        "Bell2",
        "complex_orientation_3q",
        "GHZ5",
        "QuEST-compatible QRNG3",
        "QuEST-compatible BV5",
        "Stress18",
        "sliced Stress4",
    ]
    quest_fixtures = {
        fixture["fixture_id"]
        for fixture in artifact["fixtures"]
        if "quest_cpu" in fixture["oracles"]
    }
    assert quest_fixtures == {"QuEST-compatible QRNG3", "QuEST-compatible BV5"}
    assert all(
        "direct_quimb_circuit" in fixture["oracles"]
        and "thesis_dag_complex128" in fixture["oracles"]
        for fixture in artifact["fixtures"]
    )
    assert artifact["source_commit"] == commit
    assert artifact["source_worktree_dirty"] is False
    path = tmp_path / "current-conformance.json"
    path.write_text(json.dumps(artifact), encoding="ascii")
    assert _sequential_baseline()._inspect_conformance(path, commit=commit)["passed"]


def test_sequential_conformance_direct_quimb_does_not_use_thesis_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conformance = _conformance()
    monkeypatch.setattr(
        conformance,
        "lower_tensor_network",
        lambda *_args, **_kwargs: pytest.fail("thesis lowering was called"),
    )

    state = conformance._direct_quimb_state(conformance._basis_order_2q())

    np.testing.assert_array_equal(state, np.array([0.0, 1.0, 0.0, 0.0]))


def test_sequential_conformance_phase_aligned_metric_is_diagnostic_only() -> None:
    conformance = _conformance()
    expected = np.array([1.0, 0.0], dtype=np.complex128)

    comparison = conformance._comparison(
        "global_phase",
        1j * expected,
        expected,
        policy="complex128_1e-12",
    )

    assert comparison["phase_aligned_max_abs_error"] <= 1.0e-12
    assert comparison["raw_phase_sensitive_allclose"] is False
    assert comparison["passed"] is False


def test_sequential_baseline_prepare_rewrites_only_machine_paths(tmp_path: Path) -> None:
    qualifier = _sequential_baseline()
    binaries = {}
    for name in ("host", "dpu", "init"):
        path = tmp_path / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        binaries[name] = path

    result = qualifier.prepare_configs(
        output_dir=tmp_path / "runs" / "configs",
        rank_path="/dev/dpu_rank7",
        correctness_session_root=str(tmp_path / "correctness-sessions"),
        performance_session_root=str(tmp_path / "performance-sessions"),
        expected_cpus=[2, 5],
        host_binary=str(binaries["host"]),
        dpu_binary=str(binaries["dpu"]),
        initialization_binary=str(binaries["init"]),
    )

    correctness = load_experiment_config(Path(result["correctness_config"]))
    performance = load_experiment_config(Path(result["performance_config"]))
    for config, session_name in (
        (correctness, "correctness-sessions"),
        (performance, "performance-sessions"),
    ):
        route = config["routes"]["upmem_float32_1dpu_t1"]
        assert route["options"]["rank_paths"] == ("/dev/dpu_rank7",)
        assert route["options"]["session_root"] == str((tmp_path / session_name).resolve())
        assert route["options"]["host_binary"] == str(binaries["host"].resolve())
        assert config["collection"]["machine_policy"]["affinity"] == {
            "mode": "exact_required_v1",
            "expected_cpus": (2, 5),
        }
    assert correctness["collection"]["measurement_blocks"] == 1
    assert performance["collection"]["measurement_blocks"] == 30


def _sequential_validation() -> dict[str, object]:
    return {
        "policy_reference_applicable": True,
        "policy_reference_passed": True,
        "full_precision_threshold_applicable": True,
        "full_precision_passed": True,
        "accuracy_qualified": True,
        "max_abs_error": 1.0e-7,
        "relative_l2_error": 2.0e-7,
        "norm_drift": 3.0e-7,
    }


def _sequential_physical_facts() -> dict[str, object]:
    return {
        "target_observed": "physical_hardware",
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "physical_target_verified": True,
        "hardware_release_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "startup_resource_admission_passed": True,
        "execution_resource_admission_passed": True,
        "rank_count": 1,
        "observed_rank_count": 1,
        "requested_dpus": 1,
        "allocated_dpus": 1,
        "active_dpus": 1,
        "tasklets_per_dpu": 1,
        "host_binary_sha256": "1" * 64,
        "dpu_binary_sha256": "2" * 64,
        "initialization_binary_sha256": "3" * 64,
    }


def _sequential_sample(
    *,
    case_id: str,
    plan_id: str | None,
    route_id: str,
    block_id: int,
    attempt_kind: str,
    scope: str,
    session_id: str | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "block_id": block_id,
        "attempt_kind": attempt_kind,
        "measurement": {
            "scope_id": scope,
            "total_wall_s": (
                1.0 + 0.01 * block_id
                if route_id == "numpy_same_dag"
                else 2.0 + 0.02 * block_id
            ),
        },
        "session_instance_id": session_id,
        "backend_facts": _sequential_physical_facts() if session_id else {"backend_id": route_id},
        "numeric_facts": {"numeric_policy": "split_complex_float32_v1"},
        "identities": {
            "problem_id": "1",
            "tensor_network_structure_id": "2" if plan_id is not None else None,
            "logical_plan_id": "3" if plan_id is not None else None,
            "physical_plan_id": "4" if session_id else None,
            "executable_id": "5" if session_id else None,
            "environment_id": "6",
            "validation_policy_id": "7",
        },
        "output_sha256": "8",
        "observed_affinity": [0],
        "validation": _sequential_validation(),
    }


def _sequential_session(instance: str) -> dict[str, object]:
    return {
        "session_instance_id": instance,
        "release_attempted": True,
        "release_succeeded": True,
        "release_verified": True,
        "terminal_backend_facts": _sequential_physical_facts(),
    }


def _sequential_artifacts(qualifier: object, commit: str) -> dict[str, tuple[object, ...]]:
    def manifest(template: Path, *, preflight: bool = False) -> dict[str, object]:
        config = qualifier._template_configuration(template)
        if template != qualifier.EXTERNAL_TEMPLATE:
            affinity = config["collection"]["machine_policy"]["affinity"]
            affinity.update({"mode": "exact_required_v1", "expected_cpus": [0]})
            for route in config["routes"].values():
                if route["executor"] == "upmem_physical":
                    route["options"].update(
                        {
                            "session_root": "/tmp/sessions",
                            "host_binary": "/tmp/host",
                            "dpu_binary": "/tmp/dpu",
                            "initialization_binary": "/tmp/init",
                            "rank_paths": ["/dev/dpu_rank0"],
                        }
                    )
        return {
            "run_id": template.stem,
            "experiment_id": template.stem + "-identity",
            "validation_policy_id": "validation-policy",
            "source_commit": commit,
            "source_worktree_dirty": False,
            "configuration": {
                "experiment": {
                    "experiment_identity_payload": {
                        "configuration": config,
                        "validation_policy_id": "validation-policy",
                    }
                },
                "environment": {
                    "affinity": [0],
                    "selected_cpu_ids": [0],
                    "observed_cpu_governors": {"0": "powersave"},
                    "thread_environment": {
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "NUMEXPR_NUM_THREADS": "1",
                    },
                    "machine_preflight": {
                        "machine_preflight_passed": preflight,
                        "observed_affinity": [0],
                        "selected_cpu_ids": [0],
                        "observed_cpu_governors": {"0": "powersave"},
                    },
                },
            },
        }

    correct_routes = (
        ("bell2", "unsliced"),
        ("stress4", "unsliced"),
        ("stress4", "sliced"),
    )
    correct_samples = tuple(
        _sequential_sample(
            case_id=case,
            plan_id=plan,
            route_id="upmem_float32_1dpu_t1",
            block_id=index,
            attempt_kind="measurement",
            scope="steady_execution_v1",
            session_id=f"correct-{index}",
        )
        for index, (case, plan) in enumerate(correct_routes)
    )
    correct_sessions = tuple(_sequential_session(f"correct-{index}") for index in range(3))
    performance_samples = []
    performance_sessions = []
    for block in range(32):
        kind = "warmup" if block < 2 else "measurement"
        performance_samples.append(
            _sequential_sample(
                case_id="stress18",
                plan_id="greedy",
                route_id="numpy_same_dag",
                block_id=block,
                attempt_kind=kind,
                scope="steady_execution_v1",
            )
        )
        performance_samples.append(
            _sequential_sample(
                case_id="stress18",
                plan_id="greedy",
                route_id="upmem_float32_1dpu_t1",
                block_id=block,
                attempt_kind=kind,
                scope="steady_execution_v1",
                session_id=f"performance-{block}",
            )
        )
        performance_sessions.append(_sequential_session(f"performance-{block}"))
    external_samples = tuple(
        _sequential_sample(
            case_id="stress18",
            plan_id=None,
            route_id=route,
            block_id=block,
            attempt_kind="warmup" if block == 0 else "measurement",
            scope="simulation_end_to_end_v1",
        )
        for block in range(6)
        for route in ("quimb_greedy", "quimb_cotengra_path")
    )
    return {
        "correctness": (
            manifest(qualifier.CORRECTNESS_TEMPLATE),
            correct_samples,
            correct_sessions,
            {
                "status": "completed",
                "sample_count": 3,
                "session_count": 3,
                "success_count": 3,
                "failed_count": 0,
                "unsupported_count": 0,
                "timing_scopes": ["steady_execution_v1"],
            },
        ),
        "performance": (
            manifest(qualifier.PERFORMANCE_TEMPLATE),
            tuple(performance_samples),
            tuple(performance_sessions),
            {
                "status": "completed",
                "sample_count": 64,
                "session_count": 32,
                "success_count": 64,
                "failed_count": 0,
                "unsupported_count": 0,
                "timing_scopes": ["steady_execution_v1"],
            },
        ),
        "external": (
            manifest(qualifier.EXTERNAL_TEMPLATE),
            external_samples,
            (),
            {
                "status": "completed",
                "sample_count": 12,
                "session_count": 0,
                "success_count": 12,
                "failed_count": 0,
                "unsupported_count": 0,
                "timing_scopes": ["simulation_end_to_end_v1"],
            },
        ),
    }


def _install_sequential_artifacts(
    qualifier: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifacts: dict[str, tuple[object, ...]],
    commit: str,
) -> tuple[Path, Path, Path, Path]:
    paths = tuple(tmp_path / name for name in ("correctness", "performance", "external"))
    for path in paths:
        path.mkdir()
        (path / "manifest.json").write_text("{}\n", encoding="ascii")
    conformance = tmp_path / "conformance.json"
    conformance.write_text(
        json.dumps(_sequential_conformance_artifact(qualifier, commit)),
        encoding="ascii",
    )
    by_path = dict(zip(paths, (artifacts["correctness"], artifacts["performance"], artifacts["external"])))
    monkeypatch.setattr(qualifier, "_require_clean_source", lambda: commit)
    monkeypatch.setattr(qualifier, "load_artifacts", lambda path: by_path[Path(path)][:3])
    monkeypatch.setattr(qualifier, "verify_artifacts", lambda path: by_path[Path(path)][3])
    return conformance, *paths


def _sequential_conformance_artifact(
    qualifier: object, commit: str
) -> dict[str, object]:
    analytic = {
        "basis_order_2q",
        "Bell2",
        "GHZ5",
        "QuEST-compatible QRNG3",
        "QuEST-compatible BV5",
    }
    quest = {"QuEST-compatible QRNG3", "QuEST-compatible BV5"}
    float32 = {"Stress18", "sliced Stress4"}
    fixtures = []
    for fixture_id in sorted(qualifier.REQUIRED_FIXTURES):
        oracles = ["direct_quimb_circuit", "thesis_dag_complex128"]
        comparisons = [
            ("thesis_dag_complex128_vs_direct_quimb", "complex128_1e-12")
        ]
        if fixture_id in analytic:
            oracles.append("analytic")
            comparisons.extend(
                [
                    ("direct_quimb_vs_analytic", "complex128_1e-12"),
                    ("thesis_dag_complex128_vs_analytic", "complex128_1e-12"),
                ]
            )
        if fixture_id in quest:
            oracles.append("quest_cpu")
            comparisons.extend(
                [
                    ("quest_cpu_vs_direct_quimb", "complex128_1e-12"),
                    ("quest_cpu_vs_thesis_dag_complex128", "complex128_1e-12"),
                ]
            )
        if fixture_id in float32:
            oracles.append("thesis_dag_float32")
            comparisons.append(
                ("thesis_dag_float32_vs_direct_quimb", "float32_1e-5")
            )
        fixture: dict[str, object] = {
            "fixture_id": fixture_id,
            "oracles": oracles,
            "comparisons": [
                {
                    "comparison": name,
                    "policy": policy,
                    "raw_phase_sensitive_allclose": True,
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                    "norm_drift": 0.0,
                    "phase_aligned_max_abs_error": 0.0,
                    "passed": True,
                }
                for name, policy in comparisons
            ],
            "passed": True,
        }
        if fixture_id == "sliced Stress4":
            fixture["slicing"] = {
                "node_id": "contract_24",
                "minimum_slice_count": 4,
                "sdk_simulator_coverage": (
                    "tests/test_cli_report.py::"
                    "test_sliced_conformance_retains_strict_sdk_simulator_coverage"
                ),
            }
        fixtures.append(fixture)
    return {
        "schema_version": qualifier.CONFORMANCE_SCHEMA,
        "source_commit": commit,
        "source_worktree_dirty": False,
        "execution_class": "software_only",
        "phase_aligned_metric_is_diagnostic_only": True,
        "policies": json.loads(json.dumps(qualifier.CONFORMANCE_POLICIES)),
        "fixtures": fixtures,
        "passed": True,
    }


@pytest.mark.parametrize(
    "drift",
    (
        "wrong_commit",
        "dirty_source",
        "changed_policy",
        "changed_phase_contract",
        "missing_oracle",
        "added_quest_oracle",
        "changed_comparison",
        "changed_comparison_policy",
        "raw_false_passed_true",
        "max_abs_error",
        "relative_l2_error",
        "norm_drift",
        "slicing",
    ),
)
def test_sequential_conformance_inspector_rejects_contract_drift(
    tmp_path: Path, drift: str
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifact = _sequential_conformance_artifact(qualifier, commit)
    fixtures = artifact["fixtures"]
    basis = next(row for row in fixtures if row["fixture_id"] == "basis_order_2q")
    stress = next(row for row in fixtures if row["fixture_id"] == "Stress18")
    sliced = next(row for row in fixtures if row["fixture_id"] == "sliced Stress4")
    float_comparison = next(
        row for row in stress["comparisons"] if row["policy"] == "float32_1e-5"
    )
    if drift == "wrong_commit":
        artifact["source_commit"] = "b" * 40
    elif drift == "dirty_source":
        artifact["source_worktree_dirty"] = True
    elif drift == "changed_policy":
        artifact["policies"]["float32_1e-5"]["norm_drift_max"] = 3.0e-5
    elif drift == "changed_phase_contract":
        artifact["phase_aligned_metric_is_diagnostic_only"] = False
    elif drift == "missing_oracle":
        basis["oracles"].pop()
    elif drift == "added_quest_oracle":
        basis["oracles"].append("quest_cpu")
    elif drift == "changed_comparison":
        basis["comparisons"][0]["comparison"] = "renamed"
    elif drift == "changed_comparison_policy":
        basis["comparisons"][0]["policy"] = "float32_1e-5"
    elif drift == "raw_false_passed_true":
        basis["comparisons"][0]["raw_phase_sensitive_allclose"] = False
        basis["comparisons"][0]["passed"] = True
    elif drift in {"max_abs_error", "relative_l2_error"}:
        float_comparison[drift] = 2.0e-5
    elif drift == "norm_drift":
        float_comparison[drift] = 3.0e-5
    else:
        sliced["slicing"]["minimum_slice_count"] = 8
    path = tmp_path / f"conformance-{drift}.json"
    path.write_text(json.dumps(artifact), encoding="ascii")

    with pytest.raises(ValueError):
        qualifier._inspect_conformance(path, commit=commit)


def test_sequential_conformance_phase_aligned_metric_remains_diagnostic(
    tmp_path: Path,
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifact = _sequential_conformance_artifact(qualifier, commit)
    artifact["fixtures"][0]["comparisons"][0][
        "phase_aligned_max_abs_error"
    ] = 99.0
    path = tmp_path / "conformance-phase-diagnostic.json"
    path.write_text(json.dumps(artifact), encoding="ascii")

    assert qualifier._inspect_conformance(path, commit=commit)["passed"] is True


def test_sequential_baseline_inspects_exact_four_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    summary = qualifier.inspect_baseline(
        conformance=conformance,
        correctness=correctness,
        performance=performance,
        external_context=external,
    )

    assert summary["schema_version"] == qualifier.SUMMARY_SCHEMA
    assert summary["source_commit"] == commit
    assert summary["t1_binary_hashes"] == {
        "host_binary_sha256": "1" * 64,
        "dpu_binary_sha256": "2" * 64,
        "initialization_binary_sha256": "3" * 64,
    }
    assert set(summary["inputs"]) == {
        "conformance",
        "physical_correctness",
        "physical_performance",
        "external_tn_context",
    }
    assert summary["artifact_statistics"] == "separate_no_cross_artifact_statistics_v1"
    assert summary["claim_eligible"] is False
    assert summary["claim_ineligibility_reason"] == "powersave_conditioned_diagnostic_v1"
    diagnostic = summary["powersave_diagnostic_statistics"]
    assert diagnostic["measurement_only"] is True
    assert diagnostic["resample_count"] == 10_000
    assert diagnostic["routes"]["numpy_same_dag"] == {
        "measurement_count": 30,
        "median_total_wall_s": pytest.approx(1.165),
        "raw_mad_total_wall_s": pytest.approx(0.075),
    }
    assert diagnostic["routes"]["upmem_float32_1dpu_t1"] == {
        "measurement_count": 30,
        "median_total_wall_s": pytest.approx(2.33),
        "raw_mad_total_wall_s": pytest.approx(0.15),
    }
    assert diagnostic["numpy_to_upmem_control_ratio"]["point_estimate"] == pytest.approx(0.5)
    assert len(diagnostic["numpy_to_upmem_control_ratio"]["confidence_interval_95"]) == 2


def test_sequential_baseline_diagnostic_statistics_are_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    first = qualifier.inspect_baseline(
        conformance=conformance,
        correctness=correctness,
        performance=performance,
        external_context=external,
    )
    second = qualifier.inspect_baseline(
        conformance=conformance,
        correctness=correctness,
        performance=performance,
        external_context=external,
    )

    assert first["powersave_diagnostic_statistics"] == second["powersave_diagnostic_statistics"]


@pytest.mark.parametrize(
    "drift",
    (
        "policy",
        "governor",
        "affinity",
        "sample_affinity",
        "thread",
        "relative_l2",
        "norm_drift",
    ),
)
def test_sequential_baseline_rejects_powersave_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    performance_manifest = artifacts["performance"][0]
    environment = performance_manifest["configuration"]["environment"]
    if drift == "policy":
        performance_manifest["configuration"]["experiment"]["experiment_identity_payload"][
            "configuration"
        ]["collection"]["claim_policy"] = "physical_performance_v1"
    elif drift == "governor":
        environment["machine_preflight"]["observed_cpu_governors"] = {"0": "performance"}
    elif drift == "affinity":
        environment["machine_preflight"]["observed_affinity"] = [1]
    elif drift == "sample_affinity":
        artifacts["performance"][1][0]["observed_affinity"] = [1]
    elif drift == "relative_l2":
        artifacts["performance"][1][0]["validation"]["relative_l2_error"] = 2.0e-5
    elif drift == "norm_drift":
        artifacts["performance"][1][0]["validation"]["norm_drift"] = 3.0e-5
    else:
        environment["thread_environment"]["MKL_NUM_THREADS"] = "2"
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    with pytest.raises(ValueError):
        qualifier.inspect_baseline(
            conformance=conformance,
            correctness=correctness,
            performance=performance,
            external_context=external,
        )


@pytest.mark.parametrize("drift", ("count", "route", "source", "scope", "provenance"))
def test_sequential_baseline_rejects_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    if drift == "count":
        artifacts["performance"][3]["sample_count"] = 63
    elif drift == "route":
        artifacts["external"][1][0]["route_id"] = "unexpected"
    elif drift == "source":
        artifacts["correctness"][0]["source_commit"] = "b" * 40
    elif drift == "scope":
        artifacts["performance"][3]["timing_scopes"] = ["simulation_end_to_end_v1"]
    else:
        artifacts["correctness"][1][0]["backend_facts"]["cpu_fallback_used"] = True
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    with pytest.raises(ValueError):
        qualifier.inspect_baseline(
            conformance=conformance,
            correctness=correctness,
            performance=performance,
            external_context=external,
        )


def test_sequential_baseline_rejects_one_mutated_session_binary_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    artifacts["correctness"][2][0]["terminal_backend_facts"][
        "host_binary_sha256"
    ] = "f" * 64
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    with pytest.raises(ValueError, match="consistent T1 binary triple"):
        qualifier.inspect_baseline(
            conformance=conformance,
            correctness=correctness,
            performance=performance,
            external_context=external,
        )


def test_sequential_baseline_rejects_binary_hashes_different_across_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    commit = "a" * 40
    artifacts = _sequential_artifacts(qualifier, commit)
    for session in artifacts["performance"][2]:
        session["terminal_backend_facts"]["host_binary_sha256"] = "f" * 64
    conformance, correctness, performance, external = _install_sequential_artifacts(
        qualifier, monkeypatch, tmp_path, artifacts, commit
    )

    with pytest.raises(ValueError, match="different T1 binaries"):
        qualifier.inspect_baseline(
            conformance=conformance,
            correctness=correctness,
            performance=performance,
            external_context=external,
        )


def test_sequential_baseline_bundle_is_closed_and_self_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    binaries = {}
    for name in ("host", "dpu", "init"):
        path = tmp_path / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        binaries[name] = path
    prepared = qualifier.prepare_configs(
        output_dir=tmp_path / "configs",
        rank_path="/dev/dpu_rank0",
        correctness_session_root=str(tmp_path / "correctness-sessions"),
        performance_session_root=str(tmp_path / "performance-sessions"),
        expected_cpus=[0],
        host_binary=str(binaries["host"]),
        dpu_binary=str(binaries["dpu"]),
        initialization_binary=str(binaries["init"]),
    )
    correctness_config = Path(prepared["correctness_config"])
    performance_config = Path(prepared["performance_config"])
    performance_document = yaml.safe_load(performance_config.read_text(encoding="utf-8"))
    performance_options = performance_document["routes"]["upmem_float32_1dpu_t1"][
        "options"
    ]
    for field, name in (
        ("host_binary", "host"),
        ("dpu_binary", "dpu"),
        ("initialization_binary", "init"),
    ):
        path = tmp_path / "performance-bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(binaries[name].read_bytes())
        performance_options[field] = str(path)
    performance_config.write_text(
        yaml.safe_dump(performance_document, sort_keys=False), encoding="utf-8"
    )
    binary_hashes = {
        "host_binary_sha256": qualifier._sha256(binaries["host"]),
        "dpu_binary_sha256": qualifier._sha256(binaries["dpu"]),
        "initialization_binary_sha256": qualifier._sha256(binaries["init"]),
    }
    summary = {
        "schema_version": qualifier.SUMMARY_SCHEMA,
        "status": "qualified",
        "source_commit": "a" * 40,
        "t1_binary_hashes": binary_hashes,
        "inputs": {
            "conformance": {"sha256": "1" * 64},
            "physical_correctness": {
                "artifact_sha256": "2" * 64,
                "run_id": "correct-run",
                "experiment_id": load_experiment_config(correctness_config)["experiment_id"],
            },
            "physical_performance": {
                "artifact_sha256": "3" * 64,
                "run_id": "perf-run",
                "experiment_id": load_experiment_config(performance_config)["experiment_id"],
            },
            "external_tn_context": {
                "artifact_sha256": "4" * 64,
                "run_id": "external-run",
                "experiment_id": "external-experiment",
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="ascii")
    monkeypatch.setattr(qualifier, "inspect_baseline", lambda **_kwargs: summary)
    evidence = {}
    reports = {}
    for label, record in (
        ("correctness", summary["inputs"]["physical_correctness"]),
        ("performance", summary["inputs"]["physical_performance"]),
        ("external", summary["inputs"]["external_tn_context"]),
    ):
        evidence[label] = tmp_path / "evidence" / label
        evidence[label].mkdir(parents=True)
        for filename in ("manifest.json", "samples.jsonl", "sessions.jsonl"):
            (evidence[label] / filename).write_text(filename + "\n", encoding="ascii")
        reports[label] = tmp_path / "reports" / label
        reports[label].mkdir(parents=True)
        (reports[label] / "report.json").write_text(
            json.dumps(
                {
                    "schema_version": "evidence_report_v5",
                    "status": "completed",
                    "run_id": record["run_id"],
                    "experiment_id": record["experiment_id"],
                }
            ),
            encoding="ascii",
        )
    conformance = tmp_path / "conformance.json"
    conformance.write_text("{}\n", encoding="ascii")
    output = tmp_path / "sequential-baseline-v1"

    result = qualifier.bundle_baseline(
        summary_path=summary_path,
        conformance=conformance,
        correctness=evidence["correctness"],
        performance=evidence["performance"],
        external_context=evidence["external"],
        correctness_config=correctness_config,
        performance_config=performance_config,
        correctness_report=reports["correctness"],
        performance_report=reports["performance"],
        external_context_report=reports["external"],
        output=output,
    )

    qualifier.verify_internal_hashes(output)
    assert Path(result["archive"]).is_file()
    assert Path(result["outer_checksum"]).read_text(encoding="ascii").split()[0] == qualifier._sha256(Path(result["archive"]))
    checksummed = {
        line.split(maxsplit=1)[1]
        for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    }
    actual = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert checksummed == actual
    assert not any(path.name in {"host", "dpu", "init"} for path in output.rglob("*"))
    assert json.loads(
        (output / "provenance" / "t1-binary-hashes.json").read_text(
            encoding="ascii"
        )
    ) == binary_hashes
    correct_paths = qualifier._config_binaries(
        correctness_config, summary["inputs"]["physical_correctness"]["experiment_id"]
    )
    performance_paths = qualifier._config_binaries(
        performance_config, summary["inputs"]["physical_performance"]["experiment_id"]
    )
    assert correct_paths != performance_paths
    with tarfile.open(result["archive"], "r:gz") as archive:
        assert all(member.name == output.name or member.name.startswith(output.name + "/") for member in archive.getmembers())


def test_sequential_baseline_bundle_rejects_replaced_configured_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualifier = _sequential_baseline()
    binaries = {}
    for name in ("host", "dpu", "init"):
        path = tmp_path / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        binaries[name] = path
    prepared = qualifier.prepare_configs(
        output_dir=tmp_path / "configs",
        rank_path="/dev/dpu_rank0",
        correctness_session_root=str(tmp_path / "correctness-sessions"),
        performance_session_root=str(tmp_path / "performance-sessions"),
        expected_cpus=[0],
        host_binary=str(binaries["host"]),
        dpu_binary=str(binaries["dpu"]),
        initialization_binary=str(binaries["init"]),
    )
    correctness_config = Path(prepared["correctness_config"])
    performance_config = Path(prepared["performance_config"])
    summary = {
        "schema_version": qualifier.SUMMARY_SCHEMA,
        "status": "qualified",
        "source_commit": "a" * 40,
        "t1_binary_hashes": {
            "host_binary_sha256": qualifier._sha256(binaries["host"]),
            "dpu_binary_sha256": qualifier._sha256(binaries["dpu"]),
            "initialization_binary_sha256": qualifier._sha256(binaries["init"]),
        },
        "inputs": {
            "physical_correctness": {
                "experiment_id": load_experiment_config(correctness_config)[
                    "experiment_id"
                ]
            },
            "physical_performance": {
                "experiment_id": load_experiment_config(performance_config)[
                    "experiment_id"
                ]
            },
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="ascii")
    monkeypatch.setattr(qualifier, "inspect_baseline", lambda **_kwargs: summary)
    binaries["host"].write_bytes(b"replaced-after-collection")

    with pytest.raises(ValueError, match="execution-time hash"):
        qualifier.bundle_baseline(
            summary_path=summary_path,
            conformance=tmp_path / "conformance.json",
            correctness=tmp_path / "correctness",
            performance=tmp_path / "performance",
            external_context=tmp_path / "external",
            correctness_config=correctness_config,
            performance_config=performance_config,
            correctness_report=tmp_path / "correctness-report",
            performance_report=tmp_path / "performance-report",
            external_context_report=tmp_path / "external-report",
            output=tmp_path / "bundle",
        )


def test_bootstrap_detects_repository_root_gitmodules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = _bootstrap()
    repository = tmp_path / "repository"
    implementation = repository / "thesis" / "implementation"
    venv = repository / "thesis" / ".venv"
    (repository / ".git").mkdir(parents=True)
    (repository / ".gitmodules").write_text("[submodule \"x\"]\n", encoding="ascii")
    (implementation / "ci").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []
    monkeypatch.setattr(bootstrap.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap, "_python_version", lambda _python: (3, 10))
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs["cwd"])) or SimpleNamespace(returncode=0),
    )

    bootstrap.bootstrap(root=implementation, venv=venv)

    assert calls[-1] == (["git", "submodule", "update", "--init", "--recursive"], repository)
    assert calls[0][1] == implementation


def _archive(path: Path, member_name: str, *, kind: str = "file") -> None:
    source = path.parent / "payload.txt"
    source.write_text("payload\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        if kind == "file":
            bundle.add(source, arcname=member_name)
            return
        member = tarfile.TarInfo(member_name)
        member.type = tarfile.SYMTYPE
        member.linkname = "payload.txt"
        bundle.addfile(member)


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_extracts_regular_relative_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, "evidence/manifest.json")

    _qualifier()._safe_extract_tar(archive, tmp_path / "output")

    assert (tmp_path / "output" / "evidence" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "payload\n"


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_m7c_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _m7c_qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_verifies_bundled_hashes_and_records_external_provenance(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "m7a-bundle"
    evidence = bundle / "cpu-run" / "manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (bundle / "SHA256SUMS").write_text(
        f"{digest}  runs/{bundle.name}/cpu-run/manifest.json\n"
        + "0" * 64
        + "  native/upmem/runtime/bin/dpu\n",
        encoding="utf-8",
    )

    external = _qualifier()._verify_internal_hashes(tmp_path)

    assert external == ("native/upmem/runtime/bin/dpu",)


def test_physical_preparation_preserves_template_resolved_paths(tmp_path: Path) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_smoke.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "smoke.yml"
    prepared = _physical_qualifier().prepare_config(
        template=template,
        output=output,
        mode="float32-smoke",
        rank_path="/dev/dpu_rank42",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[2, 4],
    )

    assert prepared == output
    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    source_options = source["routes"]["upmem_float32_1dpu"]["options"]
    copied_options = copied["routes"]["upmem_float32_1dpu"]["options"]
    for field in ("host_binary", "dpu_binary", "initialization_binary"):
        assert copied_options[field] == source_options[field]
    assert copied_options["session_root"] == str((tmp_path / "sessions").resolve())
    assert copied_options["rank_paths"] == ("/dev/dpu_rank42",)
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (2, 4),
    }


def test_physical_preparation_probe_has_one_measurement(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "probe.yml"
    _physical_qualifier().prepare_config(
        template=ROOT / "configs" / "tn_benchmark_physical_smoke.yml",
        output=output,
        mode="probe",
        rank_path="/dev/dpu_rank0",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[0],
    )

    config = load_experiment_config(output)
    assert config["collection"]["warmup_blocks"] == 0
    assert config["collection"]["measurement_blocks"] == 1
    assert config["routes"]["upmem_float32_1dpu"]["numeric_policy"] == (
        "split_complex_float32_v1"
    )


def test_m7c_workload_selection_is_deterministic_and_preregistered(
    tmp_path: Path,
) -> None:
    selector = _selector()
    first = selector.build_selection()
    second = selector.build_selection()

    assert first == second
    assert first["selected_primary"] == "quantization_stress_18q_l2"
    assert first["selected_secondary"] == "ghz_chain_18q"
    assert first["schema_version"] == "m7c_workload_selection_v2"
    assert first["dependency_constraints_sha256"] == hashlib.sha256(
        (ROOT / "ci" / "constraints.txt").read_bytes()
    ).hexdigest()
    assert first["selection_basis_sha256"] == selector._hash(
        {
            key: first[key]
            for key in (
                "schema_version",
                "planner_configuration",
                "selection_rule",
                "candidates",
                "selected_primary",
                "selected_secondary",
            )
        }
    )
    assert first["planner_configuration_sha256"] == selector._hash(
        selector.PLANNER_CONFIG
    )
    assert "constraints_hash" not in first
    assert "python_version" not in first
    primary = next(
        candidate
        for candidate in first["candidates"]
        if candidate["candidate_id"] == first["selected_primary"]
    )
    assert primary["logical_plan_id"] == (
        "d504919e20d95bac608dd906d46abb122f9680873679710b0584e71981648fb5"
    )
    assert primary["topologies"]["dpu4_tasklet8"][
        "collection_resource_admission_passed"
    ] is True

    path = tmp_path / "selection.json"
    selector.write_selection(path)
    selector.check_selection(path)


def test_m7c_workload_selection_rejects_nonancestor_source(tmp_path: Path) -> None:
    selector = _selector()
    selection = selector.build_selection()
    selection["source_commit"] = "0" * 40
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="not an ancestor"):
        selector.check_selection(path)


def test_m7c_workload_selection_rejects_route_matrix_drift(tmp_path: Path) -> None:
    selector = _selector()
    config = yaml.safe_load(
        (ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml").read_text(
            encoding="utf-8"
        )
    )
    config["routes"]["upmem_float32_2dpu_t8"]["options"]["dpu_count"] = 3
    drifted = tmp_path / "drifted.yml"
    drifted.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="topology drift"):
        selector.check_selection(
            ROOT / "configs" / "m7c_workload_selection.json", drifted
        )


def test_m7c_committed_selection_matches_scaling_config() -> None:
    selector = _selector()
    selection = ROOT / "configs" / "m7c_workload_selection.json"
    assert "python_version" not in json.loads(selection.read_text(encoding="utf-8"))
    for config in (
        "tn_benchmark_physical_scaling_diagnostic.yml",
        "tn_benchmark_physical_scaling.yml",
        "tn_benchmark_physical_scaling_confirmation.yml",
    ):
        selector.check_selection(selection, ROOT / "configs" / config)


def _attribution_sample(operation_timing: dict[str, float]) -> dict[str, object]:
    return {
        "status": "success",
        "attempt_kind": "measurement",
        "route_id": "upmem_float32_4dpu_t8",
        "measurement": {"total_wall_s": 2.0, "host_reduce_s": 0.05},
        "backend_facts": {
            "operation_facts": [
                {
                    "rank_count": 1,
                    "timing": operation_timing,
                }
            ]
        },
    }


def _attribution_operation_timing(
    *, request_build_breakdown: bool = True
) -> dict[str, float]:
    timing = {
        "total_wall_s": 1.8,
        "preparation_s": 0.1,
        "encode_s": 0.1,
        "rank_response_h2d_max_sum_s": 0.1,
        "rank_response_kernel_max_sum_s": 0.2,
        "rank_response_d2h_max_sum_s": 0.1,
        "rank_response_total_route_max_sum_s": 0.6,
        "request_wave_wall_sum_s": 1.2,
        "request_build_sum_s": 0.1,
        "rank_submit_parallel_wall_sum_s": 0.95,
        "rank_submit_total_max_sum_s": 0.9,
        "rank_submit_artifact_validation_max_sum_s": 0.1,
        "rank_submit_protocol_write_max_sum_s": 0.1,
        "rank_submit_response_wait_max_sum_s": 0.3,
        "rank_submit_response_validation_max_sum_s": 0.02,
        "coordinator_response_processing_sum_s": 0.1,
        "assembly_s": 0.1,
        "decode_s": 0.1,
    }
    if request_build_breakdown:
        timing.update(
            {
                "request_work_unit_materialization_sum_s": 0.02,
                "request_artifact_build_sum_s": 0.06,
                "request_payload_record_staging_sum_s": 0.03,
                "request_manifest_sidecar_staging_sum_s": 0.02,
            }
        )
    return timing


def test_m7g_attribution_derives_disjoint_request_build_components() -> None:
    attribution = _attribution()
    manifest = {"source_commit": "a" * 40}
    sample = _attribution_sample(_attribution_operation_timing())

    first = attribution.derive_attribution(manifest, (sample,))
    second = attribution.derive_attribution(manifest, (sample,))

    assert first == second
    route = first["routes"]["upmem_float32_4dpu_t8"]
    assert route["measurement_count"] == 1
    assert route["median_total_wall_s"] == pytest.approx(2.0)
    components = route["components"]
    assert components["host_request_overhead_s"]["median_s"] == pytest.approx(0.6)
    assert components["native_request_overhead_s"]["median_s"] == pytest.approx(0.2)
    assert components["operation_other_s"]["median_s"] == pytest.approx(0.2)
    assert components["coordinator_other_s"]["median_s"] == pytest.approx(0.15)
    assert route["median_unresolved_boundary_s"] == pytest.approx(0.35)
    assert route["median_accounting_residual_s"] == pytest.approx(0.0)
    assert route["nested_request_timing_medians_s"][
        "rank_submit_response_wait_max_sum_s"
    ] == pytest.approx(0.3)
    request_build = route["request_build_breakdown"]
    assert request_build is not None
    assert request_build["median_parent_s"] == pytest.approx(0.1)
    children = request_build["children"]
    assert children["work_unit_materialization_s"]["median_s"] == pytest.approx(0.02)
    assert children["payload_record_staging_s"]["median_s"] == pytest.approx(0.03)
    assert children["manifest_sidecar_staging_s"]["median_s"] == pytest.approx(0.02)
    assert children["artifact_build_residual_s"]["median_s"] == pytest.approx(0.01)
    assert children["request_build_residual_s"]["median_s"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("request_wave_wall_sum_s", None, "missing request_wave_wall_sum_s"),
        ("rank_response_total_route_max_sum_s", 0.3, "native request overhead"),
        (
            "request_payload_record_staging_sum_s",
            0.07,
            "request artifact build residual",
        ),
    ],
)
def test_m7g_attribution_rejects_missing_or_inconsistent_timing(
    field: str, value: float | None, message: str
) -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing()
    if value is None:
        del timing[field]
    else:
        timing[field] = value

    with pytest.raises(ValueError, match=message):
        attribution.derive_attribution(
            {"source_commit": "a" * 40},
            (_attribution_sample(timing),),
        )


def test_m7g_attribution_accepts_m7f_response_wait_and_omits_build_breakdown() -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing(request_build_breakdown=False)

    result = attribution.derive_attribution(
        {"source_commit": "a" * 40},
        (_attribution_sample(timing),),
    )

    route = result["routes"]["upmem_float32_4dpu_t8"]
    assert route["request_build_breakdown"] is None
    assert route["components"]["host_request_overhead_s"]["median_s"] == pytest.approx(
        0.6
    )


def test_m7g_attribution_rejects_partial_request_build_timing() -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing()
    del timing["request_manifest_sidecar_staging_sum_s"]

    with pytest.raises(ValueError, match="request-build timing is missing"):
        attribution.derive_attribution(
            {"source_commit": "a" * 40},
            (_attribution_sample(timing),),
        )


def test_m7c_scaling_preparation_preserves_all_resolved_route_paths(
    tmp_path: Path,
) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "diagnostic.yml"
    _scaling_campaign().prepare_config(
        template=template,
        output=output,
        rank_paths=["/dev/dpu_rank19"],
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[1, 3],
    )

    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    for route_id, route in source["routes"].items():
        if route["executor"] != "upmem_physical":
            continue
        source_options = route["options"]
        copied_options = copied["routes"][route_id]["options"]
        for field in ("host_binary", "dpu_binary", "initialization_binary"):
            assert copied_options[field] == source_options[field]
        assert copied_options["rank_paths"] == ("/dev/dpu_rank19",)
        assert copied_options["session_root"] == str(
            (tmp_path / "sessions" / route_id).resolve()
        )
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (1, 3),
    }


def test_parallel_scaling_config_freezes_physical_route_matrix_and_paths(
    tmp_path: Path,
) -> None:
    template = ROOT / "configs" / "tn_benchmark_parallel_scaling_diagnostic.yml"
    expected_routes = (
        "upmem_float32_1dpu_t1",
        "upmem_float32_1dpu_t2",
        "upmem_float32_1dpu_t4",
        "upmem_float32_1dpu_t8",
        "upmem_float32_2dpu_t8",
        "upmem_float32_4dpu_t8",
    )
    config = load_experiment_config(template)
    assert tuple(config["routes"]) == expected_routes
    assert tuple(config["matrix"][0]["route_ids"]) == expected_routes
    assert config["collection"]["claim_policy"] == "diagnostic_v1"
    assert config["collection"]["warmup_blocks"] == 1
    assert config["collection"]["measurement_blocks"] == 5
    assert all(route["executor"] == "upmem_physical" for route in config["routes"].values())

    output = tmp_path / "runs" / "configs" / "eth" / "parallel.yml"
    _scaling_campaign().prepare_config(
        template=template,
        output=output,
        rank_paths=["/dev/dpu_rank19"],
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[2],
    )
    prepared = load_experiment_config(output)
    for route_id, route in prepared["routes"].items():
        options = route["options"]
        assert options["rank_paths"] == ("/dev/dpu_rank19",)
        assert options["session_root"] == str(
            (tmp_path / "sessions" / route_id).resolve()
        )
        assert Path(options["host_binary"]).is_absolute()
        assert Path(options["dpu_binary"]).is_absolute()
        assert Path(options["initialization_binary"]).is_absolute()


def _parallel_scaling_rows() -> tuple[dict[str, str], ...]:
    comparisons = (
        ("tasklet_scaling", "primary", "1", "2", "1", "1"),
        ("tasklet_scaling", "primary", "1", "4", "1", "1"),
        ("tasklet_scaling", "primary", "1", "8", "1", "1"),
        ("tasklet_scaling", "secondary", "2", "4", "1", "1"),
        ("tasklet_scaling", "secondary", "2", "8", "1", "1"),
        ("tasklet_scaling", "secondary", "4", "8", "1", "1"),
        ("dpu_scaling", "primary", "8", "8", "1", "2"),
        ("dpu_scaling", "primary", "8", "8", "1", "4"),
        ("dpu_scaling", "secondary", "8", "8", "2", "4"),
    )
    return tuple(
        {
            "comparison_kind": kind,
            "comparison_role": role,
            "baseline_tasklet_count": baseline_tasklets,
            "candidate_tasklet_count": candidate_tasklets,
            "baseline_dpu_count": baseline_dpus,
            "candidate_dpu_count": candidate_dpus,
            "planned_pair_count": "5",
            "complete_pair_count": "5",
            "claim_eligible": "False",
            "claim_ineligibility_reason": "baseline_diagnostic_claim_policy",
        }
        for (
            kind,
            role,
            baseline_tasklets,
            candidate_tasklets,
            baseline_dpus,
            candidate_dpus,
        ) in comparisons
    )


def _complete_parallel_diagnostic(
    script: object,
) -> tuple[
    dict[str, object],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_parallel_scaling_diagnostic.yml"
    )
    commit = "a" * 40
    manifest: dict[str, object] = {
        "source_commit": commit,
        "source_worktree_dirty": False,
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "run_id": "00000000-0000-4000-8000-000000000001",
        "configuration": {
            "experiment": config,
            "environment": {
                "affinity": (0,),
                "observed_cpu_governors": {"0": "powersave"},
                "thread_environment": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            },
        },
    }
    times = {
        "upmem_float32_1dpu_t1": (32.0, 24.0),
        "upmem_float32_1dpu_t2": (20.0, 12.0),
        "upmem_float32_1dpu_t4": (13.0, 6.0),
        "upmem_float32_1dpu_t8": (10.0, 3.0),
        "upmem_float32_2dpu_t8": (8.0, 1.6),
        "upmem_float32_4dpu_t8": (7.0, 0.9),
    }
    utilization = {
        "upmem_float32_1dpu_t1": (1.0, 1.0),
        "upmem_float32_1dpu_t2": (0.9998, 1.0),
        "upmem_float32_1dpu_t4": (0.9993, 1.0),
        "upmem_float32_1dpu_t8": (0.9986, 1.0),
        "upmem_float32_2dpu_t8": (0.9986, 0.9904),
        "upmem_float32_4dpu_t8": (0.9986, 0.9856),
    }
    samples: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    for block_id in range(6):
        attempt_kind = "warmup" if block_id == 0 else "measurement"
        sample_index = 0 if block_id == 0 else block_id - 1
        for route_index, (route_id, (dpu_count, tasklets)) in enumerate(
            script._ROUTE_SPECS.items(), start=1
        ):
            session_id = f"{route_id}-{block_id}"
            executable_key = 4 if tasklets == 8 else {1: 1, 2: 2, 4: 3}[tasklets]
            tasklet_utilization, dpu_utilization = utilization[route_id]
            total_wall_s, kernel_s = times[route_id]
            samples.append(
                {
                    "case_id": "scaling_primary",
                    "plan_id": "greedy",
                    "route_id": route_id,
                    "attempt_kind": attempt_kind,
                    "sample_index": sample_index,
                    "block_id": block_id,
                    "status": "success",
                    "session_instance_id": session_id,
                    "identities": {
                        "environment_id": "1" * 64,
                        "problem_id": "2" * 64,
                        "tensor_network_structure_id": "3" * 64,
                        "logical_plan_id": "4" * 64,
                        "validation_policy_id": "5" * 64,
                        "physical_plan_id": f"{route_index:x}" * 64,
                        "executable_id": f"{executable_key:x}" * 64,
                    },
                    "measurement": {
                        "total_wall_s": total_wall_s * (1.0 + block_id * 0.001),
                        "kernel_s": kernel_s * (1.0 + block_id * 0.001),
                    },
                    "backend_facts": {
                        "target_observed": "physical_hardware",
                        "physical_target_verified": True,
                        "hardware_kernel_executed": True,
                        "simulator_kernel_executed": False,
                        "cpu_fallback_used": False,
                        "startup_resource_admission_passed": True,
                        "execution_resource_admission_passed": True,
                        "requested_dpus": dpu_count,
                        "allocated_dpus": dpu_count,
                        "active_dpus": dpu_count,
                        "execution_active_dpu_count": dpu_count,
                        "execution_active_rank_count": 1,
                        "tasklets_per_dpu": tasklets,
                        "dominant_work_wave_populated_dpu_slots": dpu_count,
                        "dominant_work_wave_allocated_dpu_slots": dpu_count,
                        "dominant_work_wave_tasklet_row_sufficiency_passed": True,
                        "dominant_work_wave_utilization": 1.0,
                        "arithmetic_weighted_tasklet_utilization": tasklet_utilization,
                        "arithmetic_weighted_dpu_slot_utilization": dpu_utilization,
                        "output_hash": "f" * 64,
                    },
                    "validation": {
                        "accuracy_qualified": True,
                        "policy_reference_passed": True,
                        "full_precision_passed": True,
                        "relative_l2_error": 1e-6,
                        "norm_drift": 1e-6,
                    },
                }
            )
            binary_key = f"{executable_key:x}" * 64
            sessions.append(
                {
                    "route_id": route_id,
                    "session_instance_id": session_id,
                    "status": "success",
                    "release_verified": True,
                    "terminal_backend_facts": {
                        "observed_tasklets_per_dpu": tasklets,
                        "host_binary_sha256": binary_key,
                        "dpu_binary_sha256": binary_key,
                        "initialization_binary_sha256": binary_key,
                    },
                }
            )
    return manifest, tuple(samples), tuple(sessions)


def test_parallel_scaling_summary_requires_complete_exact_resources() -> None:
    script = _parallel_scaling()
    manifest, samples, sessions = _complete_parallel_diagnostic(script)
    rows = _parallel_scaling_rows()
    summary = script.derive_summary(
        manifest=manifest,
        samples=samples,
        sessions=sessions,
        report={"schema_version": "evidence_report_v5", "scaling_count": 9},
        scaling_rows=rows,
        expected_source_commit="a" * 40,
    )
    assert summary["gate_passed"] is True
    assert summary["sample_count"] == 36
    assert summary["session_count"] == 36
    assert len(summary["primary_comparisons"]) == 5
    assert summary["primary_comparisons"][2]["kernel_speedup"] == pytest.approx(8.0)

    drifted_samples = [dict(sample) for sample in samples]
    target = next(
        sample
        for sample in drifted_samples
        if sample["route_id"] == "upmem_float32_4dpu_t8"
    )
    target["backend_facts"] = {**target["backend_facts"], "active_dpus": 3}
    with pytest.raises(ValueError, match="resource or provenance mismatch"):
        script.derive_summary(
            manifest=manifest,
            samples=tuple(drifted_samples),
            sessions=sessions,
            report={"schema_version": "evidence_report_v5", "scaling_count": 9},
            scaling_rows=rows,
            expected_source_commit="a" * 40,
        )


def _complete_m7c_diagnostic(script: object) -> tuple[dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml"
    )
    manifest = {
        "source_commit": script._source_commit(),
        "configuration": {"experiment": config},
    }
    samples: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    binary_hashes = {
        "host_binary_sha256": "1" * 64,
        "dpu_binary_sha256": "2" * 64,
        "initialization_binary_sha256": "3" * 64,
    }
    physical_routes = set(script._DIAGNOSTIC_PHYSICAL_ROUTE_IDS)
    for block_id in range(6):
        attempt_kind = "warmup" if block_id == 0 else "measurement"
        sample_index = 0 if block_id == 0 else block_id - 1
        for route_id in script._DIAGNOSTIC_ROUTE_IDS:
            physical = route_id in physical_routes
            session_id = f"{route_id}-{block_id}" if physical else None
            samples.append(
                {
                    "case_id": "scaling_primary",
                    "plan_id": "greedy",
                    "route_id": route_id,
                    "attempt_kind": attempt_kind,
                    "sample_index": sample_index,
                    "block_id": block_id,
                    "status": "success",
                    "session_instance_id": session_id,
                    "measurement": {"total_wall_s": 0.020},
                    "backend_facts": (
                        {
                            "target_observed": "physical_hardware",
                            "physical_target_verified": True,
                            "hardware_kernel_executed": True,
                            "simulator_kernel_executed": False,
                            "cpu_fallback_used": False,
                            "startup_resource_admission_passed": True,
                            "execution_resource_admission_passed": True,
                        }
                        if physical
                        else {}
                    ),
                    "validation": {
                        "accuracy_qualified": True,
                        "policy_reference_applicable": physical,
                        "policy_reference_passed": True if physical else None,
                    },
                }
            )
            if physical:
                sessions.append(
                    {
                        "route_id": route_id,
                        "session_instance_id": session_id,
                        "status": "success",
                        "release_verified": True,
                        "terminal_backend_facts": {
                            "target_observed": "physical_hardware",
                            **binary_hashes,
                        },
                    }
                )
    return manifest, tuple(samples), tuple(sessions)


def test_m7c_diagnostic_summary_requires_complete_literal_matrix() -> None:
    script = _scaling_campaign()
    manifest, samples, sessions = _complete_m7c_diagnostic(script)
    report = {"schema_version": "evidence_report_v5"}

    complete = script._diagnostic_summary(
        manifest=manifest,
        samples=samples,
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert complete["gate_passed"] is True
    assert complete["expected_route_ids"] == list(script._DIAGNOSTIC_ROUTE_IDS)
    assert complete["expected_block_ids"] == [0, 1, 2, 3, 4, 5]
    assert all(not warnings for warnings in complete["measurement_warnings"].values())

    short_samples = [dict(sample) for sample in samples]
    for sample in short_samples:
        if sample["route_id"] == "numpy_same_dag" and sample["attempt_kind"] == "measurement":
            sample["measurement"] = {"total_wall_s": 0.001}
    short = script._diagnostic_summary(
        manifest=manifest,
        samples=tuple(short_samples),
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert short["gate_passed"] is True
    assert short["measurement_warnings"]["numpy_same_dag"] == ["median_below_10ms"]

    incomplete = script._diagnostic_summary(
        manifest=manifest,
        samples=samples[:-1],
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert incomplete["gate_passed"] is False
    assert "diagnostic_block_matrix_incomplete" in incomplete["gate_reasons"]


def test_m7c_performance_run_requires_diagnostic_summary_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _scaling_campaign()
    monkeypatch.setattr(script, "_clean_worktree", lambda: None)
    monkeypatch.setattr(script, "_selector_check", lambda *args: None)
    output = tmp_path / "evidence"
    report_output = tmp_path / "report"

    with pytest.raises(ValueError, match="requires --diagnostic-summary"):
        script.run_campaign(
            selection=ROOT / "configs" / "m7c_workload_selection.json",
            config=ROOT / "configs" / "tn_benchmark_physical_scaling.yml",
            output=output,
            report_output=report_output,
        )

    assert not output.exists()
    assert not report_output.exists()


def test_m7c_campaign_binding_ignores_collection_lifecycle(tmp_path: Path) -> None:
    script = _scaling_campaign()
    diagnostic = script._plain(
        load_experiment_config(ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml")
    )
    performance = script._plain(
        load_experiment_config(ROOT / "configs" / "tn_benchmark_physical_scaling.yml")
    )
    binaries = {
        "host_binary": tmp_path / "host",
        "dpu_binary": tmp_path / "dpu",
        "initialization_binary": tmp_path / "initialization",
    }
    for path in binaries.values():
        path.write_bytes(b"m7c")
    for configuration in (diagnostic, performance):
        configuration["collection"]["machine_policy"]["affinity"] = {
            "mode": "exact_required_v1",
            "expected_cpus": [1, 3],
        }
        for route in configuration["routes"].values():
            if route["executor"] != "upmem_physical":
                continue
            route["options"]["rank_paths"] = ["/dev/dpu_rank19"]
            route["options"]["session_root"] = str(tmp_path / route["options"]["session_root"])
            for field, path in binaries.items():
                route["options"][field] = str(path)
    source_commit = script._source_commit()
    selection_sha256 = "5" * 64
    diagnostic_binding = script._campaign_binding_sha256(
        diagnostic,
        source_commit=source_commit,
        selection_sha256=selection_sha256,
        binary_hashes=script._route_binary_hashes_from_config(diagnostic),
    )
    performance_binding = script._campaign_binding_sha256(
        performance,
        source_commit=source_commit,
        selection_sha256=selection_sha256,
        binary_hashes=script._route_binary_hashes_from_config(performance),
    )

    assert diagnostic_binding == performance_binding
    assert performance["collection"]["block_cooldown_s"] == 0.0
