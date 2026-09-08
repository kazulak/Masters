"""End-to-end offline fixtures for physical wave pretest extraction."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from quantum_bench import evidence, experiment


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
generator = importlib.import_module("upmem_path_heuristic")

QUALIFIER_SPEC = importlib.util.spec_from_file_location(
    "wave_pretest_qualifier", SCRIPTS / "qualify_upmem_path_candidates.py"
)
assert QUALIFIER_SPEC is not None and QUALIFIER_SPEC.loader is not None
qualifier = importlib.util.module_from_spec(QUALIFIER_SPEC)
sys.modules[QUALIFIER_SPEC.name] = qualifier
QUALIFIER_SPEC.loader.exec_module(qualifier)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "wave_pretest_existing_fixtures", ROOT / "tests" / "test_upmem_path_heuristic_script.py"
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
existing_fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = existing_fixtures
FIXTURE_SPEC.loader.exec_module(existing_fixtures)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_canonical(path: Path, value: Any) -> None:
    path.write_text(evidence.canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(evidence.canonical_json(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _training_config(tmp_path: Path) -> tuple[Path, Path]:
    config = deepcopy(generator.load_config(ROOT / "configs" / "upmem_final_system_path_study_v2.json"))
    config["circuits"] = [
        {
            "circuit_id": "pretest_training",
            "split": "training",
            "circuit": {
                "kind": "builtin",
                "name": "quantization_stress",
                "parameters": {"n_qubits": 14, "repeat_layers": 2},
            },
        },
        {
            "circuit_id": "pretest_validation",
            "split": "validation",
            "circuit": {
                "kind": "builtin",
                "name": "quantization_stress",
                "parameters": {"n_qubits": 14, "repeat_layers": 2},
            },
        },
        {
            "circuit_id": "pretest_test",
            "split": "test",
            "circuit": {
                "kind": "builtin",
                "name": "quantization_stress",
                "parameters": {"n_qubits": 14, "repeat_layers": 2},
            },
        },
    ]
    config["candidate_generation"]["one_trial_searches"] = 1
    config_path = tmp_path / "generator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    generated = tmp_path / "generated"
    generator.generate(config_path, generated)
    return generated / "candidate_paths.json", generated / "candidate_rankings.csv"


def _profile_fixture(dataset: dict[str, Any], output_path: Path) -> dict[str, Any]:
    contract = generator.execution_contract(dataset)
    assert contract is not None
    candidate_sha = _sha256_bytes(generator._canonical_bytes(dataset))
    weights = generator.WeightVector.from_values(
        {
            **{feature: 0.25 for feature in generator.FEATURE_NAMES[:4]},
            "E_num": 0.0,
            "P_wram": 0.0,
        },
        inactive=("E_num", "P_wram"),
    )
    selected: dict[str, str] = {}
    measured: dict[str, dict[str, float]] = {}
    for circuit in dataset["circuits"]:
        if circuit["split"] != "training":
            continue
        for topology_id in ("1dpu_t8", "4dpu_t8"):
            pool = generator._wave_evaluation_pool(circuit, topology_id, contract)
            greedy = next(item for item in pool if item["is_greedy"] is True)
            selected_item = min(
                pool,
                key=lambda item: (
                    generator._wave_score(item, greedy, topology_id, weights),
                    str(item["candidate_path_id"]),
                ),
            )
            cell_id = f"{circuit['circuit_id']}:{topology_id}"
            selected[cell_id] = str(selected_item["candidate_path_id"])
            measured[cell_id] = {
                str(item["candidate_path_id"]): 1.0 for item in pool
            }
    stage_experiment = "1" * 64
    stage = {
        "stage_id": generator.WAVE_INITIAL_STAGE,
        "round_ordinal": 0,
        "prior_stage_hashes": [],
        "selection_profile_sha256": None,
        "timing_used_for_selection": False,
        "experiment_id": stage_experiment,
        "run_id": "2" * 64,
        "round_id": f"{generator.WAVE_INITIAL_STAGE}:{stage_experiment}",
        "calibration_set_sha256": "3" * 64,
        "runtime_table_sha256": "4" * 64,
    }
    profile = {
        "schema_version": generator.WAVE_CALIBRATION_PROFILE,
        "execution_profile": generator.EXECUTION_PROFILE,
        "execution_contract": dict(contract),
        "score_id": generator.WAVE_COST_MODEL_ID,
        "primary_quantity": generator.WAVE_PRIMARY_QUANTITY,
        "timing_scope": generator.CALIBRATION_TIMING_SCOPE,
        "numeric_policy": contract["numeric_policy"],
        "normalization": generator.WAVE_NORMALIZATION,
        "source_sha": dataset["source_sha"],
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": contract["execution_source"],
        "preregistration_sha256": dataset["preregistration_sha256"],
        "candidate_set_sha256": candidate_sha,
        "calibration_set_sha256": "3" * 64,
        "runtime_table_sha256": "4" * 64,
        "stage_calibration_sha256": ["3" * 64],
        "stage_runtime_table_sha256": ["4" * 64],
        "stage_id": generator.WAVE_INITIAL_STAGE,
        "round_ordinal": 0,
        "experiment_id": stage_experiment,
        "round_id": stage["round_id"],
        "prior_stage_hashes": [],
        "selection_profile_sha256": None,
        "timing_used_for_selection": False,
        "stage_count": 1,
        "adaptive_round_count": 0,
        "stage_metadata": [stage],
        "requested_model_form": "six_term",
        "fit_splits": ["training"],
        "training_cell_ids": sorted(selected),
        "weights": weights.as_mapping(),
        "feature_model": {
            "mode": "six_term",
            "active_features": list(generator.FEATURE_NAMES[:4]),
            "zero_range_features": [],
            "correlated_pairs": [],
            "matrix_rank": 4,
            "rank_tolerance": 1.0e-12,
            "reason": "deterministic schema fixture",
        },
        "selected_path_ids": selected,
        "candidate_speedups": measured,
    }
    output_path.write_bytes(generator._canonical_bytes(profile))
    assert _sha256_bytes(output_path.read_bytes()) == _sha256_bytes(
        generator._canonical_bytes(profile)
    )
    return profile


def _route_binary_manifest(normalized: dict[str, Any]) -> dict[str, str]:
    hashes = {
        "dpu_binary": "8" * 64,
        "host_binary": "9" * 64,
        "initialization_binary": "a" * 64,
    }
    result: dict[str, str] = {}
    for route in normalized["routes"].values():
        for field, digest in hashes.items():
            result[route["options"][field]] = digest
    return result


def _make_profile_and_config(
    tmp_path: Path,
    dataset_path: Path,
    dataset: dict[str, Any],
    *,
    mode: str,
    split: str,
) -> tuple[Path, Path, Path | None, dict[str, Any], dict[str, Any]]:
    profile_path = tmp_path / "physical_speedup_fit_v1.json"
    _profile_fixture(dataset, profile_path)
    selection_path: Path | None = None
    if mode == "evaluation":
        selection_path = tmp_path / f"selection-{split}.json"
        generator.evaluate_frozen_profile(
            dataset_path, profile_path, selection_path, split=split
        )
    packet_path = tmp_path / f"{mode}-{split}.yml"
    qualifier.prepare_config(
        dataset_path=dataset_path,
        output_path=packet_path,
        mode=mode,
        selection_path=selection_path,
        profile_path=profile_path,
        split=split,
    )
    normalized = json.loads(
        qualifier.canonical_json(qualifier.load_experiment_config(packet_path))
    )
    assert json.loads(
        packet_path.with_suffix(".yml.provenance.json").read_text(encoding="utf-8")
    )["calibration_set_sha256"] is None
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive = raw_dir.parent / "preregistration"
    archive.mkdir()
    physical_path = archive / "physical.yml"
    physical_path.write_bytes(packet_path.read_bytes())
    provenance_path = archive / "physical.yml.provenance.json"
    provenance_path.write_bytes(packet_path.with_suffix(".yml.provenance.json").read_bytes())
    binary_manifest = _route_binary_manifest(normalized)
    (archive / "binary_sha256.json").write_bytes(
        generator._canonical_bytes(binary_manifest)
    )
    checksum_lines = []
    for path in sorted(archive.iterdir()):
        checksum_lines.append(f"{generator._file_sha256(path)}  {path.name}\n")
    (archive / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
    return raw_dir, profile_path, provenance_path, normalized, binary_manifest


def _backend_facts(
    *,
    base_sample: dict[str, Any],
    base_session: dict[str, Any],
    dpu_count: int,
    candidate: dict[str, Any],
    topology: dict[str, Any],
    output_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_dpus = [[0, index] for index in range(dpu_count)]
    backend = deepcopy(base_sample["backend_facts"])
    backend.update({
        "requested_dpus": dpu_count,
        "allocated_dpus": dpu_count,
        "active_dpus": dpu_count,
        "active_dpu_ids": active_dpus,
        "execution_active_dpu_count": dpu_count,
        "logical_plan_id": candidate["logical_plan_id"],
        "physical_plan_id": topology["physical_plan_id"],
        "output_hash": output_sha,
    })
    terminal = deepcopy(base_session["terminal_backend_facts"])
    terminal.update({
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": dpu_count,
        "observed_dpu_count": dpu_count,
        "startup_requested_dpu_count": dpu_count,
        "startup_allocated_dpu_count": dpu_count,
        "active_dpu_ids": active_dpus,
    })
    return backend, terminal


def _raw_fixture(
    raw_dir: Path,
    normalized: dict[str, Any],
    dataset: dict[str, Any],
    binary_manifest: dict[str, str],
    base_sample: dict[str, Any],
    base_session: dict[str, Any],
) -> dict[str, Any]:
    environment = {"host": "pretest-fixture", "requested_rank_paths": ["/dev/dpu_rank1"]}
    validation_policy = dict(experiment.default_validation_policy())
    environment_digest = evidence.environment_id(environment)
    validation_digest = experiment.default_validation_policy_id()
    run_id = evidence.new_run_id()
    experiment_id = normalized["experiment_id"]
    circuits = {item["circuit_id"]: item for item in dataset["circuits"]}
    candidates = {
        (circuit["circuit_id"], candidate["candidate_path_id"]): candidate
        for circuit in dataset["circuits"]
        for candidate in circuit["candidates"]
    }
    bindings: list[dict[str, Any]] = []
    for matrix_item in normalized["matrix"]:
        case_id = matrix_item["case_id"]
        plan_id = matrix_item["plan_id"]
        candidate = candidates[(case_id, plan_id.removeprefix("path_"))]
        for route_id in matrix_item["route_ids"]:
            topology = next(
                item for item in candidate["topologies"]
                if item["topology_id"] == route_id
            )
            bindings.append({
                "case_id": case_id,
                "plan_id": plan_id,
                "route_id": route_id,
                "problem_id": circuits[case_id]["problem_id"],
                "tensor_network_structure_id": circuits[case_id]["tensor_network_structure_id"],
                "logical_plan_id": candidate["logical_plan_id"],
                "physical_plan_id": topology["physical_plan_id"],
                "executable_id": "e" * 64,
                "environment_id": environment_digest,
                "validation_policy_id": validation_digest,
            })
    bindings.sort(key=lambda item: (item["case_id"], item["plan_id"], item["route_id"]))
    manifest = {
        "schema_version": "evidence_manifest_v2",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "collection_policy_id": normalized["collection_policy_id"],
        "environment_id": environment_digest,
        "validation_policy_id": validation_digest,
        "created_at_utc": "2026-09-08T00:00:00Z",
        "source_commit": generator.EXECUTION_SOURCE,
        "source_worktree_dirty": False,
        "configuration": {
            "experiment": normalized,
            "environment": environment,
            "validation_policy": validation_policy,
            "identity_bindings": bindings,
        },
        "expected_counts": {
            "warmup": len(bindings),
            "measurement": len(bindings) * 5,
            "sessions": len(bindings) * 6,
        },
        "files": {
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
            "sessions": "sessions.jsonl",
        },
        "status": "completed",
    }
    schedule = evidence._declared_collection_attempts(manifest)
    assert schedule is not None
    samples: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    bindings_by_key = {
        (item["case_id"], item["plan_id"], item["route_id"]): item
        for item in bindings
    }
    for case_id, plan_id, route_id, attempt, sample_index, block, order in sorted(schedule):
        binding = bindings_by_key[(case_id, plan_id, route_id)]
        candidate = candidates[(case_id, plan_id.removeprefix("path_"))]
        topology = next(
            item for item in candidate["topologies"]
            if item["topology_id"] == route_id
        )
        dpu_count = topology["topology"]["dpu_count"]
        output_sha = _sha256_bytes(
            f"{case_id}:{plan_id}:{route_id}:{block}:{order}".encode()
        )
        backend, terminal = _backend_facts(
            base_sample=base_sample,
            base_session=base_session,
            dpu_count=dpu_count,
            candidate=candidate,
            topology=topology,
            output_sha=output_sha,
        )
        options = normalized["routes"][route_id]["options"]
        for field in ("dpu_binary", "host_binary", "initialization_binary"):
            terminal[f"{field}_path"] = options[field]
            terminal[f"{field}_sha256"] = binary_manifest[options[field]]
        session_id = f"session-{len(samples):04d}"
        sample = {
            "schema_version": "evidence_sample_v4",
            "sample_id": evidence.sample_id(
                run_id, case_id, route_id, attempt, sample_index,
                plan_id=plan_id, block_id=block, order_index=order,
            ),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "case_id": case_id,
            "plan_id": plan_id,
            "route_id": route_id,
            "attempt_kind": attempt,
            "sample_index": sample_index,
            "block_id": block,
            "order_index": order,
            "observed_affinity": [0],
            "background_load_1m": None,
            "session_instance_id": session_id,
            "status": "success",
            "identities": {
                field: binding[field]
                for field in (
                    "problem_id", "tensor_network_structure_id", "logical_plan_id",
                    "physical_plan_id", "executable_id", "environment_id",
                    "validation_policy_id",
                )
            },
            "measurement": {
                "scope_id": "steady_execution_v1",
                "total_wall_s": 10.0 + block,
                "lowering_s": None,
                "planning_s": None,
                "slicing_s": None,
                "mapping_s": None,
                "session_open_s": 0.0,
                "encode_s": None,
                "preparation_s": None,
                "h2d_s": None,
                "kernel_s": 1.0,
                "host_reduce_s": None,
                "d2h_s": None,
                "decode_s": None,
                "rank_work_s": None,
                "h2d_bytes": 0,
                "d2h_bytes": 0,
                "energy_j": None,
            },
            "backend_facts": backend,
            "numeric_facts": {
                "numeric_policy": generator.NUMERIC_POLICY,
                "operand_records": [],
                "operations": [],
                "raw_lane_records": [],
                "saturation_real": 0,
                "saturation_imag": 0,
            },
            "output_sha256": output_sha,
            "validation": {
                "policy_reference_applicable": True,
                "policy_reference_passed": True,
                "full_precision_threshold_applicable": True,
                "full_precision_passed": True,
                "accuracy_qualified": True,
                "max_abs_error": 0.0,
                "relative_l2_error": 0.0,
                "norm_drift": 0.0,
                "phase_aligned_max_abs_error": 0.0,
            },
            "failure": None,
        }
        session = {
            "schema_version": "evidence_session_v1",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "case_id": case_id,
            "plan_id": plan_id,
            "route_id": route_id,
            "session_instance_id": session_id,
            "session_protocol_id": generator.WAVE_SESSION_PROTOCOL,
            "open_s": 0.0,
            "session_close_s": 0.0,
            "status": "success",
            "terminal_backend_facts": terminal,
            "release_attempted": True,
            "release_succeeded": True,
            "release_verified": True,
            "failure": None,
        }
        samples.append(sample)
        sessions.append(session)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_canonical(raw_dir / "manifest.json", manifest)
    _write_jsonl(raw_dir / "samples.jsonl", samples)
    _write_jsonl(raw_dir / "sessions.jsonl", sessions)
    generator.load_artifacts(raw_dir)
    return {
        "manifest": manifest,
        "samples": samples,
        "sessions": sessions,
        "run_id": run_id,
        "bindings": bindings,
    }


def _scenario(tmp_path: Path, *, mode: str, split: str) -> dict[str, Any]:
    dataset_path, rankings_path = _training_config(tmp_path)
    del rankings_path
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_dir, profile_path, provenance_path, normalized, binary_manifest = (
        _make_profile_and_config(
            tmp_path, dataset_path, dataset, mode=mode, split=split
        )
    )
    template_root = tmp_path / "existing-fixture-template"
    template_root.mkdir()
    _, _, _, _, template_samples, template_sessions = (
        existing_fixtures._wave_calibration_fixture(template_root)
    )
    raw = _raw_fixture(
        raw_dir, normalized, dataset, binary_manifest,
        deepcopy(template_samples[0]), deepcopy(template_sessions[0]),
    )
    return {
        "raw_dir": raw_dir,
        "candidate_path": dataset_path,
        "profile_path": profile_path,
        "provenance_path": provenance_path,
        "dataset": dataset,
        "normalized": normalized,
        "binary_manifest": binary_manifest,
        **raw,
        "mode": mode,
        "split": split,
    }


@pytest.mark.parametrize(
    ("mode", "split"),
    (("confirmation", "training"), ("evaluation", "validation"), ("evaluation", "test")),
)
def test_extract_wave_pretest_full_physical_one_plus_five(
    tmp_path: Path, mode: str, split: str
) -> None:
    scenario = _scenario(tmp_path, mode=mode, split=split)
    output_dir = tmp_path / "extracted"
    result = generator.extract_wave_pretest(
        scenario["raw_dir"], scenario["candidate_path"], scenario["provenance_path"],
        scenario["profile_path"], output_dir, mode=mode, split=split,
    )
    route_count = len(scenario["bindings"])
    assert result["collection"] == {
        "warmup_blocks": 1,
        "measurement_blocks": 5,
        "blocks": [0, 1, 2, 3, 4, 5],
        "attempts_per_candidate_cell": 6,
    }
    assert result["sample_count"] == route_count * 6
    assert result["session_count"] == route_count * 6
    assert result["expected_candidate_cell_count"] == route_count
    assert result["source_sha"] == scenario["dataset"]["source_sha"]
    assert result["candidate_generation_source_sha"] == scenario["dataset"]["source_sha"]
    assert result["physical_execution_source_sha"] == generator.EXECUTION_SOURCE
    assert result["candidate_set_sha256"] == _sha256_bytes(
        generator._canonical_bytes(scenario["dataset"])
    )
    assert result["fitted_profile_sha256"] == _sha256_bytes(
        scenario["profile_path"].read_bytes()
    )
    assert result["calibration_set_sha256"] is None
    assert result["execution_provenance"]["calibration_set_sha256"] is None
    assert result["timing_scope"] == "steady_execution_v1"
    assert result["primary_quantity"] == "session_inclusive_s"
    assert {
        name for role in result["selection_roles"] for name in role["roles"]
    } == (
        {"greedy", "upmem_selected"}
        if mode == "confirmation"
        else {"greedy", "minimum_flops", "upmem_selected"}
    )
    assert {row["topology_id"] for row in result["observations"]} == {
        "1dpu_t8", "4dpu_t8"
    }
    assert all(
        row["session_inclusive_s"]
        == row["session_open_s"] + row["total_wall_s"] + row["session_close_s"]
        for row in result["observations"]
    )
    archive = scenario["raw_dir"].parent / "preregistration"
    assert result["raw_artifact_sha256"] == {
        name: generator._file_sha256(scenario["raw_dir"] / name)
        for name in ("manifest.json", "samples.jsonl", "sessions.jsonl")
    }
    assert result["private_provenance"]["checksums"] == {
        name: generator._file_sha256(archive / name)
        for name in (
            "physical.yml", "physical.yml.provenance.json", "binary_sha256.json",
            "SHA256SUMS",
        )
    }
    assert result["binary_bindings"] == {
        route_id: {
            field: {"path": scenario["normalized"]["routes"][route_id]["options"][field],
                    "sha256": scenario["binary_manifest"][scenario["normalized"]["routes"][route_id]["options"][field]]}
            for field in ("dpu_binary", "host_binary", "initialization_binary")
        }
        for route_id in scenario["normalized"]["routes"]
    }
    assert (output_dir / "path_runtime_wave_pretest.json").is_file()
    assert (output_dir / "path_runtime_wave_pretest.csv").is_file()


def _reload_raw(scenario: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads((scenario["raw_dir"] / "manifest.json").read_text())
    samples = [json.loads(line) for line in (scenario["raw_dir"] / "samples.jsonl").read_text().splitlines()]
    sessions = [json.loads(line) for line in (scenario["raw_dir"] / "sessions.jsonl").read_text().splitlines()]
    return manifest, samples, sessions


def _rewrite_raw(
    scenario: dict[str, Any], manifest: dict[str, Any],
    samples: list[dict[str, Any]], sessions: list[dict[str, Any]],
) -> None:
    _write_canonical(scenario["raw_dir"] / "manifest.json", manifest)
    _write_jsonl(scenario["raw_dir"] / "samples.jsonl", samples)
    _write_jsonl(scenario["raw_dir"] / "sessions.jsonl", sessions)


@pytest.mark.parametrize("kind", ("missing", "out_of_schedule", "one_plus_three"))
def test_extract_wave_pretest_rejects_non_exact_attempt_set(tmp_path: Path, kind: str) -> None:
    scenario = _scenario(tmp_path, mode="confirmation", split="training")
    manifest, samples, sessions = _reload_raw(scenario)
    if kind == "missing":
        removed = samples.pop()
        sessions[:] = [
            row for row in sessions
            if row["session_instance_id"] != removed["session_instance_id"]
        ]
    elif kind == "one_plus_three":
        removed_ids = {
            row["session_instance_id"] for row in samples if row["block_id"] in {4, 5}
        }
        samples[:] = [row for row in samples if row["session_instance_id"] not in removed_ids]
        sessions[:] = [row for row in sessions if row["session_instance_id"] not in removed_ids]
    elif kind == "out_of_schedule":
        sample = next(row for row in samples if row["block_id"] == 5)
        sample["block_id"] = 6
        sample["sample_index"] = 5
        sample["sample_id"] = evidence.sample_id(
            sample["run_id"], sample["case_id"], sample["route_id"],
            sample["attempt_kind"], sample["sample_index"],
            plan_id=sample["plan_id"], block_id=6, order_index=sample["order_index"],
        )
    _rewrite_raw(scenario, manifest, samples, sessions)
    with pytest.raises(ValueError, match=r"(exact|declared|count|1\+5|schedule)"):
        generator.extract_wave_pretest(
            scenario["raw_dir"], scenario["candidate_path"], scenario["provenance_path"],
            scenario["profile_path"], tmp_path / "bad", mode="confirmation", split="training",
        )


def test_extract_wave_pretest_rejects_profile_drift(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path, mode="evaluation", split="validation")
    profile = json.loads(scenario["profile_path"].read_text())
    profile["weights"][generator.FEATURE_NAMES[0]] = 0.5
    profile["weights"][generator.FEATURE_NAMES[1]] = 0.0
    scenario["profile_path"].write_bytes(generator._canonical_bytes(profile))
    with pytest.raises(ValueError, match="profile"):
        generator.extract_wave_pretest(
            scenario["raw_dir"], scenario["candidate_path"], scenario["provenance_path"],
            scenario["profile_path"], tmp_path / "bad", mode="evaluation", split="validation",
        )


@pytest.mark.parametrize("kind", ("split", "source"))
def test_extract_wave_pretest_rejects_split_or_source_drift(
    tmp_path: Path, kind: str
) -> None:
    scenario = _scenario(tmp_path, mode="confirmation", split="training")
    if kind == "split":
        with pytest.raises(ValueError, match="mode and split"):
            generator.extract_wave_pretest(
                scenario["raw_dir"], scenario["candidate_path"], scenario["provenance_path"],
                scenario["profile_path"], tmp_path / "bad", mode="confirmation", split="validation",
            )
        return
    manifest, samples, sessions = _reload_raw(scenario)
    manifest["source_commit"] = "0" * 40
    _rewrite_raw(scenario, manifest, samples, sessions)
    with pytest.raises(ValueError, match="physical execution source"):
        generator.extract_wave_pretest(
            scenario["raw_dir"], scenario["candidate_path"], scenario["provenance_path"],
            scenario["profile_path"], tmp_path / "bad", mode="confirmation", split="training",
        )
