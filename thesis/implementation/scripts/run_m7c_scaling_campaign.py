#!/usr/bin/env python3
"""Prepare, run, and inspect preregistered M7C mixed scaling campaigns.

This command is deliberately separate from ``qualify``: a mixed NumPy/physical
matrix must use ``quantum_bench.cli run --allow-physical``.  The script is an
operator entry point and performs physical execution only when ``run`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any, Mapping

import yaml

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
_PATH_FIELDS = (
    "host_binary",
    "dpu_binary",
)
_DIAGNOSTIC_SUMMARY_SCHEMA = "m7c_diagnostic_summary_v1"
_DIAGNOSTIC_ROUTE_IDS = (
    "numpy_same_dag",
    "upmem_float32_1dpu_t1",
    "upmem_float32_1dpu_t8",
    "upmem_float32_2dpu_t8",
    "upmem_float32_4dpu_t8",
)
_DIAGNOSTIC_PHYSICAL_ROUTE_IDS = _DIAGNOSTIC_ROUTE_IDS[1:]
_DIAGNOSTIC_ROUTE_SPECS = {
    "numpy_same_dag": ("numpy_dag", "split_complex_float32_v1", None),
    "upmem_float32_1dpu_t1": ("upmem_physical", "split_complex_float32_v1", (1, 1, 1)),
    "upmem_float32_1dpu_t8": ("upmem_physical", "split_complex_float32_v1", (1, 1, 8)),
    "upmem_float32_2dpu_t8": ("upmem_physical", "split_complex_float32_v1", (2, 1, 8)),
    "upmem_float32_4dpu_t8": ("upmem_physical", "split_complex_float32_v1", (4, 1, 8)),
}


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required M7C binary is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    source_commit = result.stdout.strip()
    if result.returncode or len(source_commit) != 40:
        raise ValueError("cannot determine M7C source commit")
    return source_commit


def _absolute_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _parse_cpus(value: str) -> list[int]:
    try:
        cpus = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise ValueError("expected CPUs must be a comma-separated integer list") from exc
    if not cpus or len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise ValueError("expected CPUs must be unique nonnegative integers")
    return cpus


def _physical_route_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    routes = config.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("configuration routes must be a mapping")
    return tuple(
        route_id
        for route_id, route in sorted(routes.items())
        if isinstance(route, Mapping) and route.get("executor") == "upmem_physical"
    )


def _clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("cannot inspect Git worktree")
    if result.stdout.strip():
        raise ValueError("M7C physical campaign requires a clean Git worktree")


def prepare_config(
    *,
    template: Path,
    output: Path,
    rank_paths: list[str],
    session_root: str,
    expected_cpus: list[int],
) -> Path:
    """Copy a tracked mixed template with target-specific absolute paths."""

    if output.exists():
        raise ValueError(f"prepared configuration must be absent: {output}")
    config = _plain(load_experiment_config(template))
    config.pop("collection_policy_id", None)
    config.pop("experiment_identity_payload", None)
    physical_route_ids = _physical_route_ids(config)
    if not physical_route_ids:
        raise ValueError("M7C scaling template must include physical routes")
    absolute_ranks = [_absolute_path(path) for path in rank_paths]
    absolute_session_root = Path(_absolute_path(session_root))
    for route_id in physical_route_ids:
        route = config["routes"][route_id]
        options = route["options"]
        rank_count = options["rank_count"]
        if len(absolute_ranks) != rank_count:
            raise ValueError(
                f"route {route_id} requires {rank_count} rank paths, got {len(absolute_ranks)}"
            )
        options["rank_paths"] = list(absolute_ranks)
        options["session_root"] = str(absolute_session_root / route_id)
    config["collection"]["machine_policy"]["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": expected_cpus,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = load_experiment_config(output)
    for route_id in physical_route_ids:
        source_options = config["routes"][route_id]["options"]
        prepared_options = prepared["routes"][route_id]["options"]
        for field in (*_PATH_FIELDS, "session_root"):
            if prepared_options[field] != source_options[field]:
                raise ValueError(f"prepared {route_id}.{field} resolved incorrectly")
        if tuple(prepared_options["rank_paths"]) != tuple(source_options["rank_paths"]):
            raise ValueError(f"prepared {route_id}.rank_paths resolved incorrectly")
    return output


def _run(command: list[str], *, env: Mapping[str, str], capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(env),
        capture_output=capture,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + ("\n" + result.stdout if result.stdout else "")
            + ("\n" + result.stderr if result.stderr else "")
        )
    return result.stdout


def _selector_check(
    selection: Path, config: Path | None, env: Mapping[str, str]
) -> None:
    command = [
        sys.executable,
        "scripts/select_m7c_workload.py",
        "--check",
        str(selection),
    ]
    if config is not None:
        command.extend(("--config", str(config)))
    _run(command, env=env)


def _joined_sample_facts(
    sample: Mapping[str, Any], sessions_by_id: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("physical scaling sample lacks backend facts")
    joined = dict(facts)
    session_id = sample.get("session_instance_id")
    if isinstance(session_id, str):
        session = sessions_by_id.get(session_id)
        terminal = session.get("terminal_backend_facts") if session else None
        if isinstance(terminal, Mapping):
            for field, value in terminal.items():
                joined.setdefault(field, value)
    return joined


def _route_binary_hashes_from_config(
    configuration: Mapping[str, object],
) -> Mapping[str, Mapping[str, str]]:
    hashes: dict[str, Mapping[str, str]] = {}
    for route_id in _DIAGNOSTIC_PHYSICAL_ROUTE_IDS:
        route = configuration["routes"][route_id]
        options = route["options"]
        hashes[route_id] = {
            f"{field}_sha256": _file_sha256(Path(options[field]))
            for field in _PATH_FIELDS
        }
    return hashes


def _route_binary_hashes_from_sessions(
    sessions: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Mapping[str, str]]:
    keys = tuple(f"{field}_sha256" for field in _PATH_FIELDS)
    observed: dict[str, set[tuple[str, str, str]]] = {
        route_id: set() for route_id in _DIAGNOSTIC_PHYSICAL_ROUTE_IDS
    }
    for session in sessions:
        route_id = session.get("route_id")
        if route_id not in observed:
            continue
        facts = session.get("terminal_backend_facts")
        if not isinstance(facts, Mapping):
            raise ValueError(f"M7C session lacks terminal facts for {route_id}")
        values = tuple(facts.get(key) for key in keys)
        if any(not isinstance(value, str) or len(value) != 64 for value in values):
            raise ValueError(f"M7C session lacks binary hashes for {route_id}")
        observed[route_id].add(values)
    hashes: dict[str, Mapping[str, str]] = {}
    for route_id, values in observed.items():
        if len(values) != 1:
            raise ValueError(f"M7C session binary hashes are inconsistent for {route_id}")
        hashes[route_id] = dict(zip(keys, next(iter(values)), strict=True))
    return hashes


def _campaign_binding_payload(
    configuration: Mapping[str, object],
    *,
    source_commit: str,
    selection_sha256: str,
    binary_hashes: Mapping[str, Mapping[str, str]],
) -> Mapping[str, object]:
    matrix = configuration["matrix"]
    if len(matrix) != 1:
        raise ValueError("M7C campaign binding requires exactly one matrix entry")
    matrix_item = matrix[0]
    case_id = matrix_item["case_id"]
    plan_id = matrix_item["plan_id"]
    route_ids = tuple(matrix_item["route_ids"])
    if plan_id is None:
        raise ValueError("M7C campaign binding requires a logical plan")
    route_records: list[dict[str, object]] = []
    for route_id in route_ids:
        route = configuration["routes"][route_id]
        record: dict[str, object] = {
            "route_id": route_id,
            "executor": route["executor"],
            "numeric_policy": route["numeric_policy"],
        }
        if route["executor"] == "upmem_physical":
            options = route["options"]
            record.update(
                {
                    "topology": {
                        "dpu_count": options["dpu_count"],
                        "rank_count": options["rank_count"],
                        "tasklets_per_dpu": options["tasklets_per_dpu"],
                    },
                    "rank_paths": list(options["rank_paths"]),
                    "binary_sha256": dict(binary_hashes[route_id]),
                }
            )
        route_records.append(record)
    affinity = configuration["collection"]["machine_policy"]["affinity"]
    expected_cpus = affinity["expected_cpus"]
    return {
        "source_commit": source_commit,
        "selection_sha256": selection_sha256,
        "case": {
            "case_id": case_id,
            "circuit": {
                "kind": configuration["cases"][case_id]["circuit"]["kind"],
                "name": configuration["cases"][case_id]["circuit"]["name"],
                "parameters": dict(configuration["cases"][case_id]["circuit"]["parameters"]),
            },
        },
        "plan": {
            "plan_id": plan_id,
            "planner": dict(configuration["plans"][plan_id]["planner"]),
            "slicing": configuration["plans"][plan_id]["slicing"],
        },
        "routes": route_records,
        "machine": {
            "expected_cpus": None if expected_cpus is None else list(expected_cpus),
        },
    }


def _campaign_binding_sha256(
    configuration: Mapping[str, object],
    *,
    source_commit: str,
    selection_sha256: str,
    binary_hashes: Mapping[str, Mapping[str, str]],
) -> str:
    return _hash(
        _campaign_binding_payload(
            configuration,
            source_commit=source_commit,
            selection_sha256=selection_sha256,
            binary_hashes=binary_hashes,
        )
    )


def _diagnostic_contract_reasons(configuration: Mapping[str, object]) -> tuple[str, ...]:
    reasons: list[str] = []
    collection = configuration["collection"]
    if collection["claim_policy"] != "diagnostic_v1":
        reasons.append("diagnostic_claim_policy_mismatch")
    if collection["warmup_blocks"] != 1 or collection["measurement_blocks"] != 5:
        reasons.append("diagnostic_block_count_mismatch")
    if tuple(configuration["routes"]) != _DIAGNOSTIC_ROUTE_IDS:
        reasons.append("diagnostic_route_ids_mismatch")
    if set(configuration["cases"]) != {"scaling_primary"}:
        reasons.append("diagnostic_case_mismatch")
    else:
        circuit = configuration["cases"]["scaling_primary"]["circuit"]
        if (
            circuit["kind"] != "builtin"
            or circuit["name"] != "quantization_stress"
            or dict(circuit["parameters"]) != {"n_qubits": 18, "repeat_layers": 2}
        ):
            reasons.append("diagnostic_circuit_mismatch")
    if set(configuration["plans"]) != {"greedy"}:
        reasons.append("diagnostic_plan_mismatch")
    else:
        plan = configuration["plans"]["greedy"]
        if (
            dict(plan["planner"])
            != {"engine": "opt_einsum", "mode": "greedy"}
            or plan["slicing"] is not None
        ):
            reasons.append("diagnostic_planner_mismatch")
    matrix = configuration["matrix"]
    if (
        len(matrix) != 1
        or matrix[0]["case_id"] != "scaling_primary"
        or matrix[0]["plan_id"] != "greedy"
        or tuple(matrix[0]["route_ids"]) != _DIAGNOSTIC_ROUTE_IDS
    ):
        reasons.append("diagnostic_matrix_mismatch")
    for route_id, (executor, numeric_policy, topology) in _DIAGNOSTIC_ROUTE_SPECS.items():
        route = configuration["routes"].get(route_id)
        if not isinstance(route, Mapping):
            continue
        if route["executor"] != executor or route["numeric_policy"] != numeric_policy:
            reasons.append(f"diagnostic_route_contract_mismatch:{route_id}")
            continue
        if topology is None:
            if route["options"]:
                reasons.append(f"diagnostic_route_options_mismatch:{route_id}")
            continue
        options = route["options"]
        observed = (
            options["dpu_count"],
            options["rank_count"],
            options["tasklets_per_dpu"],
        )
        if observed != topology:
            reasons.append(f"diagnostic_route_topology_mismatch:{route_id}")
    return tuple(sorted(set(reasons)))


def _diagnostic_expected_attempts() -> set[tuple[str, str, str, str, int, int]]:
    attempts: set[tuple[str, str, str, str, int, int]] = set()
    for block_id in range(6):
        attempt_kind = "warmup" if block_id == 0 else "measurement"
        sample_index = 0 if block_id == 0 else block_id - 1
        attempts.update(
            (
                "scaling_primary",
                "greedy",
                route_id,
                attempt_kind,
                sample_index,
                block_id,
            )
            for route_id in _DIAGNOSTIC_ROUTE_IDS
        )
    return attempts


def _measurement_statistics(
    samples: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, float | None], Mapping[str, float | None], Mapping[str, list[str]]]:
    medians: dict[str, float | None] = {}
    relative_mads: dict[str, float | None] = {}
    warnings: dict[str, list[str]] = {}
    for route_id in _DIAGNOSTIC_ROUTE_IDS:
        values: list[float] = []
        for sample in samples:
            if sample.get("route_id") != route_id or sample.get("attempt_kind") != "measurement":
                continue
            measurement = sample.get("measurement")
            value = measurement.get("total_wall_s") if isinstance(measurement, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                values.append(float(value))
        route_warnings: list[str] = []
        if len(values) != 5:
            medians[route_id] = None
            relative_mads[route_id] = None
            route_warnings.append("runtime_statistic_unavailable")
        else:
            route_median = float(median(values))
            raw_mad = float(median([abs(value - route_median) for value in values]))
            medians[route_id] = route_median
            relative_mads[route_id] = raw_mad / route_median if route_median > 0 else None
            if route_median <= 0:
                route_warnings.append("runtime_statistic_unavailable")
            elif route_median < 0.010:
                route_warnings.append("median_below_10ms")
            if relative_mads[route_id] is not None and relative_mads[route_id] > 0.15:
                route_warnings.append("relative_mad_above_0_15")
        warnings[route_id] = route_warnings
    return medians, relative_mads, warnings


def _diagnostic_summary(
    *,
    manifest: Mapping[str, Any],
    samples: tuple[Mapping[str, Any], ...],
    sessions: tuple[Mapping[str, Any], ...],
    report: Mapping[str, Any],
    selection_sha256: str,
) -> Mapping[str, object]:
    configuration = manifest["configuration"]["experiment"]
    reasons = list(_diagnostic_contract_reasons(configuration))
    expected_attempts = _diagnostic_expected_attempts()
    observed_attempts = {
        (
            sample.get("case_id"),
            sample.get("plan_id"),
            sample.get("route_id"),
            sample.get("attempt_kind"),
            sample.get("sample_index"),
            sample.get("block_id"),
        )
        for sample in samples
    }
    all_blocks_complete = len(samples) == len(expected_attempts) and observed_attempts == expected_attempts
    if not all_blocks_complete:
        reasons.append("diagnostic_block_matrix_incomplete")
    observed_routes = {sample.get("route_id") for sample in samples}
    all_routes_successful = (
        observed_routes == set(_DIAGNOSTIC_ROUTE_IDS)
        and len(samples) == len(expected_attempts)
        and all(sample.get("status") == "success" for sample in samples)
    )
    if not all_routes_successful:
        reasons.append("diagnostic_route_or_status_incomplete")
    expected_measurement_pairs = {
        (route_id, block_id)
        for route_id in _DIAGNOSTIC_ROUTE_IDS
        for block_id in range(1, 6)
    }
    observed_measurement_pairs = {
        (sample.get("route_id"), sample.get("block_id"))
        for sample in samples
        if sample.get("attempt_kind") == "measurement" and sample.get("status") == "success"
    }
    complete_pairs = observed_measurement_pairs == expected_measurement_pairs
    if not complete_pairs:
        reasons.append("diagnostic_measurement_pairs_incomplete")
    validation_passed = True
    resource_admission_passed = True
    physical_provenance_passed = True
    sessions_by_id = {
        str(session.get("session_instance_id")): session for session in sessions
    }
    for sample in samples:
        validation = sample.get("validation")
        if not isinstance(validation, Mapping) or validation.get("accuracy_qualified") is not True:
            validation_passed = False
        elif (
            validation.get("policy_reference_applicable") is True
            and validation.get("policy_reference_passed") is not True
        ):
            validation_passed = False
        route_id = sample.get("route_id")
        if route_id not in _DIAGNOSTIC_PHYSICAL_ROUTE_IDS:
            continue
        if not isinstance(validation, Mapping) or validation.get("policy_reference_passed") is not True:
            validation_passed = False
        try:
            facts = _joined_sample_facts(sample, sessions_by_id)
        except ValueError:
            resource_admission_passed = False
            physical_provenance_passed = False
            continue
        if (
            facts.get("startup_resource_admission_passed") is not True
            or facts.get("execution_resource_admission_passed") is not True
        ):
            resource_admission_passed = False
        required_provenance = {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
        }
        if any(facts.get(field) != expected for field, expected in required_provenance.items()):
            physical_provenance_passed = False
    physical_sessions = [
        session for session in sessions if session.get("route_id") in _DIAGNOSTIC_PHYSICAL_ROUTE_IDS
    ]
    if len(physical_sessions) != 24:
        physical_provenance_passed = False
        resource_admission_passed = False
    for session in physical_sessions:
        facts = session.get("terminal_backend_facts")
        if (
            session.get("status") != "success"
            or session.get("release_verified") is not True
            or not isinstance(facts, Mapping)
            or facts.get("target_observed") != "physical_hardware"
        ):
            physical_provenance_passed = False
    if not validation_passed:
        reasons.append("diagnostic_validation_failed")
    if not resource_admission_passed:
        reasons.append("diagnostic_resource_admission_failed")
    if not physical_provenance_passed:
        reasons.append("diagnostic_physical_provenance_failed")
    binary_hashes: Mapping[str, Mapping[str, str]] | None = None
    binding_sha256: str | None = None
    try:
        binary_hashes = _route_binary_hashes_from_sessions(sessions)
        binding_sha256 = _campaign_binding_sha256(
            configuration,
            source_commit=str(manifest["source_commit"]),
            selection_sha256=selection_sha256,
            binary_hashes=binary_hashes,
        )
    except (KeyError, TypeError, ValueError):
        reasons.append("diagnostic_campaign_binding_unavailable")
    medians, relative_mads, warnings = _measurement_statistics(samples)
    if report.get("schema_version") != "evidence_report_v5":
        raise ValueError("M7C scaling report must use evidence_report_v5")
    gate_passed = (
        not _diagnostic_contract_reasons(configuration)
        and all_routes_successful
        and all_blocks_complete
        and complete_pairs
        and validation_passed
        and resource_admission_passed
        and physical_provenance_passed
        and binding_sha256 is not None
    )
    return {
        "schema_version": _DIAGNOSTIC_SUMMARY_SCHEMA,
        "source_commit": manifest["source_commit"],
        "diagnostic_config_sha256": _hash(configuration),
        "selection_sha256": selection_sha256,
        "campaign_binding_sha256": binding_sha256,
        "expected_route_ids": list(_DIAGNOSTIC_ROUTE_IDS),
        "expected_block_ids": list(range(6)),
        "all_routes_successful": all_routes_successful,
        "all_blocks_complete": all_blocks_complete,
        "validation_passed": validation_passed,
        "resource_admission_passed": resource_admission_passed,
        "physical_provenance_passed": physical_provenance_passed,
        "median_runtime_s": medians,
        "relative_mad": relative_mads,
        "measurement_warnings": warnings,
        "gate_passed": gate_passed,
        "gate_reasons": sorted(set(reasons)),
    }


def _write_summary(path: Path, summary: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(summary), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _validate_diagnostic_summary(
    *, selection: Path, configuration: Mapping[str, object], summary_path: Path
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required_fields = {
        "schema_version",
        "source_commit",
        "diagnostic_config_sha256",
        "selection_sha256",
        "campaign_binding_sha256",
        "expected_route_ids",
        "expected_block_ids",
        "all_routes_successful",
        "all_blocks_complete",
        "validation_passed",
        "resource_admission_passed",
        "physical_provenance_passed",
        "median_runtime_s",
        "relative_mad",
        "measurement_warnings",
        "gate_passed",
        "gate_reasons",
    }
    if not isinstance(summary, Mapping) or set(summary) != required_fields:
        raise ValueError("M7C diagnostic summary has an unsupported contract")
    if summary["schema_version"] != _DIAGNOSTIC_SUMMARY_SCHEMA:
        raise ValueError("M7C diagnostic summary has an unsupported schema")
    if summary["gate_passed"] is not True:
        raise ValueError("M7C performance campaign requires a passed diagnostic summary")
    if summary["source_commit"] != _source_commit():
        raise ValueError("M7C diagnostic summary source commit does not match HEAD")
    selection_sha256 = _file_sha256(selection)
    if summary["selection_sha256"] != selection_sha256:
        raise ValueError("M7C diagnostic summary selection does not match the current file")
    if summary["expected_route_ids"] != list(_DIAGNOSTIC_ROUTE_IDS) or summary[
        "expected_block_ids"
    ] != list(range(6)):
        raise ValueError("M7C diagnostic summary route or block matrix is not preregistered")
    expected_binding = _campaign_binding_sha256(
        configuration,
        source_commit=_source_commit(),
        selection_sha256=selection_sha256,
        binary_hashes=_route_binary_hashes_from_config(configuration),
    )
    if summary["campaign_binding_sha256"] != expected_binding:
        raise ValueError("M7C diagnostic summary does not match the performance campaign")


def _validate_physical_campaign(
    *,
    configuration: Mapping[str, object],
    samples: tuple[Mapping[str, Any], ...],
    sessions: tuple[Mapping[str, Any], ...],
) -> Mapping[str, object]:
    physical_ids = {
        route_id
        for route_id, route in configuration["routes"].items()
        if route["executor"] == "upmem_physical"
    }
    if not physical_ids:
        raise ValueError("M7C campaign evidence has no physical route")
    attempts_per_route = (
        configuration["collection"]["warmup_blocks"]
        + configuration["collection"]["measurement_blocks"]
    )
    expected_sessions = len(physical_ids) * attempts_per_route
    if len(sessions) != expected_sessions:
        raise ValueError("M7C fresh-session campaign has an unexpected session count")
    sessions_by_id = {
        str(session["session_instance_id"]): session for session in sessions
    }
    for sample in samples:
        if sample["route_id"] not in physical_ids:
            continue
        facts = _joined_sample_facts(sample, sessions_by_id)
        validation = sample["validation"]
        if not isinstance(validation, Mapping):
            raise ValueError("physical scaling sample lacks validation")
        required = {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "startup_resource_admission_passed": True,
            "execution_resource_admission_passed": True,
        }
        for field, expected in required.items():
            if facts.get(field) != expected:
                raise ValueError(f"physical scaling sample requires {field}={expected!r}")
        if validation.get("policy_reference_passed") is not True:
            raise ValueError("physical scaling sample failed policy-reference validation")
        if validation.get("accuracy_qualified") is not True:
            raise ValueError("float32 physical scaling sample is not accuracy-qualified")
    for session in sessions:
        if session["route_id"] not in physical_ids:
            continue
        facts = session["terminal_backend_facts"]
        if session["release_verified"] is not True or not isinstance(facts, Mapping):
            raise ValueError("physical scaling session lacks verified release facts")
        if facts.get("target_observed") != "physical_hardware":
            raise ValueError("physical scaling session did not observe hardware")
    return {
        "physical_route_count": len(physical_ids),
        "expected_fresh_sessions": expected_sessions,
    }


def _validate_performance_report(
    *, report_dir: Path, configuration: Mapping[str, object]
) -> Mapping[str, object]:
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("schema_version") != "evidence_report_v5":
        raise ValueError("M7C scaling report must use evidence_report_v5")
    result: dict[str, object] = {
        "report_schema_version": report.get("schema_version"),
        "scaling_count": report.get("scaling_count"),
    }
    if configuration["collection"]["claim_policy"] != "physical_performance_v1":
        return result
    with (report_dir / "scaling.csv").open(newline="", encoding="utf-8") as stream:
        scaling_rows = list(csv.DictReader(stream))
    primary = [row for row in scaling_rows if row.get("comparison_role") == "primary"]
    required_primary = {
        ("tasklet_scaling", "1", "8"),
        ("dpu_scaling", "1", "2"),
        ("dpu_scaling", "1", "4"),
    }
    observed_primary = {
        (
            row.get("comparison_kind"),
            row.get("baseline_tasklet_count")
            if row.get("comparison_kind") == "tasklet_scaling"
            else row.get("baseline_dpu_count"),
            row.get("candidate_tasklet_count")
            if row.get("comparison_kind") == "tasklet_scaling"
            else row.get("candidate_dpu_count"),
        )
        for row in primary
    }
    if observed_primary != required_primary or any(
        row.get("claim_eligible") != "True" for row in primary
    ):
        raise ValueError("physical performance scaling report failed its claim gate")
    result["primary_scaling_claims_eligible"] = True
    return result


def run_campaign(
    *,
    selection: Path,
    config: Path,
    output: Path,
    report_output: Path,
    diagnostic_summary: Path | None = None,
) -> Mapping[str, object]:
    """Run the one permitted mixed route command after immutable prechecks."""

    _clean_worktree()
    normalized = load_experiment_config(config)
    if not _physical_route_ids(normalized):
        raise ValueError("M7C campaign config must include physical routes")
    claim_policy = normalized["collection"]["claim_policy"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
    _selector_check(selection, config, env)
    if claim_policy == "physical_performance_v1":
        if diagnostic_summary is None:
            raise ValueError("physical_performance_v1 requires --diagnostic-summary")
        _validate_diagnostic_summary(
            selection=selection,
            configuration=normalized,
            summary_path=diagnostic_summary,
        )
    elif diagnostic_summary is not None:
        raise ValueError("diagnostic_v1 must not receive --diagnostic-summary")
    if output.exists() or report_output.exists():
        raise ValueError("M7C campaign outputs must be absent")
    _run(
        [
            sys.executable,
            "-m",
            "quantum_bench.cli",
            "run",
            "--config",
            str(config),
            "--output",
            str(output),
            "--allow-physical",
        ],
        env=env,
    )
    _run(
        [sys.executable, "-m", "quantum_bench.cli", "verify", "--input", str(output)],
        env=env,
    )
    _run(
        [
            sys.executable,
            "-m",
            "quantum_bench.cli",
            "report",
            "--input",
            str(output),
            "--output",
            str(report_output),
        ],
        env=env,
    )
    summary_output = (
        report_output / "diagnostic_summary.json"
        if claim_policy == "diagnostic_v1"
        else None
    )
    return inspect_campaign(
        input_dir=output,
        report_dir=report_output,
        selection=selection,
        summary_output=summary_output,
    )


def inspect_campaign(
    *,
    input_dir: Path,
    report_dir: Path,
    selection: Path,
    summary_output: Path | None = None,
) -> Mapping[str, object]:
    """Inspect a mixed campaign and derive a diagnostic summary when applicable."""

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
    _selector_check(selection, None, env)
    manifest, samples, sessions = load_artifacts(input_dir)
    verification = verify_artifacts(input_dir)
    if verification.get("status") != "completed":
        raise ValueError("M7C campaign artifact is not completed")
    configuration = manifest["configuration"]["experiment"]
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    selection_sha256 = _file_sha256(selection)
    if configuration["collection"]["claim_policy"] == "diagnostic_v1":
        summary = _diagnostic_summary(
            manifest=manifest,
            samples=samples,
            sessions=sessions,
            report=report,
            selection_sha256=selection_sha256,
        )
        if summary_output is not None:
            _write_summary(summary_output, summary)
        if summary["gate_passed"] is not True:
            raise ValueError("M7C diagnostic gate failed: " + ", ".join(summary["gate_reasons"]))
        return summary
    if summary_output is not None:
        raise ValueError("only diagnostic_v1 may write a diagnostic summary")
    result: dict[str, object] = {
        "status": "completed",
        "verification": verification,
        **_validate_physical_campaign(
            configuration=configuration,
            samples=samples,
            sessions=sessions,
        ),
        **_validate_performance_report(report_dir=report_dir, configuration=configuration),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--rank-path", action="append", required=True)
    prepare.add_argument("--session-root", required=True)
    prepare.add_argument("--expected-cpus", required=True)
    run = commands.add_parser("run")
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report-output", type=Path, required=True)
    run.add_argument("--diagnostic-summary", type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--report-output", type=Path, required=True)
    inspect.add_argument("--selection", type=Path, required=True)
    inspect.add_argument("--summary-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_config(
                template=args.template.resolve(),
                output=args.output.resolve(),
                rank_paths=args.rank_path,
                session_root=args.session_root,
                expected_cpus=_parse_cpus(args.expected_cpus),
            )
            payload: Mapping[str, object] = {"status": "prepared", "output": str(output)}
        elif args.command == "run":
            payload = run_campaign(
                selection=args.selection.resolve(),
                config=args.config.resolve(),
                output=args.output.resolve(),
                report_output=args.report_output.resolve(),
                diagnostic_summary=(
                    args.diagnostic_summary.resolve()
                    if args.diagnostic_summary is not None
                    else None
                ),
            )
        else:
            payload = inspect_campaign(
                input_dir=args.input.resolve(),
                report_dir=args.report_output.resolve(),
                selection=args.selection.resolve(),
                summary_output=(
                    args.summary_output.resolve() if args.summary_output is not None else None
                ),
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
