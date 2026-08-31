#!/usr/bin/env python3
"""Compare old and optimized payload staging on matched diagnostic cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from quantum_bench.evidence import load_artifacts
from quantum_bench.report import _TERMINAL_AUTHORITY_FIELDS, verify_artifacts

try:
    from analyze_m7d_attribution import _sample_components
except ImportError:  # pragma: no cover
    _sample_components = None


CASES = ("quantization_stress_18q_l2", "hs_18q_d1", "ghz_chain_18q")
ROUTES = ("upmem_float32_1dpu_t8", "upmem_float32_4dpu_t8")
BLOCKS = (0, 1, 2, 3, 4, 5)
MEASUREMENTS = (1, 2, 3, 4, 5)
COMPONENTS = (
    "total_wall_s",
    "kernel_s",
    "payload_record_staging_s",
    "host_request_overhead_s",
    "native_request_overhead_s",
    "h2d_s",
    "d2h_s",
    "accounting_residual_s",
)


def _mad(values: list[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    facts = dict(sample.get("backend_facts", {}))
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        conflicts = sorted(
            field
            for field in _TERMINAL_AUTHORITY_FIELDS
            if field in facts and field in terminal and facts[field] != terminal[field]
        )
        if conflicts:
            raise ValueError(
                "terminal physical facts conflict for " + ", ".join(conflicts)
            )
        for field, value in terminal.items():
            facts.setdefault(field, value)
    return facts


def _load(path: Path, expected_source: str) -> tuple[dict[str, Any], dict[tuple[str, str, str, int], dict[str, Any]]]:
    verification = verify_artifacts(path)
    expected = {
        "status": "completed",
        "sample_count": 36,
        "session_count": 36,
        "success_count": 36,
        "failed_count": 0,
        "unsupported_count": 0,
        "accuracy_qualified": True,
    }
    for field, value in expected.items():
        if verification.get(field) != value:
            raise ValueError(f"{path}: {field} is {verification.get(field)!r}")
    manifest, samples, sessions = load_artifacts(path)
    if manifest.get("source_commit") != expected_source:
        raise ValueError(f"{path}: source commit does not match {expected_source}")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError(f"{path}: source worktree was dirty")
    by_id = {str(session["session_instance_id"]): session for session in sessions}
    if len(by_id) != 36:
        raise ValueError(f"{path}: session IDs are not unique")
    result: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for sample in samples:
        key = (
            str(sample.get("case_id")),
            str(sample.get("route_id")),
            str(sample.get("attempt_kind")),
            int(sample.get("block_id")),
        )
        if key in result or key[:2] not in {(case, route) for case in CASES for route in ROUTES}:
            raise ValueError(f"{path}: unexpected or duplicate sample {key}")
        if sample.get("status") != "success" or sample.get("session_instance_id") not in by_id:
            raise ValueError(f"{path}: unsuccessful or unbound sample {key}")
        facts = _joined_facts(sample, by_id)
        for field, value in (
            ("physical_target_verified", True),
            ("hardware_kernel_executed", True),
            ("simulator_kernel_executed", False),
            ("cpu_fallback_used", False),
            ("execution_resource_admission_passed", True),
        ):
            if facts.get(field) != value:
                raise ValueError(f"{path}: {key} has invalid {field}")
        if sample.get("measurement", {}).get("scope_id") != "steady_execution_v1":
            raise ValueError(f"{path}: {key} has a different timing scope")
        result[key] = sample
    expected_keys = {
        (case, route, kind, block)
        for case in CASES
        for route in ROUTES
        for kind, block in (("warmup", 0), *(('measurement', block) for block in MEASUREMENTS))
    }
    if set(result) != expected_keys:
        raise ValueError(f"{path}: case/route/block matrix is incomplete")
    return manifest, result


def _scientific_configuration(manifest: dict[str, Any]) -> dict[str, Any]:
    experiment = json.loads(json.dumps(manifest["configuration"]["experiment"]))
    experiment.pop("experiment_id", None)
    experiment.pop("label", None)
    experiment.pop("experiment_identity_payload", None)
    for route in experiment.get("routes", {}).values():
        options = route.get("options", {})
        for field in ("session_root", "host_binary", "dpu_binary", "initialization_binary"):
            options.pop(field, None)
    return experiment


def _cell(samples: dict[tuple[str, str, str, int], dict[str, Any]], case: str, route: str) -> dict[str, Any]:
    values = [samples[(case, route, "measurement", block)] for block in MEASUREMENTS]
    if _sample_components is None:
        raise ValueError("request attribution helper is unavailable")
    measurements = [sample["measurement"] for sample in values]
    components = [_sample_components(sample) for sample in values]
    if any(component is None for component in components):
        raise ValueError(f"{case}/{route}: request attribution is missing")
    rows: dict[str, dict[str, float]] = {}
    for field in COMPONENTS:
        series = [
            float(measurement[field]) if field in measurement else float(component[field])
            for measurement, component in zip(measurements, components)
        ]
        rows[field] = {"median_s": float(median(series)), "raw_mad_s": _mad(series)}
    return rows


def inspect(
    *, baseline: Path, candidate: Path, baseline_source: str, candidate_source: str, output_dir: Path
) -> dict[str, Any]:
    baseline_manifest, baseline_samples = _load(baseline, baseline_source)
    candidate_manifest, candidate_samples = _load(candidate, candidate_source)
    if _scientific_configuration(baseline_manifest) != _scientific_configuration(candidate_manifest):
        raise ValueError("scientific configurations differ between A/B runs")
    cells: list[dict[str, Any]] = []
    for case in CASES:
        for route in ROUTES:
            old = _cell(baseline_samples, case, route)
            new = _cell(candidate_samples, case, route)
            payload_old = old["payload_record_staging_s"]["median_s"]
            payload_new = new["payload_record_staging_s"]["median_s"]
            total_old = old["total_wall_s"]["median_s"]
            total_new = new["total_wall_s"]["median_s"]
            cells.append(
                {
                    "case_id": case,
                    "route": route,
                    "baseline": old,
                    "candidate": new,
                    "payload_ratio": payload_new / payload_old,
                    "total_wall_ratio": total_new / total_old,
                }
            )
    reduced = [cell["payload_ratio"] <= 0.90 for cell in cells]
    stable_payload = [cell["payload_ratio"] <= 1.05 for cell in cells]
    stable_total = [cell["total_wall_ratio"] <= 1.05 for cell in cells]
    summary = {
        "analysis_version": "payload_staging_ab_diagnostic_v1",
        "baseline_source_commit": baseline_source,
        "candidate_source_commit": candidate_source,
        "baseline_experiment_id": baseline_manifest["experiment_id"],
        "candidate_experiment_id": candidate_manifest["experiment_id"],
        "case_ids": list(CASES),
        "route_ids": list(ROUTES),
        "measurement_blocks": list(MEASUREMENTS),
        "cell_count": len(cells),
        "cells": cells,
        "gate_passed": sum(reduced) >= 5 and all(stable_payload) and all(stable_total),
        "gate_rule": "payload median <= 0.90 in at least 5/6 cells; payload and total-wall ratios <= 1.05 in every cell",
        "claim_eligible": False,
        "claim_ineligibility_reason": "diagnostic_claim_policy",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "payload_staging_ab_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = ("case_id", "route", "payload_old_median_s", "payload_new_median_s", "payload_ratio", "total_old_median_s", "total_new_median_s", "total_wall_ratio")
    with (output_dir / "payload_staging_ab.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "case_id": cell["case_id"],
                    "route": cell["route"],
                    "payload_old_median_s": cell["baseline"]["payload_record_staging_s"]["median_s"],
                    "payload_new_median_s": cell["candidate"]["payload_record_staging_s"]["median_s"],
                    "payload_ratio": cell["payload_ratio"],
                    "total_old_median_s": cell["baseline"]["total_wall_s"]["median_s"],
                    "total_new_median_s": cell["candidate"]["total_wall_s"]["median_s"],
                    "total_wall_ratio": cell["total_wall_ratio"],
                }
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = inspect(
            baseline=args.baseline.resolve(),
            candidate=args.candidate.resolve(),
            baseline_source=args.baseline_source,
            candidate_source=args.candidate_source,
            output_dir=args.output_dir.resolve(),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "inspected", "gate_passed": summary["gate_passed"]}, sort_keys=True))
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
