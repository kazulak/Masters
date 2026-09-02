#!/usr/bin/env python3
"""Strict descriptive analysis of frozen UPMEM path evaluation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping


METRICS = (
    "total_wall_s",
    "session_inclusive_s",
    "kernel_s",
    "h2d_s",
    "d2h_s",
    "request_build_s",
    "request_wave_s",
)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain objects")
    return rows


def _median(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("timing observations must be finite and nonnegative")
    return float(statistics.median(values))


def _mad(values: list[float]) -> float:
    center = _median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _operation_total(sample: Mapping[str, Any], name: str) -> float:
    facts = sample["backend_facts"].get("operation_facts", [])
    values = [item.get("timing", {}).get(name, 0.0) for item in facts]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"operation timing {name} is invalid")
    return float(sum(values))


def _expected(
    dataset: Mapping[str, Any], selection: Mapping[str, Any], split: str
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    if selection.get("schema_version") != "upmem_path_frozen_selection_v1":
        raise ValueError("selection schema is invalid")
    if selection.get("split") != split or selection.get("timing_used_for_selection") is not False:
        raise ValueError("selection split or timing-independence contract is invalid")
    dataset_sha = _sha256(_canonical_bytes(dataset))
    if selection.get("candidate_set_sha256") != dataset_sha:
        raise ValueError("selection candidate set does not match dataset")
    circuits = {item["circuit_id"]: item for item in dataset["circuits"]}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    paths: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selection["selections"]:
        cell = (row["circuit_id"], row["topology_id"])
        if cell in cells or circuits[cell[0]]["split"] != split:
            raise ValueError("selection cell is duplicated or belongs to another split")
        cells[cell] = row
        candidates = {
            item["candidate_path_id"]: item for item in circuits[cell[0]]["candidates"]
        }
        for role, field in (
            ("greedy", "greedy_path_id"),
            ("minimum_flops", "minimum_flops_path_id"),
            ("upmem_selected", "upmem_selected_path_id"),
        ):
            candidate_id = row[field]
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ValueError("selection references an unknown candidate")
            topology = next(
                item for item in candidate["topologies"] if item["topology_id"] == cell[1]
            )
            admission = topology.get("resource_admission", {})
            if topology.get("feasible") is not True or admission.get(
                "collection_resource_admission_passed"
            ) is not True:
                raise ValueError("selection references an inadmissible candidate")
            key = (*cell, candidate_id)
            paths.setdefault(
                key,
                {
                    "logical_plan_id": candidate["logical_plan_id"],
                    "physical_plan_id": topology["physical_plan_id"],
                    "dpu_count": topology["topology"]["dpu_count"],
                    "tasklets_per_dpu": topology["topology"]["tasklets_per_dpu"],
                    "roles": [],
                },
            )["roles"].append(role)
    return cells, paths


def analyze(
    raw_dir: Path,
    candidate_path: Path,
    profile_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    split: str,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    dataset = _load(candidate_path)
    profile = _load(profile_path)
    selection = _load(selection_path)
    candidate_sha = _sha256(_canonical_bytes(dataset))
    profile_sha = _sha256(_canonical_bytes(profile))
    if profile.get("candidate_set_sha256") != candidate_sha:
        raise ValueError("profile candidate set does not match dataset")
    if selection.get("fitted_profile_sha256") != profile_sha:
        raise ValueError("selection does not match fitted profile")
    cells, expected_paths = _expected(dataset, selection, split)

    raw_dir = Path(raw_dir)
    manifest = _load(raw_dir / "manifest.json")
    samples = _jsonl(raw_dir / "samples.jsonl")
    sessions = _jsonl(raw_dir / "sessions.jsonl")
    collection = manifest.get("configuration", {}).get("experiment", {}).get(
        "collection", {}
    )
    if (
        manifest.get("status") != "completed"
        or manifest.get("source_worktree_dirty") is not False
        or collection.get("claim_policy") != "diagnostic_v1"
        or collection.get("warmup_blocks") != 1
        or collection.get("measurement_blocks") != 5
    ):
        raise ValueError("manifest collection contract is invalid")
    expected_matrix = set(expected_paths)
    actual_matrix = {
        (item["case_id"], route, item["plan_id"].removeprefix("path_"))
        for item in manifest["configuration"]["experiment"]["matrix"]
        for route in item["route_ids"]
    }
    if actual_matrix != expected_matrix:
        raise ValueError("manifest matrix does not match frozen selection")
    sessions_by_id = {item["session_instance_id"]: item for item in sessions}
    expected_count = len(expected_paths) * 6
    if len(samples) != expected_count or len(sessions) != expected_count:
        raise ValueError("sample/session count does not match frozen evaluation")
    if len(sessions_by_id) != len(sessions):
        raise ValueError("session IDs are not unique")

    observed: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    for sample in samples:
        candidate_id = sample["plan_id"].removeprefix("path_")
        key = (sample["case_id"], sample["route_id"], candidate_id)
        expected = expected_paths.get(key)
        if expected is None:
            raise ValueError("sample is outside frozen selection")
        block = sample["block_id"]
        attempt = sample["attempt_kind"]
        if (attempt, block) not in {("warmup", 0), *(("measurement", i) for i in range(1, 6))}:
            raise ValueError("sample block schedule is invalid")
        identity = (*key, block)
        if identity in seen:
            raise ValueError("duplicate evaluation observation")
        seen.add(identity)
        session = sessions_by_id.get(sample["session_instance_id"])
        if session is None:
            raise ValueError("sample session is missing")
        if sample.get("status") != "success" or session.get("status") != "success":
            raise ValueError("evaluation contains a failed attempt")
        if (
            sample.get("experiment_id") != manifest.get("experiment_id")
            or sample.get("run_id") != manifest.get("run_id")
        ):
            raise ValueError("sample experiment/run identity does not match manifest")
        if not all(session.get(field) is True for field in (
            "release_attempted", "release_succeeded", "release_verified"
        )):
            raise ValueError("evaluation session release is not verified")
        for field in ("experiment_id", "run_id", "case_id", "plan_id", "route_id"):
            if session.get(field) != sample.get(field):
                raise ValueError("sample/session identity mismatch")
        facts = dict(sample["backend_facts"])
        terminal = session.get("terminal_backend_facts", {})
        if not isinstance(terminal, dict):
            raise ValueError("session terminal backend facts are missing")
        for field, value in terminal.items():
            facts.setdefault(field, value)
        required = {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "collection_resource_admission_passed": True,
            "execution_resource_admission_passed": True,
            "startup_resource_admission_passed": True,
            "request_transport": "packed_operation_v1",
        }
        if any(facts.get(field) != value for field, value in required.items()):
            raise ValueError("sample physical contract is not qualified")
        if any(
            facts.get(field) != value
            for field, value in {
                "requested_dpus": expected["dpu_count"],
                "allocated_dpus": expected["dpu_count"],
                "active_dpus": expected["dpu_count"],
                "tasklets_per_dpu": expected["tasklets_per_dpu"],
            }.items()
        ):
            raise ValueError("sample resource facts do not match selected topology")
        validation = sample["validation"]
        if (
            validation.get("accuracy_qualified") is not True
            or validation.get("full_precision_passed") is not True
            or validation.get("policy_reference_passed") is not True
        ):
            raise ValueError("sample numerical validation is not qualified")
        identities = sample["identities"]
        if (
            identities.get("logical_plan_id") != expected["logical_plan_id"]
            or identities.get("physical_plan_id") != expected["physical_plan_id"]
        ):
            raise ValueError("sample plan identity does not match frozen candidate")
        measurement = sample["measurement"]
        if measurement.get("scope_id") != "steady_execution_v1":
            raise ValueError("sample timing scope is invalid")
        if attempt == "measurement":
            row = {
                "total_wall_s": float(measurement["total_wall_s"]),
                "session_inclusive_s": float(session["open_s"])
                + float(measurement["total_wall_s"])
                + float(session["session_close_s"]),
                "kernel_s": float(measurement["kernel_s"]),
                "h2d_s": float(measurement["h2d_s"]),
                "d2h_s": float(measurement["d2h_s"]),
                "request_build_s": _operation_total(sample, "request_build_sum_s"),
                "request_wave_s": _operation_total(sample, "request_wave_wall_sum_s"),
            }
            if any(not math.isfinite(value) or value < 0 for value in row.values()):
                raise ValueError("sample contains invalid timing")
            observed.setdefault(key, []).append(row)
    expected_seen = {(*key, block) for key in expected_paths for block in range(6)}
    if seen != expected_seen or any(len(rows) != 5 for rows in observed.values()):
        raise ValueError("evaluation observations are incomplete")

    path_rows = []
    stats: dict[tuple[str, str, str], dict[str, float]] = {}
    for key in sorted(expected_paths):
        summary: dict[str, Any] = {
            "circuit_id": key[0],
            "split": split,
            "topology_id": key[1],
            "candidate_path_id": key[2],
            "roles": sorted(expected_paths[key]["roles"]),
            "measurement_count": 5,
        }
        for metric in METRICS:
            values = [row[metric] for row in observed[key]]
            summary[f"median_{metric}"] = _median(values)
            summary[f"raw_mad_{metric}"] = _mad(values)
        stats[key] = summary
        path_rows.append(summary)

    comparisons = []
    for cell, selection_row in sorted(cells.items()):
        greedy = stats[(*cell, selection_row["greedy_path_id"])]
        flop = stats[(*cell, selection_row["minimum_flops_path_id"])]
        selected = stats[(*cell, selection_row["upmem_selected_path_id"])]
        row: dict[str, Any] = {
            "circuit_id": cell[0],
            "split": split,
            "topology_id": cell[1],
            "greedy_path_id": selection_row["greedy_path_id"],
            "minimum_flops_path_id": selection_row["minimum_flops_path_id"],
            "upmem_selected_path_id": selection_row["upmem_selected_path_id"],
            "upmem_score": selection_row["upmem_score"],
            "explanation": selection_row["explanation"],
        }
        for name, path_summary in (
            ("greedy", greedy),
            ("minimum_flops", flop),
            ("upmem_selected", selected),
        ):
            for metric in METRICS:
                row[f"{name}_median_{metric}"] = path_summary[f"median_{metric}"]
                row[f"{name}_raw_mad_{metric}"] = path_summary[f"raw_mad_{metric}"]
        for metric in ("total_wall_s", "session_inclusive_s", "kernel_s"):
            selected_value = selected[f"median_{metric}"]
            row[f"upmem_vs_greedy_{metric}_speedup"] = (
                greedy[f"median_{metric}"] / selected_value
            )
            row[f"upmem_vs_flops_{metric}_speedup"] = (
                flop[f"median_{metric}"] / selected_value
            )
        row["upmem_vs_greedy_total_wall_percent_reduction"] = 100.0 * (
            1.0 - 1.0 / row["upmem_vs_greedy_total_wall_s_speedup"]
        )
        comparisons.append(row)
    speedups = [row["upmem_vs_greedy_total_wall_s_speedup"] for row in comparisons]
    geometric = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    result = {
        "schema_version": f"upmem_path_heuristic_{split}_v1",
        "split": split,
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": manifest["source_commit"],
        "candidate_set_sha256": candidate_sha,
        "fitted_profile_sha256": profile_sha,
        "selection_sha256": _sha256(_canonical_bytes(selection)),
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "timing_scope": "steady_execution_v1",
        "claim_policy": "diagnostic_v1",
        "sample_count": len(samples),
        "session_count": len(sessions),
        "all_successful": True,
        "path_statistics": path_rows,
        "comparisons": comparisons,
        "geometric_mean_upmem_vs_greedy_speedup": geometric,
        "minimum_cell_speedup": min(speedups),
        "maximum_cell_speedup": max(speedups),
        "improved_cell_count": sum(value > 1.0 for value in speedups),
        "unchanged_cell_count": sum(value == 1.0 for value in speedups),
        "regressed_cell_count": sum(value < 1.0 for value in speedups),
        "raw_artifact_sha256": {
            name: _sha256((raw_dir / name).read_bytes())
            for name in ("manifest.json", "samples.jsonl", "sessions.jsonl")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"heuristic_{split}.json").write_bytes(_canonical_bytes(result))
    columns = [key for key in comparisons[0] if key != "explanation"] + [
        "explanation_json"
    ]
    with (output_dir / "heuristic_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in comparisons:
            writer.writerow(
                {
                    **{key: value for key, value in row.items() if key != "explanation"},
                    "explanation_json": json.dumps(
                        row["explanation"], sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--candidate-paths", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.raw_dir,
        args.candidate_paths,
        args.profile,
        args.selection,
        args.output_dir,
        split=args.split,
    )
    print(
        json.dumps(
            {
                "geometric_mean_speedup": result[
                    "geometric_mean_upmem_vs_greedy_speedup"
                ],
                "sample_count": result["sample_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
