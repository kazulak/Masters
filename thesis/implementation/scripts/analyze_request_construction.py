#!/usr/bin/env python3
"""Analyze the measured request payload/record construction boundary."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from quantum_bench.evidence import load_artifacts
from quantum_bench.report import verify_artifacts


ANALYSIS_VERSION = "request_construction_attribution_v1"
EPSILON_S = 1e-6
CHILDREN = (
    "payload_materialization_s",
    "payload_file_write_s",
    "payload_hashing_s",
    "payload_record_construction_s",
)
TIMING_FIELDS = (
    "request_payload_record_staging_sum_s",
    "request_payload_materialization_sum_s",
    "request_payload_file_write_sum_s",
    "request_payload_hashing_sum_s",
    "request_payload_record_construction_sum_s",
    "request_payload_record_count",
    "request_payload_files_created",
    "request_payload_bytes_staged",
    "request_payload_bytes_hashed",
)
COUNTER_FIELDS = (
    "request_payload_record_count",
    "request_payload_files_created",
    "request_payload_bytes_staged",
    "request_payload_bytes_hashed",
)
CSV_FIELDS = (
    "case_id",
    "route_id",
    "dpu_count",
    "tasklets_per_dpu",
    "measurement_count",
    "median_total_wall_s",
    "raw_mad_total_wall_s",
    "median_payload_record_staging_s",
    "raw_mad_payload_record_staging_s",
    "payload_parent_share",
    "median_payload_materialization_s",
    "raw_mad_payload_materialization_s",
    "median_payload_file_write_s",
    "raw_mad_payload_file_write_s",
    "median_payload_hashing_s",
    "raw_mad_payload_hashing_s",
    "median_payload_record_construction_s",
    "raw_mad_payload_record_construction_s",
    "median_payload_residual_s",
    "raw_mad_payload_residual_s",
    "median_record_count",
    "median_files_created",
    "median_bytes_staged",
    "median_bytes_hashed",
    "dominant_child",
    "dominant_child_median_s",
    "dominant_child_parent_share",
)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _seconds(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nonnegative(value: float, field: str) -> float:
    if value < -EPSILON_S:
        raise ValueError(f"{field} is materially negative")
    return max(0.0, value)


def _raw_mad(values: Sequence[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _operation_attribution(operation: Mapping[str, Any]) -> dict[str, float | int]:
    timing = _mapping(operation.get("timing"), "operation timing")
    missing = [field for field in TIMING_FIELDS if field not in timing]
    if missing:
        raise ValueError(f"operation timing is missing {missing[0]}")
    parent = _seconds(
        timing["request_payload_record_staging_sum_s"],
        "request_payload_record_staging_sum_s",
    )
    children = {
        name: _seconds(timing[field], field)
        for name, field in zip(
            CHILDREN,
            TIMING_FIELDS[1:5],
            strict=True,
        )
    }
    residual = _nonnegative(
        parent - sum(children.values()), "request payload residual"
    )
    counters = {
        field: _integer(timing[field], field) for field in COUNTER_FIELDS
    }
    if counters["request_payload_files_created"] != (
        2 * counters["request_payload_record_count"]
    ):
        raise ValueError("payload file count does not match record count")
    if counters["request_payload_bytes_hashed"] != counters["request_payload_bytes_staged"]:
        raise ValueError("hashed bytes do not match staged bytes")
    return {
        "parent_s": parent,
        **children,
        "residual_s": residual,
        **counters,
    }


def _sample_attribution(sample: Mapping[str, Any]) -> dict[str, Any]:
    facts = _mapping(sample.get("backend_facts"), "sample backend_facts")
    operations = facts.get("operation_facts")
    if not isinstance(operations, Sequence) or isinstance(operations, (str, bytes)):
        raise ValueError("operation_facts must be a sequence")
    if not operations:
        raise ValueError("operation_facts must not be empty")
    measurement = _mapping(sample.get("measurement"), "sample measurement")
    result: dict[str, Any] = {
        "case_id": sample.get("case_id"),
        "route_id": sample.get("route_id"),
        "block_id": sample.get("block_id"),
        "total_wall_s": _seconds(measurement.get("total_wall_s"), "total_wall_s"),
    }
    totals: dict[str, float | int] = {
        "parent_s": 0.0,
        **{name: 0.0 for name in CHILDREN},
        "residual_s": 0.0,
        **{field: 0 for field in COUNTER_FIELDS},
    }
    topology: tuple[object, object] | None = None
    for index, raw_operation in enumerate(operations):
        operation = _mapping(raw_operation, f"operation_facts[{index}]")
        attribution = _operation_attribution(operation)
        for field, value in attribution.items():
            totals[field] += value
        current_topology = (
            operation.get("requested_dpu_count"),
            operation.get("tasklets_per_dpu"),
        )
        if topology is None:
            topology = current_topology
        elif topology != current_topology:
            raise ValueError("request construction topology changed within sample")
    if result["case_id"] is None or result["route_id"] is None:
        raise ValueError("sample case_id and route_id are required")
    if topology is None or any(value is None for value in topology):
        raise ValueError("request construction topology facts are required")
    result.update(totals)
    result["dpu_count"], result["tasklets_per_dpu"] = topology
    return result


def _summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_values = [float(value["total_wall_s"]) for value in values]
    parent_values = [float(value["parent_s"]) for value in values]
    child_stats: dict[str, dict[str, float]] = {}
    for child in CHILDREN:
        child_values = [float(value[child]) for value in values]
        child_stats[child] = {
            "median_s": float(median(child_values)),
            "raw_mad_s": _raw_mad(child_values),
            "median_parent_share": float(
                median(
                    value[child] / value["parent_s"] if value["parent_s"] else 0.0
                    for value in values
                )
            ),
            "median_total_share": float(
                median(
                    value[child] / value["total_wall_s"]
                    if value["total_wall_s"]
                    else 0.0
                    for value in values
                )
            ),
        }
    dominant = max(CHILDREN, key=lambda child: child_stats[child]["median_s"])
    return {
        "measurement_count": len(values),
        "median_total_wall_s": float(median(total_values)),
        "raw_mad_total_wall_s": _raw_mad(total_values),
        "median_parent_s": float(median(parent_values)),
        "raw_mad_parent_s": _raw_mad(parent_values),
        "parent_share": float(
            median(
                value["parent_s"] / value["total_wall_s"]
                if value["total_wall_s"]
                else 0.0
                for value in values
            )
        ),
        "children": child_stats,
        "residual": {
            "median_s": float(median(value["residual_s"] for value in values)),
            "raw_mad_s": _raw_mad([float(value["residual_s"]) for value in values]),
        },
        "counters": {
            field: {
                "median": float(median(value[field] for value in values)),
                "raw_mad": _raw_mad([float(value[field]) for value in values]),
            }
            for field in COUNTER_FIELDS
        },
        "dominant_child": dominant,
    }


def _row(case_id: str, route_id: str, values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = _summary(values)
    children = summary["children"]
    dominant = summary["dominant_child"]
    return {
        "case_id": case_id,
        "route_id": route_id,
        "dpu_count": values[0]["dpu_count"],
        "tasklets_per_dpu": values[0]["tasklets_per_dpu"],
        "measurement_count": summary["measurement_count"],
        "median_total_wall_s": summary["median_total_wall_s"],
        "raw_mad_total_wall_s": summary["raw_mad_total_wall_s"],
        "median_payload_record_staging_s": summary["median_parent_s"],
        "raw_mad_payload_record_staging_s": summary["raw_mad_parent_s"],
        "payload_parent_share": summary["parent_share"],
        **{
            key: children[child][field]
            for child in CHILDREN
            for key, field in (
                (f"median_{child}", "median_s"),
                (f"raw_mad_{child}", "raw_mad_s"),
            )
        },
        "median_payload_residual_s": summary["residual"]["median_s"],
        "raw_mad_payload_residual_s": summary["residual"]["raw_mad_s"],
        "median_record_count": summary["counters"]["request_payload_record_count"]["median"],
        "median_files_created": summary["counters"]["request_payload_files_created"]["median"],
        "median_bytes_staged": summary["counters"]["request_payload_bytes_staged"]["median"],
        "median_bytes_hashed": summary["counters"]["request_payload_bytes_hashed"]["median"],
        "dominant_child": dominant,
        "dominant_child_median_s": children[dominant]["median_s"],
        "dominant_child_parent_share": children[dominant]["median_parent_share"],
        "summary": summary,
    }


def derive_attribution(
    manifest: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive request-construction facts from successful measurement samples."""

    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("manifest source_commit is invalid")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("manifest experiment_id is invalid")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in samples:
        if sample.get("status") != "success" or sample.get("attempt_kind") != "measurement":
            continue
        attribution = _sample_attribution(sample)
        key = (str(attribution["case_id"]), str(attribution["route_id"]))
        grouped.setdefault(key, []).append(attribution)
    if not grouped:
        raise ValueError("evidence contains no successful measurements")
    rows = [
        _row(case_id, route_id, grouped[(case_id, route_id)])
        for case_id, route_id in sorted(grouped)
    ]
    dominant_children = {row["dominant_child"] for row in rows}
    if len(dominant_children) == 1:
        next_action = f"Bounded optimization of {next(iter(dominant_children))}."
    else:
        next_action = "No stable dominant child; do not optimize yet."
    return {
        "analysis_version": ANALYSIS_VERSION,
        "source_commit": source_commit,
        "experiment_id": experiment_id,
        "measurement_cells": rows,
        "next_action": next_action,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    rows = result["measurement_cells"]
    lines = [
        "# Request Construction Attribution",
        "",
        f"Source: `{result['source_commit']}`  ",
        f"Experiment: `{result['experiment_id']}`  ",
        "Measurements only; warmups are excluded. Children are disjoint "
        "subregions of `request_payload_record_staging_s`.",
        "",
        "| Circuit | Route | Total wall (s) | Payload parent (s) | Parent share | Dominant child | Child (s) | Child share | Residual (s) |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {route_id} | {median_total_wall_s:.6f} | "
            "{median_payload_record_staging_s:.6f} | {payload_parent_share:.1%} | "
            "{dominant_child} | {dominant_child_median_s:.6f} | "
            "{dominant_child_parent_share:.1%} | {median_payload_residual_s:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Accounting invariant:",
            "",
            "```text",
            "request_payload_record_staging_s >= sum(disjoint children)",
            "request_payload_residual_s = parent - sum(disjoint children)",
            "```",
            "",
            f"Next action: {result['next_action']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"analysis output directory must be empty: {output_dir}")
    verification = verify_artifacts(input_dir)
    manifest, samples, _ = load_artifacts(input_dir)
    result = derive_attribution(manifest, samples)
    result["verification"] = verification
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request_construction_attribution.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "request_construction_attribution.csv", result["measurement_cells"])
    _write_markdown(output_dir / "request_construction_analysis.md", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.input, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
