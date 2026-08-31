#!/usr/bin/env python3
"""Measure request-record construction cost and evaluate a template go/no-go gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

from quantum_bench.upmem.protocol import (
    NUMERIC_FLOAT32,
    V4Profile,
    V4WorkUnit,
    V4WorkUnitRecord,
    _safe_relative,
    build_v4_request,
)


CELL_DEFINITIONS = (
    ("ghz_chain_18q", 1),
    ("ghz_chain_18q", 4),
    ("hs_18q_d1", 1),
    ("hs_18q_d1", 4),
    ("quantization_stress_18q_l2", 1),
    ("quantization_stress_18q_l2", 4),
)
COUNT_POINTS = (1, 2, 4, 8, 16, 32, 64)
PATH_LENGTH_POINTS = (32, 64, 128, 256)
DEFAULT_REPEATS = 7
TASK_HASH = "ab" * 32
ABI_FIELDS = (
    "local_dpu_id",
    "flags",
    "tile_id",
    "batch_index",
    "m_offset",
    "n_offset",
    "k_offset",
    "m_elements",
    "n_elements",
    "k_elements",
    "a_transfer_bytes",
    "b_transfer_bytes",
    "c_transfer_bytes",
    "a_offset_bytes",
    "b_offset_bytes",
    "c_offset_bytes",
)
CSV_FIELDS = (
    "kind",
    "case_id",
    "dpu_count",
    "record_count",
    "repeat_count",
    "median_full_record_s",
    "raw_mad_full_record_s",
    "min_full_record_s",
    "max_full_record_s",
    "median_template_fill_s",
    "raw_mad_template_fill_s",
    "min_template_fill_s",
    "max_template_fill_s",
    "template_equivalent",
    "q_estimate",
    "q_lower_bound",
    "physical_record_share",
    "physical_noise_floor",
    "predicted_attempt_reduction",
    "passes_noise_gate",
    "factor",
    "factor_value",
    "median_ns",
    "raw_mad_ns",
)


def _raw_mad(values: list[float]) -> float:
    center = statistics.median(values)
    return float(statistics.median(abs(value - center) for value in values))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "raw_mad": _raw_mad(values),
        "min": min(values),
        "max": max(values),
    }


def _payload(length: int, seed: int) -> bytes:
    return bytes((seed + index) % 251 for index in range(length))


def _work_units(dpu_count: int, *, seed: int = 1) -> list[V4WorkUnit]:
    payload = _payload(12, seed)
    return [
        V4WorkUnit(
            local_dpu_id=dpu_id,
            tile_id=1000 + dpu_id,
            batch_index=0,
            m_offset=dpu_id,
            n_offset=0,
            k_offset=0,
            m_elements=1,
            n_elements=1,
            k_elements=3,
            a_payload=payload,
            b_payload=payload,
        )
        for dpu_id in range(dpu_count)
    ]


def _build(root: Path, dpu_count: int, sequence: int) -> Any:
    return build_v4_request(
        root,
        profile=V4Profile(dpu_count=dpu_count, numeric_mode=NUMERIC_FLOAT32),
        canonical_batch_count=1,
        canonical_m=dpu_count,
        canonical_n=1,
        canonical_k=3,
        work_units=_work_units(dpu_count),
        task_contract_sha256=TASK_HASH,
        request_sequence=sequence,
    )


def _fill_template(records: tuple[V4WorkUnitRecord, ...]) -> tuple[float, tuple[V4WorkUnitRecord, ...]]:
    static_values = [tuple(getattr(record, field) for field in ABI_FIELDS) for record in records]
    dynamic_values = [
        (record.a_path, record.b_path, record.c_path, record.a_sha256, record.b_sha256)
        for record in records
    ]
    started = time.perf_counter_ns()
    filled = tuple(
        V4WorkUnitRecord(
            *static,
            a_path=dynamic[0],
            b_path=dynamic[1],
            c_path=dynamic[2],
            a_sha256=dynamic[3],
            b_sha256=dynamic[4],
        )
        for static, dynamic in zip(static_values, dynamic_values, strict=True)
    )
    elapsed = (time.perf_counter_ns() - started) / 1_000_000_000
    return elapsed, filled


def _measure_cell(
    root: Path, *, case_id: str, dpu_count: int, repeats: int
) -> dict[str, Any]:
    full_values: list[float] = []
    template_values: list[float] = []
    equivalent = True
    record_count = 0
    for index in range(repeats + 1):
        artifact = _build(root / case_id / str(dpu_count) / str(index), dpu_count, index)
        template_s, filled = _fill_template(artifact.work_units)
        if index == 0:
            continue
        full_values.append(float(artifact.payload_record_construction_s))
        template_values.append(template_s)
        record_count = len(artifact.work_units)
        equivalent = equivalent and filled == artifact.work_units
    full = _stats(full_values)
    template = _stats(template_values)
    full_lower = max(1e-12, full["median"] - 2.0 * full["raw_mad"])
    template_upper = template["median"] + 2.0 * template["raw_mad"]
    q_estimate = max(0.0, 1.0 - template["median"] / max(full["median"], 1e-12))
    q_lower_bound = max(0.0, 1.0 - template_upper / full_lower)
    return {
        "kind": "physical_geometry",
        "case_id": case_id,
        "dpu_count": dpu_count,
        "record_count": record_count,
        "repeat_count": repeats,
        "full": full,
        "template": template,
        "template_equivalent": equivalent,
        "q_estimate": q_estimate,
        "q_lower_bound": q_lower_bound,
    }


def _measure_path_lengths(repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length in PATH_LENGTH_POINTS:
        timings: list[float] = []
        value = "x" * length
        for _ in range(repeats):
            started = time.perf_counter_ns()
            for _ in range(1000):
                _safe_relative(value)
            timings.append((time.perf_counter_ns() - started) / 1000.0)
        stats = _stats(timings)
        rows.append(
            {
                "kind": "isolated_factor",
                "factor": "relative_path_length_bytes",
                "factor_value": length,
                "median_ns": stats["median"],
                "raw_mad_ns": stats["raw_mad"],
            }
        )
    return rows


def _load_attribution(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    rows = result.get("measurement_cells")
    if not isinstance(rows, list):
        raise ValueError("attribution JSON lacks measurement_cells")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), int(row["dpu_count"]))
        indexed[key] = row
    expected = set(CELL_DEFINITIONS)
    if set(indexed) != expected:
        raise ValueError("attribution JSON does not contain the six expected cells")
    return indexed


def _go_no_go(cells: list[dict[str, Any]], attribution: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for cell in cells:
        source = attribution[(cell["case_id"], cell["dpu_count"])]
        attempt = float(source.get("median_attempt_elapsed_s", source["median_total_wall_s"]))
        record = float(source["median_payload_record_construction_s"])
        p = record / attempt if attempt else 0.0
        mad_attempt = float(source.get("raw_mad_attempt_elapsed_s", source["raw_mad_total_wall_s"]))
        noise_floor = 2.0 * mad_attempt / attempt if attempt else 0.0
        predicted = p * cell["q_lower_bound"]
        decisions.append(
            {
                "case_id": cell["case_id"],
                "dpu_count": cell["dpu_count"],
                "physical_record_share": p,
                "physical_noise_floor": noise_floor,
                "predicted_attempt_reduction": predicted,
                "passes_noise_gate": predicted > noise_floor,
                "template_equivalent": cell["template_equivalent"],
                "q_estimate": cell["q_estimate"],
                "q_lower_bound": cell["q_lower_bound"],
            }
        )
    passing = [item for item in decisions if item["passes_noise_gate"]]
    circuits = {item["case_id"] for item in passing}
    go = (
        all(item["template_equivalent"] for item in decisions)
        and len(passing) >= 5
        and circuits == {case_id for case_id, _ in CELL_DEFINITIONS}
    )
    return {
        "gate_version": "request_template_amdahl_noise_v1",
        "decision": "go" if go else "no_go",
        "criterion": (
            "byte-identical template fill and predicted whole-attempt reduction "
            "above twice the within-cell raw-MAD noise floor in at least five of six "
            "cells, covering every circuit"
        ),
        "cells": decisions,
    }


def _csv_rows(cells: list[dict[str, Any]], path_rows: list[dict[str, Any]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, decision in zip(cells, gate["cells"], strict=True):
        rows.append(
            {
                "kind": cell["kind"],
                "case_id": cell["case_id"],
                "dpu_count": cell["dpu_count"],
                "record_count": cell["record_count"],
                "repeat_count": cell["repeat_count"],
                "median_full_record_s": cell["full"]["median"],
                "raw_mad_full_record_s": cell["full"]["raw_mad"],
                "min_full_record_s": cell["full"]["min"],
                "max_full_record_s": cell["full"]["max"],
                "median_template_fill_s": cell["template"]["median"],
                "raw_mad_template_fill_s": cell["template"]["raw_mad"],
                "min_template_fill_s": cell["template"]["min"],
                "max_template_fill_s": cell["template"]["max"],
                "template_equivalent": cell["template_equivalent"],
                "q_estimate": decision["q_estimate"],
                "q_lower_bound": decision["q_lower_bound"],
                "physical_record_share": decision["physical_record_share"],
                "physical_noise_floor": decision["physical_noise_floor"],
                "predicted_attempt_reduction": decision["predicted_attempt_reduction"],
                "passes_noise_gate": decision["passes_noise_gate"],
            }
        )
    rows.extend(path_rows)
    return rows


def analyze(output_dir: Path, attribution_path: Path, repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    if repeats < 3:
        raise ValueError("repeats must include at least three measured repetitions")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    attribution = _load_attribution(attribution_path)
    with tempfile.TemporaryDirectory(prefix="request-record-cost-") as temporary:
        root = Path(temporary)
        cells = [
            _measure_cell(root, case_id=case_id, dpu_count=dpu_count, repeats=repeats)
            for case_id, dpu_count in CELL_DEFINITIONS
        ]
        sweep = [
            _measure_cell(root, case_id="synthetic_record_count", dpu_count=count, repeats=repeats)
            for count in COUNT_POINTS
        ]
        path_rows = _measure_path_lengths(repeats)
    gate = _go_no_go(cells, attribution)
    result: dict[str, Any] = {
        "analysis_version": "request_record_cost_v1",
        "attribution_source": str(attribution_path),
        "repeats": repeats,
        "cells": cells,
        "record_count_sweep": sweep,
        "isolated_factors": path_rows,
        "go_no_go": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request_record_cost.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "request_record_cost.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_rows(cells, path_rows, gate))
    (output_dir / "request_template_go_no_go.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--attribution-json", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_dir, args.attribution_json, args.repeats), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
