#!/usr/bin/env python3
"""Inventory repeated host request boundaries for the fixed v1 six-cell study.

This is a read-only analysis.  It reuses the existing source-only circuit
characterization and derives request lifecycle counts from the accepted v4
Python/protocol facts.  It does not execute a circuit or build a request.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_SOURCE = "c4efb3f17e29672e91a0a844881ead53ccf9f2c7"
SCHEMA_VERSION = "host_request_boundary_inventory_v1"
CHARACTERIZATION_SCHEMA = "circuit_structure_resource_sensitivity_v1"
LANE_PASS_COUNT = 4
REQUEST_PAYLOAD_FILES_PER_RECORD = 2
REQUEST_METADATA_FILES = 2
CELL_DEFINITIONS = (
    ("Stress18", "quantization_stress_18q_l2", 1),
    ("Stress18", "quantization_stress_18q_l2", 4),
    ("HS18", "hs_18q_d1", 1),
    ("HS18", "hs_18q_d1", 4),
    ("GHZ18", "ghz_chain_18q", 1),
    ("GHZ18", "ghz_chain_18q", 4),
)
CSV_COLUMNS = (
    "cell_id",
    "case_name",
    "case_id",
    "route_id",
    "dpu_count",
    "tasklets_per_dpu",
    "contraction_operation_count",
    "work_unit_count",
    "wave_request_count",
    "embedded_request_count",
    "python_submit_count",
    "python_submit_callsite_count",
    "active_work_unit_count",
    "request_record_count",
    "request_directory_count",
    "request_payload_file_count",
    "request_metadata_file_count",
    "request_file_count",
    "process_count",
    "packed_operation_submit_estimate",
    "packed_submit_reduction_estimate",
)
SUBMIT_TOKEN = re.compile(r"\bSUBMIT\s+")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _integer(value: object, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        adjective = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {adjective}")
    return value


def _physical_plan(candidate: Mapping[str, Any], dpu_count: int) -> Mapping[str, Any]:
    plans = candidate.get("physical_plans")
    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        raise ValueError("candidate physical_plans must be a sequence")
    for plan in plans:
        item = _mapping(plan, "physical plan")
        topology = _mapping(item.get("topology"), "physical plan topology")
        if (
            topology.get("dpu_count") == dpu_count
            and topology.get("rank_count") == 1
            and topology.get("tasklets_per_dpu") == 8
        ):
            return item
    raise ValueError(f"characterization lacks a 1-rank {dpu_count}-DPU/T8 plan")


def _python_submit_callsite_count(source_root: Path = ROOT) -> int:
    path = source_root / "src" / "quantum_bench" / "upmem" / "native_session.py"
    count = len(SUBMIT_TOKEN.findall(path.read_text(encoding="utf-8")))
    if count != 1:
        raise ValueError(f"expected one Python SUBMIT callsite, found {count}")
    return count


def _build_characterization() -> Mapping[str, Any]:
    """Build the existing source-only characterization without writing it."""

    path = ROOT / "scripts" / "characterize_circuit_resources.py"
    spec = importlib.util.spec_from_file_location("host_boundary_characterization", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load characterization utility: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return _mapping(module.build_characterization(), "characterization")


def _load_characterization(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), "characterization")


def _candidate_index(characterization: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if characterization.get("schema_version") != CHARACTERIZATION_SCHEMA:
        raise ValueError("characterization schema is not recognized")
    candidates = characterization.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("characterization candidates must be a sequence")
    indexed = {}
    for candidate in candidates:
        item = _mapping(candidate, "candidate")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        indexed[candidate_id] = item
    expected = {case_id for _, case_id, _ in CELL_DEFINITIONS}
    if set(indexed) < expected:
        raise ValueError("characterization is missing a selected six-cell case")
    return indexed


def _cell(
    candidate: Mapping[str, Any],
    *,
    case_name: str,
    case_id: str,
    dpu_count: int,
    submit_callsite_count: int,
) -> dict[str, Any]:
    network = _mapping(candidate.get("tensor_network"), "tensor network")
    circuit = _mapping(candidate.get("circuit"), "circuit")
    physical = _physical_plan(candidate, dpu_count)
    topology = _mapping(physical.get("topology"), "physical plan topology")
    contraction_count = _integer(
        network.get("contraction_count"), "contraction_operation_count", positive=True
    )
    work_units = _integer(physical.get("work_unit_count"), "work_unit_count", positive=True)
    wave_count = _integer(physical.get("wave_count"), "wave_request_count", positive=True)
    embedded_count = _integer(
        physical.get("estimated_native_request_count_four_real_products"),
        "embedded_request_count",
        positive=True,
    )
    expected_embedded = LANE_PASS_COUNT * wave_count
    if embedded_count != expected_embedded:
        raise ValueError(
            f"{case_id}/{dpu_count} embedded request count does not match four lane passes"
        )
    record_count = embedded_count * dpu_count
    payload_file_count = record_count * REQUEST_PAYLOAD_FILES_PER_RECORD
    metadata_file_count = embedded_count * REQUEST_METADATA_FILES
    packed_estimate = contraction_count
    return {
        "cell_id": f"{case_id}:{dpu_count}dpu:t8",
        "case_name": case_name,
        "case_id": case_id,
        "route_id": f"upmem_float32_{dpu_count}dpu_t8",
        "dpu_count": dpu_count,
        "tasklets_per_dpu": _integer(topology.get("tasklets_per_dpu"), "tasklets_per_dpu", positive=True),
        "contraction_operation_count": contraction_count,
        "work_unit_count": work_units,
        "wave_request_count": wave_count,
        "embedded_request_count": embedded_count,
        "python_submit_count": embedded_count,
        "python_submit_callsite_count": submit_callsite_count,
        "active_work_unit_count": work_units,
        "request_record_count": record_count,
        "request_directory_count": embedded_count,
        "request_payload_file_count": payload_file_count,
        "request_metadata_file_count": metadata_file_count,
        "request_file_count": payload_file_count + metadata_file_count,
        "process_count": 1,
        "packed_operation_submit_estimate": packed_estimate,
        "packed_submit_reduction_estimate": embedded_count - packed_estimate,
        "circuit_name": _mapping(circuit, "circuit").get("name"),
    }


def build_inventory(
    characterization: Mapping[str, Any] | None = None,
    *,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    """Return the deterministic inventory; this function has no file outputs."""

    facts = _candidate_index(
        _build_characterization() if characterization is None else characterization
    )
    submit_callsite_count = _python_submit_callsite_count(source_root)
    cells = [
        _cell(
            facts[case_id],
            case_name=case_name,
            case_id=case_id,
            dpu_count=dpu_count,
            submit_callsite_count=submit_callsite_count,
        )
        for case_name, case_id, dpu_count in CELL_DEFINITIONS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_source_commit": ACCEPTED_SOURCE,
        "characterization_source_commit": characterization.get("source_sha")
        if characterization is not None
        else ACCEPTED_SOURCE,
        "execution": {
            "kind": "read_only_source_analysis",
            "hardware_executed": False,
            "simulator_executed": False,
            "packed_envelope_implemented": False,
        },
        "basis": {
            "characterization": "characterize_circuit_resources.py source-only physical plans",
            "embedded_request": "four complex real-lane passes per active rank wave",
            "python_submit": "one V4Session.submit call per embedded request at one rank",
        "records": "one dense v4 record per DPU in each embedded request; active work units are reported separately",
            "request_files": "two payload files per record plus manifest.txt and sidecar.bin per request",
            "processes": "one persistent native process for the one-rank session",
            "packed_operation_submit": "estimate only: one proposed packed submit per contraction operation",
        },
        "cells": cells,
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {column: row[column] for column in CSV_COLUMNS}
        for row in inventory["cells"]
    ]


def _markdown(inventory: Mapping[str, Any]) -> str:
    lines = [
        "# Host Request Boundary Reduction Feasibility v1",
        "",
        "Read-only deterministic inventory for the six selected 18-qubit cells.",
        "The packed-operation column is an estimate only; no packed envelope or runtime path is implemented.",
        "",
        "| Cell | Route | Contractions | Waves | Embedded requests | Python SUBMIT | Records | Request dirs | Request files | Processes | Packed submits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inventory["cells"]:
        lines.append(
            "| {case_name} | {dpu_count}-DPU/T8 | {contraction_operation_count} | "
            "{wave_request_count} | {embedded_request_count} | {python_submit_count} | "
            "{request_record_count} | {request_directory_count} | {request_file_count} | "
            "{process_count} | {packed_operation_submit_estimate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Count Basis",
            "",
            "- Embedded requests are four complex lane passes for each active rank wave.",
            "- Python `SUBMIT` count is one `V4Session.submit` invocation per embedded request.",
        "- Records are dense: each embedded request carries one record per configured DPU; active logical work units are reported separately.",
            "- Request files include two payload files per record, one manifest, and one sidecar per request.",
            "- Process count is one persistent native process for the single rank.",
            "- Packed submits estimate one future packed submit per contraction operation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory(
    json_output: Path,
    csv_output: Path,
    markdown_output: Path,
    *,
    characterization: Mapping[str, Any] | None = None,
    source_root: Path = ROOT,
) -> tuple[Path, Path, Path]:
    """Write outputs only to the three explicitly supplied paths."""

    outputs = (Path(json_output), Path(csv_output), Path(markdown_output))
    if len(set(outputs)) != len(outputs):
        raise ValueError("JSON, CSV, and Markdown output paths must be distinct")
    inventory = build_inventory(characterization, source_root=source_root)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_bytes(_json_bytes(inventory))
    with outputs[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_rows(inventory))
    outputs[2].write_text(_markdown(inventory), encoding="utf-8")
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--characterization", type=Path)
    args = parser.parse_args(argv)
    try:
        characterization = (
            _load_characterization(args.characterization)
            if args.characterization is not None
            else None
        )
        paths = write_inventory(
            args.json_output.resolve(),
            args.csv_output.resolve(),
            args.markdown_output.resolve(),
            characterization=characterization,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "written", "outputs": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
