#!/usr/bin/env python3
"""Inspect a packed-operation-only six-route physical diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_parallel_scaling import inspect_artifacts as inspect_parallel_artifacts  # noqa: E402
from quantum_bench.evidence import canonical_json, load_artifacts  # noqa: E402
from quantum_bench.report import verify_artifacts  # noqa: E402


PACKED_TRANSPORT = "packed_operation_v1"
_PACKED_COUNTERS = (
    "packed_operation_count",
    "packed_operation_request_count",
    "packed_operation_max_descriptor_count",
    "packed_operation_max_bytes",
    "packed_operation_max_payload_bytes",
)


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    facts = dict(_mapping(sample.get("backend_facts"), "sample backend facts"))
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        for field, value in terminal.items():
            if field in facts and facts[field] != value:
                raise ValueError(f"sample and terminal facts conflict for {field}")
            facts.setdefault(field, value)
    return facts


def validate_packed_transport(
    samples: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Require packed transport and return compact envelope-bound facts."""

    session_map = {
        str(session.get("session_instance_id")): session for session in sessions
    }
    if len(session_map) != len(sessions):
        raise ValueError("packed diagnostic has duplicate session identities")
    route_facts: dict[str, dict[str, int]] = {}
    for sample in samples:
        facts = _joined_facts(sample, session_map)
        if facts.get("request_transport") != PACKED_TRANSPORT:
            raise ValueError("sample does not prove packed_operation_v1 transport")
        session = session_map.get(str(sample.get("session_instance_id")))
        terminal = session.get("terminal_backend_facts") if session else None
        if not isinstance(terminal, Mapping) or terminal.get("request_transport") != PACKED_TRANSPORT:
            raise ValueError("terminal session does not prove packed transport")
        counts = {
            field: _nonnegative_int(facts.get(field), field)
            for field in _PACKED_COUNTERS
        }
        if counts["packed_operation_count"] < 1:
            raise ValueError("successful sample contains no packed operation")
        if counts["packed_operation_request_count"] < 1:
            raise ValueError("successful sample contains no packed request")
        if counts["packed_operation_max_descriptor_count"] < 1:
            raise ValueError("successful sample contains no packed descriptor")
        route_id = str(sample.get("route_id"))
        current = route_facts.setdefault(
            route_id,
            {
                "samples": 0,
                "max_operation_count": 0,
                "max_request_count": 0,
                "max_descriptor_count": 0,
                "max_envelope_bytes": 0,
                "max_payload_bytes": 0,
            },
        )
        current["samples"] += 1
        current["max_operation_count"] = max(
            current["max_operation_count"], counts["packed_operation_count"]
        )
        current["max_request_count"] = max(
            current["max_request_count"], counts["packed_operation_request_count"]
        )
        current["max_descriptor_count"] = max(
            current["max_descriptor_count"],
            counts["packed_operation_max_descriptor_count"],
        )
        current["max_envelope_bytes"] = max(
            current["max_envelope_bytes"], counts["packed_operation_max_bytes"]
        )
        current["max_payload_bytes"] = max(
            current["max_payload_bytes"],
            counts["packed_operation_max_payload_bytes"],
        )
    if not route_facts:
        raise ValueError("packed diagnostic contains no samples")
    return {
        "transport": PACKED_TRANSPORT,
        "route_facts": route_facts,
        "sample_count": len(samples),
        "session_count": len(sessions),
    }


def _verify_generic_report(report_path: Path) -> Mapping[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read generic report: {report_path}") from exc
    if not isinstance(report, Mapping):
        raise ValueError("generic report must be a mapping")
    if report.get("status") != "completed":
        raise ValueError("generic report is not completed")
    if report.get("speedup_count") != 0:
        raise ValueError("diagnostic generic report emitted speedup rows")
    verification = report.get("verification")
    if isinstance(verification, Mapping):
        if verification.get("claim_eligible_aggregate_count", 0) not in (0, None):
            raise ValueError("diagnostic generic report emitted claim-eligible aggregates")
    scaling_path = report_path.with_name("scaling.csv")
    if scaling_path.is_file():
        with scaling_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if any(str(row.get("claim_eligible", "")).lower() == "true" for row in rows):
            raise ValueError("diagnostic scaling report emitted claim-eligible rows")
    return report


def inspect(
    *,
    input_dir: Path,
    summary_output: Path,
    output_dir: Path | None,
    expected_source_commit: str,
    report_path: Path | None,
) -> Mapping[str, Any]:
    verification = verify_artifacts(input_dir)
    if verification.get("status") != "completed":
        raise ValueError("packed diagnostic evidence is not completed")
    manifest, samples, sessions = load_artifacts(input_dir)
    base_summary = inspect_parallel_artifacts(
        input_dir=input_dir,
        summary_output=summary_output,
        output_dir=output_dir,
        expected_source_commit=expected_source_commit,
    )
    if base_summary.get("claim_eligible") is not False:
        raise ValueError("packed diagnostic became claim eligible")
    if base_summary.get("claim_policy") != "diagnostic_v1":
        raise ValueError("packed diagnostic claim policy changed")
    packed = validate_packed_transport(samples, sessions)
    report = _verify_generic_report(report_path) if report_path is not None else None
    summary = {
        **_plain(base_summary),
        "packed_transport": packed,
        "generic_report": {
            "speedup_count": report.get("speedup_count") if report else None,
            "claim_eligible_aggregate_count": (
                report.get("verification", {}).get("claim_eligible_aggregate_count")
                if report and isinstance(report.get("verification"), Mapping)
                else None
            ),
        },
        "gate_passed": True,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(_plain(summary), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = inspect(
            input_dir=args.input.resolve(),
            summary_output=args.summary_output.resolve(),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            expected_source_commit=args.expected_source_commit,
            report_path=args.report.resolve() if args.report else None,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
