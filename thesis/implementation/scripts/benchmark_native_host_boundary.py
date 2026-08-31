#!/usr/bin/env python3
"""Compare deterministic Python and C prepared-stage serialization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from typing import Any


PACKET_HEADER = struct.Struct("<8sIIIIIIQ")
RECORD = struct.Struct("<IIQQQQQIIIIIIIII")
MAGIC = b"NHPV1\0\0\0"
LANE_COUNT = 4
VERSION = 1


def _record(index: int, payload_bytes: int) -> bytes:
    del payload_bytes
    return RECORD.pack(
        index % 64,
        0,
        index,
        index % 4,
        index * 2,
        index * 3,
        0,
        1,
        1,
        1,
        4,
        4,
        4,
        (index * 8) % (512 * 1024 - 8),
        (index * 8 + 8) % (512 * 1024 - 8),
        (index * 8 + 16) % (512 * 1024 - 8),
    )


def build_fixture(name: str, record_count: int, payload_bytes: int) -> dict[str, Any]:
    records = tuple(_record(index, payload_bytes) for index in range(record_count))
    payload = bytes((index * 17 + len(name)) % 256 for index in range(payload_bytes))
    header = PACKET_HEADER.pack(
        MAGIC,
        VERSION,
        PACKET_HEADER.size,
        record_count,
        LANE_COUNT,
        1,
        payload_bytes,
        20260831,
    )
    packet = header + b"".join(records) + payload
    canonical = b"".join(records) + payload
    return {
        "name": name,
        "record_count": record_count,
        "payload_bytes": payload_bytes,
        "packet": packet,
        "canonical": canonical,
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _python_arm(fixture: dict[str, Any], iterations: int) -> dict[str, Any]:
    records = fixture["packet"][PACKET_HEADER.size : PACKET_HEADER.size + fixture["record_count"] * RECORD.size]
    payload = fixture["packet"][PACKET_HEADER.size + fixture["record_count"] * RECORD.size :]
    started = time.perf_counter()
    result = b""
    digest = ""
    for _ in range(iterations):
        result_builder = bytearray()
        for offset in range(0, len(records), RECORD.size):
            result_builder.extend(records[offset : offset + RECORD.size])
        result_builder.extend(payload)
        result = bytes(result_builder)
        digest = hashlib.sha256(result).hexdigest()
    return {
        "steady_s": time.perf_counter() - started,
        "canonical_sha256": digest,
        "canonical_bytes": len(result),
    }


def _run_c(binary: Path, fixture: dict[str, Any], iterations: int, root: Path) -> dict[str, Any]:
    packet_path = root / f"{fixture['name']}.packet"
    output_path = root / f"{fixture['name']}.output"
    packet_path.write_bytes(fixture["packet"])
    started = time.perf_counter()
    result = subprocess.run(
        [str(binary), str(packet_path), str(output_path), str(iterations)],
        check=False,
        capture_output=True,
        text=True,
    )
    process_s = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "prepared-stage probe failed")
    facts = json.loads(result.stdout)
    if facts.get("packet_sha256") != fixture["packet_sha256"]:
        raise ValueError(f"C packet hash mismatch: {fixture['name']}")
    facts["process_s"] = process_s
    facts["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return facts


def run(binary: Path, output_dir: Path, iterations: int) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    fixtures = (
        build_fixture("stress_1d_t8", 888, 1_528_416),
        build_fixture("stress_4d_t8", 2_544, 1_528_416),
        build_fixture("hs_1d_t8", 224, 1_528_416),
        build_fixture("ghz_4d_t8", 1_856, 1_528_416),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="native-host-boundary-") as temporary:
        root = Path(temporary)
        for fixture in fixtures:
            python_facts = _python_arm(fixture, iterations)
            c_facts = _run_c(binary, fixture, iterations, root)
            if python_facts["canonical_sha256"] != fixture["canonical_sha256"]:
                raise ValueError(f"Python canonical hash mismatch: {fixture['name']}")
            if c_facts["canonical_sha256"] != fixture["canonical_sha256"]:
                raise ValueError(f"C canonical hash mismatch: {fixture['name']}")
            if c_facts["output_sha256"] != fixture["canonical_sha256"]:
                raise ValueError(f"C output hash mismatch: {fixture['name']}")
            rows.append(
                {
                    "fixture": fixture["name"],
                    "record_count": fixture["record_count"],
                    "payload_bytes": fixture["payload_bytes"],
                    "canonical_bytes": len(fixture["canonical"]),
                    "iterations": iterations,
                    "python_steady_s": python_facts["steady_s"],
                    "c_setup_s": c_facts["setup_s"],
                    "c_steady_s": c_facts["steady_s"],
                    "c_process_s": c_facts["process_s"],
                    "packet_sha256": fixture["packet_sha256"],
                    "canonical_sha256": fixture["canonical_sha256"],
                    "equivalent": True,
                }
            )
    result = {
        "analysis_version": "native_host_feasibility_v1",
        "iterations": iterations,
        "fixtures": rows,
        "scope": "host-only deterministic serialization; no UPMEM SDK or DPU allocation",
    }
    (output_dir / "native_host_feasibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = tuple(rows[0])
    with (output_dir / "native_host_feasibility.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Native Host Feasibility",
        "",
        "Host-only deterministic prepared-stage serialization; no SDK or DPU allocation.",
        "",
        "| Fixture | Records | Payload bytes | Python steady (s) | C setup (s) | C steady (s) | Process (s) | Equivalent |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['fixture']} | {row['record_count']} | {row['payload_bytes']} | "
            f"{row['python_steady_s']:.6f} | {row['c_setup_s']:.6f} | "
            f"{row['c_steady_s']:.6f} | {row['c_process_s']:.6f} | yes |"
        )
    (output_dir / "native_host_feasibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    print(json.dumps(run(args.binary, args.output_dir, args.iterations), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
