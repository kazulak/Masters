#!/usr/bin/env python3
"""Host-only four-lane versus one-envelope probe. No native execution route."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

from quantum_bench.numerics import encode_complex_tensor
from quantum_bench.upmem.packed_operation import build_packed_v4_request, pack_operation
from quantum_bench.upmem.protocol import NUMERIC_HOST_PACKED_INT8, V4Profile, V4WorkUnit, _record_abi_fields

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("four_lane_envelopes", "one_complex_envelope")
SEED = 20260905


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded_json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def unpack_envelope(data):
    """Independent transport decoder; preserve exact embedded bytes for parity."""
    if len(data) < 96:
        raise ValueError("truncated header")
    magic, version, header, count, width, flags, reserved, start, body, size, sequence, sha = struct.unpack_from(
        "<8s6I4Q32s", data)
    if (magic, version, header, width, flags, reserved, start) != (b"UPOENV2\0", 2, 96, 200, 0, 0, 96):
        raise ValueError("unsupported header")
    if not count or count > (len(data) - 96) // 200 or body != 96 + count * 200 or size != len(data):
        raise ValueError("invalid descriptor bounds")
    if hashlib.sha256(data[:64] + bytes(32) + data[96:]).digest() != sha:
        raise ValueError("envelope digest mismatch")
    result = []
    cursor = body
    previous = sequence - 1
    for index in range(count):
        descriptor = data[96 + index * 200:96 + (index + 1) * 200]
        if hashlib.sha256(descriptor[:168] + bytes(32)).digest() != descriptor[168:]:
            raise ValueError("descriptor digest mismatch")
        seq, mo, ml, so, sl, po, pl, records, reserved, outputs = struct.unpack_from("<7Q2IQ", descriptor)
        if seq <= previous or (index == 0 and seq != sequence) or reserved or not records or not outputs:
            raise ValueError("request order or record count")
        previous = seq
        parts = []
        for i, (offset, length) in enumerate(((mo, ml), (so, sl), (po, pl))):
            if offset != cursor or not length or offset > size or length > size - offset:
                raise ValueError("invalid or overlapping region")
            part = data[offset:offset + length]
            if hashlib.sha256(part).digest() != descriptor[72 + 32 * i:104 + 32 * i]:
                raise ValueError("embedded digest mismatch")
            parts.append(part)
            cursor += length
        result.append((seq, records, outputs, *parts))
    if cursor != size:
        raise ValueError("unaccounted trailing data")
    return result


def synthetic(shape, shift):
    index = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return ((index + shift) % 31 - 15) / 17 + 1j * (((index + shift) % 23 - 11) / 13)


def prepare_operation(op, policy, dpus, root, arm, sequence):
    """Both arms perform the same encoding, slicing, templates and requests."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    setup_started = time.perf_counter_ns()
    left = synthetic((op["b"], op["m"], op["k"]), 1)
    right = synthetic((op["b"], op["k"], op["n"]), 7)
    profile = V4Profile(dpu_count=dpus, tasklets_per_dpu=8,
                        numeric_mode=NUMERIC_HOST_PACKED_INT8 if policy == "complex_int8_shared_scale_v1" else "float32")
    waves = sorted({u["wave"] for u in op["work_units"]})
    setup_ns = time.perf_counter_ns() - setup_started
    started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
    encoded_left = encode_complex_tensor(np.array(left, copy=True), policy)
    encoded_right = encode_complex_tensor(np.array(right, copy=True), policy)
    lanes = ((encoded_left.real, encoded_right.real), (encoded_left.imag, encoded_right.imag),
             (encoded_left.real, encoded_right.imag), (encoded_left.imag, encoded_right.real))
    templates = {}
    paths, expected = [], []
    pending = []
    payload_bytes = written_bytes = template_bytes = 0
    materialized_bytes = 0
    write_ns = packing_ns = 0

    def write(requests):
        nonlocal written_bytes, write_ns, packing_ns
        begin = time.perf_counter_ns()
        packed = pack_operation(root, requests=requests, operation_sequence=requests[0].request_sequence,
                                filename=f"envelope_{requests[0].request_sequence:016d}.bin")
        packing_ns += time.perf_counter_ns() - begin
        begin = time.perf_counter_ns()
        packed.path.write_bytes(packed.data)
        write_ns += time.perf_counter_ns() - begin
        paths.append(packed.path)
        written_bytes += len(packed.data)

    for lane_index, (a, b) in enumerate(lanes):
        lane_requests = []
        # Synthetic contract identity contains all input planes and scale facts.
        contract = digest(encoded_json({"node_id": op["node_id"], "lane": lane_index,
                                        "policy": policy, "left_scale": encoded_left.scale,
                                        "right_scale": encoded_right.scale}) + a.tobytes() + b.tobytes())
        for wave in waves:
            units = []
            for tile_index, unit in enumerate(op["work_units"]):
                if unit["wave"] != wave:
                    continue
                if unit["logical_rank"] != 0 or unit["batch_size"] != 1:
                    raise ValueError("probe requires accepted single-rank, one-batch work units")
                batch, m, n, k = (unit[f"{x}_start"] for x in ("batch", "m", "n", "k"))
                ms, ns, ks = (unit[f"{x}_size"] for x in ("m", "n", "k"))
                aa = np.ascontiguousarray(a[batch, m:m + ms, k:k + ks])
                bb = np.ascontiguousarray(b[batch, k:k + ks, n:n + ns])
                materialized_bytes += aa.nbytes + bb.nbytes
                units.append(V4WorkUnit(unit["logical_dpu"], tile_index, batch, m, n, k, ms, ns, ks, aa, bb))
            request = build_packed_v4_request(
                root, profile=profile, canonical_batch_count=op["b"], canonical_m=op["m"],
                canonical_n=op["n"], canonical_k=op["k"], work_units=units,
                task_contract_sha256=contract, request_sequence=sequence,
                record_templates=templates.get(wave))
            if wave not in templates:
                templates[wave] = {u.local_dpu_id: _record_abi_fields(
                    u, profile=profile, canonical_batch_count=op["b"], canonical_m=op["m"],
                    canonical_k=op["k"], canonical_n=op["n"], validate_payload=False,
                    validate_geometry=False) for u in units}
                template_bytes += sum(8 * len(t) for t in templates[wave].values())
            sequence += 1
            lane_requests.append(request)
            payload_bytes += len(request.payload_bytes)
            expected.append((request.request_sequence, len(request.work_units), request.request_output_elements,
                             request.manifest_sha256, request.sidecar_sha256, request.payload_sha256))
        if arm == ARMS[0]:
            write(lane_requests)
        else:
            pending.extend(lane_requests)
    if pending:
        write(pending)
    cpu_ns, preparation_ns = time.process_time_ns() - cpu_started, time.perf_counter_ns() - started
    validation_started = time.perf_counter_ns()
    observed, output_paths = [], []
    for path in paths:
        for seq, records, outputs, manifest, sidecar, payload in unpack_envelope(path.read_bytes()):
            observed.append((seq, records, outputs, digest(manifest), digest(sidecar), digest(payload)))
            output_paths.extend(line.split()[5] for line in manifest.decode().splitlines()[1:])
    if observed != expected or len(set(output_paths)) != len(output_paths):
        raise ValueError("embedded request equivalence or output identity failure")
    validation_ns = time.perf_counter_ns() - validation_started
    return {
        "node_id": op["node_id"], "preparation_ns": preparation_ns, "cpu_ns": cpu_ns,
        "setup_ns": setup_ns, "validation_ns": validation_ns, "packing_ns": packing_ns,
        "filesystem_write_ns": write_ns, "files_created": len(paths), "bytes_written": written_bytes,
        "payload_bytes": payload_bytes, "materialized_operand_bytes": materialized_bytes,
        "template_field_bytes_estimate": template_bytes, "request_count": len(expected),
        "work_unit_count": len(op["work_units"]), "dpu_launches_executed": 0,
        "native_processes_started": 0, "request_equivalence_sha256": digest(encoded_json(observed)),
        "output_paths_sha256": digest(encoded_json(output_paths)), "status": "passed",
    }, sequence


def run_arm(cell, arm, storage):
    observations = []
    sequence = 0
    status, error = "passed", None
    wall_started = time.perf_counter_ns()
    try:
        for op in cell["operations"]:
            root = Path(tempfile.mkdtemp(prefix="operation-", dir=storage))
            row, sequence = prepare_operation(op, cell["numeric_policy"], cell["topology"]["dpu_count"], root, arm, sequence)
            cleanup = time.perf_counter_ns()
            shutil.rmtree(root)
            row["cleanup_ns"] = time.perf_counter_ns() - cleanup
            observations.append(row)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    return {"status": status, "error": error, "operations": observations,
            "arm_process_elapsed_ns": time.perf_counter_ns() - wall_started,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "rss_scope": "fresh_arm_process_including_setup_validation_cleanup",
            "cpu_affinity": sorted(os.sched_getaffinity(0))}


def summarize(rows):
    cells = []
    for cell_id in sorted({r["cell_id"] for r in rows}):
        selected = [r for r in rows if r["cell_id"] == cell_id and r["block"] > 0]
        summary = {"cell_id": cell_id, "arms": {}}
        for arm in ARMS:
            values = [sum(o["preparation_ns"] for o in r["operations"]) for r in selected if r["arm"] == arm]
            med = statistics.median(values)
            summary["arms"][arm] = {"median_ns": med, "mad_ns": statistics.median(abs(v - med) for v in values),
                                     "minimum_ns": min(values), "maximum_ns": max(values), "measured_count": len(values)}
        ratios = []
        savings = []
        for block in sorted({r["block"] for r in selected}):
            pair = {r["arm"]: sum(o["preparation_ns"] for o in r["operations"]) for r in selected if r["block"] == block}
            ratios.append(pair[ARMS[0]] / pair[ARMS[1]])
            savings.append(pair[ARMS[0]] - pair[ARMS[1]])
        summary.update(paired_median_speedup=statistics.median(ratios), paired_median_saved_ns=statistics.median(savings))
        cells.append(summary)
    return cells


def benchmark(census_path, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    data = census_path.read_bytes()
    census = json.loads(data)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("benchmark requires clean source")
    if census["source_sha"] != sha or census["source_dirty"]:
        raise ValueError("census must come from exact clean benchmark source")
    cells = [c for c in census["cells"] if "greedy" in c["path_roles"]]
    if len(cells) != 16 or any(c["status"] != "eligible" for c in cells):
        raise ValueError("all sixteen frozen greedy cells required")
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("pin the host-only benchmark to one CPU")
    output.mkdir(parents=True, exist_ok=True)
    schedule, rng = [], random.Random(SEED)
    for block in range(8):
        for cell in cells:
            arms = list(ARMS)
            rng.shuffle(arms)
            schedule.extend({"cell_id": cell["cell_id"], "block": block, "arm": arm, "position": pos}
                            for pos, arm in enumerate(arms))
    (output / "preregistration.json").write_bytes(encoded_json({
        "source_sha": sha, "census_sha256": digest(data), "seed": SEED, "schedule": schedule,
        "host": platform.platform(), "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "filesystem_device": output.stat().st_dev, "durability": "transient_no_fsync",
        "measurement_scope": "host_only_preparation", "physical": False, "sdk": False,
        "amdahl_projection": None, "reason": "affected_physical_fraction_not_measured",
    }))
    by_id = {c["cell_id"]: c for c in cells}
    rows = []
    for index, item in enumerate(schedule):
        started = time.perf_counter_ns()
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                    input=json.dumps({"cell": by_id[item["cell_id"]], "arm": item["arm"], "storage": str(output)}),
                                    capture_output=True, text=True, timeout=600,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess([], -1, "", f"host arm timeout: {exc}")
        try:
            row = json.loads(result.stdout)
        except ValueError:
            row = {"status": "failed", "error": "worker produced no JSON", "operations": []}
        row.update(item, parent_process_elapsed_ns=time.perf_counter_ns() - started,
                   exit_status=result.returncode, stderr=result.stderr,
                   circuit_id=by_id[item["cell_id"]]["circuit_id"],
                   numeric_policy=by_id[item["cell_id"]]["numeric_policy"],
                   dpu_count=by_id[item["cell_id"]]["topology"]["dpu_count"])
        rows.append(row)
        (output / "host_batching_raw.json").write_bytes(encoded_json(rows))
        print(f"{index + 1}/{len(schedule)} {row['circuit_id']} {item['arm']} {row['status']}", flush=True)
        if result.returncode or row["status"] != "passed":
            raise ValueError("failed arm preserved; no retry or replacement")
        if item["position"] == 1:
            first, second = rows[-2:]
            for a, b in zip(first["operations"], second["operations"], strict=True):
                for key in ("node_id", "request_equivalence_sha256", "output_paths_sha256", "payload_bytes", "request_count"):
                    if a[key] != b[key]:
                        raise ValueError(f"paired equivalence failed: {key}")
                pair = {first["arm"]: a, second["arm"]: b}
                if pair[ARMS[0]]["bytes_written"] - pair[ARMS[1]]["bytes_written"] != 288:
                    raise ValueError("unexpected outer envelope byte difference")
    flat = [{**{k: r[k] for k in ("cell_id", "circuit_id", "numeric_policy", "dpu_count", "arm", "block", "position")}, **o}
            for r in rows for o in r["operations"]]
    with (output / "host_batching_raw.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / "host_batching_summary.json").write_bytes(encoded_json(summarize(rows)))
    (output / "SHA256SUMS").write_text("".join(f"{digest(p.read_bytes())}  {p.name}\n"
                                              for p in sorted(output.iterdir()) if p.is_file()), encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        request = json.load(sys.stdin)
        print(json.dumps(run_arm(request["cell"], request["arm"], request["storage"])))
    elif args.census and args.output_dir:
        # Seal a partial stage too; never turn a failure into replacement samples.
        fresh = not args.output_dir.exists() or not any(args.output_dir.iterdir())
        try:
            benchmark(args.census, args.output_dir)
        except Exception as exc:
            if fresh and args.output_dir.exists():
                (args.output_dir / "failure.json").write_bytes(encoded_json({"status": "failed", "error": str(exc)}))
            raise
        finally:
            if fresh and args.output_dir.exists():
                files = sorted(p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
                (args.output_dir / "SHA256SUMS").write_text("".join(
                    f"{digest(p.read_bytes())}  {p.relative_to(args.output_dir).as_posix()}\n" for p in files), encoding="ascii")
    else:
        parser.error("--census and --output-dir required")


if __name__ == "__main__":
    main()
