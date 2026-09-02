#!/usr/bin/env python3
"""Prepare and inspect the bounded quantized UPMEM v1 experiments.

This operator tool never opens hardware. Execution remains an explicit
``quantum_bench.cli run|qualify --allow-physical`` action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
POLICY = "complex_int8_shared_scale_v1"
FLOAT32 = "split_complex_float32_v1"
PACKED = "packed_operation_v1"
KINDS = {
    "simulator": ("quantized-upmem-simulator-v1", 4, 4),
    "pilot": ("quantized-upmem-physical-pilot-v1", 4, 4),
    "diagnostic": ("quantized-upmem-physical-diagnostic-v1", 180, 180),
}
PATH_FIELDS = ("host_binary", "dpu_binary", "initialization_binary")


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise ValueError("cannot determine source HEAD")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(root: Path) -> Path:
    """Write sorted relative checksums without changing canonical evidence."""

    if not root.is_dir():
        raise ValueError(f"evidence directory is missing: {root}")
    target = root / "SHA256SUMS"
    entries = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != target
    ]
    target.write_text("\n".join(entries) + "\n", encoding="ascii")
    return target


def verify_checksums(root: Path) -> None:
    target = root / "SHA256SUMS"
    if not target.is_file():
        raise ValueError("SHA256SUMS is missing")
    observed: set[str] = set()
    lines = target.read_text(encoding="ascii").splitlines()
    if lines != sorted(lines, key=lambda line: line.split("  ", 1)[1]):
        raise ValueError("SHA256SUMS entries are not sorted")
    for line in lines:
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not name:
            raise ValueError("malformed SHA256SUMS entry")
        path = root / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {name}")
        observed.add(name)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != target
    }
    if observed != expected:
        raise ValueError("SHA256SUMS inventory differs from evidence directory")


def prepare(
    template: Path,
    output: Path,
    *,
    rank_path: str,
    session_root: Path,
    expected_cpu: int,
    binary_root: Path,
) -> Path:
    if output.exists():
        raise ValueError(f"prepared configuration already exists: {output}")
    config = _plain(load_experiment_config(template))
    identity = config.pop("experiment_identity_payload", None)
    if not isinstance(identity, Mapping) or not isinstance(identity.get("label"), str):
        raise ValueError("template lacks its experiment identity label")
    config["experiment_id"] = identity["label"]
    config.pop("collection_policy_id", None)
    physical = [
        (route_id, route)
        for route_id, route in config["routes"].items()
        if route["executor"] == "upmem_physical"
    ]
    if not physical:
        raise ValueError("template has no physical routes")
    config["collection"]["machine_policy"]["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": [expected_cpu],
    }
    for route_id, route in physical:
        options = route["options"]
        tasklets = int(options["tasklets_per_dpu"])
        options["rank_paths"] = [str(Path(rank_path).resolve())]
        options["session_root"] = str((session_root / route_id).resolve())
        options["host_binary"] = str(
            (binary_root / f"host_upmem_execution_plan_v4_t{tasklets}").resolve()
        )
        options["dpu_binary"] = str(
            (binary_root / f"dpu_gemm_tile_v4_t{tasklets}").resolve()
        )
        options["initialization_binary"] = str(
            (binary_root / f"dpu_simplepim_management_init_t{tasklets}").resolve()
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    load_experiment_config(output)
    return output


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("sample lacks backend facts")
    result = dict(facts)
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        for key, value in terminal.items():
            result.setdefault(key, value)
    return result


def inspect(root: Path, *, kind: str, expected_source: str) -> dict[str, Any]:
    experiment_id, expected_samples, expected_sessions = KINDS[kind]
    manifest, samples, session_rows = load_artifacts(root)
    verification = verify_artifacts(root)
    required = {
        "status": "completed",
        "sample_count": expected_samples,
        "session_count": expected_sessions,
        "success_count": expected_samples,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for field, expected in required.items():
        if verification.get(field) != expected:
            raise ValueError(f"unexpected {field}: {verification.get(field)!r}")
    if manifest.get("source_commit") != expected_source:
        raise ValueError("evidence source SHA does not match expected source")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("execution source was dirty")
    config = manifest["configuration"]["experiment"]
    identity = config.get("experiment_identity_payload")
    label = identity.get("label") if isinstance(identity, Mapping) else None
    if label != experiment_id:
        raise ValueError("experiment identity does not match the selected gate")
    sessions = {str(row["session_instance_id"]): row for row in session_rows}
    if len(sessions) != expected_sessions:
        raise ValueError("session IDs are not unique")
    physical = kind != "simulator"
    for row in session_rows:
        facts = row.get("terminal_backend_facts")
        if row.get("status") != "success" or row.get("release_verified") is not True:
            raise ValueError("session failed or release was not verified")
        if not isinstance(facts, Mapping):
            raise ValueError("session lacks terminal facts")
        expected_target = "physical_hardware" if physical else "sdk_simulator"
        if facts.get("target_observed") != expected_target:
            raise ValueError("session target provenance is wrong")
        if facts.get("cpu_fallback_used") is not False:
            raise ValueError("CPU fallback was reported")
        if physical and (
            facts.get("physical_target_verified") is not True
            or facts.get("hardware_kernel_executed") is not True
        ):
            raise ValueError("physical execution was not verified")
        if not physical and facts.get("simulator_kernel_executed") is not True:
            raise ValueError("simulator kernel was not executed")
        for field in ("binary_identity_verified", "native_identity_verified"):
            if facts.get(field) is not True:
                raise ValueError(f"session requires {field}=true")
    observed: set[tuple[str, str]] = set()
    for sample in samples:
        route_id = str(sample["route_id"])
        case_id = str(sample["case_id"])
        observed.add((case_id, route_id))
        route = config["routes"][route_id]
        policy = route["numeric_policy"]
        numeric = sample.get("numeric_facts")
        validation = sample.get("validation")
        facts = _joined_facts(sample, sessions)
        if not isinstance(numeric, Mapping) or numeric.get("numeric_policy") != policy:
            raise ValueError("sample numerical policy identity is wrong")
        if not isinstance(validation, Mapping) or validation.get("policy_reference_passed") is not True:
            raise ValueError("sample did not match CPU same-policy replay")
        if policy == FLOAT32 and validation.get("accuracy_qualified") is not True:
            raise ValueError("float32 sample failed complex128 qualification")
        if policy == POLICY and validation.get("full_precision_threshold_applicable") is not False:
            raise ValueError("int8 sample incorrectly applied an accuracy threshold")
        if sample.get("output_sha256") is None:
            raise ValueError("sample output hash is missing")
        if facts.get("request_transport") != PACKED:
            raise ValueError("sample did not use packed operation transport")
        options = route["options"]
        expected_resource = {
            "requested_dpus": options["dpu_count"],
            "allocated_dpus": options["dpu_count"],
            "tasklets_per_dpu": options["tasklets_per_dpu"],
        }
        for field, expected in expected_resource.items():
            if facts.get(field) != expected:
                raise ValueError(f"sample resource mismatch for {field}")
        if facts.get("cpu_fallback_used") is not False:
            raise ValueError("sample reported fallback")
    expected_pairs = {
        (str(item["case_id"]), str(route_id))
        for item in config["matrix"]
        for route_id in item["route_ids"]
    }
    if observed != expected_pairs:
        raise ValueError("observed case/route matrix is incomplete")
    return {
        "status": "passed",
        "kind": kind,
        "source_commit": expected_source,
        "experiment_id": manifest["experiment_id"],
        "sample_count": len(samples),
        "session_count": len(session_rows),
        "matrix_cell_count": len(expected_pairs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--template", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--rank-path", default="/dev/dpu_rank1")
    prep.add_argument("--session-root", type=Path, required=True)
    prep.add_argument("--expected-cpu", type=int, default=0)
    prep.add_argument(
        "--binary-root", type=Path, default=ROOT / "native/upmem/runtime/bin"
    )
    check = commands.add_parser("inspect")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--kind", choices=tuple(KINDS), required=True)
    check.add_argument("--expected-source", default=None)
    sums = commands.add_parser("checksums")
    sums.add_argument("--input", type=Path, required=True)
    sums.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = {
                "status": "prepared",
                "output": str(
                    prepare(
                        args.template.resolve(),
                        args.output.resolve(),
                        rank_path=args.rank_path,
                        session_root=args.session_root,
                        expected_cpu=args.expected_cpu,
                        binary_root=args.binary_root,
                    )
                ),
            }
        elif args.command == "inspect":
            result = inspect(
                args.input.resolve(),
                kind=args.kind,
                expected_source=args.expected_source or _git_head(),
            )
        elif args.verify:
            verify_checksums(args.input.resolve())
            result = {"status": "verified"}
        else:
            result = {"status": "written", "path": str(write_checksums(args.input.resolve()))}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
