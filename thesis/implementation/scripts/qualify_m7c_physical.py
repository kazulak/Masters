#!/usr/bin/env python3
"""Prepare and inspect bounded M7C one-DPU physical qualifications.

The script never opens UPMEM hardware itself.  ``prepare`` writes an ignored,
machine-specific configuration from a tracked template; the operator then uses
the physical-only ``quantum_bench.cli qualify`` command.  ``inspect`` verifies
the resulting canonical evidence before any later physical attempt proceeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Mapping

import yaml

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
_PATH_FIELDS = (
    "session_root",
    "host_binary",
    "dpu_binary",
)
_NUMERIC_POLICIES = frozenset(
    {"split_complex_float32_v1", "split_complex_int8_shared_scale_v1"}
)


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _absolute_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _parse_cpus(value: str) -> list[int]:
    try:
        cpus = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise ValueError("expected CPUs must be a comma-separated integer list") from exc
    if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("expected CPUs must be unique nonnegative integers")
    return cpus


def _physical_route_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    routes = config["routes"]
    if not isinstance(routes, Mapping):  # normalized by the loader
        raise ValueError("configuration routes must be a mapping")
    result = tuple(
        route_id
        for route_id, route in sorted(routes.items())
        if isinstance(route, Mapping) and route.get("executor") == "upmem_physical"
    )
    if len(result) != 1:
        raise ValueError("M7C one-DPU template must declare exactly one physical route")
    return result


def _replace_route_id(
    config: dict[str, Any], *, old_route_id: str, new_route_id: str
) -> None:
    routes = config["routes"]
    route = routes.pop(old_route_id)
    routes[new_route_id] = route
    for matrix_item in config["matrix"]:
        matrix_item["route_ids"] = [
            new_route_id if route_id == old_route_id else route_id
            for route_id in matrix_item["route_ids"]
        ]


def prepare_config(
    *,
    template: Path,
    output: Path,
    mode: str,
    rank_path: str,
    session_root: str,
    expected_cpus: list[int],
    host_binary: str | None = None,
    dpu_binary: str | None = None,
) -> Path:
    """Resolve tracked paths once, then write a portable ignored ETH config."""

    if output.exists():
        raise ValueError(f"prepared configuration must be absent: {output}")
    normalized = _plain(load_experiment_config(template))
    # These are loader-derived values, not accepted YAML configuration fields.
    normalized.pop("collection_policy_id", None)
    normalized.pop("experiment_identity_payload", None)
    if normalized["schema_version"] != "tn_benchmark_v3":
        raise ValueError("M7C physical preparation requires tn_benchmark_v3")
    old_route_id = _physical_route_ids(normalized)[0]
    route = normalized["routes"][old_route_id]
    options = route["options"]
    if (
        options["dpu_count"] != 1
        or options["rank_count"] != 1
        or options["tasklets_per_dpu"] != 1
    ):
        raise ValueError("M7C one-DPU template must request one rank, DPU, and tasklet")

    if mode not in {"probe", "float32-smoke", "int8-smoke"}:
        raise ValueError(f"unsupported M7C physical mode: {mode}")
    numeric_policy = (
        "split_complex_int8_shared_scale_v1"
        if mode == "int8-smoke"
        else "split_complex_float32_v1"
    )
    route["numeric_policy"] = numeric_policy
    for field, override in (
        ("host_binary", host_binary),
        ("dpu_binary", dpu_binary),
    ):
        options[field] = _absolute_path(override) if override is not None else options[field]
    options["session_root"] = _absolute_path(session_root)
    options["rank_paths"] = [_absolute_path(rank_path)]

    new_route_id = (
        "upmem_int8_1dpu" if numeric_policy == "split_complex_int8_shared_scale_v1" else "upmem_float32_1dpu"
    )
    _replace_route_id(normalized, old_route_id=old_route_id, new_route_id=new_route_id)
    normalized["experiment_id"] = f"m7c-one-dpu-{mode}"
    collection = normalized["collection"]
    if mode == "probe":
        collection["warmup_blocks"] = 0
        collection["measurement_blocks"] = 1
    machine_policy = collection["machine_policy"]
    machine_policy["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": expected_cpus,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8"
    )
    # A reload from the ignored location proves that all source-relative paths
    # retain their original target after copying the template.
    prepared = load_experiment_config(output)
    prepared_route = prepared["routes"][new_route_id]
    for field in _PATH_FIELDS:
        if prepared_route["options"][field] != options[field]:
            raise ValueError(f"prepared {field} no longer resolves to the template target")
    if tuple(prepared_route["options"]["rank_paths"]) != tuple(options["rank_paths"]):
        raise ValueError("prepared rank paths no longer resolve to the requested target")
    return output


def _require_bool(facts: Mapping[str, Any], field: str, expected: bool) -> None:
    if facts.get(field) is not expected:
        raise ValueError(f"physical evidence requires {field}={expected}")


def _require_integer(facts: Mapping[str, Any], field: str, expected: int) -> None:
    if facts.get(field) != expected:
        raise ValueError(f"physical evidence requires {field}={expected}")


def _joined_sample_facts(
    sample: Mapping[str, Any], sessions_by_id: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("physical sample lacks backend facts")
    joined = dict(facts)
    session_id = sample.get("session_instance_id")
    if isinstance(session_id, str):
        session = sessions_by_id.get(session_id)
        terminal = session.get("terminal_backend_facts") if session else None
        if isinstance(terminal, Mapping):
            for field, value in terminal.items():
                joined.setdefault(field, value)
    return joined


def inspect_physical_artifacts(
    *,
    input_dir: Path,
    expected_samples: int,
    expected_sessions: int,
    numeric_policy: str,
) -> Mapping[str, Any]:
    """Assert one-DPU evidence provenance, output, and numerical contracts."""

    if numeric_policy not in _NUMERIC_POLICIES:
        raise ValueError(f"unsupported numeric policy: {numeric_policy}")
    manifest, samples, sessions = load_artifacts(input_dir)
    summary = verify_artifacts(input_dir)
    required_summary = {
        "status": "completed",
        "sample_count": expected_samples,
        "session_count": expected_sessions,
        "success_count": expected_samples,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for field, expected in required_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"unexpected physical verification {field}: {summary.get(field)!r}")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("physical evidence must bind to a clean source worktree")

    sessions_by_id = {
        str(session["session_instance_id"]): session for session in sessions
    }
    if len(sessions_by_id) != expected_sessions:
        raise ValueError("physical evidence session IDs must be unique")
    for session in sessions_by_id.values():
        if session.get("status") != "success" or session.get("release_verified") is not True:
            raise ValueError("physical session was not successfully released")
        facts = session.get("terminal_backend_facts")
        if not isinstance(facts, Mapping):
            raise ValueError("physical session lacks terminal backend facts")
        for field, expected in (
            ("target_observed", "physical_hardware"),
            ("hardware_kernel_executed", True),
            ("simulator_kernel_executed", False),
            ("cpu_fallback_used", False),
            ("physical_target_verified", True),
            ("hardware_release_verified", True),
            ("binary_identity_verified", True),
            ("native_identity_verified", True),
        ):
            if facts.get(field) != expected:
                raise ValueError(f"physical session requires {field}={expected!r}")

    for sample in samples:
        if sample.get("status") != "success":
            raise ValueError("completed physical qualification requires successful samples")
        if sample.get("output_sha256") is None:
            raise ValueError("physical qualification sample lacks an output hash")
        numeric = sample.get("numeric_facts")
        facts = _joined_sample_facts(sample, sessions_by_id)
        validation = sample.get("validation")
        if not isinstance(numeric, Mapping) or numeric.get("numeric_policy") != numeric_policy:
            raise ValueError("physical sample has an unexpected numeric policy")
        if not isinstance(validation, Mapping):
            raise ValueError("physical sample lacks validation facts")
        for field, expected in (
            ("requested_dpus", 1),
            ("allocated_dpus", 1),
            ("active_dpus", 1),
            ("tasklets_per_dpu", 1),
        ):
            _require_integer(facts, field, expected)
        for field, expected in (
            ("startup_resource_admission_passed", True),
            ("execution_resource_admission_passed", True),
        ):
            _require_bool(facts, field, expected)
        if validation.get("policy_reference_passed") is not True:
            raise ValueError("physical qualification requires policy-reference correctness")
        if numeric_policy == "split_complex_float32_v1":
            for field, expected in (
                ("full_precision_threshold_applicable", True),
                ("full_precision_passed", True),
                ("accuracy_qualified", True),
            ):
                _require_bool(validation, field, expected)
        else:
            if validation.get("full_precision_threshold_applicable") is not False:
                raise ValueError("int8 qualification must retain its no-threshold rule")
            if validation.get("full_precision_passed") is not None:
                raise ValueError("int8 qualification must not report a threshold pass")
            _require_bool(validation, "accuracy_qualified", False)
            if validation.get("max_abs_error") is None or validation.get("relative_l2_error") is None:
                raise ValueError("int8 qualification must retain full-precision error metrics")
    return summary


def archive_evidence(*, input_dir: Path, output: Path) -> tuple[Path, Path]:
    """Archive an already-inspected run and write an adjacent SHA-256 file."""

    if output.exists():
        raise ValueError(f"physical archive must be absent: {output}")
    input_root = input_dir.resolve()
    if not input_root.is_dir():
        raise ValueError(f"physical evidence directory does not exist: {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(input_root, arcname=input_root.name, recursive=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_name(output.name + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return output, checksum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mode", choices=("probe", "float32-smoke", "int8-smoke"), required=True)
    prepare.add_argument("--rank-path", required=True)
    prepare.add_argument("--session-root", required=True)
    prepare.add_argument("--expected-cpus", required=True)
    prepare.add_argument("--host-binary")
    prepare.add_argument("--dpu-binary")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--expected-samples", type=int, required=True)
    inspect.add_argument("--expected-sessions", type=int, required=True)
    inspect.add_argument("--numeric-policy", choices=sorted(_NUMERIC_POLICIES), required=True)
    archive = commands.add_parser("archive")
    archive.add_argument("--input", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_config(
                template=args.template.resolve(),
                output=args.output.resolve(),
                mode=args.mode,
                rank_path=args.rank_path,
                session_root=args.session_root,
                expected_cpus=_parse_cpus(args.expected_cpus),
                host_binary=args.host_binary,
                dpu_binary=args.dpu_binary,
            )
            payload: Mapping[str, Any] = {"status": "prepared", "output": str(output)}
        elif args.command == "inspect":
            payload = inspect_physical_artifacts(
                input_dir=args.input.resolve(),
                expected_samples=args.expected_samples,
                expected_sessions=args.expected_sessions,
                numeric_policy=args.numeric_policy,
            )
        else:
            archive_path, checksum = archive_evidence(
                input_dir=args.input.resolve(), output=args.output.resolve()
            )
            payload = {
                "status": "archived",
                "archive": str(archive_path),
                "checksum": str(checksum),
            }
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
