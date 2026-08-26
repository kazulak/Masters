#!/usr/bin/env python3
"""Prepare, run, and inspect preregistered M7C mixed scaling campaigns.

This command is deliberately separate from ``qualify``: a mixed NumPy/physical
matrix must use ``quantum_bench.cli run --allow-physical``.  The script is an
operator entry point and performs physical execution only when ``run`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
_PATH_FIELDS = (
    "host_binary",
    "dpu_binary",
    "initialization_binary",
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
    if not cpus or len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise ValueError("expected CPUs must be unique nonnegative integers")
    return cpus


def _physical_route_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    routes = config.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("configuration routes must be a mapping")
    return tuple(
        route_id
        for route_id, route in sorted(routes.items())
        if isinstance(route, Mapping) and route.get("executor") == "upmem_physical"
    )


def _clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("cannot inspect Git worktree")
    if result.stdout.strip():
        raise ValueError("M7C physical campaign requires a clean Git worktree")


def prepare_config(
    *,
    template: Path,
    output: Path,
    rank_paths: list[str],
    session_root: str,
    expected_cpus: list[int],
) -> Path:
    """Copy a tracked mixed template with target-specific absolute paths."""

    if output.exists():
        raise ValueError(f"prepared configuration must be absent: {output}")
    config = _plain(load_experiment_config(template))
    config.pop("collection_policy_id", None)
    config.pop("experiment_identity_payload", None)
    physical_route_ids = _physical_route_ids(config)
    if not physical_route_ids:
        raise ValueError("M7C scaling template must include physical routes")
    absolute_ranks = [_absolute_path(path) for path in rank_paths]
    absolute_session_root = Path(_absolute_path(session_root))
    for route_id in physical_route_ids:
        route = config["routes"][route_id]
        options = route["options"]
        rank_count = options["rank_count"]
        if len(absolute_ranks) != rank_count:
            raise ValueError(
                f"route {route_id} requires {rank_count} rank paths, got {len(absolute_ranks)}"
            )
        options["rank_paths"] = list(absolute_ranks)
        options["session_root"] = str(absolute_session_root / route_id)
    config["collection"]["machine_policy"]["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": expected_cpus,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared = load_experiment_config(output)
    for route_id in physical_route_ids:
        source_options = config["routes"][route_id]["options"]
        prepared_options = prepared["routes"][route_id]["options"]
        for field in (*_PATH_FIELDS, "session_root"):
            if prepared_options[field] != source_options[field]:
                raise ValueError(f"prepared {route_id}.{field} resolved incorrectly")
        if tuple(prepared_options["rank_paths"]) != tuple(source_options["rank_paths"]):
            raise ValueError(f"prepared {route_id}.rank_paths resolved incorrectly")
    return output


def _run(command: list[str], *, env: Mapping[str, str], capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(env),
        capture_output=capture,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "command failed: "
            + " ".join(command)
            + ("\n" + result.stdout if result.stdout else "")
            + ("\n" + result.stderr if result.stderr else "")
        )
    return result.stdout


def _selector_check(selection: Path, config: Path, env: Mapping[str, str]) -> None:
    _run(
        [
            sys.executable,
            "scripts/select_m7c_workload.py",
            "--check",
            str(selection),
            "--config",
            str(config),
        ],
        env=env,
    )


def _joined_sample_facts(
    sample: Mapping[str, Any], sessions_by_id: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("physical scaling sample lacks backend facts")
    joined = dict(facts)
    session_id = sample.get("session_instance_id")
    if isinstance(session_id, str):
        session = sessions_by_id.get(session_id)
        terminal = session.get("terminal_backend_facts") if session else None
        if isinstance(terminal, Mapping):
            for field, value in terminal.items():
                joined.setdefault(field, value)
    return joined


def run_campaign(
    *, selection: Path, config: Path, output: Path, report_output: Path
) -> Mapping[str, object]:
    """Run the one permitted mixed route command after immutable prechecks."""

    _clean_worktree()
    if output.exists() or report_output.exists():
        raise ValueError("M7C campaign outputs must be absent")
    normalized = load_experiment_config(config)
    if not _physical_route_ids(normalized):
        raise ValueError("M7C campaign config must include physical routes")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
    _selector_check(selection, config, env)
    _run(
        [
            sys.executable,
            "-m",
            "quantum_bench.cli",
            "run",
            "--config",
            str(config),
            "--output",
            str(output),
            "--allow-physical",
        ],
        env=env,
    )
    _run(
        [sys.executable, "-m", "quantum_bench.cli", "verify", "--input", str(output)],
        env=env,
    )
    _run(
        [
            sys.executable,
            "-m",
            "quantum_bench.cli",
            "report",
            "--input",
            str(output),
            "--output",
            str(report_output),
        ],
        env=env,
    )
    return inspect_campaign(input_dir=output, report_dir=report_output)


def inspect_campaign(*, input_dir: Path, report_dir: Path | None = None) -> Mapping[str, object]:
    """Check mixed-route provenance without making a performance conclusion."""

    manifest, samples, sessions = load_artifacts(input_dir)
    summary = verify_artifacts(input_dir)
    if summary.get("status") != "completed":
        raise ValueError("M7C campaign artifact is not completed")
    configuration = manifest["configuration"]["experiment"]
    routes = configuration["routes"]
    physical_ids = {
        route_id
        for route_id, route in routes.items()
        if route["executor"] == "upmem_physical"
    }
    if not physical_ids:
        raise ValueError("M7C campaign evidence has no physical route")
    attempts_per_route = (
        configuration["collection"]["warmup_blocks"]
        + configuration["collection"]["measurement_blocks"]
    )
    expected_sessions = len(physical_ids) * attempts_per_route
    if summary.get("session_count") != expected_sessions:
        raise ValueError("M7C fresh-session campaign has an unexpected session count")
    sessions_by_id = {
        str(session["session_instance_id"]): session for session in sessions
    }
    for sample in samples:
        if sample["route_id"] not in physical_ids:
            continue
        facts = _joined_sample_facts(sample, sessions_by_id)
        validation = sample["validation"]
        if not isinstance(validation, Mapping):
            raise ValueError("physical scaling sample lacks validation")
        required = {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "startup_resource_admission_passed": True,
            "execution_resource_admission_passed": True,
        }
        for field, expected in required.items():
            if facts.get(field) != expected:
                raise ValueError(f"physical scaling sample requires {field}={expected!r}")
        if validation.get("policy_reference_passed") is not True:
            raise ValueError("physical scaling sample failed policy-reference validation")
        if validation.get("accuracy_qualified") is not True:
            raise ValueError("float32 physical scaling sample is not accuracy-qualified")
    for session in sessions:
        if session["route_id"] not in physical_ids:
            continue
        facts = session["terminal_backend_facts"]
        if session["release_verified"] is not True or not isinstance(facts, Mapping):
            raise ValueError("physical scaling session lacks verified release facts")
        if facts.get("target_observed") != "physical_hardware":
            raise ValueError("physical scaling session did not observe hardware")
    result: dict[str, object] = {
        "status": "completed",
        "verification": summary,
        "physical_route_count": len(physical_ids),
        "expected_fresh_sessions": expected_sessions,
    }
    if report_dir is not None:
        report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
        result["report_schema_version"] = report.get("schema_version")
        result["scaling_count"] = report.get("scaling_count")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--rank-path", action="append", required=True)
    prepare.add_argument("--session-root", required=True)
    prepare.add_argument("--expected-cpus", required=True)
    run = commands.add_parser("run")
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--report-output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            output = prepare_config(
                template=args.template.resolve(),
                output=args.output.resolve(),
                rank_paths=args.rank_path,
                session_root=args.session_root,
                expected_cpus=_parse_cpus(args.expected_cpus),
            )
            payload: Mapping[str, object] = {"status": "prepared", "output": str(output)}
        elif args.command == "run":
            payload = run_campaign(
                selection=args.selection.resolve(),
                config=args.config.resolve(),
                output=args.output.resolve(),
                report_output=args.report_output.resolve(),
            )
        else:
            payload = inspect_campaign(
                input_dir=args.input.resolve(),
                report_dir=args.report_output.resolve() if args.report_output else None,
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
