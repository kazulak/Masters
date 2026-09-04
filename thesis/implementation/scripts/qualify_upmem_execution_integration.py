#!/usr/bin/env python3
"""Prepare and inspect the bounded Phase A UPMEM integration gates.

The tool only prepares configurations and validates finalized canonical evidence.
It never starts an SDK or physical execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import yaml

from quantum_bench.evidence import canonical_json, executable_id, identity_hash, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
FLOAT32 = "split_complex_float32_v1"
INT8 = "complex_int8_shared_scale_v1"
PACKED = "packed_operation_v1"
KERNEL_POLICY = "dpu_real_tile_v4_wram_panel_v1"
KERNEL_IMPLEMENTATION = "upmem_sdk_hardware_v4_wram_panel_kernel"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

CASES = ("bell2", "stress4")
ROUTE_SPECS = {
    "float32_1dpu_t1": (FLOAT32, 1, 1, 1),
    "int8_1dpu_t1": (INT8, 1, 1, 1),
    "float32_1dpu_t8": (FLOAT32, 1, 1, 8),
    "int8_1dpu_t8": (INT8, 1, 1, 8),
    "float32_4dpu_t8": (FLOAT32, 4, 1, 8),
    "int8_3dpu_t8": (INT8, 3, 1, 8),
    "int8_4dpu_t8": (INT8, 4, 1, 8),
}
ROUTES = tuple(ROUTE_SPECS)
SDK_CELLS = frozenset((case_id, "greedy", route_id) for case_id in CASES for route_id in ROUTES)
PHYSICAL_CELLS = frozenset(
    {
        ("bell2", "greedy", "float32_1dpu_t1"),
        ("bell2", "greedy", "int8_1dpu_t1"),
        ("stress4", "greedy", "float32_1dpu_t8"),
        ("stress4", "greedy", "int8_1dpu_t8"),
        ("stress4", "greedy", "float32_4dpu_t8"),
        ("stress4", "greedy", "int8_3dpu_t8"),
        ("stress4", "greedy", "int8_4dpu_t8"),
    }
)
KINDS = {
    "sdk": {
        "label": "upmem-execution-integration-sdk-v1",
        "executor": "upmem_sdk_simulator",
        "cells": SDK_CELLS,
        "target": "sdk_simulator",
    },
    "physical": {
        "label": "upmem-execution-integration-physical-v1",
        "executor": "upmem_physical",
        "cells": PHYSICAL_CELLS,
        "target": "physical_hardware",
    },
}
PATH_FIELDS = ("host_binary", "dpu_binary", "initialization_binary")


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    value = result.stdout.strip()
    if result.returncode or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("cannot determine source HEAD")
    return value


def _kind(kind: str) -> Mapping[str, Any]:
    try:
        return KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown integration qualification kind: {kind}") from None


def _expected_cells(config: Mapping[str, Any]) -> frozenset[tuple[str, str | None, str]]:
    observed: set[tuple[str, str | None, str]] = set()
    matrix = config.get("matrix")
    if not isinstance(matrix, (list, tuple)):
        raise ValueError("experiment matrix is missing")
    for item in matrix:
        if not isinstance(item, Mapping):
            raise ValueError("experiment matrix contains a non-mapping")
        case_id = item.get("case_id")
        plan_id = item.get("plan_id")
        route_ids = item.get("route_ids")
        if not isinstance(case_id, str) or not isinstance(route_ids, (list, tuple)):
            raise ValueError("experiment matrix row is malformed")
        for route_id in route_ids:
            cell = (case_id, plan_id, route_id)
            if cell in observed:
                raise ValueError(f"duplicate declared matrix cell: {cell}")
            observed.add(cell)
    return frozenset(observed)


def _validate_frozen_config(
    config_value: Mapping[str, Any], *, kind: str
) -> frozenset[tuple[str, str | None, str]]:
    spec = _kind(kind)
    config = _plain(config_value)
    if not isinstance(config, dict):
        raise ValueError("embedded experiment configuration is not a mapping")
    identity = config.get("experiment_identity_payload")
    if not isinstance(identity, Mapping) or identity.get("label") != spec["label"]:
        raise ValueError("experiment identity label is not frozen for this gate")
    if identity_hash("quantum_bench.experiment_id.v3", identity) != config.get("experiment_id"):
        raise ValueError("experiment identity hash does not match its manifest payload")

    expected_cases = {
        "bell2": {
            "circuit": {
                "kind": "builtin",
                "name": "bell_2q",
                "path": None,
                "parameters": {},
            }
        },
        "stress4": {
            "circuit": {
                "kind": "builtin",
                "name": "quantization_stress",
                "path": None,
                "parameters": {"n_qubits": 4, "repeat_layers": 2},
            }
        },
    }
    if config.get("cases") != expected_cases:
        raise ValueError("case set is not the frozen Bell2/stress4 matrix")
    if config.get("plans") != {
        "greedy": {
            "planner": {"engine": "opt_einsum", "mode": "greedy"},
            "slicing": None,
        }
    }:
        raise ValueError("plan set is not the frozen greedy plan")

    collection = config.get("collection")
    if not isinstance(collection, Mapping):
        raise ValueError("collection policy is missing")
    if any(
        collection.get(field) != expected
        for field, expected in (
            ("claim_policy", "diagnostic_v1"),
            ("base_seed", 20260905),
            ("warmup_blocks", 0),
            ("measurement_blocks", 1),
            ("session_policy", "fresh_session_per_attempt_v1"),
            ("block_cooldown_s", 0.0),
        )
    ):
        raise ValueError("collection timing/session policy is not frozen")
    machine = collection.get("machine_policy")
    if not isinstance(machine, Mapping) or machine.get("affinity") != {
        "mode": "exact_required_v1",
        "expected_cpus": [0],
    }:
        raise ValueError("machine CPU affinity is not the observed CPU 0 gate")

    routes = config.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(ROUTES):
        raise ValueError("route set is not frozen")
    for route_id, (policy, dpu_count, rank_count, tasklets) in ROUTE_SPECS.items():
        route = routes[route_id]
        if not isinstance(route, Mapping):
            raise ValueError(f"route is not a mapping: {route_id}")
        if route.get("executor") != spec["executor"] or route.get("numeric_policy") != policy:
            raise ValueError(f"route identity is not frozen: {route_id}")
        options = route.get("options")
        if not isinstance(options, Mapping):
            raise ValueError(f"route options are missing: {route_id}")
        for field, expected in (
            ("dpu_count", dpu_count),
            ("rank_count", rank_count),
            ("tasklets_per_dpu", tasklets),
        ):
            if options.get(field) != expected:
                raise ValueError(f"route resource is not frozen for {route_id}: {field}")
        if any(not isinstance(options.get(field), str) or not options[field] for field in PATH_FIELDS):
            raise ValueError(f"route binary identity paths are missing: {route_id}")
        if not isinstance(options.get("session_root"), str) or not options["session_root"]:
            raise ValueError(f"route session root is missing: {route_id}")
        rank_paths = options.get("rank_paths")
        if spec["executor"] == "upmem_physical":
            if not isinstance(rank_paths, (list, tuple)) or len(rank_paths) != rank_count:
                raise ValueError(f"physical route rank set is not exact: {route_id}")
            if Path(str(rank_paths[0])).name != "dpu_rank1":
                raise ValueError(f"physical route is not bound to rank1: {route_id}")
        elif rank_paths is not None:
            raise ValueError(f"SDK route unexpectedly declares physical rank paths: {route_id}")

    cells = _expected_cells(config)
    if cells != spec["cells"] or len(cells) != (14 if kind == "sdk" else 7):
        raise ValueError("declared matrix cells do not match the frozen gate")
    return cells


def prepare(
    template: Path,
    output: Path,
    *,
    rank_path: str,
    session_root: Path,
    expected_cpu: int,
    binary_root: Path,
) -> Path:
    """Materialize explicit physical paths and re-hash the new identity payload."""

    if output.exists():
        raise ValueError(f"prepared configuration already exists: {output}")
    if expected_cpu != 0 or Path(rank_path).resolve().name != "dpu_rank1":
        raise ValueError("physical integration gate requires CPU 0 and dpu_rank1")
    config = _plain(load_experiment_config(template))
    _validate_frozen_config(config, kind="physical")
    identity = config.pop("experiment_identity_payload", None)
    if not isinstance(identity, Mapping) or not isinstance(identity.get("label"), str):
        raise ValueError("template lacks its experiment identity label")
    config["experiment_id"] = identity["label"]
    config.pop("collection_policy_id", None)
    physical = [
        (route_id, route)
        for route_id, route in config.get("routes", {}).items()
        if route.get("executor") == "upmem_physical"
    ]
    if not physical:
        raise ValueError("template has no physical routes")
    config["collection"]["machine_policy"]["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": [expected_cpu],
    }
    resolved_session_root = session_root.resolve()
    resolved_binary_root = binary_root.resolve()
    resolved_rank_path = str(Path(rank_path).resolve())
    for route_id, route in physical:
        options = route["options"]
        tasklets = int(options["tasklets_per_dpu"])
        options["rank_paths"] = [resolved_rank_path]
        options["session_root"] = str((resolved_session_root / route_id).resolve())
        options["host_binary"] = str(
            (resolved_binary_root / f"host_upmem_execution_plan_v4_t{tasklets}").resolve()
        )
        options["dpu_binary"] = str(
            (resolved_binary_root / f"dpu_gemm_tile_v4_t{tasklets}").resolve()
        )
        options["initialization_binary"] = str(
            (resolved_binary_root / f"dpu_simplepim_management_init_t{tasklets}").resolve()
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepared = load_experiment_config(output)
    _validate_frozen_config(prepared, kind="physical")
    if identity_hash(
        "quantum_bench.experiment_id.v3", prepared["experiment_identity_payload"]
    ) != prepared["experiment_id"]:
        raise ValueError("prepared configuration identity manifest hash is invalid")
    return output


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("sample lacks backend facts")
    joined = dict(facts)
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        for key, value in terminal.items():
            joined.setdefault(key, value)
    return joined


def _require_hash(value: object, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 identity")


def _require_exact(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, value in expected.items():
        if mapping.get(field) != value:
            raise ValueError(f"{label} requires {field}={value!r}")


def _expected_executable_id(facts: Mapping[str, Any], *, executor: str) -> str:
    files = {}
    for field in PATH_FIELDS:
        digest = facts.get(f"{field}_sha256")
        _require_hash(digest, f"{field}_sha256")
        files[field] = digest
    return executable_id(
        {
            "executor": executor,
            "abi_version": 4,
            "static_file_sha256": files,
            "request_transport": PACKED,
            "source_commit": None,
            "dependency_versions": {},
        }
    )


def inspect(root: Path, *, kind: str, expected_source: str) -> dict[str, Any]:
    """Fail closed on source, matrix, provenance, replay, and resource drift."""

    spec = _kind(kind)
    manifest, samples, session_rows = load_artifacts(root)
    verification = verify_artifacts(root)
    expected_count = len(spec["cells"])
    _require_exact(
        verification,
        {
            "status": "completed",
            "sample_count": expected_count,
            "session_count": expected_count,
            "success_count": expected_count,
            "failed_count": 0,
            "unsupported_count": 0,
        },
        "verification",
    )
    if manifest.get("source_commit") != expected_source:
        raise ValueError("evidence source SHA does not match expected source")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("execution source was dirty")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("evidence configuration is missing")
    config = configuration.get("experiment")
    if not isinstance(config, Mapping):
        raise ValueError("embedded experiment configuration is missing")
    cells = _validate_frozen_config(config, kind=kind)
    if manifest.get("experiment_id") != config.get("experiment_id"):
        raise ValueError("manifest and embedded experiment IDs differ")

    sessions: dict[str, Mapping[str, Any]] = {}
    for row in session_rows:
        if not isinstance(row, Mapping):
            raise ValueError("session row is not a mapping")
        session_id = row.get("session_instance_id")
        if not isinstance(session_id, str) or not session_id or session_id in sessions:
            raise ValueError("session IDs are not unique")
        sessions[session_id] = row
        if row.get("status") != "success" or row.get("release_verified") is not True:
            raise ValueError("session failed or release was not verified")
        terminal = row.get("terminal_backend_facts")
        if not isinstance(terminal, Mapping):
            raise ValueError("session lacks terminal facts")
        _require_exact(
            terminal,
            {
                "target_observed": spec["target"],
                "cpu_fallback_used": False,
                "binary_identity_verified": True,
                "native_identity_verified": True,
            },
            "session provenance",
        )
        if terminal.get("target_observed") == "physical_hardware":
            _require_exact(
                terminal,
                {
                    "physical_target_verified": True,
                    "hardware_kernel_executed": True,
                    "simulator_kernel_executed": False,
                    "hardware_release_verified": True,
                },
                "physical session provenance",
            )
        else:
            _require_exact(
                terminal,
                {
                    "simulator_target_verified": True,
                    "simulator_kernel_executed": True,
                    "hardware_kernel_executed": False,
                },
                "SDK session provenance",
            )

    observed_cells: set[tuple[str, str | None, str]] = set()
    observed_sessions: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("sample row is not a mapping")
        cell = (sample.get("case_id"), sample.get("plan_id"), sample.get("route_id"))
        if cell not in cells or cell in observed_cells:
            raise ValueError(f"sample cell is missing, unexpected, or duplicated: {cell}")
        observed_cells.add(cell)
        session_id = sample.get("session_instance_id")
        if not isinstance(session_id, str) or session_id not in sessions or session_id in observed_sessions:
            raise ValueError("samples do not bind one fresh session per cell")
        observed_sessions.add(session_id)
        if sample.get("status") != "success":
            raise ValueError("integration qualification requires successful samples")
        if sample.get("attempt_kind") != "measurement" or sample.get("sample_index") != 0 or sample.get("block_id") != 0:
            raise ValueError("integration matrix must contain one measured block per cell")
        if sample.get("observed_affinity") != [0]:
            raise ValueError("sample CPU affinity is not the observed CPU 0")
        measurement = sample.get("measurement")
        if not isinstance(measurement, Mapping) or measurement.get("scope_id") != "steady_execution_v1":
            raise ValueError("integration qualification requires steady execution timing")
        output_hash = sample.get("output_sha256")
        _require_hash(output_hash, "sample.output_sha256")

        route_id = str(sample["route_id"])
        policy, dpu_count, rank_count, tasklets = ROUTE_SPECS[route_id]
        numeric = sample.get("numeric_facts")
        validation = sample.get("validation")
        identities = sample.get("identities")
        if not isinstance(numeric, Mapping) or numeric.get("numeric_policy") != policy:
            raise ValueError("sample numerical policy identity is wrong")
        operations = numeric.get("operations")
        if not isinstance(operations, (list, tuple)) or not operations:
            raise ValueError("sample lacks per-operation numeric identity")
        if any(not isinstance(operation, Mapping) or operation.get("numeric_policy") != policy for operation in operations):
            raise ValueError("operation numerical policy identity is wrong")
        if not isinstance(validation, Mapping) or validation.get("policy_reference_passed") is not True:
            raise ValueError("sample did not pass same-policy CPU replay")
        if policy == FLOAT32:
            _require_exact(
                validation,
                {
                    "full_precision_threshold_applicable": True,
                    "full_precision_passed": True,
                    "accuracy_qualified": True,
                },
                "float32 validation",
            )
        else:
            if validation.get("full_precision_threshold_applicable") is not False:
                raise ValueError("int8 validation incorrectly applied a float32 threshold")
            if validation.get("full_precision_passed") is not None or validation.get("accuracy_qualified") is not False:
                raise ValueError("int8 validation incorrectly claimed full-precision equality")
        if not isinstance(identities, Mapping):
            raise ValueError("sample identities are missing")
        _require_hash(identities.get("executable_id"), "sample.identities.executable_id")
        _require_hash(identities.get("physical_plan_id"), "sample.identities.physical_plan_id")

        facts = _joined_facts(sample, sessions)
        if identities["executable_id"] != _expected_executable_id(
            facts, executor=spec["executor"]
        ):
            raise ValueError("sample executable identity does not match recorded binary hashes")
        _require_exact(
            facts,
            {
                "request_transport": PACKED,
                "target_observed": spec["target"],
                "cpu_fallback_used": False,
                "requested_dpus": dpu_count,
                "allocated_dpus": dpu_count,
                "active_dpus": dpu_count,
                "rank_count": rank_count,
                "tasklets_per_dpu": tasklets,
                "startup_resource_admission_passed": True,
                "execution_resource_admission_passed": True,
                "kernel_policy": KERNEL_POLICY,
                "kernel_implementation_id": KERNEL_IMPLEMENTATION,
            },
            "sample execution provenance",
        )
        _require_exact(
            facts,
            {
                "observed_rank_count": rank_count,
                "observed_tasklets_per_dpu": tasklets,
                "requested_dpu_count": dpu_count,
                "allocated_dpu_count": dpu_count,
                "observed_dpu_count": dpu_count,
            },
            "observed sample resources",
        )
        if facts.get("physical_plan_id") != identities.get("physical_plan_id"):
            raise ValueError("sample physical plan identity does not match execution facts")
        if spec["target"] == "physical_hardware":
            _require_exact(
                facts,
                {
                    "physical_target_verified": True,
                    "hardware_kernel_executed": True,
                    "simulator_kernel_executed": False,
                },
                "physical sample provenance",
            )
        else:
            _require_exact(
                facts,
                {
                    "simulator_kernel_executed": True,
                    "hardware_kernel_executed": False,
                },
                "SDK sample provenance",
            )

    if observed_cells != cells or observed_sessions != set(sessions):
        raise ValueError("observed cells or fresh-session bindings are incomplete")
    return {
        "status": "passed",
        "kind": kind,
        "source_commit": expected_source,
        "experiment_id": manifest["experiment_id"],
        "sample_count": len(samples),
        "session_count": len(session_rows),
        "matrix_cell_count": len(cells),
        "correctness_only": kind == "physical",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--template", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--rank-path", required=True)
    prep.add_argument("--session-root", type=Path, required=True)
    prep.add_argument("--expected-cpu", type=int, default=0)
    prep.add_argument("--binary-root", type=Path, required=True)
    check = commands.add_parser("inspect")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--kind", choices=tuple(KINDS), required=True)
    check.add_argument("--expected-source", default=None)
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
        else:
            result = inspect(
                args.input.resolve(),
                kind=args.kind,
                expected_source=args.expected_source or _git_head(),
            )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
