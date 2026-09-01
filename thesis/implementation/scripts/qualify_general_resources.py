#!/usr/bin/env python3
"""Build, prepare, and inspect Generalized Hierarchical Resource Qualification v1.

This qualifier deliberately never allocates physical hardware.  Operators run the
prepared configuration with the physical-only benchmark command, then pass its
canonical evidence to ``inspect``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from quantum_bench.cli import _job, _plan_dag
from quantum_bench.evidence import (
    canonical_json,
    executable_id,
    identity_hash,
    load_artifacts,
    problem_id,
    tensor_network_structure_id,
)
from quantum_bench.experiment import load_experiment_config
from quantum_bench.lowering import contraction_dag_hash
from quantum_bench.report import verify_artifacts
from quantum_bench.upmem.plan import (
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
)


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "upmem" / "runtime"
TEMPLATE = ROOT / "configs" / "tn_benchmark_resource_general_correctness.yml"
BUILD_OUTPUT_NAME = "resource_general_builds.json"
SUMMARY_OUTPUT_NAME = "resource_general_summary.json"
_ROUTE_SPECS = {
    "upmem_float32_1dpu_t3": (1, 3),
    "upmem_float32_1dpu_t7": (1, 7),
    "upmem_float32_1dpu_t12": (1, 12),
    "upmem_float32_1dpu_t24": (1, 24),
    "upmem_float32_3dpu_t8": (3, 8),
}
_BINARY_FIELDS = ("host_binary", "dpu_binary", "initialization_binary")
_BINARY_HASH_FIELDS = {
    "host_binary": "host_binary_sha256",
    "dpu_binary": "dpu_binary_sha256",
    "initialization_binary": "initialization_binary_sha256",
}
_SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9A-Fa-f]+)\s+\S+\s+([0-9A-Fa-f]+)\s+\S+\s+([A-Z]*)\s"
)
_SHA256 = re.compile(r"[0-9a-f]{40}")
_CANONICAL_ID = re.compile(r"[0-9a-f]{64}")
_IRAM_START = 0x80000000
_IRAM_LIMIT_BYTES = 24 * 1024
_WRAM_LIMIT_BYTES = 64 * 1024


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _require_clean_source(expected_source: str | None = None) -> str:
    if _git_output("status", "--porcelain"):
        raise ValueError("general-resource inspection requires a clean Git worktree")
    current = _git_output("rev-parse", "HEAD")
    if expected_source is None:
        return current
    if not _SHA256.fullmatch(expected_source):
        raise ValueError("expected source commit must be a 40-hex SHA")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_source, current],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("expected source commit is not an ancestor of current HEAD")
    return expected_source


def _absolute(value: str, *, relative_to: Path | None = None) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return str(path.resolve())


def _parse_cpus(value: str) -> list[int]:
    try:
        cpus = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise ValueError("expected CPUs must be a comma-separated integer list") from exc
    if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("expected CPUs must be unique nonnegative integers")
    return cpus


def _require_absent_ignored_output(path: Path, expected_name: str) -> None:
    if path.name != expected_name:
        raise ValueError(f"output must be named {expected_name}")
    if path.exists():
        raise ValueError(f"qualification output must be absent: {path}")
    if path.resolve().is_relative_to(ROOT):
        check = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path.resolve())],
            cwd=ROOT,
            check=False,
        )
        if check.returncode != 0:
            raise ValueError("repository qualification output must be ignored")


def _template_configuration(template: Path = TEMPLATE) -> dict[str, Any]:
    normalized = _plain(load_experiment_config(template))
    payload = normalized.get("experiment_identity_payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("configuration"), Mapping):
        raise ValueError("tracked template lacks an experiment identity payload")
    return _plain(payload["configuration"])


def _validate_template(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "resource-general-correctness-stress18":
        raise ValueError("unexpected resource-general experiment ID")
    collection = config.get("collection")
    if not isinstance(collection, Mapping) or collection.get("claim_policy") != "diagnostic_v1":
        raise ValueError("resource-general qualification must remain diagnostic")
    if collection.get("warmup_blocks") != 0 or collection.get("measurement_blocks") != 1:
        raise ValueError("resource-general qualification requires zero warmups and one measurement")
    if collection.get("session_policy") != "fresh_session_per_attempt_v1":
        raise ValueError("resource-general qualification requires fresh sessions")
    case = config.get("cases", {}).get("quantization_stress_18q")
    circuit = case.get("circuit") if isinstance(case, Mapping) else None
    if not isinstance(circuit, Mapping) or circuit.get("name") != "quantization_stress":
        raise ValueError("resource-general qualification requires Stress18")
    if circuit.get("parameters") != {"n_qubits": 18, "repeat_layers": 2}:
        raise ValueError("resource-general qualification requires exact Stress18 parameters")
    plans = config.get("plans")
    greedy = plans.get("greedy") if isinstance(plans, Mapping) else None
    if not isinstance(greedy, Mapping) or greedy.get("slicing") is not None:
        raise ValueError("resource-general qualification forbids slicing")
    if greedy.get("planner") != {"engine": "opt_einsum", "mode": "greedy"}:
        raise ValueError("resource-general qualification requires opt_einsum greedy")
    routes = config.get("routes")
    if not isinstance(routes, Mapping) or set(routes) != set(_ROUTE_SPECS):
        raise ValueError("resource-general qualification has an unexpected route set")
    for route_id, (dpus, tasklets) in _ROUTE_SPECS.items():
        route = routes[route_id]
        if route.get("executor") != "upmem_physical" or route.get("numeric_policy") != "split_complex_float32_v1":
            raise ValueError(f"unexpected route policy for {route_id}")
        options = route.get("options")
        if not isinstance(options, Mapping) or (
            options.get("rank_count"), options.get("dpu_count"), options.get("tasklets_per_dpu")
        ) != (1, dpus, tasklets):
            raise ValueError(f"unexpected resources for {route_id}")
        expected_paths = _binary_paths(tasklets)
        for field, expected in expected_paths.items():
            if Path(str(options.get(field, ""))).name != expected.name:
                raise ValueError(f"unexpected {field} for {route_id}")
    matrix = config.get("matrix")
    if matrix != [
        {
            "case_id": "quantization_stress_18q",
            "plan_id": "greedy",
            "route_ids": list(_ROUTE_SPECS),
        }
    ]:
        raise ValueError("resource-general qualification matrix is not exact")


def _embedded_semantic_configuration(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or _CANONICAL_ID.fullmatch(experiment_id) is None:
        raise ValueError("manifest experiment_id must be a canonical 64-hex ID")
    configuration = manifest.get("configuration")
    experiment = configuration.get("experiment") if isinstance(configuration, Mapping) else None
    if not isinstance(experiment, Mapping) or experiment.get("experiment_id") != experiment_id:
        raise ValueError("manifest experiment_id does not match configuration.experiment")
    payload = experiment.get("experiment_identity_payload")
    semantic = payload.get("configuration") if isinstance(payload, Mapping) else None
    if not isinstance(semantic, Mapping):
        raise ValueError("embedded experiment lacks its semantic identity configuration")
    if experiment.get("schema_version") != "tn_benchmark_v3":
        raise ValueError("resource-general evidence requires tn_benchmark_v3")
    if identity_hash("quantum_bench.experiment_id.v3", payload) != experiment_id:
        raise ValueError("manifest experiment_id does not match its canonical identity payload")
    _validate_template(semantic)
    return semantic


def _expected_execution_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    job = _job(config["cases"]["quantization_stress_18q"])
    network, _, dag, _ = _plan_dag(job, config["plans"]["greedy"])
    routes: dict[str, Any] = {}
    for route_id, (dpus, tasklets) in _ROUTE_SPECS.items():
        route = config["routes"][route_id]
        plan = plan_upmem(
            dag,
            numeric_policy=route["numeric_policy"],
            topology=UpmemTopology(
                dpu_count=dpus,
                rank_count=1,
                tasklets_per_dpu=tasklets,
            ),
        )
        routes[route_id] = {
            "plan": plan,
            "physical_plan_id": physical_plan_id(plan),
            "kernel_policy": plan.kernel_policy,
        }
    return {
        "problem_id": problem_id(job),
        "tensor_network_structure_id": tensor_network_structure_id(network),
        "logical_plan_id": contraction_dag_hash(dag),
        "routes": routes,
    }


def _binary_paths(tasklets: int) -> dict[str, Path]:
    return {
        "host_binary": NATIVE / "bin" / f"host_upmem_execution_plan_v4_t{tasklets}",
        "dpu_binary": NATIVE / "bin" / f"dpu_gemm_tile_v4_t{tasklets}",
        "initialization_binary": NATIVE / "bin" / f"dpu_simplepim_management_init_t{tasklets}",
    }


def _parse_readelf_sections(output: str) -> tuple[dict[str, Any], ...]:
    sections: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = _SECTION.match(line)
        if match is None:
            continue
        name, kind, address, size, flags = match.groups()
        sections.append(
            {
                "name": name,
                "type": kind,
                "address": int(address, 16),
                "size_bytes": int(size, 16),
                "flags": flags,
            }
        )
    return tuple(sections)


def _readelf_sections(path: Path) -> tuple[dict[str, Any], ...]:
    result = subprocess.run(
        ["readelf", "-SW", str(path)], check=True, capture_output=True, text=True
    )
    sections = _parse_readelf_sections(result.stdout)
    if not sections:
        raise ValueError(f"readelf returned no section rows for {path}")
    return sections


def _dpu_memory_facts(path: Path) -> Mapping[str, Any]:
    sections = _readelf_sections(path)
    text_sections = [section for section in sections if section["name"] == ".text"]
    if len(text_sections) != 1:
        raise ValueError("DPU ELF must contain exactly one .text section")
    text = text_sections[0]
    text_bytes = text["size_bytes"]
    text_end = text["address"] + text_bytes
    if text["address"] != _IRAM_START or text_end > _IRAM_START + _IRAM_LIMIT_BYTES:
        raise ValueError("DPU .text exceeds the 24 KiB IRAM address range")
    wram_sections = [
        section
        for section in sections
        if "A" in section["flags"]
        and section["name"] != ".text"
        and not section["name"].startswith(".mram")
        and section["name"] != ".atomic"
    ]
    for section in wram_sections:
        end = section["address"] + section["size_bytes"]
        if section["address"] < 0 or end > _WRAM_LIMIT_BYTES:
            raise ValueError("DPU allocated WRAM section exceeds the 64 KiB address range")
    wram_bytes = sum(section["size_bytes"] for section in wram_sections)
    if wram_bytes > _WRAM_LIMIT_BYTES:
        raise ValueError("DPU allocated WRAM data/stacks exceed 64 KiB")
    return {
        "text_address": text["address"],
        "text_bytes": text_bytes,
        "text_end_address": text_end,
        "text_limit_bytes": _IRAM_LIMIT_BYTES,
        "allocated_wram_data_and_stacks_bytes": wram_bytes,
        "wram_limit_bytes": _WRAM_LIMIT_BYTES,
        "wram_sections": wram_sections,
        "sections": sections,
    }


def _json_events(output: str) -> tuple[Mapping[str, Any], ...]:
    events: list[Mapping[str, Any]] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            events.append(event)
    return tuple(events)


def _host_argument_probe(paths: Mapping[str, Path], tasklets: int, root: Path) -> Mapping[str, Any]:
    root.mkdir(parents=True, exist_ok=True)

    def probe(observed_tasklets: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(paths["host_binary"]),
                "--target", "hardware",
                "--session-root", str(root),
                "--dpus", "1",
                "--tasklets", str(observed_tasklets),
                "--initialization-binary", str(paths["initialization_binary"]),
                "--dpu-binary", str(paths["dpu_binary"]),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    matching = probe(tasklets)
    alternate = probe(1 if tasklets != 1 else 2)
    matching_events = _json_events(matching.stdout)
    alternate_events = _json_events(alternate.stdout)
    matching_stage = next((event.get("failure_stage") for event in matching_events if event.get("event") == "STARTUP"), None)
    alternate_stage = next((event.get("failure_stage") for event in alternate_events if event.get("event") == "STARTUP"), None)
    if matching_stage != "hardware_opt_in_missing":
        raise ValueError("matching host tasklet probe did not reach target validation")
    if alternate_stage != "tasklet_binary_mismatch":
        raise ValueError("alternate host tasklet probe did not report tasklet_binary_mismatch")
    return {
        "matching_tasklets": tasklets,
        "matching_failure_stage": matching_stage,
        "alternate_tasklets": 1 if tasklets != 1 else 2,
        "alternate_failure_stage": alternate_stage,
        "nonallocating": True,
    }


def build(*, output: Path) -> Mapping[str, Any]:
    _require_absent_ignored_output(output, BUILD_OUTPUT_NAME)
    records = []
    for tasklets in range(1, 25):
        command = ["make", "-B", "-C", str(NATIVE), "v4", f"NR_TASKLETS={tasklets}"]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        captured = result.stdout + result.stderr
        if result.returncode != 0:
            raise ValueError(f"T{tasklets} force-build failed: {captured}")
        if f"-DNR_TASKLETS={tasklets}" not in captured:
            raise ValueError(f"T{tasklets} build did not capture its NR_TASKLETS define")
        paths = _binary_paths(tasklets)
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"T{tasklets} build did not produce all T-specific binaries")
        records.append(
            {
                "tasklets_per_dpu": tasklets,
                "build_command": command,
                "captured_tasklet_define": f"-DNR_TASKLETS={tasklets}",
                "binaries": {
                    field: {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for field, path in paths.items()
                },
                "target_link_verified": True,
                "dpu_memory": {
                    "contraction": _dpu_memory_facts(paths["dpu_binary"]),
                    "initialization": _dpu_memory_facts(paths["initialization_binary"]),
                },
                "host_argument_probe": _host_argument_probe(
                    paths, tasklets, output.parent / f"host-probe-t{tasklets}"
                ),
            }
        )
    payload = {"schema_version": "resource_general_builds_v1", "builds": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return payload


def prepare(*, output: Path, rank_path: str, session_root: str, expected_cpus: Sequence[int]) -> Mapping[str, Any]:
    _require_absent_ignored_output(output, "resource_general_prepared.yml")
    config = _template_configuration()
    _validate_template(config)
    routes = config["routes"]
    for route_id, route in routes.items():
        options = route["options"]
        options["rank_paths"] = [_absolute(rank_path)]
        options["session_root"] = _absolute(str(Path(session_root) / route_id))
        for field in _BINARY_FIELDS:
            options[field] = _absolute(options[field], relative_to=TEMPLATE.parent)
    config["collection"]["machine_policy"]["affinity"] = {
        "mode": "exact_required_v1",
        "expected_cpus": list(expected_cpus),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    prepared = load_experiment_config(output)
    for route_id in _ROUTE_SPECS:
        options = prepared["routes"][route_id]["options"]
        if tuple(options["rank_paths"]) != (_absolute(rank_path),):
            raise ValueError(f"prepared rank path mismatch for {route_id}")
        for field in ("session_root", *_BINARY_FIELDS):
            if not Path(options[field]).is_absolute():
                raise ValueError(f"prepared {field} is not absolute for {route_id}")
    return {"status": "prepared", "output": str(output.resolve())}


def _joined_facts(sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("sample lacks backend facts")
    result = dict(facts)
    session_id = sample.get("session_instance_id")
    terminal = sessions.get(str(session_id), {}).get("terminal_backend_facts")
    if isinstance(terminal, Mapping):
        for field, value in terminal.items():
            result.setdefault(field, value)
    return result


def _expected_executable_id(facts: Mapping[str, Any]) -> str:
    files: dict[str, str] = {}
    for binary_field, fact_field in _BINARY_HASH_FIELDS.items():
        value = facts.get(fact_field)
        if not isinstance(value, str) or _CANONICAL_ID.fullmatch(value) is None:
            raise ValueError(f"physical evidence lacks canonical {fact_field}")
        files[binary_field] = value
    return executable_id(
        {
            "executor": "upmem_physical",
            "abi_version": 4,
            "static_file_sha256": files,
            "request_transport": "packed_operation_v1",
            "source_commit": None,
            "dependency_versions": {},
        }
    )


def _t24_plan_facts(plan: Any) -> Mapping[str, Any]:
    local_m = [unit.m_size for stage in plan.stages for unit in stage.work_units]
    maximum = max(local_m)
    if maximum >= 24:
        raise ValueError("T24 plan does not demonstrate the expected idle-tasklet condition")
    admission = collection_resource_admission(plan)
    return {
        "max_relevant_local_m": maximum,
        "idle_tasklets_per_dpu": 24 - maximum,
        "tasklet_row_sufficiency_passed": admission["tasklet_row_sufficiency_passed"],
        "collection_resource_admission_passed": admission["collection_resource_admission_passed"],
    }


def inspect(
    *, input_dir: Path, output: Path, expected_source_commit: str | None = None
) -> Mapping[str, Any]:
    _require_absent_ignored_output(output, SUMMARY_OUTPUT_NAME)
    manifest, samples, sessions = load_artifacts(input_dir)
    verification = verify_artifacts(input_dir)
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or _SHA256.fullmatch(source_commit) is None:
        raise ValueError("physical evidence source_commit must be a 40-hex SHA")
    current_commit = _require_clean_source(expected_source_commit)
    if source_commit != current_commit:
        raise ValueError("physical evidence source_commit does not match current HEAD")
    expected_counts = {
        "status": "completed",
        "sample_count": 5,
        "session_count": 5,
        "success_count": 5,
        "failed_count": 0,
        "unsupported_count": 0,
    }
    for field, expected in expected_counts.items():
        if verification.get(field) != expected:
            raise ValueError(f"canonical evidence has unexpected {field}: {verification.get(field)!r}")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("physical evidence must bind to a clean source worktree")
    semantic_config = _embedded_semantic_configuration(manifest)
    expected_contract = _expected_execution_contract(semantic_config)
    sessions_by_id = {str(session.get("session_instance_id")): session for session in sessions}
    if len(sessions_by_id) != 5:
        raise ValueError("physical evidence must contain five unique sessions")
    route_rows: dict[str, Mapping[str, Any]] = {}
    timing_scopes: set[str] = set()
    physical_ids: dict[str, str] = {}
    executable_ids: dict[str, str] = {}
    route_summary: dict[str, Any] = {}
    for sample in samples:
        route_id = sample.get("route_id")
        if route_id not in _ROUTE_SPECS or route_id in route_rows or sample.get("status") != "success":
            raise ValueError("physical evidence does not contain one successful sample per route")
        if (
            sample.get("case_id") != "quantization_stress_18q"
            or sample.get("plan_id") != "greedy"
        ):
            raise ValueError("physical evidence requires quantization_stress_18q with greedy")
        if (
            sample.get("attempt_kind") != "measurement"
            or sample.get("sample_index") != 0
            or sample.get("block_id") != 0
            or not isinstance(sample.get("output_sha256"), str)
            or _CANONICAL_ID.fullmatch(sample["output_sha256"]) is None
        ):
            raise ValueError("physical qualification requires one measured output hash per route")
        measurement = sample.get("measurement")
        if not isinstance(measurement, Mapping) or measurement.get("scope_id") != "steady_execution_v1":
            raise ValueError("physical qualification requires steady_execution_v1 timing scope")
        sample_identities = sample.get("identities")
        validation = sample.get("validation")
        numeric = sample.get("numeric_facts")
        if not isinstance(sample_identities, Mapping) or not isinstance(validation, Mapping) or not isinstance(numeric, Mapping):
            raise ValueError("physical sample lacks identity, validation, or numeric facts")
        for field in (
            "problem_id",
            "tensor_network_structure_id",
            "logical_plan_id",
        ):
            if sample_identities.get(field) != expected_contract[field]:
                raise ValueError(f"physical sample has an unexpected {field}")
        for field in ("physical_plan_id", "executable_id"):
            value = sample_identities.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"physical sample lacks {field}")
            (physical_ids if field == "physical_plan_id" else executable_ids)[route_id] = value
        expected_route = expected_contract["routes"][route_id]
        if physical_ids[route_id] != expected_route["physical_plan_id"]:
            raise ValueError(f"physical sample has an unexpected physical_plan_id for {route_id}")
        if numeric.get("numeric_policy") != "split_complex_float32_v1":
            raise ValueError("physical sample has an unexpected numeric policy")
        if not all(
            validation.get(field) is True
            for field in (
                "policy_reference_applicable",
                "policy_reference_passed",
                "full_precision_threshold_applicable",
                "full_precision_passed",
                "accuracy_qualified",
            )
        ):
            raise ValueError("physical sample did not pass replay and float32 validation")
        facts = _joined_facts(sample, sessions_by_id)
        for field, values in (("rank_response_timing_scope", timing_scopes),):
            value = facts.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"physical sample lacks {field}")
            values.add(value)
        if facts.get("kernel_policy") != expected_route["kernel_policy"]:
            raise ValueError(f"physical sample has an unexpected kernel_policy for {route_id}")
        dpus, tasklets = _ROUTE_SPECS[route_id]
        for field, expected in (
            ("requested_dpus", dpus),
            ("allocated_dpus", dpus),
            ("tasklets_per_dpu", tasklets),
            ("target_observed", "physical_hardware"),
            ("hardware_kernel_executed", True),
            ("simulator_kernel_executed", False),
            ("cpu_fallback_used", False),
            ("physical_target_verified", True),
            ("hardware_release_verified", True),
            ("binary_identity_verified", True),
            ("native_identity_verified", True),
            ("startup_resource_admission_passed", True),
            ("execution_resource_admission_passed", True),
        ):
            if facts.get(field) != expected:
                raise ValueError(f"physical evidence requires {field}={expected!r} for {route_id}")
        expected_admission = collection_resource_admission(expected_route["plan"])
        expected_resource_facts = {
            "active_dpus": dpus,
            "dominant_work_wave_populated_dpu_slots": expected_admission[
                "dominant_work_wave_populated_dpu_slots"
            ],
            "dominant_wave_useful_slots": expected_admission["dominant_wave_useful_slots"],
            "dominant_work_wave_allocated_dpu_slots": expected_admission[
                "dominant_work_wave_allocated_dpu_slots"
            ],
        }
        for field, expected in expected_resource_facts.items():
            if facts.get(field) != expected:
                raise ValueError(
                    f"physical sample has an unexpected {field} for {route_id}"
                )
        active = facts["active_dpus"]
        populated = facts["dominant_work_wave_populated_dpu_slots"]
        useful = facts["dominant_wave_useful_slots"]
        collection_bools = (
            "tasklet_row_sufficiency_passed",
            "dominant_work_wave_tasklet_row_sufficiency_passed",
            "collection_resource_admission_passed",
        )
        collection_ratios = (
            "dominant_work_wave_utilization",
            "arithmetic_weighted_dpu_slot_utilization",
            "arithmetic_weighted_tasklet_utilization",
        )
        for field in collection_bools:
            if type(facts.get(field)) is not bool:
                raise ValueError(f"physical sample lacks boolean collection resource fact {field}")
        for field in collection_ratios:
            value = facts.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"physical sample lacks finite collection resource fact {field}")
        for field in (*collection_bools, *collection_ratios):
            if facts[field] != expected_admission[field]:
                raise ValueError(f"physical sample has an unexpected {field} for {route_id}")
        expected_executable = _expected_executable_id(facts)
        if executable_ids[route_id] != expected_executable:
            raise ValueError(f"physical sample has an unexpected executable_id for {route_id}")
        session = sessions_by_id.get(str(sample.get("session_instance_id")))
        terminal = session.get("terminal_backend_facts") if session else None
        if not isinstance(session, Mapping) or session.get("status") != "success" or session.get("release_verified") is not True or not isinstance(terminal, Mapping):
            raise ValueError("physical session was not successfully released")
        if terminal.get("observed_tasklets_per_dpu") != tasklets or terminal.get("allocated_dpu_count") != dpus:
            raise ValueError(f"physical session resources mismatch for {route_id}")
        for field, expected in (
            ("target_observed", "physical_hardware"),
            ("hardware_kernel_executed", True),
            ("simulator_kernel_executed", False),
            ("cpu_fallback_used", False),
            ("physical_target_verified", True),
        ):
            if terminal.get(field) != expected:
                raise ValueError(f"physical session requires {field}={expected!r} for {route_id}")
        route_rows[route_id] = sample
        route_summary[route_id] = {
            "requested_dpus": dpus,
            "allocated_dpus": dpus,
            "requested_tasklets_per_dpu": tasklets,
            "observed_tasklets_per_dpu": tasklets,
            "active_dpus": active,
            "populated_dpu_slots": populated,
            "useful_dpu_slots": useful,
            "dominant_wave_zero_work_dpu_slots": dpus - populated,
            "collection_resource_facts": {
                field: facts[field]
                for field in (
                    "tasklet_row_sufficiency_passed",
                    "dominant_work_wave_tasklet_row_sufficiency_passed",
                    "dominant_work_wave_utilization",
                    "arithmetic_weighted_dpu_slot_utilization",
                    "arithmetic_weighted_tasklet_utilization",
                    "collection_resource_admission_passed",
                )
                if field in facts
            },
            "output_sha256": sample["output_sha256"],
            "physical_plan_id": physical_ids[route_id],
            "executable_id": executable_ids[route_id],
        }
    if (
        set(route_rows) != set(_ROUTE_SPECS)
        or len(timing_scopes) != 1
    ):
        raise ValueError("physical evidence does not bind one problem, plan, and policy")
    if len(set(physical_ids.values())) != 5 or len(set(executable_ids.values())) != 5:
        raise ValueError("route physical plans or tasklet executables are not distinct as required")
    t24 = _t24_plan_facts(expected_contract["routes"]["upmem_float32_1dpu_t24"]["plan"])
    payload = {
        "schema_version": "resource_general_summary_v1",
        "source": {"commit": source_commit, "worktree_dirty": False},
        "routes": route_summary,
        "idle_partial_facts": {"t24": t24},
        "validation": {"replay_and_float32_passed": True, "output_hashes_present": True},
        "provenance": {
            "experiment_id": manifest["experiment_id"],
            "identities": {
                field: expected_contract[field]
                for field in (
                    "problem_id",
                    "tensor_network_structure_id",
                    "logical_plan_id",
                )
            },
            "kernel_policy": expected_contract["routes"]["upmem_float32_1dpu_t3"]["kernel_policy"],
            "numeric_policy": "split_complex_float32_v1",
            "measurement_scope": "steady_execution_v1",
            "rank_response_timing_scope": next(iter(timing_scopes)),
        },
        "claim_ineligibility": {
            "performance_claim_eligible": False,
            "reason": "diagnostic_v1_resource_correctness_only",
        },
        "overall_pass": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--rank-path", required=True)
    prepare_parser.add_argument("--session-root", required=True)
    prepare_parser.add_argument("--expected-cpus", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--input", type=Path, required=True)
    inspect_parser.add_argument("--output", type=Path, required=True)
    inspect_parser.add_argument("--expected-source-commit")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build(output=args.output.resolve())
        elif args.command == "prepare":
            payload = prepare(
                output=args.output.resolve(),
                rank_path=args.rank_path,
                session_root=args.session_root,
                expected_cpus=_parse_cpus(args.expected_cpus),
            )
        else:
            payload = inspect(
                input_dir=args.input.resolve(),
                output=args.output.resolve(),
                expected_source_commit=args.expected_source_commit,
            )
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
