from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

from quantum_bench.bench.config import load_suite, route_config_for
from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.bench.summary import write_summary
from quantum_bench.circuits import load_circuit, manifest
from quantum_bench.core.jsonio import append_jsonl, write_json
from quantum_bench.core.records import BenchmarkCaseResult, BenchmarkContext, ExecutionProfile, RouteDecision, RouteIdentity, RouteOutput, RouteResult, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.tn import build_tensor_network, plan_task_graph
from quantum_bench.providers import route_registry
from quantum_bench.validation import compute_reference, validate


def run_suite(suite_path: Path, root_dir: Path) -> Path:
    suite = load_suite(suite_path)
    run_dir = create_run_dir(root_dir, suite["suite_id"])
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "plots" / ".matplotlib"))
    (run_dir / "plots" / ".matplotlib").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "resolved_suite.yml").write_text(yaml.safe_dump(suite, sort_keys=True), encoding="utf-8")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    routes = route_registry(root_dir)

    for case in suite["cases"]:
        case_id = str(case["case_id"])
        case_dir = run_dir / "cases" / case_id
        try:
            generated = _generate_case(case, suite, root_dir, case_dir)
        except Exception as exc:
            _record_case_setup_failure(run_dir, suite, case, exc)
            if suite["route_policy"].get("fail_fast"):
                raise
            continue

        for route_name in suite["route_policy"]["routes"]:
            route_config = route_config_for(suite, route_name)
            route = routes.get(route_name)
            if route is None:
                if route_config.get("required"):
                    raise ValueError(f"Required route is unknown: {route_name}")
                _record_route_unavailable(run_dir, suite, case, generated, route_name, route_config, "unknown route")
                continue
            context0 = _context(root_dir, run_dir, suite, case, route_config, 0)
            can_execute, skip_reason = route.can_execute(generated["graph"], context0)
            estimate = route.estimate(generated["graph"], context0)
            identity = _effective_identity(route.identity, route_config)
            decision = RouteDecision(
                route_name,
                route.backend_family,
                "selected" if can_execute else "skipped",
                skip_reason,
                estimate.estimated_flops,
                estimate.estimated_bytes,
                estimate.estimated_peak_memory,
                generated["graph"].path_summary.text,
                wram_fit=None if route.backend_family != "upmem" else can_execute,
                notes=estimate.notes,
            )
            append_jsonl(case_dir / "route_decisions.jsonl", decision)
            if not can_execute:
                if route_config.get("required"):
                    raise RuntimeError(f"Required route {route_name} cannot execute case {case_id}: {skip_reason}")
                for repeat_id in range(int(suite["repeats"])):
                    _record_skip(run_dir, suite, case, generated, identity, route.backend_family, skip_reason, repeat_id, route_config)
                continue

            for warmup_id in range(int(suite["warmups"])):
                _run_repeat(route, generated, suite, case, route_config, root_dir, run_dir, -(warmup_id + 1), persist=False)
            for repeat_id in range(int(suite["repeats"])):
                try:
                    _run_repeat(route, generated, suite, case, route_config, root_dir, run_dir, repeat_id, persist=True)
                except Exception as exc:
                    _record_repeat_failure(run_dir, suite, case, generated, identity, route.backend_family, repeat_id, exc, route_config)
                    if suite["route_policy"].get("fail_fast"):
                        raise

    write_summary(run_dir)
    return run_dir


def _generate_case(case: dict[str, Any], suite: dict[str, Any], root_dir: Path, case_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    circuit = load_circuit(case, root_dir)
    network = build_tensor_network(circuit)
    generate_s = time.perf_counter() - start
    graph = plan_task_graph(network, suite["planner"]["optimize"])
    write_json(case_dir / "circuit.json", manifest(circuit))
    write_json(case_dir / "task_graph.json", graph)
    return {"circuit": circuit, "network": network, "graph": graph, "generate_s": generate_s}


def _run_repeat(route: object, generated: dict[str, Any], suite: dict[str, Any], case: dict[str, Any], route_config: dict[str, Any], root_dir: Path, run_dir: Path, repeat_id: int, persist: bool) -> RouteResult:
    context = _context(root_dir, run_dir, suite, case, route_config, repeat_id)
    identity = _effective_identity(route.identity, route_config)
    start = time.perf_counter()
    prepared = route.prepare(generated["graph"], generated["network"], context)
    result = route.execute(prepared, context)
    validation = None
    validation_s = 0.0
    reference_s = 0.0
    if identity.validation_mode == "compare_output":
        reference, reference_s = compute_reference(generated["network"], suite["planner"]["optimize"])
        validation_start = time.perf_counter()
        if result.output.array is None:
            validation = None
            validation_s = time.perf_counter() - validation_start
            status = "failed"
            error = result.error or "validation required but route did not return an output array"
        else:
            validation = validate(result.output.array, reference, suite["tolerances"])
            validation_s = time.perf_counter() - validation_start
            status = result.status if validation.passed else "failed"
            error = result.error if validation.passed else "validation failed"
    elif identity.validation_mode == "benchmark_only":
        result.output.metadata.setdefault("validation_policy", "benchmark_only: route returns metrics only")
        status = result.status
        error = result.error
    elif identity.validation_mode == "skip_with_reason":
        result.output.metadata.setdefault("validation_policy", "skip_with_reason")
        status = result.status
        error = result.error
    else:
        status = result.status
        error = result.error
    total_s = time.perf_counter() - start
    profile = _merged_profile(result.profile, generated["generate_s"], generated["graph"].planning_time_s, reference_s, validation_s, total_s)
    result.profile = profile
    result.status = status
    result.error = error
    if persist:
        record = _case_record(run_dir.name, suite, case, generated, result, validation, repeat_id, identity, route_config)
        append_jsonl(run_dir / "raw" / f"{case['case_id']}.jsonl", record)
        write_json(run_dir / "validation" / f"{case['case_id']}_{result.route}_{repeat_id}.json", validation)
    return result


def _case_record(run_id: str, suite: dict[str, Any], case: dict[str, Any], generated: dict[str, Any], result: RouteResult, validation: object, repeat_id: int, identity: RouteIdentity, route_config: dict[str, Any]) -> BenchmarkCaseResult:
    circuit = generated["circuit"]
    graph = generated["graph"]
    return BenchmarkCaseResult(
        run_id=run_id,
        suite_id=suite["suite_id"],
        case_id=str(case["case_id"]),
        repeat_id=repeat_id,
        route=result.route,
        role=identity.role,
        simulation_method=identity.simulation_method,
        kernel_family=identity.kernel_family,
        hardware_target=identity.hardware_target,
        execution_mode=identity.execution_mode,
        output_contract=identity.output_contract,
        validation_mode=identity.validation_mode,
        backend_family=result.backend_family,
        status=result.status,
        skip_reason=result.error if result.status == "skipped" else None,
        n_qubits=circuit.n_qubits,
        depth=len(circuit.operations),
        circuit_family=str(case.get("circuit", {}).get("name", circuit.name)),
        gate_set=tuple(sorted({op.gate for op in circuit.operations})),
        planner=graph.path_summary.planner,
        path_summary=graph.path_summary.text,
        flops=sum(task.estimated_flops for task in graph.tasks),
        bytes=sum(task.estimated_bytes for task in graph.tasks),
        timings=to_jsonable(result.profile),
        total_time_s=result.profile.total_s,
        energy_joules=result.energy_joules,
        energy_source=result.energy_source,
        validation=to_jsonable(validation) if validation is not None else None,
        error=result.error if result.status != "skipped" else None,
    )


def _record_skip(run_dir: Path, suite: dict[str, Any], case: dict[str, Any], generated: dict[str, Any], identity: RouteIdentity, backend_family: str, reason: str | None, repeat_id: int, route_config: dict[str, Any]) -> None:
    result = RouteResult(identity.route_id, backend_family, "skipped", RouteOutput(identity.output_contract), ExecutionProfile(), None, "unavailable", reason)
    append_jsonl(run_dir / "raw" / f"{case['case_id']}.jsonl", _case_record(run_dir.name, suite, case, generated, result, None, repeat_id, identity, route_config))


def _record_route_unavailable(run_dir: Path, suite: dict[str, Any], case: dict[str, Any], generated: dict[str, Any], route_name: str, route_config: dict[str, Any], reason: str) -> None:
    identity = _unknown_identity(route_name)
    for repeat_id in range(int(suite["repeats"])):
        _record_skip(run_dir, suite, case, generated, identity, "unknown", reason, repeat_id, route_config)


def _record_case_setup_failure(run_dir: Path, suite: dict[str, Any], case: dict[str, Any], exc: Exception) -> None:
    record = {
        "run_id": run_dir.name,
        "suite_id": suite["suite_id"],
        "case_id": str(case.get("case_id", "unknown")),
        "repeat_id": 0,
        "route": "setup",
        "role": "setup",
        "simulation_method": None,
        "kernel_family": None,
        "hardware_target": None,
        "execution_mode": None,
        "output_contract": None,
        "validation_mode": None,
        "backend_family": "host",
        "status": "failed",
        "skip_reason": None,
        "n_qubits": None,
        "depth": None,
        "circuit_family": str(case.get("circuit", {}).get("name", "unknown")),
        "gate_set": [],
        "planner": suite.get("planner", {}).get("engine"),
        "path_summary": "",
        "flops": 0,
        "bytes": 0,
        "timings": {},
        "total_time_s": 0.0,
        "energy_joules": None,
        "energy_source": "unavailable",
        "validation": None,
        "error": f"{exc}\n{traceback.format_exc()}",
    }
    append_jsonl(run_dir / "raw" / f"{record['case_id']}.jsonl", record)


def _record_repeat_failure(run_dir: Path, suite: dict[str, Any], case: dict[str, Any], generated: dict[str, Any], identity: RouteIdentity, backend_family: str, repeat_id: int, exc: Exception, route_config: dict[str, Any]) -> None:
    result = RouteResult(identity.route_id, backend_family, "failed", RouteOutput(identity.output_contract), ExecutionProfile(), None, "unavailable", f"{exc}\n{traceback.format_exc()}")
    append_jsonl(run_dir / "raw" / f"{case['case_id']}.jsonl", _case_record(run_dir.name, suite, case, generated, result, None, repeat_id, identity, route_config))


def _context(root_dir: Path, run_dir: Path, suite: dict[str, Any], case: dict[str, Any], route_config: dict[str, Any], repeat_id: int) -> BenchmarkContext:
    return BenchmarkContext(
        root_dir=root_dir,
        run_dir=run_dir,
        suite=suite,
        case=case,
        route_config=route_config,
        repeat_id=repeat_id,
        tolerances=suite["tolerances"],
        timeout_s=suite.get("timeout_s"),
        memory_guard_gib=suite.get("memory_guard_gib"),
    )


def _merged_profile(profile: ExecutionProfile, generate_s: float, planning_s: float, reference_s: float, validation_s: float, total_s: float) -> ExecutionProfile:
    return ExecutionProfile(
        generate_s=generate_s,
        planning_s=planning_s,
        lowering_s=profile.lowering_s,
        prepare_s=profile.prepare_s,
        h2d_s=profile.h2d_s,
        kernel_s=profile.kernel_s,
        d2h_s=profile.d2h_s,
        reduction_s=profile.reduction_s + reference_s,
        validation_s=validation_s,
        total_s=total_s,
    )


def _effective_identity(identity: RouteIdentity, route_config: dict[str, Any]) -> RouteIdentity:
    role = route_config.get("role") or identity.role
    if role == identity.role:
        return identity
    return RouteIdentity(
        route_id=identity.route_id,
        display_name=identity.display_name,
        role=role,
        simulation_method=identity.simulation_method,
        kernel_family=identity.kernel_family,
        hardware_target=identity.hardware_target,
        execution_mode=identity.execution_mode,
        output_contract=identity.output_contract,
        validation_mode=identity.validation_mode,
    )


def _unknown_identity(route_name: str) -> RouteIdentity:
    return RouteIdentity(
        route_id=route_name,
        display_name=route_name,
        role="probe_only",
        simulation_method="unknown",
        kernel_family="none",
        hardware_target="unknown",
        execution_mode="unknown",
        output_contract="none",
        validation_mode="skip_with_reason",
    )
