"""Execution runtime for the physical one-DPU path/quantization study.

The runtime deliberately keeps the narrow experimental boundary visible:
one native interactive host owns one DPU for one circuit case, tasks execute
serially, and each logical TaskGraph dependency receives the physical output
of its predecessor.  CPU contraction is used only after the measured section
for validation, never as a substitute for a failed native output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from quantum_bench.bench.reporting import write_normalized_records, write_run_manifest
from quantum_bench.bench.run_dirs import EVIDENCE_ARTIFACT_KIND, sanitize
from quantum_bench.core.jsonio import write_json, write_jsonl
from quantum_bench.core.records import JsonDict, TaskGraph, TensorSpec, TensorValue
from quantum_bench.environment import capture_environment
from quantum_bench.routing.generic_numeric_contract import classify_numeric
from quantum_bench.routing.generic_prepare import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    generic_loop_reference_int32,
    prepare_generic_task,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareInteractiveSession,
    HardwareInteractiveSessionError,
    HardwareSessionBuild,
    HardwareSessionExecution,
    HardwareSessionTask,
    build_hardware_session,
    load_session_output,
    start_hardware_session,
    write_session_task,
)
from quantum_bench.targets.upmem.hardware_taskgraph_study import (
    HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
    HardwareTaskGraphStudyProfile,
    HardwareTaskGraphStudySuite,
    hardware_taskgraph_study_profile_metadata,
)
from quantum_bench.tn.contract import contract_binary_task
from quantum_bench.tn.execution import order_final_tensor
from quantum_bench.tn.execution_bundle import (
    canonical_hash,
    execution_identity_metadata,
    executor_config_hash,
    with_execution_identity,
)
from quantum_bench.tn.network import TensorNetworkValue


STUDY_RUNTIME_SCHEMA_VERSION = "upmem_hardware_taskgraph_study_runtime_v1"
STUDY_BENCHMARK_SCHEMA_VERSION = "upmem_hardware_taskgraph_study_benchmark_v1"
STUDY_TIMING_SCOPE = "one_dpu_steady_state_full_taskgraph_v1"


@dataclass(frozen=True)
class StudyGraphExecution:
    status: str
    reason: str | None
    output: np.ndarray | None
    summary: JsonDict
    task_metrics: tuple[JsonDict, ...]


def run_study_suite(
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphStudySuite,
    *,
    environment: Mapping[str, str],
):
    """Build once, run each case in one persistent physical session, write evidence."""

    # Imported late to avoid a module-import cycle with the public command.
    from quantum_bench.bench.upmem_hardware_taskgraph_study import (
        UpmemHardwareTaskGraphStudyResult,
        _failure_stage,
        prepare_study_case,
    )

    from shutil import copy2

    copy2(suite.suite_path, run_dir / "config" / "resolved_suite.yml")
    write_json(
        run_dir / "config" / "hardware_profile.json",
        hardware_taskgraph_study_profile_metadata(suite.profile),
    )
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    run_manifest = write_run_manifest(
        run_dir,
        run_kind="upmem_hardware_taskgraph_path_quantization_study",
        suite_id=str(suite.suite["suite_id"]),
        suite_path=str(suite.suite_path),
        artifact_kind=EVIDENCE_ARTIFACT_KIND,
        route_label="upmem_hw_taskgraph_study",
        route_id=HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        backend_id=suite.profile.backend_id,
        execution_scope="physical_single_dpu_persistent_taskgraph_study",
        evidence_type="physical_hardware_one_dpu_steady_state",
        upmem_execution_mode="sdk_hardware_single_dpu_persistent_taskgraph",
        artifact_retention="full",
        summary="upmem_hardware_taskgraph_study_summary.json",
        root_dir=root_dir,
    )

    try:
        native_build = build_hardware_session(
            root_dir,
            run_dir / "native_session",
            profile=suite.profile,  # structural profile compatibility
            environment=environment,
        )
    except Exception as exc:
        failure = _build_failure_record(suite, str(exc), run_manifest)
        write_normalized_records(run_dir, [failure])
        summary_path = run_dir / "upmem_hardware_taskgraph_study_summary.json"
        write_json(
            summary_path,
            {
                "schema_version": STUDY_RUNTIME_SCHEMA_VERSION,
                "status": "failed",
                "suite_id": suite.suite["suite_id"],
                "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
                "failure_stage": _failure_stage(
                    str(exc), default="native_build_failed"
                ),
                "reason": str(exc),
                "normalized_records": "normalized_records.jsonl",
            },
        )
        run_manifest.update(
            {
                "summary": summary_path.name,
                "hardware_available": "not_verified_by_execution",
            }
        )
        write_json(run_dir / "run_manifest.json", run_manifest)
        return UpmemHardwareTaskGraphStudyResult(run_dir, summary_path, "failed", 1)

    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    case_statuses: list[JsonDict] = []
    stop_after_failure = False
    for case in suite.suite["cases"]:
        case_id = str(case["case_id"])
        if stop_after_failure:
            case_statuses.append(
                {"case_id": case_id, "status": "not_attempted_after_prior_failure"}
            )
            continue
        try:
            prepared = prepare_study_case(
                root_dir,
                run_dir / "cases" / sanitize(case_id),
                suite,
                case,
            )
            case_records, case_warmups, case_status = _run_case(
                root_dir=root_dir,
                run_dir=run_dir,
                suite=suite,
                case=case,
                prepared=prepared,
                native_build=native_build,
                environment=environment,
                source_commit=run_manifest.get("benchmark_source_commit"),
            )
        except Exception as exc:
            case_records = [
                _build_failure_record(suite, str(exc), run_manifest, case=case)
            ]
            case_warmups = []
            case_status = {
                "case_id": case_id,
                "status": "failed",
                "failure_stage": _failure_stage(
                    str(exc), default="hardware_profile_violation"
                ),
                "reason": str(exc),
            }
        records.extend(case_records)
        warmups.extend(case_warmups)
        case_statuses.append(case_status)
        if case_status.get("status") != "passed":
            stop_after_failure = True

    write_jsonl(run_dir / "warmups.jsonl", warmups)
    write_normalized_records(run_dir, records)
    completed = bool(records) and all(
        record.get("status") == "completed" for record in records
    )
    summary_path = run_dir / "upmem_hardware_taskgraph_study_summary.json"
    write_json(
        summary_path,
        {
            "schema_version": STUDY_RUNTIME_SCHEMA_VERSION,
            "status": "completed" if completed else "failed",
            "suite_id": suite.suite["suite_id"],
            "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
            "backend_id": suite.profile.backend_id,
            "row_count": len(records),
            "warmup_count": len(warmups),
            "case_statuses": case_statuses,
            "hardware_profile": hardware_taskgraph_study_profile_metadata(
                suite.profile
            ),
            "native_build": _native_build_metadata(native_build, root_dir),
            "normalized_records": "normalized_records.jsonl",
            "warmups": "warmups.jsonl",
            "claim_boundary": (
                "measured physical one-DPU path/numeric-mode comparison only; no cross-backend speedup, energy, parallel scheduling, or multi-DPU claim"
            ),
        },
    )
    run_manifest.update(
        {
            "summary": summary_path.name,
            "hardware_available": "verified_by_execution"
            if completed
            else "not_verified_by_execution",
        }
    )
    write_json(run_dir / "run_manifest.json", run_manifest)
    return UpmemHardwareTaskGraphStudyResult(
        run_dir, summary_path, "completed" if completed else "failed", len(records)
    )


def _run_case(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphStudySuite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    native_build: HardwareSessionBuild,
    environment: Mapping[str, str],
    source_commit: object,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict]:
    case_id = str(case["case_id"])
    all_variants = tuple(
        (variant_id, mode)
        for variant_id in (item.variant_id for item in suite.variants)
        for mode in suite.profile.numeric_modes
    )
    records: list[JsonDict] = []
    warmups: list[JsonDict] = []
    executions: list[
        tuple[str, str, int, tuple[str, ...], int, StudyGraphExecution]
    ] = []
    warmup_failure: tuple[str, str, int, StudyGraphExecution] | None = None
    session_start_started = time.perf_counter()
    session: HardwareInteractiveSession | None = None
    startup_error: str | None = None
    startup_metadata: JsonDict = {}
    try:
        session = start_hardware_session(
            native_build,
            session_id=f"study-{sanitize(case_id)}",
            profile=suite.profile,  # structural profile compatibility
            environment=environment,
        )
        startup_metadata = session.startup_metadata
    except Exception as exc:
        startup_error = str(exc)
    session_startup_time_s = time.perf_counter() - session_start_started
    if session is None:
        failure = _build_failure_record(
            suite,
            startup_error or "hardware_allocation_failed",
            {"benchmark_source_commit": source_commit},
            case=case,
        )
        failure["session_startup_time_s"] = session_startup_time_s
        return (
            [failure],
            warmups,
            {
                "case_id": case_id,
                "status": "failed",
                "failure_stage": failure.get("failure_stage"),
                "reason": startup_error,
            },
        )

    close_metadata: JsonDict = {}
    try:
        for warmup_id in range(int(suite.suite["warmups"])):
            ordered = _rotated_variants(all_variants, warmup_id)
            for order_index, (variant_id, mode) in enumerate(ordered):
                execution = _execute_variant(
                    session=session,
                    native_build=native_build,
                    root_dir=root_dir,
                    work_dir=_execution_work_dir(
                        native_build, case_id, "warmup", warmup_id, variant_id, mode
                    ),
                    graph=prepared["variants"][variant_id]["graph"],
                    network=prepared["network"],
                    reference_output=prepared["variants"][variant_id][
                        "reference_output"
                    ],
                    quantization_mode=mode,
                    profile=suite.profile,
                    request_prefix=f"warmup-{warmup_id:02d}-{variant_id}-{mode}",
                )
                warmups.append(
                    _warmup_row(
                        case,
                        variant_id,
                        mode,
                        warmup_id,
                        ordered,
                        order_index,
                        execution,
                    )
                )
                if execution.status != "completed":
                    warmup_failure = (variant_id, mode, warmup_id, execution)
                    break
            if warmup_failure is not None:
                break

        if warmup_failure is None:
            for repeat_id in range(int(suite.suite["repeats"])):
                ordered = _rotated_variants(all_variants, repeat_id)
                order_names = tuple(f"{variant}/{mode}" for variant, mode in ordered)
                for order_index, (variant_id, mode) in enumerate(ordered):
                    execution = _execute_variant(
                        session=session,
                        native_build=native_build,
                        root_dir=root_dir,
                        work_dir=_execution_work_dir(
                            native_build, case_id, "repeat", repeat_id, variant_id, mode
                        ),
                        graph=prepared["variants"][variant_id]["graph"],
                        network=prepared["network"],
                        reference_output=prepared["variants"][variant_id][
                            "reference_output"
                        ],
                        quantization_mode=mode,
                        profile=suite.profile,
                        request_prefix=f"repeat-{repeat_id:02d}-{variant_id}-{mode}",
                    )
                    executions.append(
                        (
                            variant_id,
                            mode,
                            repeat_id,
                            order_names,
                            order_index,
                            execution,
                        )
                    )
                    if execution.status != "completed":
                        break
                if executions and executions[-1][-1].status != "completed":
                    break
    finally:
        close = session.close(timeout_s=suite.profile.timeout_s)
        close_metadata = {
            "hardware_release_verified": close.release_confirmed,
            "release_time_s": close.release_time_s,
            "session_close_status": close.status,
            "session_close_failure_stage": close.failure_stage,
            "session_process_returncode": close.process_returncode,
            "session_stdout_snippet": close.stdout_snippet,
            "session_stderr_snippet": close.stderr_snippet,
        }

    if warmup_failure is not None:
        variant_id, mode, warmup_id, execution = warmup_failure
        failure = _build_failure_record(
            suite,
            execution.reason or "warmup_native_execution_failed",
            {"benchmark_source_commit": source_commit},
            case=case,
        )
        failure.update(
            {
                "phase": "warmup",
                "warmup_id": warmup_id,
                "path_variant_id": variant_id,
                "quantization_mode": mode,
                "session_startup_time_s": session_startup_time_s,
                **close_metadata,
            }
        )
        return (
            [failure],
            warmups,
            {
                "case_id": case_id,
                "status": "failed",
                "phase": "warmup",
                "failure_stage": failure.get("failure_stage"),
            },
        )

    for variant_id, mode, repeat_id, order_names, order_index, execution in executions:
        record = _normalized_record(
            root_dir=root_dir,
            run_dir=run_dir,
            suite=suite,
            case=case,
            prepared=prepared,
            variant_id=variant_id,
            quantization_mode=mode,
            repeat_id=repeat_id,
            variant_order=order_names,
            variant_order_index=order_index,
            result=execution,
            native_build=native_build,
            session_startup_time_s=session_startup_time_s,
            startup_metadata=startup_metadata,
            close_metadata=close_metadata,
            source_commit=source_commit,
        )
        records.append(record)

    failed = next(
        (record for record in records if record.get("status") != "completed"), None
    )
    if failed is not None:
        return (
            records,
            warmups,
            {
                "case_id": case_id,
                "status": "failed",
                "phase": "repeat",
                "failure_stage": failed.get("failure_stage"),
                "attempted_repeats": len(
                    {record.get("repeat_id") for record in records}
                ),
            },
        )
    if close_metadata.get("hardware_release_verified") is not True:
        for record in records:
            record["status"] = "failed"
            record["failure_stage"] = (
                close_metadata.get("session_close_failure_stage")
                or "hardware_release_failed"
            )
            record["reason"] = "hardware_release_failed"
        return (
            records,
            warmups,
            {
                "case_id": case_id,
                "status": "failed",
                "failure_stage": close_metadata.get("session_close_failure_stage")
                or "hardware_release_failed",
            },
        )
    return (
        records,
        warmups,
        {
            "case_id": case_id,
            "status": "passed",
            "warmups": len(warmups),
            "timed_rows": len(records),
            "session_startup_time_s": session_startup_time_s,
            **close_metadata,
        },
    )


def _execute_variant(
    *,
    session: HardwareInteractiveSession,
    native_build: HardwareSessionBuild,
    root_dir: Path,
    work_dir: Path,
    graph: TaskGraph,
    network: TensorNetworkValue,
    reference_output: np.ndarray,
    quantization_mode: str,
    profile: HardwareTaskGraphStudyProfile,
    request_prefix: str,
) -> StudyGraphExecution:
    """Execute graph dependencies through native outputs in one live session."""

    started = time.perf_counter()
    graph = with_execution_identity(graph)
    execution_metadata = {
        **execution_identity_metadata(graph, plan_reused=True),
        "executor_config_hash": executor_config_hash(
            HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
            {
                "hardware_profile_version": profile.version,
                "backend_id": profile.backend_id,
                "quantization_mode": quantization_mode,
                "complex_policy": profile.complex_policy,
                "session_protocol": profile.session_protocol,
                "session_scope": "case_benchmark_block",
            },
        ),
    }
    if quantization_mode not in profile.numeric_modes:
        return _failure_execution(
            graph,
            profile,
            quantization_mode,
            execution_metadata,
            "hardware_profile_violation: unsupported numeric mode",
            started,
        )
    try:
        work_dir.resolve().relative_to(native_build.session_root.resolve())
    except ValueError:
        return _failure_execution(
            graph,
            profile,
            quantization_mode,
            execution_metadata,
            "hardware_profile_violation: study work directory must be inside persistent native session root",
            started,
        )
    work_dir.mkdir(parents=True, exist_ok=False)
    caps = GenericTaskPreparationCaps(
        max_rank=profile.max_rank,
        max_tensor_elements=profile.max_tensor_elements,
        max_contracted_combinations=profile.max_contracted_combinations,
    )
    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    tensor_hashes = {
        tensor_id: _array_hash(array) for tensor_id, array in tensors.items()
    }
    physical_output_hashes: dict[str, str] = {}
    completed: set[str] = set()
    metrics: list[JsonDict] = []
    totals = _empty_totals()
    final_labels: tuple[int, ...] | None = None

    # Everything from task preparation through native request completion and
    # output reconstruction is the steady-state full-graph wall interval.
    steady_started = time.perf_counter()
    for task_index, task in enumerate(graph.tasks):
        if task.id in completed:
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                "duplicate_contraction_detected",
                started,
                steady_started,
            )
        missing = [
            dependency
            for dependency in task.dependencies
            if dependency not in completed
        ]
        if missing:
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                f"runtime_task_dependency_missing:{','.join(missing)}",
                started,
                steady_started,
            )
        if any(tensor_id not in tensors for tensor_id in task.input_tensor_ids):
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                "runtime_input_tensor_missing",
                started,
                steady_started,
            )

        left = TensorValue(
            _spec_for(task.input_tensor_ids[0], labels, tensors),
            tensors[task.input_tensor_ids[0]],
        )
        right = TensorValue(
            _spec_for(task.input_tensor_ids[1], labels, tensors),
            tensors[task.input_tensor_ids[1]],
        )
        input_hashes = {
            tensor_id: tensor_hashes[tensor_id] for tensor_id in task.input_tensor_ids
        }
        input_from_physical = {
            tensor_id: physical_output_hashes[tensor_id]
            for tensor_id in task.input_tensor_ids
            if tensor_id in physical_output_hashes
        }
        task_dir = work_dir / "logical_tasks" / f"{task_index:04d}_{sanitize(task.id)}"
        task_dir.mkdir(parents=True, exist_ok=False)
        prepared_started = time.perf_counter()
        try:
            component_tasks, component_order, component_preparations = (
                _prepare_component_tasks(
                    task_dir=task_dir,
                    task=task,
                    task_index=task_index,
                    left=left,
                    right=right,
                    quantization_mode=quantization_mode,
                    caps=caps,
                    profile=profile,
                )
            )
        except Exception as exc:
            metrics.append(_failed_task_metric(task, task_index, str(exc)))
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                str(exc),
                started,
                steady_started,
            )
        prepare_time = time.perf_counter() - prepared_started
        totals["host_prepare_time_s"] += prepare_time
        try:
            native = session.submit(
                component_tasks, request_id=f"{request_prefix}-{task_index:04d}"
            )
        except HardwareInteractiveSessionError as exc:
            metrics.append(
                _failed_task_metric(
                    task, task_index, str(exc), failure_stage=exc.failure_stage
                )
            )
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                str(exc),
                started,
                steady_started,
            )
        if native.status != "completed":
            metrics.append(
                _failed_task_metric(
                    task,
                    task_index,
                    native.failure_stage or "hardware_session_failed",
                    execution=native,
                )
            )
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                native.failure_stage or "hardware_session_failed",
                started,
                steady_started,
            )
        reconstructed_started = time.perf_counter()
        try:
            component_outputs = {
                name: load_session_output(session_task)
                for name, session_task in zip(component_order, component_tasks)
            }
            output, representation = _combine_components(component_outputs)
        except Exception as exc:
            metrics.append(
                _failed_task_metric(task, task_index, str(exc), execution=native)
            )
            return _stopped_execution(
                graph,
                profile,
                quantization_mode,
                execution_metadata,
                metrics,
                totals,
                str(exc),
                started,
                steady_started,
            )
        reconstruction_time = time.perf_counter() - reconstructed_started
        totals["host_reconstruction_time_s"] += reconstruction_time
        response_tasks = (
            native.response.get("tasks")
            if isinstance(native.response.get("tasks"), list)
            else []
        )
        timing = _response_timing(response_tasks)
        if totals["allocation_time_s"] is None:
            totals["allocation_time_s"] = _number_or_none(
                native.response.get("allocation_time_s")
            )
        if totals["binary_load_time_s"] is None:
            totals["binary_load_time_s"] = _number_or_none(
                native.response.get("binary_load_time_s")
            )
        for key in ("h2d_time_s", "kernel_time_s", "d2h_time_s"):
            totals[key] += timing[key]
        totals["application_visible_h2d_bytes"] += sum(
            item.application_visible_h2d_bytes for item in component_tasks
        )
        totals["application_visible_d2h_bytes"] += sum(
            item.application_visible_d2h_bytes for item in component_tasks
        )
        exact_match = _component_exact_integer_match(
            component_order, component_tasks, component_preparations
        )
        output_hash = _array_hash(output)
        metric = {
            "task_id": task.id,
            "task_index": task_index,
            "status": "completed",
            "input_tensor_ids": list(task.input_tensor_ids),
            "output_tensor_id": task.output_tensor_id,
            "input_tensor_hashes": input_hashes,
            "input_physical_output_hashes": input_from_physical,
            "physical_dependency_input_count": len(input_from_physical),
            "output_tensor_hash": output_hash,
            "output_is_physical_native_result": True,
            "component_task_ids": [item.task_id for item in component_tasks],
            "complex_representation": representation,
            "split_complex_component_count": len(component_tasks)
            if representation == "split_real_imag"
            else 0,
            "quantization_mode": quantization_mode,
            "input_dtype_on_dpu": "float32" if quantization_mode == "none" else "int8",
            "accumulator_dtype_on_dpu": "float32"
            if quantization_mode == "none"
            else "int32",
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "application_visible_h2d_bytes": sum(
                item.application_visible_h2d_bytes for item in component_tasks
            ),
            "application_visible_d2h_bytes": sum(
                item.application_visible_d2h_bytes for item in component_tasks
            ),
            "application_visible_transfer_bytes": sum(
                item.application_visible_transfer_bytes for item in component_tasks
            ),
            "host_prepare_time_s": prepare_time,
            "host_reconstruction_time_s": reconstruction_time,
            **timing,
            "session_process_time_s": native.process_time_s,
            "exact_integer_match": exact_match
            if quantization_mode == "per_task_input_quantize"
            else None,
            "session_response_artifact": str(
                native.response_path.relative_to(native_build.session_root)
            ),
        }
        metrics.append(metric)
        tensors[task.output_tensor_id] = output
        tensor_hashes[task.output_tensor_id] = output_hash
        physical_output_hashes[task.output_tensor_id] = output_hash
        labels[task.output_tensor_id] = task.output_labels
        completed.add(task.id)
        final_labels = task.output_labels

    if final_labels is None:
        return _stopped_execution(
            graph,
            profile,
            quantization_mode,
            execution_metadata,
            metrics,
            totals,
            "final_tensor_missing",
            started,
            steady_started,
        )
    final_id = graph.tasks[-1].output_tensor_id
    output, transposed = order_final_tensor(
        np.asarray(tensors[final_id]), final_labels, graph.network.output_labels
    )
    steady_elapsed = time.perf_counter() - steady_started
    totals["steady_state_graph_execution_s"] = steady_elapsed
    known = (
        totals["h2d_time_s"]
        + totals["kernel_time_s"]
        + totals["d2h_time_s"]
        + totals["host_prepare_time_s"]
        + totals["host_reconstruction_time_s"]
    )
    totals["host_control_time_s"] = max(0.0, steady_elapsed - known)
    # All CPU validation follows this point and cannot alter a timing result.
    validation_started = time.perf_counter()
    policy_output = _policy_reference_graph(graph, network, quantization_mode, caps)
    policy_metrics = _array_metrics(
        policy_output, output, tolerance=_tolerance(quantization_mode)
    )
    full_precision_metrics = _array_metrics(
        reference_output, output, tolerance=_tolerance(quantization_mode)
    )
    validation_time_s = time.perf_counter() - validation_started
    dependency_safe = all(
        metric["physical_dependency_input_count"]
        == len(
            [
                tensor_id
                for tensor_id in metric["input_tensor_ids"]
                if tensor_id.startswith("result_")
            ]
        )
        for metric in metrics
    )
    exact_integer_match = (
        all(metric.get("exact_integer_match") is True for metric in metrics)
        if quantization_mode == "per_task_input_quantize"
        else None
    )
    validation_status = "passed" if policy_metrics["passed"] else "failed"
    summary: JsonDict = {
        "schema_version": STUDY_RUNTIME_SCHEMA_VERSION,
        "status": "completed" if validation_status == "passed" else "failed",
        "reason": None if validation_status == "passed" else "output_validation_failed",
        "failure_stage": None
        if validation_status == "passed"
        else "output_validation_failed",
        "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        "backend_id": profile.backend_id,
        "quantization_mode": quantization_mode,
        "hardware_execution": True,
        "hardware_kernel_executed": len(metrics) == len(graph.tasks),
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "target_observed": "hardware",
        "task_count": len(graph.tasks),
        "native_task_execution_count": len(metrics),
        "source_task_count": len(graph.tasks),
        "source_task_completion_count": len(completed),
        "physical_dependency_chain_verified": dependency_safe,
        "final_output_transposed_to_contract": transposed,
        "validation_status": validation_status,
        "policy_reference_validation": policy_metrics,
        "full_precision_accuracy": full_precision_metrics,
        "max_abs_error": full_precision_metrics["max_abs_error"],
        "l2_error": full_precision_metrics["l2_error"],
        "validation_max_abs_error": policy_metrics["max_abs_error"],
        "exact_integer_match": exact_integer_match,
        "application_visible_h2d_bytes": totals["application_visible_h2d_bytes"],
        "application_visible_d2h_bytes": totals["application_visible_d2h_bytes"],
        "application_visible_transfer_bytes": totals["application_visible_h2d_bytes"]
        + totals["application_visible_d2h_bytes"],
        "actual_h2d_bytes": totals["application_visible_h2d_bytes"],
        "actual_d2h_bytes": totals["application_visible_d2h_bytes"],
        "actual_transfer_bytes": totals["application_visible_h2d_bytes"]
        + totals["application_visible_d2h_bytes"],
        **totals,
        "validation_time_s": validation_time_s,
        "total_route_time_s": time.perf_counter() - started,
        "timing_scope": STUDY_TIMING_SCOPE,
        "timing_is_bringup_only": False,
        "hardware_timing_available": True,
        "hardware_speedup_applicable": False,
        "task_metrics": metrics,
        "native_source_tree_hash": native_build.source_tree_hash,
        "host_binary_hash": native_build.host_binary_hash,
        "dpu_binary_hash": native_build.dpu_binary_hash,
        "native_build_command": list(native_build.build_command),
        "sdk_tools": native_build.sdk_tools,
        **execution_metadata,
    }
    return StudyGraphExecution(
        status=str(summary["status"]),
        reason=summary["reason"],
        output=np.asarray(output),
        summary=summary,
        task_metrics=tuple(metrics),
    )


def _prepare_component_tasks(
    *,
    task_dir: Path,
    task,
    task_index: int,
    left: TensorValue,
    right: TensorValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
    profile: HardwareTaskGraphStudyProfile,
) -> tuple[list[HardwareSessionTask], tuple[str, ...], dict[str, Any]]:
    left_array = np.asarray(left.array)
    right_array = np.asarray(right.array)
    if (
        classify_numeric(left_array).has_nonfinite
        or classify_numeric(right_array).has_nonfinite
    ):
        raise RuntimeError("nonfinite_values_not_supported")
    components = (
        _split_components(left_array, right_array)
        if (
            classify_numeric(left_array).has_nonzero_imaginary
            or classify_numeric(right_array).has_nonzero_imaginary
        )
        else {"real": (_real_array(left_array), _real_array(right_array))}
    )
    tasks: list[HardwareSessionTask] = []
    preparations: dict[str, Any] = {}
    for component_index, (name, (left_part, right_part)) in enumerate(
        components.items()
    ):
        preparation = prepare_generic_task(
            GenericTaskPreparationInput(
                task=task,
                left_tensor=TensorValue(left.spec, left_part),
                right_tensor=TensorValue(right.spec, right_part),
                quantization_mode=quantization_mode,  # type: ignore[arg-type]
                caps=caps,
                route_id=HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
            )
        )
        if preparation.status != "prepared":
            raise RuntimeError(preparation.reason or preparation.status)
        tasks.append(
            write_session_task(
                task_dir,
                sequence=task_index * 8 + component_index,
                task_id=f"{task.id}__{name}",
                preparation=preparation,
                max_rank=profile.max_rank,
            )
        )
        preparations[name] = preparation
    return tasks, tuple(components), preparations


def _combine_components(outputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if "real" in outputs:
        return np.asarray(outputs["real"]), "real"
    required = ("ar_br", "ai_bi", "ar_bi", "ai_br")
    if any(name not in outputs for name in required):
        raise RuntimeError("result_transfer_failed: split-complex component missing")
    return (
        (outputs["ar_br"] - outputs["ai_bi"])
        + 1j * (outputs["ar_bi"] + outputs["ai_br"]),
        "split_real_imag",
    )


def _policy_reference_graph(
    graph: TaskGraph,
    network: TensorNetworkValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
) -> np.ndarray:
    tensors = {tensor.spec.id: np.asarray(tensor.array) for tensor in network.tensors}
    labels = {tensor.spec.id: tensor.spec.labels for tensor in network.tensors}
    final_labels: tuple[int, ...] | None = None
    for task in graph.tasks:
        left = TensorValue(
            _spec_for(task.input_tensor_ids[0], labels, tensors),
            tensors[task.input_tensor_ids[0]],
        )
        right = TensorValue(
            _spec_for(task.input_tensor_ids[1], labels, tensors),
            tensors[task.input_tensor_ids[1]],
        )
        tensors[task.output_tensor_id] = _policy_reference_task(
            task, left, right, quantization_mode, caps
        )
        labels[task.output_tensor_id] = task.output_labels
        final_labels = task.output_labels
    if final_labels is None:
        raise RuntimeError("final_tensor_missing")
    return order_final_tensor(
        tensors[graph.tasks[-1].output_tensor_id],
        final_labels,
        graph.network.output_labels,
    )[0]


def _policy_reference_task(
    task,
    left: TensorValue,
    right: TensorValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
) -> np.ndarray:
    left_array = np.asarray(left.array)
    right_array = np.asarray(right.array)
    complex_task = (
        classify_numeric(left_array).has_nonzero_imaginary
        or classify_numeric(right_array).has_nonzero_imaginary
    )
    if not complex_task:
        preparation = _prepare_policy_component(
            task, left, right, quantization_mode, caps
        )
        return contract_binary_task(
            task,
            _converted_operand(preparation, "left"),
            _converted_operand(preparation, "right"),
        )
    components = _split_components(left_array, right_array)
    prepared = {
        name: _prepare_policy_component(
            task,
            TensorValue(left.spec, left_part),
            TensorValue(right.spec, right_part),
            quantization_mode,
            caps,
        )
        for name, (left_part, right_part) in components.items()
    }
    left_policy = _converted_operand(
        prepared["ar_br"], "left"
    ) + 1j * _converted_operand(prepared["ai_bi"], "left")
    right_policy = _converted_operand(
        prepared["ar_br"], "right"
    ) + 1j * _converted_operand(prepared["ar_bi"], "right")
    return contract_binary_task(task, left_policy, right_policy)


def _prepare_policy_component(
    task,
    left: TensorValue,
    right: TensorValue,
    quantization_mode: str,
    caps: GenericTaskPreparationCaps,
):
    preparation = prepare_generic_task(
        GenericTaskPreparationInput(
            task=task,
            left_tensor=left,
            right_tensor=right,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
            caps=caps,
            route_id=HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        )
    )
    if preparation.status != "prepared" or preparation.prepared_operands is None:
        raise RuntimeError(preparation.reason or preparation.status)
    return preparation


def _converted_operand(preparation, side: str) -> np.ndarray:
    operands = preparation.prepared_operands
    if operands is None:
        raise RuntimeError("prepared_operands_missing")
    value = operands.left_operand if side == "left" else operands.right_operand
    if value is None:
        raise RuntimeError("prepared_operand_missing")
    if operands.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT:
        return np.asarray(value, dtype=np.float32)
    conversion = (
        preparation.left_conversion if side == "left" else preparation.right_conversion
    )
    if conversion is None:
        raise RuntimeError("quantization_conversion_missing")
    return np.asarray(value, dtype=np.float64) * float(conversion.scale)


def _component_exact_integer_match(
    order: Sequence[str],
    tasks: Sequence[HardwareSessionTask],
    preparations: Mapping[str, Any],
) -> bool:
    for name, session_task in zip(order, tasks):
        if session_task.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT:
            continue
        operands = preparations[name].prepared_operands
        if operands is None:
            return False
        expected = generic_loop_reference_int32(
            operands.left_quantized,
            operands.right_quantized,
            output_shape=tuple(session_task.output_shape),
            left_strides=preparations[name].left_strides,
            right_strides=preparations[name].right_strides,
            output_strides=preparations[name].output_strides,
            output_to_left_axes=preparations[name].output_to_left_axes,
            output_to_right_axes=preparations[name].output_to_right_axes,
            contracted_to_left_axes=preparations[name].contracted_to_left_axes,
            contracted_to_right_axes=preparations[name].contracted_to_right_axes,
            contracted_dims=preparations[name].contracted_dims,
        )
        actual = np.fromfile(session_task.output_path, dtype="<i4")
        if actual.size != expected.size or not np.array_equal(
            actual.reshape(expected.shape), expected
        ):
            return False
    return True


def _normalized_record(
    *,
    root_dir: Path,
    run_dir: Path,
    suite: HardwareTaskGraphStudySuite,
    case: Mapping[str, Any],
    prepared: Mapping[str, Any],
    variant_id: str,
    quantization_mode: str,
    repeat_id: int,
    variant_order: Sequence[str],
    variant_order_index: int,
    result: StudyGraphExecution,
    native_build: HardwareSessionBuild,
    session_startup_time_s: float,
    startup_metadata: Mapping[str, Any],
    close_metadata: Mapping[str, Any],
    source_commit: object,
) -> JsonDict:
    variant = prepared["variants"][variant_id]
    graph = variant["graph"]
    summary = dict(result.summary)
    # A completed kernel result is not valid functionality evidence when the
    # persistent session cannot confirm that it released its allocated DPU.
    # Apply this before writing the per-repeat artifact so its status matches
    # the normalized record.
    if (
        summary.get("status") == "completed"
        and close_metadata.get("hardware_release_verified") is not True
    ):
        summary.update(
            {
                "status": "failed",
                "reason": "hardware_release_failed",
                "failure_stage": close_metadata.get("session_close_failure_stage")
                or "hardware_release_failed",
            }
        )
    record_status = str(summary.get("status", result.status))
    artifact_dir = (
        run_dir
        / "cases"
        / sanitize(str(case["case_id"]))
        / "study_executions"
        / f"repeat_{repeat_id:02d}"
        / f"{variant_id}__{quantization_mode}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    summary_path = artifact_dir / "runtime_summary.json"
    write_json(summary_path, summary)
    output_path: Path | None = None
    if result.output is not None:
        output_path = artifact_dir / "final_output.npy"
        np.save(output_path, np.asarray(result.output), allow_pickle=False)
    validation = (
        summary.get("policy_reference_validation")
        if isinstance(summary.get("policy_reference_validation"), Mapping)
        else {}
    )
    full_precision = (
        summary.get("full_precision_accuracy")
        if isinstance(summary.get("full_precision_accuracy"), Mapping)
        else {}
    )
    return {
        "schema_version": STUDY_BENCHMARK_SCHEMA_VERSION,
        "source_artifact": "upmem_hardware_taskgraph_study_summary.json",
        "run_id": run_dir.name,
        "suite_id": suite.suite["suite_id"],
        "case_id": case["case_id"],
        "workload_id": case["workload_id"],
        "repeat_id": repeat_id,
        "measurement_round": repeat_id,
        "variant_order": list(variant_order),
        "variant_order_index": variant_order_index,
        "benchmark_n_qubits": prepared["circuit"].n_qubits,
        "actual_n_qubits": prepared["circuit"].n_qubits,
        "actual_n_qubits_source": "circuit_spec",
        "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_one_dpu_path_quantization_study",
        "route_role_description": "one physical DPU, one persistent native host session per circuit case; same circuit/TN under two selected contraction paths and two numeric modes",
        "route_limitation_scope": "within-route one-DPU steady-state timing comparison only; no CPU/GPU speedup, energy, scheduler, planner-performance, or multi-DPU claim",
        "execution_model": "tensor_network",
        "execution_plan_kind": "taskgraph_serial_physical_one_dpu_persistent",
        "execution_plan_executed": record_status == "completed",
        "parallelism_mode": "sequential",
        "parallelism_evidence_type": "executed",
        "contraction_execution_target": "upmem",
        "execution_target": "upmem",
        "accelerator_kind": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu_persistent_taskgraph",
        "execution_backend": suite.profile.backend_id,
        "target_requested": "hardware",
        "target_observed": summary.get("target_observed"),
        "hardware_execution": summary.get("hardware_execution") is True,
        "hardware_kernel_executed": summary.get("hardware_kernel_executed") is True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_functionality_evidence": record_status == "completed",
        "hardware_timing_available": True,
        "hardware_speedup_applicable": False,
        "speedup_claim_allowed": False,
        "within_route_timing_comparison_allowed": True,
        "cross_backend_speedup_applicable": False,
        "hardware_profile_version": suite.profile.version,
        "session_protocol": suite.profile.session_protocol,
        "session_scope": "case_benchmark_block",
        "persistent_session_reused": True,
        "session_startup_time_s": session_startup_time_s,
        "requested_dpu_count": suite.profile.requested_dpu_count,
        "allocated_dpu_count": 1 if summary.get("hardware_execution") is True else None,
        "tasklets_per_dpu": suite.profile.tasklets_per_dpu,
        "hardware_allocation_verified": summary.get("hardware_execution") is True,
        "hardware_release_verified": close_metadata.get("hardware_release_verified"),
        "upmem_parallelism_mode": "sequential",
        "upmem_parallelism_evidence_type": "hardware_executed",
        "task_assignment_strategy": "sequential_single_dpu",
        "dpu_group_count": 1,
        "multi_dpu_execution": False,
        "path_variant_id": variant_id,
        "path_variant_label": variant["variant"].label,
        "planner": variant["variant"].planner,
        "planner_config_hash": canonical_hash(variant["variant"].planner),
        "planner_numeric_scope": "custom_upmem_v2 is a float32/split-complex modeled selector; per-task int8 is a separate runtime ablation",
        "quantization_mode": quantization_mode,
        "input_dtype_on_dpu": "float32" if quantization_mode == "none" else "int8",
        "accumulator_dtype_on_dpu": "float32"
        if quantization_mode == "none"
        else "int32",
        "complex_policy": suite.profile.complex_policy,
        "hardware_numeric_coverage": case.get("hardware_numeric_coverage"),
        "kernel_family": "generic_loop_fallback",
        "generic_kernel_strategy": "mram_resident_output_tiled_v1",
        "native_max_rank": suite.profile.max_rank,
        "native_max_tensor_elements": suite.profile.max_tensor_elements,
        "generic_output_tile_elements": suite.profile.output_tile_elements,
        "mram_resident_operands": True,
        "wram_output_tiled": True,
        "task_count": summary.get("task_count"),
        "source_task_count": summary.get("source_task_count"),
        "source_task_completion_count": summary.get("source_task_completion_count"),
        "native_task_execution_count": summary.get("native_task_execution_count"),
        "physical_dependency_chain_verified": summary.get(
            "physical_dependency_chain_verified"
        ),
        "validation_method": "physical_native_components_plus_post_timing_cpu_policy_reference",
        "validation_status": summary.get("validation_status"),
        "validation_max_abs_error": summary.get("validation_max_abs_error"),
        "max_abs_error": summary.get("max_abs_error"),
        "l2_error": summary.get("l2_error"),
        "full_precision_max_abs_error": full_precision.get("max_abs_error"),
        "exact_integer_match": summary.get("exact_integer_match"),
        "output_contract": "final_tensor",
        "output_contract_label": "physical_persistent_taskgraph_final_tensor",
        "output_contract_is_exact": False,
        "exact_output_comparable": False,
        "full_statevector_validation_available": False,
        "performance_tier": True,
        "status": record_status,
        "reason": summary.get("reason", result.reason),
        "failure_stage": summary.get("failure_stage"),
        "application_visible_h2d_bytes": summary.get("application_visible_h2d_bytes"),
        "application_visible_d2h_bytes": summary.get("application_visible_d2h_bytes"),
        "application_visible_transfer_bytes": summary.get(
            "application_visible_transfer_bytes"
        ),
        "actual_h2d_bytes": summary.get("actual_h2d_bytes"),
        "actual_d2h_bytes": summary.get("actual_d2h_bytes"),
        "actual_transfer_bytes": summary.get("actual_transfer_bytes"),
        "allocation_time_s": startup_metadata.get(
            "allocation_time_s", summary.get("allocation_time_s")
        ),
        "binary_load_time_s": startup_metadata.get(
            "binary_load_time_s", summary.get("binary_load_time_s")
        ),
        "release_time_s": close_metadata.get("release_time_s"),
        "h2d_time_s": summary.get("h2d_time_s"),
        "kernel_time_s": summary.get("kernel_time_s"),
        "d2h_time_s": summary.get("d2h_time_s"),
        "host_prepare_time_s": summary.get("host_prepare_time_s"),
        "host_reconstruction_time_s": summary.get("host_reconstruction_time_s"),
        "host_control_time_s": summary.get("host_control_time_s"),
        "steady_state_graph_execution_s": summary.get("steady_state_graph_execution_s"),
        "validation_time_s": summary.get("validation_time_s"),
        "total_build_time_s": native_build.build_time_s,
        "total_route_time_s": summary.get("total_route_time_s"),
        "timing_scope": summary.get("timing_scope"),
        "timing_is_bringup_only": False,
        "source_commit": source_commit,
        "hostname": socket.gethostname(),
        "sdk_metadata": native_build.sdk_tools,
        "native_source_tree_hash": native_build.source_tree_hash,
        "host_binary_hash": native_build.host_binary_hash,
        "dpu_binary_hash": native_build.dpu_binary_hash,
        "input_hash": _network_input_hash(prepared["network"]),
        "output_hash": _array_hash(result.output)
        if result.output is not None
        else None,
        "execution_bundle_artifact": str(variant["bundle_path"].relative_to(run_dir)),
        "circuit_semantics_hash": graph.circuit_semantics_hash,
        "tensor_network_hash": graph.tensor_network_hash,
        "contraction_plan_hash": graph.contraction_plan_hash,
        "contraction_path_structure_hash": summary.get(
            "contraction_path_structure_hash"
        ),
        "plan_reused": True,
        "planning_in_timed_region": False,
        "executor_config_hash": summary.get("executor_config_hash"),
        "task_metrics_artifact": str(summary_path.relative_to(run_dir)),
        "final_output_artifact": str(output_path.relative_to(run_dir))
        if output_path
        else None,
        "notes": {
            "policy_reference_validation": validation,
            "full_precision_accuracy": full_precision,
            "timing_boundary": "steady_state excludes native build, session allocation/load, session release, and CPU validation",
        },
    }


def _warmup_row(
    case: Mapping[str, Any],
    variant_id: str,
    mode: str,
    warmup_id: int,
    ordered: Sequence[tuple[str, str]],
    order_index: int,
    result: StudyGraphExecution,
) -> JsonDict:
    return {
        "case_id": case["case_id"],
        "path_variant_id": variant_id,
        "quantization_mode": mode,
        "warmup_id": warmup_id,
        "variant_order": [f"{variant}/{numeric}" for variant, numeric in ordered],
        "variant_order_index": order_index,
        "status": result.status,
        "reason": result.reason,
        "timing_excluded_from_statistics": True,
        "summary": result.summary,
    }


def _build_failure_record(
    suite: HardwareTaskGraphStudySuite,
    error: str,
    run_manifest: Mapping[str, Any],
    *,
    case: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "schema_version": STUDY_BENCHMARK_SCHEMA_VERSION,
        "suite_id": suite.suite["suite_id"],
        "case_id": case.get("case_id") if case else "native_build",
        "workload_id": case.get("workload_id") if case else "native_build",
        "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        "backend_id": suite.profile.backend_id,
        "backend_family": "upmem_sdk",
        "benchmark_role": "physical_one_dpu_path_quantization_study",
        "contraction_execution_target": "upmem",
        "upmem_execution_mode": "sdk_hardware_single_dpu_persistent_taskgraph",
        "target_requested": "hardware",
        "target_observed": None,
        "hardware_execution": False,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "hardware_timing_available": False,
        "hardware_speedup_applicable": False,
        "timing_is_bringup_only": False,
        "session_scope": "case_benchmark_block",
        "status": "failed",
        "reason": error,
        "failure_stage": _failure_stage_text(
            error, default="hardware_allocation_failed"
        ),
        "source_commit": run_manifest.get("benchmark_source_commit"),
    }


def _failure_execution(
    graph: TaskGraph,
    profile: HardwareTaskGraphStudyProfile,
    quantization_mode: str,
    execution_metadata: JsonDict,
    reason: str,
    started: float,
) -> StudyGraphExecution:
    summary: JsonDict = {
        "schema_version": STUDY_RUNTIME_SCHEMA_VERSION,
        "status": "failed",
        "reason": reason,
        "failure_stage": _failure_stage_text(
            reason, default="hardware_profile_violation"
        ),
        "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        "backend_id": profile.backend_id,
        "quantization_mode": quantization_mode,
        "hardware_execution": False,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "total_route_time_s": time.perf_counter() - started,
        "timing_scope": STUDY_TIMING_SCOPE,
        "timing_is_bringup_only": False,
        "hardware_timing_available": False,
        **execution_metadata,
    }
    return StudyGraphExecution("failed", reason, None, summary, ())


def _stopped_execution(
    graph: TaskGraph,
    profile: HardwareTaskGraphStudyProfile,
    quantization_mode: str,
    execution_metadata: JsonDict,
    metrics: list[JsonDict],
    totals: JsonDict,
    reason: str,
    started: float,
    steady_started: float,
) -> StudyGraphExecution:
    totals = dict(totals)
    totals["steady_state_graph_execution_s"] = time.perf_counter() - steady_started
    known = (
        totals["h2d_time_s"]
        + totals["kernel_time_s"]
        + totals["d2h_time_s"]
        + totals["host_prepare_time_s"]
        + totals["host_reconstruction_time_s"]
    )
    totals["host_control_time_s"] = max(
        0.0, totals["steady_state_graph_execution_s"] - known
    )
    summary: JsonDict = {
        "schema_version": STUDY_RUNTIME_SCHEMA_VERSION,
        "status": "failed",
        "reason": reason,
        "failure_stage": _failure_stage_text(reason, default="kernel_launch_failed"),
        "route_id": HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
        "backend_id": profile.backend_id,
        "quantization_mode": quantization_mode,
        "hardware_execution": bool(metrics),
        "hardware_kernel_executed": any(
            metric.get("hardware_kernel_executed") is True for metric in metrics
        ),
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "task_metrics": metrics,
        "total_route_time_s": time.perf_counter() - started,
        "timing_scope": STUDY_TIMING_SCOPE,
        "timing_is_bringup_only": False,
        "hardware_timing_available": False,
        **totals,
        **execution_metadata,
    }
    return StudyGraphExecution("failed", reason, None, summary, tuple(metrics))


def _empty_totals() -> JsonDict:
    return {
        "steady_state_graph_execution_s": 0.0,
        "h2d_time_s": 0.0,
        "kernel_time_s": 0.0,
        "d2h_time_s": 0.0,
        "host_prepare_time_s": 0.0,
        "host_reconstruction_time_s": 0.0,
        "host_control_time_s": 0.0,
        "application_visible_h2d_bytes": 0,
        "application_visible_d2h_bytes": 0,
        "allocation_time_s": None,
        "binary_load_time_s": None,
    }


def _response_timing(tasks: Sequence[Any]) -> JsonDict:
    fields = ("h2d_time_s", "kernel_time_s", "d2h_time_s")
    return {
        field: sum(
            float((item.get("timing") or {}).get(field) or 0.0)
            for item in tasks
            if isinstance(item, Mapping)
        )
        for field in fields
    }


def _failed_task_metric(
    task,
    task_index: int,
    reason: str,
    *,
    failure_stage: str | None = None,
    execution: HardwareSessionExecution | None = None,
) -> JsonDict:
    return {
        "task_id": task.id,
        "task_index": task_index,
        "status": "failed",
        "reason": reason,
        "failure_stage": failure_stage
        or _failure_stage_text(reason, default="kernel_launch_failed"),
        "input_tensor_ids": list(task.input_tensor_ids),
        "output_tensor_id": task.output_tensor_id,
        "hardware_kernel_executed": False,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "session_response_artifact": str(execution.response_path)
        if execution
        else None,
    }


def _spec_for(
    tensor_id: str,
    labels: Mapping[str, tuple[int, ...]],
    tensors: Mapping[str, np.ndarray],
) -> TensorSpec:
    array = np.asarray(tensors[tensor_id])
    return TensorSpec(
        tensor_id,
        tuple(labels[tensor_id]),
        tuple(int(dim) for dim in array.shape),
        "intermediate",
        str(array.dtype),
    )


def _split_components(
    left: np.ndarray, right: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "ar_br": (np.asarray(left.real), np.asarray(right.real)),
        "ai_bi": (np.asarray(left.imag), np.asarray(right.imag)),
        "ar_bi": (np.asarray(left.real), np.asarray(right.imag)),
        "ai_br": (np.asarray(left.imag), np.asarray(right.real)),
    }


def _real_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array.real if np.iscomplexobj(array) else array)


def _array_metrics(
    expected: np.ndarray, actual: np.ndarray, *, tolerance: float
) -> JsonDict:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        return {
            "passed": False,
            "max_abs_error": None,
            "l2_error": None,
            "tolerance": tolerance,
            "reason": "shape_mismatch",
        }
    difference = actual_array - expected_array
    return {
        "passed": bool(
            np.allclose(actual_array, expected_array, rtol=tolerance, atol=tolerance)
        ),
        "max_abs_error": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "l2_error": float(np.linalg.norm(difference.ravel())),
        "tolerance": tolerance,
    }


def _tolerance(mode: str) -> float:
    return 1.0e-5 if mode == "none" else 1.0e-3


def _rotated_variants(
    items: Sequence[tuple[str, str]], round_id: int
) -> tuple[tuple[str, str], ...]:
    offset = round_id % len(items)
    return tuple((*items[offset:], *items[:offset]))


def _execution_work_dir(
    build: HardwareSessionBuild,
    case_id: str,
    phase: str,
    iteration: int,
    variant_id: str,
    mode: str,
) -> Path:
    return (
        build.session_root
        / "study_runs"
        / sanitize(case_id)
        / phase
        / f"{iteration:02d}_{sanitize(variant_id)}_{sanitize(mode)}"
    )


def _native_build_metadata(build: HardwareSessionBuild, root: Path) -> JsonDict:
    return {
        "source_tree_hash": build.source_tree_hash,
        "host_binary_hash": build.host_binary_hash,
        "dpu_binary_hash": build.dpu_binary_hash,
        "build_time_s": build.build_time_s,
        "build_command": list(build.build_command),
        "sdk_tools": build.sdk_tools,
        "session_root": str(build.session_root.relative_to(root))
        if build.session_root.is_relative_to(root)
        else str(build.session_root),
    }


def _network_input_hash(network: TensorNetworkValue) -> str:
    digest = hashlib.sha256()
    for tensor in network.tensors:
        digest.update(tensor.spec.id.encode("utf-8"))
        digest.update(_array_hash(np.asarray(tensor.array)).encode("ascii"))
    return digest.hexdigest()


def _array_hash(array: np.ndarray | None) -> str:
    if array is None:
        return ""
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(int(dim) for dim in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _failure_stage_text(error: str, *, default: str) -> str:
    for stage in (
        "hardware_profile_violation",
        "sdk_discovery_failed",
        "native_build_failed",
        "hardware_allocation_failed",
        "binary_load_failed",
        "argument_transfer_failed",
        "operand_transfer_failed",
        "kernel_launch_failed",
        "kernel_timeout",
        "request_timeout",
        "result_transfer_failed",
        "output_manifest_failed",
        "output_validation_failed",
        "hardware_release_failed",
    ):
        if stage in error:
            return stage
    return default


def _number_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None
