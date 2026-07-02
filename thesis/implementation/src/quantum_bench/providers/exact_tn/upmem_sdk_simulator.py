from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import (
    BenchmarkContext,
    CircuitSpec,
    ContractionTask,
    ExecutionProfile,
    JsonDict,
    PathSummary,
    RouteCapabilities,
    RouteEstimate,
    RouteIdentity,
    RouteOutput,
    RouteProbe,
    RouteResult,
    TaskGraph,
    TensorNetworkSpec,
    TensorSpec,
    TensorValue,
    to_jsonable,
)
from quantum_bench.targets.upmem.evidence import (
    UPMEM_ACCELERATOR_KIND,
    UPMEM_EXECUTION_BACKEND_SDK,
)
from quantum_bench.targets.upmem.taskgraph_runtime import (
    CONTRACTION_EXECUTION_TARGET,
    UPMEM_EXECUTION_MODE,
    execute_upmem_taskgraph_runtime,
)
from quantum_bench.targets.upmem.runtime_checks import (
    strict_upmem_runtime_assertions,
    summary_dpu_invocations,
    upmem_sdk_simulator_preflight_payload,
)
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import TensorNetworkValue


class UpmemTnSdkSimulatorQuantizedRoute:
    name = "upmem_tn_sdk_simulator_quantized"
    backend_family = UPMEM_EXECUTION_BACKEND_SDK
    identity = RouteIdentity(
        route_id=name,
        display_name="UPMEM SDK simulator tensor network (quantized)",
        role="strict_upmem_sdk_simulator_candidate",
        simulation_method="exact_tensor_network",
        kernel_family="upmem_taskgraph_quantized",
        hardware_target="upmem",
        execution_mode="sdk_simulator",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        return RouteProbe(
            self.name,
            True,
            metadata={
                "route_registered": True,
                "benchmark_rows_require_preflight": True,
                "quantization_mode": "per_task_input_quantize",
                "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
                "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            },
        )

    def capabilities(self) -> RouteCapabilities:
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("quest_compatible",),
            can_return_output=True,
            can_measure_energy=False,
            metadata={
                "route_registered": True,
                "external_execution_required": True,
                "preflight_required": True,
                "quantized_execution": True,
                "quantization_mode": "per_task_input_quantize",
                "whole_network_quantized_at_initialization": False,
                "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
                "upmem_execution_mode": UPMEM_EXECUTION_MODE,
                "execution_backend": UPMEM_EXECUTION_BACKEND_SDK,
                "hardware_execution": False,
                "hardware_timing_available": False,
                "hardware_speedup_applicable": False,
                "native_sdk_control_path": True,
                "simplepim_api_used": False,
                "cpu_fallback_used": False,
            },
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        options = _route_options(context)
        if not bool(options.get("execute_external", False)):
            self._write_preflight(context, upmem_sdk_simulator_preflight_payload("blocked", "upmem_sdk_simulator_execute_external_required"))
            return False, "upmem_sdk_simulator_execute_external_required"
        if not graph.tasks:
            self._write_preflight(context, upmem_sdk_simulator_preflight_payload("blocked", "empty_task_graph_not_supported"))
            return False, "empty_task_graph_not_supported"
        if bool(options.get("skip_preflight", False)):
            self._write_preflight(context, upmem_sdk_simulator_preflight_payload("skipped", "preflight_skipped_by_route_option"))
            return True, None

        try:
            result = execute_upmem_taskgraph_runtime(
                graph=_preflight_graph(),
                network=_preflight_network(),
                case_id="upmem_sdk_simulator_preflight",
                policy="generic-only",
                quantization_mode="per_task_input_quantize",
                bridge_root=self._preflight_dir(context) / "bridge",
                execute_external=True,
                reference_output=np.asarray(2.0 + 0.0j, dtype=np.complex128),
            )
        except Exception as exc:  # pragma: no cover - defensive blocker path
            payload = upmem_sdk_simulator_preflight_payload("blocked", f"upmem_sdk_simulator_preflight_exception:{exc}")
            self._write_preflight(context, payload)
            return False, str(payload["reason"])
        payload = upmem_sdk_simulator_preflight_payload(
            "passed" if result.status == "completed" and summary_dpu_invocations(result.summary) > 0 else "blocked",
            None if result.status == "completed" and summary_dpu_invocations(result.summary) > 0 else result.reason or result.status,
            summary=result.summary,
        )
        self._write_preflight(context, payload)
        if payload["status"] != "passed":
            return False, str(payload["reason"] or "upmem_sdk_simulator_preflight_failed")
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.max_intermediate_bytes,
            metadata={
                "execution_model": "tensor_network",
                "backend_family": self.backend_family,
                "quantized_execution": True,
                "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
                "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            },
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict[str, Any]:
        return {
            "graph": graph,
            "network": network,
            "policy": str(_route_options(context).get("policy", "dense-then-generic")),
            "quantization_mode": str(_route_options(context).get("quantization_mode", "per_task_input_quantize")),
        }

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        policy = str(payload["policy"])
        quantization_mode = str(payload["quantization_mode"])
        total_start = time.perf_counter()
        reference_start = time.perf_counter()
        reference_output, reference_metadata = execute_task_sequence_np_einsum(graph, network)
        reference_time_s = time.perf_counter() - reference_start
        runtime = execute_upmem_taskgraph_runtime(
            graph=graph,
            network=network,
            case_id=str(context.case.get("case_id", graph.network.circuit.name)),
            policy=policy,
            quantization_mode=quantization_mode,  # type: ignore[arg-type]
            bridge_root=_route_repeat_dir(context) / "upmem_taskgraph_bridge",
            execute_external=bool(_route_options(context).get("execute_external", False)),
            reference_output=reference_output,
        )
        assertions = strict_upmem_runtime_assertions(runtime.summary)
        metadata = _route_metadata(runtime.summary, assertions, reference_time_s, reference_metadata)
        if runtime.status != "completed":
            return _failed(self.name, self.backend_family, runtime.reason or runtime.status, metadata=metadata, total_s=time.perf_counter() - total_start)
        if assertions["status"] != "passed":
            return _failed(self.name, self.backend_family, assertions["reason"], metadata=metadata, total_s=time.perf_counter() - total_start)
        if runtime.output is None:
            return _failed(self.name, self.backend_family, "upmem_runtime_output_missing", metadata=metadata, total_s=time.perf_counter() - total_start)

        output = np.asarray(runtime.output, dtype=np.complex128)
        total_s = time.perf_counter() - total_start
        return RouteResult(
            route=self.name,
            backend_family=self.backend_family,
            status="passed",
            output=RouteOutput(
                contract=self.identity.output_contract,
                array=output,
                shape=tuple(int(dim) for dim in output.shape),
                dtype=str(output.dtype),
                metadata={"output_labels": runtime.output_labels},
            ),
            profile=ExecutionProfile(
                prepare_s=float(reference_time_s),
                h2d_s=float(runtime.summary.get("total_bridge_time_s", 0.0) or 0.0),
                kernel_s=float(runtime.summary.get("total_kernel_time_s", 0.0) or 0.0),
                total_s=float(total_s),
                validation_s=float((runtime.summary.get("final_validation") or {}).get("validation_time_s", 0.0) or 0.0),
            ),
            energy_joules=None,
            energy_source="unavailable",
            metadata=metadata,
        )

    def _preflight_dir(self, context: BenchmarkContext) -> Path:
        return _route_repeat_dir(context) / "preflight"

    def _write_preflight(self, context: BenchmarkContext, payload: JsonDict) -> None:
        write_json(self._preflight_dir(context) / "upmem_sdk_simulator_preflight.json", payload)


def _route_options(context: BenchmarkContext) -> JsonDict:
    return dict(context.route_config.get("options") or {})


def _route_repeat_dir(context: BenchmarkContext) -> Path:
    case_id = str(context.case.get("case_id", "case"))
    repeat_label = f"repeat_{context.repeat_id}" if context.repeat_id >= 0 else f"warmup_{abs(context.repeat_id) - 1}"
    return context.run_dir / "cases" / _sanitize(case_id) / "routes" / UpmemTnSdkSimulatorQuantizedRoute.name / repeat_label


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _preflight_graph() -> TaskGraph:
    circuit = CircuitSpec(name="upmem_sdk_simulator_preflight", n_qubits=0, operations=(), source={"kind": "preflight"})
    left_spec = TensorSpec("left", (0,), (1,), "dense", dtype="float64")
    right_spec = TensorSpec("right", (0,), (1,), "dense", dtype="float64")
    network_spec = TensorNetworkSpec(circuit, (left_spec, right_spec), (), "a,a->")
    task = ContractionTask(
        id="preflight_task",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="a,a->",
        input_shapes=((1,), (1,)),
        output_shape=(),
        left_labels=(0,),
        right_labels=(0,),
        contracted_labels=(0,),
        output_labels=(),
        gemm_m=1,
        gemm_k=1,
        gemm_n=1,
        structure="dense",
        estimated_flops=2,
        estimated_bytes=16,
    )
    summary = PathSummary(
        planner="preflight",
        optimize="greedy",
        path_length=1,
        largest_intermediate=1,
        naive_flops=None,
        optimized_flops=None,
        text="upmem sdk simulator preflight",
        planner_engine="preflight",
        planner_id="preflight",
        planner_kind="preflight",
        optimize_mode="greedy",
        task_count=1,
        total_estimated_flops=2,
        peak_intermediate_bytes=8,
        max_intermediate_bytes=8,
    )
    return TaskGraph(network_spec, (task,), ((0, 1),), summary, planning_time_s=0.0)


def _preflight_network() -> TensorNetworkValue:
    graph = _preflight_graph()
    specs = graph.network.tensors
    return TensorNetworkValue(
        graph.network,
        [
            TensorValue(specs[0], np.asarray([1.0], dtype=np.float64)),
            TensorValue(specs[1], np.asarray([2.0], dtype=np.float64)),
        ],
    )


def _route_metadata(summary: JsonDict, assertions: JsonDict, reference_time_s: float, reference_metadata: JsonDict) -> JsonDict:
    final_validation = dict(summary.get("final_validation") or {})
    return to_jsonable(
        {
            "execution_backend": UPMEM_EXECUTION_BACKEND_SDK,
            "backend_family": UPMEM_EXECUTION_BACKEND_SDK,
            "quantized_execution": True,
            "quantization_mode": summary.get("quantization_mode", "per_task_input_quantize"),
            "policy": summary.get("policy"),
            "whole_network_quantized_at_initialization": False,
            "contraction_execution_target": CONTRACTION_EXECUTION_TARGET,
            "upmem_execution_mode": UPMEM_EXECUTION_MODE,
            "hardware_execution": False,
            "hardware_benchmark_result": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "cpu_fallback_task_count": assertions.get("cpu_fallback_task_count", 0),
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "task_count": assertions.get("task_count", 0),
            "upmem_task_count": assertions.get("upmem_task_count", 0),
            "dpu_program_invocations": assertions.get("dpu_program_invocations", 0),
            "upmem_program_executed": assertions.get("upmem_program_executed", False),
            "strict_runtime_assertions": assertions,
            "runtime_summary": summary,
            "kernel_family_counts": summary.get("kernel_family_counts", {}),
            "backend_counts": summary.get("backend_counts", {}),
            "final_validation": final_validation,
            "cpu_reference_time_s": float(reference_time_s),
            "cpu_reference_used_for_validation_only": True,
            "cpu_reference_artifacts_feed_runtime_tensors": False,
            "reference_metadata": reference_metadata,
            "accelerator_kind": UPMEM_ACCELERATOR_KIND,
        }
    )


def _failed(route: str, backend_family: str, error: str, *, metadata: JsonDict, total_s: float) -> RouteResult:
    return RouteResult(
        route=route,
        backend_family=backend_family,
        status="failed",
        output=RouteOutput(contract="final_tensor"),
        profile=ExecutionProfile(total_s=float(total_s)),
        energy_joules=None,
        energy_source="unavailable",
        error=error,
        metadata=metadata,
    )
