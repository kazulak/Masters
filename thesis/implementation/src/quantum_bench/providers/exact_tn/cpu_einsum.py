from __future__ import annotations

import time

import numpy as np

from quantum_bench.core.records import (
    BenchmarkContext,
    ExecutionProfile,
    RouteCapabilities,
    RouteEstimate,
    RouteIdentity,
    RouteOutput,
    RouteProbe,
    RouteResult,
    TaskGraph,
)
from quantum_bench.environment import read_rapl_uj
from quantum_bench.tn.execution_bundle import execution_identity_metadata, executor_config_hash, with_execution_identity
from quantum_bench.tn.execution import execute_task_sequence_np_einsum
from quantum_bench.tn.network import TensorNetworkValue


class CpuTnEinsumExactRoute:
    name = "cpu_tn_einsum_exact"
    backend_family = "cpu"
    identity = RouteIdentity(
        route_id=name,
        display_name="CPU exact tensor network (NumPy einsum)",
        role="internal_debug_baseline",
        simulation_method="exact_tensor_network",
        kernel_family="einsum_contraction",
        hardware_target="cpu",
        execution_mode="in_process_python",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        return RouteProbe(self.name, True, metadata={"numpy": np.__version__})

    def capabilities(self) -> RouteCapabilities:
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file"),
            can_return_output=True,
            can_measure_energy=True,
            metadata={"numpy": np.__version__},
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.largest_intermediate * 16 if graph.path_summary.largest_intermediate else None,
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": with_execution_identity(graph), "network": network, "prepare_s": 0.0}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        energy_start = read_rapl_uj()
        start = time.perf_counter()
        output, metadata = execute_task_sequence_np_einsum(graph, network)
        kernel_s = time.perf_counter() - start
        energy_end = read_rapl_uj()
        energy_joules = None
        energy_source = "unavailable"
        if energy_start is not None and energy_end is not None and energy_end >= energy_start:
            energy_joules = (energy_end - energy_start) / 1_000_000.0
            energy_source = "rapl_measured" if energy_joules > 0 else "rapl_zero_or_too_short"
        array = np.asarray(output, dtype=np.complex128)
        return RouteResult(
            route=self.name,
            backend_family=self.backend_family,
            status="passed",
            output=RouteOutput(
                contract=self.identity.output_contract,
                array=array,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
            ),
            profile=ExecutionProfile(prepare_s=float(payload.get("prepare_s", 0.0)), kernel_s=kernel_s, total_s=kernel_s),
            energy_joules=energy_joules,
            energy_source=energy_source,
            metadata={
                **metadata,
                **execution_identity_metadata(graph, plan_reused=True),
                "executor_config_hash": executor_config_hash(self.name, dict(context.route_config.get("options") or {})),
            },
        )


def _execute_task_sequence(graph: TaskGraph, network: TensorNetworkValue) -> tuple[np.ndarray, dict]:
    return execute_task_sequence_np_einsum(graph, network)
