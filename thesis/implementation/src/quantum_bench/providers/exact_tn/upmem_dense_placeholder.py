from __future__ import annotations

import os
import shutil

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
from quantum_bench.targets.upmem import estimate_dense_task_graph
from quantum_bench.tn.network import TensorNetworkValue


class UpmemDenseInt8PlaceholderRoute:
    name = "upmem_dense_int8_placeholder"
    backend_family = "upmem"
    identity = RouteIdentity(
        route_id=name,
        display_name="UPMEM dense int8 GEMM placeholder",
        role="candidate",
        simulation_method="exact_tensor_network",
        kernel_family="dense_gemm",
        hardware_target="upmem_dpu",
        execution_mode="native_binary",
        output_contract="none",
        validation_mode="skip_with_reason",
    )

    def probe(self) -> RouteProbe:
        has_sdk = bool(os.environ.get("UPMEM_HOME")) or shutil.which("dpu-upmem-dpurte-clang") is not None
        reason = None if has_sdk else "UPMEM SDK not detected; set UPMEM_HOME or expose dpu-upmem-dpurte-clang"
        return RouteProbe(self.name, has_sdk, reason, {"UPMEM_HOME": os.environ.get("UPMEM_HOME")})

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file"),
            can_return_output=False,
            can_measure_energy=False,
            metadata={
                "available": probe.available,
                "reason": probe.reason,
                "target_layer": "quantum_bench.targets.upmem",
                **probe.metadata,
            },
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        schedule = estimate_dense_task_graph(graph)
        reject_reason = schedule.first_reject_reason()
        if reject_reason:
            return False, f"{reject_reason}; WRAM tiling is not implemented yet"
        return False, "UPMEM dense target estimate is available; native dense execution is not implemented yet"

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        schedule = estimate_dense_task_graph(graph)
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            schedule.total_host_to_dpu_bytes + schedule.total_dpu_to_host_bytes,
            schedule.max_working_set_bytes,
            (*schedule.notes(), "native dense execution not implemented"),
            tile_shape={
                "model": "untiled_dense_gemm",
                "max_working_set_bytes": schedule.max_working_set_bytes,
                "wram_bytes": schedule.hardware.wram_bytes,
            },
            wram_fit=schedule.all_tasks_fit_without_tiling,
            metadata=schedule.metadata(),
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> object:
        return {"graph": graph, "network": network}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        return RouteResult(
            self.name,
            self.backend_family,
            "skipped",
            RouteOutput(contract=self.identity.output_contract),
            ExecutionProfile(),
            None,
            "unavailable",
            "native UPMEM dense execution is not implemented yet",
        )
