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
            supported_workload_families=("bell_2q", "ghz_4q", "ghz_chain"),
            can_return_output=False,
            can_measure_energy=False,
            metadata={"available": probe.available, "reason": probe.reason, **probe.metadata},
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        supported = {"bell_2q", "ghz_4q", "ghz_chain"}
        circuit_name = str(context.case.get("circuit", {}).get("name", ""))
        if circuit_name not in supported:
            return False, f"legacy raw dense route currently supports only {sorted(supported)}"
        for task in graph.tasks:
            if task.gemm_k > 256:
                return False, f"task {task.id} requires K tiling (k={task.gemm_k}); legacy route supports k<=256"
        return False, "native raw UPMEM route is probed but not yet ported into canonical runtime"

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            64 * 1024,
            ("64 KiB WRAM tile guard", "legacy dense int8 GEMM route"),
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
            "native raw UPMEM route is not executable in this canonical slice",
        )
