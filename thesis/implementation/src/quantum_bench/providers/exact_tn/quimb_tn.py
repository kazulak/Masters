from __future__ import annotations

import time
from importlib import metadata as importlib_metadata

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
from quantum_bench.tn.network import TensorNetworkValue


class QuimbTnExactRoute:
    name = "quimb_tn_exact"
    backend_family = "quimb"
    identity = RouteIdentity(
        route_id=name,
        display_name="Quimb exact tensor network",
        role="serious_external_tn_baseline",
        simulation_method="exact_tensor_network",
        kernel_family="external_tn_contraction",
        hardware_target="cpu",
        execution_mode="in_process_external_library",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        versions = _dependency_versions()
        if versions["quimb"] is None:
            return RouteProbe(
                self.name,
                False,
                "quimb is not installed; install the external TN dependencies before enabling quimb_tn_exact",
                metadata=versions,
            )
        return RouteProbe(self.name, True, metadata=versions)

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("builtin", "qasm_file", "quest_compatible"),
            can_return_output=True,
            can_measure_energy=True,
            metadata={
                "available": probe.available,
                "reason": probe.reason,
                "external_library": True,
                "exact_output_comparable": True,
                "accelerator_kind": "none",
                **probe.metadata,
            },
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            sum(task.estimated_bytes for task in graph.tasks),
            graph.path_summary.largest_intermediate * 16 if graph.path_summary.largest_intermediate else None,
            metadata={
                "execution_model": "tensor_network",
                "backend_family": self.backend_family,
                "external_library": True,
            },
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        options = dict(context.route_config.get("options") or {})
        return {
            "graph": graph,
            "network": network,
            "optimize": str(options.get("optimize", "greedy")),
            "prepare_s": 0.0,
        }

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        optimize = str(payload.get("optimize", "greedy"))
        versions = _dependency_versions()
        try:
            import quimb.tensor as qtn
        except ImportError as exc:  # pragma: no cover - covered by probe/unit monkeypatches
            return _failed(self.name, self.backend_family, str(exc), metadata=versions)

        total_start = time.perf_counter()
        lowering_start = time.perf_counter()
        tensors = [
            qtn.Tensor(
                np.asarray(tensor.array, dtype=np.complex128),
                inds=tuple(_index_name(label) for label in tensor.spec.labels),
                tags=(tensor.spec.id,),
            )
            for tensor in network.tensors
        ]
        output_inds = tuple(_index_name(label) for label in graph.network.output_labels)
        tensor_network = qtn.TensorNetwork(tensors)
        lowering_s = time.perf_counter() - lowering_start

        energy_start = read_rapl_uj()
        kernel_start = time.perf_counter()
        try:
            contracted = tensor_network.contract(output_inds=output_inds, optimize=optimize)
        except Exception as exc:  # pragma: no cover - defensive runtime failure path
            return _failed(
                self.name,
                self.backend_family,
                f"quimb contraction failed: {exc}",
                total_s=time.perf_counter() - total_start,
                metadata={**versions, "optimize": optimize},
            )
        kernel_s = time.perf_counter() - kernel_start
        energy_end = read_rapl_uj()
        total_s = time.perf_counter() - total_start

        array, actual_inds, transposed = _contracted_array(contracted, output_inds)
        energy_joules = None
        energy_source = "unavailable"
        if energy_start is not None and energy_end is not None and energy_end >= energy_start:
            energy_joules = (energy_end - energy_start) / 1_000_000.0
            energy_source = "rapl_measured" if energy_joules > 0 else "rapl_zero_or_too_short"
        metadata = {
            "execution_engine": "quimb_tensor_network_contract",
            "dependency_versions": versions,
            "external_library": True,
            "accelerator_kind": "none",
            "optimize": optimize,
            "tensor_count": len(network.tensors),
            "output_inds": output_inds,
            "actual_output_inds": actual_inds,
            "final_transpose_applied": transposed,
            "planning_time_s": None,
            "planning_time_included_in_kernel_s": True,
            "lowering_time_s": lowering_s,
            "tn_task_count": len(graph.tasks),
            "tn_max_intermediate_bytes": graph.path_summary.max_intermediate_bytes,
            "tn_estimated_flops": graph.path_summary.total_estimated_flops,
            "tn_estimated_bytes": sum(task.estimated_bytes for task in graph.tasks),
        }
        return RouteResult(
            route=self.name,
            backend_family=self.backend_family,
            status="passed",
            output=RouteOutput(
                contract=self.identity.output_contract,
                array=array,
                shape=tuple(int(dim) for dim in array.shape),
                dtype=str(array.dtype),
                metadata={"output_inds": output_inds},
            ),
            profile=ExecutionProfile(
                lowering_s=lowering_s,
                kernel_s=kernel_s,
                total_s=total_s,
            ),
            energy_joules=energy_joules,
            energy_source=energy_source,
            metadata=metadata,
        )


def _dependency_versions() -> dict[str, str | None]:
    return {
        "quimb": _module_version("quimb"),
        "cotengra": _module_version("cotengra"),
        "opt_einsum": _module_version("opt_einsum"),
    }


def _module_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _index_name(label: int) -> str:
    return f"i{int(label)}"


def _contracted_array(contracted: object, output_inds: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...], bool]:
    if hasattr(contracted, "inds") and hasattr(contracted, "data"):
        actual_inds = tuple(str(item) for item in contracted.inds)
        array = np.asarray(contracted.data, dtype=np.complex128)
        if actual_inds == output_inds:
            return array, actual_inds, False
        if len(actual_inds) != len(output_inds) or set(actual_inds) != set(output_inds):
            raise ValueError(f"Quimb output indices {actual_inds} do not match requested {output_inds}")
        axes = tuple(actual_inds.index(index) for index in output_inds)
        return np.asarray(np.transpose(array, axes), dtype=np.complex128), actual_inds, True
    return np.asarray(contracted, dtype=np.complex128), (), False


def _failed(
    route: str,
    backend_family: str,
    error: str,
    total_s: float = 0.0,
    metadata: dict | None = None,
) -> RouteResult:
    return RouteResult(
        route=route,
        backend_family=backend_family,
        status="failed",
        output=RouteOutput(contract="final_tensor"),
        profile=ExecutionProfile(total_s=total_s),
        energy_joules=None,
        energy_source="unavailable",
        error=error,
        metadata=metadata or {},
    )
