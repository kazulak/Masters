from __future__ import annotations

import time
import warnings
from importlib import metadata as importlib_metadata
from typing import Any

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
        tensor_network, output_inds, lowering_s = _build_quimb_network(qtn, graph, network)

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


class QuimbTnSlicedExactRoute:
    name = "quimb_tn_sliced_exact"
    backend_family = "quimb"
    identity = RouteIdentity(
        route_id=name,
        display_name="Quimb sliced exact tensor network",
        role="explicit_slicing_evidence",
        simulation_method="exact_tensor_network",
        kernel_family="external_tn_contraction",
        hardware_target="cpu",
        execution_mode="in_process_external_library_sliced",
        output_contract="final_tensor",
        validation_mode="compare_output",
    )

    def probe(self) -> RouteProbe:
        versions = _dependency_versions()
        if versions["quimb"] is None:
            return RouteProbe(
                self.name,
                False,
                "quimb is not installed; install the external TN dependencies before enabling quimb_tn_sliced_exact",
                metadata=versions,
            )
        if versions["cotengra"] is None:
            return RouteProbe(
                self.name,
                False,
                "cotengra is not installed; explicit slicing evidence requires cotengra",
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
                "slicing_enabled": True,
                "slicing_backend": "cotengra",
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
                "slicing_enabled": True,
            },
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        options = dict(context.route_config.get("options") or {})
        strategy = str(options.get("slicing_strategy") or "target_slices")
        target_slices = int(options.get("target_slices", 2) or 2)
        if strategy != "target_slices":
            raise ValueError(f"unsupported cotengra slicing_strategy: {strategy}")
        if target_slices < 2:
            raise ValueError("quimb_tn_sliced_exact requires target_slices >= 2")
        return {
            "graph": graph,
            "network": network,
            "methods": str(options.get("methods") or options.get("optimize") or "greedy"),
            "max_repeats": int(options.get("max_repeats", 1) or 1),
            "target_slices": target_slices,
            "slicing_strategy": strategy,
            "require_slicing": bool(options.get("require_slicing", True)),
            "prepare_s": 0.0,
        }

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        payload = dict(prepared)  # type: ignore[arg-type]
        graph: TaskGraph = payload["graph"]
        network: TensorNetworkValue = payload["network"]
        methods = str(payload.get("methods") or "greedy")
        max_repeats = int(payload.get("max_repeats", 1) or 1)
        target_slices = int(payload.get("target_slices", 2) or 2)
        slicing_strategy = str(payload.get("slicing_strategy") or "target_slices")
        require_slicing = bool(payload.get("require_slicing", True))
        versions = _dependency_versions()
        try:
            import cotengra as ctg
            import quimb.tensor as qtn
        except ImportError as exc:  # pragma: no cover - covered by probe/unit monkeypatches
            return _failed(self.name, self.backend_family, str(exc), metadata=versions)

        total_start = time.perf_counter()
        try:
            tensor_network, output_inds, lowering_s = _build_quimb_network(qtn, graph, network)
            planning_start = time.perf_counter()
            optimizer = _cotengra_optimizer(
                ctg,
                methods=methods,
                max_repeats=max_repeats,
                slicing_opts={"target_slices": target_slices},
            )
            tree = tensor_network.contract(output_inds=output_inds, optimize=optimizer, get="tree")
            planning_s = time.perf_counter() - planning_start
            tree_metadata = _sliced_tree_metadata(tree, require_slicing=require_slicing)
            comparison_metadata = _unsliced_tree_comparison(
                ctg,
                tensor_network,
                output_inds,
                methods=methods,
                max_repeats=max_repeats,
                slicing_total_flops=tree_metadata.get("slicing_total_flops"),
                slicing_max_intermediate_size=tree_metadata.get("slicing_max_intermediate_size"),
            )
        except Exception as exc:
            return _failed(
                self.name,
                self.backend_family,
                f"quimb/cotengra slicing plan failed: {exc}",
                total_s=time.perf_counter() - total_start,
                metadata={
                    **versions,
                    "slicing_enabled": True,
                "slicing_backend": "cotengra",
                "slicing_strategy": slicing_strategy,
                "target_slices": target_slices,
                "slicing_reconstruction_status": "not_executed",
                "slice_parallel_execution": False,
            },
            )

        energy_start = read_rapl_uj()
        kernel_start = time.perf_counter()
        try:
            contracted = tensor_network.contract(output_inds=output_inds, optimize=tree)
        except Exception as exc:
            return _failed(
                self.name,
                self.backend_family,
                f"quimb sliced contraction failed: {exc}",
                total_s=time.perf_counter() - total_start,
                metadata={
                    **versions,
                    "slicing_enabled": True,
                    "slicing_backend": "cotengra",
                    "slicing_strategy": slicing_strategy,
                    "target_slices": target_slices,
                    **tree_metadata,
                    **comparison_metadata,
                    "slicing_reconstruction_status": "failed",
                    "slice_parallel_execution": False,
                },
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
            "execution_engine": "quimb_cotengra_sliced_contraction_tree",
            "dependency_versions": versions,
            "external_library": True,
            "accelerator_kind": "none",
            "methods": methods,
            "max_repeats": max_repeats,
            "tensor_count": len(network.tensors),
            "output_inds": output_inds,
            "actual_output_inds": actual_inds,
            "final_transpose_applied": transposed,
            "planning_time_s": planning_s,
            "planning_time_included_in_kernel_s": False,
            "lowering_time_s": lowering_s,
            "tn_task_count": len(graph.tasks),
            "tn_max_intermediate_bytes": graph.path_summary.max_intermediate_bytes,
            "tn_estimated_flops": graph.path_summary.total_estimated_flops,
            "tn_estimated_bytes": sum(task.estimated_bytes for task in graph.tasks),
            "parallelism_mode": "slicing",
            "parallelism_evidence_type": "executed",
            "execution_plan_kind": "cotengra_sliced_contraction_tree",
            "execution_plan_executed": True,
            "slicing_enabled": True,
            "slicing_backend": "cotengra",
            "slicing_strategy": slicing_strategy,
            "target_slices": target_slices,
            "slicing_reconstruction_status": "completed",
            "slice_parallel_execution": False,
            "intra_contraction_parallelism_source": "cotengra_slicing",
            "frontier_scheduler_enabled": False,
            "modeled_parallelism_available": False,
            "slice_worker_count": 1,
            **tree_metadata,
            **comparison_metadata,
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
                planning_s=planning_s,
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


def _build_quimb_network(qtn: Any, graph: TaskGraph, network: TensorNetworkValue) -> tuple[Any, tuple[str, ...], float]:
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
    return qtn.TensorNetwork(tensors), output_inds, time.perf_counter() - lowering_start


def _cotengra_optimizer(ctg: Any, *, methods: str, max_repeats: int, slicing_opts: dict[str, int] | None = None) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Couldn't find `optuna`, `cmaes`, or `nevergrad`.*")
        return ctg.HyperOptimizer(
            methods=methods,
            max_repeats=max(1, int(max_repeats)),
            parallel=False,
            progbar=False,
            slicing_opts=slicing_opts,
            on_trial_error="raise",
        )


def _sliced_tree_metadata(tree: Any, *, require_slicing: bool) -> dict[str, Any]:
    if not hasattr(tree, "nslices") or not hasattr(tree, "sliced_inds"):
        raise ValueError("slicing_api_unavailable: contraction tree does not expose nslices/sliced_inds")
    slice_count = int(getattr(tree, "nslices") or 0)
    sliced_inds = getattr(tree, "sliced_inds") or {}
    if require_slicing and slice_count <= 1:
        raise ValueError("slicing_not_applied: cotengra returned an unsliced contraction tree")
    if require_slicing and not sliced_inds:
        raise ValueError("slicing_not_applied: cotengra returned no sliced indices")
    sliced_index_sizes = {
        str(index): int(getattr(info, "size", info if isinstance(info, int) else 0))
        for index, info in sorted(dict(sliced_inds).items(), key=lambda item: str(item[0]))
    }
    return {
        "slice_count": slice_count,
        "sliced_indices": tuple(sliced_index_sizes),
        "sliced_index_sizes": sliced_index_sizes,
        "slicing_total_flops": _tree_number(tree, "total_flops"),
        "slicing_max_intermediate_size": _tree_number(tree, "max_size"),
    }


def _unsliced_tree_comparison(
    ctg: Any,
    tensor_network: Any,
    output_inds: tuple[str, ...],
    *,
    methods: str,
    max_repeats: int,
    slicing_total_flops: int | float | None,
    slicing_max_intermediate_size: int | float | None,
) -> dict[str, Any]:
    try:
        unsliced_tree = tensor_network.contract(
            output_inds=output_inds,
            optimize=_cotengra_optimizer(ctg, methods=methods, max_repeats=max_repeats),
            get="tree",
        )
    except Exception as exc:  # pragma: no cover - defensive metadata-only path
        return {"unsliced_tree_cost_status": f"unavailable:{type(exc).__name__}"}
    unsliced_total_flops = _tree_number(unsliced_tree, "total_flops")
    unsliced_max_intermediate_size = _tree_number(unsliced_tree, "max_size")
    flop_ratio = _safe_ratio(slicing_total_flops, unsliced_total_flops)
    memory_ratio = _safe_ratio(slicing_max_intermediate_size, unsliced_max_intermediate_size)
    return {
        "unsliced_tree_cost_status": "available",
        "unsliced_total_flops": unsliced_total_flops,
        "slicing_flop_ratio": flop_ratio,
        "slicing_flop_metric_source": "cotengra_contraction_tree_total_flops",
        "slicing_flop_change_kind": _ratio_change_kind(flop_ratio),
        "slicing_flop_inflation_factor": flop_ratio if flop_ratio is not None and flop_ratio >= 1.0 else None,
        "slicing_flop_inflation": flop_ratio if flop_ratio is not None and flop_ratio >= 1.0 else None,
        "unsliced_max_intermediate_size": unsliced_max_intermediate_size,
        "slicing_memory_ratio": memory_ratio,
        "slicing_memory_reduction_factor": (1.0 / memory_ratio) if memory_ratio is not None and 0.0 < memory_ratio < 1.0 else None,
    }


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return float(numerator) / float(denominator)


def _ratio_change_kind(ratio: float | None) -> str:
    if ratio is None:
        return "unavailable"
    if ratio < 1.0:
        return "reduction"
    if ratio > 1.0:
        return "inflation"
    return "equal"


def _tree_number(tree: Any, method_name: str) -> int | None:
    method = getattr(tree, method_name, None)
    if not callable(method):
        return None
    value = method()
    if value is None:
        return None
    return int(value)


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
