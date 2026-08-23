"""Direct Quimb and cotengra tensor-network simulation baselines."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import random
import time
from typing import Any

import numpy as np

from quantum_bench.lowering import lower_tensor_network
from quantum_bench.model import SimulationJob
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    JsonValue,
    Measurement,
    UnsupportedExecution,
)


_SCOPE = "simulation_end_to_end_v1"
_QUIMB_OPTIMIZERS = frozenset(
    {
        "greedy",
        "optimal",
    }
)
_COTENGRA_METHODS = frozenset({"greedy", "labels"})


def run_quimb(job: SimulationJob, *, optimize: str = "greedy") -> ExecutionSample:
    """Run a canonical job using Quimb's requested contraction optimizer."""

    _validate_options(optimize=optimize, methods=None, max_repeats=None)
    return _run(job, backend="quimb", optimize=optimize, methods=None, max_repeats=None)


def run_cotengra(
    job: SimulationJob, *, methods: str = "greedy", max_repeats: int = 1
) -> ExecutionSample:
    """Run a canonical job using a deterministic cotengra HyperOptimizer."""

    _validate_options(optimize=None, methods=methods, max_repeats=max_repeats)
    return _run(
        job,
        backend="cotengra",
        optimize=None,
        methods=methods,
        max_repeats=max_repeats,
    )


def _run(
    job: SimulationJob,
    *,
    backend: str,
    optimize: str | None,
    methods: str | None,
    max_repeats: int | None,
) -> ExecutionSample:
    _validate_job(job)
    started = time.perf_counter()
    backend_facts = dict(
        _backend_facts(
            backend, optimize=optimize, methods=methods, max_repeats=max_repeats
        )
    )

    lowering_started = time.perf_counter()
    try:
        network, inputs = lower_tensor_network(job)
    except (ValueError, TypeError) as exc:
        raise UnsupportedExecution(
            "preflight", f"unsupported SimulationJob: {exc}", "simulation_job"
        ) from exc
    except Exception as exc:
        raise ExecutionFailed("lowering", str(exc), backend_facts) from exc
    try:
        import quimb.tensor as qtn

        tensor_network, output_inds = _build_quimb_network(qtn, network, inputs)
        lowering_s = time.perf_counter() - lowering_started
    except Exception as exc:
        raise ExecutionFailed("lowering", str(exc), backend_facts) from exc

    planning_started = time.perf_counter()
    try:
        if backend == "quimb":
            tree = tensor_network.contract(
                output_inds=output_inds, optimize=optimize, get="tree"
            )
        else:
            import cotengra as ctg

            optimizer = _make_cotengra_optimizer(
                ctg, methods=methods, max_repeats=max_repeats
            )
            tree = _plan_cotengra_tree(
                tensor_network,
                output_inds=output_inds,
                optimizer=optimizer,
            )
        path_pairs = _tree_path_pairs(tree)
        backend_facts["contraction_path_fingerprint"] = _tree_path_fingerprint(
            path_pairs
        )
        backend_facts["contraction_path_length"] = len(path_pairs)
        planning_s = time.perf_counter() - planning_started
    except Exception as exc:
        raise ExecutionFailed("planning", str(exc), backend_facts) from exc

    kernel_started = time.perf_counter()
    try:
        contracted = tensor_network.contract(output_inds=output_inds, optimize=tree)
        kernel_s = time.perf_counter() - kernel_started
    except Exception as exc:
        raise ExecutionFailed("kernel", str(exc), backend_facts) from exc

    decode_started = time.perf_counter()
    try:
        final_tensor = _contracted_array(contracted, output_inds)
        output = _tensor_to_quest_statevector(final_tensor)
        if not np.isfinite(output).all():
            raise ValueError("decoded output contains non-finite values")
        decode_s = time.perf_counter() - decode_started
    except Exception as exc:
        raise ExecutionFailed("decode", str(exc), backend_facts) from exc

    return ExecutionSample(
        output=output,
        measurement=Measurement(
            scope_id=_SCOPE,
            total_wall_s=time.perf_counter() - started,
            lowering_s=lowering_s,
            planning_s=planning_s,
            kernel_s=kernel_s,
            decode_s=decode_s,
        ),
        backend_facts=backend_facts,
        numeric_facts={
            "output_dtype": str(output.dtype),
            "statevector_basis_order": "quest_little_endian_integer_index",
        },
    )


def _validate_options(
    *, optimize: str | None, methods: str | None, max_repeats: int | None
) -> None:
    if optimize is not None and (
        not isinstance(optimize, str) or optimize not in _QUIMB_OPTIMIZERS
    ):
        raise UnsupportedExecution(
            "preflight", f"unsupported Quimb optimizer: {optimize!r}", "optimize"
        )
    if methods is not None and (
        not isinstance(methods, str) or methods not in _COTENGRA_METHODS
    ):
        raise UnsupportedExecution(
            "preflight", f"unsupported cotengra methods: {methods!r}", "methods"
        )
    if max_repeats is not None and (
        isinstance(max_repeats, bool)
        or not isinstance(max_repeats, int)
        or max_repeats < 1
    ):
        raise UnsupportedExecution(
            "preflight",
            f"max_repeats must be a positive integer, got {max_repeats!r}",
            "max_repeats",
        )


def _validate_job(job: SimulationJob) -> None:
    if not isinstance(job, SimulationJob):
        raise UnsupportedExecution(
            "preflight", "job is not a canonical SimulationJob", "simulation_job"
        )
    if job.query != "pre_measurement_statevector":
        raise UnsupportedExecution(
            "preflight", f"unsupported query: {job.query!r}", "simulation_query"
        )
    if job.parameters:
        raise UnsupportedExecution(
            "preflight",
            "parameterized simulation jobs are unsupported",
            "simulation_parameters",
        )
    if job.seed is not None:
        raise UnsupportedExecution(
            "preflight",
            "seeded simulation jobs are unsupported",
            "deterministic_simulation",
        )
    n_qubits = job.circuit.n_qubits
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise UnsupportedExecution(
            "preflight", "the circuit must contain at least one qubit", "qubit_count"
        )
    for operation in job.circuit.operations:
        if any(
            isinstance(wire, bool)
            or not isinstance(wire, int)
            or wire < 0
            or wire >= n_qubits
            for wire in operation.wires
        ):
            raise UnsupportedExecution(
                "preflight", "circuit operation has an invalid wire", "circuit_wires"
            )


def _backend_facts(
    backend: str,
    *,
    optimize: str | None,
    methods: str | None,
    max_repeats: int | None,
) -> Mapping[str, JsonValue]:
    if backend == "quimb":
        return {
            "backend_id": "quimb_tn_v1",
            "backend_family": "quimb",
            "hardware_execution": False,
            "optimizer": optimize,
        }
    return {
        "backend_id": "cotengra_tn_v1",
        "backend_family": "cotengra",
        "hardware_execution": False,
        "optimizer": "cotengra.HyperOptimizer",
        "methods": methods,
        "max_repeats": max_repeats,
        "deterministic_planning_seed": 0,
        "deterministic_planning_rngs": (
            "python_random",
            "numpy_legacy",
            "cotengra_hyperoptimizer",
        ),
    }


def _build_quimb_network(
    qtn: Any, network: Any, inputs: Mapping[str, np.ndarray]
) -> tuple[Any, tuple[str, ...]]:
    tensors = []
    for spec in network.tensors:
        data = np.array(inputs[spec.id], dtype=np.complex128, copy=True, order="C")
        inds = tuple(_label_name(label) for label in spec.labels)
        tensors.append(qtn.Tensor(data=data, inds=inds, tags=(f"qbench_{spec.id}",)))
    output_inds = tuple(_label_name(label) for label in network.output_labels)
    return qtn.TensorNetwork(tensors), output_inds


def _make_cotengra_optimizer(
    ctg: Any, *, methods: str | None, max_repeats: int | None
) -> Any:
    return ctg.HyperOptimizer(
        methods=methods,
        max_repeats=max_repeats,
        parallel=False,
        progbar=False,
        optlib="random",
        seed=0,
        on_trial_error="raise",
    )


def _plan_cotengra_tree(
    tensor_network: Any,
    *,
    output_inds: tuple[str, ...],
    optimizer: Any,
) -> Any:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(0)
    np.random.seed(0)
    try:
        return tensor_network.contract(
            output_inds=output_inds,
            optimize=optimizer,
            get="tree",
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _tree_path_pairs(tree: Any) -> tuple[tuple[int, int], ...]:
    pairs = tuple(tuple(int(index) for index in pair) for pair in tree.get_path())
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("contraction path contains a non-pair entry")
    return pairs


def _tree_path_fingerprint(path_pairs: tuple[tuple[int, int], ...]) -> str:
    canonical = json.dumps(path_pairs, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _label_name(label: int) -> str:
    return f"qbench_wire_{int(label)}"


def _contracted_array(contracted: Any, output_inds: tuple[str, ...]) -> np.ndarray:
    if hasattr(contracted, "inds") and hasattr(contracted, "data"):
        actual_inds = tuple(str(index) for index in contracted.inds)
        array = np.asarray(contracted.data, dtype=np.complex128)
        if actual_inds == output_inds:
            return array
        if len(actual_inds) != len(output_inds) or set(actual_inds) != set(output_inds):
            raise ValueError(
                f"Quimb output indices {actual_inds} do not match requested {output_inds}"
            )
        axes = tuple(actual_inds.index(index) for index in output_inds)
        return np.asarray(np.transpose(array, axes), dtype=np.complex128)
    return np.asarray(contracted, dtype=np.complex128)


def _tensor_to_quest_statevector(tensor: np.ndarray) -> np.ndarray:
    array = np.asarray(tensor, dtype=np.complex128)
    if array.ndim == 0:
        return np.array(array.reshape(1), dtype=np.complex128, copy=True, order="C")
    if any(dimension != 2 for dimension in array.shape):
        raise ValueError(
            f"statevector conversion requires qubit dimensions of 2, got shape {array.shape}"
        )
    output = np.empty(1 << array.ndim, dtype=np.complex128)
    for index in range(output.size):
        bits = tuple((index >> wire) & 1 for wire in range(array.ndim))
        output[index] = array[bits]
    return output


__all__ = ["run_quimb", "run_cotengra"]
