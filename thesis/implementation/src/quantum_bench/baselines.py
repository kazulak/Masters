"""Direct Quimb, cotengra, and QuEST simulation baselines."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from quantum_bench.circuits import quest_compatible_circuit
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
_QUEST_ALIASES = {
    "qrng": "QRNG",
    "bb84": "BB84",
    "bb_n": "BB84",
    "bv": "BV",
    "bernstein_vazirani": "BV",
    "edc": "EDC",
    "dense_coding": "EDC",
    "hs": "HS",
    "hidden_shift": "HS",
    "xor": "XOR",
    "parity": "XOR",
}
_QUEST_STATE_SCHEMA = "quest_state_dump_v1"
_QUEST_BASIS_ORDER = "quest_little_endian_integer_index"


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


def run_quest_cpu(
    job: SimulationJob,
    *,
    runner: Path | None = None,
    timeout_s: float = 120.0,
    max_output_amplitudes: int = 1 << 12,
) -> ExecutionSample:
    """Run a structurally validated canonical job through the QuEST CPU binary."""

    runner_path, runner_sha256 = _preflight_quest_runner(runner)
    started = time.perf_counter()
    canonical, algorithm, repeat_layers, input_qubits, allocated_qubits = (
        _validate_quest_cpu_request(
            job,
            timeout_s=timeout_s,
            max_output_amplitudes=max_output_amplitudes,
        )
    )
    command = [
        str(runner_path),
        "--algo",
        algorithm,
        "--json",
        "--max-output-amplitudes",
        str(max_output_amplitudes),
        "--dump-state-json",
        "<state_dump.json>",
        "--repeat-layers",
        str(repeat_layers),
    ]
    if algorithm == "HS":
        command.extend(["--logical-qubits", str(input_qubits)])
    else:
        command.extend(["--qubits", str(input_qubits)])

    backend_facts: dict[str, JsonValue] = {
        "backend_id": "quest_cpu_full_state_v1",
        "backend_family": "quest",
        "execution_class": "external_process_cpu",
        "hardware_execution": True,
        "physical_upmem_execution": False,
        "target_observed": "cpu",
        "runner": str(runner_path),
        "runner_sha256": runner_sha256,
        "command": tuple(command),
        "query": job.query,
        "basis_order": _QUEST_BASIS_ORDER,
        "source_family": algorithm,
        "native_timing_scope": "quest_compute_only",
        "native_compute_energy_j": None,
        "energy_source": "unavailable",
    }

    try:
        with tempfile.TemporaryDirectory(prefix="qbench-quest-") as temp_dir:
            dump_path = Path(temp_dir) / "state_dump.json"
            command_with_dump = [
                str(dump_path) if item == "<state_dump.json>" else item
                for item in command
            ]
            result = subprocess.run(
                command_with_dump,
                cwd=runner_path.parent.parent,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            backend_facts["returncode"] = result.returncode
            backend_facts["native_stderr"] = result.stderr.strip().replace(
                str(dump_path), "<state_dump.json>"
            )
            if result.returncode != 0:
                raise _QuestRuntimeError(
                    "process", "QuEST process returned nonzero status"
                )
            decode_started = time.perf_counter()
            try:
                payload = json.loads(result.stdout)
            except (json.JSONDecodeError, TypeError) as exc:
                raise _QuestRuntimeError(
                    "stdout", "QuEST stdout is not valid JSON"
                ) from exc
            native = _validate_quest_stdout(
                payload,
                algorithm=algorithm,
                input_qubits=input_qubits,
                allocated_qubits=allocated_qubits,
                repeat_layers=repeat_layers,
                expected_one_qubit=_quest_gate_count(canonical, 1),
                expected_two_qubit=_quest_gate_count(canonical, 2),
            )
            backend_facts.update(native)
            output = _load_quest_state_dump(
                dump_path,
                allocated_qubits=allocated_qubits,
                quest_version=native["quest_version"],
            )
            decode_s = time.perf_counter() - decode_started
    except subprocess.TimeoutExpired as exc:
        backend_facts["timeout_s"] = timeout_s
        raise ExecutionFailed(
            "process", "QuEST process timed out", backend_facts
        ) from exc
    except OSError as exc:
        raise ExecutionFailed("process", str(exc), backend_facts) from exc
    except _QuestRuntimeError as exc:
        raise ExecutionFailed(exc.stage, exc.reason, backend_facts) from exc
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise ExecutionFailed("decode", str(exc), backend_facts) from exc

    backend_facts["state_dump_requested"] = True
    backend_facts["native_state_dump_time_s"] = native["native_state_dump_time_s"]
    backend_facts["quest_version"] = native["quest_version"]
    return ExecutionSample(
        output=output,
        measurement=Measurement(
            scope_id=_SCOPE,
            total_wall_s=time.perf_counter() - started,
            kernel_s=native["native_compute_time_s"],
            decode_s=decode_s,
        ),
        backend_facts=backend_facts,
        numeric_facts={
            "output_dtype": "complex128",
            "statevector_basis_order": _QUEST_BASIS_ORDER,
            "reference_dtype": "complex128",
        },
    )


class _QuestRuntimeError(RuntimeError):
    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(reason)


def _default_quest_runner() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "native"
        / "quest_cpu"
        / "bin"
        / "quest_runner"
    )


def _preflight_quest_runner(runner: Path | None) -> tuple[Path, str]:
    try:
        runner_path = (
            (_default_quest_runner() if runner is None else Path(runner))
            .expanduser()
            .resolve()
        )
    except (OSError, RuntimeError, TypeError) as exc:
        raise UnsupportedExecution(
            "preflight", f"runner path is invalid: {exc}", "quest_runner"
        ) from exc
    if not runner_path.is_file() or not os.access(runner_path, os.X_OK):
        raise UnsupportedExecution(
            "preflight", "runner is not an executable regular file", "quest_runner"
        )
    try:
        runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise UnsupportedExecution(
            "preflight", f"runner cannot be read: {exc}", "quest_runner"
        ) from exc
    return runner_path, runner_sha256


def _validate_quest_cpu_request(
    job: SimulationJob,
    *,
    timeout_s: float,
    max_output_amplitudes: int,
) -> tuple[Any, str, int, int, int]:
    if not isinstance(job, SimulationJob):
        raise UnsupportedExecution(
            "preflight", "job is not a SimulationJob", "simulation_job"
        )
    source = job.circuit.source
    if source.get("kind") != "quest_compatible":
        raise UnsupportedExecution(
            "preflight", "source kind is not quest_compatible", "circuit_source"
        )
    if source.get("deterministic_unitary") is not True:
        raise UnsupportedExecution(
            "preflight", "circuit is not deterministic_unitary", "determinism"
        )
    if job.query != "pre_measurement_statevector":
        raise UnsupportedExecution("preflight", "unsupported query", "simulation_query")
    if job.parameters:
        raise UnsupportedExecution(
            "preflight", "parameters are unsupported", "simulation_parameters"
        )
    if job.seed is not None:
        raise UnsupportedExecution(
            "preflight", "seeded jobs are unsupported", "simulation_seed"
        )
    source_name = source.get("name")
    algorithm = (
        _QUEST_ALIASES.get(str(source_name).lower())
        if isinstance(source_name, str)
        else None
    )
    if algorithm is None:
        raise UnsupportedExecution(
            "preflight", "unsupported canonical circuit name", "circuit_family"
        )
    depth = source.get("depth")
    if isinstance(depth, bool) or (depth is not None and depth != 1):
        raise UnsupportedExecution(
            "preflight", "depth must be absent or 1", "circuit_depth"
        )
    repeat_layers = source.get("repeat_layers", 1)
    if (
        isinstance(repeat_layers, bool)
        or not isinstance(repeat_layers, int)
        or repeat_layers < 1
    ):
        raise UnsupportedExecution(
            "preflight", "repeat_layers must be positive", "repeat_layers"
        )
    try:
        canonical = quest_compatible_circuit(str(source_name), dict(source))
    except (TypeError, ValueError) as exc:
        raise UnsupportedExecution("preflight", str(exc), "canonical_circuit") from exc
    if (
        job.circuit.n_qubits != canonical.n_qubits
        or job.circuit.operations != canonical.operations
    ):
        raise UnsupportedExecution(
            "preflight",
            "circuit differs from canonical operations",
            "canonical_circuit",
        )
    input_qubits = canonical.n_qubits // 2 if algorithm == "HS" else canonical.n_qubits
    allocated_qubits = canonical.n_qubits
    if algorithm == "HS" and input_qubits * 2 != allocated_qubits:
        raise UnsupportedExecution(
            "preflight", "HS qubit layout is invalid", "qubit_layout"
        )
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise UnsupportedExecution(
            "preflight", "timeout_s must be finite and positive", "timeout"
        )
    if (
        isinstance(max_output_amplitudes, bool)
        or not isinstance(max_output_amplitudes, int)
        or max_output_amplitudes <= 0
        or max_output_amplitudes < (1 << allocated_qubits)
    ):
        raise UnsupportedExecution(
            "preflight", "max_output_amplitudes is below the state size", "output_cap"
        )
    return canonical, algorithm, repeat_layers, input_qubits, allocated_qubits


def _quest_gate_count(circuit: Any, arity: int) -> int:
    return sum(len(operation.wires) == arity for operation in circuit.operations)


def _validate_quest_stdout(
    payload: Any,
    *,
    algorithm: str,
    input_qubits: int,
    allocated_qubits: int,
    repeat_layers: int,
    expected_one_qubit: int,
    expected_two_qubit: int,
) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise _QuestRuntimeError("stdout", "QuEST stdout JSON is not an object")
    if payload.get("status") != "ok":
        raise _QuestRuntimeError("status", "QuEST status is not ok")
    for key, expected in (
        ("algo", algorithm),
        ("input_qubits", input_qubits),
        ("allocated_qubits", allocated_qubits),
        # Canonical algorithms do not consume --depth; the native runner reports 0.
        ("depth", 0),
        ("repeat_layers", repeat_layers),
        ("state_dump_requested", True),
        ("one_qubit_gates", expected_one_qubit),
        ("two_qubit_gates", expected_two_qubit),
    ):
        actual = payload.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise _QuestRuntimeError(
                "stdout", f"QuEST field {key} does not match expected value"
            )
    quest_version = payload.get("quest_version")
    if not isinstance(quest_version, str) or not quest_version:
        raise _QuestRuntimeError("stdout", "QuEST version is missing")
    native_time = payload.get("time_s")
    dump_time = payload.get("state_dump_time_s")
    if not _finite_nonnegative(native_time) or not _finite_nonnegative(dump_time):
        raise _QuestRuntimeError("stdout", "QuEST timing fields are invalid")
    energy = payload.get("energy_joules")
    energy_source = payload.get("energy_source")
    if energy is None:
        if energy_source != "unavailable":
            raise _QuestRuntimeError("stdout", "QuEST energy metadata pair is invalid")
    elif not _finite_nonnegative(energy) or energy_source != "rapl_measured":
        raise _QuestRuntimeError("stdout", "QuEST energy metadata pair is invalid")
    return {
        "quest_version": quest_version,
        "native_depth": 0,
        "native_compute_time_s": float(native_time),
        "native_state_dump_time_s": float(dump_time),
        "native_compute_energy_j": None if energy is None else float(energy),
        "energy_source": energy_source,
    }


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _load_quest_state_dump(
    path: Path, *, allocated_qubits: int, quest_version: str
) -> np.ndarray:
    if not path.is_file():
        raise ValueError("state dump is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_count = 1 << allocated_qubits
    if payload.get("schema_version") != _QUEST_STATE_SCHEMA:
        raise ValueError("state dump schema is invalid")
    if payload.get("basis_order") != _QUEST_BASIS_ORDER:
        raise ValueError("state dump basis order is invalid")
    dump_version = payload.get("quest_version")
    if not isinstance(dump_version, str) or not dump_version.strip():
        raise ValueError("state dump QuEST version is invalid")
    if dump_version != quest_version:
        raise ValueError("state dump QuEST version does not match stdout")
    if (
        payload.get("allocated_qubits") != allocated_qubits
        or payload.get("amplitude_count") != expected_count
    ):
        raise ValueError("state dump dimensions are invalid")
    real = np.asarray(payload.get("real"), dtype=np.float64)
    imag = np.asarray(payload.get("imag"), dtype=np.float64)
    if (
        real.ndim != 1
        or imag.ndim != 1
        or real.size != expected_count
        or imag.size != expected_count
    ):
        raise ValueError("state dump arrays are invalid")
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("state dump contains non-finite values")
    output = np.asarray(real + 1j * imag, dtype=np.complex128)
    output.setflags(write=False)
    return output


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


__all__ = ["run_quimb", "run_cotengra", "run_quest_cpu"]
