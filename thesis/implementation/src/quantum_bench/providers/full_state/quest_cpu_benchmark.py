from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

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
from quantum_bench.tn.network import TensorNetworkValue


QUEST_EXACT_STATE_SCHEMA_VERSION = "quest_state_dump_v1"
DEFAULT_MAX_OUTPUT_QUBITS = 12
DEFAULT_MAX_OUTPUT_AMPLITUDES = 1 << DEFAULT_MAX_OUTPUT_QUBITS
DEFAULT_MAX_OUTPUT_BYTES = DEFAULT_MAX_OUTPUT_AMPLITUDES * np.dtype(np.complex128).itemsize
QUEST_COMPARABLE_ALGOS = {"BB84", "BV", "EDC", "HS", "QRNG", "XOR"}


class QuestCpuFullStateBenchmarkRoute:
    name = "quest_cpu_full_state_benchmark"
    backend_family = "cpu_full_state"
    identity = RouteIdentity(
        route_id=name,
        display_name="QuEST CPU full-state benchmark",
        role="baseline",
        simulation_method="full_state_vector",
        kernel_family="external_full_state",
        hardware_target="cpu",
        execution_mode="external_process",
        output_contract="metrics_only",
        validation_mode="benchmark_only",
    )

    def __init__(self, root_dir: Path):
        self.root = root_dir / "native" / "quest_cpu"
        self.runner = self.root / "bin" / "quest_runner"

    def probe(self) -> RouteProbe:
        return _probe_quest_runner(self.name, self.root, self.runner)

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=("BB84", "BV", "EDC", "HS", "QRNG", "XOR", "RANDOM"),
            can_return_output=False,
            can_measure_energy=True,
            metadata={"available": probe.available, "reason": probe.reason, **probe.metadata},
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        algo = self._algo(context)
        if algo is None:
            return False, "QuEST adapter supports only BB84, BV, EDC, HS, QRNG, XOR, RANDOM benchmark families"
        return True, None

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        allocated = graph.network.circuit.n_qubits
        bytes_estimate = (2**allocated) * 16
        return RouteEstimate(self.name, sum(task.estimated_flops for task in graph.tasks), bytes_estimate, bytes_estimate)

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": graph, "network": network}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        graph: TaskGraph = dict(prepared)["graph"]  # type: ignore[arg-type]
        algo = self._algo(context)
        if algo is None:
            return self._failed("unsupported QuEST algorithm")
        cmd = [str(self.runner), "--algo", algo, "--json"]
        if algo == "HS" and "logical_qubits" in context.case.get("circuit", {}):
            cmd.extend(["--logical-qubits", str(context.case["circuit"]["logical_qubits"])])
        else:
            cmd.extend(["--qubits", str(graph.network.circuit.n_qubits)])
        depth = context.case.get("circuit", {}).get("depth")
        if depth is not None:
            cmd.extend(["--depth", str(depth)])
        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, check=False)
        total_s = time.perf_counter() - start
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return self._failed((result.stderr or result.stdout or "QuEST did not emit JSON").strip(), total_s)
        quest_status = str(payload.get("status") or "")
        status = "passed" if result.returncode == 0 and quest_status in {"ok", "passed"} else "failed"
        return RouteResult(
            self.name,
            self.backend_family,
            status,
            RouteOutput(contract=self.identity.output_contract, metadata={"quest": payload}),
            ExecutionProfile(kernel_s=float(payload.get("time_s") or 0.0), total_s=total_s),
            payload.get("energy_joules"),
            str(payload.get("energy_source") or "unavailable"),
            None if status == "passed" else str(payload.get("error") or result.stderr or "QuEST failed"),
            {"quest": payload, "command": cmd},
        )

    def _algo(self, context: BenchmarkContext) -> str | None:
        name = str(context.case.get("circuit", {}).get("name", "")).upper()
        return name if name in {"BB84", "BV", "EDC", "HS", "QRNG", "XOR", "RANDOM"} else None

    def _failed(self, error: str, total_s: float = 0.0) -> RouteResult:
        return RouteResult(
            self.name,
            self.backend_family,
            "failed",
            RouteOutput(contract=self.identity.output_contract),
            ExecutionProfile(total_s=total_s),
            None,
            "unavailable",
            error,
        )


class QuestCpuFullStateExactRoute:
    name = "quest_cpu_full_state_exact"
    backend_family = "quest"
    identity = RouteIdentity(
        route_id=name,
        display_name="QuEST CPU full-state exact output",
        role="baseline",
        simulation_method="full_state_vector",
        kernel_family="full_state_vector",
        hardware_target="cpu",
        execution_mode="external_process",
        output_contract="statevector",
        validation_mode="compare_statevector",
    )

    def __init__(self, root_dir: Path):
        self.root = root_dir / "native" / "quest_cpu"
        self.runner = self.root / "bin" / "quest_runner"

    def probe(self) -> RouteProbe:
        return _probe_quest_runner(self.name, self.root, self.runner)

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=tuple(sorted(QUEST_COMPARABLE_ALGOS)),
            can_return_output=True,
            can_measure_energy=True,
            metadata={
                "available": probe.available,
                "reason": probe.reason,
                "deterministic_statevector_only": True,
                "default_max_output_qubits": DEFAULT_MAX_OUTPUT_QUBITS,
                **probe.metadata,
            },
        )

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        probe = self.probe()
        if not probe.available:
            return False, probe.reason
        circuit = context.case.get("circuit", {})
        if circuit.get("kind") != "quest_compatible":
            return False, "not_quest_compatible_circuit"
        if not circuit.get("deterministic_unitary", True):
            return False, "non_deterministic_statevector_circuit"
        algo = _quest_algo_from_case(context)
        if algo is None or algo not in QUEST_COMPARABLE_ALGOS:
            return False, "unsupported_quest_comparable_algorithm"
        return _check_output_caps(graph.network.circuit.n_qubits, context.route_config.get("options") or {})

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        bytes_estimate = (2**graph.network.circuit.n_qubits) * np.dtype(np.complex128).itemsize
        return RouteEstimate(
            self.name,
            sum(task.estimated_flops for task in graph.tasks),
            int(bytes_estimate),
            int(bytes_estimate),
            metadata={"execution_model": "full_state", "output_kind": "statevector"},
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": graph, "network": network}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        graph: TaskGraph = dict(prepared)["graph"]  # type: ignore[arg-type]
        options = context.route_config.get("options") or {}
        cap_ok, cap_reason = _check_output_caps(graph.network.circuit.n_qubits, options)
        if not cap_ok:
            return _quest_exact_failed(self.name, self.backend_family, cap_reason or "state_output_cap_exceeded")
        algo = _quest_algo_from_case(context)
        if algo is None:
            return _quest_exact_failed(self.name, self.backend_family, "unsupported_quest_comparable_algorithm")

        case_id = str(context.case.get("case_id", graph.network.circuit.name))
        rel_dump = Path("cases") / _sanitize(case_id) / "quest_full_state" / f"repeat_{context.repeat_id}" / "state_dump.json"
        dump_path = context.run_dir / rel_dump
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        max_amplitudes = int(options.get("max_output_amplitudes", DEFAULT_MAX_OUTPUT_AMPLITUDES))
        cmd = [str(self.runner), "--algo", algo, "--json", "--dump-state-json", str(dump_path), "--max-output-amplitudes", str(max_amplitudes)]
        if algo == "HS" and "logical_qubits" in context.case.get("circuit", {}):
            cmd.extend(["--logical-qubits", str(context.case["circuit"]["logical_qubits"])])
        else:
            cmd.extend(["--qubits", str(graph.network.circuit.n_qubits)])
        depth = context.case.get("circuit", {}).get("depth")
        if depth is not None:
            cmd.extend(["--depth", str(depth)])

        start = time.perf_counter()
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, check=False)
        total_s = time.perf_counter() - start
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                (result.stderr or result.stdout or "QuEST did not emit JSON").strip(),
                total_s,
                metadata={"command": cmd},
            )
        quest_status = str(payload.get("status") or "")
        if result.returncode != 0 or quest_status not in {"ok", "passed"}:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                str(payload.get("error") or result.stderr or "QuEST failed"),
                total_s,
                metadata={"quest": payload, "command": cmd},
            )
        try:
            state, dump_payload = _load_state_dump(dump_path, graph.network.circuit.n_qubits)
        except ValueError as exc:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                str(exc),
                total_s,
                metadata={"quest": payload, "command": cmd, "state_dump_artifact": rel_dump.as_posix()},
            )

        return RouteResult(
            self.name,
            self.backend_family,
            "passed",
            RouteOutput(
                contract=self.identity.output_contract,
                array=state,
                artifact_path=rel_dump,
                shape=tuple(int(dim) for dim in state.shape),
                dtype=str(state.dtype),
                metadata={
                    "state_dump_artifact": rel_dump.as_posix(),
                    "basis_order": dump_payload.get("basis_order"),
                    "output_kind": "statevector",
                    "execution_model": "full_state",
                },
            ),
            ExecutionProfile(kernel_s=float(payload.get("time_s") or 0.0), total_s=total_s),
            payload.get("energy_joules"),
            str(payload.get("energy_source") or "unavailable"),
            None,
            {"quest": payload, "state_dump": _state_dump_metadata(dump_payload), "command": cmd},
        )


def _quest_algo_from_case(context: BenchmarkContext) -> str | None:
    name = str(context.case.get("circuit", {}).get("name", "")).upper()
    if name in {"BB_N"}:
        name = "BB84"
    if name in {"DENSE_CODING"}:
        name = "EDC"
    if name in {"PARITY"}:
        name = "XOR"
    if name in {"BERNSTEIN_VAZIRANI"}:
        name = "BV"
    if name in {"HIDDEN_SHIFT"}:
        name = "HS"
    return name if name in QUEST_COMPARABLE_ALGOS or name == "RANDOM" else None


def _check_output_caps(n_qubits: int, options: dict) -> tuple[bool, str | None]:
    max_qubits = int(options.get("max_output_qubits", DEFAULT_MAX_OUTPUT_QUBITS))
    max_amplitudes = int(options.get("max_output_amplitudes", DEFAULT_MAX_OUTPUT_AMPLITUDES))
    max_bytes = int(options.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES))
    amplitude_count = 1 << int(n_qubits)
    output_bytes = amplitude_count * np.dtype(np.complex128).itemsize
    if n_qubits > max_qubits:
        return False, "state_output_qubit_cap_exceeded"
    if amplitude_count > max_amplitudes:
        return False, "state_output_amplitude_cap_exceeded"
    if output_bytes > max_bytes:
        return False, "state_output_byte_cap_exceeded"
    return True, None


def _load_state_dump(path: Path, n_qubits: int) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise ValueError("quest_state_dump_missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != QUEST_EXACT_STATE_SCHEMA_VERSION:
        raise ValueError("quest_state_dump_schema_invalid")
    expected = 1 << int(n_qubits)
    if int(payload.get("amplitude_count", -1)) != expected:
        raise ValueError("quest_state_dump_amplitude_count_mismatch")
    real = np.asarray(payload.get("real", []), dtype=np.float64)
    imag = np.asarray(payload.get("imag", []), dtype=np.float64)
    if real.shape != (expected,) or imag.shape != (expected,):
        raise ValueError("quest_state_dump_shape_invalid")
    return (real + 1j * imag).astype(np.complex128, copy=False), payload


def _state_dump_metadata(payload: dict) -> dict:
    return {
        "schema_version": payload.get("schema_version"),
        "basis_order": payload.get("basis_order"),
        "allocated_qubits": payload.get("allocated_qubits"),
        "amplitude_count": payload.get("amplitude_count"),
        "quest_version": payload.get("quest_version"),
    }


def _probe_quest_runner(route: str, root: Path, runner: Path) -> RouteProbe:
    metadata = {"runner": str(runner), "quest_root": str(root)}
    if not root.exists():
        return RouteProbe(route, False, f"QuEST implementation not found at {root}", metadata=metadata)
    if not runner.exists():
        return RouteProbe(route, False, f"QuEST runner not built at {runner}; run make in {root}", metadata=metadata)
    try:
        result = subprocess.run([str(runner), "--help"], cwd=root, capture_output=True, text=True, check=False, timeout=5.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RouteProbe(route, False, f"QuEST runner failed to start: {exc}", metadata=metadata)
    if result.returncode != 0:
        reason = (result.stderr.strip() or result.stdout.strip() or "QuEST runner failed to start").splitlines()[0]
        return RouteProbe(route, False, f"QuEST runner failed to start: {reason}", metadata=metadata)
    return RouteProbe(route, True, metadata=metadata)


def _quest_exact_failed(
    route: str,
    backend_family: str,
    error: str,
    total_s: float = 0.0,
    metadata: dict | None = None,
) -> RouteResult:
    return RouteResult(
        route,
        backend_family,
        "failed",
        RouteOutput(contract="statevector"),
        ExecutionProfile(total_s=total_s),
        None,
        "unavailable",
        error,
        metadata or {},
    )


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "case"
