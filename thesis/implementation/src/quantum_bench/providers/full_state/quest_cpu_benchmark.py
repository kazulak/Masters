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
STATE_OUTPUT_MODE_FULL_DUMP = "full_dump"
STATE_OUTPUT_MODE_NONE = "none"
VALIDATION_METHOD_FULL_STATEVECTOR = "full_statevector"
VALIDATION_METHOD_NATIVE_STATUS_GATE_COUNTS = "native_status_gate_counts"


class QuestCpuFullStateExactRoute:
    name = "quest_cpu_full_state_exact"
    backend_family = "quest"
    identity = RouteIdentity(
        route_id=name,
        display_name="QuEST CPU full-state exact output",
        role="serious_full_state_baseline",
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
        settings, reason = _quest_output_settings(context)
        if reason:
            return False, reason
        if settings["state_output_mode"] == STATE_OUTPUT_MODE_NONE:
            return True, None
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
        settings, settings_reason = _quest_output_settings(context)
        if settings_reason:
            return _quest_exact_failed(self.name, self.backend_family, settings_reason)
        state_output_mode = str(settings["state_output_mode"])
        if state_output_mode != STATE_OUTPUT_MODE_NONE:
            cap_ok, cap_reason = _check_output_caps(graph.network.circuit.n_qubits, options)
            if not cap_ok:
                return _quest_exact_failed(self.name, self.backend_family, cap_reason or "state_output_cap_exceeded")
        algo = _quest_algo_from_case(context)
        if algo is None:
            return _quest_exact_failed(self.name, self.backend_family, "unsupported_quest_comparable_algorithm")

        case_id = str(context.case.get("case_id", graph.network.circuit.name))
        rel_dump: Path | None = None
        dump_path: Path | None = None
        max_amplitudes = int(options.get("max_output_amplitudes", DEFAULT_MAX_OUTPUT_AMPLITUDES))
        cmd = [str(self.runner), "--algo", algo, "--json", "--max-output-amplitudes", str(max_amplitudes)]
        if state_output_mode != STATE_OUTPUT_MODE_NONE:
            rel_dump = Path("cases") / _sanitize(case_id) / "quest_full_state" / f"repeat_{context.repeat_id}" / "state_dump.json"
            dump_path = context.run_dir / rel_dump
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--dump-state-json", str(dump_path)])
        if algo == "HS" and "logical_qubits" in context.case.get("circuit", {}):
            cmd.extend(["--logical-qubits", str(context.case["circuit"]["logical_qubits"])])
        else:
            cmd.extend(["--qubits", str(graph.network.circuit.n_qubits)])
        depth = context.case.get("circuit", {}).get("depth")
        if depth is not None:
            cmd.extend(["--depth", str(depth)])
        repeat_layers = int(context.case.get("circuit", {}).get("repeat_layers", 1) or 1)
        cmd.extend(["--repeat-layers", str(repeat_layers)])

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
        metadata_common = _quest_runtime_metadata(payload, cmd, total_s=total_s, settings=settings)
        if state_output_mode == STATE_OUTPUT_MODE_NONE:
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput(
                    contract="metrics_only",
                    array=None,
                    metadata={
                        "output_kind": "metrics_only",
                        "execution_model": "full_state",
                        **metadata_common,
                    },
                ),
                ExecutionProfile(kernel_s=float(payload.get("time_s") or 0.0), total_s=total_s),
                payload.get("energy_joules"),
                str(payload.get("energy_source") or "unavailable"),
                None,
                metadata_common,
            )

        assert dump_path is not None
        assert rel_dump is not None
        try:
            state, dump_payload = _load_state_dump(dump_path, graph.network.circuit.n_qubits)
        except ValueError as exc:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                str(exc),
                total_s,
                metadata={"quest": payload, "command": cmd, "state_dump_artifact": rel_dump.as_posix(), **metadata_common},
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
                    **metadata_common,
                },
            ),
            ExecutionProfile(kernel_s=float(payload.get("time_s") or 0.0), total_s=total_s),
            payload.get("energy_joules"),
            str(payload.get("energy_source") or "unavailable"),
            None,
            {"quest": payload, "state_dump": _state_dump_metadata(dump_payload), "command": cmd, **metadata_common},
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


def _quest_output_settings(context: BenchmarkContext) -> tuple[dict, str | None]:
    options = dict(context.route_config.get("options") or {})
    metadata = dict(context.suite.get("metadata") or {})
    state_output_mode = str(options.get("state_output_mode") or metadata.get("state_output_mode") or STATE_OUTPUT_MODE_FULL_DUMP)
    validation_method = str(options.get("validation_method") or metadata.get("validation_method") or VALIDATION_METHOD_FULL_STATEVECTOR)
    performance_tier = bool(options.get("performance_tier", metadata.get("performance_tier", False)))
    if state_output_mode not in {STATE_OUTPUT_MODE_FULL_DUMP, STATE_OUTPUT_MODE_NONE}:
        return {}, f"unsupported_state_output_mode:{state_output_mode}"
    if state_output_mode == STATE_OUTPUT_MODE_NONE:
        if validation_method != VALIDATION_METHOD_NATIVE_STATUS_GATE_COUNTS:
            return {}, "metrics_only_requires_native_status_gate_counts_validation"
        if not performance_tier:
            return {}, "metrics_only_requires_performance_tier_true"
    return {
        "state_output_mode": state_output_mode,
        "validation_method": validation_method,
        "performance_tier": performance_tier,
        "output_contract": "metrics_only" if state_output_mode == STATE_OUTPUT_MODE_NONE else "statevector",
        "output_contract_label": "metrics_only" if state_output_mode == STATE_OUTPUT_MODE_NONE else "full_statevector",
        "output_contract_is_exact": state_output_mode != STATE_OUTPUT_MODE_NONE,
        "exact_output_comparable": state_output_mode != STATE_OUTPUT_MODE_NONE,
        "full_statevector_validation_available": state_output_mode != STATE_OUTPUT_MODE_NONE,
        "output_contract_note": (
            "Runtime/performance tier only; no full statevector artifact was requested."
            if state_output_mode == STATE_OUTPUT_MODE_NONE
            else "Full statevector artifact requested for exact output comparison."
        ),
    }, None


def _quest_runtime_metadata(payload: dict, command: list[str], *, total_s: float, settings: dict) -> dict:
    validation_status = "passed_native_status" if settings["state_output_mode"] == STATE_OUTPUT_MODE_NONE else "passed"
    return {
        "quest": payload,
        "command": command,
        "state_output_mode": settings["state_output_mode"],
        "output_contract": settings["output_contract"],
        "output_contract_label": settings["output_contract_label"],
        "output_contract_is_exact": settings["output_contract_is_exact"],
        "output_contract_note": settings["output_contract_note"],
        "output_contract_source": "suite_or_route_options",
        "output_contract_enforced": True,
        "validation_method": settings["validation_method"],
        "validation_status": validation_status,
        "performance_tier": settings["performance_tier"],
        "output_contract_metrics_only": settings["state_output_mode"] == STATE_OUTPUT_MODE_NONE,
        "exact_output_comparable": settings["exact_output_comparable"],
        "full_statevector_validation_available": settings["full_statevector_validation_available"],
        "native_process_wall_time_s": float(total_s),
        "quest_simulation_compute_time_s": float(payload.get("time_s") or 0.0),
        "state_dump_requested": bool(payload.get("state_dump_requested", settings["state_output_mode"] != STATE_OUTPUT_MODE_NONE)),
        "state_dump_time_s": float(payload.get("state_dump_time_s") or 0.0),
        "repeat_layers": int(payload.get("repeat_layers") or 1),
        "timing_scope": "compute_only_native_and_process_wall",
        "energy_measurement_status": _energy_measurement_status(payload),
    }


def _energy_measurement_status(payload: dict) -> str:
    source = str(payload.get("energy_source") or "unavailable")
    value = payload.get("energy_joules")
    if source == "unavailable" or value is None:
        return "unavailable"
    try:
        joules = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if joules > 0:
        return "measured_positive"
    return "available_zero_delta"


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
