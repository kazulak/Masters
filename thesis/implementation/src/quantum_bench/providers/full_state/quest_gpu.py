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
from quantum_bench.providers.full_state.quest_cpu_benchmark import (
    DEFAULT_MAX_OUTPUT_QUBITS,
    QUEST_COMPARABLE_ALGOS,
    STATE_OUTPUT_MODE_NONE,
    _check_output_caps,
    _quest_output_settings,
    _quest_runtime_metadata,
    _load_state_dump,
    _quest_algo_from_case,
    _quest_exact_failed,
    _sanitize,
    _state_dump_metadata,
)
from quantum_bench.tn.network import TensorNetworkValue


QUEST_GPU_ROUTE_ID = "quest_gpu_full_state_exact"
QUEST_GPU_VERIFICATION_SCHEMA_VERSION = "quest_gpu_verification_v1"
QUEST_GPU_VERIFICATION_DIR = Path("build") / "gpu_verification"
QUEST_GPU_VERIFICATION_ARTIFACT = QUEST_GPU_VERIFICATION_DIR / f"{QUEST_GPU_ROUTE_ID}.json"


class QuestGpuFullStateExactRoute:
    name = QUEST_GPU_ROUTE_ID
    backend_family = "quest"
    identity = RouteIdentity(
        route_id=name,
        display_name="QuEST GPU full-state exact output",
        role="optional_gpu_candidate",
        simulation_method="full_state_vector",
        kernel_family="full_state_vector",
        hardware_target="gpu",
        execution_mode="external_process",
        output_contract="statevector",
        validation_mode="compare_statevector",
    )

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.verification_path = quest_gpu_verification_path(root_dir)

    def probe(self) -> RouteProbe:
        verification = load_quest_gpu_verification(self.root_dir)
        metadata = {
            "verification_artifact": str(self.verification_path),
            "gpu_backend_verified": bool(verification and verification.get("gpu_backend_verified")),
            "gpu_program_executed": bool(verification and verification.get("gpu_program_executed")),
        }
        if not verification:
            return RouteProbe(self.name, False, "quest_gpu_verification_missing", metadata=metadata)
        metadata.update(_verification_metadata(verification))
        if not _verification_is_valid(verification):
            return RouteProbe(self.name, False, str(verification.get("blocker_reason") or "quest_gpu_verification_invalid"), metadata=metadata)
        runner = Path(str(verification.get("runner_path") or ""))
        if not runner.exists():
            return RouteProbe(self.name, False, "quest_gpu_runner_missing", metadata=metadata)
        return RouteProbe(self.name, True, metadata=metadata)

    def capabilities(self) -> RouteCapabilities:
        probe = self.probe()
        return RouteCapabilities(
            identity=self.identity,
            supported_workload_families=tuple(sorted(QUEST_COMPARABLE_ALGOS)),
            can_return_output=probe.available,
            can_measure_energy=False,
            metadata={
                "available": probe.available,
                "reason": probe.reason,
                "deterministic_statevector_only": True,
                "default_max_output_qubits": DEFAULT_MAX_OUTPUT_QUBITS,
                "gpu_records_require_real_gpu_execution": True,
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
            metadata={"execution_model": "full_state", "output_kind": "statevector", "contraction_execution_target": "gpu"},
        )

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> dict:
        return {"graph": graph, "network": network}

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        graph: TaskGraph = dict(prepared)["graph"]  # type: ignore[arg-type]
        verification = load_quest_gpu_verification(self.root_dir)
        if not verification or not _verification_is_valid(verification):
            return _quest_exact_failed(self.name, self.backend_family, "quest_gpu_verification_missing_or_invalid")
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

        runner = Path(str(verification.get("runner_path") or ""))
        runner_root = Path(str(verification.get("runner_root") or runner.parent.parent))
        if not runner.exists():
            return _quest_exact_failed(self.name, self.backend_family, "quest_gpu_runner_missing", metadata=_verification_metadata(verification))

        case_id = str(context.case.get("case_id", graph.network.circuit.name))
        rel_dump: Path | None = None
        dump_path: Path | None = None
        max_amplitudes = int(options.get("max_output_amplitudes", 1 << DEFAULT_MAX_OUTPUT_QUBITS))
        cmd = [str(runner), "--algo", algo, "--json", "--max-output-amplitudes", str(max_amplitudes)]
        if state_output_mode != STATE_OUTPUT_MODE_NONE:
            rel_dump = Path("cases") / _sanitize(case_id) / "quest_gpu_full_state" / f"repeat_{context.repeat_id}" / "state_dump.json"
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
        timeout_s = float(context.timeout_s) if context.timeout_s is not None else None
        try:
            result = subprocess.run(cmd, cwd=runner_root, capture_output=True, text=True, check=False, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                f"quest_gpu_timeout:{timeout_s}",
                time.perf_counter() - start,
                metadata={"command": cmd, "timeout_s": timeout_s, **_verification_metadata(verification)},
            )
        total_s = time.perf_counter() - start
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                (result.stderr or result.stdout or "QuEST GPU runner did not emit JSON").strip(),
                total_s,
                metadata={"command": cmd, **_verification_metadata(verification)},
            )
        quest_status = str(payload.get("status") or "")
        if result.returncode != 0 or quest_status not in {"ok", "passed"}:
            return _quest_exact_failed(
                self.name,
                self.backend_family,
                str(payload.get("error") or result.stderr or "QuEST GPU runner failed"),
                total_s,
                metadata={"quest": payload, "command": cmd, **_verification_metadata(verification)},
            )
        verification_metadata = _verification_metadata(verification)
        metadata_common = {
            **_quest_runtime_metadata(payload, cmd, total_s=total_s, settings=settings),
            "gpu_synchronized": bool(verification.get("gpu_synchronized", True)),
            **verification_metadata,
        }
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
            {
                "state_dump": _state_dump_metadata(dump_payload),
                **metadata_common,
            },
        )


def quest_gpu_verification_path(root_dir: Path) -> Path:
    return root_dir / QUEST_GPU_VERIFICATION_ARTIFACT


def load_quest_gpu_verification(root_dir: Path) -> dict | None:
    path = quest_gpu_verification_path(root_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != QUEST_GPU_VERIFICATION_SCHEMA_VERSION:
        return None
    return payload


def _verification_is_valid(payload: dict) -> bool:
    return bool(
        payload.get("status") == "verified"
        and payload.get("gpu_backend_verified") is True
        and payload.get("gpu_program_executed") is True
        and payload.get("gpu_device_name")
        and payload.get("runner_path")
        and payload.get("accelerator_kind") in {"amd_gpu", "nvidia_gpu"}
    )


def _verification_metadata(payload: dict) -> dict:
    return {
        "gpu_backend_verified": bool(payload.get("gpu_backend_verified", False)),
        "gpu_program_executed": bool(payload.get("gpu_program_executed", False)),
        "gpu_device_name": payload.get("gpu_device_name"),
        "gpu_runtime_stack": payload.get("gpu_runtime_stack"),
        "gpu_toolkit_metadata": payload.get("gpu_toolkit_metadata"),
        "accelerator_kind": payload.get("accelerator_kind"),
        "verification_artifact": payload.get("verification_artifact") or payload.get("artifact_path"),
        "verification_backend": payload.get("verification_backend"),
    }
