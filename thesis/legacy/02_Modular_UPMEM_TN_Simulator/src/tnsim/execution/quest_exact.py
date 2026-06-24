from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
import time

import numpy as np

from tnsim.core.model import ExecutionRun, TensorNetwork
from .energy import estimate_energy


RESULT_RE = re.compile(r"QUEST_RESULT\s+(\{.*\})")


def execute_quest_exact(network: TensorNetwork, config: dict, output_dir: Path) -> ExecutionRun:
    root_dir = Path(__file__).resolve().parents[3]
    baseline_dir = root_dir / "baselines" / "quest_exact"
    runner = baseline_dir / "bin" / "quest_exact_runner"
    _ensure_runner(baseline_dir, runner)

    output_path = output_dir / "quest_output.bin"
    cmd = [str(runner), "--output", str(output_path)]
    source = network.circuit.source
    if source.get("kind") == "qasm_file":
        cmd.extend(["--qasm", str(source["path"])])
    else:
        name = source.get("name", network.circuit.name)
        cmd.extend(["--circuit", str(name), "--qubits", str(network.circuit.n_qubits)])

    wall_start = time.perf_counter()
    completed = subprocess.run(cmd, cwd=baseline_dir, capture_output=True, text=True, check=True)
    wall_seconds = time.perf_counter() - wall_start
    metrics = _parse_result(completed.stdout)
    output = _read_complex_vector(output_path, network.circuit.n_qubits)

    measured_energy = float(metrics.get("energy_joules", 0.0))
    measured_source = str(metrics.get("energy_source", "unknown"))
    if measured_energy > 0.0 and measured_source == "rapl":
        energy_joules = measured_energy
        energy_source = "rapl"
        watts = energy_joules / max(float(metrics["execution_seconds"]), 1.0e-300)
    else:
        energy_joules, energy_source, watts = estimate_energy(float(metrics["execution_seconds"]), config)
        energy_source = f"{energy_source}_fallback_after_{measured_source}"

    profile = {
        "task_id": "quest_exact_whole_circuit",
        "route": "quest_exact_statevector",
        "data_format": "complex_f64_host",
        "status": "ok",
        "prepare_seconds": 0.0,
        "execute_seconds": float(metrics["execution_seconds"]),
        "total_seconds": float(metrics["execution_seconds"]),
        "wall_seconds_including_subprocess": wall_seconds,
        "host_to_device_bytes": 0,
        "device_to_host_bytes": 0,
        "host_tensor_read_bytes": 0,
        "host_tensor_write_bytes": int(output.nbytes),
        "energy_joules": energy_joules,
        "energy_source": energy_source,
        "estimated_power_watts": watts,
        "output_tensor_id": "quest_statevector",
    }
    return ExecutionRun(
        output=output,
        profiles=[profile],
        execution_seconds=float(metrics["execution_seconds"]),
        energy_joules=energy_joules,
        energy_source=energy_source,
        estimated_power_watts=watts,
    )


def _ensure_runner(baseline_dir: Path, runner: Path) -> None:
    if runner.exists():
        return
    subprocess.run(["make"], cwd=baseline_dir, check=True)


def _parse_result(stdout: str) -> dict:
    match = RESULT_RE.search(stdout)
    if not match:
        raise ValueError(f"Could not parse QuEST result line from stdout:\n{stdout}")
    return json.loads(match.group(1))


def _read_complex_vector(path: Path, n_qubits: int) -> np.ndarray:
    raw = np.fromfile(path, dtype="<f8")
    expected = (1 << n_qubits) * 2
    if raw.size != expected:
        raise ValueError(f"QuEST output size mismatch: expected {expected} f64 values, got {raw.size}")
    pairs = raw.reshape((-1, 2))
    return (pairs[:, 0] + 1j * pairs[:, 1]).astype(np.complex128)

