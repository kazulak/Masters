from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

from tnsim.core.model import ExecutionRun, TensorNetwork
from .energy import estimate_energy


SUPPORTED_CIRCUITS = {"bell_2q", "ghz_4q"}


def execute_mvp_upmem(network: TensorNetwork, config: dict, output_dir: Path) -> ExecutionRun:
    circuit = _mvp_circuit_name(network)
    root_dir = Path(__file__).resolve().parents[3]
    mvp_dir = root_dir.parent / "01_MVP_DenseGEMM"
    data_dir = mvp_dir / "data_exchange"
    task_graph_path = data_dir / "task_graph.json"
    tensor_data_path = data_dir / "tensor_data.bin"
    output_path = data_dir / "output_amplitudes.bin"
    log_path = data_dir / "execution_log.json"

    _prepare_once(output_dir, mvp_dir, circuit)
    _run_mvp_host(mvp_dir)
    output = _read_mvp_output(output_path)
    mvp_log = _read_json(log_path)

    _copy_mvp_artifacts(output_dir, task_graph_path, tensor_data_path, output_path, log_path)

    execution_seconds = float(mvp_log["summary"]["total_wall_seconds"])
    energy_joules, energy_source, watts = estimate_energy(execution_seconds, config)
    profiles = _convert_profiles(mvp_log, energy_source, watts)
    return ExecutionRun(
        output=output,
        profiles=profiles,
        execution_seconds=execution_seconds,
        energy_joules=energy_joules,
        energy_source=f"{energy_source}_upmem_simulator",
        estimated_power_watts=watts,
    )


def _prepare_once(output_dir: Path, mvp_dir: Path, circuit: str) -> None:
    marker = output_dir / "mvp_prepared.json"
    data_marker = mvp_dir / "data_exchange" / "v2_prepared_circuit.json"
    already_prepared = (
        _marker_matches(marker, circuit)
        and _marker_matches(data_marker, circuit)
        and (mvp_dir / "mvp_host").exists()
    )
    if already_prepared:
        return

    env = os.environ.copy()
    subprocess.run([sys.executable, "python_frontend/generate_plan.py", circuit], cwd=mvp_dir, env=env, check=True)
    subprocess.run(["make"], cwd=mvp_dir, check=True)
    marker.write_text(json.dumps({"circuit": circuit}, sort_keys=True) + "\n", encoding="utf-8")
    data_marker.write_text(json.dumps({"circuit": circuit}, sort_keys=True) + "\n", encoding="utf-8")


def _marker_matches(marker: Path, circuit: str) -> bool:
    if not marker.exists():
        return False
    try:
        return _read_json(marker).get("circuit") == circuit
    except (OSError, json.JSONDecodeError):
        return False


def _run_mvp_host(mvp_dir: Path) -> None:
    subprocess.run(
        [
            "./mvp_host",
            "data_exchange/task_graph.json",
            "data_exchange/tensor_data.bin",
            "data_exchange/output_amplitudes.bin",
            "data_exchange/execution_log.json",
        ],
        cwd=mvp_dir,
        check=True,
    )


def _mvp_circuit_name(network: TensorNetwork) -> str:
    name = str(network.circuit.source.get("name", network.circuit.name))
    if network.circuit.source.get("kind") == "qasm_file":
        name = Path(str(network.circuit.source["path"])).stem
    if name not in SUPPORTED_CIRCUITS:
        raise ValueError(
            f"raw_upmem_dense MVP supports only {sorted(SUPPORTED_CIRCUITS)}; "
            f"got {name!r}."
        )
    return name


def _read_mvp_output(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        n_elem = struct.unpack("<i", handle.read(4))[0]
        real = np.frombuffer(handle.read(n_elem * 8), dtype="<f8")
        imag = np.frombuffer(handle.read(n_elem * 8), dtype="<f8")
    return (real + 1j * imag).astype(np.complex128)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copy_mvp_artifacts(output_dir: Path, *paths: Path) -> None:
    for path in paths:
        shutil.copyfile(path, output_dir / f"mvp_{path.name}")


def _convert_profiles(mvp_log: dict, energy_source: str, watts: float) -> list[dict]:
    profiles = []
    for item in mvp_log.get("profiles", []):
        timing = item.get("timing_seconds", {})
        bytes_record = item.get("bytes", {})
        total = float(timing.get("total", 0.0))
        profiles.append(
            {
                "task_id": f"mvp_task_{item.get('task_id')}",
                "route": "raw_upmem_dense",
                "data_format": item.get("selected_format", "complex_i8_tile_scaled"),
                "status": item.get("status", "ok"),
                "prepare_seconds": float(timing.get("host_prepare", 0.0)),
                "execute_seconds": total,
                "total_seconds": total,
                "host_pack_quantize_seconds": float(timing.get("host_pack_quantize", 0.0)),
                "h2d_dma_seconds": float(timing.get("h2d_dma", 0.0)),
                "dpu_kernel_seconds": float(timing.get("dpu_kernel", 0.0)),
                "d2h_dma_seconds": float(timing.get("d2h_dma", 0.0)),
                "host_dequantize_accumulate_seconds": float(timing.get("host_dequantize_accumulate", 0.0)),
                "host_to_device_bytes": int(bytes_record.get("host_to_dpu", 0)),
                "device_to_host_bytes": int(bytes_record.get("dpu_to_host", 0)),
                "host_tensor_read_bytes": 0,
                "host_tensor_write_bytes": 0,
                "energy_joules": total * watts,
                "energy_source": f"{energy_source}_upmem_simulator",
                "estimated_power_watts": watts,
                "output_tensor_id": item.get("output_key"),
            }
        )
    return profiles
