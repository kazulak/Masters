from __future__ import annotations

import shutil
from importlib import metadata as importlib_metadata
from pathlib import Path

from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.providers import route_registry


SIMULATION_BACKEND_PROBE_SCHEMA_VERSION = "simulation_backend_probe_v1"


def probe_simulation_backends(root_dir: Path) -> JsonDict:
    routes = route_registry(root_dir)
    route_reports: list[JsonDict] = []
    for route_id in sorted(routes):
        route = routes[route_id]
        probe = route.probe()
        capabilities = route.capabilities()
        route_reports.append(
            {
                "route_id": route_id,
                "available": probe.available,
                "reason": probe.reason,
                "backend_family": route.backend_family,
                "execution_model": _execution_model(route.identity.simulation_method),
                "contraction_execution_target": _target(route.identity.hardware_target),
                "accelerator_kind": _accelerator_kind(route.identity.hardware_target),
                "output_contract": route.identity.output_contract,
                "can_return_output": capabilities.can_return_output,
                "metadata": {**probe.metadata, **capabilities.metadata},
            }
        )
    payload = {
        "schema_version": SIMULATION_BACKEND_PROBE_SCHEMA_VERSION,
        "routes": route_reports,
        "optional_libraries": {
            "cotengra": _module_status("cotengra"),
            "quimb": _module_status("quimb"),
            "cupy": _module_status("cupy"),
            "torch": _module_status("torch"),
            "jax": _module_status("jax"),
        },
        "gpu_probe": {
            "rocminfo": _command_status("rocminfo"),
            "rocm_smi": _command_status("rocm-smi"),
            "hipcc": _command_status("hipcc"),
            "cuda_only_assumption_used": False,
            "gpu_execution_backend_added": False,
            "gpu_benchmark_records_emitted": False,
            "status": "feasibility_only",
        },
        "notes": {
            "unavailable_optional_libraries_do_not_create_benchmark_records": True,
            "gpu_records_require_real_gpu_execution": True,
        },
    }
    return to_jsonable(payload)


def _module_status(name: str) -> JsonDict:
    version = _module_version(name)
    return {"available": version is not None, "version": version}


def _module_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _command_status(name: str) -> JsonDict:
    path = shutil.which(name)
    return {"available": path is not None, "path": path}


def _execution_model(simulation_method: str) -> str:
    return "full_state" if "full_state" in simulation_method else "tensor_network"


def _target(hardware_target: str) -> str:
    if "gpu" in hardware_target:
        return "gpu"
    if "upmem" in hardware_target:
        return "upmem"
    return "cpu"


def _accelerator_kind(hardware_target: str) -> str:
    if "amd" in hardware_target:
        return "amd_gpu"
    if "nvidia" in hardware_target or "cuda" in hardware_target:
        return "nvidia_gpu"
    if "gpu" in hardware_target:
        return "gpu"
    if "upmem" in hardware_target:
        return "upmem"
    return "none"
