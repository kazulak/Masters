from __future__ import annotations

import subprocess
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
    gpu_hardware = _gpu_hardware_probe()
    gpu_candidates = _gpu_candidate_matrix(root_dir, gpu_hardware)
    payload = {
        "schema_version": SIMULATION_BACKEND_PROBE_SCHEMA_VERSION,
        "routes": route_reports,
        "optional_libraries": {
            "cotengra": _module_status("cotengra"),
            "quimb": _module_status("quimb"),
            "cupy": _module_status("cupy"),
            "torch": _module_status("torch"),
            "jax": _module_status("jax"),
            "qiskit": _module_status("qiskit"),
            "qiskit_aer": _module_status("qiskit_aer"),
            "cudaq": _module_status("cudaq"),
            "cuquantum": _module_status("cuquantum"),
        },
        "gpu_probe": {
            "hardware": gpu_hardware,
            "rocminfo": _command_status("rocminfo"),
            "rocm_smi": _command_status("rocm-smi"),
            "hipcc": _command_status("hipcc"),
            "nvidia_smi": _command_status("nvidia-smi"),
            "nvcc": _command_status("nvcc"),
            "gpu_candidates": gpu_candidates,
            "cuda_only_assumption_used": False,
            "gpu_execution_backend_added": any(bool(candidate["benchmark_route_eligible"]) for candidate in gpu_candidates),
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


def _gpu_hardware_probe() -> JsonDict:
    lspci_lines = _lspci_gpu_lines()
    amd = [line for line in lspci_lines if any(marker in line.lower() for marker in ("amd", "ati", "radeon"))]
    nvidia = [line for line in lspci_lines if "nvidia" in line.lower()]
    return {
        "amd_gpu_pci_detected": bool(amd),
        "nvidia_gpu_pci_detected": bool(nvidia),
        "amd_gpu_pci_devices": amd,
        "nvidia_gpu_pci_devices": nvidia,
        "dev_kfd_present": Path("/dev/kfd").exists(),
        "dev_dri_present": Path("/dev/dri").exists(),
    }


def _lspci_gpu_lines() -> list[str]:
    try:
        result = subprocess.run(["lspci"], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return []
    if result.returncode != 0:
        return []
    out = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in ("vga", "3d", "display", "radeon", "nvidia", "amd/ati")):
            out.append(line.strip())
    return out


def _gpu_candidate_matrix(root_dir: Path, hardware: JsonDict) -> list[JsonDict]:
    quest_cmake = root_dir / "external" / "QuEST" / "CMakeLists.txt"
    quest_text = quest_cmake.read_text(encoding="utf-8", errors="ignore") if quest_cmake.exists() else ""
    quest_has_hip = "ENABLE_HIP" in quest_text
    quest_has_cuda = "ENABLE_CUDA" in quest_text
    rocm_tools = {name: _command_status(name) for name in ("hipcc", "rocminfo", "rocm-smi")}
    cuda_tools = {name: _command_status(name) for name in ("nvcc", "nvidia-smi")}
    torch_gpu = _torch_gpu_execution_status()
    candidates = [
        _candidate(
            "quest_gpu_full_state_hip",
            "tailored_quantum_gpu",
            1,
            "amd_rocm",
            classification="amd_rocm_candidate_not_verified" if quest_has_hip else "feasible_later_not_benchmarkable_now",
            blocker="quest_enable_hip_source_only_not_benchmark_evidence" if quest_has_hip else "quest_enable_hip_not_detected",
            source_paths=["external/QuEST/CMakeLists.txt"] if quest_has_hip else [],
            dependencies=rocm_tools,
            gpu_execution_verified=False,
            usable_current=bool(quest_has_hip and hardware.get("amd_gpu_pci_detected") and all(tool["available"] for tool in rocm_tools.values()) and False),
        ),
        _candidate(
            "quest_gpu_full_state_cuda",
            "tailored_quantum_gpu",
            2,
            "nvidia_cuda",
            classification="nvidia_cuda_only_not_usable_here" if not hardware.get("nvidia_gpu_pci_detected") else "feasible_later_not_benchmarkable_now",
            blocker="requires_nvidia_cuda_build_and_minimal_gpu_execution",
            source_paths=["external/QuEST/CMakeLists.txt"] if quest_has_cuda else [],
            dependencies=cuda_tools,
            gpu_execution_verified=False,
        ),
        _python_candidate("qiskit_aer_gpu_statevector", "cuda_quantum_stack", 3, "nvidia_cuda", "qiskit_aer", hardware, "qiskit_aer_gpu_execution_not_verified"),
        _python_candidate("qiskit_aer_gpu_tensor_network", "cuda_quantum_stack", 4, "nvidia_cuda", "qiskit_aer", hardware, "qiskit_aer_cuquantum_execution_not_verified"),
        _python_candidate("cudaq_gpu_full_state", "cuda_quantum_stack", 5, "nvidia_cuda", "cudaq", hardware, "cudaq_gpu_execution_not_verified"),
        _python_candidate("cudaq_gpu_tensor_network", "cuda_quantum_stack", 6, "nvidia_cuda", "cudaq", hardware, "cudaq_tn_gpu_execution_not_verified"),
        _python_candidate("cuquantum_statevector", "cuda_quantum_stack", 7, "nvidia_cuda", "cuquantum", hardware, "cuquantum_integration_not_implemented"),
        _python_candidate("cuquantum_tensor_network", "cuda_quantum_stack", 8, "nvidia_cuda", "cuquantum", hardware, "cutensornet_integration_not_implemented"),
        _quimb_gpu_candidate(hardware),
        _candidate(
            "torch_rocm_generic_full_state",
            "generic_tensor_gpu",
            10,
            "amd_rocm",
            classification=torch_gpu["classification"],
            blocker=torch_gpu["reason"],
            source_paths=[],
            dependencies={"torch": _module_status("torch"), "hipcc": _command_status("hipcc")},
            gpu_execution_verified=bool(torch_gpu["gpu_execution_verified"]),
            usable_current=bool(torch_gpu["gpu_execution_verified"]),
            extra={"minimal_execution": torch_gpu},
        ),
    ]
    return [to_jsonable(candidate) for candidate in candidates]


def _candidate(
    candidate_id: str,
    category: str,
    priority: int,
    stack: str,
    *,
    classification: str,
    blocker: str,
    source_paths: list[str],
    dependencies: JsonDict,
    gpu_execution_verified: bool,
    usable_current: bool = False,
    extra: JsonDict | None = None,
) -> JsonDict:
    return {
        "candidate_id": candidate_id,
        "candidate_category": category,
        "preferred_priority": priority,
        "target_gpu_stack": stack,
        "classification": "usable_current_machine" if usable_current and gpu_execution_verified else classification,
        "benchmark_route_eligible": bool(gpu_execution_verified),
        "gpu_execution_verified": bool(gpu_execution_verified),
        "source_support_is_not_benchmark_evidence": True,
        "evidence_paths": source_paths,
        "detected_dependencies": dependencies,
        "blocker_reason": None if gpu_execution_verified else blocker,
        **(extra or {}),
    }


def _python_candidate(
    candidate_id: str,
    category: str,
    priority: int,
    stack: str,
    module: str,
    hardware: JsonDict,
    blocker: str,
) -> JsonDict:
    module_status = _module_status(module)
    if not module_status["available"]:
        classification = "dependency_missing"
        reason = f"{module}_not_installed"
    elif stack == "nvidia_cuda" and not hardware.get("nvidia_gpu_pci_detected"):
        classification = "nvidia_cuda_only_not_usable_here"
        reason = "requires_nvidia_gpu_and_cuda_runtime"
    else:
        classification = "feasible_later_not_benchmarkable_now"
        reason = blocker
    return _candidate(
        candidate_id,
        category,
        priority,
        stack,
        classification=classification,
        blocker=reason,
        source_paths=[],
        dependencies={module: module_status},
        gpu_execution_verified=False,
    )


def _quimb_gpu_candidate(hardware: JsonDict) -> JsonDict:
    deps = {
        "quimb": _module_status("quimb"),
        "cotengra": _module_status("cotengra"),
        "cupy": _module_status("cupy"),
        "torch": _module_status("torch"),
        "jax": _module_status("jax"),
    }
    if not deps["quimb"]["available"] or not deps["cotengra"]["available"]:
        classification = "dependency_missing"
        blocker = "quimb_or_cotengra_not_installed"
    elif not any(deps[name]["available"] for name in ("cupy", "torch", "jax")):
        classification = "cpu_only_current_environment"
        blocker = "no_gpu_array_backend_detected_for_quimb_cotengra"
    elif not (hardware.get("amd_gpu_pci_detected") or hardware.get("nvidia_gpu_pci_detected")):
        classification = "feasible_later_not_benchmarkable_now"
        blocker = "gpu_hardware_not_detected"
    else:
        classification = "feasible_later_not_benchmarkable_now"
        blocker = "quimb_cotengra_gpu_execution_not_verified"
    return _candidate(
        "quimb_cotengra_gpu",
        "tailored_quantum_gpu",
        9,
        "gpu_array_backend",
        classification=classification,
        blocker=blocker,
        source_paths=[],
        dependencies=deps,
        gpu_execution_verified=False,
    )


def _torch_gpu_execution_status() -> JsonDict:
    status = _module_status("torch")
    if not status["available"]:
        return {
            "classification": "dependency_missing",
            "reason": "torch_not_installed",
            "gpu_execution_verified": False,
        }
    try:
        import torch  # type: ignore[import-not-found]

        hip = getattr(getattr(torch, "version", object()), "hip", None)
        cuda_available = bool(torch.cuda.is_available())
        if not hip:
            return {
                "classification": "cpu_only_current_environment",
                "reason": "torch_installed_without_hip_runtime",
                "gpu_execution_verified": False,
                "torch_version": getattr(torch, "__version__", status["version"]),
                "torch_hip_version": hip,
                "torch_cuda_available": cuda_available,
            }
        if not cuda_available:
            return {
                "classification": "amd_rocm_candidate_not_verified",
                "reason": "torch_hip_present_but_no_gpu_device_available",
                "gpu_execution_verified": False,
                "torch_version": getattr(torch, "__version__", status["version"]),
                "torch_hip_version": hip,
                "torch_cuda_available": cuda_available,
            }
        device = torch.device("cuda")
        tensor = torch.ones((4,), device=device)
        out = tensor + tensor
        torch.cuda.synchronize()
        verified = bool(out.is_cuda and float(out.sum().item()) == 8.0)
        return {
            "classification": "usable_current_machine" if verified else "amd_rocm_candidate_not_verified",
            "reason": None if verified else "torch_tensor_execution_not_verified_on_gpu",
            "gpu_execution_verified": verified,
            "torch_version": getattr(torch, "__version__", status["version"]),
            "torch_hip_version": hip,
            "torch_cuda_available": cuda_available,
            "device_name": torch.cuda.get_device_name(0),
        }
    except Exception as exc:
        return {
            "classification": "amd_rocm_candidate_not_verified",
            "reason": f"torch_gpu_probe_failed: {exc}",
            "gpu_execution_verified": False,
        }


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
