from __future__ import annotations

import json
import re
import subprocess
import shutil
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.providers import route_registry
from quantum_bench.providers.full_state.quest_gpu import QUEST_GPU_VERIFICATION_SCHEMA_VERSION, quest_gpu_verification_path


SIMULATION_BACKEND_PROBE_SCHEMA_VERSION = "simulation_backend_probe_v1"
GPU_VERIFY_CHOICES = ("none", "auto", "quest-hip", "quest-cuda", "torch-rocm")


def probe_simulation_backends(root_dir: Path, *, verify_gpu: str = "none") -> JsonDict:
    if verify_gpu not in GPU_VERIFY_CHOICES:
        raise ValueError(f"Unsupported --verify-gpu value: {verify_gpu}")
    gpu_hardware = _gpu_hardware_probe()
    gpu_verification = _verify_gpu_backend(root_dir, verify_gpu, gpu_hardware) if verify_gpu != "none" else None
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
    gpu_candidates = _gpu_candidate_matrix(root_dir, gpu_hardware, gpu_verification)
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
            "gpu_verification": gpu_verification,
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
    rocminfo = _rocminfo_gpu_status()
    render_nodes = _render_nodes()
    return {
        "amd_gpu_pci_detected": bool(amd),
        "nvidia_gpu_pci_detected": bool(nvidia),
        "amd_gpu_pci_devices": amd,
        "nvidia_gpu_pci_devices": nvidia,
        "dev_kfd_present": Path("/dev/kfd").exists(),
        "dev_dri_present": Path("/dev/dri").exists(),
        "dev_dri_renderD128_present": Path("/dev/dri/renderD128").exists(),
        "dev_dri_render_node_present": bool(render_nodes),
        "dev_dri_render_nodes": render_nodes,
        "rocminfo_gpu_agent_detected": bool(rocminfo["gpu_agent_detected"]),
        "rocminfo_gfx_targets": rocminfo["gfx_targets"],
        "rocminfo_returncode": rocminfo["returncode"],
    }


def _render_nodes() -> list[str]:
    dri = Path("/dev/dri")
    if not dri.exists():
        return []
    return sorted(path.as_posix() for path in dri.glob("renderD*"))


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


def _rocminfo_gpu_status() -> JsonDict:
    if shutil.which("rocminfo") is None:
        return {"available": False, "returncode": None, "gpu_agent_detected": False, "gfx_targets": []}
    try:
        result = subprocess.run(["rocminfo"], cwd=Path.cwd(), capture_output=True, text=True, check=False, timeout=15)
        returncode = result.returncode
        stdout = result.stdout
    except subprocess.TimeoutExpired:
        return {"available": False, "returncode": None, "gpu_agent_detected": False, "gfx_targets": [], "timed_out": True}
    except OSError as exc:
        return {"available": False, "returncode": None, "gpu_agent_detected": False, "gfx_targets": [], "error": str(exc)}
    gfx_targets = sorted(set(re.findall(r"\bgfx[0-9a-fA-F]+\b", stdout)))
    return {
        "available": returncode == 0,
        "returncode": returncode,
        "gpu_agent_detected": bool(gfx_targets),
        "gfx_targets": gfx_targets,
    }


def _gpu_candidate_matrix(root_dir: Path, hardware: JsonDict, verification: JsonDict | None = None) -> list[JsonDict]:
    quest_cmake = root_dir / "external" / "QuEST" / "CMakeLists.txt"
    quest_text = quest_cmake.read_text(encoding="utf-8", errors="ignore") if quest_cmake.exists() else ""
    quest_has_hip = "ENABLE_HIP" in quest_text
    quest_has_cuda = "ENABLE_CUDA" in quest_text
    rocm_tools = {name: _command_status(name) for name in ("hipcc", "rocminfo", "rocm-smi")}
    cuda_tools = {name: _command_status(name) for name in ("nvcc", "nvidia-smi")}
    torch_gpu = _torch_gpu_execution_status()
    hip_verified = bool(verification and verification.get("selected_backend") == "quest-hip" and verification.get("status") == "verified")
    cuda_verified = bool(verification and verification.get("selected_backend") == "quest-cuda" and verification.get("status") == "verified")
    hip_blocker = verification.get("blocker_reason") if verification and verification.get("selected_backend") == "quest-hip" else None
    cuda_blocker = verification.get("blocker_reason") if verification and verification.get("selected_backend") == "quest-cuda" else None
    candidates = [
        _candidate(
            "quest_gpu_full_state_hip",
            "tailored_quantum_gpu",
            1,
            "amd_rocm",
            classification="usable_current_machine" if hip_verified else ("amd_rocm_candidate_not_verified" if quest_has_hip else "feasible_later_not_benchmarkable_now"),
            blocker=None if hip_verified else (hip_blocker or ("quest_enable_hip_source_only_not_benchmark_evidence" if quest_has_hip else "quest_enable_hip_not_detected")),
            source_paths=["external/QuEST/CMakeLists.txt"] if quest_has_hip else [],
            dependencies=rocm_tools,
            gpu_execution_verified=hip_verified,
            usable_current=hip_verified,
            extra={"verification_artifact": verification.get("artifact_path") if verification and verification.get("selected_backend") == "quest-hip" else None},
        ),
        _candidate(
            "quest_gpu_full_state_cuda",
            "tailored_quantum_gpu",
            2,
            "nvidia_cuda",
            classification="usable_current_machine" if cuda_verified else ("nvidia_cuda_only_not_usable_here" if not hardware.get("nvidia_gpu_pci_detected") else "feasible_later_not_benchmarkable_now"),
            blocker=None if cuda_verified else (cuda_blocker or "requires_nvidia_cuda_build_and_minimal_gpu_execution"),
            source_paths=["external/QuEST/CMakeLists.txt"] if quest_has_cuda else [],
            dependencies=cuda_tools,
            gpu_execution_verified=cuda_verified,
            usable_current=cuda_verified,
            extra={"verification_artifact": verification.get("artifact_path") if verification and verification.get("selected_backend") == "quest-cuda" else None},
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
            benchmark_route_eligible=False,
            extra={"minimal_execution": torch_gpu},
        ),
    ]
    return [to_jsonable(candidate) for candidate in candidates]


def _verify_gpu_backend(root_dir: Path, requested_backend: str, hardware: JsonDict) -> JsonDict:
    selected_backend = _select_gpu_backend(requested_backend, hardware)
    if selected_backend is None:
        return _write_gpu_verification_artifact(
            root_dir,
            {
                "status": "blocked",
                "requested_backend": requested_backend,
                "selected_backend": None,
                "blocker_reason": "no_plausible_tailored_gpu_backend_detected",
                "missing_prerequisites": ["amd_or_nvidia_gpu"],
                "attempted_steps": [{"step": "select_backend", "status": "blocked"}],
            },
        )
    if selected_backend == "torch-rocm":
        torch_status = _torch_gpu_execution_status()
        verified = bool(torch_status.get("gpu_execution_verified"))
        return _write_gpu_verification_artifact(
            root_dir,
            {
                "status": "verified" if verified else "blocked",
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "verification_backend": "torch-rocm",
                "candidate_category": "generic_tensor_gpu",
                "accelerator_kind": "amd_gpu",
                "gpu_runtime_stack": "amd_rocm",
                "gpu_backend_verified": verified,
                "gpu_program_executed": verified,
                "gpu_device_name": torch_status.get("device_name"),
                "blocker_reason": None if verified else torch_status.get("reason"),
                "attempted_steps": [{"step": "torch_rocm_minimal_tensor_execution", "status": "passed" if verified else "blocked"}],
                "torch_status": torch_status,
            },
            artifact_name="torch_rocm_generic_full_state.json",
        )

    stack = "amd_rocm" if selected_backend == "quest-hip" else "nvidia_cuda"
    accelerator_kind = "amd_gpu" if selected_backend == "quest-hip" else "nvidia_gpu"
    prereq = _quest_gpu_prerequisites(selected_backend, hardware)
    attempted_steps = [{"step": "preflight", "status": "passed" if not prereq["missing"] else "blocked"}]
    if prereq["missing"]:
        return _write_gpu_verification_artifact(
            root_dir,
            {
                "status": "blocked",
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "verification_backend": selected_backend,
                "candidate_category": "tailored_quantum_gpu",
                "accelerator_kind": accelerator_kind,
                "gpu_runtime_stack": stack,
                "gpu_backend_verified": False,
                "gpu_program_executed": False,
                "blocker_reason": f"missing_prerequisites:{','.join(prereq['missing'])}",
                "missing_prerequisites": prereq["missing"],
                "detected_dependencies": prereq["dependencies"],
                "attempted_steps": attempted_steps,
            },
        )

    runner_root = root_dir / "native" / "quest_gpu"
    runner_bin_dir = root_dir / "build" / "native" / "quest_gpu" / ("hip" if selected_backend == "quest-hip" else "cuda") / "bin"
    runner = runner_bin_dir / "quest_gpu_runner"
    hip_smoke_payload: JsonDict | None = None
    if selected_backend == "quest-hip":
        smoke = runner_bin_dir / "hip_smoke"
        smoke_build_cmd = ["make", "clean-hip-smoke", "hip-smoke", "GPU_BACKEND=hip", "HIP_ARCHITECTURES=gfx1032"]
        smoke_build_result = _run_command(smoke_build_cmd, cwd=runner_root, timeout_s=120)
        attempted_steps.append(
            {
                "step": "build_hip_smoke",
                "status": "passed" if smoke_build_result["returncode"] == 0 else "failed",
                "command": smoke_build_cmd,
            }
        )
        if smoke_build_result["returncode"] != 0:
            return _write_gpu_verification_artifact(
                root_dir,
                {
                    "status": "failed",
                    "requested_backend": requested_backend,
                    "selected_backend": selected_backend,
                    "verification_backend": selected_backend,
                    "candidate_category": "tailored_quantum_gpu",
                    "accelerator_kind": accelerator_kind,
                    "gpu_runtime_stack": stack,
                    "gpu_backend_verified": False,
                    "gpu_program_executed": False,
                    "blocker_reason": "hip_smoke_build_failed",
                    "detected_dependencies": prereq["dependencies"],
                    "attempted_steps": attempted_steps,
                    "hip_smoke_build_result": smoke_build_result,
                },
            )
        smoke_run_cmd = [str(smoke)]
        smoke_run_result = _run_command(smoke_run_cmd, cwd=runner_root, timeout_s=30)
        hip_smoke_payload = _json_from_stdout(smoke_run_result.get("stdout") or "")
        smoke_verified = bool(
            smoke_run_result["returncode"] == 0
            and hip_smoke_payload
            and hip_smoke_payload.get("status") == "ok"
            and hip_smoke_payload.get("gpu_program_executed") is True
        )
        attempted_steps.append(
            {
                "step": "minimal_hip_smoke_run",
                "status": "passed" if smoke_verified else "failed",
                "command": smoke_run_cmd,
            }
        )
        if not smoke_verified:
            return _write_gpu_verification_artifact(
                root_dir,
                {
                    "status": "failed",
                    "requested_backend": requested_backend,
                    "selected_backend": selected_backend,
                    "verification_backend": selected_backend,
                    "candidate_category": "tailored_quantum_gpu",
                    "accelerator_kind": accelerator_kind,
                    "gpu_runtime_stack": stack,
                    "gpu_backend_verified": False,
                    "gpu_program_executed": False,
                    "gpu_device_name": _gpu_device_name_from_hip_smoke(hip_smoke_payload),
                    "blocker_reason": "hip_smoke_run_failed",
                    "detected_dependencies": prereq["dependencies"],
                    "attempted_steps": attempted_steps,
                    "hip_smoke_build_result": smoke_build_result,
                    "hip_smoke_run_result": smoke_run_result,
                    "hip_smoke_payload": hip_smoke_payload,
                },
            )

    build_cmd = ["make", "clean-all", "all", f"GPU_BACKEND={'hip' if selected_backend == 'quest-hip' else 'cuda'}"]
    if selected_backend == "quest-hip":
        build_cmd.append("HIP_ARCHITECTURES=gfx1032")
    build_result = _run_command(build_cmd, cwd=runner_root, timeout_s=300)
    attempted_steps.append({"step": "build_quest_gpu_runner", "status": "passed" if build_result["returncode"] == 0 else "failed", "command": build_cmd})
    if build_result["returncode"] != 0:
        return _write_gpu_verification_artifact(
            root_dir,
            {
                "status": "failed",
                "requested_backend": requested_backend,
                "selected_backend": selected_backend,
                "verification_backend": selected_backend,
                "candidate_category": "tailored_quantum_gpu",
                "accelerator_kind": accelerator_kind,
                "gpu_runtime_stack": stack,
                "gpu_backend_verified": False,
                "gpu_program_executed": False,
                "blocker_reason": "quest_gpu_build_failed",
                "detected_dependencies": prereq["dependencies"],
                "attempted_steps": attempted_steps,
                "hip_smoke_payload": hip_smoke_payload,
                "build_result": build_result,
            },
        )

    dump_path = root_dir / "build" / "gpu_verification" / "quest_gpu_minimal_state_dump.json"
    run_cmd = [str(runner), "--algo", "QRNG", "--qubits", "2", "--json", "--dump-state-json", str(dump_path), "--max-output-amplitudes", "4"]
    run_result = _run_command(run_cmd, cwd=runner_root, timeout_s=60)
    attempted_steps.append({"step": "minimal_quest_gpu_run", "status": "passed" if run_result["returncode"] == 0 else "failed", "command": run_cmd})
    payload = _json_from_stdout(run_result.get("stdout") or "")
    verified = bool(run_result["returncode"] == 0 and payload and payload.get("status") in {"ok", "passed"} and dump_path.exists())
    return _write_gpu_verification_artifact(
        root_dir,
        {
            "status": "verified" if verified else "failed",
            "requested_backend": requested_backend,
            "selected_backend": selected_backend,
            "verification_backend": selected_backend,
            "candidate_category": "tailored_quantum_gpu",
            "accelerator_kind": accelerator_kind,
            "gpu_runtime_stack": stack,
            "gpu_backend_verified": verified,
            "gpu_program_executed": verified,
            "gpu_device_name": _gpu_device_name(selected_backend) or _gpu_device_name_from_hip_smoke(hip_smoke_payload),
            "gpu_toolkit_metadata": prereq["dependencies"],
            "gpu_synchronized": True,
            "runner_path": str(runner),
            "runner_root": str(runner_root),
            "blocker_reason": None if verified else "quest_gpu_minimal_run_failed",
            "detected_dependencies": prereq["dependencies"],
            "attempted_steps": attempted_steps,
            "hip_smoke_payload": hip_smoke_payload,
            "build_result": build_result,
            "minimal_run_result": run_result,
        },
    )


def _select_gpu_backend(requested_backend: str, hardware: JsonDict) -> str | None:
    if requested_backend == "auto":
        if hardware.get("amd_gpu_pci_detected"):
            return "quest-hip"
        if hardware.get("nvidia_gpu_pci_detected"):
            return "quest-cuda"
        return None
    return requested_backend


def _quest_gpu_prerequisites(selected_backend: str, hardware: JsonDict) -> JsonDict:
    if selected_backend == "quest-hip":
        checks = {
            "amd_gpu_pci": {"available": bool(hardware.get("amd_gpu_pci_detected"))},
            "dev_kfd": {"available": bool(hardware.get("dev_kfd_present"))},
            "dev_dri": {"available": bool(hardware.get("dev_dri_present"))},
            "dev_dri_renderD128": {"available": bool(hardware.get("dev_dri_renderD128_present")), "required": False},
            "dev_dri_render_node": {
                "available": bool(hardware.get("dev_dri_render_node_present") or hardware.get("dev_dri_renderD128_present")),
                "nodes": hardware.get("dev_dri_render_nodes") or [],
            },
            "rocminfo_gpu_agent": {
                "available": bool(hardware.get("rocminfo_gpu_agent_detected")),
                "gfx_targets": hardware.get("rocminfo_gfx_targets") or [],
                "returncode": hardware.get("rocminfo_returncode"),
            },
            "hipcc": _command_status("hipcc"),
            "rocminfo": _command_status("rocminfo"),
            "cmake": _command_status("cmake"),
            "make": _command_status("make"),
        }
    else:
        checks = {
            "nvidia_gpu_pci": {"available": bool(hardware.get("nvidia_gpu_pci_detected"))},
            "nvidia-smi": _command_status("nvidia-smi"),
            "nvcc": _command_status("nvcc"),
            "cmake": _command_status("cmake"),
            "make": _command_status("make"),
        }
    missing = [name for name, status in checks.items() if status.get("required", True) and not status.get("available")]
    return {"dependencies": checks, "missing": missing}


def _run_command(cmd: list[str], *, cwd: Path, timeout_s: float) -> JsonDict:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout_s)
        return {
            "command": [str(part) for part in cmd],
            "cwd": str(cwd),
            "returncode": result.returncode,
            "stdout": _bounded(result.stdout),
            "stderr": _bounded(result.stderr),
            "timeout_s": timeout_s,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": [str(part) for part in cmd],
            "cwd": str(cwd),
            "returncode": None,
            "stdout": _bounded(exc.stdout or ""),
            "stderr": _bounded(exc.stderr or ""),
            "timeout_s": timeout_s,
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "command": [str(part) for part in cmd],
            "cwd": str(cwd),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timeout_s": timeout_s,
            "os_error": True,
        }


def _write_gpu_verification_artifact(root_dir: Path, payload: JsonDict, *, artifact_name: str | None = None) -> JsonDict:
    path = quest_gpu_verification_path(root_dir) if artifact_name is None else root_dir / "build" / "gpu_verification" / artifact_name
    path.parent.mkdir(parents=True, exist_ok=True)
    final_payload = {
        "schema_version": QUEST_GPU_VERIFICATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(path),
        **payload,
    }
    path.write_text(json.dumps(to_jsonable(final_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return to_jsonable(final_payload)


def _json_from_stdout(stdout: str) -> JsonDict | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def _gpu_device_name(selected_backend: str) -> str | None:
    if selected_backend == "quest-hip":
        result = _run_command(["rocm-smi", "--showproductname"], cwd=Path.cwd(), timeout_s=10)
        if result.get("returncode") == 0 and result.get("stdout"):
            lines = [line.strip() for line in str(result["stdout"]).splitlines()]
            card = next((line.split("Card Series:", 1)[1].strip() for line in lines if "Card Series:" in line), "")
            gfx = next((line.split("GFX Version:", 1)[1].strip() for line in lines if "GFX Version:" in line), "")
            if card and gfx:
                return f"{card} ({gfx})"
            return card or gfx or None
    if selected_backend == "quest-cuda":
        result = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], cwd=Path.cwd(), timeout_s=10)
        if result.get("returncode") == 0 and result.get("stdout"):
            return str(result["stdout"]).splitlines()[0].strip() or None
    return None


def _gpu_device_name_from_hip_smoke(payload: JsonDict | None) -> str | None:
    if not payload:
        return None
    device_name = str(payload.get("gpu_device_name") or "").strip()
    arch = str(payload.get("gcn_arch_name") or "").strip()
    if device_name and arch:
        return f"{device_name} ({arch})"
    return device_name or arch or None


def _bounded(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


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
    benchmark_route_eligible: bool | None = None,
    extra: JsonDict | None = None,
) -> JsonDict:
    return {
        "candidate_id": candidate_id,
        "candidate_category": category,
        "preferred_priority": priority,
        "target_gpu_stack": stack,
        "classification": "usable_current_machine" if usable_current and gpu_execution_verified else classification,
        "benchmark_route_eligible": bool(gpu_execution_verified if benchmark_route_eligible is None else benchmark_route_eligible),
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
