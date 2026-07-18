from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

import quantum_bench


RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")


def capture_environment(root_dir: Path) -> dict[str, Any]:
    thread_variables = (
        "BENCH_CPU_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_PROC_BIND",
        "OMP_PLACES",
    )
    return {
        "quantum_bench_version": quantum_bench.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": read_physical_cpu_count(),
        "cpu_affinity": read_cpu_affinity(),
        "cpu_frequency_governor": read_cpu_frequency_governor(),
        "mem_total_kib": read_mem_total_kib(),
        "numpy": np.__version__,
        "opt_einsum": _module_version("opt_einsum"),
        "quimb": _module_version("quimb"),
        "cotengra": _module_version("cotengra"),
        "matplotlib": _module_version("matplotlib"),
        "compiler": first_line(["cc", "--version"]),
        "git_commit": first_line(["git", "rev-parse", "HEAD"], cwd=root_dir.parents[1]) if (root_dir.parents[1] / ".git").exists() else None,
        "benchmark_threads": {key: os.environ.get(key) for key in thread_variables},
        "openmp": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")},
        "blas": read_numpy_blas_config(),
        "upmem": {
            "UPMEM_HOME": os.environ.get("UPMEM_HOME"),
            "dpu_compiler": shutil.which("dpu-upmem-dpurte-clang"),
        },
        "gpu": {
            "rocminfo": shutil.which("rocminfo"),
            "rocm_smi": shutil.which("rocm-smi"),
            "amd_smi": shutil.which("amd-smi"),
            "hipcc": shutil.which("hipcc"),
            "cupy": _module_version("cupy"),
            "torch": _module_version("torch"),
            "jax": _module_version("jax"),
            "gpu_execution_backend_added": False,
        },
        "simplepim": {
            "SIMPLEPIM_HOME": os.environ.get("SIMPLEPIM_HOME"),
            "SIMPLEPIM_BIN": os.environ.get("SIMPLEPIM_BIN"),
            "SIMPLEPIM_LIB": os.environ.get("SIMPLEPIM_LIB"),
            "command_path": None,
            "available": False,
            "probe_status": "retired",
            "skip_reason": "SimplePIM probing is retired; the generic SDK loop is the active UPMEM simulator path",
        },
        "rapl": {
            "path": str(RAPL_PATH),
            "available": RAPL_PATH.is_readable() if hasattr(RAPL_PATH, "is_readable") else os.access(RAPL_PATH, os.R_OK),
            "powercap_energy_uj_paths": _powercap_energy_paths(),
        },
        "sudo": {
            "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
            "sudo_uid": os.environ.get("SUDO_UID"),
            "sudo_gid": os.environ.get("SUDO_GID"),
        },
    }


def read_rapl_uj() -> int | None:
    try:
        return int(RAPL_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _powercap_energy_paths() -> list[str]:
    root = Path("/sys/class/powercap")
    try:
        return sorted(str(path) for path in root.glob("**/energy_uj") if os.access(path, os.R_OK))
    except OSError:
        return []


def read_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def read_mem_total_kib() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def read_cpu_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return None


def read_cpu_frequency_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def read_physical_cpu_count() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    packages_and_cores: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    try:
        lines = cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in [*lines, ""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                packages_and_cores.add((current["physical id"], current["core id"]))
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    if packages_and_cores:
        return len(packages_and_cores)
    return os.cpu_count()


def read_numpy_blas_config() -> dict[str, Any]:
    try:
        config = np.show_config(mode="dicts")
    except (TypeError, AttributeError):
        return {"name": None, "version": None, "configuration": None}
    dependencies = config.get("Build Dependencies", {}) if isinstance(config, dict) else {}
    blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
    if not isinstance(blas, dict):
        return {"name": None, "version": None, "configuration": None}
    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "configuration": blas.get("openblas configuration"),
    }


def first_line(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else None


def _module_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")
