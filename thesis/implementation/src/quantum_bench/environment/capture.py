from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

import quantum_bench


RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")


def capture_environment(root_dir: Path) -> dict[str, Any]:
    return {
        "quantum_bench_version": quantum_bench.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "mem_total_kib": read_mem_total_kib(),
        "numpy": np.__version__,
        "opt_einsum": _module_version("opt_einsum"),
        "matplotlib": _module_version("matplotlib"),
        "compiler": first_line(["cc", "--version"]),
        "git_commit": first_line(["git", "rev-parse", "HEAD"], cwd=root_dir.parents[1]) if (root_dir.parents[1] / ".git").exists() else None,
        "openmp": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
        "upmem": {
            "UPMEM_HOME": os.environ.get("UPMEM_HOME"),
            "dpu_compiler": shutil.which("dpu-upmem-dpurte-clang"),
        },
        "rapl": {
            "path": str(RAPL_PATH),
            "available": RAPL_PATH.is_readable() if hasattr(RAPL_PATH, "is_readable") else os.access(RAPL_PATH, os.R_OK),
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
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")
