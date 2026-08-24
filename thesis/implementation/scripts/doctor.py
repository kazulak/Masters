from __future__ import annotations

# Lightweight prerequisite check for thesis evidence commands. It does not run
# benchmarks and must not emit benchmark records.

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum_bench_mplconfig")
    print("Thesis benchmark doctor")
    _check_python()
    _check_import("quantum_bench", required=True)
    for module in ("numpy", "opt_einsum", "cotengra", "quimb", "yaml", "matplotlib"):
        _check_import(module, required=False)
    _check_cpu_benchmark_profile()
    _check_quest_cpu()
    _check_rocm()
    _check_upmem_sdk()
    return 0


def _check_python() -> None:
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info >= (3, 10):
        _line("PASS", "python", version)
    else:
        _line("BLOCKED", "python", f"{version}; Python >= 3.10 is required")


def _check_import(module: str, *, required: bool) -> None:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - exact import errors vary
        status = "BLOCKED" if required else "WARN"
        _line(status, f"dependency:{module}", str(exc))
        return
    version = getattr(imported, "__version__", "available")
    _line("PASS", f"dependency:{module}", str(version))


def _check_quest_cpu() -> None:
    runner = ROOT / "native" / "quest_cpu" / "bin" / "quest_runner"
    if runner.exists():
        _line("PASS", "quest_cpu", str(runner.relative_to(ROOT)))
    else:
        _line("BLOCKED", "quest_cpu", "native runner missing; run `make build-quest-cpu`")


def _check_cpu_benchmark_profile() -> None:
    physical = _read_physical_cpu_count()
    logical = os.cpu_count()
    affinity = _read_cpu_affinity()
    governor = _read_cpu_frequency_governor()
    configured = os.environ.get("BENCH_CPU_THREADS")
    details = (
        f"physical={physical or 'unknown'}; logical={logical or 'unknown'}; "
        f"affinity={','.join(str(cpu) for cpu in affinity) if affinity else 'unknown'}; "
        f"governor={governor or 'unknown'}; BENCH_CPU_THREADS={configured or 'unset'}"
    )
    _line("PASS" if configured else "WARN", "cpu_benchmark_profile", details)
    if physical:
        _line("PASS", "recommended_thesis_run", "make run CONFIG=configs/tn_benchmark_reset.yml")


def _check_rocm() -> None:
    hipcc = shutil.which("hipcc")
    rocminfo = shutil.which("rocminfo")
    kfd = Path("/dev/kfd").exists()
    dri = Path("/dev/dri").exists()
    device = _detect_rocm_device(rocminfo)
    details = [
        f"hipcc={hipcc or 'missing'}",
        f"rocminfo={rocminfo or 'missing'}",
        f"/dev/kfd={'yes' if kfd else 'no'}",
        f"/dev/dri={'yes' if dri else 'no'}",
        f"device={device or 'not_detected'}",
    ]
    if hipcc and rocminfo and kfd and dri and device:
        _line("PASS", "gpu_rocm", "; ".join(details))
    else:
        _line("WARN", "gpu_rocm", "; ".join(details))


def _detect_rocm_device(rocminfo: str | None) -> str | None:
    if not rocminfo:
        return None
    try:
        result = subprocess.run([rocminfo], text=True, capture_output=True, check=False, timeout=10)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "Name:" in line and ("gfx" in line or "AMD" in line or "Radeon" in line):
            return " ".join(line.split())
    return None


def _check_upmem_sdk() -> None:
    clang = shutil.which("dpu-upmem-dpurte-clang")
    pkg_config = shutil.which("dpu-pkg-config")
    make = shutil.which("make")
    details = [
        f"dpu-upmem-dpurte-clang={clang or 'missing'}",
        f"dpu-pkg-config={pkg_config or 'missing'}",
        f"make={make or 'missing'}",
    ]
    if clang and pkg_config and make:
        _line("PASS", "upmem_sdk", "; ".join(details))
    else:
        _line("WARN", "upmem_sdk", "; ".join(details))


def _line(status: str, name: str, details: str) -> None:
    print(f"{status} {name}: {details}")


def _read_cpu_affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(os.sched_getaffinity(0))
    except OSError:
        return None


def _read_cpu_frequency_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _read_physical_cpu_count() -> int | None:
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
    return len(packages_and_cores) if packages_and_cores else os.cpu_count()


if __name__ == "__main__":
    raise SystemExit(main())
