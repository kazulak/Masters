from __future__ import annotations

from datetime import datetime
from pathlib import Path

from quantum_bench.bench.run_dirs import sanitize, update_latest_symlink
from quantum_bench.core.jsonio import write_json
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem import SimplePimDenseMicrobenchInput, prepare_simplepim_dense_microbench


def run_simplepim_microbench(
    root_dir: Path,
    *,
    gemm_m: int,
    gemm_k: int,
    gemm_n: int,
    route_dtype: str = "int8",
    source_dtype: str = "float32",
    seed: int = 0,
    dry_run: bool = True,
) -> tuple[Path, Path, str]:
    run_dir = _create_microbench_run_dir(root_dir)
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    microbench_input = SimplePimDenseMicrobenchInput(
        gemm_m=gemm_m,
        gemm_k=gemm_k,
        gemm_n=gemm_n,
        route_dtype=route_dtype,
        source_dtype=source_dtype,
        seed=seed,
        dry_run=dry_run,
    )
    result = prepare_simplepim_dense_microbench(microbench_input, execute=not dry_run)
    artifact_path = run_dir / "simplepim_microbench.json"
    write_json(artifact_path, result)
    return run_dir, artifact_path, result.status


def _create_microbench_run_dir(root_dir: Path) -> Path:
    runs_dir = root_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stamp}_{sanitize('simplepim_microbench')}"
    run_dir = runs_dir / base
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{base}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    update_latest_symlink(runs_dir, run_dir)
    return run_dir
