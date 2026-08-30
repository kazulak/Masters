from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "inspect_circuit_resource_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("inspect_circuit_resource_sensitivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inspector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspector
SPEC.loader.exec_module(inspector)


def _stats(total: float, kernel: float, *, dpus: int, tasklets: int) -> dict[str, object]:
    return {
        "median_total_wall_s": total,
        "median_kernel_s": kernel,
        "dpu_count": dpus,
        "tasklets_per_dpu": tasklets,
    }


def test_configuration_matches_preregistered_matrix() -> None:
    config = load_experiment_config(ROOT / "configs" / "tn_benchmark_circuit_resource_sensitivity_diagnostic.yml")
    selection = json.loads((ROOT / "configs" / "circuit_resource_sensitivity_selection.json").read_text(encoding="utf-8"))
    inspector._validate_configuration(config, selection)


def test_comparisons_are_within_circuit_and_separate_axes() -> None:
    stats = {
        "upmem_float32_1dpu_t1": _stats(30.0, 24.0, dpus=1, tasklets=1),
        "upmem_float32_1dpu_t4": _stats(12.0, 6.0, dpus=1, tasklets=4),
        "upmem_float32_1dpu_t8": _stats(9.0, 3.0, dpus=1, tasklets=8),
        "upmem_float32_1dpu_t12": _stats(8.0, 2.5, dpus=1, tasklets=12),
        "upmem_float32_2dpu_t8": _stats(6.0, 1.6, dpus=2, tasklets=8),
        "upmem_float32_3dpu_t8": _stats(5.5, 1.2, dpus=3, tasklets=8),
        "upmem_float32_4dpu_t8": _stats(5.0, 0.9, dpus=4, tasklets=8),
    }
    rows = inspector._comparisons(stats)
    assert len(rows) == 6
    assert [row["comparison_kind"] for row in rows[:3]] == ["tasklet"] * 3
    assert [row["comparison_kind"] for row in rows[3:]] == ["dpu"] * 3
    assert rows[0]["candidate_route"] == "upmem_float32_1dpu_t4"
    assert rows[3]["baseline_route"] == "upmem_float32_1dpu_t8"
    assert rows[3]["kernel_speedup"] == 3.0 / 1.6
    assert all(row["diagnostic_only"] is True for row in rows)
