from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

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


def _component_row(total: float, kernel: float) -> dict[str, float]:
    row = {field: 0.0 for field in inspector.DISJOINT_COMPONENT_FIELDS}
    row.update(
        {
            "total_wall_s": total,
            "kernel_s": kernel,
            "request_build_parent_s": 1.0,
            "work_unit_materialization_s": 0.2,
            "payload_record_staging_s": 0.3,
            "manifest_sidecar_staging_s": 0.1,
            "artifact_build_residual_s": 0.2,
            "request_build_residual_s": 0.2,
        }
    )
    return row


def test_route_statistics_exclude_warmup_from_derived_values() -> None:
    samples = [
        {"attempt_kind": "warmup", "measurement": {"total_wall_s": 100.0, "kernel_s": 90.0, "h2d_s": 1.0, "d2h_s": 1.0, "h2d_bytes": 1, "d2h_bytes": 1}, "validation": {"max_abs_error": 1.0, "relative_l2_error": 1.0, "norm_drift": 1.0}},
        {"attempt_kind": "measurement", "measurement": {"total_wall_s": 10.0, "kernel_s": 2.0, "h2d_s": 2.0, "d2h_s": 3.0, "h2d_bytes": 2, "d2h_bytes": 3}, "validation": {"max_abs_error": 0.1, "relative_l2_error": 0.1, "norm_drift": 0.1}},
    ]
    facts = [
        {"arithmetic_weighted_tasklet_utilization": 0.1, "arithmetic_weighted_dpu_slot_utilization": 0.1, "dominant_work_wave_utilization": 0.1},
        {"arithmetic_weighted_tasklet_utilization": 0.9, "arithmetic_weighted_dpu_slot_utilization": 0.8, "dominant_work_wave_utilization": 0.7},
    ]
    components = [_component_row(100.0, 90.0), _component_row(10.0, 2.0)]
    result = inspector._route_statistics(samples, facts, components)
    assert result["measurement_count"] == 1
    assert result["median_total_wall_s"] == 10.0
    assert result["median_kernel_s"] == 2.0
    assert result["tasklet_utilization"] == 0.9
    assert result["component_medians_s"]["kernel_s"] == 2.0


def test_terminal_authority_conflicts_are_rejected() -> None:
    sample = {"backend_facts": {"physical_target_verified": True}}
    sessions = {"s": {"terminal_backend_facts": {"physical_target_verified": False}}}
    with pytest.raises(ValueError, match="terminal physical facts conflict"):
        inspector._joined_facts({**sample, "session_instance_id": "s"}, sessions)


def test_session_references_are_bijective() -> None:
    samples = [
        {"session_instance_id": "s1", "case_id": "case", "route_id": "route"},
        {"session_instance_id": "s1", "case_id": "case", "route_id": "route"},
    ]
    sessions = {"s1": {"case_id": "case", "route_id": "route"}, "s2": {"case_id": "case", "route_id": "route"}}
    with pytest.raises(ValueError, match="bijection"):
        inspector._validate_session_bijection(samples, sessions, 2)
