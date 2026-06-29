from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_generic_task_bridge_prepares_real_task_and_skips_when_external_disabled(tmp_path: Path) -> None:
    result = run_generic_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=0,
        backend="upmem_sdk_simulator_generic_loop",
        execute_external=False,
    )
    summary = _load(result.summary_path)
    encoded = json.dumps(summary)

    assert result.status == "skipped"
    assert result.reason == "generic_external_execution_disabled"
    assert summary["schema_version"] == "generic_task_bridge_v1"
    assert summary["route_id"] == "generic_loop_fallback"
    assert summary["kernel_family"] == "generic_loop_fallback"
    assert summary["execution_target"] == "upmem_simulator"
    assert summary["execution_scope"] == "task_level"
    assert summary["bridge_execution_status"] == "not_implemented"
    assert summary["external_command_executed"] is False
    assert summary["execution_implemented"] is False
    assert summary["metadata"]["validation_target"] == "expected_quantized_reference_output"
    assert summary["metadata"]["full_precision_reference_is_validation_target"] is False
    assert summary["normalized_result"]["kernel_family"] == "generic_loop_fallback"
    assert summary["normalized_result"]["hardware_speedup"] == "not_applicable"
    assert (result.run_dir / "bridge" / "input_manifest.json").exists()
    assert (result.run_dir / "bridge" / "output_manifest.json").exists()
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded


def test_generic_task_bridge_rejects_out_of_range_task_index(tmp_path: Path) -> None:
    result = run_generic_task_bridge(tmp_path, case="bell_2q", task_index=999, execute_external=False)
    summary = _load(result.summary_path)

    assert result.status == "unsupported"
    assert result.reason == "target_task_index_out_of_range"
    assert summary["normalized_result"]["kernel_family"] == "generic_loop_fallback"
    assert summary["normalized_result"]["unsupported_task_count"] == 1
