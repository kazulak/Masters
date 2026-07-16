from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from quantum_bench.bench.upmem_hardware_generic_mvp import (
    _prepare_case_bridge,
    prepare_upmem_hardware_generic_mvp,
    validate_hardware_generic_execution_request,
)
from quantum_bench.targets.upmem.generic_bridge import (
    GENERIC_BRIDGE_ID,
    GENERIC_BRIDGE_SCHEMA_VERSION,
    execute_generic_bridge,
)
import quantum_bench.targets.upmem.generic_bridge as generic_bridge_module
from quantum_bench.targets.upmem.hardware_generic_mvp import (
    HARDWARE_GENERIC_MVP_BACKEND_ID,
    load_hardware_generic_mvp_suite,
    validate_hardware_generic_mvp_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_generic_mvp.yml"


def _runner_module():
    path = ROOT / "native" / "upmem" / "simplepim" / "upmem_sdk_generic_loop_runner.py"
    spec = importlib.util.spec_from_file_location("generic_hardware_runner_for_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_hardware_suite_is_fixed_and_taskgraph_shaped() -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)

    assert suite.suite_id == "upmem_hardware_generic_mvp"
    assert [case.case_id for case in suite.cases] == ["generic_real_abc_cde_2"]
    assert suite.profile.requested_dpu_count == 1
    assert suite.profile.tasklets_per_dpu == 1
    assert suite.profile.output_tile_elements == 8
    assert suite.cases[0].output_shape == (2, 2, 2, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_dpu_count", 2),
        ("tasklets_per_dpu", 2),
        ("max_rank", 5),
        ("max_tensor_elements", 17),
        ("output_tile_elements", 16),
        ("repetitions", 6),
    ],
)
def test_generic_hardware_suite_rejects_profile_expansion(tmp_path: Path, field: str, value: object) -> None:
    payload = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    payload["profile"][field] = value
    path = tmp_path / "expanded.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        load_hardware_generic_mvp_suite(path)


def test_generic_hardware_manifest_is_fixed_and_preparation_is_nonexecuting(tmp_path: Path) -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)
    _prepare_case_bridge(suite.cases[0], tmp_path / "bridge", suite)
    payload = json.loads((tmp_path / "bridge" / "input_manifest.json").read_text(encoding="utf-8"))
    validate_hardware_generic_mvp_manifest(payload, profile=suite.profile)
    assert payload["native_index_metadata"]["generic_output_tile_count"] == 2

    payload["native_index_metadata"]["output_element_count"] = 17
    with pytest.raises(ValueError, match="hardware_profile_violation"):
        validate_hardware_generic_mvp_manifest(payload, profile=suite.profile)

    plan = prepare_upmem_hardware_generic_mvp(tmp_path, suite_path=SUITE_PATH, build=False, environment={})
    summary = json.loads(plan.summary_path.read_text(encoding="utf-8"))
    assert plan.status == "prepared"
    assert summary["dpu_allocation_attempted"] is False
    assert summary["dpu_launch_attempted"] is False


def test_generic_hardware_execution_fails_closed_without_opt_in(tmp_path: Path) -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)
    bridge = tmp_path / "bridge"
    _prepare_case_bridge(suite.cases[0], bridge, suite)

    result = execute_generic_bridge(
        bridge / "input_manifest.json",
        backend=HARDWARE_GENERIC_MVP_BACKEND_ID,
        execute_external=True,
        env={},
    )

    assert result.execution_status == "failed"
    assert result.reason == "hardware_opt_in_missing"
    assert result.external_command_executed is False


def test_generic_hardware_bridge_sanitizes_simulator_and_requires_exact_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)
    bridge = tmp_path / "bridge"
    _prepare_case_bridge(suite.cases[0], bridge, suite)
    output_path = bridge / "outputs" / "upmem_sdk_hardware_generic_loop_output.npy"
    output_path.parent.mkdir(exist_ok=True)
    expected = np.load(bridge / "references" / "expected_quantized_reference_output.npy", allow_pickle=False)
    np.save(output_path, expected, allow_pickle=False)
    captured: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command: object, **kwargs: object) -> Completed:
        captured["command"] = command
        captured["env"] = kwargs.get("env")
        captured["timeout"] = kwargs.get("timeout")
        manifest = {
            "schema_version": GENERIC_BRIDGE_SCHEMA_VERSION,
            "bridge_id": GENERIC_BRIDGE_ID,
            "manifest_kind": "generic_contraction_bridge_output",
            "backend": HARDWARE_GENERIC_MVP_BACKEND_ID,
            "status": "upmem_sdk_hardware_generic_loop_executed",
            "input_manifest": "input_manifest.json",
            "route_id": suite.profile.route_id,
            "task_id": "task",
            "output_blob": {
                "relative_path": "outputs/upmem_sdk_hardware_generic_loop_output.npy",
                "dtype": str(expected.dtype),
                "shape": [2, 2, 2, 2],
                "representation": "hardware_output",
                "nbytes": int(expected.nbytes),
                "role": "generic_loop_output",
            },
            "validation_metrics": {"exact_integer_passed": True, "passed": True, "max_abs_error": 0.0},
            "compute_time_s": 0.0,
            "write_time_s": 0.0,
            "total_time_s": 0.0,
            "external_command_executed": True,
            "execution_implemented": True,
            "error": None,
            "metadata": {
                "hardware_status_json": {
                    "success": True,
                    "failure_stage": None,
                    "allocation_profile": "backend=hw",
                    "requested_dpus": 1,
                    "allocated_dpus": 1,
                    "tasklets": 1,
                },
                "hardware_kernel_executed": True,
                "native_kernel_executed": True,
                "simulator_kernel_executed": False,
                "cpu_fallback_used": False,
                "application_visible_transfer_bytes": {"h2d": 80, "d2h": 64, "total": 144},
            },
        }
        (bridge / "output_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return Completed()

    monkeypatch.setattr(generic_bridge_module.subprocess, "run", fake_run)
    result = execute_generic_bridge(
        bridge / "input_manifest.json",
        backend=HARDWARE_GENERIC_MVP_BACKEND_ID,
        execute_external=True,
        env={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "UPMEM_PROFILE": "backend=simulator", "UPMEM_PROFILE_BASE": "backend=hw"},
    )

    assert result.execution_status == "upmem_sdk_hardware_generic_loop_executed"
    assert "--target" in captured["command"]
    assert "hardware" in captured["command"]
    assert "--timeout-seconds" in captured["command"]
    assert "30" in captured["command"]
    assert captured.get("timeout") == 65.0
    assert isinstance(captured["env"], dict)
    assert "DPU_BACKEND" not in captured["env"]
    assert "UPMEM_PROFILE" not in captured["env"]

    invalid = json.loads((bridge / "output_manifest.json").read_text(encoding="utf-8"))
    invalid["metadata"]["hardware_status_json"]["allocated_dpus"] = 2
    (bridge / "output_manifest.json").write_text(json.dumps(invalid), encoding="utf-8")

    def fake_noop_run(command: object, **kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(generic_bridge_module.subprocess, "run", fake_noop_run)
    invalid_result = execute_generic_bridge(
        bridge / "input_manifest.json",
        backend=HARDWARE_GENERIC_MVP_BACKEND_ID,
        execute_external=True,
        env={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
    )
    assert invalid_result.execution_status == "failed"
    assert invalid_result.reason == "output_validation_failed"


def test_runner_hardware_branch_never_falls_back_without_opt_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)
    bridge = tmp_path / "bridge"
    _prepare_case_bridge(suite.cases[0], bridge, suite)
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    monkeypatch.delenv("DPU_BACKEND", raising=False)
    runner = _runner_module()

    rc = runner.main(
        [
            "--input-manifest", str(bridge / "input_manifest.json"),
            "--output-manifest", str(bridge / "output_manifest.json"),
            "--backend-id", HARDWARE_GENERIC_MVP_BACKEND_ID,
            "--target", "hardware",
        ]
    )

    output = json.loads((bridge / "output_manifest.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert output["metadata"]["failure_stage"] == "hardware_opt_in_missing"
    assert output["metadata"]["simulator_kernel_executed"] is False
    assert output["metadata"]["cpu_fallback_used"] is False


def test_runner_hardware_branch_rejects_alternate_native_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    suite = load_hardware_generic_mvp_suite(SUITE_PATH)
    bridge = tmp_path / "bridge"
    _prepare_case_bridge(suite.cases[0], bridge, suite)
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.delenv("DPU_BACKEND", raising=False)
    runner = _runner_module()

    rc = runner.main(
        [
            "--input-manifest", str(bridge / "input_manifest.json"),
            "--output-manifest", str(bridge / "output_manifest.json"),
            "--backend-id", HARDWARE_GENERIC_MVP_BACKEND_ID,
            "--target", "hardware",
            "--source-dir", str(tmp_path),
        ]
    )

    output = json.loads((bridge / "output_manifest.json").read_text(encoding="utf-8"))
    assert rc == 1
    assert output["metadata"]["failure_stage"] == "hardware_profile_violation"
    assert output["metadata"]["hardware_kernel_executed"] is False


def test_generic_hardware_request_rejects_missing_opt_in_and_simulator_selector() -> None:
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        validate_hardware_generic_execution_request(execute=True, environment={})
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        validate_hardware_generic_execution_request(
            execute=True,
            environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "DPU_BACKEND": "simulator"},
        )
