from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quantum_bench.targets.upmem.hardware_taskgraph import (
    HARDWARE_TASKGRAPH_BACKEND_ID,
    HARDWARE_TASKGRAPH_ROUTE_ID,
    hardware_taskgraph_profile_metadata,
    load_hardware_taskgraph_suite,
    validate_hardware_taskgraph_execution_request,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "suites" / "upmem_hardware_taskgraph_correctness.yml"


def test_hardware_taskgraph_suite_is_single_dpu_and_covers_both_numeric_paths() -> None:
    loaded = load_hardware_taskgraph_suite(SUITE_PATH)

    assert loaded.profile.backend_id == HARDWARE_TASKGRAPH_BACKEND_ID
    assert loaded.profile.route_id == HARDWARE_TASKGRAPH_ROUTE_ID
    assert loaded.profile.requested_dpu_count == 1
    assert loaded.profile.tasklets_per_dpu == 1
    assert loaded.profile.numeric_modes == ("none", "per_task_input_quantize")
    assert {case["hardware_numeric_coverage"] for case in loaded.suite["cases"]} == {
        "real",
        "split_complex",
    }
    metadata = hardware_taskgraph_profile_metadata(loaded.profile)
    assert metadata["native_build_reuse_required"] is True
    assert metadata["logical_task_session_only"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_dpu_count", 2),
        ("tasklets_per_dpu", 2),
        ("max_tensor_elements", 257),
        ("numeric_modes", ["none"]),
        ("performance_claim_applicable", True),
    ],
)
def test_hardware_taskgraph_profile_rejects_expansion(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    payload["metadata"]["hardware_profile"][field] = value
    path = tmp_path / "expanded.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        load_hardware_taskgraph_suite(path)


def test_hardware_taskgraph_execution_requires_explicit_opt_in_and_no_simulator_selector() -> (
    None
):
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        validate_hardware_taskgraph_execution_request(execute=True, environment={})
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        validate_hardware_taskgraph_execution_request(
            execute=True,
            environment={
                "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                "DPU_BACKEND": "simulator",
            },
        )
