from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from quantum_bench.targets.upmem.hardware_taskgraph_study import (
    HARDWARE_TASKGRAPH_STUDY_BACKEND_ID,
    HARDWARE_TASKGRAPH_STUDY_ROUTE_ID,
    hardware_taskgraph_study_profile_metadata,
    load_hardware_taskgraph_study_suite,
    validate_hardware_taskgraph_study_execution_request,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs/suites/upmem_hardware_taskgraph_path_quantization.yml"


def test_one_dpu_study_suite_is_fixed_and_has_two_distinct_planners() -> None:
    loaded = load_hardware_taskgraph_study_suite(SUITE_PATH)

    assert loaded.profile.backend_id == HARDWARE_TASKGRAPH_STUDY_BACKEND_ID
    assert loaded.profile.route_id == HARDWARE_TASKGRAPH_STUDY_ROUTE_ID
    assert loaded.profile.requested_dpu_count == 1
    assert loaded.profile.tasklets_per_dpu == 1
    assert loaded.suite["warmups"] == 2
    assert loaded.suite["repeats"] == 7
    assert len(loaded.suite["cases"]) == 13
    assert [variant.variant_id for variant in loaded.variants] == [
        "opt_einsum_greedy",
        "custom_upmem_v2_balanced",
    ]
    metadata = hardware_taskgraph_study_profile_metadata(loaded.profile)
    assert metadata["session_scope"] == "case_benchmark_block"
    assert metadata["hardware_speedup_applicable"] is False
    assert metadata["multi_dpu_execution"] is False


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("metadata", "hardware_profile", "requested_dpu_count"), 2),
        (("metadata", "hardware_profile", "tasklets_per_dpu"), 2),
        (("defaults", "warmups"), 1),
        (("defaults", "repeats"), 5),
    ],
)
def test_one_dpu_study_rejects_profile_or_measurement_expansion(
    tmp_path: Path, location: tuple[str, ...], value: object
) -> None:
    payload = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    target = payload
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    path = tmp_path / "invalid.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        load_hardware_taskgraph_study_suite(path)


def test_one_dpu_study_rejects_mutated_path_variant(tmp_path: Path) -> None:
    payload = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    payload["path_variants"][1]["planner"]["weight_profile"] = "compute_oriented"
    path = tmp_path / "invalid_variant.yml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hardware_profile_violation"):
        load_hardware_taskgraph_study_suite(path)


def test_one_dpu_study_requires_opt_in_and_rejects_simulator_selector() -> None:
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE=1"):
        validate_hardware_taskgraph_study_execution_request(
            execute=True, environment={}
        )
    with pytest.raises(ValueError, match="DPU_BACKEND"):
        validate_hardware_taskgraph_study_execution_request(
            execute=True,
            environment={
                "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1",
                "DPU_BACKEND": "simulator",
            },
        )
