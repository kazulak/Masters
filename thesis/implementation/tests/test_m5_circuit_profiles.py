from __future__ import annotations

from pathlib import Path

import pytest

from quantum_bench.bench.m5_circuit_study import (
    _build_plans,
    load_study_config,
    plan_study,
)
from quantum_bench.lowering import contraction_dag_hash


ROOT = Path(__file__).parents[1]
SUITES = ROOT / "configs" / "suites"
PROFILE_NAMES = (
    "m5_circuit_smoke.yml",
    "m5_circuit_canonical.yml",
    "m5_circuit_large.yml",
    "m5_circuit_scaling.yml",
)


def _config(name: str) -> dict:
    return load_study_config(SUITES / name)


def _physical_engines(config: dict) -> list[dict]:
    return [
        item
        for item in config["engine_variants"]
        if item["topology"]["backend"] == "upmem"
    ]


def _requested_dpus(engine: dict) -> int:
    return len(engine["topology"]["device_ids"])


def _assert_physical_contract(engine: dict) -> None:
    assert engine["engine"] == "m5_whole_circuit_v4"
    assert engine["timeout_enforcement"] == "engine_subprocess"
    assert engine["topology"]["tasklets_per_device"] == 1
    assert engine["executor_config"] == {
        "profile": "m5_whole_circuit_v4_v1",
        "abi": "execution_plan_v4",
        "session_protocol": "persistent_rank_session_v1",
        "dispatch_mode": "bulk_set_synchronous_v1",
        "kernel_identity": "dpu_gemm_tile_v4",
        "execution_class": "physical_v4_output_tile",
    }


def test_all_m5_profiles_load_and_use_explicit_unique_case_ids() -> None:
    for name in PROFILE_NAMES:
        config = _config(name)
        case_ids = [case["case_id"] for case in config["cases"]]
        assert case_ids
        assert len(case_ids) == len(set(case_ids)), name
        assert config["schema_version"] == "m5_circuit_study_v2"
        assert config["metadata"]["physical_rank_paths_are_placeholders"] is True
        for engine in _physical_engines(config):
            _assert_physical_contract(engine)


def test_smoke_profile_is_small_and_plan_only(tmp_path: Path) -> None:
    config = _config("m5_circuit_smoke.yml")
    assert len(config["cases"]) == 1
    assert config["cases"][0]["circuit"] == {
        "kind": "quest_compatible",
        "name": "BV",
        "n_qubits": 4,
    }
    assert [item["planner"]["optimize"] for item in config["planner_variants"]] == [
        "greedy"
    ]
    assert {item["policy"] for item in config["numeric_policies"]} == {
        "float32_real",
        "host_packed_int8",
    }
    physical = _physical_engines(config)
    assert len(physical) == 1
    assert physical[0]["topology"]["rank_paths"] == ["/dev/dpu_rank0"]
    assert _requested_dpus(physical[0]) == 1

    plan_path = plan_study(ROOT, SUITES / "m5_circuit_smoke.yml")
    assert plan_path.is_file()


def test_canonical_profile_has_exact_42_cases_and_requested_matrix() -> None:
    config = _config("m5_circuit_canonical.yml")
    assert len(config["cases"]) == 42
    expected_families = {"QRNG", "BV", "XOR", "BB84", "EDC", "HS"}
    expected_sizes = {8, 10, 12, 14, 16, 18, 20}
    for family in expected_families:
        cases = [case for case in config["cases"] if case["family"] == family]
        assert len(cases) == 7
        sizes = {
            case["circuit"].get("allocated_qubits", case["circuit"].get("n_qubits"))
            for case in cases
        }
        assert sizes == expected_sizes
        if family == "HS":
            assert all(
                case["circuit"]["logical_qubits"]
                == case["circuit"]["allocated_qubits"] // 2
                for case in cases
            )
    assert config["metadata"]["path_comparison"] == (
        "two_standard_executable_baselines_not_upmem_aware_planning"
    )
    assert (
        "descriptive_single_rank_dpu_scaling"
        not in config["metadata"]["claims_allowed"]
    )
    assert {item["planner"]["engine"] for item in config["planner_variants"]} == {
        "opt_einsum",
        "cotengra",
    }
    cotengra = next(
        item
        for item in config["planner_variants"]
        if item["id"] == "cotengra_flops_seed0"
    )
    assert cotengra["planner"] == {
        "engine": "cotengra",
        "objective": "flops",
        "methods": "greedy",
        "max_repeats": 1,
        "seed": 0,
    }
    assert {item["policy"] for item in config["numeric_policies"]} == {
        "float32_real",
        "host_packed_int8",
    }
    assert len(config["engine_variants"]) == 2
    assert config["repeats"] == 3


def test_canonical_planning_resolves_two_distinct_standard_paths_per_case() -> None:
    config = _config("m5_circuit_canonical.yml")
    plans = _build_plans(ROOT, config)
    assert len(plans) == 84
    by_case: dict[str, list[object]] = {}
    for plan in plans:
        by_case.setdefault(plan.case["case_id"], []).append(plan.dag)
    assert set(by_case) == {case["case_id"] for case in config["cases"]}
    for dags in by_case.values():
        assert len(dags) == 2
        assert len({contraction_dag_hash(dag) for dag in dags}) == 2


@pytest.mark.parametrize(
    ("profile", "case_count", "dpu_counts", "warmups", "repeats"),
    [
        ("m5_circuit_large.yml", 15, {64}, 0, 1),
        ("m5_circuit_scaling.yml", 1, {1, 2, 4, 8, 16, 32, 64, 128}, 1, 5),
    ],
)
def test_large_and_scaling_profiles_have_bounded_requested_topologies(
    profile: str,
    case_count: int,
    dpu_counts: set[int],
    warmups: int,
    repeats: int,
) -> None:
    config = _config(profile)
    assert len(config["cases"]) == case_count
    assert config["warmups"] == warmups
    assert config["repeats"] == repeats
    if profile == "m5_circuit_large.yml":
        assert config["metadata"]["path_comparison"] == (
            "two_standard_executable_baselines_not_upmem_aware_planning"
        )
        assert (
            "descriptive_single_rank_dpu_scaling"
            not in config["metadata"]["claims_allowed"]
        )
        assert [item["id"] for item in config["planner_variants"]] == [
            "opt_einsum_greedy",
            "cotengra_flops_seed0",
        ]
        assert {case["family"] for case in config["cases"]} == {"BV", "EDC", "HS"}
        for family in {"BV", "EDC", "HS"}:
            cases = [case for case in config["cases"] if case["family"] == family]
            assert {
                case["circuit"].get("allocated_qubits", case["circuit"].get("n_qubits"))
                for case in cases
            } == {22, 24, 26, 28, 30}
        assert config["resource_limits"]["max_live_bytes"] == 2147483648
        assert config["resource_limits"]["max_output_bytes"] == 2147483648
    else:
        case = config["cases"][0]
        assert case["family"] == "EDC"
        assert case["circuit"] == {
            "kind": "quest_compatible",
            "name": "EDC",
            "n_qubits": 20,
        }
    physical = _physical_engines(config)
    assert {_requested_dpus(item) for item in physical} == dpu_counts
    for item in physical:
        _assert_physical_contract(item)


def test_scaling_profile_varies_only_physical_topology() -> None:
    config = _config("m5_circuit_scaling.yml")
    assert len(config["planner_variants"]) == 1
    assert config["planner_variants"][0]["planner"] == {
        "engine": "opt_einsum",
        "optimize": "greedy",
    }
    assert {item["policy"] for item in config["numeric_policies"]} == {
        "float32_real",
        "host_packed_int8",
    }
    physical = _physical_engines(config)
    assert len(physical) == 8
    assert len(physical[-1]["topology"]["rank_paths"]) == 2
