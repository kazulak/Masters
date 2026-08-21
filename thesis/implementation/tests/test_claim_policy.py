from __future__ import annotations

from quantum_bench.evidence import Claim, ClaimPolicy, ExecutionMode


def _row(
    *,
    engine: str,
    runtime: float = 1.0,
    path: str = "greedy",
    numeric: str = "float32",
    dpu_count: int = 1,
) -> dict[str, object]:
    is_upmem = "upmem" in engine
    row: dict[str, object] = {
        "engine_id": engine,
        "case_id": "case",
        "path_variant_id": path,
        "numeric_policy": numeric,
        "timing_s": runtime,
        "status": "completed",
        "scientific_validation_status": "passed",
        "exact_once": True,
        "no_fallback_used": True,
        "repeat_id": 0,
        "timing_scope": "whole_route_v1",
        "circuit_semantics_hash": "circuit",
        "tensor_network_hash": "network",
        "contraction_plan_hash": f"plan-{path}",
        "executor_config_hash": "executor",
        "route_modules": {
            "tensor_network": {"implementation": "tn"},
            "planner": {"implementation": path},
            "numeric": {"implementation": numeric},
            "executor": {"implementation": engine},
            "topology": {"implementation": f"upmem-{dpu_count}" if is_upmem else "cpu"},
        },
    }
    if "int8" in numeric:
        row.update(
            numeric_policy_id=numeric,
            packed_int8_transfer=True,
            numeric_transport="host_packed_int8_mram",
        )
    if is_upmem:
        row.update(
            target_observed="physical_hardware",
            hardware_allocation_verified=True,
            native_kernel_executed=True,
            hardware_kernel_executed=True,
            simulator=False,
            simulator_kernel_executed=False,
            cpu_fallback=False,
            cpu_fallback_used=False,
            release_succeeded=True,
            hardware_speedup_applicable=True,
            timing_is_bringup_only=False,
            requested_dpu_count=dpu_count,
            allocated_dpu_count=dpu_count,
            rank_count=1,
            engine_metadata={
                "active_dpu_ids": list(range(dpu_count)),
                "active_rank_indices": [0],
            },
        )
    return row


def _v2(row: dict[str, object], dag_hash: str) -> dict[str, object]:
    row["contraction_dag_schema_version"] = "contraction_dag_v2"
    row["contraction_dag_hash"] = dag_hash
    row["exact_once_scope"] = "host_dag_node_completion_per_route"
    row["host_dag_node_completion_coverage"] = True
    if "upmem" in str(row["engine_id"]):
        row["native_identity_verified"] = True
    return row


def _m5_v2(row: dict[str, object], dag_hash: str) -> dict[str, object]:
    return _v2(row, dag_hash)


def test_cpu_and_physical_upmem_rows_are_timing_admissible() -> None:
    policy = ClaimPolicy()

    assert policy.execution_mode(_row(engine="cpu_numpy")) is ExecutionMode.CPU_HOST
    assert policy.evaluate_row(Claim.TIMING, _row(engine="cpu_numpy")).allowed
    assert policy.evaluate_row(Claim.TIMING, _row(engine="upmem_m5")).allowed

    modeled_cpu = _row(engine="cpu_numpy")
    modeled_cpu["execution_mode"] = "model"
    assert not policy.evaluate_row(Claim.TIMING, modeled_cpu).allowed

    spoofed_cpu = _row(engine="cpu_numpy")
    spoofed_cpu.update(execution_mode="cpu_host", modeled=True)
    assert policy.execution_mode(spoofed_cpu) is ExecutionMode.MODEL
    assert not policy.evaluate_row(Claim.TIMING, spoofed_cpu).allowed

    measured_cpu = _row(engine="cpu_numpy")
    measured_cpu["timing_origin"] = "host_timer"
    assert policy.evaluate_row(Claim.TIMING, measured_cpu).allowed
    measured_cpu["timing_origin"] = "analytic_model"
    assert not policy.evaluate_row(Claim.TIMING, measured_cpu).allowed

    physical_with_planner_model = _row(engine="upmem_m5")
    physical_with_planner_model.update(
        model_id="planner-model-v2",
        cost_model_id="upmem-path-cost-v2",
        resource_model_id="mram-wram-v1",
    )
    assert (
        policy.execution_mode(physical_with_planner_model)
        is ExecutionMode.PHYSICAL_HARDWARE
    )
    assert policy.evaluate_row(Claim.TIMING, physical_with_planner_model).allowed

    analytic_cpu = _row(engine="cpu_numpy")
    analytic_cpu.update(execution_mode="cpu_host", simulation_method="analytic_model")
    assert policy.execution_mode(analytic_cpu) is ExecutionMode.MODEL
    assert not policy.evaluate_row(Claim.TIMING, analytic_cpu).allowed


def test_current_m5_dag_v2_requires_native_identity_and_host_dag_coverage() -> None:
    policy = ClaimPolicy()
    row = _m5_v2(_row(engine="upmem_m5"), "dag-m5")

    row.pop("native_identity_verified")
    missing_native = policy.evaluate_row(Claim.TIMING, row)
    assert not missing_native.allowed
    assert (
        "DAG-v2 physical UPMEM/M5 admission requires native_identity_verified=True"
        in missing_native.reasons
    )

    row["native_identity_verified"] = False
    false_native = policy.evaluate_row(Claim.PHYSICAL_EXECUTION, row)
    assert not false_native.allowed
    assert (
        "DAG-v2 physical UPMEM/M5 admission requires native_identity_verified=True"
        in false_native.reasons
    )

    row["native_identity_verified"] = True
    row.pop("host_dag_node_completion_coverage")
    missing_coverage = policy.evaluate_row(Claim.FUNCTIONAL_CORRECTNESS, row)
    assert not missing_coverage.allowed
    assert any(
        "host DAG coverage is not native kernel exact-once evidence" in reason
        for reason in missing_coverage.reasons
    )

    row["host_dag_node_completion_coverage"] = False
    false_coverage = policy.evaluate_row(Claim.TIMING, row)
    assert not false_coverage.allowed
    assert any(
        "host DAG coverage is not native kernel exact-once evidence" in reason
        for reason in false_coverage.reasons
    )

    row["host_dag_node_completion_coverage"] = True
    row["exact_once_scope"] = "native_kernel_exact_once"
    wrong_scope = policy.evaluate_row(Claim.FUNCTIONAL_CORRECTNESS, row)
    assert not wrong_scope.allowed
    assert any(
        "host DAG coverage is not native kernel exact-once evidence" in reason
        for reason in wrong_scope.reasons
    )

    row["exact_once_scope"] = "host_dag_node_completion_per_route"
    assert policy.evaluate_row(Claim.FUNCTIONAL_CORRECTNESS, row).allowed
    assert policy.evaluate_row(Claim.PHYSICAL_EXECUTION, row).allowed
    assert policy.evaluate_row(Claim.TIMING, row).allowed


def test_legacy_rows_keep_exact_once_compatibility_without_new_fields() -> None:
    policy = ClaimPolicy()
    row = _row(engine="upmem_m5")

    assert "native_identity_verified" not in row
    assert "host_dag_node_completion_coverage" not in row
    assert policy.evaluate_row(Claim.FUNCTIONAL_CORRECTNESS, row).allowed
    assert policy.evaluate_row(Claim.PHYSICAL_EXECUTION, row).allowed
    assert policy.evaluate_row(Claim.TIMING, row).allowed


def test_speedup_requires_matching_scope_hashes_and_repeat_context() -> None:
    policy = ClaimPolicy()
    baseline = _row(engine="cpu_numpy")
    candidate = _row(engine="upmem_m5")

    assert policy.evaluate_pair(Claim.SPEEDUP, baseline, candidate).allowed

    candidate["timing_scope"] = "kernel_only"
    mismatch = policy.evaluate_pair(Claim.SPEEDUP, baseline, candidate)
    assert not mismatch.allowed
    assert "speedup requires matching timing scopes" in mismatch.reasons

    candidate["timing_scope"] = "whole_route_v1"
    candidate.pop("repeat_id")
    missing_repeat = policy.evaluate_pair(Claim.SPEEDUP, baseline, candidate)
    assert not missing_repeat.allowed
    assert "speedup aggregation requires repeat identifiers" in missing_repeat.reasons


def test_pair_claims_require_compatible_contraction_dag_identities() -> None:
    policy = ClaimPolicy()

    cpu = _v2(_row(engine="cpu_numpy"), "dag-a")
    upmem = _v2(_row(engine="upmem_m5"), "dag-a")
    assert policy.evaluate_pair(Claim.SPEEDUP, cpu, upmem).allowed

    mismatch = _v2(_row(engine="upmem_m5"), "dag-b")
    speedup = policy.evaluate_pair(Claim.SPEEDUP, cpu, mismatch)
    assert not speedup.allowed
    assert "speedup requires matching contraction DAG hashes" in speedup.reasons

    broken = _v2(_row(engine="upmem_m5"), "")
    numeric = policy.evaluate_pair(
        Claim.NUMERIC_ABLATION,
        _v2(_row(engine="upmem_m5", numeric="float32"), "dag-a"),
        broken,
    )
    assert not numeric.allowed
    assert (
        "numeric ablation requires complete valid contraction DAG identities"
        in numeric.reasons
    )

    legacy = _row(engine="upmem_m5", numeric="float32")
    mixed = _v2(_row(engine="upmem_m5", numeric="host_packed_int8"), "dag-a")
    mixed["contraction_plan_hash"] = legacy["contraction_plan_hash"]
    decision = policy.evaluate_pair(Claim.NUMERIC_ABLATION, legacy, mixed)
    assert not decision.allowed
    assert any("legacy and v2 rows cannot mix" in reason for reason in decision.reasons)

    path_a = _v2(_row(engine="upmem_m5", path="greedy"), "dag-a")
    path_b = _v2(_row(engine="upmem_m5", path="cotengra"), "dag-b")
    assert policy.evaluate_pair(Claim.PATH_ABLATION, path_a, path_b).allowed
    same_dag = _v2(_row(engine="upmem_m5", path="cotengra"), "dag-a")
    decision = policy.evaluate_pair(Claim.PATH_ABLATION, path_a, same_dag)
    assert not decision.allowed
    assert "path ablation requires distinct contraction DAG hashes" in decision.reasons

    scale_a = _v2(_row(engine="upmem_m5", dpu_count=1), "dag-a")
    scale_b = _v2(_row(engine="upmem_m5", dpu_count=2), "dag-b")
    decision = policy.evaluate_pair(Claim.SCALING, scale_a, scale_b)
    assert not decision.allowed
    assert "scaling requires matching contraction DAG hashes" in decision.reasons


def test_speedup_rejects_simulator_or_fallback_candidate() -> None:
    policy = ClaimPolicy()
    baseline = _row(engine="cpu_numpy")
    candidate = _row(engine="upmem_m5")
    candidate["simulator_kernel_executed"] = True

    decision = policy.evaluate_pair(Claim.SPEEDUP, baseline, candidate)

    assert not decision.allowed
    assert any("simulator" in reason for reason in decision.reasons)

    candidate = _row(engine="upmem_m5")
    candidate["execution_mode"] = "model"
    modeled = policy.evaluate_pair(Claim.SPEEDUP, baseline, candidate)
    assert not modeled.allowed
    assert "speedup candidate must use physical UPMEM hardware" in modeled.reasons


def test_energy_requires_positive_measured_non_tdp_source() -> None:
    policy = ClaimPolicy()
    row = _row(engine="upmem_m5")
    row.update(
        energy_joules=0.5,
        energy_source="external_meter_measured",
        energy_measurement_status="measured",
        energy_measurement_boundary="host_and_dimm",
        energy_sensor_id="meter-1",
        energy_measurement_interval_s=1.0,
        energy_sample_count=2,
    )
    assert policy.evaluate_row(Claim.ENERGY, row).allowed

    row["energy_source"] = "tdp_estimate"
    decision = policy.evaluate_row(Claim.ENERGY, row)
    assert not decision.allowed
    assert "energy source is not an approved measured source" in decision.reasons

    row["energy_source"] = "external_meter_measured"
    row["energy_measurement_status"] = "not_measured"
    decision = policy.evaluate_row(Claim.ENERGY, row)
    assert not decision.allowed
    assert "energy measurement status is not measured" in decision.reasons

    row["energy_measurement_status"] = "measured"
    row["energy_measurement_interval_s"] = "soon"
    decision = policy.evaluate_row(Claim.ENERGY, row)
    assert not decision.allowed
    assert "energy measurement interval is missing or non-positive" in decision.reasons

    row["energy_measurement_interval_s"] = 1.0
    row["energy_source"] = "magic_source"
    decision = policy.evaluate_row(Claim.ENERGY, row)
    assert not decision.allowed
    assert "energy source is not an approved measured source" in decision.reasons

    row["energy_source"] = "external_meter_measured"
    row["energy_measurement_status"] = "passed"
    row.pop("energy_sample_count")
    decision = policy.evaluate_row(Claim.ENERGY, row)
    assert not decision.allowed
    assert "energy measurement status is not measured" in decision.reasons
    assert (
        "energy measurement lacks positive samples or counter readings"
        in decision.reasons
    )


def test_path_numeric_and_scaling_pairs_require_exact_changed_module_role() -> None:
    policy = ClaimPolicy()

    path_a = _row(engine="upmem_m5", path="greedy")
    path_b = _row(engine="upmem_m5", path="cotengra")
    assert policy.evaluate_pair(Claim.PATH_ABLATION, path_a, path_b).allowed

    numeric_a = _row(engine="upmem_m5", numeric="float32")
    numeric_b = _row(engine="upmem_m5", numeric="host_packed_int8")
    numeric_b["contraction_plan_hash"] = numeric_a["contraction_plan_hash"]
    assert policy.evaluate_pair(Claim.NUMERIC_ABLATION, numeric_a, numeric_b).allowed

    scale_a = _row(engine="upmem_m5", dpu_count=1)
    scale_b = _row(engine="upmem_m5", dpu_count=2)
    assert policy.evaluate_pair(Claim.SCALING, scale_a, scale_b).allowed
    assert not policy.evaluate_row(Claim.SCALING, scale_a).allowed

    bad_modules = dict(path_b)
    bad_modules["route_modules"] = dict(path_b["route_modules"])
    bad_modules["route_modules"]["executor"] = {"implementation": "other"}
    decision = policy.evaluate_pair(Claim.PATH_ABLATION, path_a, bad_modules)
    assert not decision.allowed
    assert any("observed=executor,planner" in reason for reason in decision.reasons)


def test_numeric_pair_requires_verified_host_packed_transport() -> None:
    policy = ClaimPolicy()
    float32 = _row(engine="upmem_m5", numeric="float32")
    int8 = _row(engine="upmem_m5", numeric="host_packed_int8")
    int8["contraction_plan_hash"] = float32["contraction_plan_hash"]
    int8["packed_int8_transfer"] = False

    decision = policy.evaluate_pair(Claim.NUMERIC_ABLATION, float32, int8)

    assert not decision.allowed
    assert any("host-packed Int8" in reason for reason in decision.reasons)

    int8 = _row(engine="upmem_m5", numeric="host_packed_int8")
    int8["contraction_plan_hash"] = float32["contraction_plan_hash"]
    int8.pop("packed_int8_transfer")
    int8.pop("numeric_transport")
    metadata = int8["engine_metadata"]
    assert isinstance(metadata, dict)
    metadata["task_metrics"] = [
        {
            "packed_int8_transfer": True,
            "numeric_transport": "host_packed_int8_mram",
        }
    ]
    assert policy.evaluate_pair(Claim.NUMERIC_ABLATION, float32, int8).allowed


def test_controlled_pairs_require_route_modules_on_both_rows() -> None:
    policy = ClaimPolicy()
    path_a = _row(engine="upmem_m5", path="greedy")
    path_b = _row(engine="upmem_m5", path="cotengra")
    numeric_a = _row(engine="upmem_m5", numeric="float32")
    numeric_b = _row(engine="upmem_m5", numeric="host_packed_int8")
    numeric_b["contraction_plan_hash"] = numeric_a["contraction_plan_hash"]
    scale_a = _row(engine="upmem_m5", dpu_count=1)
    scale_b = _row(engine="upmem_m5", dpu_count=2)

    for claim, baseline, candidate in (
        (Claim.PATH_ABLATION, path_a, path_b),
        (Claim.NUMERIC_ABLATION, numeric_a, numeric_b),
        (Claim.SCALING, scale_a, scale_b),
    ):
        without_modules_a = dict(baseline)
        without_modules_b = dict(candidate)
        without_modules_a.pop("route_modules")
        without_modules_b.pop("route_modules")
        absent = policy.evaluate_pair(claim, without_modules_a, without_modules_b)
        assert not absent.allowed
        assert (
            "comparison requires complete route_modules on both rows" in absent.reasons
        )

        one_sided = policy.evaluate_pair(claim, baseline, without_modules_b)
        assert not one_sided.allowed
        assert (
            "comparison requires complete route_modules on both rows"
            in one_sided.reasons
        )
