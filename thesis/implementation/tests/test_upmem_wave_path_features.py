"""Pure wave surrogate qualification; no SDK, fitted profile or timing claims."""

from dataclasses import replace

import pytest

from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractionDAG, TensorSpec
from quantum_bench.upmem.execution_features import extract_execution_features
from quantum_bench.upmem.path_heuristic import (
    COST_MODEL_ID,
    WAVE_COST_MODEL_ID,
    WeightVector,
    extract_plan_features,
    extract_wave_path_features,
    score_features,
    score_wave_path_features,
)
from tests.test_upmem_execution_features import POLICIES, SCHEDULES, _plan
from tests.test_upmem_wave_runtime import fork_join


WEIGHTS = WeightVector.from_values((1, 1, 1, 1, 0, 0))


def test_host_reduction_is_one_explicit_sync_count():
    dag, _ = fork_join(k=3)
    dag = slice_contraction(dag, node_id="a", labels=(1,))
    facts = extract_wave_path_features(dag, _plan(dag))
    assert facts["sync_components"]["host_reduce_count"] == 1
    assert facts["raw"].sync_events == sum(facts["sync_components"].values())
    assert facts["raw"].dpu_work == facts["totals"]["wave_critical_real_mac_sum"]


@pytest.mark.parametrize("fuse", [False, True])
def test_d1_degeneration_and_explicit_formulas(fuse):
    dag, _ = fork_join()
    rows = [extract_wave_path_features(dag, _plan(dag, schedule=s, dpus=1),
                                      fuse_complex=fuse) for s in SCHEDULES]
    assert rows[0]["raw"] == rows[1]["raw"]
    for facts in rows:
        raw, total = facts["raw"], facts["totals"]
        assert raw.host_dpu_bytes == total["h2d_bytes"] + total["d2h_bytes"]
        assert raw.dpu_work == total["real_mac_count"]
        assert raw.mram_wram_bytes == total["local_traffic"]["mram_aligned_transfer_bytes_estimate"]
        assert raw.sync_events == sum(total[k] for k in (
            "cohort_count", "launch_count", "host_reduce_count", "barrier_events"))
        assert raw.wram_pressure == facts["static_memory"]["known_wram_buffers_bytes"]
        assert raw.numeric_overhead == 0
        assert facts["inactive_features"] == ("E_num", "P_wram")


@pytest.mark.parametrize("fuse", [False, True])
def test_equal_mac_fork_join_critical_work_and_fusion(fuse):
    dag, _ = fork_join(k=1)
    serial, wave = [extract_wave_path_features(dag, _plan(dag, schedule=s),
                                             fuse_complex=fuse) for s in SCHEDULES]
    assert serial["totals"]["real_mac_count"] == wave["totals"]["real_mac_count"] == 64
    assert serial["raw"].dpu_work == 64
    assert wave["raw"].dpu_work == 48
    assert serial["totals"]["launch_count"] == (3 if fuse else 12)
    assert wave["totals"]["launch_count"] == (2 if fuse else 8)
    assert wave["raw"].mram_wram_bytes == sum(max(
        s["local_traffic"]["mram_aligned_transfer_bytes_estimate"] for s in w["slots"]
    ) for w in wave["waves"])
    assert wave["sync_components"]["wave_critical_barrier_events"] == (22 if fuse else 40)
    assert wave["raw"].sync_events == (26 if fuse else 50)
    compute_only = WeightVector.from_values((0, 0, 1, 0, 0, 0))
    # Cross-policy diagnostics use generic math, not frozen-context scoring.
    assert score_features(wave["raw"], serial["raw"], compute_only) < 0


@pytest.mark.parametrize("field,value", [
    ("rank_count", 2), ("dpu_count", 4), ("tasklets_per_dpu", 16),
    ("numeric_policy", POLICIES[1]), ("request_transport", "other_transport"),
    ("kernel_identity", "other_kernel"), ("schedule_policy", SCHEDULES[0]),
    ("fuse_complex", True), ("geometry_policy", "outer_k1_v1"),
])
def test_reference_context_mismatch_rejected(field, value):
    dag, _ = fork_join()
    facts = extract_wave_path_features(dag, _plan(dag))
    reference = {**facts, "plan": {**facts["plan"], field: value}}
    with pytest.raises(ValueError, match=f"context mismatch: {field}"):
        score_wave_path_features(facts, reference, WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID)


def test_same_context_zero_reference_and_distinct_path_ids_allowed():
    dag, _ = fork_join()
    facts = extract_wave_path_features(dag, _plan(dag))
    assert score_wave_path_features(facts, facts, WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID) == 0
    reference = {**facts, "plan": {**facts["plan"],
                                   "logical_plan_id": "another_logical_path",
                                   "physical_plan_id": "another_physical_path"}}
    assert score_wave_path_features(facts, reference, WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID) == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_mixed_fusion_fallback_and_identity_retention(policy):
    original, _ = fork_join(k=1)
    shapes = (((256, 1), (1, 256), (256, 256)),
              ((256, 1), (1, 1), (256, 1)),
              ((256, 256), (256, 1), (256, 1)))
    nodes = tuple(replace(node, left=replace(node.left, shape=left),
                          right=replace(node.right, shape=right),
                          output=replace(node.output, shape=output))
                  for node, (left, right, output) in zip(original.nodes, shapes, strict=True))
    tensors = tuple(TensorSpec(v.tensor_id, v.labels, v.shape, "dense", dtype="complex128")
                    for node in nodes[:2] for v in (node.left, node.right))
    dag = ContractionDAG(tensors=tensors, nodes=nodes, output=replace(original.output, shape=(256, 1)))
    plan = _plan(dag, policy=policy)
    options = {"fuse_complex": True, "geometry_policy": "outer_k1_v1"}
    facts = extract_wave_path_features(dag, plan, **options)
    source = extract_execution_features(dag, plan, **options)
    assert all(facts[key] == value for key, value in source.items())
    assert facts == extract_wave_path_features(dag, plan, **options)
    # product_layout admits more fused controls with the smaller int8 planes.
    assert facts["totals"]["launch_count"] == (8 if policy == POLICIES[0] else 5)
    assert facts["totals"]["real_mac_count"] == 525312
    assert set(facts["totals"]["local_traffic"]["algorithms"]) == {"panel_compute_v1", "outer_compute_v1"}
    assert facts["plan"]["logical_plan_id"] == plan.logical_plan_id
    assert facts["plan"]["numeric_policy"] == policy
    assert facts["cost_model_id"] != COST_MODEL_ID
    if policy == POLICIES[1]:
        with pytest.raises(ValueError, match="float32"):
            score_wave_path_features(facts, facts, WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID)
    else:
        assert score_wave_path_features(facts, facts, WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID) == 0


def test_no_historical_profile_autouse_or_inactive_penalties():
    dag, _ = fork_join()
    plan = _plan(dag)
    facts = extract_wave_path_features(dag, plan)
    with pytest.raises(ValueError, match="requires serial_nodes_v1"):
        extract_plan_features(plan)
    assert extract_plan_features(_plan(dag, schedule=SCHEDULES[0])).raw.dpu_work > 0
    with pytest.raises(TypeError):
        score_wave_path_features(facts, facts, WEIGHTS)
    with pytest.raises(ValueError, match="identity"):
        score_wave_path_features(facts, facts, WEIGHTS, cost_model_id=COST_MODEL_ID)
    with pytest.raises(ValueError, match="identity"):
        score_wave_path_features({**facts, "cost_model_id": COST_MODEL_ID}, facts,
                                 WEIGHTS, cost_model_id=WAVE_COST_MODEL_ID)
    for values in ((1, 1, 1, 1, 1, 0), (1, 1, 1, 1, 0, 1)):
        with pytest.raises(ValueError, match="zero weight"):
            score_wave_path_features(facts, facts, WeightVector.from_values(values),
                                     cost_model_id=WAVE_COST_MODEL_ID)
