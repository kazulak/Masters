"""The new score observes production controls without creating tensor payloads."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.upmem.execution_features import extract_launch_cost_features
from quantum_bench.upmem.path_heuristic import LaunchCostFacts
from tests.test_upmem_execution_features import _plan, _single, fork_join
from tests.test_upmem_wave_work import _node


@pytest.mark.parametrize("fuse", [False, True])
def test_launch_facts_retain_actual_controls_and_full_four_product_work(fuse):
    dag, _ = fork_join(k=1)
    result = extract_launch_cost_features(dag, _plan(dag), fuse_complex=fuse)
    facts = LaunchCostFacts.from_mapping(result)
    execution = result["execution"]
    totals = execution["totals"]
    assert len(facts.launches) == totals["launch_count"] == (2 if fuse else 8)
    assert facts.total_w() == totals["real_mac_count"] == 64
    assert facts.total_m() == totals["local_traffic"]["mram_aligned_transfer_bytes_estimate"]
    assert facts.h == totals["h2d_bytes"] + totals["d2h_bytes"]
    assert facts.n == totals["cohort_count"] + totals["launch_count"]
    assert facts.p == sum(result["host_array_passes"].values())
    for launch, wave in zip(facts.launches, execution["waves"], strict=True):
        assert launch.w_by_dpu == tuple(slot["real_mac_count"] for slot in wave["slots"])


def test_padding_and_named_array_passes_have_hand_computed_bytes():
    # A 1x1 complex128 scalar product: four 4-byte input planes, padded to 8.
    dag = _single(_node("scalar", 1, 1, 1))
    result = extract_launch_cost_features(dag, _plan(dag, dpus=1))
    assert result["host_array_passes"] == {
        "operand_copy": 64,
        "component_cast": 48,
        "operand_reduce": 0,
        "canonical_copy": 0,
        "complex_canonical_and_encoded_copy": 64,
        "tile_contiguous_copy": 0,
        "tile_bytes_and_padding": 4 * (4 + 8 + 2 * 8),
        "envelope_payload_copy": 4 * 2 * 8,
        "lane_assembly": 64 + 12 * 4,
        "complex_decode_and_copy": 56,
        "graph_reduce": 0,
        "final_copy": 16,
    }


def test_component_view_copy_depends_on_dtype_not_peak_memory():
    dag = _single(_node("matrix", 2, 2, 2))
    narrowed = replace(
        dag, tensors=tuple(replace(t, dtype="complex64") for t in dag.tensors),
        nodes=tuple(replace(node, output=replace(node.output, dtype="complex64")) for node in dag.nodes),
    )
    wide = extract_launch_cost_features(dag, _plan(dag, dpus=1))["host_array_passes"]
    narrow = extract_launch_cost_features(narrowed, _plan(narrowed, dpus=1))["host_array_passes"]
    assert wide["component_cast"] == 8 * 24
    assert narrow["component_cast"] == 0
    assert wide["canonical_copy"] == 0
    assert narrow["canonical_copy"] == 8 * 16


def test_feature_extraction_cannot_allocate_or_execute_arrays(monkeypatch):
    dag, _ = fork_join()
    plan = _plan(dag)

    def forbidden(*args, **kwargs):
        pytest.fail("metadata extraction attempted array allocation/execution")

    for name in ("array", "asarray", "ascontiguousarray", "zeros", "empty", "full", "matmul", "einsum"):
        monkeypatch.setattr(np, name, forbidden)
    result = extract_launch_cost_features(dag, plan)
    assert result["P"] > 0


def test_other_numeric_policy_cannot_silently_reuse_float32_host_pass_model():
    dag, _ = fork_join()
    with pytest.raises(ValueError, match="float32-only"):
        extract_launch_cost_features(dag, _plan(dag, policy="complex_int8_shared_scale_v1"))


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("permuted", [False, True])
def test_canonical_copy_inventory_matches_actual_small_operand_preparation(monkeypatch, dtype, permuted):
    from quantum_bench.upmem import tiling
    from quantum_bench.upmem.runtime import _prepare_complex_operation

    node = _node("copy", 2, 3, 4)
    if permuted:
        node = replace(node, left=replace(node.left, labels=(1, 0), shape=(4, 2)))
    node = replace(node, output=replace(node.output, dtype=dtype))
    dag = _single(node)
    dag = replace(dag, tensors=tuple(replace(t, dtype=dtype) for t in dag.tensors))
    plan = _plan(dag, dpus=1)
    expected = extract_launch_cost_features(dag, plan)["host_array_passes"]["canonical_copy"]
    observed = []
    original = tiling._as_batched_matrix

    def spy(array, *args, **kwargs):
        result = original(array, *args, **kwargs)
        observed.append(0 if np.shares_memory(array, result) else array.nbytes + result.nbytes)
        return result

    monkeypatch.setattr(tiling, "_as_batched_matrix", spy)
    _prepare_complex_operation(
        node, np.ones(node.left.shape, dtype=dtype), np.ones(node.right.shape, dtype=dtype),
        plan.stages[0], plan.numeric_policy,
    )
    assert expected == sum(observed)


def test_output_permutation_and_shape_restore_do_not_force_an_extra_copy(monkeypatch):
    from quantum_bench.upmem.tiling import lower_binary_contraction

    node = _node("transpose_output", 2, 3, 4)
    node = replace(node, output_labels=(2, 0), output=replace(
        node.output, labels=(2, 0), shape=(3, 2),
    ))
    lowering = lower_binary_contraction(
        node, np.ones(node.left.shape, dtype=np.float32), np.ones(node.right.shape, dtype=np.float32),
    )
    allocations = []
    original = np.zeros

    def zeros(*args, **kwargs):
        result = original(*args, **kwargs)
        allocations.append(result)
        return result

    partials = {tile.id: np.ones((tile.m_size, tile.n_size), dtype=np.float32) for tile in lowering.tiles}
    monkeypatch.setattr(np, "zeros", zeros)
    output = lowering.assemble(partials)
    assert output.shape == node.output.shape
    assert not output.flags.c_contiguous
    assert np.shares_memory(output, allocations[0])


def test_frozen_family_workload_greedy_facts_are_metadata_only(monkeypatch):
    from tests.test_upmem_family_aligned_packet import path_study

    manifest = json.loads((Path(__file__).resolve().parents[1] /
        "configs/upmem_family_aligned_workload_v1.json").read_text())
    for instance in manifest["instances"]:
        circuit = path_study._circuit_from_definition(instance["circuit"])
        network, _ = path_study.lower_tensor_network(path_study.make_simulation_job(circuit))
        path, _ = path_study.plan_opt_einsum(network, optimize="greedy")
        dag = path_study.build_contraction_dag(network, path)
        for dpus in (1, 4):
            plan = _plan(dag, dpus=dpus)

            def forbidden(*args, **kwargs):
                pytest.fail("family fact extraction attempted tensor allocation")

            with monkeypatch.context() as patch:
                for name in ("array", "asarray", "ascontiguousarray", "zeros", "empty", "full"):
                    patch.setattr(np, name, forbidden)
                result = extract_launch_cost_features(dag, plan)
            facts = LaunchCostFacts.from_mapping(result)
            assert all(value > 0 for value in facts.as_tuple())
