"""Conservative resident admission, not a production residency executor."""

from dataclasses import replace

import numpy as np
import pytest

from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.locality_probe import _orders, resident_pair_probe_layout, slice_branch_facts
from quantum_bench.upmem.plan import UpmemTopology, plan_upmem
from quantum_bench.upmem.tiling import canonicalize_binary_contraction
from quantum_bench.upmem.wave_protocol import FOUR_PRODUCT_PANEL, WaveControl


def pair(m=2, k=3, n=4, q=5, *, side="left", transpose=False, consume_transposed=False):
    def tensor(name, labels, shape, producer=None):
        return TensorSpec(name, labels, shape, "dense", produced_by=producer)

    def view(t):
        return TensorView(tensor_id=t.id, labels=t.labels, shape=t.shape)

    a, b = tensor("a", (0, 1), (m, k)), tensor("b", (1, 2), (k, n))
    x = tensor("x", (2, 0) if transpose else (0, 2), (n, m) if transpose else (m, n), "first")
    first = ContractNode(node_id="first", left=view(a), right=view(b), output=x,
                         contracted_labels=(1,), output_labels=x.labels, dependencies=())
    if side == "left":
        reduction, free, size, out_size = (0, 2, m, n) if consume_transposed else (2, 0, n, m)
        c = tensor("c", (reduction, 3), (size, q))
        out = tensor("out", (free, 3), (out_size, q), "second")
        left, right = view(x), view(c)
    else:
        reduction = 0
        c = tensor("c", (3, 0), (q, m))
        out = tensor("out", (3, 2), (q, n), "second")
        left, right = view(c), view(x)
    second = ContractNode(node_id="second", left=left, right=right, output=out,
                          contracted_labels=(reduction,), output_labels=out.labels, dependencies=("first",))
    return ContractionDAG(tensors=(a, b, c), nodes=(first, second), output=view(out))


def layout(dag, policy="split_complex_float32_v1"):
    plan = plan_upmem(dag, numeric_policy=policy,
                      topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1))
    return resident_pair_probe_layout(dag, plan, "first", "second")


@pytest.mark.parametrize("side,transpose", [("left", False), ("right", False), ("left", True), ("right", True)])
def test_native_orders_and_explicit_joint_layout(side, transpose):
    dag = pair(side=side, transpose=transpose)
    result = layout(dag)
    assert result["eligible_for_native_probe"]
    assert not result["runtime_admitted"] and not result["native_execution_implemented"]
    assert result["rejection_reasons"] == []
    for i, name in enumerate(("first", "second")):
        node = dag.nodes[i]
        canonical = canonicalize_binary_contraction(node, np.zeros(node.left.shape), np.zeros(node.right.shape))
        assert _orders(node)[2] == canonical.canonical_output_labels
        m, n, k = result[f"{name}_geometry"]
        WaveControl(dpu_id=0, tasklets=8, flags=0, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL,
                    operation_index=i, wave_id=i, request_sequence=1, tile_id=i, batch_index=0,
                    m=m, n=n, k=k, k_offset=0, planes=result[f"{name}_planes"]).validate()
    resident_indices = (0, 1) if side == "left" else (2, 3)
    assert tuple(result["second_planes"][i] for i in resident_indices) == result["retained_planes"]
    unique_regions = sorted(set((*result["first_planes"], *result["retained_planes"], *result["second_planes"])))
    assert all(a + size <= b for (a, size), (b, _) in zip(unique_regions, unique_regions[1:]))
    assert result["live_mram_bytes"] == sum(size for _, size in unique_regions)
    assert result["eliminable_intermediate_payload_bytes"] == 6 * result["retained_planes"][0][1]
    first, second = dag.nodes
    left = np.arange(np.prod(first.left.shape), dtype=np.float32).reshape(first.left.shape)
    right = np.arange(np.prod(first.right.shape), dtype=np.float32).reshape(first.right.shape)
    canonical = canonicalize_binary_contraction(first, left, right)
    native_output = (canonical.left @ canonical.right).reshape(2, 4)
    declared_output = native_output.T if transpose else native_output
    external_view = second.right if side == "left" else second.left
    external = np.ones(external_view.shape, dtype=np.float32)
    left, right = (declared_output, external) if side == "left" else (external, declared_output)
    consumer = canonicalize_binary_contraction(second, left, right)
    native_input = consumer.left if side == "left" else consumer.right
    np.testing.assert_array_equal(native_input.ravel(), native_output.ravel())


@pytest.mark.parametrize("dag,policy,reason", [
    (pair(), "complex_int8_shared_scale_v1", "shared_scale_int8_residency_not_qualified"),
    (pair(k=257), "split_complex_float32_v1", "full_operand_single_tile_without_split_k_required"),
    (pair(transpose=True, consume_transposed=True), "split_complex_float32_v1", "intermediate_requires_layout_permutation"),
    (pair(m=128, k=8, n=128, q=128), "split_complex_float32_v1", "joint_live_mram_exceeds_512KiB"),
])
def test_rejects_unproven_residency(dag, policy, reason):
    result = layout(dag, policy)
    assert not result["eligible_for_native_probe"]
    assert reason in result["rejection_reasons"]


def test_rejects_plan_tampering_and_unknown_nodes():
    dag = pair()
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1",
                      topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1))
    with pytest.raises(ValueError, match="recomputation"):
        resident_pair_probe_layout(dag, replace(plan, stages=plan.stages[::-1]), "first", "second")
    with pytest.raises(ValueError, match="distinct declared"):
        resident_pair_probe_layout(dag, plan, "unknown", "second")
    assert not resident_pair_probe_layout(dag, plan, "second", "first")["eligible_for_native_probe"]


def test_complete_slices_remain_full_shaped_and_missing_contributions_are_rejected():
    dag = pair(k=4)
    sliced = slice_contraction(dag, node_id="first", labels=(1,))
    facts = slice_branch_facts(sliced, dag.nodes[0])
    assert facts["slice_count"] == 4
    assert facts["output_elements_per_partial"] == 8
    assert facts["partial_output_elements_total"] == 32
    assert facts["scalar_partial_outputs"] is False
    reduction = next(node for node in sliced.nodes if node.node_id == facts["reduction_node_id"])
    missing = replace(reduction, inputs=reduction.inputs[:-1], dependencies=reduction.dependencies[:-1])
    malformed = replace(sliced, nodes=tuple(missing if node == reduction else node for node in sliced.nodes))
    with pytest.raises(ValueError, match="Cartesian product"):
        slice_branch_facts(malformed, dag.nodes[0])
    duplicated = replace(reduction, inputs=(*reduction.inputs, reduction.inputs[0]))
    malformed = replace(sliced, nodes=tuple(duplicated if node == reduction else node for node in sliced.nodes))
    with pytest.raises(ValueError, match="each full partial once"):
        slice_branch_facts(malformed, dag.nodes[0])
    plan = plan_upmem(sliced, numeric_policy="split_complex_float32_v1",
                      topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1))
    result = resident_pair_probe_layout(sliced, plan, facts["partial_node_ids"][0], "second")
    assert "sliced_views_not_supported_by_resident_probe" in result["rejection_reasons"]


def test_fanout_is_not_silently_discarded():
    dag = pair()
    second = dag.nodes[1]
    extra = replace(second, node_id="extra", output=replace(second.output, id="extra_out", produced_by="extra"))
    dag = replace(dag, nodes=(*dag.nodes, extra))
    assert "intermediate_has_other_consumers" in layout(dag)["rejection_reasons"]
