"""Static admission facts for a bounded two-contract float32 residency probe.

These facts do not enable residency in the production executor. A native probe
must still implement and qualify the intermediate reconstruction they describe.
"""

from itertools import product
from math import prod

from quantum_bench.lowering import validate_contraction_dag
from quantum_bench.model import ContractNode, ContractionDAG, ReduceNode
from quantum_bench.upmem.plan import UpmemPlan, physical_plan_id, validate_upmem_plan
from quantum_bench.upmem.wave_protocol import FOUR_PRODUCT_PANEL, MRAM_BYTES, aligned_bytes, product_layout


def _orders(node: ContractNode):
    left, right, output = node.left.labels, node.right.labels, node.output_labels
    shared = set(left) & set(right)
    batch = tuple(label for label in left if label in shared and label in output)
    contracted = tuple(label for label in left if label in shared and label not in output)
    free_left = tuple(label for label in left if label in output and label not in batch)
    free_right = tuple(label for label in right if label in output and label not in batch)
    unary = tuple(label for label in (*left, *right) if label not in output and label not in shared)
    return batch + free_left + contracted, batch + contracted + free_right, batch + free_left + free_right, unary


def resident_pair_probe_layout(
    dag: ContractionDAG, plan: UpmemPlan, producer_id: str, consumer_id: str,
) -> dict:
    """Prove a conservative static layout, or return explicit rejection reasons.

No buffer recycling is assumed: both product sets and the reconstructed pair
remain allocated. Only the second operation's resident inputs alias that pair.
"""
    validate_upmem_plan(dag, plan)
    nodes = {node.node_id: node for node in dag.nodes}
    if producer_id not in nodes or consumer_id not in nodes or producer_id == consumer_id:
        raise ValueError("resident probe requires two distinct declared nodes")
    first, second = nodes[producer_id], nodes[consumer_id]
    result = {
        "scope": "static_two_contract_float32_probe_admission_v1",
        "logical_plan_id": plan.logical_plan_id, "physical_plan_id": physical_plan_id(plan),
        "producer_id": producer_id, "consumer_id": consumer_id,
        "numeric_policy": plan.numeric_policy,
        "eligible_for_native_probe": False, "runtime_admitted": False,
        "native_execution_implemented": False, "rejection_reasons": [],
        "live_mram_bytes": None, "eliminable_intermediate_payload_bytes": None,
    }
    reasons = result["rejection_reasons"]
    if plan.numeric_policy != "split_complex_float32_v1":
        reasons.append("shared_scale_int8_residency_not_qualified")
    if plan.topology.rank_count != 1:
        reasons.append("one_rank_required")
    if not isinstance(first, ContractNode) or not isinstance(second, ContractNode):
        reasons.append("two_contract_nodes_required")
        return result
    positions = [(i, stage) for i, stage in enumerate(plan.stages)
                 if producer_id in stage.node_ids or consumer_id in stage.node_ids]
    if (len(positions) != 2 or positions[0][1].node_ids != (producer_id,)
            or positions[1][1].node_ids != (consumer_id,)
            or positions[1][0] != positions[0][0] + 1):
        reasons.append("consecutive_single_node_stages_required")
    uses = [side for side in ("left", "right") if getattr(second, side).tensor_id == first.output.id]
    if len(uses) != 1:
        reasons.append("exactly_one_consumer_operand_must_be_resident")
    for node in dag.nodes:
        views = (node.left, node.right) if isinstance(node, ContractNode) else node.inputs
        if node.node_id != consumer_id and any(view.tensor_id == first.output.id for view in views):
            reasons.append("intermediate_has_other_consumers")
            break
    if dag.output.tensor_id == first.output.id:
        reasons.append("intermediate_is_query_output")
    if any(view.slice_spec for node in (first, second) for view in (node.left, node.right)):
        reasons.append("sliced_views_not_supported_by_resident_probe")
    first_orders, second_orders = _orders(first), _orders(second)
    if first_orders[3] or second_orders[3]:
        reasons.append("host_unary_reduction_required")
    if len(uses) == 1:
        side = uses[0]
        result["resident_operand"] = side
        if first_orders[2] != second_orders[0 if side == "left" else 1]:
            reasons.append("intermediate_requires_layout_permutation")
        result["resident_label_order"] = first_orders[2]
    units = {node_id: tuple(unit for stage in plan.stages for unit in stage.work_units
                           if unit.node_id == node_id) for node_id in (producer_id, consumer_id)}
    if any(len(group) != 1 for group in units.values()):
        reasons.append("full_operand_single_tile_without_split_k_required")
    else:
        a, b = units[producer_id][0], units[consumer_id][0]
        if (a.logical_rank, a.logical_dpu) != (b.logical_rank, b.logical_dpu):
            reasons.append("same_dpu_ownership_required")
        result["logical_slot"] = (a.logical_rank, a.logical_dpu)
        if any(unit.batch_start != 0 or unit.batch_size != 1 or unit.m_start != 0
               or unit.n_start != 0 or unit.k_start != 0 for unit in (a, b)):
            reasons.append("whole_unbatched_tile_required")
        for node, unit in ((first, a), (second, b)):
            if (prod(node.output.shape) != unit.m_size * unit.n_size
                    or prod(node.left.shape) != unit.m_size * unit.k_size
                    or prod(node.right.shape) != unit.k_size * unit.n_size):
                reasons.append("whole_unbatched_operand_geometry_required")
    if reasons:
        result["rejection_reasons"] = sorted(set(reasons))
        return result

    first_unit, second_unit = units[producer_id][0], units[consumer_id][0]
    first_shape = (first_unit.m_size, first_unit.n_size, first_unit.k_size)
    second_shape = (second_unit.m_size, second_unit.n_size, second_unit.k_size)
    try:
        first_planes = product_layout(*first_shape, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL)
        second_template = product_layout(*second_shape, numeric_mode=0, kernel=FOUR_PRODUCT_PANEL)
    except ValueError as exc:
        reasons.append("individual_fused_layout_not_admitted")
        result["layout_error"] = str(exc)
        return result
    cursor = sum(length for _, length in first_planes)
    intermediate_span = aligned_bytes(prod(first.output.shape) * 4)
    retained = ((cursor, intermediate_span), (cursor + intermediate_span, intermediate_span))
    cursor += 2 * intermediate_span
    resident_indices = (0, 1) if uses[0] == "left" else (2, 3)
    second_planes = []
    for index, (_, length) in enumerate(second_template):
        if index in resident_indices:
            if length != intermediate_span:
                raise ValueError("resident input byte length disagrees with proven geometry")
            second_planes.append(retained[resident_indices.index(index)])
        else:
            second_planes.append((cursor, length))
            cursor += length
    result.update(first_geometry=first_shape, second_geometry=second_shape,
                  first_planes=first_planes, retained_planes=retained, second_planes=tuple(second_planes),
                  live_mram_bytes=cursor, intermediate_elements=prod(first.output.shape),
                  eliminable_intermediate_payload_bytes=6 * intermediate_span,
                  movement_scope="four_product_fused_control_pair_padded_payload_only",
                  extra_local_reconstruction_payload_bytes=6 * intermediate_span,
                  reconstruction_required="lane_assembly_positive_zero_then_rr_minus_ii_ri_plus_ir_float32",
                  buffer_policy="no_recycling_all_live_v1")
    if cursor > MRAM_BYTES:
        reasons.append("joint_live_mram_exceeds_512KiB")
    result["eligible_for_native_probe"] = not reasons
    return result


def slice_branch_facts(dag: ContractionDAG, original_node: ContractNode) -> dict:
    """Describe every full-shaped partial and its explicit reduction."""
    validate_contraction_dag(dag)
    if original_node.left.slice_spec or original_node.right.slice_spec:
        raise ValueError("nested slice probes are unsupported")
    reductions = [node for node in dag.nodes if isinstance(node, ReduceNode)
                  and node.output.id == original_node.output.id]
    if len(reductions) != 1:
        raise ValueError("sliced contraction must have one explicit output reduction")
    reduction = reductions[0]
    nodes = {node.node_id: node for node in dag.nodes}
    partials = tuple(nodes[node_id] for node_id in reduction.dependencies)
    if not partials or any(not isinstance(node, ContractNode) or node.output.shape != original_node.output.shape
                           or node.output_labels != original_node.output_labels for node in partials):
        raise ValueError("slice partials must preserve all original output indices")
    if (tuple(view.tensor_id for view in reduction.inputs) != tuple(node.output.id for node in partials)
            or any(view.slice_spec for view in reduction.inputs)):
        raise ValueError("slice reduction must consume each full partial once in dependency order")
    labels = reduction.reduced_labels
    dimensions = dict(zip(original_node.left.labels, original_node.left.shape))
    shared = set(original_node.left.labels) & set(original_node.right.labels)
    if not labels or any(label not in original_node.contracted_labels or label not in shared for label in labels):
        raise ValueError("slice reduction must cover original internal indices")
    expected = tuple(tuple(zip(labels, values)) for values in product(*(range(dimensions[label]) for label in labels)))
    observed = []
    for partial in partials:
        assignment = tuple(sorted((original_node.left.labels[axis], value) for axis, value in partial.left.slice_spec))
        right_assignment = tuple(sorted((original_node.right.labels[axis], value) for axis, value in partial.right.slice_spec))
        if right_assignment != assignment:
            raise ValueError("both operands must use the same slice assignment")
        observed.append(assignment)
        for source, view in ((original_node.left, partial.left), (original_node.right, partial.right)):
            remaining = tuple((label, size) for label, size in zip(source.labels, source.shape) if label not in labels)
            if (view.tensor_id != source.tensor_id or view.labels != tuple(label for label, _ in remaining)
                    or view.shape != tuple(size for _, size in remaining)):
                raise ValueError("slice operand differs from the original contraction")
        if partial.contracted_labels != tuple(label for label in original_node.contracted_labels if label not in labels):
            raise ValueError("slice contraction has different reduction semantics")
    if tuple(observed) != expected:
        raise ValueError("slice assignments must cover the complete ordered Cartesian product")
    elements = prod(original_node.output.shape)
    return {"slice_count": len(partials), "partial_node_ids": tuple(node.node_id for node in partials),
            "reduction_node_id": reduction.node_id, "output_elements_per_partial": elements,
            "partial_output_elements_total": len(partials) * elements,
            "output_assembly": "sum_internal_slices_full_shaped_partials",
            "scalar_partial_outputs": elements == 1}
