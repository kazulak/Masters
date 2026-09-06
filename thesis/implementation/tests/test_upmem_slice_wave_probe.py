"""Bounded SDK qualification for fixed-index sliced contraction waves."""

from collections import Counter
from itertools import combinations
from math import prod

import numpy as np
import pytest

from quantum_bench.cpu import replay_upmem_plan_once, run_complex128_reference
from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network, slice_contraction
from quantum_bench.model import ContractNode, ReduceNode, make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.locality_probe import slice_branch_facts
from quantum_bench.upmem.plan import UpmemResources, UpmemTopology, physical_plan_id, plan_upmem
from quantum_bench.upmem.runtime import open_upmem_simulator
from tests import test_upmem_wave_runtime as wave_runtime_tests


wave_runtime_binaries = wave_runtime_tests.binaries
_POLICY = "split_complex_float32_v1"
_PACKED_WAVE = "packed_wave_v1"
_SERIAL = "serial_nodes_v1"
_STATIC = "static_dag_waves_v1"
_DPUS = (2, 4)
_TASKLETS = 8
_TOL = dict(atol=2e-6, rtol=2e-6)
_CASES = (
    ("quantization_stress", {"n_qubits": 4}),
    ("edc", {"n_qubits": 4}),
)

@pytest.fixture(scope="module")
def greedy_dags():
    result = {}
    for name, params in _CASES:
        circuit = builtin_circuit(name, params)
        assert circuit.source["name"] == name
        assert circuit.source["n_qubits"] == 4
        network, inputs = lower_tensor_network(make_simulation_job(circuit))
        path, _ = plan_opt_einsum(network, optimize="greedy")
        result[name] = (build_contraction_dag(network, path), inputs)
    return result


def _plan(dag, schedule, dpu_count):
    return plan_upmem(dag, numeric_policy=_POLICY, schedule_policy=schedule,
                      topology=UpmemTopology(dpu_count=dpu_count,
                                             tasklets_per_dpu=_TASKLETS, rank_count=1))


def _cohort_disjoint(plan, siblings):
    siblings = set(siblings)
    for stage in plan.stages:
        members = tuple(node_id for node_id in stage.node_ids if node_id in siblings)
        if len(members) < 2:
            continue
        slots = [
            {(unit.logical_rank, unit.logical_dpu) for unit in stage.work_units
             if unit.node_id == node_id}
            for node_id in members
        ]
        if all(slots) and all(a.isdisjoint(b) for a, b in combinations(slots, 2)):
            return True
    return False


def _slice_case(base, wanted):
    for target in base.nodes:
        if not isinstance(target, ContractNode) or not any(
            target.node_id in node.dependencies for node in base.nodes
        ):
            continue
        dimensions = dict(zip(target.left.labels, target.left.shape))
        labels = tuple(label for label in target.contracted_labels if dimensions[label] > 1)
        for width in range(1, len(labels) + 1):
            for selected in combinations(labels, width):
                if prod(dimensions[label] for label in selected) != wanted:
                    continue
                dag = slice_contraction(base, node_id=target.node_id, labels=selected)
                facts = slice_branch_facts(dag, target)
                siblings = tuple(facts["partial_node_ids"])
                if all(_cohort_disjoint(_plan(dag, _STATIC, dpu), siblings) for dpu in _DPUS):
                    return {"dag": dag, "target": target, "facts": facts, "siblings": siblings}
    return None


def _assert_slice(base, case):
    dag, target, facts, siblings = (case[key] for key in ("dag", "target", "facts", "siblings"))
    assert dag.output == base.output
    assert facts["slice_count"] == len(siblings)
    elements = prod(target.output.shape)
    assert facts["output_elements_per_partial"] == elements
    assert facts["partial_output_elements_total"] == len(siblings) * elements
    assert facts["output_assembly"] == "sum_internal_slices_full_shaped_partials"
    nodes = {node.node_id: node for node in dag.nodes}
    partials = tuple(nodes[node_id] for node_id in siblings)
    reduction = nodes[facts["reduction_node_id"]]
    assert all(isinstance(node, ContractNode) and node.output_labels == target.output_labels
               and node.output.shape == target.output.shape for node in partials)
    assert isinstance(reduction, ReduceNode)
    assert reduction.output.labels == target.output_labels
    assert reduction.output.shape == target.output.shape
    assert tuple(view.tensor_id for view in reduction.inputs) == tuple(node.output.id for node in partials)
    assert all(view.shape == target.output.shape for view in reduction.inputs)
    assert any(facts["reduction_node_id"] in node.dependencies for node in dag.nodes)


def _assert_order(dag, plan, facts):
    stage_index = {node_id: index for index, stage in enumerate(plan.stages)
                   for node_id in stage.node_ids}
    reduction = facts["reduction_node_id"]
    siblings = facts["partial_node_ids"]
    consumers = [node for node in dag.nodes if reduction in node.dependencies]
    assert consumers
    assert max(stage_index[node_id] for node_id in siblings) < stage_index[reduction]
    assert stage_index[reduction] < min(stage_index[node.node_id] for node in consumers)


def _assert_runtime_cohort(sample, siblings):
    groups = {}
    for operation in sample.backend_facts["operation_facts"]:
        groups.setdefault(operation["cohort_id"], []).append(operation)
    found = False
    for group in groups.values():
        members = [operation for operation in group if operation["node_id"] in siblings]
        if len(members) < 2:
            continue
        assert len({tuple(item["cohort_node_ids"]) for item in members}) == 1
        slots = [set(item["active_dpu_ids"]) for item in members]
        assert all(slots)
        assert all(left.isdisjoint(right) for left, right in combinations(slots, 2))
        found = True
    assert found, "no runtime cohort contained two slice siblings"


def _assert_sample(sample, replay, reference, dag, plan, siblings=()):
    np.testing.assert_array_equal(sample.output, replay.output)
    np.testing.assert_allclose(
        np.asarray(sample.output, dtype=np.complex128), reference, **_TOL
    )
    assert sample.output.shape == dag.output.shape == reference.shape
    facts = sample.backend_facts
    assert facts["logical_plan_id"] == plan.logical_plan_id == contraction_dag_hash(dag)
    assert facts["physical_plan_id"] == physical_plan_id(plan)
    assert facts["schedule_policy"] == plan.schedule_policy
    assert facts["request_transport"] == _PACKED_WAVE
    assert facts["target_observed"] == "sdk_simulator"
    assert facts["simulator_kernel_executed"] is True
    assert facts["cpu_fallback_used"] is False
    assert facts["physical_plan_consumed"] is True
    assert facts["timing_claim_applicable"] is False
    assert facts["complex_launch_policy"] == "four_real_launches_v1"
    contracts = {node.node_id for node in dag.nodes if isinstance(node, ContractNode)}
    operations = tuple(facts["operation_facts"])
    assert len(operations) == len(contracts)
    assert {item["node_id"] for item in operations} == contracts
    assert set(siblings) <= contracts
    assert all(item["lane_pass_count"] == 4 for item in operations)
    numeric = sample.numeric_facts
    assert numeric["numeric_policy"] == _POLICY
    assert {item["node_id"] for item in numeric["operations"]} == contracts
    units = tuple(unit for stage in plan.stages if stage.kind == "contract_batch"
                  for unit in stage.work_units)
    tiles = Counter(unit.node_id for unit in units)
    records = tuple(numeric["raw_lane_records"])
    assert len(records) == 4 * len(units)
    assert Counter(item["node_id"] for item in records) == Counter({node_id: 4 * count
                                                                      for node_id, count in tiles.items()})


def _run(tmp_path, binaries, dag, inputs, reference, schedule, dpu_count, tag, siblings=()):
    plan = _plan(dag, schedule, dpu_count)
    replay = replay_upmem_plan_once(dag, plan, inputs)
    host, dpu, init = binaries[_TASKLETS]
    resources = UpmemResources(session_root=str(tmp_path / tag), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport=_PACKED_WAVE)
    session = open_upmem_simulator(dag, plan, resources, timeout_s=30.0, fuse_complex=False)
    try:
        samples = (session.run_once(inputs), session.run_once(inputs))
    finally:
        closed = session.close()
    assert closed["native_identity_verified"] is True
    assert closed["hardware_release_verified"] is True
    assert not list((tmp_path / tag).rglob("wave-*.bin"))
    for sample in samples:
        _assert_sample(sample, replay, reference, dag, plan, siblings)
    return plan, samples


@pytest.mark.parametrize("case_name", [name for name, _ in _CASES])
def test_sliced_wave_probe_replays_full_statevector(
    tmp_path, greedy_dags, wave_runtime_binaries, case_name
):
    base, inputs = greedy_dags[case_name]
    reference = run_complex128_reference(base, inputs)
    for dpu in _DPUS:
        _run(tmp_path, wave_runtime_binaries, base, inputs, reference, _SERIAL, dpu,
             f"{case_name}-unsliced-d{dpu}")
    cases = {count: _slice_case(base, count) for count in (2, 4)}
    assert cases[2] is not None
    if case_name == "quantization_stress":
        assert cases[4] is not None
    for count, case in cases.items():
        if case is None:
            continue
        _assert_slice(base, case)
        routes = {}
        for schedule in (_SERIAL, _STATIC):
            for dpu in _DPUS:
                plan, samples = _run(
                    tmp_path, wave_runtime_binaries, case["dag"], inputs, reference,
                    schedule, dpu, f"{case_name}-slice{count}-{schedule}-d{dpu}",
                    case["siblings"],
                )
                _assert_order(case["dag"], plan, case["facts"])
                routes[schedule, dpu] = plan, samples
                if schedule == _STATIC:
                    if dpu == 4:
                        assert any(0 < len({(u.logical_rank, u.logical_dpu) for u in s.work_units if u.wave == w}) < dpu
                                   for s in plan.stages if s.kind == "contract_batch" for w in {u.wave for u in s.work_units})
                    for sample in samples:
                        _assert_runtime_cohort(sample, case["siblings"])
                    assert any(
                        item["cohort_idle_h2d_bytes"] > 0
                        for item in samples[0].backend_facts["operation_facts"]
                    )
        for dpu in _DPUS:
            serial, serial_samples = routes[_SERIAL, dpu]
            static, static_samples = routes[_STATIC, dpu]
            assert serial.logical_plan_id == static.logical_plan_id
            assert serial.numeric_policy == static.numeric_policy == _POLICY
            assert serial.topology == static.topology
            assert serial.kernel_policy == static.kernel_policy
            assert serial.intermediate_policy == static.intermediate_policy
            assert physical_plan_id(serial) != physical_plan_id(static)
            for serial_sample, static_sample in zip(serial_samples, static_samples):
                np.testing.assert_array_equal(serial_sample.output, static_sample.output)
