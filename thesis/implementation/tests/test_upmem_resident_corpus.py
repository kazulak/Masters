"""Replay the frozen Stress16 pair through the isolated SDK resident probe."""

import hashlib
import json
from pathlib import Path

import numpy as np

from quantum_bench import cpu
from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network
from quantum_bench.model import make_simulation_job
from quantum_bench.upmem.locality_probe import resident_pair_probe_layout
from quantum_bench.upmem.plan import UpmemTopology, physical_plan_id, plan_upmem
from tests import test_upmem_resident_kernel_simulator as probe


resident_sdk = probe.sdk


def test_selected_stress_pair_preserves_complete_statevector(resident_sdk, monkeypatch):
    pool = Path(__file__).resolve().parents[1] / "thesis_results/upmem_path_heuristic_generalization_v1/software/candidate_paths.json"
    data = pool.read_bytes()
    assert hashlib.sha256(data).hexdigest() == "d95150ddf89f6aafa861000b0db2d8447d64456a035c5404463a878c3a319049"
    circuit = next(c for c in json.loads(data)["circuits"] if c["circuit_id"] == "quantization_stress_16q_l2")
    candidate = next(p for p in circuit["candidates"] if p["is_greedy"])
    assert candidate["candidate_path_id"] == "31a9997e87f38005e081aad56952a30bd31277dafe425c44de78613365a45e02"
    network, inputs = lower_tensor_network(make_simulation_job(
        builtin_circuit(circuit["circuit"]["name"], circuit["circuit"]["parameters"])))
    dag = build_contraction_dag(network, tuple(tuple(pair) for pair in candidate["path"]))
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1",
                      topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1))
    assert plan.logical_plan_id == candidate["logical_plan_id"]
    assert physical_plan_id(plan) == "90b181769e4c2c061f2b44e7fa5f61df39ac0a9354b86ad38f6b0db65c4bb3df"
    names = ("contract_121", "contract_122")
    admitted = resident_pair_probe_layout(dag, plan, *names)
    assert admitted["eligible_for_native_probe"] and admitted["live_mram_bytes"] == 93184

    # Observe the accepted CPU policy at its exact tile boundary, without
    # regenerating a path or changing any contraction/reduction geometry.
    captured = {}
    original = cpu._replay_tile_lanes

    def observe(left, right, unit, policy):
        lanes = original(left, right, unit, policy)
        if unit.node_id in names:
            assert unit.node_id not in captured
            captured[unit.node_id] = {
                "inputs": tuple(a.copy() for a in (left.real, left.imag, right.real, right.imag)),
                "products": tuple(a.copy() for a in lanes),
            }
        return lanes

    monkeypatch.setattr(cpu, "_replay_tile_lanes", observe)
    expected = cpu.replay_upmem_plan_once(dag, plan, inputs)
    monkeypatch.setattr(cpu, "_replay_tile_lanes", original)
    assert set(captured) == set(names)
    info = probe.make_plan(admitted["first_geometry"], admitted["second_geometry"], "right", 1, 8)
    assert info["retained"] == admitted["retained_planes"]
    assert info["controls"][0].planes == admitted["first_planes"]
    assert info["controls"][1].planes == admitted["second_planes"]
    arena = bytearray(probe.GUARD * probe.MRAM_BYTES)
    for span, array in zip(info["controls"][0].planes[:4], captured[names[0]]["inputs"]):
        raw = array.tobytes()
        arena[span[0]:span[0] + len(raw)] = raw
    for span, array in zip(info["controls"][1].planes[:2], captured[names[1]]["inputs"][:2]):
        raw = array.tobytes()
        arena[span[0]:span[0] + len(raw)] = raw
    resident_values = captured[names[1]]["inputs"][2:]
    host_patch = b"".join(array.tobytes() + probe.GUARD * (span[1] - array.nbytes)
                          for span, array in zip(info["retained"], resident_values))
    item = {"info": info}
    first = probe.request(1, item, 0, bytes(arena))
    baseline = probe.run(resident_sdk, 8, (first, probe.request(2, item, info["retained"][0][0], host_patch)))
    resident = probe.run(resident_sdk, 8, (first, probe.request(3, item, 0, b"")))
    for arm in (baseline, resident):
        for record, control in zip(arm, info["controls"]):
            probe.assert_success(record, control)
        probe.assert_products(arm[0][1], info["controls"][0].planes[4:], captured[names[0]]["products"])
        probe.assert_products(arm[1][1], info["retained"], resident_values)
        probe.assert_products(arm[1][1], info["controls"][1].planes[4:], captured[names[1]]["products"])
    assert baseline[1][1] == resident[1][1]

    # Feed only the measured consumer lanes into the original reference DAG.
    # This is a test oracle, not a production CPU/DPU placement policy.
    control = info["controls"][1]
    native_lanes = tuple(np.frombuffer(resident[1][1], dtype="<f4", count=control.m * control.n,
                                      offset=span[0]).reshape(control.m, control.n).copy()
                         for span in control.planes[4:])
    injections = []

    def inject(left, right, unit, policy):
        if unit.node_id == names[1]:
            injections.append(unit.stable_tile_id)
            return native_lanes
        return original(left, right, unit, policy)

    monkeypatch.setattr(cpu, "_replay_tile_lanes", inject)
    reconstructed = cpu.replay_upmem_plan_once(dag, plan, inputs)
    assert len(injections) == 1
    assert reconstructed.output.size == 65536
    np.testing.assert_array_equal(reconstructed.output, expected.output)
    np.testing.assert_allclose(reconstructed.output, cpu.run_complex128_reference(dag, inputs), atol=2e-6, rtol=2e-6)
