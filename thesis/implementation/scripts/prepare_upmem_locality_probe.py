#!/usr/bin/env python3
"""Prepare two development locality probes from the retained census, no execution."""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from math import prod
from pathlib import Path
import subprocess

from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network, slice_contraction
from quantum_bench.model import ContractNode, make_simulation_job
from quantum_bench.upmem.locality_probe import resident_pair_probe_layout, slice_branch_facts
from quantum_bench.upmem.plan import UpmemTopology, physical_plan_id, plan_upmem
from quantum_bench.upmem.tiling import canonical_label_geometry
from quantum_bench.upmem.wave_protocol import COMPLETION_BYTES, CONTROL, REAL_PANEL, product_layout


ROOT = Path(__file__).resolve().parents[1]
CENSUS_SHA256 = "b26a20e821c1510c6975c4990b2224c42d3f656cf98c0e14c958d7cfe19c3095"
CIRCUITS = ("quantization_stress_16q_l2", "edc_14q")
POLICY = "split_complex_float32_v1"


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def plan_counts(plan):
    """Planned unfused padded payload/control counts, not measured traffic."""
    if plan.numeric_policy != POLICY:
        raise ValueError("locality probe counters require split-complex float32")
    waves = active = macs = payload_in = payload_out = 0
    for stage in plan.stages:
        units = stage.work_units
        if not units:
            continue
        groups = (units,) if plan.schedule_policy == "static_dag_waves_v1" else tuple(
            tuple(u for u in units if u.node_id == node) for node in stage.node_ids)
        waves += sum(len({u.wave for u in group}) for group in groups)
        for unit in units:
            spans = product_layout(unit.m_size, unit.n_size, unit.k_size, numeric_mode=0, kernel=REAL_PANEL)
            active += 1
            macs += 4 * unit.m_size * unit.n_size * unit.k_size
            payload_in += 4 * (spans[0][1] + spans[2][1])
            payload_out += 4 * spans[4][1]
    slots = waves * plan.topology.dpu_count
    return {"scope": "planned_unfused_panel_padded_payload_control_completion_v1",
            "physical_plan_id": physical_plan_id(plan), "logical_plan_id": plan.logical_plan_id,
            "schedule_policy": plan.schedule_policy, "work_unit_count": active,
            "packed_cohort_count": sum(len(s.node_ids) if plan.schedule_policy == "serial_nodes_v1" else 1
                                       for s in plan.stages if s.work_units),
            "launch_count": 4 * waves, "active_slot_launch_count": 4 * active,
            "idle_slot_launch_count": 4 * (slots - active), "real_mac_count": macs,
            "padded_h2d_payload_bytes": payload_in, "padded_d2h_payload_bytes": payload_out,
            "control_h2d_bytes": 4 * slots * CONTROL.size, "completion_d2h_bytes": 4 * slots * COMPLETION_BYTES,
            "host_reduce_count": sum(s.kind == "host_reduce" for s in plan.stages)}


def choose_slice(dag):
    choices = []
    for node in dag.nodes:
        if not isinstance(node, ContractNode) or node.left.slice_spec or node.right.slice_spec:
            continue
        dimensions = dict(zip(node.left.labels, node.left.shape))
        shared = set(node.left.labels) & set(node.right.labels)
        labels = tuple(sorted(label for label in node.contracted_labels if label in shared and 1 < dimensions[label] <= 4))
        b, m, k, n = canonical_label_geometry(node.left.labels, node.left.shape,
                                            node.right.labels, node.right.shape, node.output_labels)
        for count in (1, 2):
            for chosen in combinations(labels, count):
                slices = prod(dimensions[label] for label in chosen)
                if slices in (2, 4):
                    choices.append((-4 * b * m * k * n, -slices, node.node_id, chosen))
    # The highest-work deterministic decomposition must actually expose sibling
    # concurrency at both small resource points; no timings enter this choice.
    for _, _, node_id, labels in sorted(choices):
        sliced = slice_contraction(dag, node_id=node_id, labels=labels)
        original = next(n for n in dag.nodes if n.node_id == node_id)
        facts = slice_branch_facts(sliced, original)
        siblings = set(facts["partial_node_ids"])
        arms = []
        for dpus in (2, 4):
            topology = UpmemTopology(dpu_count=dpus, tasklets_per_dpu=8, rank_count=1)
            serial = plan_upmem(sliced, numeric_policy=POLICY, topology=topology)
            concurrent = plan_upmem(sliced, numeric_policy=POLICY, topology=topology, schedule_policy="static_dag_waves_v1")
            cohorts = [s for s in concurrent.stages if len(siblings.intersection(s.node_ids)) >= 2]
            if not cohorts:
                break
            unsliced = plan_upmem(dag, numeric_policy=POLICY, topology=topology)
            counts = [plan_counts(p) for p in (unsliced, serial, concurrent)]
            arms.append({"dpu_count": dpus, "tasklets": 8, "unsliced_serial": counts[0],
                         "sliced_serial": counts[1], "sliced_concurrent": counts[2],
                         "sibling_cohort_node_ids": [s.node_ids for s in cohorts],
                         "real_mac_work_ratio": counts[1]["real_mac_count"] / counts[0]["real_mac_count"]})
        if len(arms) == 2:
            return {"node_id": node_id, "labels": labels, "facts": facts, "arms": arms,
                    "logical_plan_id": contraction_dag_hash(sliced),
                    "selection_rule": "descending_original_real_macs_then_slice_count_then_node_and_labels"}
    return {"status": "no_admitted_concurrent_slice", "candidate_count": len(choices)}


def _source_identity(*, allow_dirty):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    if dirty and not allow_dirty:
        raise ValueError("clean source required; use --allow-dirty-preview only for development")
    return source, dirty


def prepare(census_path, *, allow_dirty=False):
    data = census_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CENSUS_SHA256:
        raise ValueError("frozen frontier census checksum mismatch")
    census = json.loads(data)
    source, dirty = _source_identity(allow_dirty=allow_dirty)
    subprocess.run(["git", "merge-base", "--is-ancestor", census["source_sha"], source], cwd=ROOT, check=True)
    rows = []
    eligible = []
    for circuit_id in CIRCUITS:
        cells = [c for c in census["cells"] if c["circuit_id"] == circuit_id and c["numeric_policy"] == POLICY
                 and c["topology"]["dpu_count"] == 1 and "greedy" in c["path_roles"]]
        if len(cells) != 1 or cells[0]["status"] != "eligible":
            raise ValueError(f"missing unique frozen greedy circuit: {circuit_id}")
        cell = cells[0]
        network, _ = lower_tensor_network(make_simulation_job(builtin_circuit(cell["circuit"]["name"],
                                                                             cell["circuit"]["parameters"])))
        dag = build_contraction_dag(network, tuple(tuple(pair) for pair in cell["path"]))
        plan = plan_upmem(dag, numeric_policy=POLICY, topology=UpmemTopology(**cell["topology"]))
        if contraction_dag_hash(dag) != cell["logical_plan_id"] or physical_plan_id(plan) != cell["physical_plan_id"]:
            raise ValueError("frozen logical/physical identity mismatch")
        producers = {n.output.id: n for n in dag.nodes if isinstance(n, ContractNode)}
        pairs = []
        for consumer in dag.nodes:
            if isinstance(consumer, ContractNode):
                for tensor_id in sorted({consumer.left.tensor_id, consumer.right.tensor_id} & producers.keys()):
                    admission = resident_pair_probe_layout(dag, plan, producers[tensor_id].node_id, consumer.node_id)
                    pairs.append(admission)
                    if admission["eligible_for_native_probe"]:
                        eligible.append({"circuit_id": circuit_id, **admission})
        reasons = Counter(reason for row in pairs for reason in row["rejection_reasons"])
        rows.append({"circuit_id": circuit_id, "circuit": cell["circuit"], "candidate_path_id": cell["candidate_path_id"],
                     "logical_plan_id": cell["logical_plan_id"], "physical_plan_id": cell["physical_plan_id"],
                     "resident_pairs": pairs, "resident_rejection_counts": dict(sorted(reasons.items())),
                     "slice_probe": choose_slice(dag)})
    eligible.sort(key=lambda r: (-r["eliminable_intermediate_payload_bytes"], r["circuit_id"],
                                r["producer_id"], r["consumer_id"]))
    return {"scope": "bounded_locality_probe_preparation_v1", "source_sha": source, "source_dirty": dirty,
            "preview_only": dirty, "census_source_sha": census["source_sha"], "census_ancestry_verified": True,
            "census_sha256": CENSUS_SHA256, "numeric_policy": POLICY, "request_transport": "packed_wave_v1",
            "slice_kernel_policy": "panel_only_v1_unfused", "hardware_execution": False,
            "sdk_execution": False, "timing_claim_applicable": False, "circuits": rows,
            "resident_selected_pair": eligible[0] if eligible else None,
            "resident_eligible_pair_count": len(eligible), "resident_native_implementation_pending": True,
            "physical_execution_authorized_by_this_artifact": False,
            "physical_gate": "P0_and_source_SDK_qualification_and_separate_budgeted_packet_required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-preview", action="store_true")
    args = parser.parse_args()
    result = prepare(args.census, allow_dirty=args.allow_dirty_preview)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "locality_probe_preparation.json"
    target.write_bytes(canonical_bytes(result))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.output / "SHA256SUMS").write_text(f"{digest}  {target.name}\n")
    print(json.dumps({"artifact": str(target), "sha256": digest,
                      "resident_eligible_pairs": result["resident_eligible_pair_count"],
                      "selected_pair": result["resident_selected_pair"]}, sort_keys=True))


if __name__ == "__main__":
    main()
