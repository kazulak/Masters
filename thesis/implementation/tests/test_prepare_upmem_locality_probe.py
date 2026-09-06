"""Probe preparation is deterministic and does not launch or regenerate paths."""

import importlib.util
from pathlib import Path

import pytest

from quantum_bench.upmem.plan import UpmemTopology, plan_upmem
from tests.test_upmem_locality_probe import pair
from tests.test_upmem_wave_runtime import fork_join


SPEC = importlib.util.spec_from_file_location("prepare_locality", Path(__file__).resolve().parents[1]
                                            / "scripts/prepare_upmem_locality_probe.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rejects_changed_census_before_lowering(tmp_path):
    path = tmp_path / "census.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        MODULE.prepare(path)


def test_dirty_source_requires_explicit_preview(monkeypatch):
    def output(command, **kwargs):
        return "a" * 40 if command[1] == "rev-parse" else "?? untracked-file"
    monkeypatch.setattr(MODULE.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="clean source required"):
        MODULE._source_identity(allow_dirty=False)
    assert MODULE._source_identity(allow_dirty=True) == ("a" * 40, True)


def test_counters_reject_other_numeric_policies():
    plan = plan_upmem(pair(), numeric_policy="complex_int8_shared_scale_v1",
                      topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1))
    with pytest.raises(ValueError, match="require split-complex float32"):
        MODULE.plan_counts(plan)


def test_planned_counters_distinguish_real_launches_and_idle_slots():
    dag, _ = fork_join(k=3)
    topo = UpmemTopology(dpu_count=4, tasklets_per_dpu=8, rank_count=1)
    serial = plan_upmem(dag, numeric_policy=MODULE.POLICY, topology=topo)
    parallel = plan_upmem(dag, numeric_policy=MODULE.POLICY, topology=topo, schedule_policy="static_dag_waves_v1")
    a, b = (MODULE.plan_counts(p) for p in (serial, parallel))
    assert (a["launch_count"], b["launch_count"]) == (12, 8)
    assert (a["packed_cohort_count"], b["packed_cohort_count"]) == (3, 2)
    assert (a["idle_slot_launch_count"], b["idle_slot_launch_count"]) == (36, 20)
    assert a["real_mac_count"] == b["real_mac_count"] == 128
    assert a["padded_h2d_payload_bytes"] == b["padded_h2d_payload_bytes"] == 512
    assert a["padded_d2h_payload_bytes"] == b["padded_d2h_payload_bytes"] == 192
    assert a["control_h2d_bytes"] == 12 * 4 * 144
    assert b["completion_d2h_bytes"] == 8 * 4 * 72


def test_slice_choice_is_deterministic_and_keeps_all_partials():
    dag = pair(k=4)
    a, b = MODULE.choose_slice(dag), MODULE.choose_slice(dag)
    assert MODULE.canonical_bytes(a) == MODULE.canonical_bytes(b)
    assert a["facts"]["slice_count"] in (2, 4)
    assert len(a["arms"]) == 2
    for arm in a["arms"]:
        assert arm["real_mac_work_ratio"] == 1
        assert arm["sibling_cohort_node_ids"]
        assert arm["sliced_serial"]["logical_plan_id"] == arm["sliced_concurrent"]["logical_plan_id"]
        assert arm["sliced_serial"]["logical_plan_id"] != arm["unsliced_serial"]["logical_plan_id"]
        assert arm["sliced_serial"]["host_reduce_count"] == 1
        assert arm["sliced_concurrent"]["host_reduce_count"] == 1


def test_no_slice_target_is_an_explicit_result():
    result = MODULE.choose_slice(pair(m=1, n=1, k=1, q=1))
    assert result == {"status": "no_admitted_concurrent_slice", "candidate_count": 0}
