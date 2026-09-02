from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_upmem_path_evaluation.py"
SPEC = importlib.util.spec_from_file_location("analyze_upmem_path_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def _fixture() -> tuple[dict, dict]:
    candidate_id = "a" * 64
    topology = {
        "topology_id": "1dpu_t8",
        "feasible": True,
        "physical_plan_id": "b" * 64,
        "topology": {"dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8},
        "resource_admission": {"collection_resource_admission_passed": True},
    }
    dataset = {
        "source_sha": "c" * 40,
        "circuits": [{
            "circuit_id": "held-out",
            "split": "test",
            "candidates": [{
                "candidate_path_id": candidate_id,
                "logical_plan_id": "d" * 64,
                "topologies": [topology],
            }],
        }],
    }
    selection = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "split": "test",
        "timing_used_for_selection": False,
        "candidate_set_sha256": analyzer._sha256(analyzer._canonical_bytes(dataset)),
        "selections": [{
            "circuit_id": "held-out",
            "topology_id": "1dpu_t8",
            "greedy_path_id": candidate_id,
            "minimum_flops_path_id": candidate_id,
            "upmem_selected_path_id": candidate_id,
            "upmem_score": 0.0,
            "explanation": [],
        }],
    }
    return dataset, selection


def test_raw_median_and_mad_are_sample_paired() -> None:
    values = [1.0, 2.0, 100.0, 3.0, 4.0]
    assert analyzer._median(values) == 3.0
    assert analyzer._mad(values) == 1.0


def test_expected_deduplicates_coincident_path_roles() -> None:
    dataset, selection = _fixture()
    cells, paths = analyzer._expected(dataset, selection, "test")
    assert set(cells) == {("held-out", "1dpu_t8")}
    path = next(iter(paths.values()))
    assert path["roles"] == ["greedy", "minimum_flops", "upmem_selected"]


def test_expected_rejects_timing_dependent_selection() -> None:
    dataset, selection = _fixture()
    selection["timing_used_for_selection"] = True
    with pytest.raises(ValueError, match="timing-independence"):
        analyzer._expected(dataset, selection, "test")
