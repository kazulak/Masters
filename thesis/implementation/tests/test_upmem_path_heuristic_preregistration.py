from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "upmem_path_heuristic_v1.json"


def test_preregistration_freezes_splits_topologies_and_search() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["schema_version"] == "upmem_path_heuristic_preregistration_v1"
    assert record["candidate_generation"] == {
        "cotengra_method": "greedy",
        "cotengra_objective": "flops",
        "master_seed": 20260902,
        "maximum_planned_work_units": 500,
        "one_trial_searches": 64,
        "opt_einsum_reference": "greedy",
        "physical_lowering_timeout_s": 60.0,
        "physical_lowering_worker_address_space_bytes": 2147483648,
    }
    splits = {
        item["circuit_id"]: item["split"]
        for item in record["circuits"]
    }
    assert splits == {
        "quantization_stress_18q_l2": "training",
        "hs_18q_d1": "training",
        "ghz_chain_18q": "training",
        "bv_16q": "training",
        "edc_16q": "validation",
        "bv_18q": "test",
    }
    assert [item["topology_id"] for item in record["topologies"]] == [
        "1dpu_t8",
        "4dpu_t8",
    ]
    assert record["calibration"] == {
        "candidates_per_cell_maximum": 6,
        "measurement_blocks": 3,
        "warmup_blocks": 1,
    }
    assert record["validation_and_test"]["measurement_blocks"] == 5


def test_preregistration_contains_no_observed_runtime_or_fitted_weights() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in (
        "total_wall",
        "kernel_s",
        "measured_runtime",
        "fitted_weights",
        "speedup",
    ):
        assert forbidden not in lowered
