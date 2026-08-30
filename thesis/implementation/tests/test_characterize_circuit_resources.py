from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "characterize_circuit_resources.py"
SPEC = importlib.util.spec_from_file_location("characterize_circuit_resources", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
characterize = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = characterize
SPEC.loader.exec_module(characterize)


def test_characterization_is_deterministic_and_covers_fixed_matrix(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    characterize.write_characterization(first)
    characterize.write_characterization(second)
    assert (first / "characterization.json").read_bytes() == (second / "characterization.json").read_bytes()
    assert (first / "characterization.csv").read_bytes() == (second / "characterization.csv").read_bytes()
    record = json.loads((first / "characterization.json").read_text(encoding="utf-8"))
    assert [candidate["candidate_id"] for candidate in record["candidates"]] == [
        "quantization_stress_18q_l2", "ghz_chain_18q", "bv_18q", "hs_18q_d1"
    ]
    expected = {(1, tasklets) for tasklets in range(1, 25)} | {(dpus, 8) for dpus in range(2, 5)}
    for candidate in record["candidates"]:
        assert len(candidate["physical_plans"]) == 27
        assert {(p["topology"]["dpu_count"], p["topology"]["tasklets_per_dpu"]) for p in candidate["physical_plans"]} == expected
        for physical in candidate["physical_plans"]:
            assert physical["estimated_native_payload_record_count_four_real_products"] == 4 * physical["work_unit_count"]
    with (first / "characterization.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 4 * 27
    characterize.check_characterization(first)


def test_characterization_is_source_only_and_selection_uses_no_timing(tmp_path: Path) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "open_upmem(" not in source
    assert "open_upmem_simulator(" not in source
    output = tmp_path / "characterization"
    characterize.write_characterization(output)
    record = json.loads((output / "characterization.json").read_text(encoding="utf-8"))
    assert record["execution"] == {"hardware_executed": False, "kind": "source_only_characterization", "simulator_executed": False}
    assert "planning_time" not in json.dumps(record, sort_keys=True)
    assert "entropy" not in json.dumps(record, sort_keys=True)
    selection = characterize.build_selection(record)
    assert selection["selected_case_ids"] == ["quantization_stress_18q_l2", "hs_18q_d1", "ghz_chain_18q"]
    assert selection["selection_rule"]["timing_used"] is False
