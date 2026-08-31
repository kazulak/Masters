from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host_boundary_inventory.py"
SPEC = importlib.util.spec_from_file_location("host_boundary_inventory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


def _characterization() -> dict[str, object]:
    values = {
        "quantization_stress_18q_l2": (141, 222, 222),
        "hs_18q_d1": (53, 56, 56),
        "ghz_chain_18q": (35, 371, 371),
    }
    candidates = []
    for case_id, (contractions, one_wave, one_work) in values.items():
        candidates.append(
            {
                "candidate_id": case_id,
                "circuit": {"name": case_id},
                "tensor_network": {"contraction_count": contractions},
                "physical_plans": [
                    {
                        "topology": {"dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8},
                        "work_unit_count": one_work,
                        "wave_count": one_wave,
                        "estimated_native_request_count_four_real_products": 4 * one_wave,
                    },
                    {
                        "topology": {"dpu_count": 4, "rank_count": 1, "tasklets_per_dpu": 8},
                        "work_unit_count": one_work,
                        "wave_count": {"quantization_stress_18q_l2": 159, "hs_18q_d1": 53, "ghz_chain_18q": 116}[case_id],
                        "estimated_native_request_count_four_real_products": 4 * {"quantization_stress_18q_l2": 159, "hs_18q_d1": 53, "ghz_chain_18q": 116}[case_id],
                    },
                ],
            }
        )
    return {
        "schema_version": inventory.CHARACTERIZATION_SCHEMA,
        "source_sha": inventory.ACCEPTED_SOURCE,
        "candidates": candidates,
    }


def _source_root(tmp_path: Path) -> Path:
    source = tmp_path / "src" / "quantum_bench" / "upmem"
    source.mkdir(parents=True)
    (source / "native_session.py").write_text(
        'self._write(f"SUBMIT {manifest_rel} {artifact.manifest_sha256}\\n")\n',
        encoding="utf-8",
    )
    return tmp_path


def test_inventory_has_deterministic_six_cell_counts(tmp_path: Path) -> None:
    first = inventory.build_inventory(_characterization(), source_root=_source_root(tmp_path / "one"))
    second = inventory.build_inventory(_characterization(), source_root=_source_root(tmp_path / "two"))

    assert first == second
    assert [(row["case_name"], row["dpu_count"]) for row in first["cells"]] == [
        ("Stress18", 1), ("Stress18", 4),
        ("HS18", 1), ("HS18", 4),
        ("GHZ18", 1), ("GHZ18", 4),
    ]
    assert [
        (
            row["contraction_operation_count"],
            row["wave_request_count"],
            row["embedded_request_count"],
            row["request_record_count"],
            row["packed_operation_submit_estimate"],
        )
        for row in first["cells"]
    ] == [
        (141, 222, 888, 888, 141),
        (141, 159, 636, 2544, 141),
        (53, 56, 224, 224, 53),
        (53, 53, 212, 848, 53),
        (35, 371, 1484, 1484, 35),
        (35, 116, 464, 1856, 35),
    ]
    stress_4 = first["cells"][1]
    assert stress_4["contraction_operation_count"] == 141
    assert stress_4["wave_request_count"] == 159
    assert stress_4["embedded_request_count"] == 636
    assert stress_4["python_submit_count"] == 636
    assert stress_4["request_record_count"] == 2544
    assert stress_4["request_directory_count"] == 636
    assert stress_4["request_payload_file_count"] == 5088
    assert stress_4["request_file_count"] == 6360
    assert stress_4["process_count"] == 1
    assert stress_4["packed_operation_submit_estimate"] == 141


def test_outputs_have_expected_shape_and_import_is_read_only(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path / "source")
    before = set(tmp_path.iterdir())
    json_path = tmp_path / "out" / "inventory.json"
    csv_path = tmp_path / "out" / "inventory.csv"
    markdown_path = tmp_path / "out" / "inventory.md"
    inventory.write_inventory(
        json_path,
        csv_path,
        markdown_path,
        characterization=_characterization(),
        source_root=source_root,
    )

    assert before == {path for path in tmp_path.iterdir() if path.name != "out"}
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["schema_version"] == inventory.SCHEMA_VERSION
    assert record["execution"]["packed_envelope_implemented"] is False
    assert len(record["cells"]) == 6
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert set(rows[0]) == set(inventory.CSV_COLUMNS)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Host Request Boundary Reduction Feasibility v1" in markdown
    assert "packed envelope" in markdown


def test_duplicate_output_paths_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "same"
    with pytest.raises(ValueError, match="distinct"):
        inventory.write_inventory(
            path,
            path,
            tmp_path / "other",
            characterization=_characterization(),
            source_root=_source_root(tmp_path / "source"),
        )
