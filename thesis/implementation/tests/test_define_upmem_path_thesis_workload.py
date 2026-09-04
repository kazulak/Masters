from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "define_upmem_path_thesis_workload.py"
SPEC = importlib.util.spec_from_file_location("define_upmem_path_thesis_workload", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
define = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = define
SPEC.loader.exec_module(define)

CONFIG = ROOT / "configs" / "upmem_path_thesis_workload_v1.json"
PILOT_CONFIG = ROOT / "configs" / "upmem_path_heuristic_v1.json"


def _config_copy(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path, value


def _remove_final_tests(value: dict[str, object]) -> None:
    for item in value["declared_workload"]:
        if item["split"] == "test":
            item["split"] = "training"
    for item in value["circuits"]:
        if item["split"] == "test":
            item["split"] = "training"


def test_manifest_is_deterministic_and_freezes_workload_contract(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    define.write_manifest(first, CONFIG, PILOT_CONFIG)
    define.write_manifest(second, CONFIG, PILOT_CONFIG)

    assert (first / define.MANIFEST_FILENAME).read_bytes() == (
        second / define.MANIFEST_FILENAME
    ).read_bytes()
    assert (first / define.CSV_FILENAME).read_bytes() == (
        second / define.CSV_FILENAME
    ).read_bytes()

    manifest = json.loads(
        (first / define.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == define.MANIFEST_SCHEMA_VERSION
    assert len(manifest["workload"]) == 15
    assert manifest["summary"] == {
        "correctness_only_count": 3,
        "declared_instance_count": 15,
        "eligible_family_count": 6,
        "heuristic_eligible_count": 12,
        "pilot_development_count": 6,
        "unobserved_eligible_count": 6,
        "unobserved_final_test_count": 2,
    }
    assert manifest["contract"]["numeric_policy"] == define.NUMERIC_POLICY
    assert manifest["contract"]["transport"] == define.TRANSPORT
    assert manifest["contract"]["topologies"] == list(define.TOPOLOGIES)
    assert manifest["contract"]["calibration"]["candidate_roles"] == [
        "greedy",
        "minimum_flops",
        "minimum_peak_intermediate",
        "minimum_writes",
        "frozen_v1_selected",
        "feature_diverse",
    ]
    assert manifest["contract"]["calibration"]["frozen_v1_profile"][
        "sha256"
    ] == "cc1e3deb6b5a227b4efe9c84e43679d385cb9b65da76e293ad0d074889cb868a"
    assert len(manifest["source_sha"]) == 40
    for field in (
        "config_sha256",
        "pilot_config_sha256",
        "circuits_sha256",
        "descriptor_source_sha256",
        "generator_source_sha256",
    ):
        assert len(manifest[field]) == 64

    by_id = {item["circuit_id"]: item for item in manifest["workload"]}
    assert by_id["bv_18q"]["split"] == "pilot_development"
    assert by_id["bv_18q"]["historical_physical_status"] == define.PILOT_STATUS
    assert by_id["ghz_chain_14q"]["split"] == "test"
    assert by_id["ghz_chain_14q"]["historical_physical_status"] == define.UNOBSERVED_STATUS
    assert by_id["bell_2q"]["classification"] == "correctness_only"
    assert by_id["bell_2q"]["circuit_definition"] == {
        "kind": "builtin",
        "name": "bell_2q",
        "parameters": {},
    }
    assert by_id["quantization_stress_16q_l2"]["descriptors"]["two_qubit_gate_count"] > 0

    with (first / define.CSV_FILENAME).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    assert tuple(rows[0]) == define.CSV_COLUMNS
    assert {row["circuit_id"] for row in rows} == set(by_id)

    define.check_manifest(first, CONFIG, PILOT_CONFIG)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("numeric_policy", "complex_int8_shared_scale_v1"), "numeric_policy"),
        (lambda value: value["topologies"].__setitem__(0, {**value["topologies"][0], "dpu_count": 2}), "topologies"),
        (lambda value: next(item for item in value["declared_workload"] if item["circuit_id"] == "bv_18q").__setitem__("split", "test"), "pilot"),
        (_remove_final_tests, "final tests"),
    ],
)
def test_manifest_rejects_contract_and_split_violations(
    tmp_path: Path, mutation: object, message: str
) -> None:
    config_path, value = _config_copy(tmp_path)
    mutation(value)
    config_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        define.build_manifest(config_path, PILOT_CONFIG)


def test_manifest_does_not_execute_backend_or_planner() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "open_upmem(" not in source
    assert "open_upmem_simulator(" not in source
    assert "plan_opt_einsum(" not in source
    assert "plan_cotengra(" not in source
