from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_request_record_invariance.py"
SPEC = importlib.util.spec_from_file_location("analyze_request_record_invariance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_invariance_audit_covers_all_cells_and_source_fields(tmp_path: Path) -> None:
    result = audit.analyze(tmp_path)

    assert len(result["fixtures"]) == 6
    assert len(result["fields"]) == 6 * len(audit.FIELD_CLASSES)
    assert tuple(result["record_field_order"]) == tuple(audit.FIELD_CLASSES)

    variations = {
        item["variation"]: set(item["changed_fields"])
        for item in result["variations"]
        if item["case_id"] == "ghz_chain_18q" and item["dpu_count"] == 1
    }
    assert variations["session_root"] == set()
    assert variations["tasklet_topology"] == set()
    assert variations["request_sequence"] == {"a_path", "b_path", "c_path"}
    assert variations["payload_contents"] == {"a_sha256", "b_sha256"}
    assert "a_transfer_bytes" in variations["numeric_policy"]
    assert "k_elements" in variations["geometry"]

    for artifact in result["artifact_equivalence"]:
        assert artifact["same_relative_artifact_bytes"] is True
        assert artifact["relative_file_count"] == 2 * artifact["dpu_count"] + 2
        assert artifact["baseline_accounting"]["payload_file_count_matches"] is True
        assert artifact["baseline_accounting"]["payload_byte_count_matches"] is True
        assert artifact["baseline_accounting"]["hash_byte_count_matches"] is True


def test_invariance_audit_fails_closed_on_record_schema_drift(monkeypatch) -> None:
    monkeypatch.setitem(audit.FIELD_CLASSES, "unexpected", "plan_static")

    try:
        audit._validate_source_fields()
    except ValueError as error:
        assert "fields changed" in str(error)
    else:  # pragma: no cover - protects the fail-closed contract.
        raise AssertionError("schema drift must fail closed")
