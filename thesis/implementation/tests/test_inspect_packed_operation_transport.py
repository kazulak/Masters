from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inspect_packed_operation_transport.py"
SPEC = importlib.util.spec_from_file_location("inspect_packed_operation_transport", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
inspector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspector
SPEC.loader.exec_module(inspector)


def _sample(*, transport: str = "packed_operation_v1") -> tuple[dict[str, object], dict[str, object]]:
    session_id = "session-1"
    facts = {
        "request_transport": transport,
        "packed_operation_count": 2,
        "packed_operation_request_count": 8,
        "packed_operation_max_descriptor_count": 4,
        "packed_operation_max_bytes": 1024,
        "packed_operation_max_payload_bytes": 768,
    }
    sample = {"session_instance_id": session_id, "route_id": "route", "backend_facts": facts}
    session = {
        "session_instance_id": session_id,
        "terminal_backend_facts": {"request_transport": transport},
    }
    return sample, session


def test_packed_transport_facts_are_required_and_summarized() -> None:
    sample, session = _sample()
    summary = inspector.validate_packed_transport([sample], [session])
    assert summary["transport"] == "packed_operation_v1"
    assert summary["route_facts"]["route"]["max_descriptor_count"] == 4


def test_directory_transport_is_rejected() -> None:
    sample, session = _sample(transport="directory_v1")
    with pytest.raises(ValueError, match="packed_operation_v1"):
        inspector.validate_packed_transport([sample], [session])


def test_missing_packed_counter_is_rejected() -> None:
    sample, session = _sample()
    del sample["backend_facts"]["packed_operation_max_bytes"]
    with pytest.raises(ValueError, match="packed_operation_max_bytes"):
        inspector.validate_packed_transport([sample], [session])
