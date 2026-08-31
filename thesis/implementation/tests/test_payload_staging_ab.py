from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inspect_payload_staging_ab.py"
SPEC = importlib.util.spec_from_file_location("inspect_payload_staging_ab", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_ab_contract_uses_only_two_routes_and_measured_blocks() -> None:
    assert module.CASES == (
        "quantization_stress_18q_l2",
        "hs_18q_d1",
        "ghz_chain_18q",
    )
    assert module.ROUTES == (
        "upmem_float32_1dpu_t8",
        "upmem_float32_4dpu_t8",
    )
    assert module.BLOCKS == (0, 1, 2, 3, 4, 5)
    assert module.MEASUREMENTS == (1, 2, 3, 4, 5)


def test_scientific_configuration_ignores_only_run_specific_paths() -> None:
    manifest = {
        "configuration": {
            "experiment": {
                "experiment_id": "run-specific",
                "label": "run-specific",
                "experiment_identity_payload": {
                    "configuration": {"experiment_id": "run-specific", "path": "/tmp/a"}
                },
                "routes": {
                    "route": {
                        "options": {
                            "session_root": "/tmp/a",
                            "host_binary": "/tmp/host-a",
                            "dpu_binary": "/tmp/dpu-a",
                            "initialization_binary": "/tmp/init-a",
                            "dpu_count": 1,
                        }
                    }
                },
            }
        }
    }
    normalized = module._scientific_configuration(manifest)
    assert "experiment_id" not in normalized
    assert "label" not in normalized
    assert "experiment_identity_payload" not in normalized
    assert normalized["routes"]["route"]["options"] == {"dpu_count": 1}


def test_joined_facts_uses_bound_terminal_physical_facts() -> None:
    sample = {
        "backend_facts": {"cpu_fallback_used": False},
        "session_instance_id": "session-1",
    }
    sessions = {
        "session-1": {
            "terminal_backend_facts": {
                "physical_target_verified": True,
                "hardware_kernel_executed": True,
            }
        }
    }

    assert module._joined_facts(sample, sessions) == {
        "cpu_fallback_used": False,
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
    }


def test_main_fails_closed_when_ab_gate_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(module, "inspect", lambda **_: {"gate_passed": False})

    assert module.main(
        [
            "--baseline", str(tmp_path / "baseline"),
            "--candidate", str(tmp_path / "candidate"),
            "--baseline-source", "a" * 40,
            "--candidate-source", "b" * 40,
            "--output-dir", str(tmp_path / "output"),
        ]
    ) == 1
