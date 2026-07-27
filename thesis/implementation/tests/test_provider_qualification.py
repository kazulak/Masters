from __future__ import annotations
import json
import os
from pathlib import Path
import pytest
import quantum_bench.bench.provider_qualification as harness
from quantum_bench.providers.qualification import (
    SIMPLEPIM_RUNNER_SCHEMA_VERSION,
    load_provider_catalog,
    parse_runner_result,
    provider_source_fingerprint,
    simplepim_runner_schema_errors,
    source_gate_failure,
)
ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "configs/qualification/upmem_provider_m1.yml"
def test_catalog_keeps_executable_and_blocked_lanes_truthful() -> None:
    catalog = load_provider_catalog(CATALOG)
    assert catalog.get("simplepim").executable is True
    assert {p.provider_id for p in catalog.providers if p.status == "blocked"} == {
        "atim",
        "pid-comm",
        "sparsep",
    }
    assert all(not p.executable for p in catalog.providers if p.status == "blocked")
def test_prepare_plan_is_unique_and_does_not_build_or_allocate(tmp_path: Path) -> None:
    result = harness.prepare_provider_qualification(ROOT, catalog_path=CATALOG)
    second = harness.prepare_provider_qualification(ROOT, catalog_path=CATALOG)
    assert result.status == second.status == "prepared"
    assert result.plan_dir != second.plan_dir
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assert plan["execution_policy"] == {
        "mode": "prepare-only",
        "build_attempted": False,
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "runner_execute_invoked": False,
    }
    simplepim = next(row for row in plan["providers"] if row["id"] == "simplepim")
    assert simplepim["runner_prepare"]["status"] == "prepared"
def test_prepare_reports_blocked_provider_without_invoking_runner() -> None:
    result = harness.prepare_provider_qualification(ROOT, catalog_path=CATALOG, provider_id="atim")
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    row = plan["providers"][0]
    assert result.status == "prepared"
    assert row["preparation_status"] == "blocked"
    assert row["runner_prepare"] is None
def test_source_gate_requires_pinned_clean_checkout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    provider = load_provider_catalog(CATALOG).get("simplepim")
    provider = provider.__class__(**{**provider.__dict__, "source_path": "source", "pinned_commit": "deadbeef"})
    fingerprint = provider_source_fingerprint(provider, tmp_path)
    assert fingerprint["clean"] is False
    assert "Git checkout" in (source_gate_failure(provider, fingerprint) or "")
def test_shared_contract_accepts_prepare_and_rejects_simulator_execution() -> None:
    provider = load_provider_catalog(CATALOG).get("simplepim")
    payload = {
        "schema_version": SIMPLEPIM_RUNNER_SCHEMA_VERSION,
        "provider_id": "simplepim",
        "probe_id": "simplepim_va_map_zip_v1",
        "status": "prepared",
        "target": None,
        "target_observed": None,
        "requested_dpu_count": 1,
        "observed_dpu_count": None,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "hardware_preflight_verified": False,
        "device_evidence": [],
        "native_execution": False,
        "validation_performed": False,
        "exact_validation": False,
        "fallback": False,
        "simulator_kernel_executed": False,
        "release_status": "not_attempted",
        "backend_profile": "backend=hw",
        "source_hash": "a" * 64,
        "source_hashes": {"combined_sha256": "a" * 64},
        "command_fingerprint": "b" * 64,
        "effective_compilers": {},
        "staged_patch": {
            "path": "patches/simplepim-map-unroll-rest.patch",
            "applied": False,
        },
        "binary_hashes": {},
        "input_hashes": {},
        "output_hash": None,
        "logical_transfer_bytes": {},
        "payload_sizes_8_byte_aligned": True,
        "physical_transfer_bytes_available": False,
        "physical_transfer_bytes": None,
        "timing": {},
        "failure_stage": None,
        "reason": "prepare_only_no_compiler_or_hardware_invoked",
        "commands": {},
    }
    assert simplepim_runner_schema_errors(payload, provider, mode="prepare") == ()
    payload["hardware_preflight_verified"] = True
    payload["device_evidence"] = [{"exists": True, "character_device": True, "readable": True, "writable": True}]
    assert "prepared result must not claim staging, build, or hardware execution" in simplepim_runner_schema_errors(payload, provider, mode="prepare")
    payload["hardware_preflight_verified"] = False
    payload["device_evidence"] = []
    payload["simulator_kernel_executed"] = True
    payload["unexpected"] = True
    assert simplepim_runner_schema_errors(payload, provider, mode="prepare")
def test_public_execute_uses_canonical_identity_not_absolute_checkout(tmp_path: Path) -> None:
    custom = tmp_path / "catalog.yml"
    custom.write_text(CATALOG.read_text(encoding="utf-8"), encoding="utf-8")
    catalog = load_provider_catalog(custom)
    harness._require_canonical_identity(catalog, catalog.get("simplepim"))
    custom.write_text(CATALOG.read_text(encoding="utf-8").replace("catalog_id: upmem_provider_m1", "catalog_id: wrong"), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical catalog/provider identity"):
        harness.execute_provider_qualification(ROOT, catalog_path=custom, provider_id="simplepim")
def test_evidence_copies_preflight_and_records_provenance(monkeypatch) -> None:
    real_call = harness._call_runner
    real_source_gate = harness.require_provider_source
    source_checks = []
    def checked_source(*args):
        source_checks.append(True)
        return real_source_gate(*args)
    def fake_call(root, plan_dir, provider, mode, env, host_cc, **kwargs):
        if mode == "--prepare-only":
            return real_call(root, plan_dir, provider, mode, env, host_cc, **kwargs)
        plan_dir.mkdir(parents=True)
        return {"status": "failed", "reason": "controlled failure", "returncode": 1, "raw_path": str(plan_dir / "raw_runner_result.json"), "command": ["controlled"], "environment": {}, "stdout": None, "stderr": None, "timeout_cleanup": None, "contract_errors": ()}
    monkeypatch.setattr(harness, "require_provider_source", checked_source)
    monkeypatch.setattr(harness, "_call_runner", fake_call)
    result = harness._execute_provider_qualification_for_test(
        ROOT, catalog_path=CATALOG, provider_id="simplepim",
        environment={**os.environ, "UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"},
        hook=harness._TEST_EXECUTION_HOOK,
    )
    summary = json.loads(result.result_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert (result.run_dir / "raw_runner_preflight.json").is_file()
    assert summary["repository_fingerprint"]["head_commit"]
    assert summary["runner_fingerprint"]["sha256"]
    assert summary["resource_release_status"] == "unconfirmed"
    assert summary["execution_classification"] == "internal_test_non_evidence"
    assert manifest["qualification"]["raw_preflight_sha256"] == summary["raw_runner_preflight"]["sha256"]
    assert len(source_checks) == 2
def test_orchestrator_timeout_drain_is_bounded(tmp_path: Path, monkeypatch) -> None:
    class Process:
        pid = 123
        returncode = None
        stdout = None
        stderr = None
        def __init__(self):
            self.timeouts = []
        def communicate(self, *, timeout):
            self.timeouts.append(timeout)
            raise harness.subprocess.TimeoutExpired(["runner"], timeout, output=b"partial", stderr=b"error")
    process = Process()
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(harness.os, "killpg", lambda *args: None)
    completed, error, cleanup = harness._run_process(["runner"], tmp_path, {}, 0.1)
    assert process.timeouts == [0.1, 2, 2]
    assert completed and completed.stdout == "partial" and completed.stderr == "error"
    assert error == "runner timed out after 0.1 seconds"
    assert cleanup and cleanup["verified"] is False and cleanup["output_capture_complete"] is False
def test_result_parser_keeps_requested_and_observed_counts_separate() -> None:
    provider = load_provider_catalog(CATALOG).get("simplepim")
    payload = {
        "requested_dpu_count": 1,
        "observed_dpu_count": 1,
        "configured_tasklets_per_dpu": 12,
        "observed_tasklets_per_dpu": None,
        "status": "failed",
        "reason": "no device",
    }
    result = parse_runner_result(payload, provider)
    assert result.status == "failed"
    assert result.observed_dpus == 1
    assert result.configured_tasklets == 12
    assert result.observed_tasklets is None
