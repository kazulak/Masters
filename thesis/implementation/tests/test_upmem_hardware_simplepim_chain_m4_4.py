from __future__ import annotations

from pathlib import Path
import json

import pytest

from quantum_bench.bench import upmem_hardware_simplepim_chain_m4_4 as m44


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "configs" / "suites" / "upmem_hardware_simplepim_chain_m4_4.yml"


def _response(manifest: dict[str, object]) -> dict[str, object]:
    rows = [
        {"repeat_id": 0, "warmup": True, "reference_int64": -11654, "result_int64": -11654, "exact_integer_match": True, "scatter_time_s": 1.0, "virtual_zip_time_s": 1.0, "map_time_s": 1.0, "reduction_time_s": 1.0, "total_time_s": 4.0},
        *[
            {"repeat_id": index, "warmup": False, "reference_int64": -11654, "result_int64": -11654, "exact_integer_match": True, "scatter_time_s": 1.0, "virtual_zip_time_s": 1.0, "map_time_s": 1.0, "reduction_time_s": 1.0, "total_time_s": 4.0, "total_route_time_s": 1.0}
            for index in range(5)
        ],
    ]
    return {
        "schema_version": m44.NATIVE_SCHEMA_VERSION,
        "profile_id": m44.PROFILE_ID,
        "backend_id": m44.BACKEND_ID,
        "route_id": m44.ROUTE_ID,
        "case_id": manifest["case_id"],
        "fixture_version": manifest["fixture_version"],
        "circuit_semantics_hash": manifest["circuit_semantics_hash"],
        "tensor_network_hash": manifest["tensor_network_hash"],
        "contraction_plan_hash": manifest["contraction_plan_hash"],
        "contraction_path_structure_hash": manifest["contraction_path_structure_hash"],
        "graph_binding_sha256": manifest["graph_binding_sha256"],
        "graph_binding_validated": True,
        "task_graph_integrated": True,
        "input_dtype": "int8",
        "accumulator_dtype": "int32",
        "length": 256,
        "task_count": 2,
        "path": manifest["expected_path"],
        "task_order": ["task_0", "task_1"],
        "task_dependencies": [[], ["task_0"]],
        "operation_kinds": [
            "elementwise_product_i8_i8",
            "scalar_product_i32_i8_reduce_i64",
        ],
        "target_requested": "physical_hardware",
        "target_observed": "physical_hardware",
        "requested_dpu_count": 1,
        "allocated_dpu_count": 1,
        "tasklets_per_dpu": 1,
        "effective_operator_tasklets": 1,
        "final_reduction_location": "host",
        "intermediate_residency": "device_mram",
        "cpu_fallback_used": False,
        "simulator_kernel_executed": False,
        "hardware_speedup_applicable": False,
        "native_taskgraph_protocol": True,
        "status": "completed",
        "validation_status": "passed",
        "provider_initialized": True,
        "simplepim_operator_api_used": True,
        "native_kernel_executed": True,
        "hardware_kernel_executed": True,
        "all_tasks_completed": True,
        "exact_integer_match": True,
        "release_confirmed": True,
        "hardware_functionality_evidence": True,
        "input_sha256": manifest["input_sha256"],
        "reference_int64": manifest["reference_int64"],
        "application_visible_h2d_bytes": 768,
        "application_visible_d2h_bytes": 8,
        "application_visible_transfer_bytes": 776,
        "repetitions": rows,
    }


def test_suite_is_fixed_chain_profile() -> None:
    suite = m44.load_suite(SUITE)
    assert suite["profile"]["requested_dpu_count"] == 1
    assert suite["profile"]["tasklets_per_dpu"] == 1
    assert suite["workload"]["id"] == m44.CHAIN_CASE_ID
    assert suite["workload"]["fixture_version"] == m44.CHAIN_FIXTURE_VERSION
    assert suite["workload"]["task_graph"]["dependencies"] == [["task_1", "task_0"]]
    assert suite["workload"]["task_graph"]["binding_protocol"] == "M44_GRAPH_BINDING_V1"


def test_hardware_requires_opt_in_and_rejects_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="hardware_opt_in_missing"):
        m44.execute(ROOT, suite_path=SUITE, environment={})
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    with pytest.raises(ValueError, match="simulator selector"):
        m44.execute(ROOT, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1", "DPU_BACKEND": "simulator"})


def test_response_requires_device_resident_intermediate_and_host_reduction(tmp_path: Path) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    manifest = json.loads((Path(result["plan_dir"]) / "input_manifest.json").read_text())
    payload = _response(manifest)
    m44._validate_response(payload, manifest=manifest)
    payload["final_reduction_location"] = "device"
    with pytest.raises(ValueError, match="final host reduction"):
        m44._validate_response(payload, manifest=manifest)


def test_response_rejects_malformed_repetition_and_transfer(tmp_path: Path) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    manifest = json.loads((Path(result["plan_dir"]) / "input_manifest.json").read_text())
    payload = _response(manifest)
    payload["repetitions"][1]["total_time_s"] = float("nan")
    with pytest.raises(ValueError, match="timing"):
        m44._validate_response(payload, manifest=manifest)
    payload = _response(manifest)
    payload["application_visible_d2h_bytes"] = 16
    with pytest.raises(ValueError, match="transfer contract"):
        m44._validate_response(payload, manifest=manifest)


def test_records_preserve_warmup_marker(tmp_path: Path) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    manifest = json.loads((Path(result["plan_dir"]) / "input_manifest.json").read_text())
    payload = _response(manifest)
    m44._validate_response(payload, manifest=manifest)
    records = m44._records(payload, manifest, "response.json")
    assert len(records) == 6
    assert [row["warmup"] for row in records] == [True, False, False, False, False, False]
    assert all(record["graph_binding_sha256"] == manifest["graph_binding_sha256"] for record in records)
    assert all(record["input_sha256"] == manifest["input_sha256"] for record in records)
    assert all(record["path"] == manifest["expected_path"] for record in records)


def test_response_cannot_admit_taskgraph_without_binding_validation(tmp_path: Path) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    manifest = json.loads((Path(result["plan_dir"]) / "input_manifest.json").read_text())
    payload = _response(manifest)
    payload["graph_binding_validated"] = False
    with pytest.raises(ValueError, match="graph binding was not validated"):
        m44._validate_response(payload, manifest=manifest)
    payload = _response(manifest)
    payload["task_graph_integrated"] = False
    with pytest.raises(ValueError, match="TaskGraph integration"):
        m44._validate_response(payload, manifest=manifest)


def test_response_rejects_structural_taskgraph_drift(tmp_path: Path) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    manifest = json.loads((Path(result["plan_dir"]) / "input_manifest.json").read_text())
    payload = _response(manifest)
    payload["task_dependencies"] = [[], []]
    with pytest.raises(ValueError, match="task_dependencies"):
        m44._validate_response(payload, manifest=manifest)


def test_suite_rejects_taskgraph_metadata_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    altered = tmp_path / SUITE.name
    altered.write_text(
        SUITE.read_text(encoding="utf-8").replace(
            "binding_protocol: M44_GRAPH_BINDING_V1",
            "binding_protocol: WRONG_PROTOCOL",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(m44, "_canonical_suite", lambda: altered)
    with pytest.raises(ValueError, match="task graph contract"):
        m44.load_suite(altered)


def test_missing_response_is_not_a_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.setattr(m44, "_run", lambda *args, **kwargs: {"command": [], "returncode": 0, "timed_out": False, "elapsed_s": 0.0, "stdout": "", "stderr": ""})
    result = m44.execute(tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"})
    assert result["status"] == "failed"


def test_native_failure_response_forces_failed_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m44, "_response_path", lambda _native: tmp_path / "response.json")
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            (tmp_path / "response.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "failure_stage": "allocation",
                        "reason": "no DPU available",
                        "allocation_attempted": True,
                        "release_confirmed": False,
                    }
                ),
                encoding="utf-8",
            )
            return {"command": command, "returncode": 1, "timed_out": False, "elapsed_s": 0.0, "stdout": "", "stderr": ""}
        return {"command": command, "returncode": 0, "timed_out": False, "elapsed_s": 0.0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(m44, "_run", fake_run)
    result = m44.execute(tmp_path, suite_path=SUITE, environment={"UPMEM_ALLOW_PHYSICAL_HARDWARE": "1"})
    summary = json.loads(Path(result["artifact"]).read_text())
    assert result["status"] == "failed"
    assert summary["status"] == "failed"
    assert summary["failure_stage"] == "allocation"
    assert summary["failure_reason"] == "no DPU available"
    assert summary["allocation_attempted"] is True
    assert summary["release_confirmed"] is False
    assert summary.get("row_count", 0) == 0


def test_prepare_build_only_invokes_native_make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> dict[str, object]:
        calls.append(command)
        assert kwargs["cwd"] == tmp_path / m44.NATIVE_REL
        assert kwargs["env"]["UPMEM_ALLOW_PHYSICAL_HARDWARE"] == "1"  # type: ignore[index]
        return {"command": command, "returncode": 0, "timed_out": False, "elapsed_s": 0.0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(m44, "_run", fake_run)
    result = m44.prepare(tmp_path, suite_path=SUITE, build=True, environment={})
    assert result["status"] == "prepared"
    assert calls == [["make", "clean", "build"]]


def test_native_command_uses_m4_4_staging_path(tmp_path: Path) -> None:
    (tmp_path / "operands.bin").write_bytes(b"chain")
    (tmp_path / "graph_binding.txt").write_text("binding\n", encoding="ascii")
    command, cwd = m44._native_command(
        tmp_path / m44.NATIVE_REL,
        tmp_path / "operands.bin",
        tmp_path / "graph_binding.txt",
        tmp_path / "response.json",
        input_sha256="1" * 64,
        graph_binding_sha256="2" * 64,
    )
    assert command[0].endswith("build/simplepim_chain_m4_4/staged/benchmarks/chain_m4_4/bin/chain_host")
    assert cwd == Path(command[0]).parent.parent
    assert "--input-sha256" in command
    assert "--graph-binding" in command
    assert "--graph-binding-sha256" in command
    assert command[command.index("--input-sha256") + 1] == "1" * 64
    assert command[command.index("--graph-binding-sha256") + 1] == "2" * 64


def test_native_command_uses_prepared_hashes_after_input_tamper(tmp_path: Path) -> None:
    operands = tmp_path / "operands.bin"
    binding = tmp_path / "graph_binding.txt"
    operands.write_bytes(b"original")
    binding.write_text("original\n", encoding="ascii")
    input_hash = m44._sha256(operands)
    binding_hash = m44._sha256(binding)
    operands.write_bytes(b"tampered")
    binding.write_text("tampered\n", encoding="ascii")
    command, _ = m44._native_command(
        tmp_path / m44.NATIVE_REL,
        operands,
        binding,
        tmp_path / "response.json",
        input_sha256=input_hash,
        graph_binding_sha256=binding_hash,
    )
    assert command[command.index("--input-sha256") + 1] == input_hash
    assert command[command.index("--graph-binding-sha256") + 1] == binding_hash


def test_graph_binding_is_deterministic_and_manifest_records_its_hash(tmp_path: Path) -> None:
    first = m44.prepare(tmp_path / "first", suite_path=SUITE)
    second = m44.prepare(tmp_path / "second", suite_path=SUITE)
    first_dir = Path(first["plan_dir"])
    second_dir = Path(second["plan_dir"])
    first_binding = (first_dir / "graph_binding.txt").read_bytes()
    second_binding = (second_dir / "graph_binding.txt").read_bytes()
    assert first_binding == second_binding
    assert first_binding.decode("ascii").splitlines()[-1] == "END"
    assert "\n\n" not in first_binding.decode("ascii")
    first_manifest = json.loads((first_dir / "input_manifest.json").read_text())
    assert first_manifest["graph_binding_sha256"] == m44._sha256(first_dir / "graph_binding.txt")
    assert first_manifest["graph_binding_file"] == "graph_binding.txt"
    lines = first_binding.decode("ascii").splitlines()
    assert lines[0] == "M44_GRAPH_BINDING_V1"
    assert lines[2] == "TASK_COUNT\t2"
    assert lines[12:15] == ["PATH_COUNT\t2", "PATH\t0\t0\t1", "PATH\t1\t0\t1"]
    assert lines[15] == "TASK\t0\ttask_0\t-\tchain_a,chain_b\tresult_0\telementwise_product_i8_i8"
    assert lines[16] == "TASK\t1\ttask_1\ttask_0\tchain_c,result_0\tresult_1\tscalar_product_i32_i8_reduce_i64"


@pytest.mark.parametrize("field", ["circuit_semantics_hash", "contraction_plan_hash", "graph_binding_sha256"])
def test_response_rejects_missing_or_mismatched_binding_identity(tmp_path: Path, field: str) -> None:
    result = m44.prepare(tmp_path, suite_path=SUITE)
    plan_dir = Path(result["plan_dir"])
    manifest = json.loads((plan_dir / "input_manifest.json").read_text())
    payload = _response(manifest)
    payload.pop(field)
    with pytest.raises(ValueError, match="graph binding"):
        m44._validate_response(payload, manifest=manifest)
    payload = _response(manifest)
    payload[field] = "0" * 64
    with pytest.raises(ValueError, match="graph binding"):
        m44._validate_response(payload, manifest=manifest)
