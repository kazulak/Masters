from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantum_bench.bench import __main__ as bench_main
from quantum_bench.bench.dense_route_coverage import run_dense_route_coverage
import quantum_bench.bench.dense_route_coverage as coverage_module
import quantum_bench.targets.upmem.dense_bridge as dense_bridge_module


ROOT = Path(__file__).resolve().parents[1]


def _stub_path() -> Path:
    return ROOT / "native" / "upmem" / "simplepim" / "simplepim_dense_stub.py"


def _load_payload(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "dense_route_coverage.json").read_text(encoding="utf-8"))


def test_dense_route_coverage_bell_has_initial_and_replayed_bridge_ready_tasks(tmp_path: Path) -> None:
    run_dir = run_dense_route_coverage(tmp_path, case="bell_2q", env={})
    payload = _load_payload(run_dir)
    rows = payload["rows"]
    encoded = json.dumps(payload)

    assert payload["schema_version"] == "dense_route_coverage_v1"
    assert payload["summary"]["total_tasks"] == len(rows)
    assert any(row["materialization_status"] == "initial_inputs_available" for row in rows)
    assert any(row["materialization_status"] == "materialized" for row in rows)
    assert any(row["final_readiness_level"] == "bridge_manifest_ready" for row in rows)
    assert all(row["bridge_artifact_written"] is False for row in rows)
    assert all(row["external_command_executed"] is False for row in rows)
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded
    assert (run_dir / "cases" / "bell_2q" / "dense_route_coverage.jsonl").exists()


def test_dense_route_coverage_ghz_chain_three_qubits_is_supported(tmp_path: Path) -> None:
    run_dir = run_dense_route_coverage(tmp_path, case="ghz_chain", n_qubits=3, env={})
    payload = _load_payload(run_dir)

    assert payload["suite_id"] == "ghz_3q"
    assert payload["summary"]["total_tasks"] > 0
    assert {row["case_id"] for row in payload["rows"]} == {"ghz_3q"}


def test_dense_route_coverage_suite_uses_default_planner_only(tmp_path: Path) -> None:
    run_dir = run_dense_route_coverage(
        tmp_path,
        suite_path=ROOT / "configs" / "suites" / "diagnostics" / "planner_compare.yml",
        env={},
    )
    payload = _load_payload(run_dir)
    rows = payload["rows"]

    assert {row["case_id"] for row in rows} == {"bell_2q", "ghz_3q"}
    assert {row["optimize_mode"] for row in rows} == {"greedy"}
    assert len({row["planner_id"] for row in rows}) == 1
    assert payload["planner"]["optimize"] == "greedy"


def test_dense_route_coverage_csv_uses_json_strings_for_nested_fields(tmp_path: Path) -> None:
    run_dir = run_dense_route_coverage(tmp_path, case="bell_2q", env={})
    with (run_dir / "dense_route_coverage.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    parsed_sources = json.loads(rows[0]["input_sources"])
    parsed_validation = json.loads(rows[0]["validation_metrics"])
    parsed_conversion = json.loads(rows[0]["conversion_error_metrics"])
    assert isinstance(parsed_sources, dict)
    assert isinstance(parsed_validation, dict)
    assert isinstance(parsed_conversion, dict)


def test_dense_route_coverage_default_does_not_call_subprocess(tmp_path: Path, monkeypatch) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("default dense-route-coverage must not call subprocess")

    monkeypatch.setattr(coverage_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    run_dir = run_dense_route_coverage(tmp_path, case="bell_2q", env={})
    payload = _load_payload(run_dir)

    assert all(row["external_stub_checked"] is False for row in payload["rows"])
    assert all(row["external_command_executed"] is False for row in payload["rows"])


def test_dense_route_coverage_stub_backend_without_execute_writes_metadata_without_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def forbidden_subprocess(*args: object, **kwargs: object) -> object:
        raise AssertionError("stub backend without --execute-external must not call subprocess")

    monkeypatch.setattr(coverage_module, "capture_environment", lambda root_dir: {})
    monkeypatch.setattr(dense_bridge_module.subprocess, "run", forbidden_subprocess)
    run_dir = run_dense_route_coverage(
        tmp_path,
        case="bell_2q",
        bridge_backend="simplepim_external_stub",
        max_bridge_artifacts=1,
        env={},
    )
    payload = _load_payload(run_dir)
    written = [row for row in payload["rows"] if row["bridge_artifact_written"]]

    assert len(written) == 1
    assert written[0]["external_stub_checked"] is False
    assert written[0]["external_stub_status"] == "not_implemented"
    assert written[0]["final_readiness_level"] == "bridge_manifest_ready"
    assert (run_dir / written[0]["bridge_artifact_path"]).exists()


def test_dense_route_coverage_explicit_stub_check_is_capped_and_ready(tmp_path: Path) -> None:
    run_dir = run_dense_route_coverage(
        tmp_path,
        case="bell_2q",
        bridge_backend="simplepim_external_stub",
        execute_external=True,
        max_bridge_artifacts=1,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    payload = _load_payload(run_dir)
    stub_rows = [row for row in payload["rows"] if row["external_stub_checked"]]

    assert len(stub_rows) == 1
    row = stub_rows[0]
    assert row["final_readiness_level"] == "external_stub_ready"
    assert row["external_stub_status"] == "stub_executed"
    assert row["external_command_executed"] is True
    assert row["execution_implemented"] is False
    output_manifest = run_dir / Path(row["bridge_artifact_path"]).parent / "output_manifest.json"
    manifest_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["status"] == "stub_executed"
    assert manifest_payload["output_blob"] is None
    assert manifest_payload["execution_implemented"] is False


def test_dense_route_coverage_stub_failure_keeps_bridge_manifest_ready(tmp_path: Path, monkeypatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=3, stdout="", stderr="stub failed")

    monkeypatch.setattr(dense_bridge_module.subprocess, "run", fake_run)
    run_dir = run_dense_route_coverage(
        tmp_path,
        case="bell_2q",
        bridge_backend="simplepim_external_stub",
        execute_external=True,
        max_bridge_artifacts=1,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    payload = _load_payload(run_dir)
    checked = [row for row in payload["rows"] if row["external_stub_checked"]]

    assert len(checked) == 1
    assert checked[0]["external_stub_status"] == "failed"
    assert checked[0]["final_readiness_level"] == "bridge_manifest_ready"
    assert "external_stub_failed" in checked[0]["readiness_reason"]


def test_dense_route_coverage_cli_rejects_invalid_external_combinations(monkeypatch) -> None:
    invalid_argv = [
        [
            "python -m quantum_bench.bench",
            "dense-route-coverage",
            "--case",
            "bell_2q",
            "--execute-external",
        ],
        [
            "python -m quantum_bench.bench",
            "dense-route-coverage",
            "--case",
            "bell_2q",
            "--bridge-backend",
            "simplepim_external_stub",
            "--execute-external",
        ],
    ]
    for argv in invalid_argv:
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit) as exc_info:
            bench_main.main()
        assert exc_info.value.code == 2


def test_dense_route_coverage_cli_dispatch(monkeypatch, capsys, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def fake_run_dense_route_coverage(root_dir: Path, **kwargs: object) -> Path:
        called["root_dir"] = root_dir
        called.update(kwargs)
        run_dir = tmp_path / "runs" / "fake_dense_route_coverage"
        run_dir.mkdir(parents=True)
        return run_dir

    monkeypatch.setattr(coverage_module, "run_dense_route_coverage", fake_run_dense_route_coverage)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "dense-route-coverage",
            "--case",
            "bell_2q",
            "--bridge-backend",
            "simplepim_external_stub",
            "--max-bridge-artifacts",
            "1",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert called["case"] == "bell_2q"
    assert called["bridge_backend"] == "simplepim_external_stub"
    assert called["max_bridge_artifacts"] == 1
    assert called["execute_external"] is False
    assert payload["status"] == "completed"
    assert payload["run_dir"].endswith("fake_dense_route_coverage")


def test_readiness_level_precedence_for_synthetic_rows() -> None:
    base = {
        "materialization_status": "initial_inputs_available",
        "materialization_reason": None,
        "supported_by_dense_estimate": True,
        "estimate_reject_reason": None,
        "requires_tiling": False,
        "tiling_implemented": False,
        "requires_host_aggregation": False,
        "dense_prepare_status": "simplepim_unavailable",
        "dense_prepare_reason": None,
        "bridge_manifest_eligible": True,
        "readiness_reason": None,
    }

    assert coverage_module._readiness_level({**base, "materialization_status": "unsupported"})[0] == "not_materializable"
    assert coverage_module._readiness_level({**base, "supported_by_dense_estimate": False})[0] == "blocked_unsupported_shape"
    assert coverage_module._readiness_level({**base, "requires_tiling": True})[0] == "blocked_requires_tiling"
    assert coverage_module._readiness_level({**base, "dense_prepare_status": "failed"})[0] == "dense_prepare_failed"
    assert coverage_module._readiness_level(base)[0] == "bridge_manifest_ready"
