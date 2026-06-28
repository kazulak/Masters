from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from quantum_bench.bench import __main__ as bench_main
from quantum_bench.bench.dense_task_bridge import run_dense_task_bridge
import quantum_bench.bench.dense_task_bridge as dense_task_bridge_module

ROOT = Path(__file__).resolve().parents[1]


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stub_path() -> Path:
    return ROOT / "native" / "upmem" / "simplepim" / "simplepim_dense_stub.py"


def test_dense_task_bridge_harness_runs_real_bell_task_with_mock_backend(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=0,
        backend="mock_numpy_dequantized",
        env={},
    )
    summary = _load_summary(result.summary_path)
    encoded = json.dumps(summary)

    assert result.status == "completed"
    assert result.reason is None
    assert summary["schema_version"] == "dense_task_bridge_v1"
    assert summary["status"] == "completed"
    assert summary["case_id"] == "bell_2q"
    assert summary["task_index"] == 0
    assert summary["route_id"] == "dense_gemm"
    assert summary["bridge_backend_id"] == "mock_numpy_dequantized"
    assert summary["bridge_execution_status"] == "mock_executed"
    assert summary["external_command_executed"] is False
    assert summary["execution_implemented"] is False
    assert summary["metadata"]["developer_only"] is True
    assert summary["metadata"]["one_task_only"] is True
    assert summary["metadata"]["normal_routing_unchanged"] is True
    assert summary["metadata"]["bridge_manifest_written"] is True
    assert summary["materialization"]["mode"] == "initial-only"
    assert summary["materialization"]["status"] == "initial_inputs_available"
    assert summary["materialization"]["reason"] is None
    assert summary["artifacts"] == {
        "input_manifest": "bridge/input_manifest.json",
        "output_blob": "bridge/outputs/mock_dequantized_output.npy",
        "output_manifest": "bridge/output_manifest.json",
    }
    assert not Path(summary["artifacts"]["input_manifest"]).is_absolute()
    assert not Path(summary["artifacts"]["output_manifest"]).is_absolute()
    assert not Path(summary["artifacts"]["output_blob"]).is_absolute()
    assert (result.run_dir / "environment.json").exists()
    assert (result.run_dir / "dense_task_bridge_summary.json").exists()
    assert (result.run_dir / "bridge" / "input_manifest.json").exists()
    assert (result.run_dir / "bridge" / "output_manifest.json").exists()
    assert (result.run_dir / "bridge" / "outputs" / "mock_dequantized_output.npy").exists()
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded
    json.dumps(summary)


def test_dense_task_bridge_harness_auto_selects_first_bridgeable_task(tmp_path: Path) -> None:
    result = run_dense_task_bridge(tmp_path, case="bell_2q", backend="mock_numpy_dequantized", env={})
    summary = _load_summary(result.summary_path)

    assert result.status == "completed"
    assert summary["task_index"] == 0
    assert summary["bridge_execution_status"] == "mock_executed"
    assert summary["materialization"]["mode"] == "initial-only"
    assert summary["materialization"]["status"] == "initial_inputs_available"


def test_dense_task_bridge_cpu_replay_with_omitted_task_index_keeps_auto_selection_path(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        backend="mock_numpy_dequantized",
        materialization="cpu-replay",
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "completed"
    assert summary["task_index"] == 0
    assert summary["bridge_execution_status"] == "mock_executed"
    assert summary["materialization"]["mode"] == "cpu-replay"
    assert summary["materialization"]["status"] == "initial_inputs_available"
    assert summary["materialization"]["replayed_task_count"] == 0


def test_dense_task_bridge_simplepim_external_is_nonexecuting_and_nonfatal(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=0,
        backend="simplepim_external",
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "skipped"
    assert result.reason == "simplepim_unavailable"
    assert summary["status"] == "skipped"
    assert summary["bridge_execution_status"] == "skipped"
    assert summary["bridge_execution_reason"] == "simplepim_unavailable"
    assert summary["external_command_executed"] is False
    assert summary["execution_implemented"] is False
    assert summary["artifacts"] == {
        "input_manifest": "bridge/input_manifest.json",
        "output_manifest": "bridge/output_manifest.json",
    }
    assert (result.run_dir / "bridge" / "input_manifest.json").exists()
    assert (result.run_dir / "bridge" / "output_manifest.json").exists()
    assert not (result.run_dir / "bridge" / "outputs" / "simplepim_output.npy").exists()
    output_manifest = _load_summary(result.run_dir / "bridge" / "output_manifest.json")
    assert output_manifest["status"] == "skipped"
    assert output_manifest["output_blob"] is None


def test_dense_task_bridge_simplepim_external_stub_executes_only_contract_process(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=0,
        backend="simplepim_external_stub",
        execute_external=True,
        env={"SIMPLEPIM_STUB_BIN": str(_stub_path())},
    )
    summary = _load_summary(result.summary_path)
    encoded = json.dumps(summary)

    assert result.status == "completed"
    assert result.reason == "external_stub_contract_executed"
    assert summary["status"] == "completed"
    assert summary["reason"] == "external_stub_contract_executed"
    assert summary["bridge_backend_id"] == "simplepim_external_stub"
    assert summary["bridge_execution_status"] == "stub_executed"
    assert summary["bridge_execution_reason"] == "external_stub_contract_executed"
    assert summary["external_command_executed"] is True
    assert summary["execution_implemented"] is False
    assert summary["bridge_validation_metrics"] == {
        "status": "not_applicable",
        "reason": "stub_writes_no_output_blob",
    }
    assert summary["artifacts"] == {
        "input_manifest": "bridge/input_manifest.json",
        "output_manifest": "bridge/output_manifest.json",
    }
    assert "output_blob" not in summary["artifacts"]
    assert (result.run_dir / "bridge" / "input_manifest.json").exists()
    assert (result.run_dir / "bridge" / "output_manifest.json").exists()
    assert not (result.run_dir / "bridge" / "outputs" / "simplepim_output.npy").exists()
    output_manifest = _load_summary(result.run_dir / "bridge" / "output_manifest.json")
    assert output_manifest["status"] == "stub_executed"
    assert output_manifest["output_blob"] is None
    assert output_manifest["external_command_executed"] is True
    assert output_manifest["execution_implemented"] is False
    assert output_manifest["metadata"]["native_kernel_executed"] is False
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded


def test_dense_task_bridge_rejects_unmaterialized_intermediate_task(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=1,
        backend="mock_numpy_dequantized",
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "unsupported"
    assert result.reason == "intermediate_tensor_inputs_not_materialized"
    assert summary["status"] == "unsupported"
    assert summary["reason"] == "intermediate_tensor_inputs_not_materialized"
    assert summary["task_index"] == 1
    assert summary["materialization"]["mode"] == "initial-only"
    assert summary["materialization"]["status"] == "unsupported"
    assert summary["materialization"]["reason"] == "intermediate_tensor_inputs_not_materialized"
    assert summary["artifacts"] == {}
    assert not (result.run_dir / "bridge" / "input_manifest.json").exists()


def test_dense_task_bridge_cpu_replay_processes_later_task(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=1,
        materialization="cpu-replay",
        backend="mock_numpy_dequantized",
        env={},
    )
    summary = _load_summary(result.summary_path)
    encoded = json.dumps(summary)

    assert result.status == "completed"
    assert result.reason is None
    assert summary["task_index"] == 1
    assert summary["bridge_execution_status"] == "mock_executed"
    assert summary["materialization"]["mode"] == "cpu-replay"
    assert summary["materialization"]["status"] == "materialized"
    assert summary["materialization"]["reason"] is None
    assert summary["materialization"]["replayed_task_count"] == 1
    assert summary["materialization"]["replayed_task_ids"] == ["task_0"]
    assert summary["materialization"]["dead_tensor_release_implemented"] is False
    assert summary["artifacts"]["input_manifest"] == "bridge/input_manifest.json"
    assert not Path(summary["artifacts"]["input_manifest"]).is_absolute()
    assert "prepared_operands" not in encoded
    assert "left_matrix" not in encoded
    assert "right_matrix" not in encoded
    assert str(tmp_path) not in encoded


def test_dense_task_bridge_cpu_replay_on_initial_task_records_initial_inputs(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=0,
        materialization="cpu-replay",
        backend="mock_numpy_dequantized",
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "completed"
    assert summary["task_index"] == 0
    assert summary["materialization"]["mode"] == "cpu-replay"
    assert summary["materialization"]["status"] == "initial_inputs_available"
    assert summary["materialization"]["reason"] is None
    assert summary["materialization"]["replayed_task_count"] == 0


def test_dense_task_bridge_cpu_replay_keeps_simplepim_external_nonexecuting(tmp_path: Path) -> None:
    result = run_dense_task_bridge(
        tmp_path,
        case="bell_2q",
        task_index=1,
        materialization="cpu-replay",
        backend="simplepim_external",
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "skipped"
    assert result.reason == "simplepim_unavailable"
    assert summary["bridge_execution_status"] == "skipped"
    assert summary["materialization"]["status"] == "materialized"
    assert summary["external_command_executed"] is False
    assert summary["execution_implemented"] is False


def test_dense_task_bridge_cli_dispatch_prints_run_and_summary_paths(monkeypatch, capsys, tmp_path: Path) -> None:
    called: dict[str, object] = {}

    def fake_run_dense_task_bridge(root_dir: Path, **kwargs: object) -> SimpleNamespace:
        called["root_dir"] = root_dir
        called.update(kwargs)
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / "fake_dense_task_bridge",
            summary_path=tmp_path / "runs" / "fake_dense_task_bridge" / "dense_task_bridge_summary.json",
            status="completed",
            reason=None,
        )

    monkeypatch.setattr(
        dense_task_bridge_module,
        "run_dense_task_bridge",
        fake_run_dense_task_bridge,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "dense-task-bridge",
            "--case",
            "bell_2q",
            "--materialization",
            "cpu-replay",
            "--backend",
            "simplepim_external_stub",
            "--execute-external",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert called["case"] == "bell_2q"
    assert called["task_index"] is None
    assert called["materialization"] == "cpu-replay"
    assert called["backend"] == "simplepim_external_stub"
    assert called["execute_external"] is True
    assert payload["status"] == "completed"
    assert payload["reason"] is None
    assert payload["run_dir"].endswith("fake_dense_task_bridge")
    assert payload["summary_path"].endswith("dense_task_bridge_summary.json")


def test_dense_task_bridge_cli_accepts_upmem_sdk_simulator_backend_only_with_execute_external(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    called: dict[str, object] = {}

    def fake_run_dense_task_bridge(root_dir: Path, **kwargs: object) -> SimpleNamespace:
        called.update(kwargs)
        return SimpleNamespace(
            run_dir=tmp_path / "runs" / "fake_upmem_dense_task_bridge",
            summary_path=tmp_path / "runs" / "fake_upmem_dense_task_bridge" / "dense_task_bridge_summary.json",
            status="completed",
            reason=None,
        )

    monkeypatch.setattr(dense_task_bridge_module, "run_dense_task_bridge", fake_run_dense_task_bridge)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "dense-task-bridge",
            "--case",
            "bell_2q",
            "--backend",
            "upmem_sdk_simulator_dense",
            "--execute-external",
        ],
    )

    assert bench_main.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert called["backend"] == "upmem_sdk_simulator_dense"
    assert called["execute_external"] is True
    assert payload["status"] == "completed"

    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m quantum_bench.bench",
            "dense-task-bridge",
            "--case",
            "bell_2q",
            "--backend",
            "upmem_sdk_simulator_dense",
        ],
    )
    try:
        bench_main.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - parser.error always raises
        raise AssertionError("upmem_sdk_simulator_dense must require --execute-external")


def test_dense_task_bridge_supports_synthetic_l2_developer_case(tmp_path: Path, monkeypatch) -> None:
    def fake_execute(input_manifest_path: Path, **kwargs: object) -> SimpleNamespace:
        manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
        assert kwargs["backend"] == "upmem_sdk_simulator_dense"
        assert manifest["metadata"]["execution_class_hint"] == "L2_SINGLE_DPU_MRAM"
        assert manifest["metadata"]["kernel_strategy_hint"] == "l2_single_dpu_mram_wram_tiled_v1"
        assert manifest["tile_plan"]["requires_tiling"] is True
        return SimpleNamespace(
            execution_status="upmem_sdk_simulator_executed",
            reason="upmem_sdk_simulator_executed",
            error=None,
            error_type=None,
            output_manifest_path=Path("output_manifest.json"),
            output_blob_path=Path("outputs/fake.npy"),
            output_manifest=SimpleNamespace(
                metadata={
                    "backend_family": "upmem_sdk",
                    "execution_class": "L2_SINGLE_DPU_MRAM",
                    "kernel_strategy": "l2_single_dpu_mram_wram_tiled_v1",
                    "simulator_kernel_executed": True,
                    "hardware_kernel_executed": False,
                },
                validation_metrics={"passed": True, "max_abs_error": 0.0},
            ),
            external_command_executed=True,
            execution_implemented=True,
            metadata={
                "backend_family": "upmem_sdk",
                "execution_class": "L2_SINGLE_DPU_MRAM",
                "simulator_kernel_executed": True,
                "hardware_kernel_executed": False,
            },
        )

    monkeypatch.setattr(dense_task_bridge_module, "execute_dense_bridge", fake_execute)

    result = run_dense_task_bridge(
        tmp_path,
        case="synthetic_l2_square",
        task_index=0,
        backend="upmem_sdk_simulator_dense",
        execute_external=True,
        env={},
    )
    summary = _load_summary(result.summary_path)

    assert result.status == "completed"
    assert summary["case_id"] == "synthetic_l2_square"
    assert summary["workload"]["not_real_quantum_circuit"] is True
    assert summary["preparation"]["status"] == "requires_executable_tiling_not_implemented"
    assert summary["tile_plan"]["requires_tiling"] is True
    assert summary["bridge_execution_status"] == "upmem_sdk_simulator_executed"
    assert summary["metadata"]["bridge_manifest_written"] is True


def test_dense_task_bridge_harness_does_not_reference_subprocess() -> None:
    source = inspect.getsource(dense_task_bridge_module)

    assert "subprocess" not in source
