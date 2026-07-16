from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json

import numpy as np

from scripts import research_benchmark_pack as research_pack
from quantum_bench.bench.upmem_hardware_taskgraph_study import prepare_study_case
from quantum_bench.bench.upmem_hardware_taskgraph_study_runtime import (
    _execute_variant,
    _rotated_variants,
    run_study_suite,
)
from quantum_bench.routing.generic_prepare import (
    GENERIC_MODE_FLOAT32_NO_QUANT,
    generic_loop_reference_int32,
)
from quantum_bench.targets.upmem.hardware_session import (
    HardwareSessionBuild,
    HardwareSessionClose,
    HardwareSessionExecution,
)
from quantum_bench.targets.upmem.hardware_taskgraph_study import (
    load_hardware_taskgraph_study_suite,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakePersistentSession:
    """Controlled native-shaped executor; it does not claim hardware coverage."""

    def __init__(self, root: Path, *, fail: bool = False) -> None:
        self.root = root
        self.fail = fail
        self.requests: list[tuple[str | None, list[str]]] = []

    def submit(self, tasks, *, request_id=None, timeout_s=None):
        del timeout_s
        self.requests.append((request_id, [task.task_id for task in tasks]))
        response_path = self.root / f"{len(self.requests):04d}_response.json"
        response_path.write_text("{}\n", encoding="utf-8")
        if self.fail:
            return HardwareSessionExecution(
                status="failed",
                failure_stage="kernel_launch_failed",
                response_path=response_path,
                response={"tasks": []},
                process_time_s=0.001,
                command=("fake",),
                stdout_snippet="",
                stderr_snippet="",
            )
        response_tasks = []
        for task in tasks:
            preparation = task.preparation
            operands = preparation.prepared_operands
            assert operands is not None
            if task.operand_mode == GENERIC_MODE_FLOAT32_NO_QUANT:
                raw = np.asarray(operands.expected_reference_output, dtype="<f4")
            else:
                raw = generic_loop_reference_int32(
                    operands.left_quantized,
                    operands.right_quantized,
                    output_shape=task.output_shape,
                    left_strides=preparation.left_strides,
                    right_strides=preparation.right_strides,
                    output_strides=preparation.output_strides,
                    output_to_left_axes=preparation.output_to_left_axes,
                    output_to_right_axes=preparation.output_to_right_axes,
                    contracted_to_left_axes=preparation.contracted_to_left_axes,
                    contracted_to_right_axes=preparation.contracted_to_right_axes,
                    contracted_dims=preparation.contracted_dims,
                )
            np.asarray(raw).tofile(task.output_path)
            response_tasks.append(
                {
                    "task_id": task.task_id,
                    "status": "completed",
                    "timing": {
                        "h2d_time_s": 0.001,
                        "kernel_time_s": 0.002,
                        "d2h_time_s": 0.001,
                    },
                }
            )
        return HardwareSessionExecution(
            status="completed",
            failure_stage=None,
            response_path=response_path,
            response={
                "status": "completed",
                "allocation_time_s": 0.05,
                "binary_load_time_s": 0.06,
                "tasks": response_tasks,
            },
            process_time_s=0.004 * len(tasks),
            command=("fake",),
            stdout_snippet="",
            stderr_snippet="",
        )


class _FakePersistentSessionForSuite(_FakePersistentSession):
    def __init__(self, root: Path, *, release_confirmed: bool = True) -> None:
        super().__init__(root)
        self.release_confirmed = release_confirmed

    @property
    def startup_metadata(self):
        return {
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "allocation_time_s": 0.05,
            "binary_load_time_s": 0.06,
        }

    def close(self, *, timeout_s=None):
        del timeout_s
        return HardwareSessionClose(
            status="closed" if self.release_confirmed else "failed",
            failure_stage=None if self.release_confirmed else "hardware_release_failed",
            release_confirmed=self.release_confirmed,
            release_time_s=0.01,
            process_returncode=0 if self.release_confirmed else 1,
            stdout_snippet="",
            stderr_snippet="",
        )


def _build(tmp_path: Path) -> HardwareSessionBuild:
    root = tmp_path / "native_session"
    root.mkdir()
    host = root / "host"
    dpu = root / "dpu"
    host.write_bytes(b"host")
    dpu.write_bytes(b"dpu")
    return HardwareSessionBuild(
        session_root=root,
        source_snapshot=root,
        build_dir=root,
        host_binary=host,
        dpu_binary=dpu,
        source_tree_hash="source",
        host_binary_hash="host",
        dpu_binary_hash="dpu",
        build_time_s=0.0,
        build_command=("fake",),
        sdk_tools={},
    )


def _prepared_case(tmp_path: Path):
    suite = load_hardware_taskgraph_study_suite(
        ROOT / "configs/suites/upmem_hardware_taskgraph_path_quantization.yml"
    )
    case = suite.suite["cases"][0]
    prepared = prepare_study_case(ROOT, tmp_path / "case", suite, case)
    return suite, case, prepared


def test_persistent_execution_feeds_native_outputs_to_downstream_tasks(
    tmp_path: Path,
) -> None:
    suite, _case, prepared = _prepared_case(tmp_path)
    build = _build(tmp_path)
    session = _FakePersistentSession(build.session_root)
    variant = prepared["variants"]["opt_einsum_greedy"]

    result = _execute_variant(
        session=session,  # type: ignore[arg-type]
        native_build=build,
        root_dir=ROOT,
        work_dir=build.session_root / "repeat",
        graph=variant["graph"],
        network=prepared["network"],
        reference_output=variant["reference_output"],
        quantization_mode="none",
        profile=suite.profile,
        request_prefix="test",
    )

    assert result.status == "completed"
    assert result.summary["physical_dependency_chain_verified"] is True
    assert (
        result.summary["source_task_completion_count"]
        == result.summary["source_task_count"]
    )
    assert result.summary["steady_state_graph_execution_s"] > 0.0
    assert result.summary["allocation_time_s"] == 0.05
    assert result.summary["binary_load_time_s"] == 0.06
    assert result.summary["validation_time_s"] > 0.0
    assert any(
        metric["physical_dependency_input_count"] > 0 for metric in result.task_metrics
    )
    assert len(session.requests) == len(variant["graph"].tasks)


def test_quantized_execution_records_exact_native_int32_components(
    tmp_path: Path,
) -> None:
    suite, _case, prepared = _prepared_case(tmp_path)
    build = _build(tmp_path)
    variant = prepared["variants"]["custom_upmem_v2_balanced"]
    result = _execute_variant(
        session=_FakePersistentSession(build.session_root),  # type: ignore[arg-type]
        native_build=build,
        root_dir=ROOT,
        work_dir=build.session_root / "quantized",
        graph=variant["graph"],
        network=prepared["network"],
        reference_output=variant["reference_output"],
        quantization_mode="per_task_input_quantize",
        profile=suite.profile,
        request_prefix="quantized",
    )

    assert result.status == "completed"
    assert result.summary["exact_integer_match"] is True
    assert all(metric["exact_integer_match"] is True for metric in result.task_metrics)


def test_failed_native_submission_does_not_fallback_to_cpu(tmp_path: Path) -> None:
    suite, _case, prepared = _prepared_case(tmp_path)
    build = _build(tmp_path)
    variant = prepared["variants"]["opt_einsum_greedy"]
    result = _execute_variant(
        session=_FakePersistentSession(build.session_root, fail=True),  # type: ignore[arg-type]
        native_build=build,
        root_dir=ROOT,
        work_dir=build.session_root / "failed",
        graph=variant["graph"],
        network=prepared["network"],
        reference_output=variant["reference_output"],
        quantization_mode="none",
        profile=suite.profile,
        request_prefix="failed",
    )

    assert result.status == "failed"
    assert result.output is None
    assert result.summary["cpu_fallback_used"] is False
    assert "kernel_launch_failed" in str(result.reason)


def test_round_robin_order_counterbalances_four_variants() -> None:
    variants = (
        ("greedy", "none"),
        ("greedy", "int8"),
        ("custom", "none"),
        ("custom", "int8"),
    )
    assert _rotated_variants(variants, 0) == variants
    assert _rotated_variants(variants, 1) == variants[1:] + variants[:1]
    assert _rotated_variants(variants, 4) == variants


def test_suite_emits_four_compatible_one_dpu_timing_rows_without_hardware(
    tmp_path: Path, monkeypatch
) -> None:
    import quantum_bench.bench.upmem_hardware_taskgraph_study_runtime as runtime

    suite, _case, _prepared = _prepared_case(tmp_path)
    mini_suite = replace(
        suite,
        suite={
            **suite.suite,
            "cases": [suite.suite["cases"][0]],
            "warmups": 0,
            "repeats": 1,
        },
    )
    build = _build(tmp_path)
    monkeypatch.setattr(
        runtime, "build_hardware_session", lambda *args, **kwargs: build
    )
    monkeypatch.setattr(
        runtime,
        "start_hardware_session",
        lambda *args, **kwargs: _FakePersistentSessionForSuite(build.session_root),
    )
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "cases").mkdir()

    result = run_study_suite(ROOT, run_dir, mini_suite, environment={})

    assert result.status == "completed"
    records = [
        json.loads(line)
        for line in (run_dir / "normalized_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 4
    assert {record["quantization_mode"] for record in records} == {
        "none",
        "per_task_input_quantize",
    }
    assert len({record["contraction_path_structure_hash"] for record in records}) == 2
    assert all(record["hardware_release_verified"] is True for record in records)
    assert all(
        record["timing_scope"] == "one_dpu_steady_state_full_taskgraph_v1"
        for record in records
    )
    assert all(record["cpu_fallback_used"] is False for record in records)


def test_release_failure_marks_records_and_per_repeat_artifacts_failed(
    tmp_path: Path, monkeypatch
) -> None:
    import quantum_bench.bench.upmem_hardware_taskgraph_study_runtime as runtime

    suite, _case, _prepared = _prepared_case(tmp_path)
    mini_suite = replace(
        suite,
        suite={
            **suite.suite,
            "cases": [suite.suite["cases"][0]],
            "warmups": 0,
            "repeats": 1,
        },
    )
    build = _build(tmp_path)
    monkeypatch.setattr(
        runtime, "build_hardware_session", lambda *args, **kwargs: build
    )
    monkeypatch.setattr(
        runtime,
        "start_hardware_session",
        lambda *args, **kwargs: _FakePersistentSessionForSuite(
            build.session_root, release_confirmed=False
        ),
    )
    run_dir = tmp_path / "release-failure-run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "cases").mkdir()

    result = run_study_suite(ROOT, run_dir, mini_suite, environment={})

    assert result.status == "failed"
    records = [
        json.loads(line)
        for line in (run_dir / "normalized_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records
    assert all(record["status"] == "failed" for record in records)
    assert all(
        record["failure_stage"] == "hardware_release_failed" for record in records
    )
    assert all(record["hardware_functionality_evidence"] is False for record in records)
    summary_path = run_dir / records[0]["task_metrics_artifact"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["failure_stage"] == "hardware_release_failed"


def test_study_evidence_generates_one_dpu_report_tables_and_figures(
    tmp_path: Path, monkeypatch
) -> None:
    import quantum_bench.bench.upmem_hardware_taskgraph_study_runtime as runtime

    suite, _case, _prepared = _prepared_case(tmp_path)
    mini_suite = replace(
        suite,
        suite={
            **suite.suite,
            "cases": [suite.suite["cases"][0]],
            "warmups": 0,
            "repeats": 7,
        },
    )
    build = _build(tmp_path)
    monkeypatch.setattr(
        runtime, "build_hardware_session", lambda *args, **kwargs: build
    )
    monkeypatch.setattr(
        runtime,
        "start_hardware_session",
        lambda *args, **kwargs: _FakePersistentSessionForSuite(build.session_root),
    )
    run_dir = tmp_path / "reportable-run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "cases").mkdir()
    assert (
        run_study_suite(ROOT, run_dir, mini_suite, environment={}).status == "completed"
    )

    report_dir = tmp_path / "report"
    assert (
        research_pack._write_pack(
            ROOT,
            report_dir,
            [run_dir],
            command_results=[],
            selected_suite_keys=[],
        )
        == 0
    )

    for name in (
        "upmem_one_dpu_runtime_summary.csv",
        "upmem_one_dpu_quantization_pairs.csv",
        "upmem_one_dpu_path_pairs.csv",
    ):
        assert (report_dir / name).is_file()
    manifest = json.loads(
        (report_dir / "plot_manifest.json").read_text(encoding="utf-8")
    )
    one_dpu_plots = {
        entry["plot"]: entry["status"]
        for entry in manifest["plots"]
        if entry["plot"].startswith("upmem_one_dpu_")
    }
    assert len(one_dpu_plots) == 7
    assert all((report_dir / "plots" / name).is_file() for name in one_dpu_plots)
    assert (
        one_dpu_plots["upmem_one_dpu_path_quantization_runtime.png"]
        == "generated_valid"
    )
    assert (
        one_dpu_plots["upmem_one_dpu_quantization_ratio_by_path.png"]
        == "generated_valid"
    )
    assert (
        one_dpu_plots["upmem_one_dpu_path_ratio_by_numeric_mode.png"]
        == "generated_valid"
    )
    summary = (report_dir / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "## Physical One-DPU Path And Quantization Study" in summary
    assert "ratios are not speedups" in summary
