from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_bench.bench.config import suite_path
from quantum_bench.bench.planner_compare import compare_planners
from quantum_bench.bench.summary import write_summary
from quantum_bench.core.records import to_jsonable


def main() -> int:
    root_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(prog="python -m quantum_bench.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--suite", required=True, help="Suite path or preset name under configs/suites")

    summarize_parser = sub.add_parser("summarize")
    summarize_parser.add_argument("run_dir")

    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("run_dir")
    plot_parser.add_argument("--baseline-route")

    compare_parser = sub.add_parser("compare-planners")
    compare_parser.add_argument("--suite", required=True, help="Suite path or preset name under configs/suites")

    simplepim_parser = sub.add_parser("simplepim-microbench")
    simplepim_parser.add_argument("--dry-run", action="store_true", required=True)
    simplepim_parser.add_argument("--m", type=int, required=True)
    simplepim_parser.add_argument("--k", type=int, required=True)
    simplepim_parser.add_argument("--n", type=int, required=True)
    simplepim_parser.add_argument("--route-dtype", default="int8", choices=("int8", "int16"))
    simplepim_parser.add_argument("--source-dtype", default="float32", choices=("float32", "float64"))
    simplepim_parser.add_argument("--seed", type=int, default=0)

    dense_bridge_parser = sub.add_parser("dense-task-bridge")
    dense_bridge_parser.add_argument("--case", default="bell_2q")
    dense_bridge_parser.add_argument("--n-qubits", type=int)
    dense_bridge_parser.add_argument("--task-index", type=int)
    dense_bridge_parser.add_argument("--materialization", default="initial-only", choices=("initial-only", "cpu-replay"))
    dense_bridge_parser.add_argument(
        "--backend",
        default="mock_numpy_dequantized",
        choices=("mock_numpy_dequantized", "simplepim_external", "simplepim_external_stub", "upmem_sdk_simulator_dense"),
    )
    dense_bridge_parser.add_argument("--execute-external", action="store_true")

    generic_bridge_parser = sub.add_parser("generic-task-bridge")
    generic_bridge_parser.add_argument("--case", default="bell_2q")
    generic_bridge_parser.add_argument("--n-qubits", type=int)
    generic_bridge_parser.add_argument("--task-index", type=int, required=True)
    generic_bridge_parser.add_argument("--backend", default="upmem_sdk_simulator_generic_loop", choices=("upmem_sdk_simulator_generic_loop",))
    generic_bridge_parser.add_argument("--execute-external", action="store_true")

    upmem_runtime_parser = sub.add_parser("upmem-taskgraph-runtime")
    upmem_runtime_parser.add_argument("--case", default="bell_2q")
    upmem_runtime_parser.add_argument("--n-qubits", type=int)
    upmem_runtime_parser.add_argument("--policy", default="generic-only", choices=("generic-only", "dense-then-generic", "dense-only"))
    upmem_runtime_parser.add_argument(
        "--quantization-mode",
        default="per_task_input_quantize",
        choices=("per_task_input_quantize", "none", "persistent_network_quantized"),
    )
    upmem_runtime_parser.add_argument("--execute-external", action="store_true")

    upmem_mvp_parser = sub.add_parser("upmem-mvp-benchmark")
    upmem_mvp_parser.add_argument("--suite", required=True, help="Suite path or preset name under configs/suites")
    upmem_mvp_parser.add_argument("--policies", default="generic-only,dense-then-generic")
    upmem_mvp_parser.add_argument("--quantization-modes", default="per_task_input_quantize")
    upmem_mvp_parser.add_argument("--execute-external", action="store_true")
    upmem_mvp_parser.add_argument("--max-taskgraph-tasks", type=int, default=128)
    upmem_mvp_parser.add_argument("--fail-fast", action="store_true")
    upmem_mvp_parser.add_argument("--artifact-retention", default="compact", choices=("full", "compact", "summary-only"))

    simulation_compare_parser = sub.add_parser("simulation-backend-compare")
    simulation_compare_parser.add_argument("--suite", required=True, help="Suite path or preset name under configs/suites")
    simulation_compare_parser.add_argument("--artifact-retention", default="compact", choices=("full", "compact", "summary-only"))

    report_run_parser = sub.add_parser("report-run")
    report_run_parser.add_argument("--input", required=True)
    report_run_parser.add_argument("--output-plots", action=argparse.BooleanOptionalAction, default=True)

    prune_run_parser = sub.add_parser("prune-run")
    prune_run_parser.add_argument("--input", required=True)
    prune_run_parser.add_argument("--artifact-retention", default="compact", choices=("compact", "summary-only"))

    compare_runs_parser = sub.add_parser("compare-runs")
    compare_runs_parser.add_argument("--baseline", required=True)
    compare_runs_parser.add_argument("--candidate", required=True)
    compare_runs_parser.add_argument("--out", required=True)

    coverage_parser = sub.add_parser("dense-route-coverage")
    coverage_input = coverage_parser.add_mutually_exclusive_group(required=True)
    coverage_input.add_argument("--suite", help="Suite path or preset name under configs/suites")
    coverage_input.add_argument("--case", help="Builtin circuit case name")
    coverage_parser.add_argument("--n-qubits", type=int)
    coverage_parser.add_argument("--bridge-backend", default="none", choices=("none", "simplepim_external_stub"))
    coverage_parser.add_argument("--execute-external", action="store_true")
    coverage_parser.add_argument("--max-bridge-artifacts", type=int, default=0)

    pim_eval_parser = sub.add_parser("pim-bridge-eval")
    pim_eval_input = pim_eval_parser.add_mutually_exclusive_group(required=True)
    pim_eval_input.add_argument("--suite", help="Suite path or preset name under configs/suites")
    pim_eval_input.add_argument("--case", help="Builtin circuit case name")
    pim_eval_parser.add_argument("--n-qubits", type=int)
    pim_eval_parser.add_argument("--backend", default="upmem_sdk_simulator_dense", choices=("upmem_sdk_simulator_dense",))
    pim_eval_parser.add_argument("--execute-external", action="store_true")
    pim_eval_parser.add_argument("--dry-run", action="store_true")
    pim_eval_parser.add_argument("--max-tasks-per-case", type=int, default=64)
    pim_eval_parser.add_argument("--max-executed-tasks-per-case", type=int, default=2)
    pim_eval_parser.add_argument(
        "--task-selection",
        default="eligible-only",
        choices=("all", "eligible-only", "first-supported", "first-n"),
    )
    pim_eval_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    pim_eval_parser.add_argument("--planner")
    pim_eval_parser.add_argument("--output-plots", action=argparse.BooleanOptionalAction, default=True)
    pim_eval_parser.add_argument("--debug-failures", action="store_true")
    pim_eval_parser.add_argument("--compare-mock-on-failure", action="store_true")
    pim_eval_parser.add_argument("--keep-failure-artifacts", action="store_true")

    frontier_parser = sub.add_parser("pim-frontier-analysis")
    frontier_input = frontier_parser.add_mutually_exclusive_group(required=True)
    frontier_input.add_argument("--suite", help="Suite path or preset name under configs/suites")
    frontier_input.add_argument("--case", help="Builtin circuit case name")
    frontier_parser.add_argument("--n-qubits", type=int)
    frontier_parser.add_argument("--available-dpus", type=int, default=64)
    frontier_parser.add_argument("--per-dpu-wram-bytes", type=int, default=64 * 1024)
    frontier_parser.add_argument("--effective-wram-bytes", type=int, default=60 * 1024)
    frontier_parser.add_argument("--per-dpu-mram-bytes", type=int, default=64 * 1024 * 1024)
    frontier_parser.add_argument("--max-task-group-dpus", type=int, default=64)
    frontier_parser.add_argument("--output-plots", action=argparse.BooleanOptionalAction, default=True)

    matrix_parser = sub.add_parser("benchmark-matrix-report")
    matrix_parser.add_argument("--matrix", required=True, help="Benchmark matrix YAML path")
    matrix_parser.add_argument("--external-libs-report")
    matrix_parser.add_argument("--output-plots", action=argparse.BooleanOptionalAction, default=True)

    shadow_parser = sub.add_parser("shadow-routed-runtime")
    shadow_input = shadow_parser.add_mutually_exclusive_group(required=True)
    shadow_input.add_argument("--suite", help="Suite path or preset name under configs/suites")
    shadow_input.add_argument("--case", help="Builtin circuit case name")
    shadow_parser.add_argument("--n-qubits", type=int)
    shadow_parser.add_argument("--dense-shadow", default="prepare", choices=("none", "prepare", "bridge", "stub"))
    shadow_parser.add_argument(
        "--shadow-route-policy",
        default="cpu-only",
        choices=("cpu-only", "dense-if-estimate-supported", "dense-if-no-tiling", "dense-if-bridge-ready"),
    )
    shadow_parser.add_argument(
        "--bridge-backend",
        default="none",
        choices=("none", "mock_numpy_dequantized", "simplepim_external_stub"),
    )
    shadow_parser.add_argument("--execute-external", action="store_true")
    shadow_parser.add_argument("--max-bridge-artifacts", type=int, default=0)

    upmem_env_parser = sub.add_parser("upmem-env-check")
    upmem_env_parser.add_argument("--run-sample", action="store_true")
    upmem_env_parser.add_argument("--target", default="auto", choices=("auto", "simulator", "hardware"))
    upmem_env_parser.add_argument("--timeout-seconds", type=float, default=10.0)
    upmem_env_parser.add_argument("--simplepim-home")

    external_libs_parser = sub.add_parser("upmem-external-libs-check")
    external_libs_parser.add_argument("--simplepim-home")
    external_libs_parser.add_argument("--pid-comm-home")
    external_libs_parser.add_argument("--timeout-seconds", type=float, default=10.0)
    external_libs_parser.add_argument("--check-pid-comm-build", action="store_true")

    compare_results_parser = sub.add_parser("compare-results")
    compare_results_parser.add_argument("--inputs", nargs="+", required=True)
    compare_results_parser.add_argument("--out", required=True)

    sub.add_parser("probe")

    args = parser.parse_args()
    if args.command == "run":
        from quantum_bench.bench.runner import run_suite

        run_dir = run_suite(suite_path(args.suite, root_dir), root_dir)
        from quantum_bench.plots import plot_run

        created = plot_run(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "plots": [str(path) for path in created]}, indent=2))
        return 0
    if args.command == "summarize":
        summary = write_summary(Path(args.run_dir).resolve())
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "plot":
        from quantum_bench.plots import plot_run

        created = plot_run(Path(args.run_dir).resolve(), args.baseline_route)
        print(json.dumps({"plots": [str(path) for path in created]}, indent=2))
        return 0
    if args.command == "compare-planners":
        run_dir = compare_planners(suite_path(args.suite, root_dir), root_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "planner_comparison": str(run_dir / "planner_comparison.json"),
                    "planner_comparison_csv": str(run_dir / "planner_comparison.csv"),
                    "planner_comparison_summary": str(run_dir / "planner_comparison_summary.md"),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "simplepim-microbench":
        from quantum_bench.bench.simplepim_microbench import run_simplepim_microbench

        run_dir, artifact_path, status = run_simplepim_microbench(
            root_dir,
            gemm_m=args.m,
            gemm_k=args.k,
            gemm_n=args.n,
            route_dtype=args.route_dtype,
            source_dtype=args.source_dtype,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(artifact_path),
                    "status": status,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "dense-task-bridge":
        from quantum_bench.bench.dense_task_bridge import run_dense_task_bridge

        if args.backend == "upmem_sdk_simulator_dense" and not args.execute_external:
            parser.error("--backend upmem_sdk_simulator_dense requires --execute-external")
        result = run_dense_task_bridge(
            root_dir,
            case=args.case,
            n_qubits=args.n_qubits,
            task_index=args.task_index,
            backend=args.backend,
            execute_external=args.execute_external,
            materialization=args.materialization,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "summary_path": str(result.summary_path),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "generic-task-bridge":
        from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge

        if args.backend == "upmem_sdk_simulator_generic_loop" and not args.execute_external:
            parser.error("--backend upmem_sdk_simulator_generic_loop requires --execute-external")
        result = run_generic_task_bridge(
            root_dir,
            case=args.case,
            n_qubits=args.n_qubits,
            task_index=args.task_index,
            backend=args.backend,
            execute_external=args.execute_external,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "summary_path": str(result.summary_path),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-taskgraph-runtime":
        from quantum_bench.bench.upmem_taskgraph_runtime import run_upmem_taskgraph_runtime

        if not args.execute_external:
            parser.error("upmem-taskgraph-runtime requires --execute-external for strict UPMEM SDK DPU execution")
        if args.quantization_mode != "per_task_input_quantize":
            parser.error("only --quantization-mode per_task_input_quantize is implemented for upmem-taskgraph-runtime")
        result = run_upmem_taskgraph_runtime(
            root_dir,
            case=args.case,
            n_qubits=args.n_qubits,
            policy=args.policy,
            quantization_mode=args.quantization_mode,
            execute_external=args.execute_external,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "summary_path": str(result.summary_path),
                    "task_metrics": str(result.run_dir / "cases" / result.case_id / "upmem_taskgraph_task_metrics.jsonl"),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-mvp-benchmark":
        from quantum_bench.bench.upmem_mvp_benchmark import parse_csv_choices, run_upmem_mvp_benchmark

        try:
            result = run_upmem_mvp_benchmark(
                root_dir,
                suite_path=suite_path(args.suite, root_dir),
                policies=parse_csv_choices(args.policies),
                quantization_modes=parse_csv_choices(args.quantization_modes),
                execute_external=args.execute_external,
                max_taskgraph_tasks=args.max_taskgraph_tasks,
                fail_fast=args.fail_fast,
                artifact_retention=args.artifact_retention,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.summary_path),
                    "results_csv": str(result.run_dir / "upmem_mvp_benchmark_results.csv"),
                    "kernel_family_summary": str(result.run_dir / "kernel_family_summary.csv"),
                    "quantization_accuracy_summary": str(result.run_dir / "quantization_accuracy_summary.csv"),
                    "unsupported_reasons": str(result.run_dir / "unsupported_reasons.csv"),
                    "summary": str(result.run_dir / "comparison_summary.md"),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "simulation-backend-compare":
        from quantum_bench.bench.simulation_backend_compare import run_simulation_backend_compare

        try:
            result = run_simulation_backend_compare(
                root_dir,
                suite_path=suite_path(args.suite, root_dir),
                artifact_retention=args.artifact_retention,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.summary_path),
                    "results_csv": str(result.run_dir / "simulation_backend_compare_results.csv"),
                    "summary": str(result.run_dir / "comparison_summary.md"),
                    "status": result.status,
                    "case_count": result.case_count,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "report-run":
        from quantum_bench.bench.reporting import report_run

        try:
            result = report_run(Path(args.input), output_plots=args.output_plots)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps({"run_dir": str(result.run_dir), "report": str(result.report_path), "status": result.status, "reason": result.reason}, indent=2))
        return 0
    if args.command == "prune-run":
        from quantum_bench.bench.reporting import prune_run

        try:
            result = prune_run(Path(args.input), artifact_retention=args.artifact_retention)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact_retention_manifest": str(result.manifest_path),
                    "status": result.status,
                    "pruned_file_count": result.pruned_file_count,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "compare-runs":
        from quantum_bench.bench.reporting import compare_runs

        result = compare_runs(Path(args.baseline), Path(args.candidate), Path(args.out))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.artifact_path),
                    "summary": str(result.summary_path),
                    "status": result.status,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "dense-route-coverage":
        from quantum_bench.bench.dense_route_coverage import run_dense_route_coverage, validate_cli_options

        resolved_suite = suite_path(args.suite, root_dir) if args.suite else None
        try:
            validate_cli_options(
                suite_path=resolved_suite,
                case=args.case,
                bridge_backend=args.bridge_backend,
                execute_external=args.execute_external,
                max_bridge_artifacts=args.max_bridge_artifacts,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_dir = run_dense_route_coverage(
            root_dir,
            suite_path=resolved_suite,
            case=args.case,
            n_qubits=args.n_qubits,
            bridge_backend=args.bridge_backend,
            execute_external=args.execute_external,
            max_bridge_artifacts=args.max_bridge_artifacts,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(run_dir / "dense_route_coverage.json"),
                    "csv": str(run_dir / "dense_route_coverage.csv"),
                    "summary": str(run_dir / "dense_route_coverage_summary.md"),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "pim-bridge-eval":
        from quantum_bench.bench.pim_bridge_eval import run_pim_bridge_eval, validate_cli_options

        resolved_suite = suite_path(args.suite, root_dir) if args.suite else None
        try:
            validate_cli_options(
                suite_path=resolved_suite,
                case=args.case,
                n_qubits=args.n_qubits,
                backend=args.backend,
                execute_external=args.execute_external,
                dry_run=args.dry_run,
                max_tasks_per_case=args.max_tasks_per_case,
                max_executed_tasks_per_case=args.max_executed_tasks_per_case,
                task_selection=args.task_selection,
                timeout_seconds=args.timeout_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_dir = run_pim_bridge_eval(
            root_dir,
            suite_path=resolved_suite,
            case=args.case,
            n_qubits=args.n_qubits,
            backend=args.backend,
            execute_external=args.execute_external,
            dry_run=args.dry_run,
            max_tasks_per_case=args.max_tasks_per_case,
            max_executed_tasks_per_case=args.max_executed_tasks_per_case,
            task_selection=args.task_selection,
            timeout_seconds=args.timeout_seconds,
            planner=args.planner,
            output_plots=args.output_plots,
            debug_failures=args.debug_failures,
            compare_mock_on_failure=args.compare_mock_on_failure,
            keep_failure_artifacts=args.keep_failure_artifacts,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(run_dir / "pim_bridge_eval.json"),
                    "csv": str(run_dir / "pim_bridge_eval.csv"),
                    "cases_csv": str(run_dir / "pim_bridge_eval_cases.csv"),
                    "summary": str(run_dir / "pim_bridge_eval_summary.md"),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "pim-frontier-analysis":
        from quantum_bench.bench.pim_frontier_analysis import run_pim_frontier_analysis, validate_cli_options
        from quantum_bench.targets.upmem import UpmemResourceModel

        resolved_suite = suite_path(args.suite, root_dir) if args.suite else None
        try:
            resource_model = UpmemResourceModel(
                available_dpus=args.available_dpus,
                per_dpu_wram_bytes=args.per_dpu_wram_bytes,
                effective_wram_bytes=args.effective_wram_bytes,
                per_dpu_mram_bytes=args.per_dpu_mram_bytes,
                max_task_group_dpus=args.max_task_group_dpus,
            )
            validate_cli_options(
                suite_path=resolved_suite,
                case=args.case,
                n_qubits=args.n_qubits,
                resource_model=resource_model,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_dir = run_pim_frontier_analysis(
            root_dir,
            suite_path=resolved_suite,
            case=args.case,
            n_qubits=args.n_qubits,
            resource_model=resource_model,
            output_plots=args.output_plots,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(run_dir / "pim_frontier_analysis.json"),
                    "tasks_csv": str(run_dir / "pim_frontier_analysis_tasks.csv"),
                    "cases_csv": str(run_dir / "pim_frontier_analysis_cases.csv"),
                    "waves_csv": str(run_dir / "pim_frontier_analysis_waves.csv"),
                    "summary": str(run_dir / "pim_frontier_analysis_summary.md"),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "benchmark-matrix-report":
        from quantum_bench.bench.benchmark_matrix_report import run_benchmark_matrix_report

        matrix_path = Path(args.matrix)
        if not matrix_path.is_absolute():
            matrix_path = root_dir / matrix_path
        external_libs_report_path = Path(args.external_libs_report) if args.external_libs_report else None
        if external_libs_report_path is not None and not external_libs_report_path.is_absolute():
            external_libs_report_path = root_dir / external_libs_report_path
        try:
            run_dir = run_benchmark_matrix_report(
                root_dir,
                matrix_path,
                output_plots=args.output_plots,
                external_libs_report_path=external_libs_report_path,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(run_dir / "benchmark_matrix.json"),
                    "csv": str(run_dir / "benchmark_matrix.csv"),
                    "summary": str(run_dir / "benchmark_matrix_summary.md"),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "shadow-routed-runtime":
        from quantum_bench.bench.shadow_routed_runtime import run_shadow_routed_runtime, validate_cli_options

        resolved_suite = suite_path(args.suite, root_dir) if args.suite else None
        try:
            validate_cli_options(
                suite_path=resolved_suite,
                case=args.case,
                dense_shadow=args.dense_shadow,
                bridge_backend=args.bridge_backend,
                execute_external=args.execute_external,
                max_bridge_artifacts=args.max_bridge_artifacts,
                shadow_route_policy=args.shadow_route_policy,
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_dir = run_shadow_routed_runtime(
            root_dir,
            suite_path=resolved_suite,
            case=args.case,
            n_qubits=args.n_qubits,
            dense_shadow=args.dense_shadow,
            bridge_backend=args.bridge_backend,
            execute_external=args.execute_external,
            max_bridge_artifacts=args.max_bridge_artifacts,
            shadow_route_policy=args.shadow_route_policy,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(run_dir / "shadow_routed_runtime.json"),
                    "csv": str(run_dir / "shadow_routed_runtime.csv"),
                    "summary": str(run_dir / "shadow_routed_runtime_summary.md"),
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-env-check":
        from quantum_bench.bench.upmem_env_check import run_upmem_env_check

        try:
            run_dir, artifact_path, status = run_upmem_env_check(
                root_dir,
                run_sample=args.run_sample,
                target=args.target,
                timeout_seconds=args.timeout_seconds,
                simplepim_home=args.simplepim_home,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(artifact_path),
                    "summary": str(run_dir / "upmem_env_check_summary.md"),
                    "status": status,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-external-libs-check":
        from quantum_bench.bench.upmem_external_libs_check import run_upmem_external_libs_check

        try:
            run_dir, artifact_path, status = run_upmem_external_libs_check(
                root_dir,
                simplepim_home=args.simplepim_home,
                pid_comm_home=args.pid_comm_home,
                check_pid_comm_build=args.check_pid_comm_build,
                timeout_seconds=args.timeout_seconds,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "artifact": str(artifact_path),
                    "summary": str(run_dir / "external_pim_libraries_summary.md"),
                    "status": status,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "compare-results":
        from quantum_bench.bench.result_artifacts import compare_results

        try:
            result = compare_results((Path(item) for item in args.inputs), Path(args.out))
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.artifact_path),
                    "csv": str(result.csv_path),
                    "summary": str(result.summary_path),
                    "record_count": result.record_count,
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "probe":
        from quantum_bench.providers import route_registry

        payload = {
            name: {
                "probe": route.probe(),
                "capabilities": route.capabilities(),
            }
            for name, route in route_registry(root_dir).items()
        }
        print(json.dumps(to_jsonable(payload), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
