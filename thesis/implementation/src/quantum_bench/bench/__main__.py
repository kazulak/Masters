from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_bench.bench.config import suite_path
from quantum_bench.bench.planner_compare import compare_planners


def main() -> int:
    root_dir = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(prog="python -m quantum_bench.bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )

    compare_parser = sub.add_parser("compare-planners")
    compare_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )

    generic_bridge_parser = sub.add_parser("generic-task-bridge")
    generic_bridge_parser.add_argument("--case", default="bell_2q")
    generic_bridge_parser.add_argument("--n-qubits", type=int)
    generic_bridge_parser.add_argument("--task-index", type=int, required=True)
    generic_bridge_parser.add_argument(
        "--backend",
        default="upmem_sdk_simulator_generic_loop",
        choices=("upmem_sdk_simulator_generic_loop",),
    )
    generic_bridge_parser.add_argument("--execute-external", action="store_true")

    upmem_runtime_parser = sub.add_parser("upmem-taskgraph-runtime")
    upmem_runtime_parser.add_argument("--case", default="bell_2q")
    upmem_runtime_parser.add_argument("--n-qubits", type=int)
    upmem_runtime_parser.add_argument(
        "--policy",
        default="generic-only",
        choices=("generic-only",),
    )
    upmem_runtime_parser.add_argument(
        "--quantization-mode",
        default="per_task_input_quantize",
        choices=("per_task_input_quantize", "none"),
    )
    upmem_runtime_parser.add_argument("--execute-external", action="store_true")

    upmem_mvp_parser = sub.add_parser("upmem-mvp-benchmark")
    upmem_mvp_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )
    upmem_mvp_parser.add_argument("--policies", default="generic-only")
    upmem_mvp_parser.add_argument(
        "--quantization-modes", default="per_task_input_quantize"
    )
    upmem_mvp_parser.add_argument("--execute-external", action="store_true")
    upmem_mvp_parser.add_argument("--max-taskgraph-tasks", type=int, default=128)
    upmem_mvp_parser.add_argument("--fail-fast", action="store_true")
    upmem_mvp_parser.add_argument(
        "--artifact-retention",
        default="compact",
        choices=("full", "compact", "summary-only"),
    )

    upmem_hardware_taskgraph_resident_parser = sub.add_parser(
        "upmem-hardware-taskgraph-resident",
        help="run the guarded physical one-DPU MRAM-resident TaskGraph route",
    )
    upmem_hardware_taskgraph_resident_parser.add_argument(
        "--suite", required=True, help="MRAM-resident hardware TaskGraph suite YAML"
    )
    upmem_hardware_taskgraph_resident_mode = (
        upmem_hardware_taskgraph_resident_parser.add_mutually_exclusive_group(
            required=True
        )
    )
    upmem_hardware_taskgraph_resident_mode.add_argument(
        "--prepare-only", action="store_true"
    )
    upmem_hardware_taskgraph_resident_mode.add_argument(
        "--execute", action="store_true"
    )
    upmem_hardware_taskgraph_resident_parser.add_argument(
        "--build",
        action="store_true",
        help="build the separate resident native source during --prepare-only; never allocates a DPU",
    )

    upmem_hardware_sliced_resident_parser = sub.add_parser(
        "upmem-hardware-sliced-resident-mvp",
        help="run the guarded internal/research two-DPU sliced-resident M2 MVP",
    )
    upmem_hardware_sliced_resident_parser.add_argument(
        "--suite", required=True, help="committed M2 two-DPU sliced-resident suite YAML"
    )
    upmem_hardware_sliced_resident_mode = (
        upmem_hardware_sliced_resident_parser.add_mutually_exclusive_group(
            required=True
        )
    )
    upmem_hardware_sliced_resident_mode.add_argument(
        "--prepare-only", action="store_true"
    )
    upmem_hardware_sliced_resident_mode.add_argument("--execute", action="store_true")
    upmem_hardware_sliced_resident_parser.add_argument(
        "--build",
        action="store_true",
        help="build M2 native sources during --prepare-only; never allocates a DPU",
    )

    upmem_hardware_frontier_m3_1_parser = sub.add_parser(
        "upmem-hardware-frontier-m3-1",
        help="run the guarded physical M3.1 two-DPU frontier route",
    )
    upmem_hardware_frontier_m3_1_parser.add_argument(
        "--suite", required=True, help="committed M3.1 frontier hardware suite YAML"
    )
    upmem_hardware_frontier_m3_1_mode = (
        upmem_hardware_frontier_m3_1_parser.add_mutually_exclusive_group(required=True)
    )
    upmem_hardware_frontier_m3_1_mode.add_argument("--prepare-only", action="store_true")
    upmem_hardware_frontier_m3_1_mode.add_argument("--execute", action="store_true")
    upmem_hardware_frontier_m3_1_parser.add_argument(
        "--build",
        action="store_true",
        help="build the M3.1 native source during --prepare-only; never allocates a DPU",
    )

    upmem_generic_feasibility_parser = sub.add_parser("upmem-generic-feasibility")
    upmem_generic_feasibility_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )
    upmem_generic_feasibility_parser.add_argument(
        "--quantization-modes", default="none,per_task_input_quantize"
    )
    upmem_generic_feasibility_parser.add_argument(
        "--max-taskgraph-tasks", type=int, default=128
    )

    upmem_multi_dpu_parser = sub.add_parser("upmem-multi-dpu-assignment")
    upmem_multi_dpu_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )
    upmem_multi_dpu_parser.add_argument("--dpu-groups", type=int, default=4)
    upmem_multi_dpu_parser.add_argument(
        "--strategy",
        default="frontier_round_robin_dpu_groups",
        choices=(
            "sequential_single_dpu",
            "frontier_round_robin_dpu_groups",
            "frontier_size_aware_dpu_groups",
        ),
    )

    simulation_compare_parser = sub.add_parser("simulation-backend-compare")
    simulation_compare_parser.add_argument(
        "--suite", required=True, help="Suite path or preset name under configs/suites"
    )
    simulation_compare_parser.add_argument(
        "--artifact-retention",
        default="compact",
        choices=("full", "compact", "summary-only"),
    )

    simulation_probe_parser = sub.add_parser("simulation-backend-probe")
    simulation_probe_parser.add_argument(
        "--verify-gpu",
        default="none",
        choices=("none", "auto", "quest-hip", "quest-cuda", "torch-rocm"),
    )
    simulation_probe_parser.add_argument(
        "--verify-gpu-tn",
        default="none",
        choices=(
            "none",
            "auto",
            "cuquantum-cutensornet",
            "cudaq-tensornet",
            "qiskit-aer-tensor-network",
            "cupy-rocm-generic",
        ),
    )

    report_run_parser = sub.add_parser("report-run")
    report_run_parser.add_argument("--input", required=True)
    report_run_parser.add_argument("--out", required=True)
    report_run_parser.add_argument(
        "--output-plots", action=argparse.BooleanOptionalAction, default=True
    )

    prune_run_parser = sub.add_parser("prune-run")
    prune_run_parser.add_argument("--input", required=True)
    prune_run_parser.add_argument(
        "--artifact-retention", default="compact", choices=("compact", "summary-only")
    )

    compare_runs_parser = sub.add_parser("compare-runs")
    compare_runs_parser.add_argument("--baseline", required=True)
    compare_runs_parser.add_argument("--candidate", required=True)
    compare_runs_parser.add_argument("--out", required=True)

    matrix_parser = sub.add_parser("benchmark-matrix-report")
    matrix_parser.add_argument(
        "--matrix", required=True, help="Benchmark matrix YAML path"
    )
    matrix_parser.add_argument(
        "--output-plots", action=argparse.BooleanOptionalAction, default=True
    )

    upmem_env_parser = sub.add_parser("upmem-env-check")
    upmem_env_parser.add_argument("--run-sample", action="store_true")
    upmem_env_parser.add_argument(
        "--target", default="auto", choices=("auto", "simulator", "hardware")
    )
    upmem_env_parser.add_argument("--timeout-seconds", type=float, default=10.0)

    provider_qualification_parser = sub.add_parser(
        "provider-qualification",
        help="prepare or execute the guarded M1 physical provider qualification",
    )
    provider_qualification_parser.add_argument("--catalog", required=True)
    provider_qualification_mode = (
        provider_qualification_parser.add_mutually_exclusive_group(required=True)
    )
    provider_qualification_mode.add_argument("--prepare-only", action="store_true")
    provider_qualification_mode.add_argument("--execute", action="store_true")
    provider_qualification_parser.add_argument("--provider")

    compare_results_parser = sub.add_parser("compare-results")
    compare_results_parser.add_argument("--inputs", nargs="+", required=True)
    compare_results_parser.add_argument("--out", required=True)
    compare_results_parser.add_argument(
        "--comparison-type", default="generic_comparison"
    )

    args = parser.parse_args()
    if args.command == "run":
        from quantum_bench.bench.runner import run_suite

        run_dir = run_suite(suite_path(args.suite, root_dir), root_dir)
        print(json.dumps({"run_dir": str(run_dir)}, indent=2))
        return 0
    if args.command == "compare-planners":
        run_dir = compare_planners(suite_path(args.suite, root_dir), root_dir)
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "planner_comparison": str(run_dir / "planner_comparison.json"),
                    "planner_comparison_csv": str(run_dir / "planner_comparison.csv"),
                    "planner_comparison_summary": str(
                        run_dir / "planner_comparison_summary.md"
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "generic-task-bridge":
        from quantum_bench.bench.generic_task_bridge import run_generic_task_bridge

        if (
            args.backend == "upmem_sdk_simulator_generic_loop"
            and not args.execute_external
        ):
            parser.error(
                "--backend upmem_sdk_simulator_generic_loop requires --execute-external"
            )
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
        from quantum_bench.bench.upmem_taskgraph_runtime import (
            run_upmem_taskgraph_runtime,
        )

        if not args.execute_external:
            parser.error(
                "upmem-taskgraph-runtime requires --execute-external for strict UPMEM SDK DPU execution"
            )
        if args.quantization_mode == "none" and args.policy != "generic-only":
            parser.error(
                "--quantization-mode none currently requires --policy generic-only"
            )
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
                    "task_metrics": str(
                        result.run_dir
                        / "cases"
                        / result.case_id
                        / "upmem_taskgraph_task_metrics.jsonl"
                    ),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-mvp-benchmark":
        from quantum_bench.bench.upmem_mvp_benchmark import (
            parse_csv_choices,
            run_upmem_mvp_benchmark,
        )

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
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-hardware-taskgraph-resident":
        from quantum_bench.bench.upmem_hardware_taskgraph_resident import (
            prepare_upmem_hardware_taskgraph_resident,
            run_upmem_hardware_taskgraph_resident,
        )

        if args.build and not args.prepare_only:
            parser.error("--build is only valid with --prepare-only")
        try:
            if args.prepare_only:
                result = prepare_upmem_hardware_taskgraph_resident(
                    root_dir,
                    suite_path=suite_path(args.suite, root_dir),
                    build=args.build,
                )
                print(
                    json.dumps(
                        {
                            "plan_dir": str(result.plan_dir),
                            "artifact": str(result.summary_path),
                            "status": result.status,
                            "dpu_allocation_attempted": False,
                            "dpu_launch_attempted": False,
                        },
                        indent=2,
                    )
                )
                return 0 if result.status == "prepared" else 2
            result = run_upmem_hardware_taskgraph_resident(
                root_dir, suite_path=suite_path(args.suite, root_dir)
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.summary_path),
                    "status": result.status,
                    "row_count": result.row_count,
                },
                indent=2,
            )
        )
        return 0 if result.status == "completed" else 2
    if args.command == "upmem-hardware-sliced-resident-mvp":
        from quantum_bench.bench.upmem_hardware_sliced_resident_mvp import (
            prepare_upmem_hardware_sliced_resident_mvp,
            run_upmem_hardware_sliced_resident_mvp,
        )

        if args.build and not args.prepare_only:
            parser.error("--build is only valid with --prepare-only")
        try:
            if args.prepare_only:
                result = prepare_upmem_hardware_sliced_resident_mvp(
                    root_dir,
                    suite_path=suite_path(args.suite, root_dir),
                    build=args.build,
                )
                print(
                    json.dumps(
                        {
                            "plan_dir": str(result.plan_dir),
                            "artifact": str(result.summary_path),
                            "status": result.status,
                            "dpu_allocation_attempted": False,
                            "dpu_launch_attempted": False,
                        },
                        indent=2,
                    )
                )
                return 0 if result.status == "prepared" else 2
            result = run_upmem_hardware_sliced_resident_mvp(
                root_dir, suite_path=suite_path(args.suite, root_dir)
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.summary_path),
                    "status": result.status,
                    "row_count": result.row_count,
                },
                indent=2,
            )
        )
        return 0 if result.status == "completed" else 2
    if args.command == "upmem-hardware-frontier-m3-1":
        from quantum_bench.bench.upmem_hardware_frontier_m3_1 import (
            prepare_upmem_hardware_frontier_m3_1,
            run_upmem_hardware_frontier_m3_1,
        )

        if args.build and not args.prepare_only:
            parser.error("--build is only valid with --prepare-only")
        try:
            if args.prepare_only:
                result = prepare_upmem_hardware_frontier_m3_1(
                    root_dir,
                    suite_path=suite_path(args.suite, root_dir),
                    build=args.build,
                )
                print(
                    json.dumps(
                        {
                            "plan_dir": str(result.plan_dir),
                            "artifact": str(result.summary_path),
                            "status": result.status,
                            "dpu_allocation_attempted": False,
                            "dpu_launch_attempted": False,
                        },
                        indent=2,
                    )
                )
                return 0 if result.status == "prepared" else 2
            result = run_upmem_hardware_frontier_m3_1(
                root_dir, suite_path=suite_path(args.suite, root_dir)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.summary_path),
                    "status": result.status,
                    "row_count": result.row_count,
                },
                indent=2,
            )
        )
        return 0 if result.status == "completed" else 2
    if args.command == "upmem-generic-feasibility":
        from quantum_bench.bench.upmem_generic_feasibility import (
            parse_csv_choices,
            run_upmem_generic_feasibility,
        )

        try:
            result = run_upmem_generic_feasibility(
                root_dir,
                suite_path=suite_path(args.suite, root_dir),
                quantization_modes=parse_csv_choices(args.quantization_modes),
                max_taskgraph_tasks=args.max_taskgraph_tasks,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "summary_path": str(result.summary_path),
                    "status": result.status,
                    "row_count": result.row_count,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "upmem-multi-dpu-assignment":
        from quantum_bench.bench.upmem_multi_dpu_assignment import (
            run_upmem_multi_dpu_assignment,
        )

        try:
            result = run_upmem_multi_dpu_assignment(
                root_dir,
                suite_path=suite_path(args.suite, root_dir),
                dpu_group_count=args.dpu_groups,
                strategy=args.strategy,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.plan_path),
                    "normalized_records": str(result.normalized_records_path),
                    "status": result.status,
                    "case_count": result.case_count,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "simulation-backend-compare":
        from quantum_bench.bench.simulation_backend_compare import (
            run_simulation_backend_compare,
        )

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
                    "status": result.status,
                    "case_count": result.case_count,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "simulation-backend-probe":
        from quantum_bench.bench.simulation_backend_probe import (
            probe_simulation_backends,
        )

        print(
            json.dumps(
                probe_simulation_backends(
                    root_dir,
                    verify_gpu=args.verify_gpu,
                    verify_gpu_tn=args.verify_gpu_tn,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "report-run":
        from quantum_bench.bench.reporting import report_run

        try:
            result = report_run(
                Path(args.input),
                Path(args.out),
                output_plots=args.output_plots,
                root_dir=root_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "report": str(result.report_path),
                    "status": result.status,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "prune-run":
        from quantum_bench.bench.reporting import prune_run

        try:
            result = prune_run(
                Path(args.input), artifact_retention=args.artifact_retention
            )
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
    if args.command == "benchmark-matrix-report":
        from quantum_bench.bench.benchmark_matrix_report import (
            run_benchmark_matrix_report,
        )

        matrix_path = Path(args.matrix)
        if not matrix_path.is_absolute():
            matrix_path = root_dir / matrix_path
        try:
            run_dir = run_benchmark_matrix_report(
                root_dir,
                matrix_path,
                output_plots=args.output_plots,
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
    if args.command == "upmem-env-check":
        from quantum_bench.bench.upmem_env_check import run_upmem_env_check

        try:
            run_dir, artifact_path, status = run_upmem_env_check(
                root_dir,
                run_sample=args.run_sample,
                target=args.target,
                timeout_seconds=args.timeout_seconds,
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
    if args.command == "provider-qualification":
        from quantum_bench.bench.provider_qualification import (
            execute_provider_qualification,
            prepare_provider_qualification,
        )

        catalog = Path(args.catalog)
        try:
            if args.prepare_only:
                result = prepare_provider_qualification(
                    root_dir, catalog_path=catalog, provider_id=args.provider
                )
                print(
                    json.dumps(
                        {"plan": str(result.plan_path), "status": result.status},
                        indent=2,
                    )
                )
                return 0 if result.status == "prepared" else 2
            result = execute_provider_qualification(
                root_dir, catalog_path=catalog, provider_id=args.provider
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "result": str(result.result_path),
                    "raw_result": str(result.raw_result_path)
                    if result.raw_result_path.is_file()
                    else None,
                    "normalized_records": str(result.normalized_records_path),
                    "manifest": str(result.manifest_path),
                    "status": result.status,
                },
                indent=2,
            )
        )
        return 0 if result.status == "qualified" else 2
    if args.command == "compare-results":
        from quantum_bench.bench.result_artifacts import compare_results

        try:
            result = compare_results(
                (Path(item) for item in args.inputs),
                Path(args.out),
                comparison_type=args.comparison_type,
                root_dir=root_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "run_dir": str(result.run_dir),
                    "artifact": str(result.artifact_path),
                    "csv": str(result.csv_path),
                    "summary": str(result.summary_path),
                    "manifest": str(result.manifest_path)
                    if result.manifest_path
                    else None,
                    "record_count": result.record_count,
                    "status": "completed",
                },
                indent=2,
            )
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
