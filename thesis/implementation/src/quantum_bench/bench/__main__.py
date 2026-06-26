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
        choices=("mock_numpy_dequantized", "simplepim_external", "simplepim_external_stub"),
    )
    dense_bridge_parser.add_argument("--execute-external", action="store_true")

    coverage_parser = sub.add_parser("dense-route-coverage")
    coverage_input = coverage_parser.add_mutually_exclusive_group(required=True)
    coverage_input.add_argument("--suite", help="Suite path or preset name under configs/suites")
    coverage_input.add_argument("--case", help="Builtin circuit case name")
    coverage_parser.add_argument("--n-qubits", type=int)
    coverage_parser.add_argument("--bridge-backend", default="none", choices=("none", "simplepim_external_stub"))
    coverage_parser.add_argument("--execute-external", action="store_true")
    coverage_parser.add_argument("--max-bridge-artifacts", type=int, default=0)

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
