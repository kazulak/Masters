from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_bench.plots import plot_run
from quantum_bench.bench.config import suite_path
from quantum_bench.bench.runner import run_suite
from quantum_bench.bench.summary import write_summary
from quantum_bench.core.records import to_jsonable
from quantum_bench.providers import route_registry


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

    sub.add_parser("probe")

    args = parser.parse_args()
    if args.command == "run":
        run_dir = run_suite(suite_path(args.suite, root_dir), root_dir)
        created = plot_run(run_dir)
        print(json.dumps({"run_dir": str(run_dir), "plots": [str(path) for path in created]}, indent=2))
        return 0
    if args.command == "summarize":
        summary = write_summary(Path(args.run_dir).resolve())
        print(json.dumps(summary, indent=2))
        return 0
    if args.command == "plot":
        created = plot_run(Path(args.run_dir).resolve(), args.baseline_route)
        print(json.dumps({"plots": [str(path) for path in created]}, indent=2))
        return 0
    if args.command == "probe":
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
