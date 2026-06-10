from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tnsim.config import load_config
from tnsim.results import collect_results, discover_run_dirs, write_result_tables
from tnsim.runner import run


def run_suite(suite_path: Path, output_dir_override: Path | None = None) -> dict:
    root_dir = Path(__file__).resolve().parents[3]
    suite_path = suite_path.resolve()
    suite = _load_suite(suite_path)
    suite_id = suite.get("suite_id", suite_path.stem)
    output_dir = output_dir_override or Path(suite.get("results_dir", f"results/{suite_id}"))
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir

    run_dirs = []
    for item in suite.get("configs", []):
        config_path, run_output_override = _suite_item_paths(item, suite_path.parent, root_dir)
        log = run(config_path, run_output_override)
        run_dirs.append(Path(log["run_dir"]))

    rows = collect_results(run_dirs, suite.get("baseline_experiment"))
    outputs = write_result_tables(rows, output_dir)
    print(f"[suite] {suite_id} runs={len(run_dirs)} results={outputs['markdown']}")
    print(f"[suite] charts={outputs['speedup_svg']}, {outputs['energy_svg']}")
    return {"suite_id": suite_id, "rows": rows, "outputs": outputs}


def analyze_runs(runs_root: Path, output_dir: Path, baseline_experiment: str | None = None) -> dict:
    run_dirs = discover_run_dirs(runs_root)
    rows = collect_results(run_dirs, baseline_experiment)
    outputs = write_result_tables(rows, output_dir)
    print(f"[analyze] runs={len(run_dirs)} results={outputs['markdown']}")
    print(f"[analyze] charts={outputs['speedup_svg']}, {outputs['energy_svg']}")
    return {"rows": rows, "outputs": outputs}


def suite_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a YAML benchmark suite and write result summaries.")
    parser.add_argument("suite", type=Path, help="Suite YAML path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override suite results directory")
    args = parser.parse_args(argv)
    run_suite(args.suite, args.output_dir)


def analyze_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze existing benchmark run directories.")
    parser.add_argument("runs_root", type=Path, help="Directory containing run outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("results/manual"), help="Summary output directory")
    parser.add_argument("--baseline", type=str, default=None, help="Experiment id used for speedup comparison")
    args = parser.parse_args(argv)
    analyze_runs(args.runs_root, args.output_dir, args.baseline)


def _load_suite(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    if not isinstance(suite, dict):
        raise ValueError("Suite config must be a YAML mapping")
    return suite


def _suite_item_paths(item, suite_dir: Path, root_dir: Path) -> tuple[Path, Path | None]:
    if isinstance(item, str):
        config_path = Path(item)
        output_dir = None
    elif isinstance(item, dict):
        config_path = Path(item["path"])
        output_dir = Path(item["output_dir"]) if item.get("output_dir") else None
    else:
        raise ValueError(f"Unsupported suite config item: {item!r}")

    if not config_path.is_absolute():
        direct = suite_dir / config_path
        config_path = direct if direct.exists() else root_dir / config_path
    if output_dir is not None and not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    load_config(config_path)
    return config_path, output_dir

