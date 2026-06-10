from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from .config import load_config
from .execution import execute_backend
from .core.files import write_json, write_yaml
from .records import base_metrics_line, execution_log
from .task_graph import plan_task_graph
from .network import build_tensor_network
from .validation import compute_reference, validation_record


def run(config_path: Path, output_dir_override: Path | None = None) -> dict:
    root_dir = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    config = load_config(config_path)
    output_dir = output_dir_override or Path(config["experiment"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, output_dir / "input_config.yaml")
    write_yaml(output_dir / "resolved_config.yaml", config)

    build_start = time.perf_counter()
    network = build_tensor_network(config, root_dir)
    build_seconds = time.perf_counter() - build_start
    graph, _active, planning_seconds = plan_task_graph(network, config, root_dir)

    warmups = int(config["execution"].get("warmups", 0))
    repeats = int(config["execution"].get("repeats", 1))
    if repeats < 1:
        raise ValueError("execution.repeats must be >= 1")
    for _ in range(warmups):
        execute_backend(graph, network, config, output_dir)

    repeat_records = []
    final_output = None
    final_profiles = []
    for repeat_index in range(repeats):
        result = execute_backend(graph, network, config, output_dir)
        final_output = result.output
        final_profiles = result.profiles
        repeat_records.append(
            {
                "repeat_index": repeat_index,
                "execution_seconds": result.execution_seconds,
                "energy_joules": result.energy_joules,
                "energy_source": result.energy_source,
                "estimated_power_watts": result.estimated_power_watts,
                "n_task_profiles": len(result.profiles),
            }
        )

    reference, reference_seconds = compute_reference(network, config["planner"]["optimize"])
    validation_start = time.perf_counter()
    validation = validation_record(
        final_output,
        reference,
        config,
        time.perf_counter() - validation_start,
    )

    graph["profiles"] = final_profiles
    graph["validation"] = [validation]
    timings = {
        "build_tensor_network_seconds": build_seconds,
        "planning_seconds": planning_seconds,
        "reference_seconds": reference_seconds,
        "warmups": warmups,
        "repeats": repeats,
    }
    log = execution_log(config, graph, final_profiles, timings, repeat_records, validation)
    log["run_dir"] = str(output_dir)

    if config["outputs"]["write_task_graph"]:
        write_json(output_dir / "task_graph.json", graph)
    if config["outputs"]["write_execution_log"]:
        write_json(output_dir / "execution_log.json", log)
    if config["outputs"]["write_validation_record"]:
        write_json(output_dir / "validation_record.json", validation)
    if config["outputs"]["write_metrics_jsonl"]:
        metrics_line = base_metrics_line(log, str(output_dir))
        with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics_line, sort_keys=True) + "\n")

    print(
        "[benchmark] "
        f"{config['experiment']['id']} status={log['status']} "
        f"tasks={len(graph['tasks'])} "
        f"best={log['summary']['best_repeat_seconds']:.6e}s "
        f"energy={log['summary'].get('energy_joules'):.6e}J "
        f"max_abs={validation['metrics']['max_abs_error']:.3e} "
        f"out={output_dir}"
    )
    if not validation["passed"]:
        raise SystemExit(2)
    return log


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a YAML-defined shared TaskGraph benchmark.")
    parser.add_argument("config", type=Path, help="Benchmark YAML config path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args(argv)
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
