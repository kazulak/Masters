from __future__ import annotations

import platform
import subprocess
import sys

import numpy as np
import opt_einsum as oe

from tnsim import __version__


def execution_log(
    config: dict,
    graph: dict,
    profiles: list[dict],
    timings: dict,
    repeats: list[dict],
    validation: dict,
) -> dict:
    route_counts: dict[str, int] = {}
    for profile in profiles:
        route = profile["route"]
        route_counts[route] = route_counts.get(route, 0) + 1
    if not route_counts:
        for task in graph["tasks"]:
            route = task["selected_route"]
            route_counts[route] = route_counts.get(route, 0) + 1

    best_repeat = min(repeats, key=lambda item: item["execution_seconds"]) if repeats else {}
    energy_joules = best_repeat.get("energy_joules")
    energy_source = best_repeat.get("energy_source")
    estimated_power_watts = best_repeat.get("estimated_power_watts")

    return {
        "schema_version": "execution_log_stage1a-0.2",
        "experiment_id": config["experiment"]["id"],
        "status": "ok" if validation["passed"] else "validation_failed",
        "pipeline_role": "shared_benchmark_pipeline",
        "config": {
            "planner": config["planner"],
            "execution": config["execution"],
            "measurement": config.get("measurement", {}),
            "validation": config["validation"],
        },
        "environment": environment_record(),
        "summary": {
            "n_tasks": len(graph["tasks"]),
            "route_counts": route_counts,
            "validation_passed": validation["passed"],
            "best_repeat_seconds": best_repeat.get("execution_seconds"),
            "mean_repeat_seconds": (
                sum(item["execution_seconds"] for item in repeats) / len(repeats) if repeats else None
            ),
            "energy_joules": energy_joules,
            "energy_source": energy_source,
            "estimated_power_watts": estimated_power_watts,
        },
        "timings": timings,
        "repeats": repeats,
        "route_decisions": graph["route_decisions"],
        "profiles": profiles,
        "validation": validation,
    }


def base_metrics_line(log: dict, output_dir: str) -> dict:
    validation = log["validation"]
    return {
        "experiment_id": log["experiment_id"],
        "status": log["status"],
        "best_repeat_seconds": log["summary"]["best_repeat_seconds"],
        "mean_repeat_seconds": log["summary"]["mean_repeat_seconds"],
        "energy_joules": log["summary"].get("energy_joules"),
        "energy_source": log["summary"].get("energy_source"),
        "max_abs_error": validation["metrics"]["max_abs_error"],
        "max_rel_error": validation["metrics"]["max_rel_error"],
        "fidelity": validation["metrics"]["fidelity"],
        "output_dir": output_dir,
    }


def environment_record() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opt_einsum": getattr(oe, "__version__", "unknown"),
        "tnsim": __version__,
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

