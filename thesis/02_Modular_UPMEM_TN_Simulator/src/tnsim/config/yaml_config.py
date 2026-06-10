from __future__ import annotations

from pathlib import Path
import copy

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs.
    raise SystemExit(
        "PyYAML is required for benchmark configs. Install with "
        "`python -m pip install -r requirements.txt` from 02_Modular_UPMEM_TN_Simulator."
    ) from exc


SCHEMA_VERSION = "benchmark_config-0.1"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, dict):
        raise ValueError("Benchmark config must be a YAML mapping")
    return resolve_config(user_config)


def resolve_config(user_config: dict) -> dict:
    defaults = {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "id": "unnamed_experiment",
            "description": "",
            "seed": 0,
            "tags": ["stage_1a", "tn_cpu"],
            "output_dir": "runs/unnamed_experiment",
        },
        "workload": {
            "kind": "builtin_circuit",
            "name": "bell_2q",
        },
        "planner": {
            "engine": "opt_einsum",
            "optimize": "optimal",
        },
        "execution": {
            "warmups": 1,
            "repeats": 3,
            "routes": {
                "enabled": ["cpu_reference"],
                "forced": "cpu_reference",
                "disabled": [
                    "quest_exact_statevector",
                    "gpu_cupy",
                    "raw_upmem_dense",
                    "simplepim_default",
                    "custom_dense",
                    "sparsep",
                    "pidcomm_collective",
                ],
            },
            "data_format": {
                "name": "complex_f64_host",
                "logical_dtype": "complex_f64",
                "accumulator": "complex_f64",
                "scale_scope": "none",
            },
        },
        "measurement": {
            "energy": {
                "mode": "estimated_static_power",
                "cpu_watts": 65.0,
                "notes": "Used when hardware energy telemetry is unavailable.",
            },
        },
        "validation": {
            "reference": "numpy_full_einsum",
            "tolerances": {
                "max_abs_error": 1.0e-12,
                "max_rel_error": 1.0e-12,
                "norm_drift": 1.0e-12,
                "min_fidelity": 0.999999999999,
            },
        },
        "outputs": {
            "write_task_graph": True,
            "write_execution_log": True,
            "write_validation_record": True,
            "write_metrics_jsonl": True,
        },
    }
    config = _deep_merge(defaults, user_config)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported config schema_version: {config.get('schema_version')}")
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result

