#!/usr/bin/env python3
"""Prepare, inspect, and bundle the frozen sequential UPMEM reference baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
from statistics import median
import subprocess
import tarfile
from typing import Any, Mapping, Sequence

import yaml

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
CORRECTNESS_TEMPLATE = ROOT / "configs" / "tn_benchmark_sequential_upmem_correctness.yml"
PERFORMANCE_TEMPLATE = ROOT / "configs" / "tn_benchmark_sequential_upmem_performance.yml"
EXTERNAL_TEMPLATE = ROOT / "configs" / "tn_benchmark_external_tn_context.yml"
CORRECTNESS_CONFIG_NAME = "sequential-upmem-correctness.yml"
PERFORMANCE_CONFIG_NAME = "sequential-upmem-performance.yml"
SUMMARY_SCHEMA = "sequential_upmem_baseline_summary_v1"
CONFORMANCE_SCHEMA = "sequential_statevector_conformance_v1"
REQUIRED_FIXTURES = frozenset(
    {
        "basis_order_2q",
        "Bell2",
        "complex_orientation_3q",
        "GHZ5",
        "QuEST-compatible QRNG3",
        "QuEST-compatible BV5",
        "Stress18",
        "sliced Stress4",
    }
)
CONFORMANCE_POLICIES = {
    "complex128_1e-12": {
        "raw_allclose_atol": 1.0e-12,
        "raw_allclose_rtol": 1.0e-12,
        "max_abs_error_max": 1.0e-12,
        "relative_l2_error_max": 1.0e-12,
        "norm_drift_max": 1.0e-12,
    },
    "float32_1e-5": {
        "raw_allclose_atol": 1.0e-5,
        "raw_allclose_rtol": 1.0e-5,
        "max_abs_error_max": 1.0e-5,
        "relative_l2_error_max": 1.0e-5,
        "norm_drift_max": 2.0e-5,
    },
}
_BINARY_FIELDS = ("host_binary", "dpu_binary")
_BINARY_HASH_FIELDS = {
    "host_binary": "host_binary_sha256",
    "dpu_binary": "dpu_binary_sha256",
}
_MACHINE_PATH_FIELDS = ("session_root", *_BINARY_FIELDS)
_PHYSICAL_FACTS = {
    "target_observed": "physical_hardware",
    "hardware_kernel_executed": True,
    "simulator_kernel_executed": False,
    "cpu_fallback_used": False,
    "physical_target_verified": True,
    "hardware_release_verified": True,
    "binary_identity_verified": True,
    "native_identity_verified": True,
    "startup_resource_admission_passed": True,
    "execution_resource_admission_passed": True,
    "rank_count": 1,
    "observed_rank_count": 1,
    "requested_dpus": 1,
    "allocated_dpus": 1,
    "active_dpus": 1,
    "tasklets_per_dpu": 1,
}
_POWERSAVE_AFFINITY = [0]
_POWERSAVE_GOVERNORS = {"0": "powersave"}
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_BASELINE_BOOTSTRAP_RESAMPLES = 10_000
_BASELINE_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True) + "\n", encoding="ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*") if path.is_file()))


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_source() -> str:
    if _git_output("status", "--porcelain"):
        raise ValueError("sequential baseline inspection requires a clean Git worktree")
    return _git_output("rev-parse", "HEAD")


def _parse_cpus(value: str) -> list[int]:
    try:
        cpus = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise ValueError("expected CPUs must be a comma-separated integer list") from exc
    if not cpus or any(cpu < 0 for cpu in cpus) or len(cpus) != len(set(cpus)):
        raise ValueError("expected CPUs must be unique nonnegative integers")
    return cpus


def _template_configuration(path: Path) -> dict[str, Any]:
    normalized = _plain(load_experiment_config(path))
    payload = normalized.get("experiment_identity_payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("configuration"), Mapping):
        raise ValueError(f"tracked template lacks an experiment identity payload: {path}")
    return _plain(payload["configuration"])


def _absolute(value: str, *, relative_to: Path | None = None) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute() and relative_to is not None:
        path = relative_to / path
    return str(path.resolve())


def _prepare_configuration(
    *,
    template: Path,
    output: Path,
    rank_path: str,
    session_root: str,
    expected_cpus: Sequence[int],
    binaries: Mapping[str, str | None],
) -> Path:
    if output.exists():
        raise ValueError(f"prepared configuration must be absent: {output}")
    config = _template_configuration(template)
    physical_routes = [
        route
        for route in config["routes"].values()
        if route["executor"] == "upmem_physical"
    ]
    if len(physical_routes) != 1:
        raise ValueError("sequential template must contain exactly one physical route")
    options = physical_routes[0]["options"]
    if [options[field] for field in ("rank_count", "dpu_count", "tasklets_per_dpu")] != [1, 1, 1]:
        raise ValueError("sequential template must request one rank, DPU, and tasklet")
    options["rank_paths"] = [_absolute(rank_path)]
    options["session_root"] = _absolute(session_root)
    for field in _BINARY_FIELDS:
        configured = binaries[field] or options[field]
        options[field] = _absolute(configured, relative_to=template.parent)
    affinity = config["collection"]["machine_policy"]["affinity"]
    affinity["mode"] = "exact_required_v1"
    affinity["expected_cpus"] = list(expected_cpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    loaded = load_experiment_config(output)
    loaded_options = next(
        route["options"]
        for route in loaded["routes"].values()
        if route["executor"] == "upmem_physical"
    )
    for field in _MACHINE_PATH_FIELDS:
        if loaded_options[field] != options[field]:
            raise ValueError(f"prepared {field} did not retain its resolved path")
    if tuple(loaded_options["rank_paths"]) != tuple(options["rank_paths"]):
        raise ValueError("prepared rank path did not retain its resolved path")
    return output


def prepare_configs(
    *,
    output_dir: Path,
    rank_path: str,
    correctness_session_root: str,
    performance_session_root: str,
    expected_cpus: Sequence[int],
    host_binary: str | None = None,
    dpu_binary: str | None = None,
) -> Mapping[str, Any]:
    resolved_output = output_dir.resolve()
    if resolved_output.is_relative_to(ROOT) and subprocess.run(
        ["git", "check-ignore", "--quiet", str(resolved_output)],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("prepared configs inside the repository must use an ignored path")
    binaries = {
        "host_binary": host_binary,
        "dpu_binary": dpu_binary,
    }
    correctness = _prepare_configuration(
        template=CORRECTNESS_TEMPLATE,
        output=output_dir / CORRECTNESS_CONFIG_NAME,
        rank_path=rank_path,
        session_root=correctness_session_root,
        expected_cpus=expected_cpus,
        binaries=binaries,
    )
    performance = _prepare_configuration(
        template=PERFORMANCE_TEMPLATE,
        output=output_dir / PERFORMANCE_CONFIG_NAME,
        rank_path=rank_path,
        session_root=performance_session_root,
        expected_cpus=expected_cpus,
        binaries=binaries,
    )
    return {
        "status": "prepared",
        "correctness_config": str(correctness),
        "performance_config": str(performance),
    }


def _masked_contract(config: Mapping[str, Any], *, physical: bool) -> dict[str, Any]:
    result = _plain(config)
    if not physical:
        return result
    affinity = result["collection"]["machine_policy"]["affinity"]
    if affinity.get("mode") != "exact_required_v1":
        raise ValueError("resolved physical config requires exact CPU affinity")
    cpus = affinity.get("expected_cpus")
    if not isinstance(cpus, list) or not cpus or any(not isinstance(cpu, int) for cpu in cpus):
        raise ValueError("resolved physical config requires a nonempty expected CPU set")
    affinity["expected_cpus"] = "<machine-specific>"
    for route in result["routes"].values():
        if route["executor"] != "upmem_physical":
            continue
        options = route["options"]
        for field in _MACHINE_PATH_FIELDS:
            if not Path(options[field]).is_absolute():
                raise ValueError(f"resolved physical config requires absolute {field}")
            options[field] = "<machine-specific>"
        rank_paths = options.get("rank_paths")
        if not isinstance(rank_paths, list) or len(rank_paths) != 1 or not Path(rank_paths[0]).is_absolute():
            raise ValueError("resolved physical config requires one absolute rank path")
        options["rank_paths"] = ["<machine-specific>"]
    return result


def _experiment_configuration(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    experiment = manifest["configuration"]["experiment"]
    payload = experiment.get("experiment_identity_payload")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("configuration"), Mapping):
        raise ValueError("evidence manifest lacks its experiment identity configuration")
    return payload["configuration"]


def _require_contract(manifest: Mapping[str, Any], template: Path, *, physical: bool) -> None:
    experiment = manifest["configuration"]["experiment"]
    payload = experiment.get("experiment_identity_payload")
    if not isinstance(payload, Mapping) or payload.get("validation_policy_id") != manifest.get("validation_policy_id"):
        raise ValueError("evidence experiment identity is not bound to its validation policy")
    actual = _masked_contract(_experiment_configuration(manifest), physical=physical)
    expected_config = _template_configuration(template)
    if physical:
        expected_config["collection"]["machine_policy"]["affinity"]["mode"] = "exact_required_v1"
        expected_config["collection"]["machine_policy"]["affinity"]["expected_cpus"] = [0]
        for route in expected_config["routes"].values():
            if route["executor"] == "upmem_physical":
                for field in _MACHINE_PATH_FIELDS:
                    route["options"][field] = f"/{field}"
                route["options"]["rank_paths"] = ["/dev/dpu_rank0"]
    expected = _masked_contract(expected_config, physical=physical)
    if actual != expected:
        raise ValueError(f"evidence configuration drifted from tracked contract: {template.name}")


def _require_summary(summary: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"{label} requires {field}={value!r}, got {summary.get(field)!r}")


def _require_validation(sample: Mapping[str, Any], label: str) -> None:
    validation = sample.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{label} sample lacks validation provenance")
    for field in (
        "policy_reference_applicable",
        "policy_reference_passed",
        "full_precision_threshold_applicable",
        "full_precision_passed",
        "accuracy_qualified",
    ):
        if validation.get(field) is not True:
            raise ValueError(f"{label} sample requires validation.{field}=true")
    limits = CONFORMANCE_POLICIES["float32_1e-5"]
    for field, limit_field in (
        ("max_abs_error", "max_abs_error_max"),
        ("relative_l2_error", "relative_l2_error_max"),
        ("norm_drift", "norm_drift_max"),
    ):
        value = validation.get(field)
        limit = limits[limit_field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > limit
        ):
            raise ValueError(
                f"{label} sample requires validation.{field} <= {limit}"
            )
    if not sample.get("output_sha256"):
        raise ValueError(f"{label} sample lacks an output hash")


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    facts = sample.get("backend_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("physical sample lacks backend facts")
    joined = dict(facts)
    session_id = sample.get("session_instance_id")
    session = sessions.get(str(session_id)) if session_id is not None else None
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        for field, value in terminal.items():
            joined.setdefault(field, value)
    return joined


def _require_physical_samples(
    samples: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]], label: str
) -> Mapping[str, str]:
    sessions_by_id = {str(session["session_instance_id"]): session for session in sessions}
    if len(sessions_by_id) != len(sessions):
        raise ValueError(f"{label} physical session IDs must be unique")
    binary_hashes: set[tuple[str, ...]] = set()
    for session in sessions:
        if any(
            session.get(field) is not True
            for field in ("release_attempted", "release_succeeded", "release_verified")
        ):
            raise ValueError(f"{label} physical session was not fully released")
        terminal = session.get("terminal_backend_facts")
        if not isinstance(terminal, Mapping):
            raise ValueError(f"{label} physical session lacks terminal backend facts")
        values = tuple(terminal.get(field) for field in _BINARY_HASH_FIELDS.values())
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        ):
            raise ValueError(f"{label} physical session lacks valid T1 binary hashes")
        binary_hashes.add(values)
    if len(binary_hashes) != 1:
        raise ValueError(f"{label} physical sessions require one consistent T1 binary triple")
    for sample in samples:
        _require_validation(sample, label)
        numeric = sample.get("numeric_facts")
        if not isinstance(numeric, Mapping) or numeric.get("numeric_policy") != "split_complex_float32_v1":
            raise ValueError(f"{label} requires float32 samples")
        identities = sample.get("identities")
        if not isinstance(identities, Mapping) or any(
            identities.get(field) is None
            for field in ("problem_id", "tensor_network_structure_id", "logical_plan_id", "physical_plan_id", "executable_id", "environment_id", "validation_policy_id")
        ):
            raise ValueError(f"{label} physical sample lacks full identity provenance")
        facts = _joined_facts(sample, sessions_by_id)
        for field, expected in _PHYSICAL_FACTS.items():
            if facts.get(field) != expected:
                raise ValueError(f"{label} physical sample requires {field}={expected!r}")
    values = next(iter(binary_hashes))
    return dict(zip(_BINARY_HASH_FIELDS.values(), values, strict=True))


def _require_powersave_environment(
    manifest: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> None:
    configuration = manifest.get("configuration")
    environment = configuration.get("environment") if isinstance(configuration, Mapping) else None
    if not isinstance(environment, Mapping):
        raise ValueError("baseline diagnostic evidence lacks environment provenance")
    preflight = environment.get("machine_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("baseline diagnostic evidence requires machine preflight facts")
    for label, facts, affinity_field in (
        ("environment", environment, "affinity"),
        ("machine preflight", preflight, "observed_affinity"),
    ):
        if facts.get(affinity_field) != _POWERSAVE_AFFINITY:
            raise ValueError(f"baseline diagnostic requires {label} affinity [0]")
        if facts.get("selected_cpu_ids") != _POWERSAVE_AFFINITY:
            raise ValueError(f"baseline diagnostic requires {label} selected CPU [0]")
        if facts.get("observed_cpu_governors") != _POWERSAVE_GOVERNORS:
            raise ValueError(f"baseline diagnostic requires {label} CPU 0 governor powersave")
    if environment.get("thread_environment") != _THREAD_ENVIRONMENT:
        raise ValueError("baseline diagnostic requires single-thread CPU environment")
    if any(sample.get("observed_affinity") != _POWERSAVE_AFFINITY for sample in samples):
        raise ValueError("baseline diagnostic requires observed affinity [0] for every sample")


def _raw_mad(values: Sequence[float]) -> float:
    center = float(median(values))
    return float(median(abs(value - center) for value in values))


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    return sorted(values)[int(quantile * (len(values) - 1))]


def _paired_ratio_bootstrap(
    pairs: Sequence[tuple[float, float]], *, seed: int
) -> tuple[float, float]:
    if not pairs or any(numpy_s <= 0.0 or upmem_s <= 0.0 for numpy_s, upmem_s in pairs):
        raise ValueError("baseline diagnostic requires positive paired route timings")
    generator = random.Random(seed)
    population_size = len(pairs)
    ratios = sorted(
        float(median(pair[0] for pair in selected) / median(pair[1] for pair in selected))
        for selected in (
            [pairs[generator.randrange(population_size)] for _ in range(population_size)]
            for _ in range(_BASELINE_BOOTSTRAP_RESAMPLES)
        )
    )
    tail = (1.0 - _BASELINE_BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return _percentile(ratios, tail), _percentile(ratios, 1.0 - tail)


def _diagnostic_statistics(
    samples: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    measurements = [sample for sample in samples if sample["attempt_kind"] == "measurement"]
    route_values: dict[str, list[float]] = {"numpy_same_dag": [], "upmem_float32_1dpu_t1": []}
    pairs: dict[int, dict[str, float]] = {}
    for sample in measurements:
        route_id = sample["route_id"]
        measurement = sample.get("measurement")
        if route_id not in route_values or not isinstance(measurement, Mapping):
            raise ValueError("baseline diagnostic measurement route contract drift")
        value = measurement.get("total_wall_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
            raise ValueError("baseline diagnostic requires positive finite total-wall timings")
        route_values[route_id].append(float(value))
        block_id = sample.get("block_id")
        if isinstance(block_id, bool) or not isinstance(block_id, int):
            raise ValueError("baseline diagnostic measurement block identity is invalid")
        pairs.setdefault(block_id, {})[route_id] = float(value)
    if set(pairs) != set(range(2, 32)) or any(set(pair) != set(route_values) for pair in pairs.values()):
        raise ValueError("baseline diagnostic requires complete paired measurement blocks")
    if any(len(values) != 30 for values in route_values.values()):
        raise ValueError("baseline diagnostic requires 30 measurements per route")
    route_statistics = {
        route_id: {
            "measurement_count": len(values),
            "median_total_wall_s": float(median(values)),
            "raw_mad_total_wall_s": _raw_mad(values),
        }
        for route_id, values in route_values.items()
    }
    paired = [
        (pairs[block_id]["numpy_same_dag"], pairs[block_id]["upmem_float32_1dpu_t1"])
        for block_id in sorted(pairs)
    ]
    payload = {
        "base_seed": _experiment_configuration(manifest)["collection"]["base_seed"],
        "block_ids": sorted(pairs),
        "comparison": "same_dag_numpy_to_sequential_upmem_ratio_control_v1",
    }
    seed = int.from_bytes(
        hashlib.sha256(
            b"quantum_bench.sequential_baseline_diagnostic_bootstrap.v1\0"
            + canonical_json(payload).encode("ascii")
        ).digest()[:8],
        "big",
    )
    interval = _paired_ratio_bootstrap(paired, seed=seed)
    return {
        "measurement_only": True,
        "summary": "median_raw_mad_v1",
        "paired_ratio_method": "block_paired_median_numpy_to_upmem_ratio_bootstrap_v1",
        "confidence_interval": "percentile_bootstrap_95_v1",
        "confidence_level": _BASELINE_BOOTSTRAP_CONFIDENCE_LEVEL,
        "resample_count": _BASELINE_BOOTSTRAP_RESAMPLES,
        "routes": route_statistics,
        "numpy_to_upmem_control_ratio": {
            "point_estimate": float(
                route_statistics["numpy_same_dag"]["median_total_wall_s"]
                / route_statistics["upmem_float32_1dpu_t1"]["median_total_wall_s"]
            ),
            "confidence_interval_95": list(interval),
        },
    }


def _load_evidence(
    path: Path, *, commit: str, template: Path, physical: bool, label: str
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    manifest, samples, sessions = load_artifacts(path)
    summary = verify_artifacts(path)
    if manifest.get("source_commit") != commit or manifest.get("source_worktree_dirty") is not False:
        raise ValueError(f"{label} evidence must bind to exact current clean source {commit}")
    _require_contract(manifest, template, physical=physical)
    return manifest, samples, sessions, summary


def _expected_conformance_fixture(
    fixture_id: str,
) -> tuple[set[str], dict[str, str], Mapping[str, Any] | None]:
    oracles = {"direct_quimb_circuit", "thesis_dag_complex128"}
    comparisons = {
        "thesis_dag_complex128_vs_direct_quimb": "complex128_1e-12"
    }
    if fixture_id in {
        "basis_order_2q",
        "Bell2",
        "GHZ5",
        "QuEST-compatible QRNG3",
        "QuEST-compatible BV5",
    }:
        oracles.add("analytic")
        comparisons.update(
            {
                "direct_quimb_vs_analytic": "complex128_1e-12",
                "thesis_dag_complex128_vs_analytic": "complex128_1e-12",
            }
        )
    if fixture_id in {"QuEST-compatible QRNG3", "QuEST-compatible BV5"}:
        oracles.add("quest_cpu")
        comparisons.update(
            {
                "quest_cpu_vs_direct_quimb": "complex128_1e-12",
                "quest_cpu_vs_thesis_dag_complex128": "complex128_1e-12",
            }
        )
    if fixture_id in {"Stress18", "sliced Stress4"}:
        oracles.add("thesis_dag_float32")
        comparisons["thesis_dag_float32_vs_direct_quimb"] = "float32_1e-5"
    slicing = None
    if fixture_id == "sliced Stress4":
        slicing = {
            "node_id": "contract_24",
            "minimum_slice_count": 4,
            "sdk_simulator_coverage": (
                "tests/test_cli_report.py::"
                "test_sliced_conformance_retains_strict_sdk_simulator_coverage"
            ),
        }
    return oracles, comparisons, slicing


def _require_conformance_metric(
    comparison: Mapping[str, Any], field: str, maximum: float | None
) -> None:
    value = comparison.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"sequential conformance {field} must be finite and nonnegative")
    if maximum is not None and value > maximum:
        raise ValueError(f"sequential conformance {field} exceeds its policy gate")


def _inspect_conformance(path: Path, *, commit: str) -> Mapping[str, Any]:
    artifact = _read_json(path)
    fixtures = artifact.get("fixtures")
    if artifact.get("schema_version") != CONFORMANCE_SCHEMA or artifact.get("passed") is not True:
        raise ValueError("sequential conformance requires passed schema v1 evidence")
    if (
        artifact.get("source_commit") != commit
        or artifact.get("source_worktree_dirty") is not False
    ):
        raise ValueError(f"sequential conformance must bind to exact current clean source {commit}")
    if (
        artifact.get("execution_class") != "software_only"
        or artifact.get("phase_aligned_metric_is_diagnostic_only") is not True
        or artifact.get("policies") != CONFORMANCE_POLICIES
    ):
        raise ValueError("sequential conformance top-level policy contract drift")
    if not isinstance(fixtures, list) or len(fixtures) != 8:
        raise ValueError("sequential conformance requires exactly eight fixtures")
    ids = [fixture.get("fixture_id") for fixture in fixtures if isinstance(fixture, Mapping)]
    if len(ids) != 8 or set(ids) != REQUIRED_FIXTURES or len(set(ids)) != 8:
        raise ValueError("sequential conformance fixture identity drift")
    for fixture in fixtures:
        if fixture.get("passed") is not True:
            raise ValueError("every sequential conformance fixture must pass")
        fixture_id = fixture["fixture_id"]
        expected_oracles, expected_comparisons, expected_slicing = (
            _expected_conformance_fixture(fixture_id)
        )
        oracles = fixture.get("oracles")
        if (
            not isinstance(oracles, list)
            or len(oracles) != len(expected_oracles)
            or set(oracles) != expected_oracles
        ):
            raise ValueError(f"sequential conformance oracle drift for {fixture_id}")
        if fixture.get("slicing") != expected_slicing:
            raise ValueError(f"sequential conformance slicing drift for {fixture_id}")
        comparisons = fixture.get("comparisons")
        if not isinstance(comparisons, list) or len(comparisons) != len(
            expected_comparisons
        ):
            raise ValueError(f"sequential conformance comparison drift for {fixture_id}")
        observed: dict[str, Mapping[str, Any]] = {}
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise ValueError(f"sequential conformance comparison drift for {fixture_id}")
            name = comparison.get("comparison")
            if not isinstance(name, str) or name in observed:
                raise ValueError(f"sequential conformance comparison drift for {fixture_id}")
            observed[name] = comparison
        if set(observed) != set(expected_comparisons):
            raise ValueError(f"sequential conformance comparison drift for {fixture_id}")
        for name, comparison in observed.items():
            policy = expected_comparisons[name]
            if (
                comparison.get("policy") != policy
                or comparison.get("raw_phase_sensitive_allclose") is not True
                or comparison.get("passed") is not True
            ):
                raise ValueError(f"sequential conformance comparison failed for {fixture_id}")
            limits = CONFORMANCE_POLICIES[policy]
            for field in ("max_abs_error", "relative_l2_error", "norm_drift"):
                _require_conformance_metric(
                    comparison, field, limits[f"{field}_max"]
                )
            _require_conformance_metric(
                comparison, "phase_aligned_max_abs_error", None
            )
    return {
        "schema_version": CONFORMANCE_SCHEMA,
        "source_commit": commit,
        "fixture_count": 8,
        "passed": True,
        "sha256": _sha256(path),
    }


def _evidence_record(path: Path, manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "artifact_sha256": _tree_sha256(path),
        "manifest_sha256": _sha256(path / "manifest.json"),
        "run_id": manifest["run_id"],
        "experiment_id": manifest["experiment_id"],
        "source_commit": manifest["source_commit"],
        "sample_count": summary["sample_count"],
        "session_count": summary["session_count"],
        "timing_scopes": summary["timing_scopes"],
    }


def inspect_baseline(
    *, conformance: Path, correctness: Path, performance: Path, external_context: Path
) -> Mapping[str, Any]:
    commit = _require_clean_source()
    conformance_record = _inspect_conformance(conformance, commit=commit)
    correct_manifest, correct_samples, correct_sessions, correct_summary = _load_evidence(
        correctness, commit=commit, template=CORRECTNESS_TEMPLATE, physical=True, label="correctness"
    )
    _require_summary(
        correct_summary,
        {"status": "completed", "sample_count": 3, "session_count": 3, "success_count": 3, "failed_count": 0, "unsupported_count": 0, "timing_scopes": ["steady_execution_v1"]},
        "correctness evidence",
    )
    expected_correct_routes = {
        ("bell2", "unsliced", "upmem_float32_1dpu_t1"),
        ("stress4", "unsliced", "upmem_float32_1dpu_t1"),
        ("stress4", "sliced", "upmem_float32_1dpu_t1"),
    }
    if {(s["case_id"], s["plan_id"], s["route_id"]) for s in correct_samples} != expected_correct_routes:
        raise ValueError("correctness evidence requires Bell2 unsliced and Stress4 unsliced/sliced")
    correct_binary_hashes = _require_physical_samples(
        correct_samples, correct_sessions, "correctness"
    )

    perf_manifest, perf_samples, perf_sessions, perf_summary = _load_evidence(
        performance, commit=commit, template=PERFORMANCE_TEMPLATE, physical=True, label="performance"
    )
    _require_summary(
        perf_summary,
        {"status": "completed", "sample_count": 64, "session_count": 32, "success_count": 64, "failed_count": 0, "unsupported_count": 0, "timing_scopes": ["steady_execution_v1"]},
        "performance evidence",
    )
    route_ids = {sample["route_id"] for sample in perf_samples}
    if route_ids != {"numpy_same_dag", "upmem_float32_1dpu_t1"} or {sample["case_id"] for sample in perf_samples} != {"stress18"}:
        raise ValueError("performance evidence requires Stress18 and the exact paired routes")
    blocks: dict[int, list[Mapping[str, Any]]] = {}
    for sample in perf_samples:
        blocks.setdefault(int(sample["block_id"]), []).append(sample)
        _require_validation(sample, "performance")
        if sample["measurement"]["scope_id"] != "steady_execution_v1":
            raise ValueError("performance evidence requires steady execution timing")
    if set(blocks) != set(range(32)) or any(
        {row["route_id"] for row in rows} != route_ids or len(rows) != 2
        for rows in blocks.values()
    ):
        raise ValueError("performance evidence requires complete 2+30 paired blocks")
    if sum(row["attempt_kind"] == "warmup" for row in perf_samples) != 4 or sum(
        row["attempt_kind"] == "measurement" for row in perf_samples
    ) != 60:
        raise ValueError("performance evidence requires two warmup and 30 measured paired blocks")
    physical_samples = [row for row in perf_samples if row["route_id"] == "upmem_float32_1dpu_t1"]
    perf_binary_hashes = _require_physical_samples(
        physical_samples, perf_sessions, "performance"
    )
    if perf_binary_hashes != correct_binary_hashes:
        raise ValueError("correctness and performance evidence used different T1 binaries")
    if _experiment_configuration(perf_manifest)["collection"]["claim_policy"] != "diagnostic_v1":
        raise ValueError("baseline characterization requires diagnostic_v1")
    _require_powersave_environment(perf_manifest, perf_samples)
    diagnostic_statistics = _diagnostic_statistics(perf_samples, perf_manifest)

    external_manifest, external_samples, external_sessions, external_summary = _load_evidence(
        external_context, commit=commit, template=EXTERNAL_TEMPLATE, physical=False, label="external context"
    )
    _require_summary(
        external_summary,
        {"status": "completed", "sample_count": 12, "session_count": 0, "success_count": 12, "failed_count": 0, "unsupported_count": 0, "timing_scopes": ["simulation_end_to_end_v1"]},
        "external context evidence",
    )
    if external_sessions or {row["case_id"] for row in external_samples} != {"stress18"} or {row["route_id"] for row in external_samples} != {"quimb_greedy", "quimb_cotengra_path"}:
        raise ValueError("external context evidence requires only the two Stress18 TN routes")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "qualified",
        "source_commit": commit,
        "t1_binary_hashes": correct_binary_hashes,
        "artifact_statistics": "separate_no_cross_artifact_statistics_v1",
        "claim_eligible": False,
        "claim_ineligibility_reason": "powersave_conditioned_diagnostic_v1",
        "powersave_diagnostic_statistics": diagnostic_statistics,
        "inputs": {
            "conformance": conformance_record,
            "physical_correctness": _evidence_record(correctness, correct_manifest, correct_summary),
            "physical_performance": _evidence_record(performance, perf_manifest, perf_summary),
            "external_tn_context": _evidence_record(external_context, external_manifest, external_summary),
        },
    }


def inspect_to_file(
    *, conformance: Path, correctness: Path, performance: Path, external_context: Path, output: Path
) -> Mapping[str, Any]:
    if output.exists():
        raise ValueError(f"inspection summary must be absent: {output}")
    summary = inspect_baseline(
        conformance=conformance,
        correctness=correctness,
        performance=performance,
        external_context=external_context,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, summary)
    return summary


def _require_exact_files(directory: Path, names: set[str], label: str) -> None:
    actual = {path.relative_to(directory).as_posix() for path in _files(directory)}
    if actual != names:
        raise ValueError(f"{label} files are not the closed canonical set")


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"bundle input must be a regular directory tree: {source}")
    shutil.copytree(source, target)


def _config_binaries(path: Path, expected_experiment_id: str) -> Mapping[str, Path]:
    config = load_experiment_config(path)
    if config["experiment_id"] != expected_experiment_id:
        raise ValueError(f"resolved config does not bind to inspected evidence: {path}")
    routes = [route for route in config["routes"].values() if route["executor"] == "upmem_physical"]
    if len(routes) != 1:
        raise ValueError("resolved config must contain exactly one physical route")
    return {field: Path(routes[0]["options"][field]) for field in _BINARY_FIELDS}


def _validate_report(path: Path, record: Mapping[str, Any], label: str) -> None:
    report = _read_json(path / "report.json")
    if (
        report.get("schema_version") != "evidence_report_v5"
        or report.get("status") != "completed"
        or report.get("run_id") != record["run_id"]
        or report.get("experiment_id") != record["experiment_id"]
    ):
        raise ValueError(f"{label} report does not bind to inspected evidence")


def _write_sha256s(root: Path) -> None:
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in _files(root)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


def verify_internal_hashes(root: Path) -> None:
    checksum = root / "SHA256SUMS"
    lines = checksum.read_text(encoding="ascii").splitlines()
    expected_names = [path.relative_to(root).as_posix() for path in _files(root) if path != checksum]
    observed_names: list[str] = []
    for line in lines:
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError("invalid internal checksum line")
        expected, name = fields[0], fields[1].lstrip("*")
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()) or _sha256(path) != expected:
            raise ValueError(f"internal checksum mismatch: {name}")
        observed_names.append(name)
    if observed_names != sorted(expected_names) or len(observed_names) != len(set(observed_names)):
        raise ValueError("internal checksums do not cover the closed bundle file set")


def _verify_archive_hashes(archive: Path, root_name: str) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers() if member.isfile()}
        checksum_name = f"{root_name}/SHA256SUMS"
        checksum_member = members.get(checksum_name)
        if checksum_member is None:
            raise ValueError("archive lacks internal SHA256SUMS")
        checksum_stream = bundle.extractfile(checksum_member)
        if checksum_stream is None:
            raise ValueError("archive checksum record is unreadable")
        lines = checksum_stream.read().decode("ascii").splitlines()
        observed: list[str] = []
        for line in lines:
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError("invalid archived checksum line")
            expected, name = fields[0], fields[1].lstrip("*")
            member = members.get(f"{root_name}/{name}")
            if member is None:
                raise ValueError(f"archived checksum target is absent: {name}")
            stream = bundle.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected:
                raise ValueError(f"archived checksum mismatch: {name}")
            observed.append(name)
        archived_files = sorted(
            name.removeprefix(f"{root_name}/")
            for name in members
            if name != checksum_name
        )
        if observed != archived_files or len(observed) != len(set(observed)):
            raise ValueError("archived checksums do not cover the closed file set")


def bundle_baseline(
    *,
    summary_path: Path,
    conformance: Path,
    correctness: Path,
    performance: Path,
    external_context: Path,
    correctness_config: Path,
    performance_config: Path,
    correctness_report: Path,
    performance_report: Path,
    external_context_report: Path,
    output: Path,
) -> Mapping[str, Any]:
    archive = output.with_name(output.name + ".tar.gz")
    outer = archive.with_name(archive.name + ".sha256")
    for path in (output, archive, outer):
        if path.exists():
            raise ValueError(f"bundle output must be absent: {path}")
    inspected = inspect_baseline(
        conformance=conformance,
        correctness=correctness,
        performance=performance,
        external_context=external_context,
    )
    if _plain(_read_json(summary_path)) != _plain(inspected):
        raise ValueError("inspection summary does not match the supplied artifacts")
    input_records = inspected["inputs"]
    correct_binaries = _config_binaries(correctness_config, input_records["physical_correctness"]["experiment_id"])
    perf_binaries = _config_binaries(performance_config, input_records["physical_performance"]["experiment_id"])
    recorded_binary_hashes = inspected["t1_binary_hashes"]
    for label, binaries in (
        ("correctness", correct_binaries),
        ("performance", perf_binaries),
    ):
        for field, path in binaries.items():
            if not path.is_file():
                raise ValueError(f"T1 {field} is unavailable: {path}")
            if _sha256(path) != recorded_binary_hashes[_BINARY_HASH_FIELDS[field]]:
                raise ValueError(
                    f"{label} T1 {field} does not match its execution-time hash"
                )
    reports = {
        "physical-correctness": (correctness_report, input_records["physical_correctness"]),
        "physical-performance": (performance_report, input_records["physical_performance"]),
        "external-tn-context": (external_context_report, input_records["external_tn_context"]),
    }
    for label, (path, record) in reports.items():
        _validate_report(path, record, label)
    for path, label in ((correctness, "correctness"), (performance, "performance"), (external_context, "external context")):
        _require_exact_files(path, {"manifest.json", "samples.jsonl", "sessions.jsonl"}, label)

    output.mkdir(parents=True)
    shutil.copy2(summary_path, output / "baseline-summary.json")
    configs = output / "configs"
    configs.mkdir()
    shutil.copy2(correctness_config, configs / CORRECTNESS_CONFIG_NAME)
    shutil.copy2(performance_config, configs / PERFORMANCE_CONFIG_NAME)
    conformance_dir = output / "conformance"
    conformance_dir.mkdir()
    shutil.copy2(conformance, conformance_dir / "sequential-conformance.json")
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    _copy_tree(correctness, evidence_dir / "physical-correctness")
    _copy_tree(performance, evidence_dir / "physical-performance")
    _copy_tree(external_context, evidence_dir / "external-tn-context")
    reports_dir = output / "reports"
    reports_dir.mkdir()
    for label, (path, _record) in reports.items():
        _copy_tree(path, reports_dir / label)
    provenance = output / "provenance"
    provenance.mkdir()
    _write_json(
        provenance / "source.json",
        {"source_commit": inspected["source_commit"], "source_worktree_dirty": False},
    )
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "opt_einsum", "quimb", "cotengra", "PyYAML")}
    _write_json(
        provenance / "versions.json",
        {"python": platform.python_version(), "platform": platform.platform(), "packages": versions},
    )
    _write_json(
        provenance / "t1-binary-hashes.json",
        recorded_binary_hashes,
    )
    _write_json(
        provenance / "input-hashes.json",
        {name: record.get("sha256", record.get("artifact_sha256")) for name, record in input_records.items()},
    )
    _write_sha256s(output)
    verify_internal_hashes(output)
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname=output.name, recursive=True)
    _verify_archive_hashes(archive, output.name)
    archive_sha = _sha256(archive)
    outer.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    if _sha256(archive) != archive_sha:
        raise ValueError("outer bundle checksum verification failed")
    return {
        "status": "bundled",
        "bundle": str(output),
        "archive": str(archive),
        "outer_checksum": str(outer),
        "archive_sha256": archive_sha,
    }


def _add_inspection_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conformance", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--external-context", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--rank-path", required=True)
    prepare.add_argument("--correctness-session-root", required=True)
    prepare.add_argument("--performance-session-root", required=True)
    prepare.add_argument("--expected-cpus", required=True)
    prepare.add_argument("--host-binary")
    prepare.add_argument("--dpu-binary")
    inspect = commands.add_parser("inspect")
    _add_inspection_inputs(inspect)
    inspect.add_argument("--output", type=Path, required=True)
    bundle = commands.add_parser("bundle")
    _add_inspection_inputs(bundle)
    bundle.add_argument("--summary", type=Path, required=True)
    bundle.add_argument("--correctness-config", type=Path, required=True)
    bundle.add_argument("--performance-config", type=Path, required=True)
    bundle.add_argument("--correctness-report", type=Path, required=True)
    bundle.add_argument("--performance-report", type=Path, required=True)
    bundle.add_argument("--external-context-report", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_configs(
                output_dir=args.output_dir.resolve(),
                rank_path=args.rank_path,
                correctness_session_root=args.correctness_session_root,
                performance_session_root=args.performance_session_root,
                expected_cpus=_parse_cpus(args.expected_cpus),
                host_binary=args.host_binary,
                dpu_binary=args.dpu_binary,
            )
        elif args.command == "inspect":
            result = inspect_to_file(
                conformance=args.conformance.resolve(),
                correctness=args.correctness.resolve(),
                performance=args.performance.resolve(),
                external_context=args.external_context.resolve(),
                output=args.output.resolve(),
            )
        else:
            result = bundle_baseline(
                summary_path=args.summary.resolve(),
                conformance=args.conformance.resolve(),
                correctness=args.correctness.resolve(),
                performance=args.performance.resolve(),
                external_context=args.external_context.resolve(),
                correctness_config=args.correctness_config.resolve(),
                performance_config=args.performance_config.resolve(),
                correctness_report=args.correctness_report.resolve(),
                performance_report=args.performance_report.resolve(),
                external_context_report=args.external_context_report.resolve(),
                output=args.output.resolve(),
            )
    except (OSError, ValueError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
