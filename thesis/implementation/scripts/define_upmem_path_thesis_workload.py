#!/usr/bin/env python3
"""Freeze and describe the finite UPMEM path-heuristic thesis workload."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping
import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from quantum_bench.circuits import builtin_circuit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "upmem_path_thesis_workload_v1.json"
DEFAULT_PILOT_CONFIG = ROOT / "configs" / "upmem_path_heuristic_v1.json"
SOURCE_CIRCUITS = ROOT / "src" / "quantum_bench" / "circuits.py"
SOURCE_DESCRIPTORS = ROOT / "scripts" / "characterize_circuit_resources.py"
MANIFEST_FILENAME = "thesis_workload_manifest.json"
CSV_FILENAME = "thesis_workload_characterization.csv"
MANIFEST_SCHEMA_VERSION = "upmem_path_thesis_workload_manifest_v1"
NUMERIC_POLICY = "split_complex_float32_v1"
TRANSPORT = "packed_operation_v1"
TOPOLOGIES = (
    {"topology_id": "1dpu_t8", "dpu_count": 1, "rank_count": 1, "tasklets_per_dpu": 8},
    {"topology_id": "4dpu_t8", "dpu_count": 4, "rank_count": 1, "tasklets_per_dpu": 8},
)
ALLOWED_CLASSIFICATIONS = frozenset({"heuristic_eligible", "correctness_only"})
ALLOWED_EXPERIMENT_SPLITS = frozenset(
    {"training", "validation", "test", "pilot_development", "none"}
)
ALLOWED_CALIBRATION_SPLITS = frozenset({"training", "validation"})
PILOT_SOURCE = "upmem_path_heuristic_v1"
GENERALIZATION_SOURCE = "generalization_v1"
PILOT_STATUS = "pilot_observed"
UNOBSERVED_STATUS = "unobserved_for_path_selection"
DEPENDENCIES = ("numpy", "opt_einsum", "cotengra", "quimb", "PyYAML")

CORRECTNESS_ONLY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "bell_2q": {
        "kind": "builtin",
        "name": "bell_2q",
        "parameters": {},
    },
    "qrng_18q": {
        "kind": "builtin",
        "name": "qrng",
        "parameters": {"n_qubits": 18},
    },
    "bb84_18q": {
        "kind": "builtin",
        "name": "bb84",
        "parameters": {"n_qubits": 18},
    },
}

FAMILY_BY_NAME = {
    "bb84": "bb84",
    "bell_2q": "bell",
    "bv": "bv",
    "edc": "edc",
    "ghz_chain": "ghz_chain",
    "hs": "hs",
    "qrng": "qrng",
    "quantization_stress": "quantization_stress",
    "xor": "xor",
}

CSV_COLUMNS = (
    "circuit_id",
    "classification",
    "split",
    "family",
    "historical_physical_status",
    "candidate_source",
    "definition_source",
    "circuit_kind",
    "circuit_name",
    "circuit_parameters_json",
    "circuit_spec_name",
    "n_qubits",
    "circuit_depth",
    "gate_count",
    "one_qubit_gate_count",
    "two_qubit_gate_count",
    "interaction_edge_count",
    "interaction_density",
    "interaction_max_degree",
    "operations_sha256",
)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or len(value) != 40:
        raise ValueError("cannot determine source SHA")
    return value


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _load_descriptor_function() -> Any:
    """Reuse the established circuit descriptor implementation."""

    spec = importlib.util.spec_from_file_location(
        "_upmem_thesis_workload_circuit_descriptors", SOURCE_DESCRIPTORS
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load descriptor source: {SOURCE_DESCRIPTORS}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._circuit_descriptors


_circuit_descriptors = _load_descriptor_function()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return _load_json(path)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in workload manifest: {type(value).__name__}")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _is_hex_sha(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _canonical_definition(value: Any, field: str) -> dict[str, Any]:
    definition = _required_mapping(value, field)
    if definition.get("kind") != "builtin":
        raise ValueError(f"{field}.kind must be 'builtin'")
    name = _required_string(definition.get("name"), f"{field}.name")
    parameters = definition.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"{field}.parameters must be an object")
    if name not in FAMILY_BY_NAME:
        raise ValueError(f"unsupported builtin circuit family: {name}")
    return {
        "kind": "builtin",
        "name": name,
        "parameters": _plain(parameters),
    }


def _circuit_specs(config: dict[str, Any], pilot_config: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str]]:
    result: dict[str, tuple[dict[str, Any], str]] = {}
    for source_name, source_config in (
        ("pilot_config", pilot_config),
        ("workload_config", config),
    ):
        entries = source_config.get("circuits")
        if not isinstance(entries, list):
            raise ValueError(f"{source_name}.circuits must be a list")
        for item in entries:
            item = _required_mapping(item, f"{source_name}.circuits entry")
            circuit_id = _required_string(item.get("circuit_id"), "circuit_id")
            definition = _canonical_definition(
                item.get("circuit"), f"{source_name}.circuits[{circuit_id}].circuit"
            )
            if circuit_id in result:
                previous_definition, previous_source = result[circuit_id]
                if previous_definition != definition:
                    raise ValueError(
                        f"circuit {circuit_id} has conflicting definitions in "
                        f"{previous_source} and {source_name}"
                    )
                raise ValueError(
                    f"circuit {circuit_id} is declared in both "
                    f"{previous_source} and {source_name}"
                )
            result[circuit_id] = (definition, source_name)
    result.update(
        (circuit_id, (definition, "correctness_only_builtin"))
        for circuit_id, definition in CORRECTNESS_ONLY_DEFINITIONS.items()
        if circuit_id not in result
    )
    return result


def _workload_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = config.get("declared_workload")
    if not isinstance(entries, list) or not entries:
        raise ValueError("declared_workload must be a non-empty list")
    result = []
    seen: set[str] = set()
    for index, item in enumerate(entries):
        item = _required_mapping(item, f"declared_workload[{index}]")
        circuit_id = _required_string(item.get("circuit_id"), "circuit_id")
        if circuit_id in seen:
            raise ValueError(f"duplicate declared workload circuit: {circuit_id}")
        seen.add(circuit_id)
        result.append(dict(item))
    return result


def _validate_contract(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "upmem_path_heuristic_preregistration_v1":
        raise ValueError("unrecognized workload configuration schema")
    if config.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError(f"numeric_policy must be {NUMERIC_POLICY}")
    topologies = config.get("topologies")
    if topologies != [dict(item) for item in TOPOLOGIES]:
        raise ValueError("topologies must be the exact one-rank 1DPU/T8 and 4DPU/T8 contract")
    environment = _required_mapping(config.get("physical_environment"), "physical_environment")
    if environment.get("transport") != TRANSPORT:
        raise ValueError(f"physical_environment.transport must be {TRANSPORT}")
    if environment.get("rank_path") != "/dev/dpu_rank1":
        raise ValueError("physical_environment.rank_path must be /dev/dpu_rank1")
    if environment.get("affinity") != [0]:
        raise ValueError("physical_environment.affinity must pin CPU 0")
    calibration = _required_mapping(config.get("calibration"), "calibration")
    calibration_splits = calibration.get("splits")
    if set(calibration_splits or ()) != ALLOWED_CALIBRATION_SPLITS:
        raise ValueError("calibration.splits must contain training and validation")
    if calibration.get("measurement_blocks") != 3 or calibration.get("warmup_blocks") != 1:
        raise ValueError("calibration collection must be one warmup and three measurements")
    if calibration.get("candidates_per_cell_maximum") != 6:
        raise ValueError("calibration candidate maximum must be six")
    if calibration.get("candidate_roles") != [
        "greedy",
        "minimum_flops",
        "minimum_peak_intermediate",
        "minimum_writes",
        "frozen_v1_selected",
        "feature_diverse",
    ]:
        raise ValueError("calibration candidate roles are not frozen")
    frozen_profile = _required_mapping(
        calibration.get("frozen_v1_profile"), "calibration.frozen_v1_profile"
    )
    if frozen_profile.get("path") != (
        "thesis_results/upmem_path_heuristic_v1/fit/physical_speedup_fit_v1.json"
    ):
        raise ValueError("calibration frozen-v1 profile path is not frozen")
    if frozen_profile.get("sha256") != (
        "cc1e3deb6b5a227b4efe9c84e43679d385cb9b65da76e293ad0d074889cb868a"
    ):
        raise ValueError("calibration frozen-v1 profile hash is not frozen")
    final_policy = _required_mapping(config.get("final_test_policy"), "final_test_policy")
    if final_policy.get("timing_may_change_candidates_or_model") is not False:
        raise ValueError("final-test timing must not change candidates or the model")
    if final_policy.get("warmup_blocks") != 1 or final_policy.get("measurement_blocks") != 5:
        raise ValueError("final-test collection must be one warmup and five measurements")
    if final_policy.get("candidate_roles") != [
        "greedy", "minimum_flops", "frozen_heuristic_selected"
    ]:
        raise ValueError("final-test candidate roles are not frozen")
    if config.get("model_forms") != ["six_term", "grouped"]:
        raise ValueError("model_forms must contain exactly six_term and grouped")
    baseline = config.get("software_baseline_sha")
    if not _is_hex_sha(baseline, 40):
        raise ValueError("software_baseline_sha must be a full commit SHA")
    score = _required_mapping(config.get("score"), "score")
    if score.get("semantic_id") != "upmem_slr_cost_v1":
        raise ValueError("score.semantic_id must be upmem_slr_cost_v1")
    pilot_policy = _required_mapping(config.get("pilot_evidence_policy"), "pilot_evidence_policy")
    if pilot_policy.get("canonical_final_evidence") is not False:
        raise ValueError("pilot evidence cannot be canonical final evidence")
    if pilot_policy.get("mix_into_new_statistical_fit") is not False:
        raise ValueError("pilot evidence cannot be mixed into the new fit")


def _validate_workload_entries(
    config: dict[str, Any],
    pilot_config: dict[str, Any],
    entries: list[dict[str, Any]],
    definitions: dict[str, tuple[dict[str, Any], str]],
) -> None:
    pilot_circuits = pilot_config.get("circuits")
    if not isinstance(pilot_circuits, list):
        raise ValueError("pilot_config.circuits must be a list")
    pilot_ids = {
        _required_string(item.get("circuit_id"), "pilot circuit_id")
        for item in pilot_circuits
    }
    configured_ids = {
        _required_string(item.get("circuit_id"), "configured circuit_id")
        for item in config.get("circuits", [])
    }
    entry_ids = {str(item["circuit_id"]) for item in entries}
    if not configured_ids <= entry_ids:
        raise ValueError("every generated circuit must be declared in the workload")
    if not pilot_ids <= entry_ids:
        raise ValueError("every historical pilot circuit must remain declared")

    eligible = [item for item in entries if item.get("classification") == "heuristic_eligible"]
    if len(eligible) > 12:
        raise ValueError("at most twelve heuristic-eligible circuit instances are allowed")
    if len({item.get("family") for item in eligible}) < 4:
        raise ValueError("at least four eligible circuit families are required")

    new_config_by_id = {
        str(item["circuit_id"]): item for item in config.get("circuits", [])
    }
    for item in entries:
        circuit_id = str(item["circuit_id"])
        classification = item.get("classification")
        split = item.get("split")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise ValueError(f"{circuit_id} has unsupported classification: {classification!r}")
        if split not in ALLOWED_EXPERIMENT_SPLITS:
            raise ValueError(f"{circuit_id} has unsupported split: {split!r}")
        if circuit_id not in definitions:
            raise ValueError(f"no circuit definition is available for {circuit_id}")
        definition, definition_source = definitions[circuit_id]
        expected_family = FAMILY_BY_NAME[definition["name"]]
        if item.get("family") != expected_family:
            raise ValueError(f"{circuit_id} family does not match its builtin definition")

        if circuit_id in pilot_ids and (
            classification != "heuristic_eligible"
            or item.get("candidate_source") != PILOT_SOURCE
            or item.get("historical_physical_status") != PILOT_STATUS
            or split != "pilot_development"
        ):
            raise ValueError(f"historical pilot {circuit_id} must remain pilot_development")

        if classification == "correctness_only":
            if split != "none":
                raise ValueError(f"correctness-only circuit {circuit_id} must use split none")
            if item.get("candidate_source") is not None:
                raise ValueError(f"correctness-only circuit {circuit_id} cannot have candidates")
            if not isinstance(item.get("exclusion_reason"), str) or not item["exclusion_reason"]:
                raise ValueError(f"correctness-only circuit {circuit_id} needs an exclusion reason")
            if item.get("historical_physical_status") is not None:
                raise ValueError(f"correctness-only circuit {circuit_id} cannot claim physical status")
            continue

        if split == "none":
            raise ValueError(f"eligible circuit {circuit_id} cannot use split none")
        status = item.get("historical_physical_status")
        source = item.get("candidate_source")
        if circuit_id in pilot_ids:
            if source != PILOT_SOURCE or status != PILOT_STATUS or split != "pilot_development":
                raise ValueError(f"historical pilot {circuit_id} must remain pilot_development")
            if split == "test":
                raise ValueError(f"historical pilot {circuit_id} cannot be an untouched test")
        else:
            if source != GENERALIZATION_SOURCE or status != UNOBSERVED_STATUS:
                raise ValueError(f"new eligible circuit {circuit_id} must be unobserved generalization data")
            if split not in {"training", "validation", "test"}:
                raise ValueError(f"new eligible circuit {circuit_id} has invalid split")
            configured = new_config_by_id.get(circuit_id)
            if configured is None:
                raise ValueError(f"new eligible circuit {circuit_id} is missing from circuits")
            configured_split = configured.get("split")
            if configured_split != split:
                raise ValueError(f"{circuit_id} declared and configured splits differ")
            configured_definition = _canonical_definition(
                configured.get("circuit"), f"circuits[{circuit_id}].circuit"
            )
            if configured_definition != definition:
                raise ValueError(f"{circuit_id} has inconsistent circuit definitions")
            if definition_source != "workload_config":
                raise ValueError(f"new eligible circuit {circuit_id} must use the workload config")

    final_tests = [
        item
        for item in eligible
        if item.get("split") == "test"
        and item.get("historical_physical_status") == UNOBSERVED_STATUS
    ]
    if len(final_tests) < 2:
        raise ValueError("at least two unobserved eligible final tests are required")


def _operation_record(operation: Any) -> dict[str, Any]:
    return {
        "gate": operation.gate,
        "wires": list(operation.wires),
        "params": list(operation.params),
    }


def _describe_circuit(
    circuit_id: str,
    entry: dict[str, Any],
    definition: dict[str, Any],
    definition_source: str,
) -> dict[str, Any]:
    circuit = builtin_circuit(definition["name"], dict(definition["parameters"]))
    operations = [_operation_record(operation) for operation in circuit.operations]
    descriptors = dict(_circuit_descriptors(circuit))
    gate_counts = descriptors["gate_counts"]
    descriptors["circuit_depth"] = descriptors.pop("true_deterministic_depth")
    descriptors["one_qubit_gate_count"] = sum(
        count for gate, count in gate_counts.items() if gate in {"h", "x", "y", "z", "s", "t", "i", "rz", "ry"}
    )
    descriptors["two_qubit_gate_count"] = sum(
        count for gate, count in gate_counts.items() if gate in {"cx", "cnot", "cz", "swap"}
    )
    operations_sha256 = _sha256_bytes(_canonical_bytes(operations))
    return {
        "circuit_id": circuit_id,
        "classification": entry["classification"],
        "split": entry["split"],
        "family": entry["family"],
        "historical_physical_status": entry.get("historical_physical_status"),
        "candidate_source": entry.get("candidate_source"),
        "exclusion_reason": entry.get("exclusion_reason"),
        "definition_source": definition_source,
        "circuit_definition": definition,
        "circuit_spec": {
            "name": circuit.name,
            "n_qubits": circuit.n_qubits,
            "operations": operations,
            "source": _plain(circuit.source),
        },
        "operations_sha256": operations_sha256,
        "descriptors": descriptors,
    }


def build_manifest(
    config_path: Path = DEFAULT_CONFIG,
    pilot_config_path: Path = DEFAULT_PILOT_CONFIG,
) -> dict[str, Any]:
    """Build the deterministic manifest without executing a planner or backend."""

    config = load_config(config_path)
    pilot_config = load_config(pilot_config_path)
    _validate_contract(config)
    entries = _workload_entries(config)
    definitions = _circuit_specs(config, pilot_config)
    _validate_workload_entries(config, pilot_config, entries, definitions)

    workload = [
        _describe_circuit(
            str(entry["circuit_id"]),
            entry,
            definitions[str(entry["circuit_id"])][0],
            definitions[str(entry["circuit_id"])][1],
        )
        for entry in entries
    ]
    eligible = [item for item in workload if item["classification"] == "heuristic_eligible"]
    final_tests = [item for item in eligible if item["split"] == "test"]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_id": config["study_id"],
        "source_sha": _source_sha(),
        "software_baseline_sha": config["software_baseline_sha"],
        "config_sha256": _sha256_file(config_path),
        "pilot_config_sha256": _sha256_file(pilot_config_path),
        "circuits_sha256": _sha256_file(SOURCE_CIRCUITS),
        "descriptor_source_sha256": _sha256_file(SOURCE_DESCRIPTORS),
        "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "dependency_versions": _dependency_versions(),
        "contract": {
            "calibration": _plain(config["calibration"]),
            "numeric_policy": config["numeric_policy"],
            "transport": config["physical_environment"]["transport"],
            "topologies": [dict(item) for item in TOPOLOGIES],
            "rank_count": 1,
            "timing_selection_frozen": True,
            "physical_execution": False,
        },
        "summary": {
            "declared_instance_count": len(workload),
            "heuristic_eligible_count": len(eligible),
            "eligible_family_count": len({item["family"] for item in eligible}),
            "pilot_development_count": sum(item["split"] == "pilot_development" for item in workload),
            "unobserved_eligible_count": sum(
                item["historical_physical_status"] == UNOBSERVED_STATUS for item in eligible
            ),
            "unobserved_final_test_count": len(final_tests),
            "correctness_only_count": sum(
                item["classification"] == "correctness_only" for item in workload
            ),
        },
        "workload": workload,
    }


def _csv_row(item: dict[str, Any]) -> dict[str, Any]:
    descriptors = item["descriptors"]
    graph = descriptors["interaction_graph"]
    definition = item["circuit_definition"]
    return {
        "circuit_id": item["circuit_id"],
        "classification": item["classification"],
        "split": item["split"],
        "family": item["family"],
        "historical_physical_status": item.get("historical_physical_status"),
        "candidate_source": item.get("candidate_source"),
        "definition_source": item["definition_source"],
        "circuit_kind": definition["kind"],
        "circuit_name": definition["name"],
        "circuit_parameters_json": json.dumps(
            definition["parameters"], sort_keys=True, separators=(",", ":")
        ),
        "circuit_spec_name": item["circuit_spec"]["name"],
        "n_qubits": descriptors["n_qubits"],
        "circuit_depth": descriptors["circuit_depth"],
        "gate_count": descriptors["gate_count"],
        "one_qubit_gate_count": descriptors["one_qubit_gate_count"],
        "two_qubit_gate_count": descriptors["two_qubit_gate_count"],
        "interaction_edge_count": graph["edge_count"],
        "interaction_density": graph["density"],
        "interaction_max_degree": graph["max_degree"],
        "operations_sha256": item["operations_sha256"],
    }


def write_manifest(
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    pilot_config_path: Path = DEFAULT_PILOT_CONFIG,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config_path, pilot_config_path)
    manifest_path = output_dir / MANIFEST_FILENAME
    csv_path = output_dir / CSV_FILENAME
    manifest_path.write_bytes(_canonical_bytes(manifest))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_row(item) for item in manifest["workload"])
    return manifest_path, csv_path


def check_manifest(
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    pilot_config_path: Path = DEFAULT_PILOT_CONFIG,
) -> None:
    expected_manifest, expected_csv = write_to_memory(config_path, pilot_config_path)
    manifest_path = output_dir / MANIFEST_FILENAME
    csv_path = output_dir / CSV_FILENAME
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError("workload manifest differs from deterministic recomputation")
    if csv_path.read_bytes() != expected_csv:
        raise ValueError("workload characterization differs from deterministic recomputation")


def write_to_memory(
    config_path: Path = DEFAULT_CONFIG,
    pilot_config_path: Path = DEFAULT_PILOT_CONFIG,
) -> tuple[bytes, bytes]:
    manifest = build_manifest(config_path, pilot_config_path)
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_csv_row(item) for item in manifest["workload"])
    return _canonical_bytes(manifest), stream.getvalue().encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pilot-config", type=Path, default=DEFAULT_PILOT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_manifest(args.output_dir, args.config, args.pilot_config)
        print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}))
        return
    manifest_path, csv_path = write_manifest(
        args.output_dir, args.config, args.pilot_config
    )
    print(json.dumps({"manifest": str(manifest_path), "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
