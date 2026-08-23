from __future__ import annotations

from pathlib import Path

import pytest

from quantum_bench.experiment import load_experiment_config


def _config() -> str:
    return """\
schema_version: tn_benchmark_v1
experiment_id: focused
defaults:
  warmups: 0
  repetitions: 1
  timeout_s: 2.5
cases:
  qasm_case:
    circuit:
      kind: qasm_file
      name: null
      path: circuits/example.qasm
      parameters: {}
  builtin_case:
    circuit:
      kind: builtin
      name: bell_2q
      path: null
      parameters: {}
plans:
  p1:
    planner:
      engine: opt_einsum
      mode: greedy
      max_repeats: 1
      seed: 0
    slicing: null
routes:
  numpy:
    executor: numpy_dag
    numeric_policy: split_complex_float32_v1
    options: {}
  quest:
    executor: quest_cpu
    numeric_policy: null
    options:
      runner: bin/quest_runner
matrix:
  - case_id: qasm_case
    plan_id: p1
    route_ids: [numpy]
  - case_id: builtin_case
    plan_id: null
    route_ids: [quest]
"""


def test_loader_normalizes_paths_and_returns_recursive_immutable_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "configs" / "study.yml"
    path.parent.mkdir()
    path.write_text(_config(), encoding="utf-8")

    config = load_experiment_config(path)

    assert config["cases"]["qasm_case"]["circuit"]["path"] == str(
        (path.parent / "circuits/example.qasm").resolve()
    )
    assert config["routes"]["quest"]["options"]["runner"] == str(
        (path.parent / "bin/quest_runner").resolve()
    )
    with pytest.raises(TypeError):
        config["defaults"]["warmups"] = 1
    with pytest.raises(TypeError):
        config["matrix"][0]["route_ids"] += ("quest",)


def test_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text(
        _config().replace(
            "experiment_id: focused", "experiment_id: focused\nexperiment_id: duplicate"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_experiment_config(path)


def test_loader_rejects_unknown_fields_and_invalid_route_unions(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yml"
    unknown.write_text(
        _config().replace(
            "experiment_id: focused", "experiment_id: focused\nunknown: true"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields must be exact"):
        load_experiment_config(unknown)

    mixed = tmp_path / "mixed.yml"
    mixed.write_text(
        _config().replace("route_ids: [numpy]", "route_ids: [numpy, quest]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot mix"):
        load_experiment_config(mixed)


def test_loader_rejects_invalid_matrix_references_and_plan_nullability(
    tmp_path: Path,
) -> None:
    unknown_case = tmp_path / "unknown_case.yml"
    unknown_case.write_text(
        _config().replace("case_id: qasm_case", "case_id: missing", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown case_id"):
        load_experiment_config(unknown_case)

    missing_plan = tmp_path / "missing_plan.yml"
    missing_plan.write_text(
        _config().replace("plan_id: p1", "plan_id: null", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="plan_id is incompatible"):
        load_experiment_config(missing_plan)


def test_loader_enforces_circuit_name_path_union_and_exact_route_options(
    tmp_path: Path,
) -> None:
    bad_circuit = tmp_path / "bad_circuit.yml"
    bad_circuit.write_text(
        _config().replace(
            "name: null\n      path: circuits/example.qasm",
            "name: example\n      path: circuits/example.qasm",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name must be null"):
        load_experiment_config(bad_circuit)

    bad_options = tmp_path / "bad_options.yml"
    bad_options.write_text(
        _config().replace("options: {}", "options:\n      extra: true", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields must be exact"):
        load_experiment_config(bad_options)
