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


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qasm = path.parent / "circuits" / "example.qasm"
    qasm.parent.mkdir(parents=True, exist_ok=True)
    qasm.write_text(
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n', encoding="utf-8"
    )
    path.write_text(text, encoding="utf-8")


def test_loader_normalizes_paths_and_returns_recursive_immutable_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "configs" / "study.yml"
    _write_config(path, _config())

    config = load_experiment_config(path)

    assert config["cases"]["qasm_case"]["circuit"]["path"] == str(
        (path.parent / "circuits/example.qasm").resolve()
    )
    assert config["routes"]["quest"]["options"]["runner"] == str(
        (path.parent / "bin/quest_runner").resolve()
    )
    assert len(config["experiment_id"]) == 64
    with pytest.raises(TypeError):
        config["defaults"]["warmups"] = 1
    with pytest.raises(TypeError):
        config["matrix"][0]["route_ids"] += ("quest",)


def test_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yml"
    _write_config(
        path,
        _config().replace(
            "experiment_id: focused", "experiment_id: focused\nexperiment_id: duplicate"
        ),
    )
    with pytest.raises(ValueError, match="duplicate YAML key"):
        load_experiment_config(path)


def test_loader_rejects_unknown_fields_and_invalid_route_unions(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yml"
    _write_config(
        unknown,
        _config().replace(
            "experiment_id: focused", "experiment_id: focused\nunknown: true"
        ),
    )
    with pytest.raises(ValueError, match="fields must be exact"):
        load_experiment_config(unknown)

    mixed = tmp_path / "mixed.yml"
    _write_config(
        mixed,
        _config().replace("route_ids: [numpy]", "route_ids: [numpy, quest]"),
    )
    with pytest.raises(ValueError, match="cannot mix"):
        load_experiment_config(mixed)


def test_loader_rejects_invalid_matrix_references_and_plan_nullability(
    tmp_path: Path,
) -> None:
    unknown_case = tmp_path / "unknown_case.yml"
    _write_config(
        unknown_case,
        _config().replace("case_id: qasm_case", "case_id: missing", 1),
    )
    with pytest.raises(ValueError, match="unknown case_id"):
        load_experiment_config(unknown_case)

    missing_plan = tmp_path / "missing_plan.yml"
    _write_config(
        missing_plan,
        _config().replace("plan_id: p1", "plan_id: null", 1),
    )
    with pytest.raises(ValueError, match="plan_id is incompatible"):
        load_experiment_config(missing_plan)


def test_loader_enforces_circuit_name_path_union_and_exact_route_options(
    tmp_path: Path,
) -> None:
    bad_circuit = tmp_path / "bad_circuit.yml"
    _write_config(
        bad_circuit,
        _config().replace(
            "name: null\n      path: circuits/example.qasm",
            "name: example\n      path: circuits/example.qasm",
        ),
    )
    with pytest.raises(ValueError, match="name must be null"):
        load_experiment_config(bad_circuit)

    bad_options = tmp_path / "bad_options.yml"
    _write_config(
        bad_options,
        _config().replace("options: {}", "options:\n      extra: true", 1),
    )
    with pytest.raises(ValueError, match="fields must be exact"):
        load_experiment_config(bad_options)


def test_loader_rejects_duplicate_case_route_occurrences(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate_matrix.yml"
    text = _config().replace(
        "  - case_id: builtin_case",
        "  - case_id: qasm_case\n    plan_id: p1\n    route_ids: [numpy]\n"
        "  - case_id: builtin_case",
    )
    _write_config(duplicate, text)

    with pytest.raises(ValueError, match="pair once"):
        load_experiment_config(duplicate)


def test_loader_allows_planless_baseline_only_configuration(tmp_path: Path) -> None:
    path = tmp_path / "baseline.yml"
    text = _config().replace(
        "plans:\n  p1:\n    planner:\n      engine: opt_einsum\n"
        "      mode: greedy\n      max_repeats: 1\n      seed: 0\n"
        "    slicing: null",
        "plans: {}",
    )
    text = text.replace(
        "  - case_id: qasm_case\n    plan_id: p1\n    route_ids: [numpy]\n",
        "",
    )
    _write_config(path, text)

    config = load_experiment_config(path)

    assert dict(config["plans"]) == {}


def test_experiment_identity_changes_with_repetition_policy(tmp_path: Path) -> None:
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    _write_config(first, _config())
    _write_config(second, _config().replace("repetitions: 1", "repetitions: 2"))

    assert (
        load_experiment_config(first)["experiment_id"]
        != load_experiment_config(second)["experiment_id"]
    )


def test_loader_rejects_invalid_simulator_topology(tmp_path: Path) -> None:
    path = tmp_path / "simulator.yml"
    text = (
        _config()
        .replace(
            "  quest:\n    executor: quest_cpu\n    numeric_policy: null\n"
            "    options:\n      runner: bin/quest_runner",
            "  quest:\n    executor: upmem_sdk_simulator\n"
            "    numeric_policy: split_complex_float32_v1\n"
            "    options:\n      dpu_count: 2\n      rank_count: 1\n"
            "      tasklets_per_dpu: 1\n      session_root: native\n"
            "      host_binary: native/host\n      dpu_binary: native/dpu\n"
            "      initialization_binary: native/init",
        )
        .replace(
            "plan_id: null\n    route_ids: [quest]",
            "plan_id: p1\n    route_ids: [quest]",
        )
    )
    _write_config(path, text)

    with pytest.raises(ValueError, match="one DPU and one rank"):
        load_experiment_config(path)
