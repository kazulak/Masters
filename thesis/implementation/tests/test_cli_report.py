from __future__ import annotations

from dataclasses import fields
import csv
import os
from pathlib import Path
import json
import shutil
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

import quantum_bench.baselines as baselines
import quantum_bench.cli as cli
from quantum_bench.evidence import (
    append_sample,
    append_session,
    canonical_json,
    collection_policy_id,
    environment_id,
    finalize_artifacts,
    load_artifacts,
    sample_id,
    validation_policy_id,
    write_manifest,
)
from quantum_bench.experiment import load_experiment_config
from quantum_bench.report import (
    _assert_unique_plot_points,
    _plot_grouped_bars,
    _point,
    report_artifacts,
    verify_artifacts,
)
from quantum_bench.results import ExecutionSample, Measurement
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemStage,
    UpmemTopology,
    UpmemWorkUnit,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _command(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src"}
    return subprocess.run(
        list(args),
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_make() -> None:
    if shutil.which("make") is None:
        pytest.skip("make is unavailable")


def test_make_public_targets_are_exact_and_pidcomm_is_private() -> None:
    _require_make()
    help_result = _command("make", "help")
    assert help_result.returncode == 0
    assert "pidcomm" not in help_result.stdout.lower()
    assert "make setup" in help_result.stdout
    assert "make verify" in help_result.stdout

    dry_run = _command("make", "-n", "pidcomm-check")
    assert dry_run.returncode != 0

    database = _command("make", "-pn", "help")
    assert database.returncode == 0
    public_line = next(
        line for line in database.stdout.splitlines()
        if line.startswith("PUBLIC_TARGETS := ")
    )
    assert public_line.removeprefix("PUBLIC_TARGETS := ").split() == [
        "help",
        "setup",
        "doctor",
        "test",
        "build-quest-cpu",
        "build-upmem-runtime",
        "plan",
        "run",
        "report",
        "verify",
        "qualify",
        "sequential-conformance",
        "sequential-baseline",
        "clean-generated",
    ]


def test_physical_smoke_configuration_is_the_safe_default() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_physical_smoke.yml"
    )

    assert config["schema_version"] == "tn_benchmark_v3"
    assert config["collection"] == {
        "claim_policy": "diagnostic_v1",
        "base_seed": 20260826,
        "warmup_blocks": 1,
        "measurement_blocks": 5,
        "session_policy": "fresh_session_per_attempt_v1",
        "block_cooldown_s": 0.0,
        "machine_policy": {
            "machine_exclusivity": {"mode": "observed_v1"},
            "cpu_governor": {"mode": "observed_v1"},
            "affinity": {"mode": "observed_v1", "expected_cpus": None},
            "numa_policy": {"mode": "observed_v1"},
            "background_load": {
                "mode": "observed_v1",
                "max_load1_per_online_cpu": None,
            },
        },
    }


def _config() -> str:
    return """\
schema_version: tn_benchmark_v3
experiment_id: focused
defaults:
  timeout_s: 2.5
collection:
  claim_policy: diagnostic_v1
  base_seed: 7
  warmup_blocks: 0
  measurement_blocks: 1
  session_policy: fresh_session_per_attempt_v1
  block_cooldown_s: 0.0
  machine_policy:
    machine_exclusivity:
      mode: observed_v1
    cpu_governor:
      mode: observed_v1
    affinity:
      mode: observed_v1
      expected_cpus: null
    numa_policy:
      mode: observed_v1
    background_load:
      mode: observed_v1
      max_load1_per_online_cpu: null
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


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_repeats", "1"), ("seed", "0"), ("extra", "true")),
)
def test_planner_opt_einsum_rejects_inert_and_extra_fields(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / f"opt-einsum-{field}.yml"
    text = _config().replace(
        "      mode: greedy\n    slicing: null",
        f"      mode: greedy\n      {field}: {value}\n    slicing: null",
        1,
    )
    _write_config(path, text)

    with pytest.raises(ValueError, match="fields must be exact"):
        load_experiment_config(path)


def test_planner_cotengra_preserves_consumed_fields(tmp_path: Path) -> None:
    path = tmp_path / "cotengra.yml"
    text = _config().replace(
        "      engine: opt_einsum\n      mode: greedy",
        "      engine: cotengra\n      mode: labels\n"
        "      max_repeats: 7\n      seed: 19",
        1,
    )
    _write_config(path, text)

    config = load_experiment_config(path)

    assert dict(config["plans"]["p1"]["planner"]) == {
        "engine": "cotengra",
        "mode": "labels",
        "max_repeats": 7,
        "seed": 19,
    }


def _selected_configurations(
    config: object,
) -> list[tuple[object, str, object]]:
    return [
        (item, route_id, config["routes"][route_id])
        for item in config["matrix"]
        for route_id in item["route_ids"]
    ]


def test_sequential_upmem_correctness_config_contract() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_sequential_upmem_correctness.yml"
    )

    assert config["collection"]["claim_policy"] == "diagnostic_v1"
    assert config["collection"]["warmup_blocks"] == 0
    assert config["collection"]["measurement_blocks"] == 1
    assert set(config["cases"]) == {"bell2", "stress4"}
    assert config["cases"]["bell2"]["circuit"]["name"] == "bell_2q"
    assert dict(config["cases"]["stress4"]["circuit"]["parameters"]) == {
        "n_qubits": 4,
        "repeat_layers": 2,
    }
    assert config["plans"]["unsliced"]["slicing"] is None
    assert dict(config["plans"]["sliced"]["slicing"]) == {
        "node_id": "contract_24",
        "minimum_slice_count": 4,
    }
    assert set(config["routes"]) == {"upmem_float32_1dpu_t1"}
    route = config["routes"]["upmem_float32_1dpu_t1"]
    assert route["executor"] == "upmem_physical"
    assert route["numeric_policy"] == "split_complex_float32_v1"
    assert tuple(
        route["options"][field]
        for field in ("rank_count", "dpu_count", "tasklets_per_dpu")
    ) == (1, 1, 1)
    assert [
        (item["case_id"], item["plan_id"], tuple(item["route_ids"]))
        for item in config["matrix"]
    ] == [
        ("bell2", "unsliced", ("upmem_float32_1dpu_t1",)),
        ("stress4", "unsliced", ("upmem_float32_1dpu_t1",)),
        ("stress4", "sliced", ("upmem_float32_1dpu_t1",)),
    ]
    assert dict(cli._expected_counts(config)) == {
        "warmup": 0,
        "measurement": 3,
        "sessions": 3,
    }


def test_sequential_upmem_performance_config_uses_complete_randomized_blocks() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_sequential_upmem_performance.yml"
    )

    assert set(config["cases"]) == {"stress18"}
    assert dict(config["cases"]["stress18"]["circuit"]["parameters"]) == {
        "n_qubits": 18,
        "repeat_layers": 2,
    }
    assert config["collection"]["claim_policy"] == "diagnostic_v1"
    assert config["collection"]["warmup_blocks"] == 2
    assert config["collection"]["measurement_blocks"] == 30
    assert config["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (0,),
    }
    assert config["collection"]["machine_policy"]["cpu_governor"] == {
        "mode": "observed_v1"
    }
    assert tuple(config["routes"]) == (
        "numpy_same_dag",
        "upmem_float32_1dpu_t1",
    )
    assert {
        route_id: config["routes"][route_id]["executor"]
        for route_id in config["routes"]
    } == {
        "numpy_same_dag": "numpy_dag",
        "upmem_float32_1dpu_t1": "upmem_physical",
    }
    assert all(
        route["numeric_policy"] == "split_complex_float32_v1"
        for route in config["routes"].values()
    )
    physical = config["routes"]["upmem_float32_1dpu_t1"]
    assert tuple(
        physical["options"][field]
        for field in ("rank_count", "dpu_count", "tasklets_per_dpu")
    ) == (1, 1, 1)
    assert dict(cli._expected_counts(config)) == {
        "warmup": 4,
        "measurement": 60,
        "sessions": 32,
    }

    schedule = cli._scheduled_attempts(config, _selected_configurations(config))
    block_orders = {
        block_id: tuple(
            route_id
            for _, route_id, _, attempt in schedule
            if attempt[2] == block_id
        )
        for block_id in range(32)
    }
    expected_routes = {"numpy_same_dag", "upmem_float32_1dpu_t1"}
    assert len(schedule) == 64
    assert all(set(order) == expected_routes and len(order) == 2 for order in block_orders.values())
    assert len(set(block_orders.values())) == 2


def test_external_tn_context_config_is_two_quimb_execution_routes() -> None:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_external_tn_context.yml"
    )

    assert set(config["cases"]) == {"stress18"}
    assert dict(config["cases"]["stress18"]["circuit"]["parameters"]) == {
        "n_qubits": 18,
        "repeat_layers": 2,
    }
    assert config["collection"]["claim_policy"] == "diagnostic_v1"
    assert config["collection"]["warmup_blocks"] == 1
    assert config["collection"]["measurement_blocks"] == 5
    assert dict(config["plans"]) == {}
    assert tuple(config["routes"]) == ("quimb_greedy", "quimb_cotengra_path")
    assert config["routes"]["quimb_greedy"] == {
        "executor": "quimb",
        "numeric_policy": None,
        "options": {"optimize": "greedy"},
    }
    assert config["routes"]["quimb_cotengra_path"] == {
        "executor": "cotengra",
        "numeric_policy": None,
        "options": {"methods": "greedy", "max_repeats": 1},
    }
    assert all(item["plan_id"] is None for item in config["matrix"])
    assert baselines._SCOPE == "simulation_end_to_end_v1"
    assert dict(cli._expected_counts(config)) == {
        "warmup": 2,
        "measurement": 10,
        "sessions": 0,
    }


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


def test_loader_allows_distinct_plan_occurrences_but_rejects_exact_duplicates(
    tmp_path: Path,
) -> None:
    distinct = tmp_path / "distinct_plans.yml"
    distinct_text = (
        _config()
        .replace(
            "    slicing: null\nroutes:",
            "    slicing: null\n  p2:\n    planner:\n      engine: opt_einsum\n"
            "      mode: optimal\n"
            "    slicing: null\nroutes:",
        )
        .replace(
            "  - case_id: builtin_case",
            "  - case_id: qasm_case\n    plan_id: p2\n    route_ids: [numpy]\n"
            "  - case_id: builtin_case",
        )
    )
    _write_config(distinct, distinct_text)
    config = load_experiment_config(distinct)
    assert [entry["plan_id"] for entry in config["matrix"][:2]] == ["p1", "p2"]

    duplicate = tmp_path / "duplicate_matrix.yml"
    text = _config().replace(
        "  - case_id: builtin_case",
        "  - case_id: qasm_case\n    plan_id: p1\n    route_ids: [numpy]\n"
        "  - case_id: builtin_case",
    )
    _write_config(duplicate, text)

    with pytest.raises(ValueError, match="combination once"):
        load_experiment_config(duplicate)


def test_loader_allows_planless_baseline_only_configuration(tmp_path: Path) -> None:
    path = tmp_path / "baseline.yml"
    text = _config().replace(
        "plans:\n  p1:\n    planner:\n      engine: opt_einsum\n"
        "      mode: greedy\n"
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
    _write_config(
        second, _config().replace("measurement_blocks: 1", "measurement_blocks: 2")
    )

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


def _numpy_config(*, warmups: int = 0, repetitions: int = 1) -> str:
    return f"""\
schema_version: tn_benchmark_v3
experiment_id: cli-focused
defaults:
  timeout_s: 2.5
collection:
  claim_policy: diagnostic_v1
  base_seed: 7
  warmup_blocks: {warmups}
  measurement_blocks: {repetitions}
  session_policy: fresh_session_per_attempt_v1
  block_cooldown_s: 0.0
  machine_policy:
    machine_exclusivity:
      mode: observed_v1
    cpu_governor:
      mode: observed_v1
    affinity:
      mode: observed_v1
      expected_cpus: null
    numa_policy:
      mode: observed_v1
    background_load:
      mode: observed_v1
      max_load1_per_online_cpu: null
cases:
  bell:
    circuit:
      kind: builtin
      name: bell_2q
      path: null
      parameters: {{}}
plans:
  greedy:
    planner:
      engine: opt_einsum
      mode: greedy
    slicing: null
routes:
  numpy:
    executor: numpy_dag
    numeric_policy: split_complex_float32_v1
    options: {{}}
matrix:
  - case_id: bell
    plan_id: greedy
    route_ids: [numpy]
"""


def _two_plan_numpy_config() -> str:
    return (
        _numpy_config()
        .replace(
            "    slicing: null\nroutes:",
            "    slicing: null\n  optimal:\n    planner:\n      engine: opt_einsum\n"
            "      mode: optimal\n"
            "    slicing: null\nroutes:",
        )
        .replace(
            "    route_ids: [numpy]\n",
            "    route_ids: [numpy]\n  - case_id: bell\n"
            "    plan_id: optimal\n    route_ids: [numpy]\n",
            1,
        )
    )


def _physical_config() -> str:
    return (
        _numpy_config()
        .replace(
            "  numpy:\n    executor: numpy_dag\n    numeric_policy: split_complex_float32_v1\n    options: {}",
            "  physical:\n    executor: upmem_physical\n"
            "    numeric_policy: split_complex_float32_v1\n"
            "    options:\n"
            "      dpu_count: 1\n      rank_count: 1\n      tasklets_per_dpu: 1\n"
            "      session_root: native\n      host_binary: native/host\n"
            "      dpu_binary: native/dpu\n      initialization_binary: native/init\n"
            "      rank_paths: [/dev/dpu_rank0]",
        )
        .replace("route_ids: [numpy]", "route_ids: [physical]")
    )


def _physical_performance_config() -> str:
    return _physical_config().replace(
        "  claim_policy: diagnostic_v1\n"
        "  base_seed: 7\n"
        "  warmup_blocks: 0\n"
        "  measurement_blocks: 1\n"
        "  session_policy: fresh_session_per_attempt_v1\n"
        "  block_cooldown_s: 0.0\n"
        "  machine_policy:\n"
        "    machine_exclusivity:\n"
        "      mode: observed_v1\n"
        "    cpu_governor:\n"
        "      mode: observed_v1\n"
        "    affinity:\n"
        "      mode: observed_v1\n"
        "      expected_cpus: null\n"
        "    numa_policy:\n"
        "      mode: observed_v1\n"
        "    background_load:\n"
        "      mode: observed_v1\n"
        "      max_load1_per_online_cpu: null",
        "  claim_policy: physical_performance_v1\n"
        "  base_seed: 7\n"
        "  warmup_blocks: 2\n"
        "  measurement_blocks: 30\n"
        "  session_policy: fresh_session_per_attempt_v1\n"
        "  block_cooldown_s: 0.0\n"
        "  machine_policy:\n"
        "    machine_exclusivity:\n"
        "      mode: operator_attested_v1\n"
        "    cpu_governor:\n"
        "      mode: performance_required_v1\n"
        "    affinity:\n"
        "      mode: exact_required_v1\n"
        "      expected_cpus: [0]\n"
        "    numa_policy:\n"
        "      mode: operator_attested_v1\n"
        "    background_load:\n"
        "      mode: observed_v1\n"
        "      max_load1_per_online_cpu: 0.25",
    )


def _simulator_config() -> str:
    return (
        _numpy_config()
        .replace(
            "  numpy:\n    executor: numpy_dag\n    numeric_policy: split_complex_float32_v1\n    options: {}",
            "  simulator:\n    executor: upmem_sdk_simulator\n"
            "    numeric_policy: split_complex_float32_v1\n"
            "    options:\n"
            "      dpu_count: 1\n      rank_count: 1\n      tasklets_per_dpu: 1\n"
            "      session_root: native\n      host_binary: native/host\n"
            "      dpu_binary: native/dpu\n      initialization_binary: native/init",
        )
        .replace("route_ids: [numpy]", "route_ids: [simulator]")
    )


def test_plan_never_opens_a_session_and_writes_deterministic_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config())
    monkeypatch.setattr(
        cli, "open_upmem", lambda *args, **kwargs: pytest.fail("opened")
    )
    monkeypatch.setattr(
        cli, "open_upmem_simulator", lambda *args, **kwargs: pytest.fail("opened")
    )

    first = cli.plan_command(str(config), str(tmp_path / "first"))
    cli.plan_command(str(config), str(tmp_path / "second"))

    assert first["status"] == "planned"
    assert (tmp_path / "first" / "plan.json").read_bytes() == (
        tmp_path / "second" / "plan.json"
    ).read_bytes()
    document = json.loads((tmp_path / "first" / "plan.json").read_text())
    assert document["schema_version"] == "tn_benchmark_plan_v1"


def test_sliced_conformance_retains_strict_sdk_simulator_coverage(
    tmp_path: Path,
) -> None:
    result = cli.plan_command(
        str(ROOT / "configs" / "tn_benchmark_simulator.yml"),
        str(tmp_path / "simulator-plan"),
    )

    assert result["status"] == "planned"
    document = json.loads(
        (tmp_path / "simulator-plan" / "plan.json").read_text(encoding="utf-8")
    )
    entries = {entry["route_id"]: entry for entry in document["entries"]}
    assert set(entries) == {
        "simulator_float32_t1",
        "simulator_float32_t8",
        "simulator_int8_t1",
        "simulator_int8_t8",
    }

    float_entry = entries["simulator_float32_t1"]
    int8_entry = entries["simulator_int8_t1"]
    assert float_entry["problem_id"] == (
        "42b85161b341872ea93285b649d9fbb9d146de3f378228309826722a071d925d"
    )
    assert float_entry["tensor_network_structure_id"] == (
        "21aca2c497034eea383263931b6bdf6f1f8f03c791a587e721b92b584d38a856"
    )
    assert int8_entry["problem_id"] == float_entry["problem_id"]
    assert (
        int8_entry["tensor_network_structure_id"]
        == float_entry["tensor_network_structure_id"]
    )
    assert float_entry["logical_plan_id"] == (
        "fd59ad9414b06631f0dc068b36bf8f2b8b7e0cd72000fdf20045f35cd32ed902"
    )
    assert int8_entry["logical_plan_id"] == float_entry["logical_plan_id"]
    assert float_entry["upmem"]["physical_plan_id"] == (
        "4ff5cad04ff84fbe1ebf3cdd2bd1b8226913426dacbdc95d54fb9898cbb806c5"
    )
    assert int8_entry["upmem"]["physical_plan_id"] == (
        "644c1f556fa8665b950e445b865bcd52e014a25a0698c49d623c3377430a30c2"
    )
    assert float_entry["upmem"]["kernel_policy"] == "dpu_real_tile_v4_wram_panel_v1"
    assert int8_entry["upmem"]["kernel_policy"] == "dpu_real_tile_v4_wram_panel_v1"
    assert (
        float_entry["upmem"]["physical_plan_id"]
        != int8_entry["upmem"]["physical_plan_id"]
    )
    assert entries["simulator_float32_t8"]["upmem"]["physical_plan_id"] == (
        "eb1a228c8d24ff214298d1da6dec155cc8c109ffffbffe829940b74f7fae6171"
    )
    assert entries["simulator_int8_t8"]["upmem"]["physical_plan_id"] == (
        "a7431e780bb523e4470aecbc436bf723cef476c48bd37a349011eab4d77796c1"
    )
    assert {
        route_id: entry["upmem"]["topology"]["tasklets_per_dpu"]
        for route_id, entry in entries.items()
    } == {
        "simulator_float32_t1": 1,
        "simulator_float32_t8": 8,
        "simulator_int8_t1": 1,
        "simulator_int8_t8": 8,
    }
    assert {
        entry["logical_plan_id"] for entry in entries.values()
    } == {float_entry["logical_plan_id"]}
    assert {
        entry["upmem"]["kernel_policy"] for entry in entries.values()
    } == {"dpu_real_tile_v4_wram_panel_v1"}

    branches = [
        "contract_24__slice__label_12_value_0__label_14_value_0",
        "contract_24__slice__label_12_value_0__label_14_value_1",
        "contract_24__slice__label_12_value_1__label_14_value_0",
        "contract_24__slice__label_12_value_1__label_14_value_1",
    ]
    reduction = "contract_24__reduce__label_12__label_14"
    relevant_node_ids = {*branches, reduction}
    for entry in entries.values():
        relevant_stages = [
            (stage["kind"], stage["node_ids"])
            for stage in entry["upmem"]["stages"]
            if any(node_id in relevant_node_ids for node_id in stage["node_ids"])
        ]
        assert relevant_stages == [
            ("contract_batch", branches),
            ("host_reduce", [reduction]),
        ]


def test_validation_phase_alignment_cannot_qualify_raw_failure() -> None:
    expected = np.array([1.0, 0.0], dtype=np.complex128)
    sample = ExecutionSample(
        output=1j * expected,
        measurement=Measurement(scope_id="steady_execution_v1", total_wall_s=1.0),
        backend_facts={},
        numeric_facts={},
    )

    validation = cli._validation(
        sample=sample,
        policy_reference=None,
        full_reference=expected,
        numeric_policy="split_complex_float32_v1",
        require_raw_lanes=False,
    )

    assert validation["phase_aligned_max_abs_error"] <= 1.0e-12
    assert validation["max_abs_error"] > 1.0
    assert validation["relative_l2_error"] > 1.0
    assert validation["norm_drift"] == 0.0
    assert validation["full_precision_passed"] is False
    assert validation["accuracy_qualified"] is False


def test_validation_gates_relative_l2_in_addition_to_raw_allclose() -> None:
    expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    actual = expected + np.array([0.0, 9.0e-6, 9.0e-6, 9.0e-6])
    assert np.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)
    sample = ExecutionSample(
        output=actual,
        measurement=Measurement(scope_id="steady_execution_v1", total_wall_s=1.0),
        backend_facts={},
        numeric_facts={},
    )

    validation = cli._validation(
        sample=sample,
        policy_reference=None,
        full_reference=expected,
        numeric_policy="split_complex_float32_v1",
        require_raw_lanes=False,
    )

    assert validation["relative_l2_error"] > 1.0e-5
    assert validation["full_precision_passed"] is False


def test_run_direct_dispatch_writes_exact_evidence_files(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config(warmups=1, repetitions=2))

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)

    assert result["status"] == "completed"
    run_dir = tmp_path / "run"
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "samples.jsonl",
        "sessions.jsonl",
    }
    assert (run_dir / "sessions.jsonl").read_text() == ""
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text().splitlines()
    ]
    assert [(row["attempt_kind"], row["sample_index"]) for row in samples] == [
        ("warmup", 0),
        ("measurement", 0),
        ("measurement", 1),
    ]
    assert {row["plan_id"] for row in samples} == {"greedy"}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bindings = manifest["configuration"]["identity_bindings"]
    assert [
        (binding["case_id"], binding["plan_id"], binding["route_id"])
        for binding in bindings
    ] == [("bell", "greedy", "numpy")]
    assert samples[0]["identities"] == {
        field: bindings[0][field] for field in samples[0]["identities"]
    }


def test_run_keeps_same_case_route_samples_separate_by_plan_id(tmp_path: Path) -> None:
    config = tmp_path / "two-plans.yml"
    _write_config(config, _two_plan_numpy_config())

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)

    assert result["status"] == "completed"
    samples = [
        json.loads(line)
        for line in (tmp_path / "run" / "samples.jsonl").read_text().splitlines()
    ]
    assert sorted(
        (sample["case_id"], sample["plan_id"], sample["route_id"])
        for sample in samples
    ) == [
        ("bell", "greedy", "numpy"),
        ("bell", "optimal", "numpy"),
    ]
    assert samples[0]["sample_id"] != samples[1]["sample_id"]


def test_collection_schedule_is_deterministic_and_records_block_order(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "two-plans.yml"
    _write_config(config_path, _two_plan_numpy_config())
    config = load_experiment_config(config_path)
    selected = [
        (item, route_id, config["routes"][route_id])
        for item in config["matrix"]
        for route_id in item["route_ids"]
    ]

    first = cli._scheduled_attempts(config, selected)
    second = cli._scheduled_attempts(config, selected)

    assert first == second
    assert [(attempt[0], attempt[2], attempt[3]) for *_, attempt in first] == [
        ("measurement", 0, 0),
        ("measurement", 0, 1),
    ]


def test_block_cooldown_occurs_once_after_each_nonfinal_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "cooldown.yml"
    _write_config(
        config_path,
        _numpy_config(repetitions=2).replace(
            "block_cooldown_s: 0.0", "block_cooldown_s: 0.25"
        ),
    )
    config = load_experiment_config(config_path)
    selected = [
        (item, route_id, config["routes"][route_id])
        for item in config["matrix"]
        for route_id in item["route_ids"]
    ]
    scheduled = cli._scheduled_attempts(config, selected)
    calls: list[float] = []
    monkeypatch.setattr(cli.time, "sleep", calls.append)

    for index in range(len(scheduled)):
        cli._complete_collection_block(config, scheduled, index)

    assert calls == [0.25]


def test_physical_machine_preflight_records_static_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "physical-performance.yml"
    _write_config(config_path, _physical_performance_config())
    config = load_experiment_config(config_path)
    monkeypatch.setenv("QUANTUM_BENCH_EXCLUSIVITY_ATTESTED", "1")
    monkeypatch.setenv("QUANTUM_BENCH_NUMA_ATTESTED", "1")
    monkeypatch.setattr(cli, "_observed_affinity", lambda: [0])
    monkeypatch.setattr(
        cli, "_cpu_governors", lambda _cpu_ids: {"0": "performance"}
    )
    monkeypatch.setattr(cli, "_numa_nodes", lambda: ["node0"])
    normalized_rank_paths = ("normalized-rank-path",)
    monkeypatch.setattr(
        cli, "_physical_rank_paths", lambda _config: normalized_rank_paths
    )
    monkeypatch.setattr(
        cli, "_rank_paths_accessible", lambda paths: paths == normalized_rank_paths
    )
    monkeypatch.setattr(cli, "_tool_version", lambda command: "2023.1")
    monkeypatch.setattr(cli, "_background_load_1m", lambda: 1.0)
    monkeypatch.setattr(cli, "_online_logical_cpu_count", lambda: 8)
    monkeypatch.setattr(cli, "_utc_now", lambda: "2026-08-26T12:00:00Z")

    facts = cli._machine_preflight(config)

    assert facts["machine_preflight_passed"] is True
    assert facts["machine_preflight_reasons"] == ()
    assert facts["selected_cpu_ids"] == (0,)
    assert facts["observed_affinity"] == [0]
    assert facts["observed_cpu_governors"] == {"0": "performance"}
    assert facts["initial_load1_per_online_cpu"] == 0.125
    assert facts["exclusivity_attestation_recorded_at_utc"] == "2026-08-26T12:00:00Z"


def test_environment_provenance_is_compact_and_thread_distinguishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "environment.yml"
    _write_config(config_path, _numpy_config())
    config = load_experiment_config(config_path)
    preflight = {
        "observed_affinity": [2, 3],
        "selected_cpu_ids": [2, 3],
        "observed_cpu_governors": {"2": "schedutil", "3": "schedutil"},
    }
    monkeypatch.setattr(
        cli.np,
        "show_config",
        lambda **_kwargs: {
            "Build Dependencies": {
                "blas": {
                    "found": True,
                    "name": " scipy-openblas ",
                    "version": " 0.3.29 ",
                    "include directory": "/irrelevant/build/include",
                    "lib directory": "/irrelevant/build/lib",
                }
            },
            "Python Information": {"path": "/irrelevant/build/python"},
        },
    )
    monkeypatch.setattr(cli, "_tool_version", lambda _command: None)
    monkeypatch.setattr(cli, "_background_load_1m", lambda: None)
    monkeypatch.setattr(cli, "_numa_nodes", lambda: [])
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    first_id, first = cli._environment(config, preflight)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    second_id, second = cli._environment(config, preflight)

    assert first["numpy_version"] == np.__version__
    assert first["blas"] == {"name": "scipy-openblas", "version": "0.3.29"}
    assert "/irrelevant/build" not in canonical_json(first)
    assert first["thread_environment"] == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
    }
    assert first["affinity"] == [2, 3]
    assert first["selected_cpu_ids"] == [2, 3]
    assert first["observed_cpu_governors"] == {
        "2": "schedutil",
        "3": "schedutil",
    }
    assert second["thread_environment"]["OMP_NUM_THREADS"] == "2"
    assert first_id != second_id


def test_physical_machine_preflight_failure_finalizes_without_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.setattr(
        cli,
        "_machine_preflight",
        lambda config: {
            "machine_preflight_passed": False,
            "machine_preflight_reasons": ("rank_paths_inaccessible",),
        },
    )

    config_path = tmp_path / "physical-performance.yml"
    _write_config(config_path, _physical_performance_config())
    result = cli.run_command(
        str(config_path), str(tmp_path / "preflight-failure"), allow_physical=True
    )

    assert result["status"] == "failed"
    root = tmp_path / "preflight-failure"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["configuration"]["environment"]["machine_preflight"][
        "machine_preflight_reasons"
    ] == ["rank_paths_inaccessible"]
    assert (root / "samples.jsonl").read_text(encoding="utf-8") == ""
    assert (root / "sessions.jsonl").read_text(encoding="utf-8") == ""


def test_physical_collection_admission_requires_useful_dpu_and_tasklet_work() -> None:
    def work_unit(*, dpu: int, m_size: int = 8) -> UpmemWorkUnit:
        return UpmemWorkUnit(
            node_id="contract",
            stable_tile_id=f"contract:{dpu}",
            wave=0,
            logical_rank=0,
            logical_dpu=dpu,
            batch_start=0,
            batch_size=1,
            m_start=dpu * m_size,
            m_size=m_size,
            n_start=0,
            n_size=1,
            k_start=0,
            k_size=1,
            estimated_input_bytes=8,
            estimated_output_bytes=8,
            aligned_mram_bytes=24,
            estimated_arithmetic_work=m_size,
        )

    def plan(*, dpus: int, tasklets: int, units: tuple[UpmemWorkUnit, ...]) -> UpmemPlan:
        return UpmemPlan(
            logical_plan_id="a" * 64,
            numeric_policy="split_complex_float32_v1",
            topology=UpmemTopology(
                dpu_count=dpus, rank_count=1, tasklets_per_dpu=tasklets
            ),
            stages=(
                UpmemStage(
                    stage_id="contract_batch:contract",
                    kind="contract_batch",
                    node_ids=("contract",),
                    work_units=units,
                ),
            ),
        )

    with pytest.raises(cli.UnsupportedExecution, match="fully populated"):
        cli._require_collection_resource_admission(
            plan(dpus=2, tasklets=1, units=(work_unit(dpu=0),)),
            physical_performance_campaign=True,
        )

    cli._require_collection_resource_admission(
        plan(dpus=2, tasklets=1, units=(work_unit(dpu=0), work_unit(dpu=1))),
        physical_performance_campaign=True,
    )

    with pytest.raises(cli.UnsupportedExecution, match="dominant-work unit"):
        cli._require_collection_resource_admission(
            plan(dpus=1, tasklets=8, units=(work_unit(dpu=0, m_size=4),)),
            physical_performance_campaign=True,
        )


def test_load_rejects_tampered_experiment_identity_payload(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config())
    run_dir = tmp_path / "run"
    cli.run_command(str(config), str(run_dir), allow_physical=False)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest["configuration"]["experiment"]["experiment_identity_payload"]
    payload["label"] = "tampered"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment identity payload"):
        load_artifacts(run_dir)


def test_load_rejects_tampered_declared_collection_order(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config())
    run_dir = tmp_path / "run"
    cli.run_command(str(config), str(run_dir), allow_physical=False)

    samples_path = run_dir / "samples.jsonl"
    sample = json.loads(samples_path.read_text(encoding="utf-8"))
    sample["order_index"] = 1
    sample["sample_id"] = sample_id(
        sample["run_id"],
        sample["case_id"],
        sample["route_id"],
        sample["attempt_kind"],
        sample["sample_index"],
        plan_id=sample["plan_id"],
        block_id=sample["block_id"],
        order_index=sample["order_index"],
    )
    samples_path.write_text(canonical_json(sample) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="declared collection schedule"):
        load_artifacts(run_dir)


def test_simulator_route_uses_simulator_session_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _simulator_config())
    observed: dict[str, object] = {}

    def fake_sessions(**kwargs: object) -> tuple[tuple[object, ...], object]:
        observed.update(kwargs)
        return (), {}

    monkeypatch.setattr(cli, "run_session_samples", fake_sessions)
    monkeypatch.setattr(cli, "open_upmem_simulator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli, "open_upmem", lambda *args, **kwargs: pytest.fail("physical")
    )
    monkeypatch.setattr(cli, "finalize_artifacts", lambda *args, **kwargs: None)

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)

    assert result["status"] == "completed"
    assert observed["session_protocol_id"] == "upmem_real_tile_abi_v4"
    assert observed["open_session"]() is not None


def test_unsupported_upmem_mapping_is_retained_as_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _simulator_config())

    def reject(*_args: object, **_kwargs: object) -> object:
        raise cli.UnsupportedExecution("mapping", "shape rejected", "tile_shape")

    monkeypatch.setattr(cli, "plan_upmem", reject)

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)
    samples = [
        json.loads(line)
        for line in (tmp_path / "run" / "samples.jsonl").read_text().splitlines()
    ]

    assert result["status"] == "failed"
    assert len(samples) == 1
    assert samples[0]["status"] == "unsupported"
    assert samples[0]["failure"] == {
        "capability": "tile_shape",
        "reason": "shape rejected",
        "stage": "mapping",
    }
    assert (tmp_path / "run" / "sessions.jsonl").read_text() == ""
    manifest, loaded_samples, loaded_sessions = load_artifacts(tmp_path / "run")
    assert manifest["status"] == "failed"
    assert len(manifest["configuration"]["identity_bindings"]) == 1
    assert manifest["configuration"]["identity_bindings"][0][
        "physical_plan_id"
    ] is None
    assert loaded_samples == tuple(samples)
    assert loaded_sessions == ()


def test_physical_dual_opt_in_and_qualify_route_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical = tmp_path / "physical.yml"
    _write_config(physical, _physical_config())
    with pytest.raises(ValueError, match="--allow-physical"):
        cli.run_command(str(physical), str(tmp_path / "run"), allow_physical=False)
    with pytest.raises(ValueError, match="--allow-physical"):
        cli.qualify_command(
            str(physical), str(tmp_path / "qualify-physical"), allow_physical=False
        )
    monkeypatch.delenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", raising=False)
    with pytest.raises(ValueError, match="UPMEM_ALLOW_PHYSICAL_HARDWARE"):
        cli.run_command(str(physical), str(tmp_path / "run"), allow_physical=True)
    monkeypatch.setenv("UPMEM_ALLOW_PHYSICAL_HARDWARE", "1")
    monkeypatch.setattr(cli, "_worktree_dirty", lambda: True)
    with pytest.raises(ValueError, match="clean Git worktree"):
        cli.qualify_command(
            str(physical), str(tmp_path / "dirty-qualify"), allow_physical=True
        )

    baseline = tmp_path / "baseline.yml"
    _write_config(baseline, _numpy_config())
    with pytest.raises(ValueError, match="only upmem_physical"):
        cli.qualify_command(
            str(baseline), str(tmp_path / "qualify"), allow_physical=True
        )


def test_executable_identity_excludes_route_and_numeric_policy() -> None:
    float_route = {
        "executor": "numpy_dag",
        "numeric_policy": "split_complex_float32_v1",
        "options": {},
    }
    int8_route = {
        **float_route,
        "numeric_policy": "split_complex_int8_shared_scale_v1",
    }

    assert cli._executable_identity(float_route) == cli._executable_identity(int8_route)


def test_failed_finalization_returns_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config())
    calls: list[str] = []

    def finalize(_: Path, *, status: str) -> None:
        calls.append(status)
        if status == "completed":
            raise ValueError("aggregate failed")

    monkeypatch.setattr(cli, "finalize_artifacts", finalize)

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)

    assert result["status"] == "failed"
    assert calls == ["completed", "failed"]


def test_report_command_returns_success_for_completed_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quantum_bench.report.report_artifacts",
        lambda _input, _output: {"status": "completed"},
    )

    assert cli.main(["report", "--input", "evidence", "--output", "report"]) == 0


def test_slicing_is_selected_by_named_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        "circuit": {
            "kind": "builtin",
            "name": "bell_2q",
            "path": None,
            "parameters": {},
        }
    }
    job = cli._job(case)
    unsliced = {
        "planner": {
            "engine": "opt_einsum",
            "mode": "greedy",
        },
        "slicing": None,
    }
    _, _, dag, _ = cli._plan_dag(job, unsliced)
    node_id = next(
        node.node_id for node in dag.nodes if hasattr(node, "contracted_labels")
    )
    selected: list[str] = []
    original = cli.slice_contraction

    def sliced(*args: object, **kwargs: object):
        selected.append(str(kwargs["node_id"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "slice_contraction", sliced)
    sliced_plan = {
        **unsliced,
        "slicing": {"node_id": node_id, "minimum_slice_count": 2},
    }

    cli._plan_dag(job, sliced_plan)

    assert selected == [node_id]


def test_parser_accepts_public_commands() -> None:
    parser = cli._parser()
    assert (
        parser.parse_args(["plan", "--config", "x", "--output", "y"]).command == "plan"
    )
    assert parser.parse_args(["verify", "--input", "x"]).command == "verify"


def test_plan_and_run_reject_nonempty_output_directories(tmp_path: Path) -> None:
    config = tmp_path / "config.yml"
    _write_config(config, _numpy_config())
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("not evidence", encoding="utf-8")

    with pytest.raises(ValueError, match="absent or empty"):
        cli.plan_command(str(config), str(occupied))
    with pytest.raises(ValueError, match="absent or empty"):
        cli.run_command(str(config), str(occupied), allow_physical=False)


def test_make_help_lists_active_workflow() -> None:
    _require_make()
    result = _command("make", "help")

    assert result.returncode == 0, result.stderr
    for target in (
        "make plan",
        "make run",
        "make report",
        "make verify",
        "make build-upmem-runtime",
        "make qualify",
    ):
        assert target in result.stdout
    assert "PHYSICAL_CONFIG=<prepared-yaml>" in result.stdout


def test_make_qualify_requires_an_explicit_physical_config() -> None:
    _require_make()
    result = _command("make", "-s", "qualify", "UPMEM_ALLOW_PHYSICAL_HARDWARE=1")

    assert result.returncode == 2
    assert "Set PHYSICAL_CONFIG=<prepared ignored config>" in result.stderr


@pytest.mark.parametrize(
    ("target", "arguments", "fragment"),
    (
        ("plan", ("CONFIG=config.yml", "OUTPUT=/tmp/plan"), "quantum_bench.cli plan"),
        ("run", ("CONFIG=config.yml", "OUTPUT=/tmp/run"), "quantum_bench.cli run"),
        (
            "report",
            ("INPUT=/tmp/run", "REPORT_OUTPUT=/tmp/report"),
            "quantum_bench.cli report",
        ),
        ("verify", ("INPUT=/tmp/run",), "quantum_bench.cli verify"),
        (
            "qualify",
            ("PHYSICAL_CONFIG=config.yml", "OUTPUT=/tmp/qualify"),
            "quantum_bench.cli qualify",
        ),
    ),
)
def test_make_forwards_active_cli_commands(
    target: str, arguments: tuple[str, ...], fragment: str
) -> None:
    _require_make()
    result = _command("make", "-n", target, *arguments)

    assert result.returncode == 0, result.stderr
    assert fragment in result.stdout


def test_make_build_forwards_tasklet_count() -> None:
    _require_make()
    result = _command("make", "-n", "build-upmem-runtime", "UPMEM_TASKLETS=8")

    assert result.returncode == 0, result.stderr
    assert "NR_TASKLETS=8" in result.stdout


def test_sequential_baseline_makefile_and_docs_commands_are_consistent() -> None:
    _require_make()
    help_result = _command("make", "help")
    conformance = _command("make", "-n", "sequential-conformance")
    qualifier = _command("make", "-n", "sequential-baseline")
    document = (ROOT / "docs" / "sequential_upmem_baseline.md").read_text(
        encoding="utf-8"
    )

    assert "make sequential-conformance" in help_result.stdout
    assert "make sequential-baseline" in help_result.stdout
    assert "check_sequential_conformance.py --output" in conformance.stdout
    assert "qualify_sequential_baseline.py --help" in qualifier.stdout
    assert "make sequential-conformance" in document
    assert "make sequential-baseline" in document
    assert "PHYSICAL_CONFIG=" in document
    assert "--allow-physical" in document


def test_active_docs_declare_evidence_sample_v4_only() -> None:
    for path in (ROOT / "README.md", ROOT / "docs" / "evidence_workflow.md"):
        document = path.read_text(encoding="utf-8")
        assert "evidence_sample_v4" in document
        assert "evidence_sample_v3" not in document


def test_numpy_plan_run_verify_report_lifecycle(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yml"
    _write_config(config, _numpy_config())
    plan_dir = tmp_path / "plan"
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"

    plan = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "plan",
        "--config",
        str(config),
        "--output",
        str(plan_dir),
    )
    run = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "run",
        "--config",
        str(config),
        "--output",
        str(run_dir),
    )
    verify = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "verify",
        "--input",
        str(run_dir),
    )
    report = _command(
        PYTHON,
        "-m",
        "quantum_bench.cli",
        "report",
        "--input",
        str(run_dir),
        "--output",
        str(report_dir),
    )

    assert plan.returncode == 0, plan.stderr
    assert json.loads(plan.stdout)["status"] == "planned"
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["status"] == "completed"
    assert verify.returncode == 0, verify.stderr
    assert json.loads(verify.stdout)["status"] == "completed"
    assert report.returncode == 0, report.stderr
    assert json.loads(report.stdout)["status"] == "completed"
    assert (plan_dir / "plan.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "samples.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (report_dir / "report.json").is_file()


def _measurement(scope_id: str, total_wall_s: float) -> dict[str, object]:
    value = {field.name: None for field in fields(Measurement)}
    value.update({"scope_id": scope_id, "total_wall_s": total_wall_s})
    return value


def _validation() -> dict[str, object]:
    return {
        "policy_reference_applicable": True,
        "policy_reference_passed": True,
        "full_precision_threshold_applicable": True,
        "full_precision_passed": True,
        "accuracy_qualified": True,
        "max_abs_error": 0.01,
        "relative_l2_error": 0.02,
        "norm_drift": 0.001,
        "phase_aligned_max_abs_error": 0.01,
    }


def _sample(
    *,
    run_id: str,
    experiment_id: str,
    environment_id: str,
    validation_policy_id: str,
    case_id: str,
    route_id: str,
    plan_id: str | None = None,
    index: int,
    total_wall_s: float | None,
    scope_id: str = "steady_execution_v1",
    facts: dict[str, object] | None = None,
    session_instance_id: str | None = None,
    status: str = "success",
    sample_kind: str = "measurement",
) -> dict[str, object]:
    sample: dict[str, object] = {
        "schema_version": "evidence_sample_v4",
        "sample_id": sample_id(
            run_id,
            case_id,
            route_id,
            sample_kind,
            index,
            plan_id=plan_id,
            block_id=index,
        ),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "attempt_kind": sample_kind,
        "sample_index": index,
        "block_id": index,
        "order_index": 0,
        "observed_affinity": [0],
        "background_load_1m": 0.0,
        "session_instance_id": session_instance_id,
        "status": status,
        "identities": {
            "problem_id": "1" * 64,
            "tensor_network_structure_id": "2" * 64,
            "logical_plan_id": "3" * 64,
            "physical_plan_id": "4" * 64 if session_instance_id else None,
            "executable_id": "5" * 64 if session_instance_id else None,
            "environment_id": environment_id,
            "validation_policy_id": validation_policy_id,
        },
        "measurement": None,
        "backend_facts": facts or {"backend_id": "numpy_cpu_v1"},
        "numeric_facts": {"numeric_policy": "split_complex_float32_v1"},
        "output_sha256": None,
        "validation": None,
        "failure": None,
    }
    if status == "success":
        assert total_wall_s is not None
        sample["measurement"] = _measurement(scope_id, total_wall_s)
        sample["output_sha256"] = "b" * 64
        sample["validation"] = _validation()
    else:
        sample["failure"] = {"stage": "kernel", "reason": "failed"}
    return sample


def _session(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    instance: str,
    plan_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "evidence_session_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "session_instance_id": instance,
        "session_protocol_id": "upmem_real_tile_v4",
        "open_s": 0.1,
        "session_close_s": 0.1,
        "status": "success",
        "terminal_backend_facts": {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "cpu_fallback_used": False,
            "simulator_kernel_executed": False,
            "requested_dpus": 2,
            "allocated_dpus": 2,
            "active_dpus": 2,
        },
        "release_attempted": True,
        "release_succeeded": True,
        "release_verified": True,
        "failure": None,
    }


_ENVIRONMENT = {"host": "test-host", "os": "test-os"}
_VALIDATION_POLICY = {"reference_dtype": "complex128", "atol": 1.0e-5}
_COLLECTION_POLICY = {"base_seed": 7, "session_policy": "test"}
_PHYSICAL_COLLECTION_POLICY = {
    "claim_policy": "physical_performance_v1",
    "base_seed": 7,
    "warmup_blocks": 2,
    "measurement_blocks": 30,
    "session_policy": "fresh_session_per_attempt_v1",
    "block_cooldown_s": 0.0,
    "machine_policy": {},
}


def _physical_attempts(
    *,
    run_id: str,
    experiment_id: str,
    environment_id: str,
    validation_policy_id: str,
    case_id: str,
    route_id: str,
    total_wall_s: float,
    facts: dict[str, object] | None = None,
    session_instance_id: str | None = None,
    scope_id: str = "steady_execution_v1",
) -> list[dict[str, object]]:
    route_facts = dict(facts or {})
    if session_instance_id is not None:
        route_facts.setdefault("startup_resource_admission_passed", True)
        route_facts.setdefault("execution_resource_admission_passed", True)
    return [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id,
            validation_policy_id=validation_policy_id,
            case_id=case_id,
            route_id=route_id,
            index=index,
            total_wall_s=total_wall_s,
            facts=route_facts,
            session_instance_id=session_instance_id,
            sample_kind=attempt_kind,
            scope_id=scope_id,
        )
        for attempt_kind, count in (("warmup", 2), ("measurement", 30))
        for index in range(count)
    ]


def _artifact(
    directory: Path,
    samples: list[dict[str, object]],
    sessions: list[dict[str, object]] | None = None,
    *,
    status: str = "completed",
    source_worktree_dirty: bool = False,
    collection_policy: dict[str, object] | None = None,
    machine_preflight_passed: bool = False,
) -> Path:
    sessions = sessions or []
    environment = {
        **_ENVIRONMENT,
        **(
            {
                "machine_preflight": {
                    "machine_preflight_passed": machine_preflight_passed
                }
            }
            if collection_policy is not None
            else {}
        ),
    }
    environment_id_value = environment_id(environment)
    for sample in samples:
        identities = sample["identities"]
        if isinstance(identities, dict):
            identities["environment_id"] = environment_id_value
    run_id = str(samples[0]["run_id"])
    experiment_id = str(samples[0]["experiment_id"])
    policy_id = str(samples[0]["identities"]["validation_policy_id"])
    bindings = {
        (
            str(sample["case_id"]),
            sample["plan_id"],
            str(sample["route_id"]),
        ): {
            "case_id": sample["case_id"],
            "plan_id": sample["plan_id"],
            "route_id": sample["route_id"],
            **sample["identities"],
        }
        for sample in samples
    }
    manifest = {
        "schema_version": "evidence_manifest_v2",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "collection_policy_id": collection_policy_id(
            collection_policy or _COLLECTION_POLICY
        ),
        "environment_id": environment_id_value,
        "validation_policy_id": policy_id,
        "created_at_utc": "2026-08-24T12:00:00Z",
        "source_commit": "a" * 40,
        "source_worktree_dirty": source_worktree_dirty,
        "configuration": {
            "experiment": {
                "experiment_id": experiment_id,
                **(
                    {"collection": collection_policy}
                    if collection_policy is not None
                    else {}
                ),
            },
            "environment": environment,
            "validation_policy": _VALIDATION_POLICY,
            "identity_bindings": [
                bindings[key]
                for key in sorted(
                    bindings,
                    key=lambda key: (
                        key[0],
                        "" if key[1] is None else str(key[1]),
                        key[2],
                    ),
                )
            ],
        },
        "expected_counts": {
            "warmup": sum(sample["attempt_kind"] == "warmup" for sample in samples),
            "measurement": sum(
                sample["attempt_kind"] == "measurement" for sample in samples
            ),
            "sessions": len(sessions),
        },
        "files": {
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
            "sessions": "sessions.jsonl",
        },
        "status": "running",
    }
    write_manifest(directory / "manifest.json", manifest)
    for sample in samples:
        append_sample(directory / "samples.jsonl", sample)
    for session in sessions:
        append_session(directory / "sessions.jsonl", session)
    if not samples:
        (directory / "samples.jsonl").write_text("", encoding="utf-8")
    if not sessions:
        (directory / "sessions.jsonl").write_text("", encoding="utf-8")
    finalize_artifacts(directory, status=status)
    return directory


def _ids() -> tuple[str, str, str, str]:
    return (
        str(uuid4()),
        "e" * 64,
        environment_id(_ENVIRONMENT),
        validation_policy_id(_VALIDATION_POLICY),
    )


def test_load_rejects_noncanonical_reencoding(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id_value,
                validation_policy_id=policy_id,
                case_id="case",
                route_id="cpu",
                index=0,
                total_wall_s=1.0,
            )
        ],
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_artifacts(artifact)


def test_load_requires_all_primary_files(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    directory.mkdir()

    with pytest.raises(ValueError, match="missing required evidence file"):
        load_artifacts(directory)


def test_load_binds_manifest_identity_payloads(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id_value,
                validation_policy_id=policy_id,
                case_id="case",
                route_id="cpu",
                index=0,
                total_wall_s=1.0,
            )
        ],
    )
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["environment"]["host"] = "other-host"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment_id"):
        load_artifacts(artifact)


def test_verify_and_report_aggregate_duplicate_measurements_once(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id_value,
                validation_policy_id=policy_id,
                case_id="bell",
                route_id="cpu",
                index=index,
                total_wall_s=float(index + 1),
            )
            for index in range(2)
        ],
    )

    assert verify_artifacts(artifact)["success_count"] == 2
    report = report_artifacts(artifact, tmp_path / "report")

    assert report["status"] == "completed"
    assert report["schema_version"] == "evidence_report_v5"
    assert report["aggregate_count"] == 1
    assert report["statistics"] == {
        "summary": "median_raw_mad_v1",
        "confidence_interval": "percentile_bootstrap_95_v1",
        "confidence_level": 0.95,
        "resample_count": 10_000,
        "speedup_method": "block_paired_median_ratio_bootstrap_v1",
        "outlier_policy": "no_post_hoc_exclusion_v1",
    }
    rows = (
        (tmp_path / "report" / "aggregate.csv").read_text(encoding="utf-8").splitlines()
    )
    assert len(rows) == 2
    assert (tmp_path / "report" / "plots" / "runtime_steady_execution_v1.png").is_file()


def test_report_excludes_sdk_simulator_speedup(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="simulator",
            index=index,
            total_wall_s=0.5,
            facts={
                "backend_id": "upmem_sdk_simulator_v4",
                "target_observed": "sdk_simulator",
                "simulator_kernel_executed": True,
            },
        )
        for index in range(2)
    )
    report = report_artifacts(
        _artifact(tmp_path / "evidence", samples), tmp_path / "report"
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["simulator_execution"] == 1
    assert report["simulator_timing"]["present"] is True


def test_report_ignores_non_upmem_routes_for_speedup(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id=route,
            index=index,
            total_wall_s=2.0 if route == "cpu" else 1.0,
            facts=(
                {"backend_id": "numpy_cpu_v1"}
                if route == "cpu"
                else {
                    "backend_id": "quimb_tn_v1",
                    "backend_family": "quimb",
                    "hardware_execution": False,
                }
            ),
        )
        for route in ("cpu", "quimb")
        for index in range(2)
    ]

    report = report_artifacts(
        _artifact(tmp_path / "evidence", samples),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"] == {}
    assert report["simulator_timing"]["present"] is False


def test_report_admits_physical_speedup_from_terminal_session_facts(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    session_instance_id = "physical-session"
    samples = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        total_wall_s=3.0,
    )
    samples.extend(
        _physical_attempts(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            total_wall_s=1.0,
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=session_instance_id,
        )
    )
    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=session_instance_id,
                )
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 1
    with (tmp_path / "report" / "speedups.csv").open(newline="", encoding="utf-8") as stream:
        speedup_rows = list(csv.DictReader(stream))
    assert len(speedup_rows) == 1
    assert speedup_rows[0]["speedup"] == "3.0"
    assert speedup_rows[0]["complete_pair_count"] == "30"
    assert speedup_rows[0]["bootstrap_method"] == "block_paired_median_ratio_bootstrap_v1"
    assert float(speedup_rows[0]["speedup_ci_low"]) <= 3.0
    assert float(speedup_rows[0]["speedup_ci_high"]) >= 3.0
    assert (tmp_path / "report" / "plots" / "physical_speedup_by_case.png").is_file()


def test_report_emits_claim_gated_tasklet_scaling_csv(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    baseline_instance = "physical-t1"
    candidate_instance = "physical-t8"
    common_facts = {
        "backend_id": "upmem_sdk_hardware_v4",
        "kernel_policy": "dpu_real_tile_v4_wram_panel_v1",
        "requested_dpus": 1,
        "allocated_dpus": 1,
        "active_dpus": 1,
        "dominant_work_wave": 0,
        "dominant_work_wave_arithmetic_work": 4096,
        "dominant_work_wave_populated_dpu_slots": 1,
        "dominant_work_wave_allocated_dpu_slots": 1,
        "dominant_work_wave_utilization": 1.0,
        "arithmetic_weighted_dpu_slot_utilization": 1.0,
        "arithmetic_weighted_tasklet_utilization": 1.0,
        "fully_populated_wave_count": 1,
    }
    baseline = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="upmem_t1",
        total_wall_s=8.0,
        facts={**common_facts, "tasklets_per_dpu": 1},
        session_instance_id=baseline_instance,
    )
    candidate = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="upmem_t8",
        total_wall_s=2.0,
        facts={**common_facts, "tasklets_per_dpu": 8},
        session_instance_id=candidate_instance,
    )
    for sample in baseline:
        sample["identities"]["physical_plan_id"] = "4" * 64
        sample["identities"]["executable_id"] = "5" * 64
    for sample in candidate:
        sample["identities"]["physical_plan_id"] = "6" * 64
        sample["identities"]["executable_id"] = "7" * 64

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            [*baseline, *candidate],
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem_t1",
                    instance=baseline_instance,
                ),
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem_t8",
                    instance=candidate_instance,
                ),
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["schema_version"] == "evidence_report_v5"
    assert report["scaling_count"] == 1
    with (tmp_path / "report" / "scaling.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_id"] == experiment_id
    assert row["comparison_kind"] == "tasklet_scaling"
    assert row["comparison_role"] == "primary"
    assert row["case_id"] == "bell"
    assert row["plan_id"] == ""
    assert row["scope_id"] == "steady_execution_v1"
    assert row["problem_id"] == "1" * 64
    assert row["tensor_network_structure_id"] == "2" * 64
    assert row["logical_plan_id"] == "3" * 64
    assert row["numeric_policy"] == "split_complex_float32_v1"
    assert row["kernel_policy"] == "dpu_real_tile_v4_wram_panel_v1"
    assert row["validation_policy_id"] == policy_id
    assert row["baseline_route_id"] == "upmem_t1"
    assert row["candidate_route_id"] == "upmem_t8"
    assert row["baseline_physical_plan_id"] == "4" * 64
    assert row["candidate_physical_plan_id"] == "6" * 64
    assert row["baseline_executable_id"] == "5" * 64
    assert row["candidate_executable_id"] == "7" * 64
    assert row["baseline_tasklet_count"] == "1"
    assert row["candidate_tasklet_count"] == "8"
    assert row["resource_ratio"] == "8.0"
    assert row["planned_pair_count"] == "30"
    assert row["complete_pair_count"] == "30"
    assert row["speedup"] == "4.0"
    assert row["parallel_efficiency"] == "0.5"
    assert row["claim_eligible"] == "True"
    assert row["claim_ineligibility_reason"] == ""
    assert row["baseline_dominant_work_wave"] == "0"
    assert row["candidate_dominant_work_wave"] == "0"
    assert row["candidate_dominant_work_wave_arithmetic_work"] == "4096"
    assert row["candidate_dominant_work_wave_utilization"] == "1.0"
    assert row["candidate_arithmetic_weighted_dpu_slot_utilization"] == "1.0"
    assert row["candidate_arithmetic_weighted_tasklet_utilization"] == "1.0"
    assert row["candidate_fully_populated_wave_count"] == "1"


def test_report_emits_claim_gated_dpu_scaling_csv(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    baseline_instance = "physical-1dpu-t8"
    candidate_instance = "physical-4dpu-t8"
    common_facts = {
        "backend_id": "upmem_sdk_hardware_v4",
        "kernel_policy": "dpu_real_tile_v4_wram_panel_v1",
        "tasklets_per_dpu": 8,
        "dominant_work_wave": 0,
        "dominant_work_wave_arithmetic_work": 4096,
        "dominant_work_wave_utilization": 1.0,
        "arithmetic_weighted_tasklet_utilization": 1.0,
        "fully_populated_wave_count": 1,
    }
    baseline = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="upmem_1dpu_t8",
        total_wall_s=8.0,
        facts={
            **common_facts,
            "requested_dpus": 1,
            "allocated_dpus": 1,
            "active_dpus": 1,
            "dominant_work_wave_populated_dpu_slots": 1,
            "dominant_work_wave_allocated_dpu_slots": 1,
            "arithmetic_weighted_dpu_slot_utilization": 1.0,
        },
        session_instance_id=baseline_instance,
    )
    candidate = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="upmem_4dpu_t8",
        total_wall_s=4.0,
        facts={
            **common_facts,
            "requested_dpus": 4,
            "allocated_dpus": 4,
            "active_dpus": 4,
            "dominant_work_wave_populated_dpu_slots": 4,
            "dominant_work_wave_allocated_dpu_slots": 4,
            "arithmetic_weighted_dpu_slot_utilization": 0.98,
        },
        session_instance_id=candidate_instance,
    )
    for sample in baseline:
        sample["identities"]["physical_plan_id"] = "4" * 64
        sample["identities"]["executable_id"] = "5" * 64
    for sample in candidate:
        sample["identities"]["physical_plan_id"] = "6" * 64
        sample["identities"]["executable_id"] = "5" * 64

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            [*baseline, *candidate],
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem_1dpu_t8",
                    instance=baseline_instance,
                ),
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem_4dpu_t8",
                    instance=candidate_instance,
                ),
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["scaling_count"] == 1
    with (tmp_path / "report" / "scaling.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        row = next(csv.DictReader(stream))
    assert row["comparison_kind"] == "dpu_scaling"
    assert row["comparison_role"] == "primary"
    assert row["baseline_dpu_count"] == "1"
    assert row["candidate_dpu_count"] == "4"
    assert row["baseline_tasklet_count"] == "8"
    assert row["candidate_tasklet_count"] == "8"
    assert row["baseline_executable_id"] == row["candidate_executable_id"]
    assert row["resource_ratio"] == "4.0"
    assert row["speedup"] == "2.0"
    assert row["parallel_efficiency"] == "0.5"
    assert row["claim_eligible"] == "True"
    assert row["candidate_arithmetic_weighted_dpu_slot_utilization"] == "0.98"


def test_report_emits_validation_metrics_mad_and_unqualified_labels(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu_float32",
            index=index,
            total_wall_s=value,
        )
        for index, value in enumerate((1.0, 3.0, 5.0))
    ]
    samples.extend(
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu_int8",
            index=index,
            total_wall_s=value,
        )
        for index, value in enumerate((2.0, 4.0, 6.0))
    )
    for sample in samples:
        if sample["route_id"] != "cpu_int8":
            continue
        sample["numeric_facts"] = {
            "numeric_policy": "split_complex_int8_shared_scale_v1"
        }
        sample["validation"] = {
            **_validation(),
            "full_precision_threshold_applicable": False,
            "full_precision_passed": None,
            "accuracy_qualified": False,
        }

    report = report_artifacts(
        _artifact(tmp_path / "evidence", samples), tmp_path / "report"
    )

    with (tmp_path / "report" / "aggregate.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = {row["route_id"]: row for row in csv.DictReader(stream)}
    float32 = rows["cpu_float32"]
    int8 = rows["cpu_int8"]
    assert float32["median_total_wall_s"] == "3.0"
    assert float32["mad_total_wall_s"] == "2.0"
    assert float32["median_max_abs_error"] == "0.01"
    assert float32["median_relative_l2_error"] == "0.02"
    assert float32["median_norm_drift"] == "0.001"
    assert float32["median_phase_aligned_max_abs_error"] == "0.01"
    assert float(float32["median_total_wall_ci_low_s"]) <= 3.0
    assert float(float32["median_total_wall_ci_high_s"]) >= 3.0
    assert float32["accuracy_qualified"] == "True"
    assert float32["claim_eligible"] == "False"
    assert int8["accuracy_qualified"] == "False"
    assert int8["claim_eligible"] == "False"
    assert int8["claim_ineligibility_reason"] == "diagnostic_claim_policy"
    assert report["qualification"] == {
        "accuracy_qualified_aggregate_count": 1,
        "accuracy_unqualified_aggregate_count": 1,
        "claim_eligible_aggregate_count": 0,
    }
    assert (tmp_path / "report" / "plots" / "runtime_steady_execution_v1.png").is_file()


def test_report_rejects_scope_mismatch(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    session_instance_id = "physical-session"
    samples = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        total_wall_s=2.0,
    )
    samples.extend(
        _physical_attempts(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            total_wall_s=1.0,
            scope_id="simulation_end_to_end_v1",
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=session_instance_id,
        )
    )
    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=session_instance_id,
                )
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["timing_scope_mismatch"] == 1


def test_report_rejects_speedup_without_full_precision_threshold(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    instance = "physical-session"
    samples = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        total_wall_s=2.0,
        facts={"backend_id": "numpy_cpu_v1"},
    )
    samples.extend(
        _physical_attempts(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            total_wall_s=1.0,
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=instance,
        )
    )
    for sample in samples:
        sample["numeric_facts"] = {
            "numeric_policy": "split_complex_int8_shared_scale_v1"
        }
        sample["validation"] = {
            **_validation(),
            "full_precision_threshold_applicable": False,
            "full_precision_passed": None,
            "accuracy_qualified": False,
        }

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=instance,
                )
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["candidate_accuracy_unqualified"] == 1


def test_report_rejects_speedup_without_applicable_policy_reference(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    instance = "physical-session"
    samples = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        total_wall_s=2.0,
        facts={"backend_id": "numpy_cpu_v1"},
    )
    samples.extend(
        _physical_attempts(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            total_wall_s=1.0,
            facts={"backend_id": "upmem_sdk_hardware_v4"},
            session_instance_id=instance,
        )
    )
    for sample in samples:
        if sample["route_id"] == "upmem":
            sample["validation"] = {
                **_validation(),
                "policy_reference_applicable": False,
                "policy_reference_passed": None,
            }

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [
                _session(
                    run_id=run_id,
                    experiment_id=experiment_id,
                    case_id="bell",
                    route_id="upmem",
                    instance=instance,
                )
            ],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["candidate_validation_failed"] == 1


def test_verify_counts_warmup_validation_but_claims_use_measurements(
    tmp_path: Path,
) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    warmup = _sample(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        index=0,
        total_wall_s=1.0,
        sample_kind="warmup",
    )
    warmup["validation"] = {
        **_validation(),
        "policy_reference_passed": False,
        "full_precision_passed": False,
        "accuracy_qualified": False,
    }
    measurement = _sample(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        index=0,
        total_wall_s=1.0,
    )
    artifact = _artifact(tmp_path / "evidence", [warmup, measurement])

    verification = verify_artifacts(artifact)

    assert verification["policy_reference_applicable_count"] == 2
    assert verification["policy_reference_failure_count"] == 1
    assert verification["accuracy_qualified_count"] == 1
    assert verification["accuracy_unqualified_count"] == 1
    assert verification["policy_reference_qualified"] is True
    assert verification["accuracy_qualified"] is True


def test_verify_unvalidated_measurement_prevents_qualification(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=1.0,
        )
        for index in range(2)
    ]
    samples[1]["validation"] = None
    artifact = _artifact(tmp_path / "evidence", samples)

    verification = verify_artifacts(artifact)

    assert verification["policy_reference_applicable_count"] == 1
    assert verification["policy_reference_failure_count"] == 0
    assert verification["accuracy_qualified_count"] == 1
    assert verification["accuracy_unqualified_count"] == 0
    assert verification["policy_reference_qualified"] is False
    assert verification["accuracy_qualified"] is False


def test_report_rejects_claims_from_failed_or_dirty_artifacts(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    samples = [
        _sample(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="cpu",
            index=index,
            total_wall_s=2.0,
        )
        for index in range(2)
    ]

    failed = report_artifacts(
        _artifact(tmp_path / "failed", samples, status="failed"),
        tmp_path / "failed-report",
    )
    dirty = report_artifacts(
        _artifact(
            tmp_path / "dirty",
            samples,
            source_worktree_dirty=True,
        ),
        tmp_path / "dirty-report",
    )

    assert failed["speedup_rejections"] == {"artifact_not_completed": 1}
    assert dirty["speedup_rejections"] == {"source_worktree_dirty": 1}


def test_report_rejects_conflicting_terminal_physical_facts(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    instance = "physical-session"
    samples = _physical_attempts(
        run_id=run_id,
        experiment_id=experiment_id,
        environment_id=environment_id_value,
        validation_policy_id=policy_id,
        case_id="bell",
        route_id="cpu",
        total_wall_s=2.0,
    )
    samples.extend(
        _physical_attempts(
            run_id=run_id,
            experiment_id=experiment_id,
            environment_id=environment_id_value,
            validation_policy_id=policy_id,
            case_id="bell",
            route_id="upmem",
            total_wall_s=1.0,
            facts={
                "backend_id": "upmem_sdk_hardware_v4",
                "target_observed": "physical_hardware",
            },
            session_instance_id=instance,
        )
    )
    session = _session(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id="bell",
        route_id="upmem",
        instance=instance,
    )
    session["terminal_backend_facts"]["target_observed"] = "sdk_simulator"

    report = report_artifacts(
        _artifact(
            tmp_path / "evidence",
            samples,
            [session],
            collection_policy=_PHYSICAL_COLLECTION_POLICY,
            machine_preflight_passed=True,
        ),
        tmp_path / "report",
    )

    assert report["speedup_count"] == 0
    assert report["speedup_rejections"]["terminal_fact_conflict"] == 1


def test_failed_artifact_generates_diagnostic_report(tmp_path: Path) -> None:
    run_id, experiment_id, environment_id_value, policy_id = _ids()
    artifact = _artifact(
        tmp_path / "evidence",
        [
            _sample(
                run_id=run_id,
                experiment_id=experiment_id,
                environment_id=environment_id_value,
                validation_policy_id=policy_id,
                case_id="bell",
                route_id="cpu",
                index=0,
                total_wall_s=None,
                status="failed",
            )
        ],
        status="failed",
    )

    report = report_artifacts(artifact, tmp_path / "report")

    assert report["failed_count"] == 1
    assert report["aggregate_count"] == 0
    assert (tmp_path / "report" / "plots").is_dir()


def test_plot_key_uniqueness_and_png_smoke(tmp_path: Path) -> None:
    point = {
        "figure_id": "figure",
        "facet_id": "facet",
        "series_id": "series",
        "series_label": "Series",
        "x_value": "case",
        "x_label": "Case",
        "value": 1.0,
    }
    with pytest.raises(ValueError, match="duplicate plot point"):
        _assert_unique_plot_points([point, dict(point)])

    _plot_grouped_bars(
        tmp_path / "plot.png",
        [point],
        title="Smoke",
        ylabel="Value",
    )
    assert (tmp_path / "plot.png").is_file()


def test_plot_series_uses_readable_plan_dimension() -> None:
    base = {
        "route_id": "upmem",
        "numeric_policy": "split_complex_float32_v1",
        "case_id": "circuit",
    }
    first = _point(
        figure_id="runtime",
        facet_id="steady",
        row={**base, "plan_id": "greedy"},
        value=1.0,
    )
    second = _point(
        figure_id="runtime",
        facet_id="steady",
        row={**base, "plan_id": "cotengra"},
        value=2.0,
    )

    _assert_unique_plot_points([first, second])
    assert first["series_id"] != second["series_id"]
    assert first["series_label"].startswith("Greedy |")
