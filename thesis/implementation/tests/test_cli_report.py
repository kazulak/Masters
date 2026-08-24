from __future__ import annotations

from pathlib import Path
import json

import pytest

import quantum_bench.cli as cli
from quantum_bench.evidence import canonical_json, load_artifacts
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


def test_loader_allows_distinct_plan_occurrences_but_rejects_exact_duplicates(
    tmp_path: Path,
) -> None:
    distinct = tmp_path / "distinct_plans.yml"
    distinct_text = (
        _config()
        .replace(
            "    slicing: null\nroutes:",
            "    slicing: null\n  p2:\n    planner:\n      engine: opt_einsum\n"
            "      mode: optimal\n      max_repeats: 1\n      seed: 0\n"
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


def _numpy_config(*, warmups: int = 0, repetitions: int = 1) -> str:
    return f"""\
schema_version: tn_benchmark_v1
experiment_id: cli-focused
defaults:
  warmups: {warmups}
  repetitions: {repetitions}
  timeout_s: 2.5
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
      max_repeats: 1
      seed: 0
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
            "      mode: optimal\n      max_repeats: 1\n      seed: 0\n"
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
    assert [(row["sample_kind"], row["sample_index"]) for row in samples] == [
        ("warmup", 0),
        ("measurement", 0),
        ("measurement", 1),
    ]
    assert {row["plan_id"] for row in samples} == {"greedy"}


def test_run_keeps_same_case_route_samples_separate_by_plan_id(tmp_path: Path) -> None:
    config = tmp_path / "two-plans.yml"
    _write_config(config, _two_plan_numpy_config())

    result = cli.run_command(str(config), str(tmp_path / "run"), allow_physical=False)

    assert result["status"] == "completed"
    samples = [
        json.loads(line)
        for line in (tmp_path / "run" / "samples.jsonl").read_text().splitlines()
    ]
    assert [
        (sample["case_id"], sample["plan_id"], sample["route_id"]) for sample in samples
    ] == [
        ("bell", "greedy", "numpy"),
        ("bell", "optimal", "numpy"),
    ]
    assert samples[0]["sample_id"] != samples[1]["sample_id"]


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
            "max_repeats": 1,
            "seed": 0,
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
