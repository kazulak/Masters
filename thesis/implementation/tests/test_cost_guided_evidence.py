"""Canonical synthetic raw evidence; no validator mocks or executor calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

from quantum_bench import evidence, experiment
from quantum_bench.upmem.path_heuristic import LaunchCostFacts, validate_launch_cost_observations
import upmem_cost_guided_evidence as extractor


ROOT = Path(__file__).resolve().parents[1]


def _test_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


packet_fixture = _test_module("test_cost_guided_packet")
pretest = _test_module("test_upmem_wave_pretest_extraction")
wave = extractor.wave


def _checksums(archive):
    (archive / "SHA256SUMS").write_text("".join(
        f"{wave._file_sha256(p)}  {p.relative_to(archive)}\n"
        for p in sorted(archive.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"
    ))


def _save_raw(scenario):
    root = scenario["root"]
    pretest._write_canonical(root / "manifest.json", scenario["raw_manifest"])
    pretest._write_jsonl(root / "samples.jsonl", scenario["samples"])
    pretest._write_jsonl(root / "sessions.jsonl", scenario["sessions"])


def _fixture(tmp_path, stage="initial", *, definition=None, skipped=False, collection_changes=None):
    definition = definition or {"kind": "builtin", "name": "quantization_stress",
                                "parameters": {"n_qubits": 6, "repeat_layers": 1}}
    round_manifest, workload = packet_fixture._packet(definition)
    if stage.startswith("feedback_"):
        qualifier = packet_fixture.qualifier
        job = qualifier.make_simulation_job(qualifier._circuit_from_definition(definition))
        network, _ = qualifier.lower_tensor_network(job)
        for topology_id, cell in round_manifest["cells"].items():
            greedy = cell["selection"]["roles"]["G"]
            candidate = deepcopy(cell["candidates"][greedy])
            # At three active operands, a different penultimate pair gives a
            # distinct complete replay without running another path search.
            candidate["path"][-2] = [0, 2] if candidate["path"][-2] == [0, 1] else [0, 1]
            identifier = qualifier.path_id(candidate["path"], circuit_id=cell["circuit_id"])
            candidate["path_id"] = identifier
            dag = qualifier.build_contraction_dag(network, candidate["path"])
            plan = qualifier.plan_upmem(
                dag, numeric_policy=qualifier.FLOAT32, schedule_policy="static_dag_waves_v1",
                topology=qualifier.UpmemTopology(dpu_count=1 if topology_id.endswith("1dpu_t8") else 4,
                                                rank_count=1, tasklets_per_dpu=8),
            )
            facts = extractor.extract_launch_cost_features(dag, plan)
            execution = facts.pop("execution")
            facts.update(logical_plan_id=qualifier.contraction_dag_hash(dag),
                         physical_plan_id=qualifier.physical_plan_id(plan),
                         resource_admission=qualifier.collection_resource_admission(plan),
                         declared_host_bytes=execution["host_buffers"]["declared_executor_memory_estimate_bytes"])
            candidate["facts"] = facts
            cell["candidates"][identifier] = candidate
            cell["selection"] = {"path_ids": sorted([greedy, identifier]),
                                 "roles": {"G": greedy, "F": greedy, "R": identifier}}
        round_manifest["expected_attempts"] = 16
    config, provenance = packet_fixture.qualifier.prepare_cost_guided_config(
        round_manifest, workload, execution_root=ROOT, experiment_id="cost-evidence-fixture",
    )
    measured = 5 if stage == "evaluation" else 3
    round_manifest["stage"] = stage
    round_manifest["measurement_blocks"] = list(range(1, measured + 1))
    if skipped:
        round_manifest["cells"].pop("stress6/4dpu_t8")
        round_manifest["skipped_cells"] = {
            "stress6/4dpu_t8": {"reason": "no_new_eligible_candidate", "trace_hash": "a" * 64},
        }
        config["routes"].pop("4dpu_t8")
        for row in config["matrix"]:
            row["route_ids"] = ["1dpu_t8"]
    round_manifest["expected_attempts"] = sum(len(cell["candidates"]) for cell in round_manifest["cells"].values()) * (measured + 1)
    if stage.startswith("feedback_"):
        for cell in round_manifest["cells"].values():
            roles = cell["selection"]["roles"]
            cell["selection"]["roles"] = {"G": roles["G"], "new_best": roles["R"]}
    if stage == "evaluation":
        workload["instances"] = workload["instances"][:1]
        workload["instances"][0]["split"] = "test"
        for cell in round_manifest["cells"].values():
            cell["split"] = "test"
            cell["selection"]["roles"]["U"] = cell["selection"]["roles"]["G"]
    config["collection"] = packet_fixture.qualifier._collection(
        warmups=1, measurements=measured,
        seed=20260910 + ("initial", "feedback_1", "feedback_2", "evaluation").index(stage),
    )
    if collection_changes:
        config["collection"].update(collection_changes)
    archive = tmp_path / "preregistration"
    archive.mkdir()
    for source in provenance["qasm_source_bindings"]:
        target = archive / source["prepared_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source["source_path"]).read_bytes())
    physical = archive / "physical.yml"
    physical.write_text(yaml.safe_dump(config))
    normalized = json.loads(evidence.canonical_json(experiment.load_experiment_config(physical)))
    study = json.loads((ROOT / "configs/upmem_cost_guided_path_study_v1.json").read_text())
    binding = {"source_sha": "f" * 40, "executor_source": study["executor"]["source"],
               "study_hash": extractor._hash(study), "workload_hash": study["workload"]["sha256"]}
    round_manifest["binding_hash"] = extractor._hash(binding)
    provenance.update({
        "round_manifest_hash": extractor._hash(round_manifest), "binding_hash": extractor._hash(binding),
        "source_sha": binding["source_sha"], "workload_record_sha256": extractor._hash(workload),
        "execution_source": study["executor"]["source"],
        "configuration_sha256": wave._file_sha256(physical),
        "normalized_configuration_sha256": wave._sha256_bytes(wave._canonical_bytes(normalized)),
    })
    (archive / "physical.yml.provenance.json").write_bytes(wave._canonical_bytes(provenance))
    binaries = {options[field]: study["executor"]["binaries"][Path(options[field]).name]
                for route in normalized["routes"].values() for options in [route["options"]]
                for field in ("dpu_binary", "host_binary", "initialization_binary")}
    (archive / "binary_sha256.json").write_bytes(wave._canonical_bytes(binaries))
    _checksums(archive)
    template_root = tmp_path / "template"
    template_root.mkdir()
    _, _, _, _, templates, session_templates = pretest.existing_fixtures._wave_calibration_fixture(template_root)
    environment = {"host": "cost-guided-fixture", "requested_rank_paths": ["/dev/dpu_rank1"]}
    env_id, validation_id = evidence.environment_id(environment), experiment.default_validation_policy_id()
    bindings = []
    for cell in round_manifest["cells"].values():
        for identifier, candidate in cell["candidates"].items():
            bindings.append({
                "case_id": cell["circuit_id"], "plan_id": f"path_{identifier}", "route_id": cell["topology_id"],
                "problem_id": workload["instances"][0]["canonical_circuit"]["operation_identity"]["problem_id"],
                "tensor_network_structure_id": normalized["plans"][f"path_{identifier}"]["planner"]["tensor_network_structure_id"],
                "logical_plan_id": candidate["facts"]["logical_plan_id"],
                "physical_plan_id": candidate["facts"]["physical_plan_id"],
                "executable_id": extractor._executable_identity({
                    field: {"sha256": binaries[normalized["routes"][cell["topology_id"]]["options"][field]]}
                    for field in ("dpu_binary", "host_binary", "initialization_binary")
                }, study["executor"]), "environment_id": env_id, "validation_policy_id": validation_id,
            })
    bindings.sort(key=lambda row: (row["case_id"], row["plan_id"], row["route_id"]))
    raw_manifest = {
        "schema_version": "evidence_manifest_v2", "run_id": "00000000-0000-4000-8000-000000000001",
        "experiment_id": normalized["experiment_id"], "collection_policy_id": normalized["collection_policy_id"],
        "environment_id": env_id, "validation_policy_id": validation_id,
        "created_at_utc": "2026-09-09T00:00:00Z", "source_commit": binding["source_sha"],
        "source_worktree_dirty": False,
        "configuration": {"experiment": normalized, "environment": environment,
                          "validation_policy": dict(experiment.default_validation_policy()), "identity_bindings": bindings},
        "expected_counts": {"warmup": len(bindings), "measurement": len(bindings) * measured,
                            "sessions": len(bindings) * (measured + 1)},
        "files": {"manifest": "manifest.json", "samples": "samples.jsonl", "sessions": "sessions.jsonl"},
        "status": "completed",
    }
    samples, sessions = [], []
    for case, plan, route, attempt, sample_index, block, order in sorted(evidence._declared_collection_attempts(raw_manifest)):
        item = next(b for b in bindings if (b["case_id"], b["plan_id"], b["route_id"]) == (case, plan, route))
        candidate = round_manifest["cells"][f"{case}/{route}"]["candidates"][plan.removeprefix("path_")]
        dpu_count = 1 if route == "1dpu_t8" else 4
        backend, terminal = pretest._backend_facts(
            base_sample=templates[0], base_session=session_templates[0], dpu_count=dpu_count,
            candidate=candidate["facts"], topology=candidate["facts"],
            output_sha=pretest.existing_fixtures._fixture_runtime_facts_output_sha256(),
        )
        options = normalized["routes"][route]["options"]
        for field in ("dpu_binary", "host_binary", "initialization_binary"):
            terminal[f"{field}_path"] = options[field]
            terminal[f"{field}_sha256"] = binaries[options[field]]
        session_id = f"session-{len(samples)}"
        sample = deepcopy(templates[0])
        sample.update(
            schema_version="evidence_sample_v4", run_id=raw_manifest["run_id"],
            experiment_id=normalized["experiment_id"], case_id=case, plan_id=plan, route_id=route,
            attempt_kind=attempt, sample_index=sample_index, block_id=block, order_index=order,
            sample_id=evidence.sample_id(raw_manifest["run_id"], case, route, attempt, sample_index,
                                         plan_id=plan, block_id=block, order_index=order),
            session_instance_id=session_id, observed_affinity=[0], background_load_1m=None,
            identities={k: v for k, v in item.items() if k not in {"case_id", "plan_id", "route_id"}},
            backend_facts=backend,
        )
        sample["measurement"].update(scope_id="steady_execution_v1", total_wall_s=2.0 + block,
                                     session_open_s=0.0)
        sample["measurement"].update({field: None for field in (
            "decode_s", "encode_s", "energy_j", "host_reduce_s", "lowering_s",
            "mapping_s", "planning_s", "rank_work_s", "slicing_s",
        )})
        session = deepcopy(session_templates[0])
        session.update(schema_version="evidence_session_v1", run_id=raw_manifest["run_id"],
                       experiment_id=normalized["experiment_id"], case_id=case, plan_id=plan, route_id=route,
                       session_instance_id=session_id, open_s=0.0, session_close_s=0.5,
                       terminal_backend_facts=terminal)
        samples.append(sample)
        sessions.append(session)
    root = tmp_path / "raw"
    root.mkdir()
    scenario = dict(root=root, round_manifest=round_manifest, workload=workload, binding=binding,
                    study=study, samples=samples, sessions=sessions, raw_manifest=raw_manifest)
    _save_raw(scenario)
    (tmp_path / "terminal.json").write_text(json.dumps({
        "physical_stage_elapsed_s": 100.0, "physical_stage_elapsed_clock": "monotonic_duration",
        "source": binding["source_sha"], "worktree_status": "", "terminal_inspection_valid": True,
        "rank1_released": True, "rank_ownership_at_finally": {"dpu_rank1": "0"},
        "terminal_inspection_errors": [], "retries": 0, "replacements": 0,
        "results": {
            "physical": {"returncode": 0, "timed_out": False, "elapsed_s": 100.0,
                         "planned_attempts": len(samples), "actual_sample_rows": len(samples)},
            "canonical": {"returncode": 0, "timed_out": False},
        },
    }))
    return scenario


def _extract(s):
    return extractor.extract_round_observations(s["root"], s["round_manifest"], s["workload"], s["binding"], s["study"])


@pytest.mark.parametrize("stage", ["initial", "feedback_1", "feedback_2", "evaluation"])
def test_complete_round_immutable_relocatable(stage, tmp_path):
    s = _fixture(tmp_path, stage)
    before = deepcopy(s)
    result = _extract(s)
    assert s == before
    assert len(result["rows"]) == (12 if stage == "evaluation" else 16 if stage.startswith("feedback_") else 8)
    assert result["canonical_report"]["success_count"] == len(result["rows"])
    assert result["copy_acceptance"] == "caller_pending"
    assert result["fit_eligible_split"] is (stage != "evaluation")
    for row in result["rows"]:
        assert row["round_id"] == stage
        assert row["split"] == ("test" if stage == "evaluation" else "training")
        assert row["session_inclusive_s"] == row["total_wall_s"] + 0.5
        assert row["identity"] == result["identity"]
        assert row["identity"]["source_sha"] == "f" * 40
        assert row["identity"]["execution_source"] == wave.EXECUTION_SOURCE
        assert row["experiment_output_sha256"] != row["runtime_output_sha256"]
        assert row["execution_resource_admission_passed"] is True
        assert row["startup_resource_admission_passed"] is True
        candidate = s["round_manifest"]["cells"][row["cell_id"]]["candidates"][row["path_id"]]
        for field in ("collection_resource_admission_passed", "tasklet_row_sufficiency_passed"):
            assert row[field] is candidate["facts"]["resource_admission"][field]
    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path / "raw", relocated / "raw")
    shutil.copytree(tmp_path / "preregistration", relocated / "preregistration")
    shutil.copy2(tmp_path / "terminal.json", relocated / "terminal.json")
    s["root"] = relocated / "raw"
    assert _extract(s)["rows"] == result["rows"]


@pytest.mark.parametrize("mutation", [
    "missing_attempt", "extra_attempt", "missing_session", "reused_session", "block", "source", "dirty",
    "failed", "fallback", "release", "execution", "startup", "accuracy", "replay", "scope",
    "digest", "runtime_digest", "logical", "physical", "binary", "collection", "zero_time",
    "provenance", "sidecar_executor", "private_checksum", "expected_attempts", "split", "path", "profile",
])
def test_corruption_rejected(tmp_path, mutation):
    s = _fixture(tmp_path)
    sample, session = s["samples"][0], s["sessions"][0]
    if mutation == "missing_attempt":
        s["samples"].pop()
    elif mutation == "extra_attempt":
        s["samples"].append(deepcopy(sample))
    elif mutation == "missing_session":
        s["sessions"].pop()
    elif mutation == "reused_session":
        s["samples"][1]["session_instance_id"] = sample["session_instance_id"]
    elif mutation == "block":
        sample["block_id"] = 6
    elif mutation == "source":
        s["raw_manifest"]["source_commit"] = wave.EXECUTION_SOURCE
    elif mutation == "dirty":
        s["raw_manifest"]["source_worktree_dirty"] = True
    elif mutation == "failed":
        sample["status"] = "failed"
    elif mutation == "fallback":
        sample["backend_facts"]["cpu_fallback_used"] = True
    elif mutation == "release":
        session["release_verified"] = False
    elif mutation in {"execution", "startup"}:
        sample["backend_facts"][f"{mutation}_resource_admission_passed"] = False
    elif mutation in {"accuracy", "replay"}:
        sample["validation"]["accuracy_qualified" if mutation == "accuracy" else "policy_reference_passed"] = False
    elif mutation == "scope":
        sample["measurement"]["scope_id"] = "session_inclusive_execution_v1"
    elif mutation == "digest":
        sample["output_sha256"] = "not-a-digest"
    elif mutation == "runtime_digest":
        sample["backend_facts"]["output_hash"] = "x" * 64
    elif mutation in {"logical", "physical"}:
        sample["identities"][f"{mutation}_plan_id"] = "a" * 64
    elif mutation == "binary":
        session["terminal_backend_facts"]["dpu_binary_sha256"] = "a" * 64
    elif mutation == "collection":
        sample["backend_facts"]["collection_resource_admission_passed"] = not sample["backend_facts"]["collection_resource_admission_passed"]
    elif mutation == "zero_time":
        sample["measurement"]["total_wall_s"] = session["open_s"] = session["session_close_s"] = 0.0
    elif mutation in {"provenance", "sidecar_executor"}:
        path = s["root"].parent / "preregistration/physical.yml.provenance.json"
        sidecar = json.loads(path.read_text())
        if mutation == "provenance":
            sidecar["round_manifest_hash"] = "0" * 64
        else:
            sidecar["execution_source"] = s["binding"]["source_sha"]
        path.write_bytes(wave._canonical_bytes(sidecar))
        _checksums(path.parent)
    elif mutation == "private_checksum":
        (s["root"].parent / "preregistration/physical.yml").write_text("tamper")
    elif mutation == "expected_attempts":
        s["round_manifest"]["expected_attempts"] += 1
    elif mutation == "split":
        next(iter(s["round_manifest"]["cells"].values()))["split"] = "test"
    elif mutation == "path":
        next(iter(next(iter(s["round_manifest"]["cells"].values()))["candidates"].values()))["path"][0] = [0, 0]
    elif mutation == "profile":
        s["round_manifest"]["profile_hash"] = "0" * 64
    _save_raw(s)
    with pytest.raises(ValueError):
        _extract(s)


@pytest.mark.parametrize("elapsed", [0.0, 12.5, 100.0, -1.0, float("nan"), True])
def test_required_physical_elapsed(tmp_path, elapsed):
    s = _fixture(tmp_path)
    path = tmp_path / "terminal.json"
    terminal = json.loads(path.read_text())
    terminal["physical_stage_elapsed_s"] = terminal["results"]["physical"]["elapsed_s"] = elapsed
    path.write_text(json.dumps(terminal))
    if elapsed != 100.0:
        with pytest.raises(ValueError, match="physical_stage_elapsed_s"):
            _extract(s)
    else:
        assert _extract(s)["physical_stage_elapsed_s"] == elapsed


@pytest.mark.parametrize("field,value", [
    ("source", "e" * 40), ("worktree_status", " M source.py"),
    ("terminal_inspection_valid", False), ("rank1_released", False),
    ("terminal_inspection_errors", ["cannot inspect rank"]),
    ("rank_ownership_at_finally", {"dpu_rank1": "1"}),
    ("physical_stage_elapsed_clock", "estimated"), ("retries", 1), ("replacements", True),
])
def test_invalid_terminal_state_rejected(tmp_path, field, value):
    s = _fixture(tmp_path)
    path = tmp_path / "terminal.json"
    terminal = json.loads(path.read_text())
    terminal[field] = value
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="terminal"):
        _extract(s)


@pytest.mark.parametrize("command,field,value", [
    ("physical", "returncode", 1), ("physical", "returncode", False),
    ("canonical", "returncode", 1), ("physical", "timed_out", True),
    ("canonical", "timed_out", True), ("physical", "planned_attempts", 4),
    ("physical", "actual_sample_rows", 7), ("physical", "elapsed_s", 101.0),
])
def test_invalid_terminal_process_rejected(tmp_path, command, field, value):
    s = _fixture(tmp_path)
    path = tmp_path / "terminal.json"
    terminal = json.loads(path.read_text())
    terminal["results"][command][field] = value
    path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="terminal"):
        _extract(s)


def test_missing_terminal_rejected(tmp_path):
    s = _fixture(tmp_path)
    (tmp_path / "terminal.json").unlink()
    with pytest.raises(ValueError, match="terminal record"):
        _extract(s)


def _rebind_private(s):
    """Keep checksums consistent so negative tests reach semantic validation."""
    s["binding"]["study_hash"] = extractor._hash(s["study"])
    s["round_manifest"]["binding_hash"] = extractor._hash(s["binding"])
    path = s["root"].parent / "preregistration/physical.yml.provenance.json"
    sidecar = json.loads(path.read_text())
    sidecar.update(round_manifest_hash=extractor._hash(s["round_manifest"]),
                   binding_hash=extractor._hash(s["binding"]),
                   workload_record_sha256=extractor._hash(s["workload"]), source_sha=s["binding"]["source_sha"])
    path.write_bytes(wave._canonical_bytes(sidecar))
    _checksums(path.parent)


@pytest.mark.parametrize(("field", "message"), [
    ("path", "invalid active operands"), ("split", "split"), ("memory", "host memory"),
    ("physical", "physical plan"), ("facts", "cost facts"),
    ("source", "source_sha"), ("blocks", "blocks"), ("expected_attempts", "expected attempts"),
])
def test_self_consistent_sidecar_cannot_bypass_semantics(tmp_path, field, message):
    s = _fixture(tmp_path)
    cell = next(iter(s["round_manifest"]["cells"].values()))
    candidate = next(iter(cell["candidates"].values()))
    if field == "path":
        candidate["path"][0] = [0, 0]
    elif field == "split":
        cell["split"] = "test"
    elif field == "memory":
        candidate["facts"]["declared_host_bytes"] = 1
    elif field == "physical":
        candidate["facts"]["physical_plan_id"] = "f" * 64
    elif field == "facts":
        candidate["facts"]["P"] += 1
    elif field == "source":
        s["binding"]["source_sha"] = "F" * 40
    elif field == "blocks":
        s["round_manifest"]["measurement_blocks"] = [1, 2, 3.0]
    else:
        s["round_manifest"]["expected_attempts"] += 1
    _rebind_private(s)
    _save_raw(s)
    with pytest.raises(ValueError, match=message):
        _extract(s)


@pytest.mark.parametrize("stage", ["initial", "evaluation"])
def test_fit_row_contract_and_no_search(tmp_path, monkeypatch, stage):
    s = _fixture(tmp_path, stage)

    def forbidden(*args, **kwargs):
        pytest.fail("evidence extraction must not search or execute")

    monkeypatch.setattr(extractor.cli, "plan_opt_einsum", forbidden)
    monkeypatch.setattr(extractor.cli, "plan_cotengra", forbidden)
    result = _extract(s)
    cells = {key: {"family": cell["family"], "split": "training",
                   "greedy_path_id": cell["selection"]["roles"]["G"],
                   "path_facts": {p: LaunchCostFacts.from_mapping(c["facts"])
                                  for p, c in cell["candidates"].items()}}
             for key, cell in s["round_manifest"]["cells"].items()}
    declared = {"round_id": stage, "accepted": True, "identity": result["identity"],
                "cell_paths": {key: list(cell["path_facts"]) for key, cell in cells.items()},
                "warmup_blocks": [0], "measurement_blocks": s["round_manifest"]["measurement_blocks"]}
    if stage == "evaluation":
        with pytest.raises(ValueError, match="evaluation"):
            validate_launch_cost_observations(cells, [declared], result["rows"], expected_identity=result["identity"])
    else:
        paired = validate_launch_cost_observations(cells, [declared], result["rows"], expected_identity=result["identity"])
        assert set(paired) == set(cells)


def test_preserves_request_components_without_imputation(tmp_path):
    s = _fixture(tmp_path)
    s["samples"][0]["backend_facts"]["operation_facts"] = [
        {"timing": {"request_build_sum_s": 0.25}}, {"timing": {"request_build_sum_s": 0.75}},
    ]
    _save_raw(s)
    result = _extract(s)
    row = next(r for r in result["rows"] if r["sample_id"] == s["samples"][0]["sample_id"])
    assert row["request_build_sum_s"] == 1.0
    assert "request_payload_hashing_sum_s" not in row
    assert all("request_build_sum_s" not in r for r in result["rows"] if r is not row)


def test_qasm_archive_byte_binding(tmp_path):
    source = tmp_path / "stress.qasm"
    source.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[6];\n'
                      + "\n".join(f"ry(0.7) q[{i}];" for i in range(6))
                      + "\n"
                      + "\n".join(f"cx q[{i}],q[{i+1}];" for i in range(5)))
    definition = {"kind": "qasm_file", "name": None, "path": str(source),
                  "parameters": {"qasm_sha256": wave._file_sha256(source)}}
    s = _fixture(tmp_path, definition=definition)
    assert len(_extract(s)["rows"]) == 8
    staged = next((tmp_path / "preregistration/qasm").rglob("*.qasm"))
    staged.write_text(staged.read_text() + "\n// different bytes\n")
    _checksums(staged.parents[2])
    with pytest.raises(ValueError, match="archived QASM bytes"):
        _extract(s)


def test_skipped_cells_are_not_invented_as_observations(tmp_path):
    s = _fixture(tmp_path, "feedback_1", skipped=True)
    result = _extract(s)
    assert len(result["rows"]) == 8
    assert {row["cell_id"] for row in result["rows"]} == {"stress6/1dpu_t8"}


@pytest.mark.parametrize("stage", ["initial", "feedback_1", "feedback_2", "evaluation"])
@pytest.mark.parametrize("mutation", ["omitted", "extra_workload_instance", "overlap", "foreign_skip"])
def test_exact_active_and_skipped_workload_partition(tmp_path, stage, mutation):
    s = _fixture(tmp_path, stage)
    manifest = s["round_manifest"]
    if mutation == "omitted":
        manifest["cells"].pop("stress6/4dpu_t8")
    elif mutation == "extra_workload_instance":
        instance = deepcopy(s["workload"]["instances"][0])
        instance["instance_id"] = "unaccounted-instance"
        s["workload"]["instances"].append(instance)
    else:
        key = "stress6/4dpu_t8" if mutation == "overlap" else "unknown/4dpu_t8"
        manifest["skipped_cells"] = {key: {"reason": "no_new_eligible_candidate", "trace_hash": "a" * 64}}
    _rebind_private(s)
    with pytest.raises(ValueError, match="partition|only in feedback"):
        _extract(s)


@pytest.mark.parametrize("stage", ["initial", "evaluation"])
def test_initial_and_evaluation_cannot_replace_missing_cell_with_skip(tmp_path, stage):
    s = _fixture(tmp_path, stage)
    s["round_manifest"]["cells"].pop("stress6/4dpu_t8")
    s["round_manifest"]["skipped_cells"] = {"stress6/4dpu_t8": {"reason": "no_new_eligible_candidate"}}
    _rebind_private(s)
    with pytest.raises(ValueError, match="only in feedback"):
        _extract(s)


@pytest.mark.parametrize("reason", [None, "no_headroom", "infeasible", ""])
def test_feedback_skip_requires_explicit_no_new_candidate_reason(tmp_path, reason):
    s = _fixture(tmp_path, "feedback_1", skipped=True)
    s["round_manifest"]["skipped_cells"]["stress6/4dpu_t8"]["reason"] = reason
    _rebind_private(s)
    with pytest.raises(ValueError, match="no_new_eligible_candidate"):
        _extract(s)


@pytest.mark.parametrize(("stage", "role"), [
    ("initial", "G"), ("initial", "F"), ("initial", "R"),
    ("feedback_1", "G"), ("feedback_1", "new_best"),
    ("feedback_2", "G"), ("feedback_2", "new_best"),
    ("evaluation", "G"), ("evaluation", "F"), ("evaluation", "R"), ("evaluation", "U"),
])
def test_mandatory_stage_roles(tmp_path, stage, role):
    s = _fixture(tmp_path, stage)
    next(iter(s["round_manifest"]["cells"].values()))["selection"]["roles"].pop(role)
    _rebind_private(s)
    with pytest.raises(ValueError, match="stage roles"):
        _extract(s)


@pytest.mark.parametrize("stage", ["feedback_1", "feedback_2"])
def test_feedback_new_best_cannot_alias_greedy(tmp_path, stage):
    s = _fixture(tmp_path, stage)
    roles = next(iter(s["round_manifest"]["cells"].values()))["selection"]["roles"]
    roles["diverse_1"] = roles["new_best"]
    roles["new_best"] = roles["G"]
    _rebind_private(s)
    with pytest.raises(ValueError, match="new_best must differ from G"):
        _extract(s)


def test_rehashed_loose_accuracy_policy_is_not_the_frozen_policy(tmp_path):
    s = _fixture(tmp_path)
    policy = s["raw_manifest"]["configuration"]["validation_policy"]
    for field in ("float32_atol", "float32_rtol", "float32_relative_l2_max", "float32_norm_drift_max"):
        policy[field] = 1.0
    changed_id = evidence.validation_policy_id(policy)
    s["raw_manifest"]["validation_policy_id"] = changed_id
    for row in s["raw_manifest"]["configuration"]["identity_bindings"]:
        row["validation_policy_id"] = changed_id
    for row in s["samples"]:
        row["identities"]["validation_policy_id"] = changed_id
    _save_raw(s)
    assert extractor.verify_artifacts(s["root"])["success_count"] == 8
    payload = s["raw_manifest"]["configuration"]["experiment"]["experiment_identity_payload"]
    assert payload["validation_policy_id"] == experiment.default_validation_policy_id()
    with pytest.raises(ValueError, match="frozen validation policy"):
        _extract(s)


def test_experiment_payload_must_bind_the_same_default_validation_policy(tmp_path):
    s = _fixture(tmp_path)
    normalized = s["raw_manifest"]["configuration"]["experiment"]
    payload = normalized["experiment_identity_payload"]
    changed = dict(experiment.default_validation_policy())
    changed["float32_atol"] = 1.0
    payload["validation_policy_id"] = evidence.validation_policy_id(changed)
    changed_id = experiment._experiment_id_v3(payload)
    normalized["experiment_id"] = s["raw_manifest"]["experiment_id"] = changed_id
    for row in [*s["samples"], *s["sessions"]]:
        row["experiment_id"] = changed_id
    schedule = {(case, plan, route, attempt, block): (index, order)
                for case, plan, route, attempt, index, block, order
                in evidence._declared_collection_attempts(s["raw_manifest"])}
    for row in s["samples"]:
        row["sample_index"], row["order_index"] = schedule[
            row["case_id"], row["plan_id"], row["route_id"], row["attempt_kind"], row["block_id"]
        ]
        row["sample_id"] = evidence.sample_id(
            row["run_id"], row["case_id"], row["route_id"], row["attempt_kind"], row["sample_index"],
            plan_id=row["plan_id"], block_id=row["block_id"], order_index=row["order_index"],
        )
    _save_raw(s)
    assert extractor.verify_artifacts(s["root"])["success_count"] == 8
    with pytest.raises(ValueError, match="experiment identity payload validation policy"):
        _extract(s)


@pytest.mark.parametrize("stage", ["initial", "feedback_1", "feedback_2", "evaluation"])
@pytest.mark.parametrize("changes", [{"base_seed": 7}, {"block_cooldown_s": 1.0}])
def test_self_consistent_collection_must_match_frozen_stage_policy(tmp_path, stage, changes):
    s = _fixture(tmp_path, stage, collection_changes=changes)
    assert extractor.verify_artifacts(s["root"])["success_count"] == len(s["samples"])
    with pytest.raises(ValueError, match="frozen stage collection policy"):
        _extract(s)
