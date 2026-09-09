"""Read-only extraction of a frozen cost-guided round, before copy acceptance."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from quantum_bench import cli
from quantum_bench.evidence import executable_id, load_artifacts, problem_id, tensor_network_structure_id
from quantum_bench.experiment import default_validation_policy, default_validation_policy_id
from quantum_bench.planning import normalize_frozen_path
from quantum_bench.report import verify_artifacts
from quantum_bench.upmem.execution_features import extract_launch_cost_features

import qualify_upmem_path_candidates as qualifier
import upmem_path_heuristic as wave


def _hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _equal(actual, expected, label):
    # JSON comparison also distinguishes bools from integer resource facts.
    if _hash(actual) != _hash(expected):
        raise ValueError(f"{label} mismatch")


def _seconds(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _digest(value, label, length=64):
    if (not isinstance(value, str) or len(value) != length
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be lowercase hexadecimal of length {length}")
    return value


def _executable_identity(binaries, contract):
    # The same payload as cli._executable_identity, using verified archived
    # digests instead of reopening remote executable paths on this machine.
    return executable_id({
        "executor": "upmem_physical", "abi_version": 5,
        "static_file_sha256": {k: v["sha256"] for k, v in binaries.items()},
        "request_transport": contract["request_transport"], "source_commit": None,
        "dependency_versions": {}, "execution_policy": {
            "fuse_complex": contract["fuse_complex"], "geometry_policy": contract["geometry_policy"],
        },
    })


def _selected(round_manifest, workload, study, experiment, archive_root):
    stage = round_manifest["stage"]
    split = "test" if stage == "evaluation" else "training"
    instances = {}
    for instance in workload["instances"]:
        key = instance["instance_id"]
        if not isinstance(key, str) or not key or key in instances:
            raise ValueError("duplicate or empty workload instance identity")
        instances[key] = instance
    cells = wave._mapping(round_manifest.get("cells"), "round cells")
    if not cells:
        raise ValueError("round must declare active cells")
    topologies = {
        row["topology_id"]: {k: row[k] for k in ("dpu_count", "rank_count", "tasklets_per_dpu")}
        for row in study["executor"]["topologies"]
    }
    skipped = wave._mapping(round_manifest.get("skipped_cells", {}), "skipped cells")
    feedback = stage in {"feedback_1", "feedback_2"}
    if skipped and not feedback:
        raise ValueError("skipped cells are allowed only in feedback rounds")
    for record in skipped.values():
        record = wave._mapping(record, "skipped cell")
        if record.get("reason") != "no_new_eligible_candidate":
            raise ValueError("feedback skip requires no_new_eligible_candidate")
    workload_cells = {f"{key}/{topology}" for key, instance in instances.items()
                      if instance["split"] == split for topology in topologies}
    if set(cells) & set(skipped) or set(cells) | set(skipped) != workload_cells:
        raise ValueError("active and skipped cells must exactly partition workload split/topologies")
    required_roles = {"G", "new_best"} if feedback else {"G", "F", "R"}
    if stage == "evaluation":
        required_roles.add("U")
    expected, networks = {}, {}
    for cell_id, cell in sorted(cells.items()):
        circuit_id, topology_id = cell["circuit_id"], cell["topology_id"]
        if cell_id != f"{circuit_id}/{topology_id}" or topology_id not in topologies:
            raise ValueError("cell identity/topology mismatch")
        instance = instances.get(circuit_id)
        if instance is None or instance["split"] != split or cell["split"] != split:
            raise ValueError("round cell split does not match stage/workload")
        _equal(cell["family"], instance["family"], "cell family")
        paths, roles = cell["selection"]["path_ids"], cell["selection"]["roles"]
        limit = 3 if stage.startswith("feedback_") else 4
        if (not isinstance(paths, list) or not paths or len(paths) > limit
                or any(not isinstance(p, str) for p in paths)
                or len(paths) != len(set(paths)) or set(paths) != set(cell["candidates"])):
            raise ValueError("selected candidate set must be exact, unique and bounded")
        if (not isinstance(roles, dict) or not required_roles <= set(roles)
                or any(not isinstance(r, str) or not r for r in roles)
                or set(roles.values()) != set(paths)):
            raise ValueError("selection roles must include stage roles and cover selected paths")
        if feedback and roles["new_best"] == roles["G"]:
            raise ValueError("feedback new_best must differ from G")
        if circuit_id not in networks:
            job = qualifier.make_simulation_job(qualifier._circuit_from_definition(instance["circuit"]))
            problem = problem_id(job)
            _equal(problem, instance["canonical_circuit"]["operation_identity"]["problem_id"], "workload problem")
            network, _ = qualifier.lower_tensor_network(job)
            definition = deepcopy(experiment["cases"][circuit_id]["circuit"])
            if definition["kind"] == "qasm_file":
                source = (archive_root / definition["path"]).resolve()
                # Public QASM parameters cannot carry this private byte binding.
                expected_sha = instance["circuit"]["parameters"]["qasm_sha256"]
                _equal(wave._file_sha256(source), expected_sha, "archived QASM bytes")
                definition["path"] = str(source)
            config_job = cli._job({"circuit": definition})
            _equal(problem_id(config_job), problem, "configured problem")
            networks[circuit_id] = (network, config_job, problem)
        network, config_job, problem = networks[circuit_id]
        resources = topologies[topology_id]
        wave._wave_topology_resources(topology_id, resources)
        for identifier in paths:
            candidate = cell["candidates"][identifier]
            path = normalize_frozen_path(candidate["path"])
            _equal(candidate["path_id"], identifier, "candidate path ID")
            _equal(qualifier.path_id(path, circuit_id=circuit_id), identifier, "canonical path ID")
            facts = candidate["facts"]
            plan_key = f"path_{identifier}"
            configured_plan = experiment["plans"][plan_key]
            expected_planner = {
                "engine": "frozen_path", "mode": "replay", "path": path,
                "tensor_network_structure_id": tensor_network_structure_id(network),
                "logical_plan_id": facts["logical_plan_id"],
            }
            _equal(configured_plan["planner"], expected_planner, "frozen replay planner")
            if configured_plan.get("slicing") is not None:
                raise ValueError("frozen replay must not use slicing")
            _, _, dag, _ = cli._plan_dag(config_job, configured_plan)
            plan = qualifier.plan_upmem(
                dag, numeric_policy=qualifier.FLOAT32,
                topology=qualifier.UpmemTopology(**resources),
                schedule_policy=study["executor"]["schedule_policy"],
            )
            qualifier._require_wave_execution_coverage(plan)
            _equal(qualifier.physical_plan_id(plan), facts["physical_plan_id"], "physical plan")
            recomputed = extract_launch_cost_features(dag, plan)
            memory = recomputed["execution"]["host_buffers"]["declared_executor_memory_estimate_bytes"]
            _equal(facts["declared_host_bytes"], memory, "declared host memory")
            if memory > 512 * 1024 * 1024:
                raise ValueError("candidate exceeds frozen host memory budget")
            for field in ("model_id", "H", "P", "N", "launches"):
                _equal(facts[field], recomputed[field], f"candidate cost facts {field}")
            admission = qualifier.collection_resource_admission(plan)
            wave._wave_scaling_admission(facts["resource_admission"], resources, expected=admission)
            expected[(circuit_id, plan_key, topology_id)] = {
                "cell_id": cell_id, "path_id": identifier, "split": split,
                "family": cell["family"], "roles": sorted(r for r, p in roles.items() if p == identifier),
                "topology": {"topology": resources, "resource_admission": admission},
                "identities": {"problem_id": problem,
                               "tensor_network_structure_id": expected_planner["tensor_network_structure_id"],
                               "logical_plan_id": facts["logical_plan_id"],
                               "physical_plan_id": facts["physical_plan_id"]},
            }
    matrix = [(row["case_id"], row["plan_id"], route)
              for row in experiment["matrix"] for route in row["route_ids"]]
    if len(matrix) != len(set(matrix)) or set(matrix) != set(expected):
        raise ValueError("configuration matrix differs from exact selected set")
    for field, keys in (("cases", {k[0] for k in expected}), ("plans", {k[1] for k in expected})):
        _equal(sorted(experiment[field]), sorted(keys), f"configuration {field} set")
    return expected


def _binary_bindings(experiment, archive, contract, environment):
    binaries, used = {}, set()
    names = {"dpu_binary": "dpu_wave_v5_t8", "host_binary": "host_upmem_execution_plan_v4_t8",
             "initialization_binary": "dpu_simplepim_management_init_t8"}
    _equal(environment.get("requested_rank_paths"), ["/dev/dpu_rank1"], "environment rank paths")
    for route_id, route in experiment["routes"].items():
        _equal(route["executor"], "upmem_physical", "executor")
        _equal(route["numeric_policy"], contract["numeric_policy"], "numeric policy")
        options = route["options"]
        wave._wave_topology_resources(route_id, options)
        for key in ("request_transport", "schedule_policy", "fuse_complex", "geometry_policy"):
            _equal(options[key], contract[key], f"route {key}")
        _equal(options.get("rank_paths"), ["/dev/dpu_rank1"], "route rank paths")
        # The accepted executor defaults are the policy, not an RSS bound.
        for key, value in (("intermediate_policy", "host_roundtrip_v1"),
                           ("host_memory_budget_bytes", 512 * 1024 * 1024),
                           ("host_memory_reserve_bytes", 0)):
            _equal(options.get(key, value), value, f"route {key}")
        binaries[route_id] = {}
        for field, name in names.items():
            path = options[field]
            if not Path(path).is_absolute():
                raise ValueError("deployment binary path must be absolute")
            digest = _digest(contract["binaries"][name], name)
            _equal(archive["binary_manifest"].get(path), digest, "frozen binary digest")
            binaries[route_id][field] = {"path": path, "sha256": digest}
            used.add(path)
    _equal(sorted(archive["binary_manifest"]), sorted(used), "binary manifest paths")
    return binaries


def extract_round_observations(raw_root, round_manifest, workload, binding, study) -> dict:
    """Validate one complete physical round; copy/archive acceptance remains external.

    The canonical private sidecar requires round_manifest_hash, binding_hash,
    source_sha, execution_source, workload_record_sha256, configuration_sha256 and
    normalized_configuration_sha256. The first two use sorted compact JSON
    SHA-256 (no newline). The configuration hashes use the existing archive
    helper's byte/canonical domains. No observation is marked accepted here.
    """
    root = Path(raw_root)
    stage = round_manifest["stage"]
    stages = ("initial", "feedback_1", "feedback_2", "evaluation")
    if stage not in stages:
        raise ValueError("unsupported round stage")
    measured = 5 if stage == "evaluation" else 3
    # Bind the policy before canonical verification considers the schedule or
    # self-consistent hashes supplied by the evidence itself.
    policy_manifest = wave._wave_json_mapping(root / "manifest.json", "raw manifest")
    configuration = wave._mapping(policy_manifest.get("configuration"), "raw configuration")
    experiment = wave._mapping(configuration.get("experiment"), "raw experiment")
    _equal(configuration.get("validation_policy"), dict(default_validation_policy()),
           "frozen validation policy mapping")
    _equal(policy_manifest.get("validation_policy_id"), default_validation_policy_id(),
           "frozen validation policy ID")
    payload = wave._mapping(experiment.get("experiment_identity_payload"), "experiment identity payload")
    _equal(payload.get("validation_policy_id"), policy_manifest["validation_policy_id"],
           "experiment identity payload validation policy")
    _equal(experiment.get("collection"), qualifier._collection(
        warmups=1, measurements=measured, seed=20260910 + stages.index(stage),
    ), "frozen stage collection policy")
    report = verify_artifacts(root)
    manifest, samples, sessions = load_artifacts(root)
    _equal(manifest, policy_manifest, "raw manifest verification snapshot")
    contract = study["executor"]
    source = _digest(binding.get("source_sha"), "source_sha", 40)
    _equal(contract["source"], wave.EXECUTION_SOURCE, "frozen executor source")
    _equal(binding.get("executor_source"), contract["source"], "binding executor source")
    _equal(binding.get("study_hash"), _hash(study), "binding study hash")
    _equal(binding.get("workload_hash"), study["workload"]["sha256"], "binding workload hash")
    for key, value in (("request_transport", "packed_wave_v1"),
                       ("numeric_policy", "split_complex_float32_v1"),
                       ("schedule_policy", "static_dag_waves_v1"),
                       ("geometry_policy", "panel_only_v1"), ("fuse_complex", True)):
        _equal(contract[key], value, f"executor {key}")
    _equal(round_manifest["study_id"], study["study_id"], "round study")
    _equal(round_manifest["binding_hash"], _hash(binding), "round binding hash")
    for field in ("profile_hash", "normalization_hash"):
        _digest(round_manifest.get(field), field)
    _equal(round_manifest["warmup_blocks"], [0], "warmup blocks")
    _equal(round_manifest["measurement_blocks"], list(range(1, measured + 1)), "measurement blocks")
    archive_root = root.parent / "preregistration"
    archive = wave._wave_private_archive(root, archive_root / "physical.yml.provenance.json", manifest)
    provenance = archive["provenance"]
    for field, value in {
        "round_manifest_hash": _hash(round_manifest), "binding_hash": _hash(binding),
        "source_sha": source, "execution_source": contract["source"],
        "workload_record_sha256": _hash(workload),
    }.items():
        _equal(provenance.get(field), value, f"provenance {field}")
    _equal(manifest["source_commit"], source, "raw source commit")
    _equal(manifest["source_worktree_dirty"], False, "raw source cleanliness")
    _equal(manifest["status"], "completed", "raw completion")
    experiment = manifest["configuration"]["experiment"]
    if experiment["defaults"]["timeout_s"] != 120.0:
        raise ValueError("physical attempt timeout must remain 120 seconds")
    expected = _selected(round_manifest, workload, study, experiment, archive_root)
    count = len(expected) * (measured + 1)
    _equal(round_manifest["expected_attempts"], count, "expected attempts")
    if count > (144 if stage.startswith("feedback_") else 288 if stage == "evaluation" else 192):
        raise ValueError("round exceeds protocol attempt bound")
    if len(samples) != count or len(sessions) != count:
        raise ValueError("raw sample/session counts differ from declared attempts")
    _equal(sorted(experiment["routes"]), sorted({key[2] for key in expected}), "route set")
    binaries = _binary_bindings(experiment, archive, contract, manifest["configuration"]["environment"])
    identity = {"source_sha": source, "execution_source": contract["source"],
                "policy_id": contract["numeric_policy"], "timing_scope": "steady_execution_v1",
                "round_manifest_hash": _hash(round_manifest), "experiment_id": manifest["experiment_id"]}
    session_map = {s["session_instance_id"]: s for s in sessions}
    if len(session_map) != count:
        raise ValueError("duplicate session identity")
    bound = {}
    for item in manifest["configuration"]["identity_bindings"]:
        key = (item["case_id"], item["plan_id"], item["route_id"])
        if key in bound or key not in expected:
            raise ValueError("unexpected or duplicate raw identity binding")
        for field, value in expected[key]["identities"].items():
            _equal(item[field], value, f"raw binding {field}")
        _equal(item["executable_id"], _executable_identity(binaries[key[2]], contract), "executable identity")
        bound[key] = item
    _equal(sorted(bound), sorted(expected), "raw identity binding set")
    rows, seen, used_sessions = [], set(), set()
    for sample in samples:
        key = (sample["case_id"], sample["plan_id"], sample["route_id"])
        if key not in expected:
            raise ValueError("sample outside selected set")
        item = expected[key]
        block, attempt = sample["block_id"], sample["attempt_kind"]
        observation = (*key, block, attempt)
        if observation in seen:
            raise ValueError("duplicate selected observation")
        seen.add(observation)
        session_id = sample["session_instance_id"]
        if session_id in used_sessions or session_id not in session_map:
            raise ValueError("sample requires a unique fresh session")
        used_sessions.add(session_id)
        session = session_map[session_id]
        for field in ("run_id", "experiment_id", "case_id", "plan_id", "route_id"):
            _equal(session[field], sample[field], f"sample/session {field}")
        for field, value in sample["identities"].items():
            _equal(value, bound[key][field], f"sample identity {field}")
        _equal(sample["status"], "success", "sample status")
        facts, terminal = wave._joined_backend_facts(sample, session, allow_null_overrides=True)
        wave._require_backend_contract(sample, session, facts, item["topology"], contract,
                                       binaries[key[2]], claim_policy="diagnostic_v1")
        for field in ("logical_plan_id", "physical_plan_id"):
            _equal(facts[field], item["identities"][field], f"backend {field}")
        validation = sample["validation"]
        for field in ("policy_reference_applicable", "policy_reference_passed",
                      "full_precision_threshold_applicable", "full_precision_passed"):
            _equal(validation[field], True, f"numerical {field}")
        _digest(sample["output_sha256"], "experiment output digest")
        _digest(facts.get("output_hash"), "runtime output digest")
        measurement = sample["measurement"]
        _equal(measurement["scope_id"], "steady_execution_v1", "raw timing scope")
        wall = _seconds(measurement["total_wall_s"], "total_wall_s")
        opening = _seconds(session["open_s"], "session open_s")
        closing = _seconds(session["session_close_s"], "session close_s")
        inclusive = opening + wall + closing
        if not math.isfinite(inclusive) or inclusive <= 0:
            raise ValueError("session-inclusive time must be positive and finite")
        row = {k: deepcopy(item[k]) for k in ("cell_id", "path_id", "split", "family", "roles")}
        row.update({k: sample[k] for k in ("sample_id", "session_instance_id", "run_id", "experiment_id",
                                          "sample_index", "order_index")})
        row.update(round_id=stage, block=block, attempt_type=attempt, identity=dict(identity),
                   status="success", validation="pass", fallback=False,
                   full_precision_passed=True, policy_reference_passed=True,
                   execution_resource_admission_passed=True, startup_resource_admission_passed=True,
                   collection_resource_admission_passed=facts["collection_resource_admission_passed"],
                   session_open_s=opening, total_wall_s=wall, session_close_s=closing,
                   session_inclusive_s=inclusive, raw_identities=dict(sample["identities"]),
                   experiment_output_sha256=sample["output_sha256"], runtime_output_sha256=facts["output_hash"],
                   raw_measurement=deepcopy(measurement), numeric_facts=deepcopy(sample["numeric_facts"]))
        for field in ("kernel_s", "h2d_s", "d2h_s"):
            if field in measurement:
                row[field] = measurement[field]
        for field in ("tasklet_row_sufficiency_passed", "dominant_work_wave_tasklet_row_sufficiency_passed",
                      "dominant_work_wave_allocated_dpu_slots", "dominant_work_wave_populated_dpu_slots"):
            row[field] = facts[field]
        if sample["backend_facts"].get("operation_facts"):
            for field in ("request_build_sum_s", "request_wave_wall_sum_s", "request_artifact_build_sum_s",
                          "request_payload_record_staging_sum_s", "request_work_unit_materialization_sum_s",
                          "request_payload_materialization_sum_s", "request_payload_hashing_sum_s",
                          "request_payload_file_write_sum_s", "request_manifest_sidecar_staging_sum_s",
                          "request_build_residual_sum_s"):
                value = wave._operation_timing_total(sample, field)
                if value is not None:
                    row[field] = value
        rows.append(row)
    expected_attempts = {(*key, block, "warmup" if block == 0 else "measurement")
                         for key in expected for block in range(measured + 1)}
    if seen != expected_attempts or used_sessions != set(session_map):
        raise ValueError("raw observations do not match the exact declared round")
    result = {
        "round_id": stage, "identity": identity,
        "rows": sorted(rows, key=lambda r: (r["cell_id"], r["path_id"], r["block"])),
        "canonical_report": report, "canonical_report_hash": _hash(report),
        "raw_artifact_sha256": {name: wave._file_sha256(root / name)
                                for name in ("manifest.json", "samples.jsonl", "sessions.jsonl")},
        "private_archive_sha256": dict(archive["private_hashes"]),
        "copy_acceptance": "caller_pending", "fit_eligible_split": stage != "evaluation",
    }
    terminal_path = root.parent / "terminal.json"
    if not terminal_path.is_file():
        raise ValueError("physical stage terminal record is required")
    terminal_record = json.loads(terminal_path.read_text())
    elapsed = _seconds(terminal_record.get("physical_stage_elapsed_s"), "physical_stage_elapsed_s")
    if elapsed <= 0 or elapsed < sum(row["session_inclusive_s"] for row in rows):
        raise ValueError("physical_stage_elapsed_s cannot be shorter than its sequential attempts")
    for field, value in {
        "physical_stage_elapsed_clock": "monotonic_duration",
        "source": source, "worktree_status": "", "terminal_inspection_valid": True,
        "rank1_released": True, "terminal_inspection_errors": [], "retries": 0, "replacements": 0,
    }.items():
        _equal(terminal_record.get(field), value, f"terminal {field}")
    ownership = wave._mapping(terminal_record.get("rank_ownership_at_finally"), "terminal rank ownership")
    _equal(ownership.get("dpu_rank1"), "0", "terminal selected rank released")
    results = wave._mapping(terminal_record.get("results"), "terminal process results")
    for command in ("physical", "canonical"):
        command_result = wave._mapping(results.get(command), f"terminal {command} result")
        _equal(command_result.get("returncode"), 0, f"terminal {command} returncode")
        _equal(command_result.get("timed_out"), False, f"terminal {command} timeout")
    physical = results["physical"]
    _equal(physical.get("elapsed_s"), elapsed, "terminal physical elapsed")
    _equal(physical.get("planned_attempts"), count, "terminal planned attempts")
    _equal(physical.get("actual_sample_rows"), count, "terminal sample count")
    result["physical_stage_elapsed_s"] = elapsed
    result["terminal_sha256"] = wave._file_sha256(terminal_path)
    return result
