from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import pytest
import yaml

from quantum_bench.experiment import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_m7b.py"
PHYSICAL_SCRIPT = ROOT / "scripts" / "qualify_m7c_physical.py"
SELECTION_SCRIPT = ROOT / "scripts" / "select_m7c_workload.py"
SCALING_SCRIPT = ROOT / "scripts" / "run_m7c_scaling_campaign.py"
M7C_QUALIFIER_SCRIPT = ROOT / "scripts" / "qualify_m7c.py"
ATTRIBUTION_SCRIPT = ROOT / "scripts" / "analyze_m7d_attribution.py"


def _load_script(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _qualifier():
    return _load_script(SCRIPT, "qualify_m7b")


def _physical_qualifier():
    return _load_script(PHYSICAL_SCRIPT, "qualify_m7c_physical")


def _selector():
    return _load_script(SELECTION_SCRIPT, "select_m7c_workload")


def _scaling_campaign():
    return _load_script(SCALING_SCRIPT, "run_m7c_scaling_campaign")


def _m7c_qualifier():
    return _load_script(M7C_QUALIFIER_SCRIPT, "qualify_m7c")


def _attribution():
    return _load_script(ATTRIBUTION_SCRIPT, "analyze_m7d_attribution")


def _archive(path: Path, member_name: str, *, kind: str = "file") -> None:
    source = path.parent / "payload.txt"
    source.write_text("payload\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        if kind == "file":
            bundle.add(source, arcname=member_name)
            return
        member = tarfile.TarInfo(member_name)
        member.type = tarfile.SYMTYPE
        member.linkname = "payload.txt"
        bundle.addfile(member)


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_extracts_regular_relative_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, "evidence/manifest.json")

    _qualifier()._safe_extract_tar(archive, tmp_path / "output")

    assert (tmp_path / "output" / "evidence" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "payload\n"


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_m7c_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _m7c_qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_verifies_bundled_hashes_and_records_external_provenance(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "m7a-bundle"
    evidence = bundle / "cpu-run" / "manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (bundle / "SHA256SUMS").write_text(
        f"{digest}  runs/{bundle.name}/cpu-run/manifest.json\n"
        + "0" * 64
        + "  native/upmem/runtime/bin/dpu\n",
        encoding="utf-8",
    )

    external = _qualifier()._verify_internal_hashes(tmp_path)

    assert external == ("native/upmem/runtime/bin/dpu",)


def test_physical_preparation_preserves_template_resolved_paths(tmp_path: Path) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_smoke.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "smoke.yml"
    prepared = _physical_qualifier().prepare_config(
        template=template,
        output=output,
        mode="float32-smoke",
        rank_path="/dev/dpu_rank42",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[2, 4],
    )

    assert prepared == output
    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    source_options = source["routes"]["upmem_float32_1dpu"]["options"]
    copied_options = copied["routes"]["upmem_float32_1dpu"]["options"]
    for field in ("host_binary", "dpu_binary", "initialization_binary"):
        assert copied_options[field] == source_options[field]
    assert copied_options["session_root"] == str((tmp_path / "sessions").resolve())
    assert copied_options["rank_paths"] == ("/dev/dpu_rank42",)
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (2, 4),
    }


def test_physical_preparation_probe_has_one_measurement(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "probe.yml"
    _physical_qualifier().prepare_config(
        template=ROOT / "configs" / "tn_benchmark_physical_smoke.yml",
        output=output,
        mode="probe",
        rank_path="/dev/dpu_rank0",
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[0],
    )

    config = load_experiment_config(output)
    assert config["collection"]["warmup_blocks"] == 0
    assert config["collection"]["measurement_blocks"] == 1
    assert config["routes"]["upmem_float32_1dpu"]["numeric_policy"] == (
        "split_complex_float32_v1"
    )


def test_m7c_workload_selection_is_deterministic_and_preregistered(
    tmp_path: Path,
) -> None:
    selector = _selector()
    first = selector.build_selection()
    second = selector.build_selection()

    assert first == second
    assert first["selected_primary"] == "quantization_stress_18q_l2"
    assert first["selected_secondary"] == "ghz_chain_18q"
    assert first["schema_version"] == "m7c_workload_selection_v2"
    assert first["dependency_constraints_sha256"] == hashlib.sha256(
        (ROOT / "ci" / "constraints.txt").read_bytes()
    ).hexdigest()
    assert first["selection_basis_sha256"] == selector._hash(
        {
            key: first[key]
            for key in (
                "schema_version",
                "planner_configuration",
                "selection_rule",
                "candidates",
                "selected_primary",
                "selected_secondary",
            )
        }
    )
    assert first["planner_configuration_sha256"] == selector._hash(
        selector.PLANNER_CONFIG
    )
    assert "constraints_hash" not in first
    assert "python_version" not in first
    primary = next(
        candidate
        for candidate in first["candidates"]
        if candidate["candidate_id"] == first["selected_primary"]
    )
    assert primary["logical_plan_id"] == (
        "d504919e20d95bac608dd906d46abb122f9680873679710b0584e71981648fb5"
    )
    assert primary["topologies"]["dpu4_tasklet8"][
        "collection_resource_admission_passed"
    ] is True

    path = tmp_path / "selection.json"
    selector.write_selection(path)
    selector.check_selection(path)


def test_m7c_workload_selection_rejects_nonancestor_source(tmp_path: Path) -> None:
    selector = _selector()
    selection = selector.build_selection()
    selection["source_commit"] = "0" * 40
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="not an ancestor"):
        selector.check_selection(path)


def test_m7c_workload_selection_rejects_route_matrix_drift(tmp_path: Path) -> None:
    selector = _selector()
    config = yaml.safe_load(
        (ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml").read_text(
            encoding="utf-8"
        )
    )
    config["routes"]["upmem_float32_2dpu_t8"]["options"]["dpu_count"] = 3
    drifted = tmp_path / "drifted.yml"
    drifted.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="topology drift"):
        selector.check_selection(
            ROOT / "configs" / "m7c_workload_selection.json", drifted
        )


def test_m7c_committed_selection_matches_scaling_config() -> None:
    selector = _selector()
    selection = ROOT / "configs" / "m7c_workload_selection.json"
    assert "python_version" not in json.loads(selection.read_text(encoding="utf-8"))
    for config in (
        "tn_benchmark_physical_scaling_diagnostic.yml",
        "tn_benchmark_physical_scaling.yml",
        "tn_benchmark_physical_scaling_confirmation.yml",
    ):
        selector.check_selection(selection, ROOT / "configs" / config)


def _attribution_sample(operation_timing: dict[str, float]) -> dict[str, object]:
    return {
        "status": "success",
        "attempt_kind": "measurement",
        "route_id": "upmem_float32_4dpu_t8",
        "measurement": {"total_wall_s": 2.0, "host_reduce_s": 0.05},
        "backend_facts": {
            "operation_facts": [
                {
                    "rank_count": 1,
                    "timing": operation_timing,
                }
            ]
        },
    }


def _attribution_operation_timing(
    *, request_build_breakdown: bool = True
) -> dict[str, float]:
    timing = {
        "total_wall_s": 1.8,
        "preparation_s": 0.1,
        "encode_s": 0.1,
        "rank_response_h2d_max_sum_s": 0.1,
        "rank_response_kernel_max_sum_s": 0.2,
        "rank_response_d2h_max_sum_s": 0.1,
        "rank_response_total_route_max_sum_s": 0.6,
        "request_wave_wall_sum_s": 1.2,
        "request_build_sum_s": 0.1,
        "rank_submit_parallel_wall_sum_s": 0.95,
        "rank_submit_total_max_sum_s": 0.9,
        "rank_submit_artifact_validation_max_sum_s": 0.1,
        "rank_submit_protocol_write_max_sum_s": 0.1,
        "rank_submit_response_wait_max_sum_s": 0.3,
        "rank_submit_response_validation_max_sum_s": 0.02,
        "coordinator_response_processing_sum_s": 0.1,
        "assembly_s": 0.1,
        "decode_s": 0.1,
    }
    if request_build_breakdown:
        timing.update(
            {
                "request_work_unit_materialization_sum_s": 0.02,
                "request_artifact_build_sum_s": 0.06,
                "request_payload_record_staging_sum_s": 0.03,
                "request_manifest_sidecar_staging_sum_s": 0.02,
            }
        )
    return timing


def test_m7g_attribution_derives_disjoint_request_build_components() -> None:
    attribution = _attribution()
    manifest = {"source_commit": "a" * 40}
    sample = _attribution_sample(_attribution_operation_timing())

    first = attribution.derive_attribution(manifest, (sample,))
    second = attribution.derive_attribution(manifest, (sample,))

    assert first == second
    route = first["routes"]["upmem_float32_4dpu_t8"]
    assert route["measurement_count"] == 1
    assert route["median_total_wall_s"] == pytest.approx(2.0)
    components = route["components"]
    assert components["host_request_overhead_s"]["median_s"] == pytest.approx(0.6)
    assert components["native_request_overhead_s"]["median_s"] == pytest.approx(0.2)
    assert components["operation_other_s"]["median_s"] == pytest.approx(0.2)
    assert components["coordinator_other_s"]["median_s"] == pytest.approx(0.15)
    assert route["median_unresolved_boundary_s"] == pytest.approx(0.35)
    assert route["median_accounting_residual_s"] == pytest.approx(0.0)
    assert route["nested_request_timing_medians_s"][
        "rank_submit_response_wait_max_sum_s"
    ] == pytest.approx(0.3)
    request_build = route["request_build_breakdown"]
    assert request_build is not None
    assert request_build["median_parent_s"] == pytest.approx(0.1)
    children = request_build["children"]
    assert children["work_unit_materialization_s"]["median_s"] == pytest.approx(0.02)
    assert children["payload_record_staging_s"]["median_s"] == pytest.approx(0.03)
    assert children["manifest_sidecar_staging_s"]["median_s"] == pytest.approx(0.02)
    assert children["artifact_build_residual_s"]["median_s"] == pytest.approx(0.01)
    assert children["request_build_residual_s"]["median_s"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("request_wave_wall_sum_s", None, "missing request_wave_wall_sum_s"),
        ("rank_response_total_route_max_sum_s", 0.3, "native request overhead"),
        (
            "request_payload_record_staging_sum_s",
            0.07,
            "request artifact build residual",
        ),
    ],
)
def test_m7g_attribution_rejects_missing_or_inconsistent_timing(
    field: str, value: float | None, message: str
) -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing()
    if value is None:
        del timing[field]
    else:
        timing[field] = value

    with pytest.raises(ValueError, match=message):
        attribution.derive_attribution(
            {"source_commit": "a" * 40},
            (_attribution_sample(timing),),
        )


def test_m7g_attribution_accepts_m7f_response_wait_and_omits_build_breakdown() -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing(request_build_breakdown=False)

    result = attribution.derive_attribution(
        {"source_commit": "a" * 40},
        (_attribution_sample(timing),),
    )

    route = result["routes"]["upmem_float32_4dpu_t8"]
    assert route["request_build_breakdown"] is None
    assert route["components"]["host_request_overhead_s"]["median_s"] == pytest.approx(
        0.6
    )


def test_m7g_attribution_rejects_partial_request_build_timing() -> None:
    attribution = _attribution()
    timing = _attribution_operation_timing()
    del timing["request_manifest_sidecar_staging_sum_s"]

    with pytest.raises(ValueError, match="request-build timing is missing"):
        attribution.derive_attribution(
            {"source_commit": "a" * 40},
            (_attribution_sample(timing),),
        )


def test_m7c_scaling_preparation_preserves_all_resolved_route_paths(
    tmp_path: Path,
) -> None:
    template = ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml"
    output = tmp_path / "runs" / "configs" / "eth" / "diagnostic.yml"
    _scaling_campaign().prepare_config(
        template=template,
        output=output,
        rank_paths=["/dev/dpu_rank19"],
        session_root=str(tmp_path / "sessions"),
        expected_cpus=[1, 3],
    )

    source = load_experiment_config(template)
    copied = load_experiment_config(output)
    for route_id, route in source["routes"].items():
        if route["executor"] != "upmem_physical":
            continue
        source_options = route["options"]
        copied_options = copied["routes"][route_id]["options"]
        for field in ("host_binary", "dpu_binary", "initialization_binary"):
            assert copied_options[field] == source_options[field]
        assert copied_options["rank_paths"] == ("/dev/dpu_rank19",)
        assert copied_options["session_root"] == str(
            (tmp_path / "sessions" / route_id).resolve()
        )
    assert copied["collection"]["machine_policy"]["affinity"] == {
        "mode": "exact_required_v1",
        "expected_cpus": (1, 3),
    }


def _complete_m7c_diagnostic(script: object) -> tuple[dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    config = load_experiment_config(
        ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml"
    )
    manifest = {
        "source_commit": script._source_commit(),
        "configuration": {"experiment": config},
    }
    samples: list[dict[str, object]] = []
    sessions: list[dict[str, object]] = []
    binary_hashes = {
        "host_binary_sha256": "1" * 64,
        "dpu_binary_sha256": "2" * 64,
        "initialization_binary_sha256": "3" * 64,
    }
    physical_routes = set(script._DIAGNOSTIC_PHYSICAL_ROUTE_IDS)
    for block_id in range(6):
        attempt_kind = "warmup" if block_id == 0 else "measurement"
        sample_index = 0 if block_id == 0 else block_id - 1
        for route_id in script._DIAGNOSTIC_ROUTE_IDS:
            physical = route_id in physical_routes
            session_id = f"{route_id}-{block_id}" if physical else None
            samples.append(
                {
                    "case_id": "scaling_primary",
                    "plan_id": "greedy",
                    "route_id": route_id,
                    "attempt_kind": attempt_kind,
                    "sample_index": sample_index,
                    "block_id": block_id,
                    "status": "success",
                    "session_instance_id": session_id,
                    "measurement": {"total_wall_s": 0.020},
                    "backend_facts": (
                        {
                            "target_observed": "physical_hardware",
                            "physical_target_verified": True,
                            "hardware_kernel_executed": True,
                            "simulator_kernel_executed": False,
                            "cpu_fallback_used": False,
                            "startup_resource_admission_passed": True,
                            "execution_resource_admission_passed": True,
                        }
                        if physical
                        else {}
                    ),
                    "validation": {
                        "accuracy_qualified": True,
                        "policy_reference_applicable": physical,
                        "policy_reference_passed": True if physical else None,
                    },
                }
            )
            if physical:
                sessions.append(
                    {
                        "route_id": route_id,
                        "session_instance_id": session_id,
                        "status": "success",
                        "release_verified": True,
                        "terminal_backend_facts": {
                            "target_observed": "physical_hardware",
                            **binary_hashes,
                        },
                    }
                )
    return manifest, tuple(samples), tuple(sessions)


def test_m7c_diagnostic_summary_requires_complete_literal_matrix() -> None:
    script = _scaling_campaign()
    manifest, samples, sessions = _complete_m7c_diagnostic(script)
    report = {"schema_version": "evidence_report_v5"}

    complete = script._diagnostic_summary(
        manifest=manifest,
        samples=samples,
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert complete["gate_passed"] is True
    assert complete["expected_route_ids"] == list(script._DIAGNOSTIC_ROUTE_IDS)
    assert complete["expected_block_ids"] == [0, 1, 2, 3, 4, 5]
    assert all(not warnings for warnings in complete["measurement_warnings"].values())

    short_samples = [dict(sample) for sample in samples]
    for sample in short_samples:
        if sample["route_id"] == "numpy_same_dag" and sample["attempt_kind"] == "measurement":
            sample["measurement"] = {"total_wall_s": 0.001}
    short = script._diagnostic_summary(
        manifest=manifest,
        samples=tuple(short_samples),
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert short["gate_passed"] is True
    assert short["measurement_warnings"]["numpy_same_dag"] == ["median_below_10ms"]

    incomplete = script._diagnostic_summary(
        manifest=manifest,
        samples=samples[:-1],
        sessions=sessions,
        report=report,
        selection_sha256="4" * 64,
    )
    assert incomplete["gate_passed"] is False
    assert "diagnostic_block_matrix_incomplete" in incomplete["gate_reasons"]


def test_m7c_performance_run_requires_diagnostic_summary_before_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _scaling_campaign()
    monkeypatch.setattr(script, "_clean_worktree", lambda: None)
    monkeypatch.setattr(script, "_selector_check", lambda *args: None)
    output = tmp_path / "evidence"
    report_output = tmp_path / "report"

    with pytest.raises(ValueError, match="requires --diagnostic-summary"):
        script.run_campaign(
            selection=ROOT / "configs" / "m7c_workload_selection.json",
            config=ROOT / "configs" / "tn_benchmark_physical_scaling.yml",
            output=output,
            report_output=report_output,
        )

    assert not output.exists()
    assert not report_output.exists()


def test_m7c_campaign_binding_ignores_collection_lifecycle(tmp_path: Path) -> None:
    script = _scaling_campaign()
    diagnostic = script._plain(
        load_experiment_config(ROOT / "configs" / "tn_benchmark_physical_scaling_diagnostic.yml")
    )
    performance = script._plain(
        load_experiment_config(ROOT / "configs" / "tn_benchmark_physical_scaling.yml")
    )
    binaries = {
        "host_binary": tmp_path / "host",
        "dpu_binary": tmp_path / "dpu",
        "initialization_binary": tmp_path / "initialization",
    }
    for path in binaries.values():
        path.write_bytes(b"m7c")
    for configuration in (diagnostic, performance):
        configuration["collection"]["machine_policy"]["affinity"] = {
            "mode": "exact_required_v1",
            "expected_cpus": [1, 3],
        }
        for route in configuration["routes"].values():
            if route["executor"] != "upmem_physical":
                continue
            route["options"]["rank_paths"] = ["/dev/dpu_rank19"]
            route["options"]["session_root"] = str(tmp_path / route["options"]["session_root"])
            for field, path in binaries.items():
                route["options"][field] = str(path)
    source_commit = script._source_commit()
    selection_sha256 = "5" * 64
    diagnostic_binding = script._campaign_binding_sha256(
        diagnostic,
        source_commit=source_commit,
        selection_sha256=selection_sha256,
        binary_hashes=script._route_binary_hashes_from_config(diagnostic),
    )
    performance_binding = script._campaign_binding_sha256(
        performance,
        source_commit=source_commit,
        selection_sha256=selection_sha256,
        binary_hashes=script._route_binary_hashes_from_config(performance),
    )

    assert diagnostic_binding == performance_binding
    assert performance["collection"]["block_cooldown_s"] == 0.0
