from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import quantum_bench.evidence.canonical as canonical
from quantum_bench.evidence import (
    append_sample,
    append_session,
    canonical_json,
    environment_id,
    executable_id,
    finalize_artifacts,
    identity_hash,
    new_run_id,
    problem_id,
    require_matching_scope,
    sample_id,
    tensor_network_structure_id,
    validate_artifact_set,
    validate_manifest,
    validate_sample,
    validate_session,
    validation_policy_id,
    write_manifest,
)
from quantum_bench.circuits import builtin_circuit
from quantum_bench.experiment import run_direct_samples
from quantum_bench.model import (
    CircuitSpec,
    TensorNetwork,
    TensorSpec,
    make_simulation_job,
)
from quantum_bench.results import Measurement


_RUN_ID = new_run_id()
_OTHER_RUN_ID = new_run_id()
_EXPERIMENT_ID = "e" * 64
_ENVIRONMENT_ID = "d" * 64
_POLICY_ID = "c" * 64
_SESSION_ID = "session-1"


def _measurement() -> dict[str, object]:
    value = {field.name: None for field in fields(Measurement)}
    value.update({"scope_id": "steady_execution_v1", "total_wall_s": 1.0})
    return value


def _validation(*, passed: bool = True) -> dict[str, object]:
    return {
        "policy_reference_applicable": True,
        "policy_reference_passed": passed,
        "full_precision_threshold_applicable": False,
        "full_precision_passed": None,
        "scientific_validation_passed": passed,
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
    }


def _manifest(
    *,
    run_id: str = _RUN_ID,
    status: str = "running",
    created_at_utc: str = "2026-08-23T12:00:00Z",
    expected_counts: dict[str, int] | None = None,
) -> dict[str, object]:
    if expected_counts is None:
        expected_counts = {"warmup": 0, "measurement": 1, "sessions": 1}
    return {
        "schema_version": "evidence_manifest_v1",
        "run_id": run_id,
        "experiment_id": _EXPERIMENT_ID,
        "environment_id": _ENVIRONMENT_ID,
        "validation_policy_id": _POLICY_ID,
        "created_at_utc": created_at_utc,
        "source_commit": "a" * 40,
        "source_worktree_dirty": False,
        "configuration": {"b": 2, "a": [True, None]},
        "expected_counts": expected_counts,
        "files": {
            "manifest": "manifest.json",
            "samples": "samples.jsonl",
            "sessions": "sessions.jsonl",
        },
        "status": status,
    }


def _sample(
    status: str = "success",
    *,
    run_id: str = _RUN_ID,
    experiment_id: str = _EXPERIMENT_ID,
    case_id: str = "case-1",
    plan_id: str | None = "plan-1",
    route_id: str = "route-1",
    kind: str = "measurement",
    index: int = 0,
    session_instance_id: str | None = _SESSION_ID,
    environment_id: str = _ENVIRONMENT_ID,
    validation_policy_id: str = _POLICY_ID,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema_version": "evidence_sample_v1",
        "sample_id": sample_id(run_id, case_id, route_id, kind, index, plan_id=plan_id),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "sample_kind": kind,
        "sample_index": index,
        "session_instance_id": session_instance_id,
        "status": status,
        "identities": {
            "problem_id": "1" * 64,
            "tensor_network_structure_id": "2" * 64,
            "logical_plan_id": "3" * 64,
            "physical_plan_id": None,
            "executable_id": None,
            "environment_id": environment_id,
            "validation_policy_id": validation_policy_id,
        },
        "measurement": _measurement(),
        "backend_facts": {"provider": "cpu"},
        "numeric_facts": {"error": 0.0},
        "output_sha256": "b" * 64,
        "validation": None,
        "failure": None,
    }
    if status != "success":
        record["measurement"] = None
        record["output_sha256"] = None
        record["failure"] = {"stage": "preflight", "reason": "not available"}
        if status == "unsupported":
            record["failure"]["capability"] = "hardware"
    return record


def _session(
    status: str = "success",
    *,
    run_id: str = _RUN_ID,
    experiment_id: str = _EXPERIMENT_ID,
    case_id: str = "case-1",
    plan_id: str | None = "plan-1",
    route_id: str = "route-1",
    session_instance_id: str = _SESSION_ID,
    release_attempted: bool = True,
    release_succeeded: bool = True,
    release_verified: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "evidence_session_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "plan_id": plan_id,
        "route_id": route_id,
        "session_instance_id": session_instance_id,
        "session_protocol_id": "protocol-1",
        "open_s": 0.1,
        "session_close_s": 0.2,
        "status": status,
        "terminal_backend_facts": {"released": release_verified},
        "release_attempted": release_attempted,
        "release_succeeded": release_succeeded,
        "release_verified": release_verified,
        "failure": None
        if status == "success"
        else {"stage": "close", "reason": "failed"},
    }


def _write_artifact_files(
    directory: Path,
    manifest: dict[str, object],
    samples: list[dict[str, object]],
    sessions: list[dict[str, object]],
) -> None:
    write_manifest(directory / "manifest.json", manifest)
    samples_path = directory / "samples.jsonl"
    sessions_path = directory / "sessions.jsonl"
    if samples:
        for sample in samples:
            append_sample(samples_path, sample)
    else:
        samples_path.write_text("", encoding="utf-8")
    if sessions:
        for session in sessions:
            append_session(sessions_path, session)
    else:
        sessions_path.write_text("", encoding="utf-8")


def test_canonical_json_is_deterministic_and_rejects_non_json_values() -> None:
    assert canonical_json({"z": "ż", "a": (2, 1)}) == '{"a":[2,1],"z":"\\u017c"}'
    invalid = (math.nan, math.inf, np.int64(1), np.array([1]), {1, 2}, Path("x"))
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            canonical_json(value)


def test_identity_domains_and_all_sample_identity_fields_are_bound() -> None:
    assert identity_hash("a", {"x": 1}) != identity_hash("b", {"x": 1})
    base = sample_id(_RUN_ID, "case-a", "route-a", "measurement", 0, plan_id="plan-a")
    assert base != sample_id(
        _RUN_ID, "case-b", "route-a", "measurement", 0, plan_id="plan-a"
    )
    assert base != sample_id(
        _RUN_ID, "case-a", "route-b", "measurement", 0, plan_id="plan-a"
    )
    assert base != sample_id(
        _RUN_ID, "case-a", "route-a", "measurement", 0, plan_id="plan-b"
    )


def test_identity_constructors_are_domain_separated_and_order_stable() -> None:
    job = make_simulation_job(builtin_circuit("bell_2q"))
    network = TensorNetwork(
        circuit=job.circuit,
        tensors=(
            TensorSpec("a", (0, 1), (2, 2), "dense", "complex128"),
            TensorSpec("b", (1, 2), (2, 2), "gate", "complex128", "op"),
        ),
        output_labels=(0, 2),
        einsum_expression="ab,bc->ac",
    )
    assert len(problem_id(job)) == 64
    assert len(tensor_network_structure_id(network)) == 64
    assert environment_id({"b": 2, "a": 1}) == environment_id({"a": 1, "b": 2})
    assert validation_policy_id({"b": 2, "a": 1}) == validation_policy_id(
        {"a": 1, "b": 2}
    )
    assert executable_id({"b": 2, "a": 1}) == executable_id({"a": 1, "b": 2})
    assert environment_id({"a": 1}) != validation_policy_id({"a": 1})

    reversed_network = TensorNetwork(
        circuit=job.circuit,
        tensors=tuple(reversed(network.tensors)),
        output_labels=network.output_labels,
        einsum_expression=network.einsum_expression,
    )
    assert tensor_network_structure_id(network) == tensor_network_structure_id(
        reversed_network
    )


def test_problem_identity_uses_circuit_semantics_not_name_or_source() -> None:
    circuit = builtin_circuit("bell_2q")
    renamed = CircuitSpec(
        "different-display-name",
        circuit.n_qubits,
        circuit.operations,
        {"origin": "different-provenance"},
    )

    assert problem_id(make_simulation_job(circuit)) == problem_id(
        make_simulation_job(renamed)
    )
    assert problem_id(make_simulation_job(circuit, seed=1)) != problem_id(
        make_simulation_job(circuit, seed=2)
    )


def test_run_ids_are_unique_canonical_uuid4_values() -> None:
    first, second = new_run_id(), new_run_id()
    assert first != second
    assert UUID(first).version == 4 and str(UUID(first)) == first
    assert UUID(second).version == 4 and str(UUID(second)) == second


@pytest.mark.parametrize(
    ("validator", "record"),
    [
        (validate_manifest, _manifest()),
        (validate_sample, _sample()),
        (validate_session, _session()),
    ],
)
def test_fixed_records_reject_unknown_root_fields(validator, record) -> None:
    record["unexpected"] = True
    with pytest.raises(ValueError):
        validator(record)


def test_evidence_rejects_non_hash_identity_fields() -> None:
    manifest = _manifest()
    manifest["environment_id"] = "environment-label"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_manifest(manifest)

    sample = _sample()
    sample["identities"]["logical_plan_id"] = "logical-label"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_sample(sample)


def test_identities_and_failure_records_have_exact_fields() -> None:
    sample = _sample()
    sample["identities"]["unexpected"] = True
    with pytest.raises(ValueError):
        validate_sample(sample)

    successful = _sample()
    successful["failure"] = {"stage": "execute", "reason": "bad"}
    with pytest.raises(ValueError):
        validate_sample(successful)

    failed = _sample("failed")
    failed["failure"]["capability"] = "hardware"
    with pytest.raises(ValueError):
        validate_sample(failed)

    unsupported = _sample("unsupported")
    unsupported["failure"]["unexpected"] = True
    with pytest.raises(ValueError):
        validate_sample(unsupported)

    failed_session = _session("failed")
    failed_session["failure"]["unexpected"] = True
    with pytest.raises(ValueError):
        validate_session(failed_session)


def test_explicit_opaque_json_mappings_remain_extensible() -> None:
    manifest = _manifest()
    manifest["configuration"]["nested"] = {"future": [1, None]}
    validate_manifest(manifest)

    sample = _sample()
    sample["backend_facts"]["future"] = {"nested": [1, 2]}
    sample["numeric_facts"]["future"] = {"nested": True}
    sample["validation"] = _validation()
    validate_sample(sample)

    session = _session()
    session["terminal_backend_facts"]["future"] = {"nested": None}
    validate_session(session)


@pytest.mark.parametrize(
    "field",
    [
        "policy_reference_applicable",
        "full_precision_threshold_applicable",
        "scientific_validation_passed",
    ],
)
def test_validation_schema_requires_boolean_control_fields(field: str) -> None:
    sample = _sample()
    sample["validation"] = _validation()
    sample["validation"][field] = "true"
    with pytest.raises((TypeError, ValueError), match="validation"):
        validate_sample(sample)


def test_validation_schema_requires_applicable_pass_fields_and_conjunction() -> None:
    sample = _sample()
    report = _validation()
    report["policy_reference_passed"] = None
    sample["validation"] = report
    with pytest.raises((TypeError, ValueError), match="validation"):
        validate_sample(sample)

    report = _validation()
    report["scientific_validation_passed"] = False
    sample["validation"] = report
    with pytest.raises(ValueError, match="scientific_validation_passed"):
        validate_sample(sample)

    report = _validation()
    report["policy_reference_applicable"] = False
    report["policy_reference_passed"] = None
    report["full_precision_threshold_applicable"] = False
    report["full_precision_passed"] = None
    sample["validation"] = report
    with pytest.raises(ValueError, match="at least one"):
        validate_sample(sample)


def test_completed_artifact_rejects_successful_failed_validation() -> None:
    sample = _sample()
    sample["validation"] = _validation(passed=False)
    validate_sample(sample)

    with pytest.raises(ValueError, match="failed scientific validation"):
        validate_artifact_set(
            _manifest(status="completed"),
            [sample],
            [_session()],
        )


def test_measurement_requires_every_measurement_field() -> None:
    sample = _sample()
    assert set(sample["measurement"]) == {field.name for field in fields(Measurement)}
    sample["measurement"].pop("lowering_s")
    with pytest.raises(ValueError):
        validate_sample(sample)


def test_measurement_scope_is_frozen() -> None:
    sample = _sample()
    sample["measurement"]["scope_id"] = "sample"

    with pytest.raises(ValueError, match="measurement.scope_id"):
        validate_sample(sample)

    sample = _sample()
    sample["measurement"]["unexpected"] = None
    with pytest.raises(ValueError):
        validate_sample(sample)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_wall_s", math.inf),
        ("kernel_s", math.nan),
        ("energy_j", -0.1),
        ("h2d_bytes", True),
        ("d2h_bytes", -1),
    ],
)
def test_measurement_values_follow_measurement_contract(
    field: str, value: object
) -> None:
    sample = _sample()
    sample["measurement"][field] = value
    with pytest.raises((TypeError, ValueError)):
        validate_sample(sample)


@pytest.mark.parametrize(
    ("validator", "record"),
    [
        (validate_manifest, _manifest()),
        (validate_sample, _sample()),
        (validate_session, _session()),
    ],
)
def test_records_reject_malformed_run_uuid(validator, record) -> None:
    record["run_id"] = "not-a-uuid"
    with pytest.raises(ValueError):
        validator(record)


def test_manifest_requires_canonical_uuid4_and_rfc3339_utc_timestamp() -> None:
    validate_manifest(_manifest(created_at_utc="2026-08-23T12:00:00.123456789Z"))
    for timestamp in (
        "2026-08-23T12:00:00+00:00",
        "2026-08-23 12:00:00Z",
        "2026-02-30T12:00:00Z",
        "2026-08-23T12:00:00Zjunk",
    ):
        with pytest.raises(ValueError):
            validate_manifest(_manifest(created_at_utc=timestamp))

    version_one = _manifest(run_id="123e4567-e89b-12d3-a456-426614174000")
    with pytest.raises(ValueError):
        validate_manifest(version_one)
    uppercase = _manifest(run_id=_RUN_ID.upper())
    with pytest.raises(ValueError):
        validate_manifest(uppercase)


def test_expected_counts_are_exact_nonnegative_integers() -> None:
    for counts in (
        {"warmup": 0, "measurement": 1},
        {"warmup": 0, "measurement": 1, "sessions": 1, "extra": 0},
        {"warmup": 0, "measurement": 1, "sessions": -1},
        {"warmup": False, "measurement": 1, "sessions": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            validate_manifest(_manifest(expected_counts=counts))


@pytest.mark.parametrize(
    ("attempted", "succeeded", "verified"),
    [(False, True, False), (True, False, True)],
)
def test_session_rejects_impossible_release_flags(
    attempted: bool, succeeded: bool, verified: bool
) -> None:
    session = _session(
        "failed",
        release_attempted=attempted,
        release_succeeded=succeeded,
        release_verified=verified,
    )
    with pytest.raises(ValueError):
        validate_session(session)


def test_successful_session_requires_verified_release() -> None:
    session = _session(
        release_attempted=True,
        release_succeeded=True,
        release_verified=False,
    )
    with pytest.raises(ValueError):
        validate_session(session)
    validate_session(
        _session(
            "failed",
            release_attempted=False,
            release_succeeded=False,
            release_verified=False,
        )
    )


@pytest.mark.parametrize("status", ["success", "failed", "unsupported"])
def test_sample_status_schemas(status: str) -> None:
    validate_sample(_sample(status))


def test_full_state_sample_allows_null_tensor_network_and_logical_plan() -> None:
    sample = _sample(plan_id=None)
    sample["identities"]["tensor_network_structure_id"] = None
    sample["identities"]["logical_plan_id"] = None
    validate_sample(sample)


def test_sample_rejects_half_null_tensor_network_identity_pair() -> None:
    sample = _sample()
    sample["identities"]["logical_plan_id"] = None
    with pytest.raises(ValueError, match="null together"):
        validate_sample(sample)


def test_experiment_argument_validation_uses_same_identity_pair_rule(
    tmp_path: Path,
) -> None:
    identities = _sample()["identities"]
    identities["tensor_network_structure_id"] = None
    with pytest.raises(ValueError, match="null together"):
        run_direct_samples(
            run_id=_RUN_ID,
            experiment_id=_EXPERIMENT_ID,
            case_id="case-1",
            route_id="route-1",
            identities=identities,
            warmups=0,
            repetitions=1,
            run_once=lambda: None,
            samples_path=tmp_path / "samples.jsonl",
        )


def test_require_matching_scope_accepts_same_scope_samples() -> None:
    samples = [_sample(index=0), _sample(index=1)]

    assert require_matching_scope(samples) == "steady_execution_v1"
    assert (
        require_matching_scope(
            (sample for sample in samples), expected_scope_id="steady_execution_v1"
        )
        == "steady_execution_v1"
    )


def test_require_matching_scope_rejects_mixed_scopes() -> None:
    samples = [_sample(index=0), _sample(index=1)]
    samples[1]["measurement"]["scope_id"] = "simulation_end_to_end_v1"

    with pytest.raises(ValueError, match="one scope"):
        require_matching_scope(samples)


@pytest.mark.parametrize(
    "sample",
    [_sample(kind="warmup"), _sample("failed")],
)
def test_require_matching_scope_rejects_warmup_and_failure(sample) -> None:
    with pytest.raises(ValueError):
        require_matching_scope([_sample(index=0), sample])


def test_require_matching_scope_rejects_fewer_than_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        require_matching_scope([_sample()])


def test_require_matching_scope_rejects_invalid_expected_scope() -> None:
    with pytest.raises(ValueError, match="expected_scope_id"):
        require_matching_scope(
            [_sample(index=0), _sample(index=1)], expected_scope_id="sample"
        )


def test_manifest_files_are_exact() -> None:
    manifest = _manifest()
    manifest["files"]["extra"] = "extra.json"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_canonical_writers_do_not_change_files_for_invalid_records(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "nested" / "manifest.json"
    write_manifest(manifest_path, _manifest())
    original_manifest = manifest_path.read_bytes()
    assert original_manifest.endswith(b"\n")
    with pytest.raises(ValueError):
        write_manifest(manifest_path, {**_manifest(), "source_commit": "bad"})
    assert manifest_path.read_bytes() == original_manifest

    samples_path = tmp_path / "nested" / "samples.jsonl"
    append_sample(samples_path, _sample())
    original_samples = samples_path.read_bytes()
    with pytest.raises(ValueError):
        append_sample(samples_path, {**_sample(), "sample_id": "bad"})
    assert samples_path.read_bytes() == original_samples


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_public_manifest_writer_rejects_terminal_status(
    tmp_path: Path, status: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, _manifest())
    original = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="require status running"):
        write_manifest(manifest_path, _manifest(status=status))

    assert manifest_path.read_bytes() == original


def test_jsonl_append_failures_and_partial_files_leave_original_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples_path = tmp_path / "samples.jsonl"
    append_sample(samples_path, _sample())
    original = samples_path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(canonical.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        append_sample(samples_path, _sample(index=1))
    assert samples_path.read_bytes() == original

    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_bytes(b'{"incomplete":')
    partial = partial_path.read_bytes()
    with pytest.raises(ValueError, match="newline-terminated"):
        append_sample(partial_path, _sample())
    assert partial_path.read_bytes() == partial


def test_artifact_set_accepts_valid_completed_records() -> None:
    validate_artifact_set(
        _manifest(status="completed"),
        [_sample()],
        [_session()],
    )


def test_completed_artifact_rejects_mixed_route_timing_scopes() -> None:
    first = _sample(index=0, session_instance_id=None)
    second = _sample(index=1, session_instance_id=None)
    second["measurement"]["scope_id"] = "simulation_end_to_end_v1"
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 2, "sessions": 0},
    )

    with pytest.raises(ValueError, match="one timing scope per case and route"):
        validate_artifact_set(manifest, [first, second], [])


def test_artifact_set_rejects_duplicate_or_conflicting_ids() -> None:
    first_sample = _sample()
    conflicting_sample = deepcopy(first_sample)
    conflicting_sample["backend_facts"] = {"provider": "different"}
    duplicate_sample_manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 2, "sessions": 1},
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_artifact_set(
            duplicate_sample_manifest,
            [first_sample, conflicting_sample],
            [_session()],
        )

    conflicting_session = _session(route_id="route-2")
    duplicate_session_manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 2},
    )
    with pytest.raises(ValueError, match="duplicate session_instance_id"):
        validate_artifact_set(
            duplicate_session_manifest,
            [_sample()],
            [_session(), conflicting_session],
        )


@pytest.mark.parametrize(
    "sample",
    [
        _sample(run_id=_OTHER_RUN_ID, session_instance_id=None),
        _sample(experiment_id="f" * 64, session_instance_id=None),
        _sample(environment_id="a" * 64, session_instance_id=None),
        _sample(validation_policy_id="b" * 64, session_instance_id=None),
    ],
)
def test_artifact_set_rejects_sample_context_mismatch(sample) -> None:
    with pytest.raises(ValueError, match="does not match manifest"):
        validate_artifact_set(_manifest(status="failed"), [sample], [])


def test_artifact_set_rejects_session_context_and_linkage_mismatches() -> None:
    with pytest.raises(ValueError, match="session run_id does not match manifest"):
        validate_artifact_set(
            _manifest(status="failed"),
            [],
            [_session(run_id=_OTHER_RUN_ID)],
        )

    with pytest.raises(ValueError, match="linked session route_id"):
        validate_artifact_set(
            _manifest(status="failed"),
            [_sample()],
            [_session(route_id="route-2")],
        )

    with pytest.raises(ValueError, match="linked session plan_id"):
        validate_artifact_set(
            _manifest(status="failed"),
            [_sample(plan_id="plan-a")],
            [_session(plan_id="plan-b")],
        )

    with pytest.raises(ValueError, match="no matching session"):
        validate_artifact_set(_manifest(status="failed"), [_sample()], [])


def test_artifact_set_rejects_running_status_and_expected_count_mismatch() -> None:
    with pytest.raises(ValueError, match="completed or failed"):
        validate_artifact_set(_manifest(), [_sample()], [_session()])

    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 1, "measurement": 1, "sessions": 1},
    )
    with pytest.raises(ValueError, match="counts do not match"):
        validate_artifact_set(manifest, [_sample()], [_session()])

    failed_manifest = _manifest(
        status="failed",
        expected_counts={"warmup": 0, "measurement": 0, "sessions": 0},
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate_artifact_set(
            failed_manifest,
            [_sample(session_instance_id=None)],
            [],
        )


def test_completed_artifacts_reject_failed_sample_or_session() -> None:
    with pytest.raises(ValueError, match="every sample"):
        validate_artifact_set(
            _manifest(status="completed"),
            [_sample("failed")],
            [_session()],
        )

    with pytest.raises(ValueError, match="successful sessions"):
        validate_artifact_set(
            _manifest(status="completed"),
            [_sample()],
            [_session("failed")],
        )


def test_successful_finalization_rewrites_only_manifest_status(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    _write_artifact_files(directory, _manifest(), [_sample()], [_session()])
    samples_before = (directory / "samples.jsonl").read_bytes()
    sessions_before = (directory / "sessions.jsonl").read_bytes()

    finalize_artifacts(directory, status="completed")

    finalized = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert finalized["status"] == "completed"
    assert (directory / "manifest.json").read_bytes().endswith(b"\n")
    assert (directory / "samples.jsonl").read_bytes() == samples_before
    assert (directory / "sessions.jsonl").read_bytes() == sessions_before


def test_terminal_artifact_rejects_later_record_appends(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    _write_artifact_files(directory, _manifest(), [_sample()], [_session()])
    finalize_artifacts(directory, status="completed")
    samples_path = directory / "samples.jsonl"
    sessions_path = directory / "sessions.jsonl"
    samples_before = samples_path.read_bytes()
    sessions_before = sessions_path.read_bytes()

    with pytest.raises(ValueError, match="requires a running manifest"):
        append_sample(
            samples_path,
            _sample(index=1, session_instance_id=None),
        )
    with pytest.raises(ValueError, match="requires a running manifest"):
        append_session(
            sessions_path,
            _session(session_instance_id="session-2"),
        )

    assert samples_path.read_bytes() == samples_before
    assert sessions_path.read_bytes() == sessions_before


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_finalization_requires_running_manifest(tmp_path: Path, status: str) -> None:
    directory = tmp_path / "evidence"
    _write_artifact_files(directory, _manifest(), [_sample()], [_session()])
    finalize_artifacts(directory, status="completed")
    manifest_path = directory / "manifest.json"
    finalized = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="requires a running manifest"):
        finalize_artifacts(directory, status=status)

    assert manifest_path.read_bytes() == finalized


def test_failed_finalization_accepts_empty_incomplete_artifacts(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    manifest = _manifest(expected_counts={"warmup": 2, "measurement": 3, "sessions": 2})
    _write_artifact_files(directory, manifest, [], [])

    finalize_artifacts(directory, status="failed")

    finalized = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert finalized["status"] == "failed"


def test_invalid_finalization_leaves_running_manifest_unchanged(tmp_path: Path) -> None:
    directory = tmp_path / "evidence"
    _write_artifact_files(directory, _manifest(), [], [])
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="counts do not match"):
        finalize_artifacts(directory, status="completed")

    assert manifest_path.read_bytes() == original
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "running"


@pytest.mark.parametrize(
    "invalid_samples",
    ["{bad json}\n", "\n", "[]\n", '{"a":1,"a":2}\n'],
)
def test_finalizer_strictly_rejects_invalid_json_lines_without_rewrite(
    tmp_path: Path, invalid_samples: str
) -> None:
    directory = tmp_path / "evidence"
    manifest = _manifest(expected_counts={"warmup": 0, "measurement": 0, "sessions": 0})
    _write_artifact_files(directory, manifest, [], [])
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    (directory / "samples.jsonl").write_text(invalid_samples, encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        finalize_artifacts(directory, status="failed")

    assert manifest_path.read_bytes() == original
