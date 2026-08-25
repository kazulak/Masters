from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import errno
import json
import math
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import quantum_bench.evidence as canonical
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
import quantum_bench.experiment as experiment
from quantum_bench.experiment import run_direct_samples, run_session_samples
from quantum_bench.model import (
    CircuitSpec,
    TensorNetwork,
    TensorSpec,
    make_simulation_job,
)
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    Measurement,
    UnsupportedExecution,
)


_RUN_ID = new_run_id()
_OTHER_RUN_ID = new_run_id()
_EXPERIMENT_ID = "e" * 64
_ENVIRONMENT_ID = "d" * 64
_POLICY_ID = "c" * 64
_SESSION_ID = "session-1"
_LIFECYCLE_RUN_ID = "12345678-1234-4234-8234-1234567890ab"


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
        "accuracy_qualified": False,
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
        "configuration": {
            "b": 2,
            "a": [True, None],
            "identity_bindings": [
                {
                    "case_id": "case-1",
                    "plan_id": "plan-1",
                    "route_id": "route-1",
                    "problem_id": "1" * 64,
                    "tensor_network_structure_id": "2" * 64,
                    "logical_plan_id": "3" * 64,
                    "physical_plan_id": None,
                    "executable_id": None,
                    "environment_id": _ENVIRONMENT_ID,
                    "validation_policy_id": _POLICY_ID,
                }
            ],
        },
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
        "schema_version": "evidence_sample_v2",
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


def _execution_sample(
    *,
    output: np.ndarray | None = None,
    measurement: Measurement | None = None,
) -> ExecutionSample:
    return ExecutionSample(
        output=np.array([1, 2] if output is None else output),
        measurement=measurement
        or Measurement(scope_id="steady_execution_v1", total_wall_s=1.0),
        backend_facts={"backend": "test", "nested": (1, True)},
        numeric_facts={"value": 1},
    )


def _execution_validation(
    *,
    policy_passed: bool = True,
    full_precision_applicable: bool = False,
    full_precision_passed: bool | None = None,
) -> dict[str, object]:
    return {
        "policy_reference_applicable": True,
        "policy_reference_passed": policy_passed,
        "full_precision_threshold_applicable": full_precision_applicable,
        "full_precision_passed": full_precision_passed,
        "accuracy_qualified": full_precision_applicable
        and full_precision_passed is True,
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


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
        "accuracy_qualified",
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
    report["accuracy_qualified"] = True
    sample["validation"] = report
    with pytest.raises(ValueError, match="accuracy_qualified"):
        validate_sample(sample)

    report = _validation()
    report["policy_reference_applicable"] = False
    report["policy_reference_passed"] = None
    report["full_precision_threshold_applicable"] = False
    report["full_precision_passed"] = None
    sample["validation"] = report
    with pytest.raises(ValueError, match="at least one"):
        validate_sample(sample)


def test_completed_artifact_accepts_successful_unqualified_validation() -> None:
    sample = _sample()
    sample["validation"] = _validation(passed=False)
    validate_sample(sample)

    validate_artifact_set(_manifest(status="completed"), [sample], [_session()])


def test_sample_v1_is_rejected() -> None:
    sample = _sample()
    sample["schema_version"] = "evidence_sample_v1"
    with pytest.raises(ValueError, match="invalid schema_version"):
        validate_sample(sample)


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


def test_directory_fsync_is_skipped_on_non_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(canonical.os, "name", "nt")

    def fail_open(*args: object, **kwargs: object) -> int:
        raise AssertionError("non-POSIX directory fsync must not open a directory")

    monkeypatch.setattr(canonical.os, "open", fail_open)
    canonical._fsync_parent_directory(tmp_path)


def test_directory_fsync_uses_directory_descriptor_when_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(canonical.os, "name", "posix")

    def fake_open(path: object, flags: int) -> int:
        assert path == tmp_path
        calls.append(("open", flags))
        return 41

    monkeypatch.setattr(canonical.os, "open", fake_open)
    monkeypatch.setattr(canonical.os, "fsync", lambda descriptor: calls.append(("fsync", descriptor)))
    monkeypatch.setattr(canonical.os, "close", lambda descriptor: calls.append(("close", descriptor)))

    canonical._fsync_parent_directory(tmp_path)

    assert calls == [
        ("open", canonical.os.O_RDONLY | getattr(canonical.os, "O_DIRECTORY", 0)),
        ("fsync", 41),
        ("close", 41),
    ]


@pytest.mark.parametrize(
    "unsupported_errno", sorted(canonical._DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS)
)
def test_directory_fsync_suppresses_only_supported_filesystem_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_errno: int
) -> None:
    monkeypatch.setattr(canonical.os, "name", "posix")

    def fail_open(*args: object, **kwargs: object) -> int:
        raise OSError(unsupported_errno, "unsupported directory sync")

    monkeypatch.setattr(canonical.os, "open", fail_open)
    canonical._fsync_parent_directory(tmp_path)


@pytest.mark.parametrize(
    "unsupported_errno", sorted(canonical._DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS)
)
def test_directory_fsync_suppresses_supported_sync_errors_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_errno: int
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(canonical.os, "name", "posix")
    monkeypatch.setattr(canonical.os, "open", lambda path, flags: 47)

    def fail_fsync(descriptor: int) -> None:
        raise OSError(unsupported_errno, "unsupported directory sync")

    monkeypatch.setattr(canonical.os, "fsync", fail_fsync)
    monkeypatch.setattr(canonical.os, "close", closed.append)

    canonical._fsync_parent_directory(tmp_path)

    assert closed == [47]


def test_directory_fsync_propagates_unrelated_errors_and_closes_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(canonical.os, "name", "posix")
    monkeypatch.setattr(canonical.os, "open", lambda path, flags: 43)

    def fail_fsync(descriptor: int) -> None:
        raise OSError(errno.EBADF, "bad descriptor")

    monkeypatch.setattr(canonical.os, "fsync", fail_fsync)
    monkeypatch.setattr(canonical.os, "close", closed.append)

    with pytest.raises(OSError) as error:
        canonical._fsync_parent_directory(tmp_path)
    assert error.value.errno == errno.EBADF
    assert closed == [43]


def test_artifact_set_accepts_valid_completed_records() -> None:
    validate_artifact_set(
        _manifest(status="completed"),
        [_sample()],
        [_session()],
    )


def test_completed_artifact_rejects_undeclared_route_or_identity_tampering() -> None:
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 0},
    )
    undeclared = _sample(route_id="route-2", session_instance_id=None)
    with pytest.raises(ValueError, match="not declared by identity_bindings"):
        validate_artifact_set(manifest, [undeclared], [])

    session_manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 1},
    )
    with pytest.raises(ValueError, match="session route is not declared"):
        validate_artifact_set(
            session_manifest,
            [_sample()],
            [_session(route_id="route-2")],
        )

    for field, value in (
        ("logical_plan_id", "4" * 64),
        ("physical_plan_id", "5" * 64),
        ("executable_id", "6" * 64),
    ):
        binding = manifest["configuration"]["identity_bindings"][0]
        original = binding[field]
        binding[field] = value
        sample = _sample(session_instance_id=None)
        sample["identities"][field] = value
        sample["identities"][field] = "7" * 64
        with pytest.raises(ValueError, match=f"identities.{field}"):
            validate_artifact_set(manifest, [sample], [])
        binding[field] = original


def test_completed_artifact_rejects_duplicate_or_missing_identity_bindings() -> None:
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 0},
    )
    binding = deepcopy(manifest["configuration"]["identity_bindings"][0])
    manifest["configuration"]["identity_bindings"].append(binding)
    with pytest.raises(ValueError, match="duplicate identity_binding"):
        validate_artifact_set(manifest, [_sample(session_instance_id=None)], [])

    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 0},
    )
    manifest["configuration"]["identity_bindings"] = []
    with pytest.raises(ValueError, match="not declared by identity_bindings"):
        validate_artifact_set(manifest, [_sample(session_instance_id=None)], [])


def test_completed_artifact_requires_every_configured_matrix_route() -> None:
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 0},
    )
    manifest["configuration"]["experiment"] = {
        "matrix": [
            {
                "case_id": "case-1",
                "plan_id": "plan-1",
                "route_ids": ["route-1", "route-2"],
            }
        ]
    }

    with pytest.raises(ValueError, match="cover exactly declared routes"):
        validate_artifact_set(manifest, [_sample(session_instance_id=None)], [])


def test_failed_artifact_binds_observed_subset_without_requiring_unattempted_routes() -> None:
    manifest = _manifest(
        status="failed",
        expected_counts={"warmup": 0, "measurement": 2, "sessions": 0},
    )
    manifest["configuration"]["experiment"] = {
        "matrix": [
            {
                "case_id": "case-1",
                "plan_id": "plan-1",
                "route_ids": ["route-1", "route-2"],
            }
        ]
    }
    sample = _sample(session_instance_id=None)
    validate_artifact_set(manifest, [sample], [])

    tampered = deepcopy(sample)
    tampered["identities"]["logical_plan_id"] = "7" * 64
    with pytest.raises(ValueError, match="identities.logical_plan_id"):
        validate_artifact_set(manifest, [tampered], [])

    undeclared = _sample(route_id="route-3", session_instance_id=None)
    with pytest.raises(ValueError, match="not declared by identity_bindings"):
        validate_artifact_set(manifest, [undeclared], [])


def test_completed_artifact_binds_mixed_direct_and_session_routes() -> None:
    direct = _sample(session_instance_id=None)
    session = _session(
        case_id="case-2",
        plan_id="plan-2",
        route_id="route-2",
        session_instance_id="session-2",
    )
    upmem = _sample(
        case_id="case-2",
        plan_id="plan-2",
        route_id="route-2",
        session_instance_id="session-2",
    )
    upmem["identities"]["physical_plan_id"] = "4" * 64
    upmem["identities"]["executable_id"] = "5" * 64
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 2, "sessions": 1},
    )
    manifest["configuration"]["identity_bindings"].append(
        {
            "case_id": "case-2",
            "plan_id": "plan-2",
            "route_id": "route-2",
            **upmem["identities"],
        }
    )
    validate_artifact_set(manifest, [direct, upmem], [session])


def test_completed_artifact_accepts_planless_direct_route_binding() -> None:
    direct = _sample(plan_id=None, session_instance_id=None)
    direct["identities"]["tensor_network_structure_id"] = None
    direct["identities"]["logical_plan_id"] = None
    manifest = _manifest(
        status="completed",
        expected_counts={"warmup": 0, "measurement": 1, "sessions": 0},
    )
    manifest["configuration"]["identity_bindings"] = [
        {
            "case_id": direct["case_id"],
            "plan_id": None,
            "route_id": direct["route_id"],
            **direct["identities"],
        }
    ]
    validate_artifact_set(manifest, [direct], [])


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

    route_manifest = _manifest(status="failed")
    route_binding = deepcopy(route_manifest["configuration"]["identity_bindings"][0])
    route_binding["route_id"] = "route-2"
    route_manifest["configuration"]["identity_bindings"].append(route_binding)
    with pytest.raises(ValueError, match="linked session route_id"):
        validate_artifact_set(
            route_manifest,
            [_sample()],
            [_session(route_id="route-2")],
        )

    plan_manifest = _manifest(status="failed")
    plan_manifest["configuration"]["identity_bindings"][0]["plan_id"] = "plan-a"
    plan_binding = deepcopy(plan_manifest["configuration"]["identity_bindings"][0])
    plan_binding["plan_id"] = "plan-b"
    plan_manifest["configuration"]["identity_bindings"].append(plan_binding)
    with pytest.raises(ValueError, match="linked session plan_id"):
        validate_artifact_set(
            plan_manifest,
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


def test_direct_lifecycle_orders_samples_and_validates_each_attempt(
    tmp_path: Path,
) -> None:
    calls = 0
    validated: list[ExecutionSample] = []

    def run_once() -> ExecutionSample:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ExecutionFailed("kernel", "no result", {"rank": 1})
        return _execution_sample()

    def validate(sample: ExecutionSample) -> dict[str, object]:
        validated.append(sample)
        return _execution_validation()

    rows = run_direct_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=_sample()["identities"],
        warmups=1,
        repetitions=2,
        run_once=run_once,
        samples_path=tmp_path / "samples.jsonl",
        validate=validate,
    )

    assert [(row["sample_kind"], row["sample_index"]) for row in rows] == [
        ("warmup", 0),
        ("measurement", 0),
        ("measurement", 1),
    ]
    assert len(validated) == 2
    assert rows[0]["status"] == "success"
    assert rows[1]["status"] == "failed"
    assert rows[1]["failure"] == {"stage": "kernel", "reason": "no result"}
    assert rows[2]["status"] == "success"
    assert rows[0]["measurement"]["session_open_s"] is None
    assert all(
        value is None
        for key, value in rows[0]["measurement"].items()
        if key not in {"scope_id", "total_wall_s"}
    )


def test_direct_validation_failures_are_bounded_and_preserve_facts(
    tmp_path: Path,
) -> None:
    failed = run_direct_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="failed-validation",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        run_once=_execution_sample,
        samples_path=tmp_path / "failed.jsonl",
        validate=lambda sample: _execution_validation(policy_passed=False),
    )[0]
    assert failed["status"] == "success"
    assert failed["measurement"]["total_wall_s"] == 1.0
    assert failed["output_sha256"] is not None
    assert failed["backend_facts"] == {"backend": "test", "nested": [1, True]}
    assert failed["numeric_facts"] == {"value": 1}
    assert failed["validation"]["policy_reference_passed"] is False
    assert failed["validation"]["accuracy_qualified"] is False
    assert failed["failure"] is None

    def invalid_validator(sample: ExecutionSample) -> dict[str, object]:
        raise RuntimeError("bad\n" + ("x" * 1000))

    bounded = run_direct_samples(
        run_id="22345678-1234-4234-8234-1234567890ab",
        experiment_id=_EXPERIMENT_ID,
        case_id="validator-error",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        run_once=_execution_sample,
        samples_path=tmp_path / "validator.jsonl",
        validate=invalid_validator,
    )[0]
    assert bounded["status"] == "failed"
    assert bounded["validation"] is None
    assert "traceback" not in bounded["failure"]["reason"].lower()
    assert len(bounded["failure"]["reason"]) <= 256


def test_direct_timing_scope_errors_produce_failure_rows(tmp_path: Path) -> None:
    rows = run_direct_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        run_once=lambda: _execution_sample(
            measurement=Measurement(scope_id="invalid_scope", total_wall_s=1.0)
        ),
        samples_path=tmp_path / "samples.jsonl",
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure"] == {
        "stage": "timing_contract",
        "reason": "samples require a frozen timing scope",
    }
    assert rows[0]["measurement"] is rows[0]["output_sha256"] is None


def test_output_hash_is_shape_dtype_and_endianness_stable(tmp_path: Path) -> None:
    native = np.array([1.5, 2.5], dtype=np.float32)
    big_endian = native.astype(">f4")
    rows = [
        run_direct_samples(
            run_id=f"{index + 3}2345678-1234-4234-8234-1234567890ab",
            experiment_id=_EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=_sample()["identities"],
            warmups=0,
            repetitions=1,
            run_once=lambda output=output: _execution_sample(output=output),
            samples_path=tmp_path / f"samples-{index}.jsonl",
        )[0]
        for index, output in enumerate(
            (
                native,
                big_endian,
                np.array([[1.5, 2.5]], dtype=np.float32),
                np.array([1.5, 2.5], dtype=np.float64),
            )
        )
    ]
    assert rows[0]["output_sha256"] == rows[1]["output_sha256"]
    assert len({row["output_sha256"] for row in rows}) == 3

    for index, output in enumerate(
        (
            np.array([object()], dtype=object),
            np.array(["text"], dtype="U4"),
            np.array([(1,)], dtype=[("value", "i4")]),
            np.array([np.nan], dtype=np.float64),
        )
    ):
        row = run_direct_samples(
            run_id=f"{index + 6}2345678-1234-4234-8234-1234567890ab",
            experiment_id=_EXPERIMENT_ID,
            case_id="invalid-output",
            route_id="route",
            identities=_sample()["identities"],
            warmups=0,
            repetitions=1,
            run_once=lambda output=output: _execution_sample(output=output),
            samples_path=tmp_path / f"invalid-{index}.jsonl",
        )[0]
        assert row["status"] == "failed"
        assert row["failure"]["stage"] == "execution"


def test_persistent_session_validates_each_sample_and_closes_once(
    tmp_path: Path,
) -> None:
    class Session:
        def __init__(self) -> None:
            self.runs = 0
            self.closes = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            assert inputs == {"input": "value"}
            self.runs += 1
            return _execution_sample()

        def close(self) -> dict[str, object]:
            self.closes += 1
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    session = Session()
    validated: list[ExecutionSample] = []
    rows, session_row = run_session_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=_sample()["identities"],
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: session,
        inputs={"input": "value"},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
        validate=lambda sample: validated.append(sample) or _execution_validation(),
    )
    assert len(rows) == len(validated) == session.runs == 2
    assert session.closes == 1
    assert session_row["status"] == "success"
    assert rows[0]["measurement"]["session_open_s"] is None
    assert session_row["open_s"] is not None
    assert session_row["session_close_s"] is not None


def test_session_open_and_run_failures_stop_later_attempts(tmp_path: Path) -> None:
    calls = 0

    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            nonlocal calls
            calls += 1
            raise UnsupportedExecution("preflight", "unsupported route", "device")

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    samples_path = tmp_path / "samples.jsonl"
    sessions_path = tmp_path / "sessions.jsonl"
    rows, session_row = run_session_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="unsupported",
        route_id="route",
        identities=_sample()["identities"],
        warmups=2,
        repetitions=2,
        session_protocol_id="protocol",
        open_session=Session,
        inputs={},
        samples_path=samples_path,
        sessions_path=sessions_path,
    )
    assert len(rows) == calls == 1
    assert session_row["failure"] == {
        "stage": "preflight",
        "reason": "unsupported route",
    }

    _, open_row = run_session_samples(
        run_id="32345678-1234-4234-8234-1234567890ab",
        experiment_id=_EXPERIMENT_ID,
        case_id="open-failure",
        route_id="route",
        identities=_sample()["identities"],
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: (_ for _ in ()).throw(
            ExecutionFailed("connect", "refused", {"host": "x"})
        ),
        inputs={},
        samples_path=samples_path,
        sessions_path=sessions_path,
    )
    assert open_row["failure"] == {"stage": "connect", "reason": "refused"}
    assert len(_read_jsonl(samples_path)) == 1


def test_session_timing_and_execution_failures_stop_the_session(
    tmp_path: Path,
) -> None:
    class TimingSession:
        calls = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            self.calls += 1
            return _execution_sample(
                measurement=Measurement(
                    scope_id="steady_execution_v1", total_wall_s=1.0, mapping_s=0.1
                )
            )

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    timing_session = TimingSession()
    rows, session_row = run_session_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="timing",
        route_id="route",
        identities=_sample()["identities"],
        warmups=2,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: timing_session,
        inputs={},
        samples_path=tmp_path / "timing-samples.jsonl",
        sessions_path=tmp_path / "timing-sessions.jsonl",
    )
    assert timing_session.calls == len(rows) == 1
    assert rows[0]["failure"]["stage"] == "timing_contract"
    assert session_row["failure"]["stage"] == "timing_contract"

    class FailedSession:
        def __init__(self, outcome: Exception) -> None:
            self.outcome = outcome
            self.calls = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            self.calls += 1
            raise self.outcome

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    for index, outcome, failure in (
        (
            0,
            ExecutionFailed("kernel", "failed", {"rank": 1}),
            {"stage": "kernel", "reason": "failed"},
        ),
        (
            1,
            RuntimeError("boom"),
            {"stage": "execution", "reason": "RuntimeError: boom"},
        ),
    ):
        failed_session = FailedSession(outcome)
        rows, session_row = run_session_samples(
            run_id=f"{index + 4}2345678-1234-4234-8234-1234567890ab",
            experiment_id=_EXPERIMENT_ID,
            case_id=f"failure-{index}",
            route_id="route",
            identities=_sample()["identities"],
            warmups=1,
            repetitions=1,
            session_protocol_id="protocol",
            open_session=lambda failed_session=failed_session: failed_session,
            inputs={},
            samples_path=tmp_path / f"failure-{index}-samples.jsonl",
            sessions_path=tmp_path / f"failure-{index}-sessions.jsonl",
        )
        assert failed_session.calls == len(rows) == 1
        assert rows[0]["failure"] == session_row["failure"] == failure


def test_invalid_ids_and_collisions_are_rejected_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct_calls = 0
    open_calls = 0
    samples_path = tmp_path / "samples.jsonl"
    sessions_path = tmp_path / "sessions.jsonl"
    identities = _sample()["identities"]

    with pytest.raises(ValueError, match="canonical UUID4"):
        run_direct_samples(
            run_id="not-a-uuid",
            experiment_id=_EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=identities,
            warmups=0,
            repetitions=1,
            run_once=lambda: (_ for _ in ()).throw(AssertionError()),
            samples_path=samples_path,
        )

    def run_once() -> ExecutionSample:
        nonlocal direct_calls
        direct_calls += 1
        return _execution_sample()

    direct_kwargs = {
        "run_id": _LIFECYCLE_RUN_ID,
        "experiment_id": _EXPERIMENT_ID,
        "case_id": "case",
        "route_id": "route",
        "identities": identities,
        "warmups": 0,
        "repetitions": 1,
        "run_once": run_once,
        "samples_path": samples_path,
    }
    run_direct_samples(**direct_kwargs)
    original = samples_path.read_bytes()
    with pytest.raises(ValueError, match="planned sample IDs"):
        run_direct_samples(**direct_kwargs)
    assert direct_calls == 1
    assert samples_path.read_bytes() == original

    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _execution_sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    def open_session() -> Session:
        nonlocal open_calls
        open_calls += 1
        return Session()

    session_kwargs = {
        "run_id": _LIFECYCLE_RUN_ID,
        "experiment_id": _EXPERIMENT_ID,
        "case_id": "session-case",
        "route_id": "route",
        "identities": identities,
        "warmups": 0,
        "repetitions": 1,
        "session_protocol_id": "protocol",
        "open_session": open_session,
        "inputs": {},
        "samples_path": samples_path,
        "sessions_path": sessions_path,
    }
    run_session_samples(**session_kwargs)
    original_sessions = sessions_path.read_bytes()
    with pytest.raises(ValueError, match="planned sample IDs"):
        run_session_samples(**session_kwargs)
    assert open_calls == 1
    assert sessions_path.read_bytes() == original_sessions

    _, first_session = run_session_samples(
        **{**session_kwargs, "case_id": "first", "route_id": "first"}
    )
    monkeypatch.setattr(
        experiment, "uuid4", lambda: first_session["session_instance_id"]
    )
    with pytest.raises(ValueError, match="session_instance_id already exists"):
        run_session_samples(
            **{**session_kwargs, "case_id": "second", "route_id": "second"}
        )


def test_direct_unsupported_and_unexpected_failures_are_distinguished(
    tmp_path: Path,
) -> None:
    outcomes = iter(
        (
            UnsupportedExecution("preflight", "missing device", "accelerator"),
            RuntimeError("boom"),
        )
    )

    def run_once() -> ExecutionSample:
        raise next(outcomes)

    rows = run_direct_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=2,
        run_once=run_once,
        samples_path=tmp_path / "samples.jsonl",
    )
    assert rows[0]["status"] == "unsupported"
    assert rows[0]["failure"] == {
        "stage": "preflight",
        "reason": "missing device",
        "capability": "accelerator",
    }
    assert rows[1]["status"] == "failed"
    assert rows[1]["failure"] == {
        "stage": "execution",
        "reason": "RuntimeError: boom",
    }
    assert rows[0]["backend_facts"] == rows[0]["numeric_facts"] == {}
    assert rows[1]["backend_facts"] == rows[1]["numeric_facts"] == {}


def test_session_release_failures_and_contradictions_are_normalized(
    tmp_path: Path,
) -> None:
    class CloseFailure:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _execution_sample()

        def close(self) -> dict[str, object]:
            raise ExecutionFailed(
                "release", "release failed", {"hardware_release_attempted": True}
            )

    _, close_row = run_session_samples(
        run_id=_LIFECYCLE_RUN_ID,
        experiment_id=_EXPERIMENT_ID,
        case_id="close-failure",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=CloseFailure,
        inputs={},
        samples_path=tmp_path / "close-samples.jsonl",
        sessions_path=tmp_path / "close-sessions.jsonl",
    )
    assert close_row["failure"] == {"stage": "release", "reason": "release failed"}
    assert close_row["release_attempted"] is True
    assert close_row["release_succeeded"] is False
    assert close_row["release_verified"] is False

    class BadRelease:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _execution_sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": False,
                "hardware_release_verified": False,
            }

    _, release_row = run_session_samples(
        run_id="32345678-1234-4234-8234-1234567890ab",
        experiment_id=_EXPERIMENT_ID,
        case_id="release-failure",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=BadRelease,
        inputs={},
        samples_path=tmp_path / "release-samples.jsonl",
        sessions_path=tmp_path / "release-sessions.jsonl",
    )
    assert release_row["failure"] == {
        "stage": "session_close",
        "reason": "hardware release was not fully verified",
    }

    class ContradictoryRelease:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _execution_sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": False,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    sessions_path = tmp_path / "contradictory-sessions.jsonl"
    _, contradictory = run_session_samples(
        run_id="42345678-1234-4234-8234-1234567890ab",
        experiment_id=_EXPERIMENT_ID,
        case_id="contradictory",
        route_id="route",
        identities=_sample()["identities"],
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=ContradictoryRelease,
        inputs={},
        samples_path=tmp_path / "contradictory-samples.jsonl",
        sessions_path=sessions_path,
    )
    assert contradictory["failure"] == {
        "stage": "session_close",
        "reason": "hardware release facts are inconsistent",
    }
    assert contradictory["terminal_backend_facts"] == {
        "hardware_release_attempted": False,
        "hardware_release_succeeded": True,
        "hardware_release_verified": True,
    }
    assert (
        contradictory["release_attempted"],
        contradictory["release_succeeded"],
        contradictory["release_verified"],
    ) == (False, False, False)
    assert _read_jsonl(sessions_path) == [contradictory]
