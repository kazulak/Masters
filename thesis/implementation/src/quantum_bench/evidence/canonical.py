"""Canonical serialization and validation for experiment evidence records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from datetime import datetime
from enum import Enum as _Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable
from uuid import UUID, uuid4

from quantum_bench.model import SimulationJob, TensorNetwork
from quantum_bench.results import Measurement as _Measurement


_MANIFEST_SCHEMA = "evidence_manifest_v1"
_SAMPLE_SCHEMA = "evidence_sample_v1"
_SESSION_SCHEMA = "evidence_session_v1"
_SAMPLE_KINDS = frozenset({"warmup", "measurement"})
_SAMPLE_STATUSES = frozenset({"success", "unsupported", "failed"})
_TIMING_SCOPES = frozenset({"simulation_end_to_end_v1", "steady_execution_v1"})
_FINAL_STATUSES = frozenset({"completed", "failed"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "experiment_id",
        "environment_id",
        "validation_policy_id",
        "created_at_utc",
        "source_commit",
        "source_worktree_dirty",
        "configuration",
        "expected_counts",
        "files",
        "status",
    }
)
_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "run_id",
        "experiment_id",
        "case_id",
        "route_id",
        "sample_kind",
        "sample_index",
        "session_instance_id",
        "status",
        "identities",
        "measurement",
        "backend_facts",
        "numeric_facts",
        "output_sha256",
        "validation",
        "failure",
    }
)
_SESSION_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "experiment_id",
        "case_id",
        "route_id",
        "session_instance_id",
        "session_protocol_id",
        "open_s",
        "session_close_s",
        "status",
        "terminal_backend_facts",
        "release_attempted",
        "release_succeeded",
        "release_verified",
        "failure",
    }
)
_IDENTITY_FIELDS = frozenset(
    {
        "problem_id",
        "tensor_network_structure_id",
        "logical_plan_id",
        "physical_plan_id",
        "executable_id",
        "environment_id",
        "validation_policy_id",
    }
)
_MEASUREMENT_FIELDS = frozenset(field.name for field in fields(_Measurement))
_EXPECTED_COUNT_FIELDS = frozenset({"warmup", "measurement", "sessions"})
_FAILED_FIELDS = frozenset({"stage", "reason"})
_UNSUPPORTED_FIELDS = frozenset({"stage", "reason", "capability"})
_FILES = {
    "manifest": "manifest.json",
    "samples": "samples.jsonl",
    "sessions": "sessions.jsonl",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RFC3339_UTC_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z$"
)


def _is_numpy_value(value: object) -> bool:
    return type(value).__module__.split(".", 1)[0] == "numpy"


def _canonical_value(value: object) -> object:
    if _is_numpy_value(value) or isinstance(value, _Enum):
        raise TypeError(f"unsupported JSON value type: {type(value).__name__}")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if (
                _is_numpy_value(key)
                or isinstance(key, _Enum)
                or not isinstance(key, str)
            ):
                raise TypeError("JSON mapping keys must be strings")
            result[key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return compact, sorted, ASCII-only JSON for a supported value."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must be nonempty")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    _canonical_value(value)
    return value


def _exact_fields(
    record: Mapping[str, Any], expected: frozenset[str], kind: str
) -> None:
    actual = set(record)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(
        f"{kind} fields must be exact; missing={missing}, unexpected={unexpected}"
    )


def _finite_nonnegative(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{field} must be a finite non-negative number")


def _nullable_finite_nonnegative(value: object, field: str) -> None:
    if value is not None:
        _finite_nonnegative(value, field)


def _nonnegative_int(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _nullable_or_string(value: object, field: str) -> None:
    if value is not None:
        _nonempty_string(value, field)


def _sha256(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a lowercase SHA-256 hex string")
    if not _HASH_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex string")


def _uuid4(value: object, field: str) -> None:
    text = _nonempty_string(value, field)
    try:
        parsed = UUID(text)
    except ValueError:
        raise ValueError(f"{field} must be a canonical UUID4 string") from None
    if parsed.version != 4 or str(parsed) != text:
        raise ValueError(f"{field} must be a canonical UUID4 string")


def _created_at_utc(value: object) -> None:
    text = _nonempty_string(value, "created_at_utc")
    if not _RFC3339_UTC_RE.fullmatch(text):
        raise ValueError("created_at_utc must be RFC3339 UTC ending in Z")
    try:
        datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise ValueError("created_at_utc must be RFC3339 UTC ending in Z") from None


def identity_hash(domain: str, payload: object) -> str:
    """Hash a canonical payload whose explicit domain is part of the input."""

    _nonempty_string(domain, "domain")
    encoded = canonical_json({"domain": domain, "payload": payload}).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def problem_id(job: SimulationJob) -> str:
    """Return the identity of a simulation problem, excluding execution dtype."""

    if not isinstance(job, SimulationJob):
        raise TypeError("problem_id requires a SimulationJob")
    circuit = job.circuit
    payload = {
        "circuit": {
            "name": circuit.name,
            "n_qubits": circuit.n_qubits,
            "operations": [
                {
                    "gate": operation.gate,
                    "wires": list(operation.wires),
                    "params": list(operation.params),
                }
                for operation in circuit.operations
            ],
            "source": circuit.source,
        },
        "query": job.query,
        "parameters": [list(parameter) for parameter in job.parameters],
        "seed": job.seed,
    }
    return identity_hash("quantum_bench.problem_id.v1", payload)


def tensor_network_structure_id(network: TensorNetwork) -> str:
    """Return a path- and value-independent tensor-network structure identity."""

    if not isinstance(network, TensorNetwork):
        raise TypeError("tensor_network_structure_id requires a TensorNetwork")
    payload = {
        "tensors": [
            {
                "id": tensor.id,
                "labels": list(tensor.labels),
                "shape": list(tensor.shape),
                "structure": tensor.structure,
                "dtype": tensor.dtype,
                "produced_by": tensor.produced_by,
            }
            for tensor in network.tensors
        ],
        "output_labels": list(network.output_labels),
        "einsum_expression": network.einsum_expression,
    }
    return identity_hash("quantum_bench.tensor_network_structure_id.v1", payload)


def _mapping_identity(domain: str, value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{domain} requires a mapping")
    return identity_hash(domain, value)


def environment_id(value: Mapping[str, Any]) -> str:
    """Return the identity of a recorded execution environment mapping."""

    return _mapping_identity("quantum_bench.environment_id.v1", value)


def validation_policy_id(value: Mapping[str, Any]) -> str:
    """Return the identity of a validation-policy mapping."""

    return _mapping_identity("quantum_bench.validation_policy_id.v1", value)


def executable_id(value: Mapping[str, Any]) -> str:
    """Return the identity of an executable/provenance mapping."""

    return _mapping_identity("quantum_bench.executable_id.v1", value)


def new_run_id() -> str:
    """Return a freshly generated UUID4 string."""

    return str(uuid4())


def sample_id(
    run_id: str,
    case_id: str,
    route_id: str,
    sample_kind: str,
    sample_index: int,
) -> str:
    """Return the stable identity of one warmup or measurement sample."""

    for value, field in (
        (run_id, "run_id"),
        (case_id, "case_id"),
        (route_id, "route_id"),
        (sample_kind, "sample_kind"),
    ):
        _nonempty_string(value, field)
    if sample_kind not in _SAMPLE_KINDS:
        raise ValueError("sample_kind must be warmup or measurement")
    _nonnegative_int(sample_index, "sample_index")
    return identity_hash(
        _SAMPLE_SCHEMA,
        {
            "run_id": run_id,
            "case_id": case_id,
            "route_id": route_id,
            "sample_kind": sample_kind,
            "sample_index": sample_index,
        },
    )


def validate_manifest(record: Mapping[str, Any]) -> None:
    """Validate one evidence manifest without changing it."""

    record = _mapping(record, "manifest")
    _exact_fields(record, _MANIFEST_FIELDS, "manifest")
    if record["schema_version"] != _MANIFEST_SCHEMA:
        raise ValueError("manifest has an invalid schema_version")
    _uuid4(record["run_id"], "run_id")
    for field in ("experiment_id", "environment_id", "validation_policy_id"):
        _nonempty_string(record[field], field)
    _created_at_utc(record["created_at_utc"])
    if not isinstance(record["source_commit"], str):
        raise TypeError("source_commit must be a string")
    if not _COMMIT_RE.fullmatch(record["source_commit"]):
        raise ValueError("source_commit must be 40 lowercase hexadecimal characters")
    if not isinstance(record["source_worktree_dirty"], bool):
        raise TypeError("source_worktree_dirty must be a bool")
    _mapping(record["configuration"], "configuration")

    expected_counts = _mapping(record["expected_counts"], "expected_counts")
    _exact_fields(expected_counts, _EXPECTED_COUNT_FIELDS, "expected_counts")
    for field in _EXPECTED_COUNT_FIELDS:
        _nonnegative_int(expected_counts[field], f"expected_counts.{field}")

    files = _mapping(record["files"], "files")
    if dict(files) != _FILES:
        raise ValueError(
            "files must name manifest.json, samples.jsonl, and sessions.jsonl exactly"
        )
    if record["status"] not in {"running", "completed", "failed"}:
        raise ValueError("manifest status must be running, completed, or failed")


def _validate_measurement(value: object) -> None:
    measurement = _mapping(value, "measurement")
    _exact_fields(measurement, _MEASUREMENT_FIELDS, "measurement")
    scope_id = measurement["scope_id"]
    if not isinstance(scope_id, str):
        raise TypeError("measurement.scope_id must be a frozen timing scope string")
    if scope_id not in _TIMING_SCOPES:
        raise ValueError(
            "measurement.scope_id must be simulation_end_to_end_v1 or "
            "steady_execution_v1"
        )
    _finite_nonnegative(measurement["total_wall_s"], "measurement.total_wall_s")
    try:
        _Measurement(**dict(measurement))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid measurement: {error}") from error


def _validate_failure(value: object, status: str) -> None:
    failure = _mapping(value, "failure")
    expected = _UNSUPPORTED_FIELDS if status == "unsupported" else _FAILED_FIELDS
    _exact_fields(failure, expected, f"{status} failure")
    _nonempty_string(failure["stage"], "failure.stage")
    _nonempty_string(failure["reason"], "failure.reason")
    if status == "unsupported":
        _nonempty_string(failure["capability"], "failure.capability")


def validate_sample(record: Mapping[str, Any]) -> None:
    """Validate one canonical sample record and its identity binding."""

    record = _mapping(record, "sample")
    _exact_fields(record, _SAMPLE_FIELDS, "sample")
    if record["schema_version"] != _SAMPLE_SCHEMA:
        raise ValueError("sample has an invalid schema_version")
    _uuid4(record["run_id"], "run_id")
    for field in ("experiment_id", "case_id", "route_id"):
        _nonempty_string(record[field], field)
    _nonempty_string(record["sample_id"], "sample_id")
    _nonempty_string(record["sample_kind"], "sample_kind")
    if record["sample_kind"] not in _SAMPLE_KINDS:
        raise ValueError("sample_kind must be warmup or measurement")
    _nonnegative_int(record["sample_index"], "sample_index")
    _nullable_or_string(record["session_instance_id"], "session_instance_id")
    if record["status"] not in _SAMPLE_STATUSES:
        raise ValueError("sample status must be success, unsupported, or failed")

    identities = _mapping(record["identities"], "identities")
    _exact_fields(identities, _IDENTITY_FIELDS, "identities")
    for field in (
        "problem_id",
        "environment_id",
        "validation_policy_id",
    ):
        _nonempty_string(identities[field], f"identities.{field}")
    tensor_network_id = identities["tensor_network_structure_id"]
    logical_plan_id = identities["logical_plan_id"]
    if (tensor_network_id is None) != (logical_plan_id is None):
        raise ValueError(
            "tensor_network_structure_id and logical_plan_id must be null together"
        )
    _nullable_or_string(tensor_network_id, "identities.tensor_network_structure_id")
    _nullable_or_string(logical_plan_id, "identities.logical_plan_id")
    for field in ("physical_plan_id", "executable_id"):
        _nullable_or_string(identities[field], f"identities.{field}")

    expected_id = sample_id(
        record["run_id"],
        record["case_id"],
        record["route_id"],
        record["sample_kind"],
        record["sample_index"],
    )
    if record["sample_id"] != expected_id:
        raise ValueError("sample_id does not match the sample identity fields")
    _mapping(record["backend_facts"], "backend_facts")
    _mapping(record["numeric_facts"], "numeric_facts")
    if record["validation"] is not None:
        _mapping(record["validation"], "validation")

    status = record["status"]
    if status == "success":
        if record["measurement"] is None:
            raise ValueError("successful samples require measurement")
        _validate_measurement(record["measurement"])
        if record["output_sha256"] is None:
            raise ValueError("successful samples require output_sha256")
        _sha256(record["output_sha256"], "output_sha256")
        if record["failure"] is not None:
            raise ValueError("successful samples must have null failure")
        return

    if record["measurement"] is not None:
        raise ValueError("unsupported and failed samples must have null measurement")
    if record["output_sha256"] is not None:
        raise ValueError("unsupported and failed samples must have null output_sha256")
    if record["failure"] is None:
        raise ValueError("unsupported and failed samples require failure")
    _validate_failure(record["failure"], status)


def require_matching_scope(
    samples: Iterable[Mapping[str, Any]], *, expected_scope_id: str | None = None
) -> str:
    """Admit at least two successful measurement samples with one timing scope."""

    if isinstance(samples, (str, bytes, Mapping)):
        raise TypeError("samples must be an iterable of sample mappings")
    if expected_scope_id is not None:
        if not isinstance(expected_scope_id, str):
            raise TypeError("expected_scope_id must be a string or None")
        if expected_scope_id not in _TIMING_SCOPES:
            raise ValueError(
                "expected_scope_id must be simulation_end_to_end_v1 or "
                "steady_execution_v1"
            )
    try:
        iterator = iter(samples)
    except TypeError:
        raise TypeError("samples must be an iterable of sample mappings") from None

    shared_scope: str | None = None
    count = 0
    for sample in iterator:
        validate_sample(sample)
        if sample["status"] != "success":
            raise ValueError("scope admission requires successful samples")
        if sample["sample_kind"] != "measurement":
            raise ValueError("scope admission requires measurement samples")
        if sample["measurement"] is None:
            raise ValueError("scope admission requires non-null measurements")
        scope_id = sample["measurement"]["scope_id"]
        if shared_scope is None:
            shared_scope = scope_id
        elif scope_id != shared_scope:
            raise ValueError("scope admission requires all samples to share one scope")
        count += 1

    if count < 2:
        raise ValueError("scope admission requires at least two samples")
    if expected_scope_id is not None and shared_scope != expected_scope_id:
        raise ValueError("samples do not match expected_scope_id")
    assert shared_scope is not None
    return shared_scope


def validate_session(record: Mapping[str, Any]) -> None:
    """Validate one session lifecycle record."""

    record = _mapping(record, "session")
    _exact_fields(record, _SESSION_FIELDS, "session")
    if record["schema_version"] != _SESSION_SCHEMA:
        raise ValueError("session has an invalid schema_version")
    _uuid4(record["run_id"], "run_id")
    for field in (
        "experiment_id",
        "case_id",
        "route_id",
        "session_instance_id",
        "session_protocol_id",
    ):
        _nonempty_string(record[field], field)
    _nullable_finite_nonnegative(record["open_s"], "open_s")
    _nullable_finite_nonnegative(record["session_close_s"], "session_close_s")
    if record["status"] not in {"success", "failed"}:
        raise ValueError("session status must be success or failed")
    _mapping(record["terminal_backend_facts"], "terminal_backend_facts")
    for field in ("release_attempted", "release_succeeded", "release_verified"):
        if not isinstance(record[field], bool):
            raise TypeError(f"{field} must be a bool")

    if record["release_succeeded"] and not record["release_attempted"]:
        raise ValueError("release_succeeded requires release_attempted")
    if record["release_verified"] and not record["release_succeeded"]:
        raise ValueError("release_verified requires release_succeeded")

    if record["status"] == "success":
        if not all(
            record[field]
            for field in ("release_attempted", "release_succeeded", "release_verified")
        ):
            raise ValueError("successful sessions require verified release")
        if record["failure"] is not None:
            raise ValueError("successful sessions must have null failure")
        return

    if record["failure"] is None:
        raise ValueError("failed sessions require failure")
    _validate_failure(record["failure"], "failed")


def _record_rows(
    records: Iterable[Mapping[str, Any]], kind: str
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(records, (str, bytes, Mapping)):
        raise TypeError(f"{kind} must be an iterable of mappings")
    try:
        return tuple(records)
    except TypeError:
        raise TypeError(f"{kind} must be an iterable of mappings") from None


def validate_artifact_set(
    manifest: Mapping[str, Any],
    samples: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> None:
    """Validate final artifact records, counts, identities, and session links."""

    validate_manifest(manifest)
    if manifest["status"] not in _FINAL_STATUSES:
        raise ValueError("artifact manifest status must be completed or failed")

    sample_rows = _record_rows(samples, "samples")
    session_rows = _record_rows(sessions, "sessions")
    manifest_run_id = manifest["run_id"]
    manifest_experiment_id = manifest["experiment_id"]

    sessions_by_id: dict[str, Mapping[str, Any]] = {}
    for session in session_rows:
        validate_session(session)
        if session["run_id"] != manifest_run_id:
            raise ValueError("session run_id does not match manifest")
        if session["experiment_id"] != manifest_experiment_id:
            raise ValueError("session experiment_id does not match manifest")
        session_instance_id = session["session_instance_id"]
        if session_instance_id in sessions_by_id:
            raise ValueError(f"duplicate session_instance_id: {session_instance_id}")
        sessions_by_id[session_instance_id] = session

    sample_ids: set[str] = set()
    successful_scopes: dict[tuple[str, str], set[str]] = {}
    actual_counts = {"warmup": 0, "measurement": 0, "sessions": len(session_rows)}
    for sample in sample_rows:
        validate_sample(sample)
        if sample["run_id"] != manifest_run_id:
            raise ValueError("sample run_id does not match manifest")
        if sample["experiment_id"] != manifest_experiment_id:
            raise ValueError("sample experiment_id does not match manifest")
        identities = sample["identities"]
        if identities["environment_id"] != manifest["environment_id"]:
            raise ValueError("sample environment_id does not match manifest")
        if identities["validation_policy_id"] != manifest["validation_policy_id"]:
            raise ValueError("sample validation_policy_id does not match manifest")

        current_sample_id = sample["sample_id"]
        if current_sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {current_sample_id}")
        sample_ids.add(current_sample_id)
        actual_counts[sample["sample_kind"]] += 1
        if sample["status"] == "success":
            measurement = sample["measurement"]
            assert measurement is not None
            key = (sample["case_id"], sample["route_id"])
            successful_scopes.setdefault(key, set()).add(measurement["scope_id"])

        session_instance_id = sample["session_instance_id"]
        if session_instance_id is not None:
            session = sessions_by_id.get(session_instance_id)
            if session is None:
                raise ValueError("sample session_instance_id has no matching session")
            for field in ("run_id", "experiment_id", "case_id", "route_id"):
                if sample[field] != session[field]:
                    raise ValueError(f"sample and linked session {field} do not match")

    expected_counts = manifest["expected_counts"]
    if manifest["status"] == "completed":
        if actual_counts != dict(expected_counts):
            raise ValueError(
                f"completed artifact counts do not match expected_counts: {actual_counts}"
            )
        if any(sample["status"] != "success" for sample in sample_rows):
            raise ValueError("completed artifacts require every sample to succeed")
        if any(
            session["status"] != "success" or not session["release_verified"]
            for session in session_rows
        ):
            raise ValueError(
                "completed artifacts require successful sessions with verified release"
            )
        if any(len(scopes) != 1 for scopes in successful_scopes.values()):
            raise ValueError(
                "completed artifacts require one timing scope per case and route"
            )
        return

    for field, actual in actual_counts.items():
        if actual > expected_counts[field]:
            raise ValueError(f"failed artifact count exceeds expected_counts.{field}")


def _record_json(
    record: Mapping[str, Any], validator: Callable[[Mapping[str, Any]], None]
) -> str:
    validator(record)
    return canonical_json(record)


def write_manifest(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Atomically write an initial running manifest followed by one newline."""

    validate_manifest(record)
    if record["status"] != "running":
        raise ValueError("public manifest writes require status running")
    _write_manifest(path, record)


def _write_manifest(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Atomically write a validated manifest, including a finalized manifest."""

    _atomic_write_bytes(
        Path(path), (_record_json(record, validate_manifest) + "\n").encode("utf-8")
    )


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _append_record(
    path: str | os.PathLike[str],
    record: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], None],
) -> None:
    payload = (_record_json(record, validator) + "\n").encode("utf-8")
    target = Path(path)
    _require_open_artifact(target)
    existing = target.read_bytes() if target.exists() else b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError("existing JSONL file must be empty or newline-terminated")
    _atomic_write_bytes(target, existing + payload)


def _require_open_artifact(target: Path) -> None:
    manifest_path = target.parent / _FILES["manifest"]
    if not manifest_path.exists():
        return
    manifest = _load_json_mapping(manifest_path, "manifest")
    validate_manifest(manifest)
    if manifest["status"] != "running":
        raise ValueError("evidence append requires a running manifest")


def append_sample(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one validated sample as a canonical JSONL record."""

    _append_record(path, record, validate_sample)


def append_session(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one validated session as a canonical JSONL record."""

    _append_record(path, record, validate_session)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json(text: str, field: str) -> object:
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{field} contains invalid JSON") from error


def _load_json_mapping(path: Path, field: str) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{field} must not be blank")
    return _mapping(_strict_json(text, field), field)


def _load_json_lines(path: Path, field: str) -> tuple[Mapping[str, Any], ...]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return ()
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError(f"{field} must not contain blank lines")
    return tuple(
        _mapping(_strict_json(line, f"{field} line {index}"), f"{field} line {index}")
        for index, line in enumerate(lines, start=1)
    )


def finalize_artifacts(directory: str | os.PathLike[str], *, status: str) -> None:
    """Validate an artifact directory and atomically finalize its manifest."""

    if not isinstance(status, str):
        raise TypeError("status must be a string")
    if status not in _FINAL_STATUSES:
        raise ValueError("status must be completed or failed")

    root = Path(directory)
    manifest_path = root / _FILES["manifest"]
    manifest = _load_json_mapping(manifest_path, "manifest")
    validate_manifest(manifest)
    if manifest["status"] != "running":
        raise ValueError("artifact finalization requires a running manifest")
    samples = _load_json_lines(root / _FILES["samples"], "samples")
    sessions = _load_json_lines(root / _FILES["sessions"], "sessions")

    final_manifest = dict(manifest)
    final_manifest["status"] = status
    validate_artifact_set(final_manifest, samples, sessions)
    _write_manifest(manifest_path, final_manifest)


__all__ = [
    "append_sample",
    "append_session",
    "canonical_json",
    "finalize_artifacts",
    "identity_hash",
    "new_run_id",
    "require_matching_scope",
    "sample_id",
    "validate_artifact_set",
    "validate_manifest",
    "validate_sample",
    "validate_session",
    "write_manifest",
]
