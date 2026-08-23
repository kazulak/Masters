"""Experiment-owned repetition and session lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from quantum_bench.evidence import (
    append_sample,
    append_session,
    canonical_json,
    sample_id,
    validate_sample,
    validate_session,
)
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    JsonValue,
    Measurement,
    UnsupportedExecution,
)


_IDENTITY_FIELDS = {
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "physical_plan_id",
    "executable_id",
    "environment_id",
    "validation_policy_id",
}
_REQUIRED_IDENTITY_FIELDS = {
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "environment_id",
    "validation_policy_id",
}
_MEASUREMENT_FIELDS = (
    "scope_id",
    "total_wall_s",
    "lowering_s",
    "planning_s",
    "slicing_s",
    "mapping_s",
    "session_open_s",
    "encode_s",
    "preparation_s",
    "h2d_s",
    "kernel_s",
    "host_reduce_s",
    "d2h_s",
    "decode_s",
    "rank_work_s",
    "h2d_bytes",
    "d2h_bytes",
    "energy_j",
)
_TIMING_SCOPES = frozenset({"simulation_end_to_end_v1", "steady_execution_v1"})


def run_direct_samples(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    identities: Mapping[str, JsonValue],
    warmups: int,
    repetitions: int,
    run_once: Callable[[], ExecutionSample],
    samples_path: str | os.PathLike[str],
) -> tuple[Mapping[str, JsonValue], ...]:
    """Run and append all warmup and measurement samples for a direct route."""

    normalized_identities = _validate_arguments(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id=case_id,
        route_id=route_id,
        identities=identities,
        warmups=warmups,
        repetitions=repetitions,
        samples_path=samples_path,
        run_once=run_once,
    )
    _reject_planned_sample_id_collisions(
        samples_path,
        _planned_sample_ids(run_id, case_id, route_id, warmups, repetitions),
    )

    rows: list[Mapping[str, JsonValue]] = []
    for sample_kind, count in (("warmup", warmups), ("measurement", repetitions)):
        for sample_index in range(count):
            row = _run_sample(
                run_id=run_id,
                experiment_id=experiment_id,
                case_id=case_id,
                route_id=route_id,
                identities=normalized_identities,
                sample_kind=sample_kind,
                sample_index=sample_index,
                session_instance_id=None,
                invoke=lambda: run_once(),
            )
            append_sample(samples_path, row)
            rows.append(row)
    return tuple(rows)


def run_session_samples(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    identities: Mapping[str, JsonValue],
    warmups: int,
    repetitions: int,
    session_protocol_id: str,
    open_session: Callable[[], Any],
    inputs: Mapping[str, Any],
    samples_path: str | os.PathLike[str],
    sessions_path: str | os.PathLike[str],
) -> tuple[tuple[Mapping[str, JsonValue], ...], Mapping[str, JsonValue]]:
    """Run samples on one persistent session and append its lifecycle record."""

    normalized_identities = _validate_arguments(
        run_id=run_id,
        experiment_id=experiment_id,
        case_id=case_id,
        route_id=route_id,
        identities=identities,
        warmups=warmups,
        repetitions=repetitions,
        samples_path=samples_path,
        sessions_path=sessions_path,
        session_protocol_id=session_protocol_id,
        open_session=open_session,
        inputs=inputs,
    )
    _reject_planned_sample_id_collisions(
        samples_path,
        _planned_sample_ids(run_id, case_id, route_id, warmups, repetitions),
    )

    session_instance_id = str(uuid4())
    if session_instance_id in _existing_session_ids(sessions_path):
        raise ValueError(
            "generated session_instance_id already exists in sessions_path"
        )
    open_started = time.perf_counter()
    try:
        session = open_session()
    except UnsupportedExecution as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts={},
            failure={"stage": exc.stage, "reason": exc.reason},
        )
        append_session(sessions_path, session_row)
        return (), session_row
    except ExecutionFailed as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts=exc.backend_facts,
            failure={"stage": exc.stage, "reason": exc.reason},
        )
        append_session(sessions_path, session_row)
        return (), session_row
    except Exception as exc:
        open_s = time.perf_counter() - open_started
        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=None,
            terminal_backend_facts={},
            failure={"stage": "session_open", "reason": _unexpected_reason(exc)},
        )
        append_session(sessions_path, session_row)
        return (), session_row

    open_s = time.perf_counter() - open_started
    rows: list[Mapping[str, JsonValue]] = []
    sample_failure: Mapping[str, JsonValue] | None = None
    interface_failure: Mapping[str, JsonValue] | None = None
    try:
        run_method = getattr(session, "run_once", None)
    except Exception as exc:
        run_method = None
        interface_failure = {
            "stage": "session_open",
            "reason": _unexpected_reason(exc),
        }
    try:
        close_method = getattr(session, "close", None)
    except Exception as exc:
        close_method = None
        interface_failure = {
            "stage": "session_open",
            "reason": _unexpected_reason(exc),
        }
    if interface_failure is None and not callable(run_method):
        interface_failure = {
            "stage": "session_open",
            "reason": "opened session must expose callable run_once",
        }
    if interface_failure is None and not callable(close_method):
        interface_failure = {
            "stage": "session_open",
            "reason": "opened session must expose callable close",
        }

    try:
        if interface_failure is None:
            for sample_kind, count in (
                ("warmup", warmups),
                ("measurement", repetitions),
            ):
                for sample_index in range(count):
                    row = _run_sample(
                        run_id=run_id,
                        experiment_id=experiment_id,
                        case_id=case_id,
                        route_id=route_id,
                        identities=normalized_identities,
                        sample_kind=sample_kind,
                        sample_index=sample_index,
                        session_instance_id=session_instance_id,
                        invoke=lambda: run_method(inputs),
                        persistent_session=True,
                    )
                    append_sample(samples_path, row)
                    rows.append(row)
                    if row["status"] != "success":
                        failure = row["failure"]
                        if not isinstance(failure, Mapping):  # pragma: no cover
                            raise TypeError("failed sample has no failure mapping")
                        sample_failure = {
                            "stage": failure["stage"],
                            "reason": failure["reason"],
                        }
                        break
                if sample_failure is not None:
                    break
    finally:
        close_failure: Mapping[str, JsonValue] | None = None
        terminal_backend_facts: Mapping[str, JsonValue] = {}
        session_close_s: float | None = None
        if callable(close_method):
            close_started = time.perf_counter()
            try:
                terminal_backend_facts = _plain_json(close_method())
                if not isinstance(terminal_backend_facts, Mapping):
                    raise TypeError("session close must return a mapping")
            except ExecutionFailed as exc:
                terminal_backend_facts = _plain_json(exc.backend_facts)
                close_failure = {"stage": exc.stage, "reason": exc.reason}
            except Exception as exc:
                terminal_backend_facts = {}
                close_failure = {
                    "stage": "session_close",
                    "reason": _unexpected_reason(exc),
                }
            session_close_s = time.perf_counter() - close_started
        else:
            close_failure = {
                "stage": "session_close",
                "reason": "opened session must expose callable close",
            }

        (
            release_attempted,
            release_succeeded,
            release_verified,
            release_inconsistent,
        ) = _release_facts(terminal_backend_facts)

        if close_failure is not None:
            failure = close_failure
        elif release_inconsistent:
            failure = {
                "stage": "session_close",
                "reason": "hardware release facts are inconsistent",
            }
        elif interface_failure is not None:
            failure = interface_failure
        elif sample_failure is not None:
            failure = sample_failure
        elif not (release_attempted and release_succeeded and release_verified):
            failure = {
                "stage": "session_close",
                "reason": "hardware release was not fully verified",
            }
        else:
            failure = None

        session_row = _session_row(
            run_id=run_id,
            experiment_id=experiment_id,
            case_id=case_id,
            route_id=route_id,
            session_instance_id=session_instance_id,
            session_protocol_id=session_protocol_id,
            open_s=open_s,
            session_close_s=session_close_s,
            terminal_backend_facts=terminal_backend_facts,
            release_attempted=release_attempted,
            release_succeeded=release_succeeded,
            release_verified=release_verified,
            failure=failure,
        )
        append_session(sessions_path, session_row)

    return tuple(rows), session_row


def _validate_arguments(**values: Any) -> dict[str, JsonValue]:
    _canonical_uuid4(values["run_id"], "run_id")
    for field in ("experiment_id", "case_id", "route_id"):
        _nonempty_string(values[field], field)
    if "session_protocol_id" in values:
        _nonempty_string(values["session_protocol_id"], "session_protocol_id")
    for field in ("warmups", "repetitions"):
        value = values[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    for field in ("run_once", "open_session"):
        if field in values and not callable(values[field]):
            raise TypeError(f"{field} must be callable")
    if "inputs" in values and not isinstance(values["inputs"], Mapping):
        raise TypeError("inputs must be a mapping")
    for field in ("samples_path", "sessions_path"):
        if field in values and not isinstance(values[field], (str, os.PathLike)):
            raise TypeError(f"{field} must be path-like")

    identities = values["identities"]
    if not isinstance(identities, Mapping):
        raise TypeError("identities must be a mapping")
    if set(identities) != _IDENTITY_FIELDS:
        raise ValueError("identities must match the evidence identity schema exactly")
    normalized = _plain_json(identities)
    if not isinstance(normalized, dict):  # pragma: no cover - guarded above
        raise TypeError("identities must be a mapping")
    for field in _REQUIRED_IDENTITY_FIELDS:
        _nonempty_string(normalized[field], f"identities.{field}")
    for field in ("physical_plan_id", "executable_id"):
        if normalized[field] is not None:
            _nonempty_string(normalized[field], f"identities.{field}")
    return normalized


def _run_sample(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    identities: Mapping[str, JsonValue],
    sample_kind: str,
    sample_index: int,
    session_instance_id: str | None,
    invoke: Callable[[], Any],
    persistent_session: bool = False,
) -> Mapping[str, JsonValue]:
    base: dict[str, JsonValue] = {
        "schema_version": "evidence_sample_v1",
        "sample_id": sample_id(run_id, case_id, route_id, sample_kind, sample_index),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "route_id": route_id,
        "sample_kind": sample_kind,
        "sample_index": sample_index,
        "session_instance_id": session_instance_id,
        "identities": identities,
        "validation": None,
    }
    try:
        sample = invoke()
        if not isinstance(sample, ExecutionSample):
            raise TypeError("run_once must return ExecutionSample")
        if sample.measurement.scope_id not in _TIMING_SCOPES:
            return {
                **base,
                "status": "failed",
                "measurement": None,
                "backend_facts": {},
                "numeric_facts": {},
                "output_sha256": None,
                "failure": {
                    "stage": "timing_contract",
                    "reason": "samples require a frozen timing scope",
                },
            }
        if persistent_session and not _has_steady_session_timing(sample.measurement):
            return {
                **base,
                "status": "failed",
                "measurement": None,
                "backend_facts": {},
                "numeric_facts": {},
                "output_sha256": None,
                "failure": {
                    "stage": "timing_contract",
                    "reason": "persistent session samples require steady_execution_v1 timing",
                },
            }
        return {
            **base,
            "status": "success",
            "measurement": _measurement_mapping(sample.measurement),
            "backend_facts": _plain_json(sample.backend_facts),
            "numeric_facts": _plain_json(sample.numeric_facts),
            "output_sha256": _output_hash(sample.output),
            "failure": None,
        }
    except UnsupportedExecution as exc:
        return {
            **base,
            "status": "unsupported",
            "measurement": None,
            "backend_facts": {},
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {
                "stage": exc.stage,
                "reason": exc.reason,
                "capability": exc.capability,
            },
        }
    except ExecutionFailed as exc:
        return {
            **base,
            "status": "failed",
            "measurement": None,
            "backend_facts": _plain_json(exc.backend_facts),
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {"stage": exc.stage, "reason": exc.reason},
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "measurement": None,
            "backend_facts": {},
            "numeric_facts": {},
            "output_sha256": None,
            "failure": {"stage": "execution", "reason": _unexpected_reason(exc)},
        }


def _measurement_mapping(measurement: Measurement) -> Mapping[str, JsonValue]:
    return {field: getattr(measurement, field) for field in _MEASUREMENT_FIELDS}


def _has_steady_session_timing(measurement: Measurement) -> bool:
    return measurement.scope_id == "steady_execution_v1" and all(
        getattr(measurement, field) is None
        for field in (
            "lowering_s",
            "planning_s",
            "slicing_s",
            "mapping_s",
            "session_open_s",
        )
    )


def _planned_sample_ids(
    run_id: str,
    case_id: str,
    route_id: str,
    warmups: int,
    repetitions: int,
) -> frozenset[str]:
    return frozenset(
        sample_id(run_id, case_id, route_id, sample_kind, sample_index)
        for sample_kind, count in (("warmup", warmups), ("measurement", repetitions))
        for sample_index in range(count)
    )


def _reject_planned_sample_id_collisions(
    samples_path: str | os.PathLike[str], planned_ids: frozenset[str]
) -> None:
    if not planned_ids:
        return
    collisions = _existing_sample_ids(samples_path).intersection(planned_ids)
    if collisions:
        raise ValueError("planned sample IDs already exist in samples_path")


def _existing_sample_ids(path: str | os.PathLike[str]) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    text = target.read_text(encoding="utf-8")
    if not text:
        return set()
    if not text.endswith("\n"):
        raise ValueError("existing samples JSONL must be newline-terminated")

    sample_ids: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError("existing samples JSONL must not contain blank lines")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"existing samples JSONL line {index} is invalid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError(f"existing samples JSONL line {index} must be a mapping")
        validate_sample(record)
        current_id = record["sample_id"]
        if current_id in sample_ids:
            raise ValueError("existing samples JSONL contains duplicate sample_id")
        sample_ids.add(current_id)
    return sample_ids


def _existing_session_ids(path: str | os.PathLike[str]) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    text = target.read_text(encoding="utf-8")
    if not text:
        return set()
    if not text.endswith("\n"):
        raise ValueError("existing sessions JSONL must be newline-terminated")

    session_ids: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError("existing sessions JSONL must not contain blank lines")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"existing sessions JSONL line {index} is invalid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise TypeError(f"existing sessions JSONL line {index} must be a mapping")
        validate_session(record)
        current_id = record["session_instance_id"]
        if current_id in session_ids:
            raise ValueError(
                "existing sessions JSONL contains duplicate session_instance_id"
            )
        session_ids.add(current_id)
    return session_ids


def _session_row(
    *,
    run_id: str,
    experiment_id: str,
    case_id: str,
    route_id: str,
    session_instance_id: str,
    session_protocol_id: str,
    open_s: float | None,
    session_close_s: float | None,
    terminal_backend_facts: Mapping[str, JsonValue],
    release_attempted: bool = False,
    release_succeeded: bool = False,
    release_verified: bool = False,
    failure: Mapping[str, JsonValue] | None = None,
) -> Mapping[str, JsonValue]:
    return {
        "schema_version": "evidence_session_v1",
        "run_id": run_id,
        "experiment_id": experiment_id,
        "case_id": case_id,
        "route_id": route_id,
        "session_instance_id": session_instance_id,
        "session_protocol_id": session_protocol_id,
        "open_s": open_s,
        "session_close_s": session_close_s,
        "status": "success" if failure is None else "failed",
        "terminal_backend_facts": _plain_json(terminal_backend_facts),
        "release_attempted": release_attempted,
        "release_succeeded": release_succeeded,
        "release_verified": release_verified,
        "failure": failure,
    }


def _plain_json(value: object) -> Any:
    return json.loads(canonical_json(value))


def _output_hash(output: np.ndarray) -> str:
    dtype = output.dtype
    if dtype.fields is not None or dtype.kind not in {"b", "i", "u", "f", "c"}:
        raise TypeError(
            "output dtype must be a scalar bool, integer, float, or complex"
        )
    if dtype.kind in {"f", "c"} and not np.isfinite(output).all():
        raise ValueError("output values must be finite")
    array = np.asarray(output, dtype=dtype.newbyteorder("<"), order="C")
    digest = hashlib.sha256()
    digest.update(
        canonical_json(
            {
                "domain": "quantum_bench.output_sha256",
                "version": 1,
                "dtype": array.dtype.str,
                "shape": array.shape,
            }
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _release_facts(
    terminal_backend_facts: Mapping[str, JsonValue],
) -> tuple[bool, bool, bool, bool]:
    attempted_raw = terminal_backend_facts.get("hardware_release_attempted") is True
    succeeded_raw = terminal_backend_facts.get("hardware_release_succeeded") is True
    verified_raw = terminal_backend_facts.get("hardware_release_verified") is True
    inconsistent = (succeeded_raw and not attempted_raw) or (
        verified_raw and not succeeded_raw
    )
    release_attempted = attempted_raw
    release_succeeded = attempted_raw and succeeded_raw
    release_verified = release_succeeded and verified_raw
    return release_attempted, release_succeeded, release_verified, inconsistent


def _nonempty_string(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must be nonempty")


def _canonical_uuid4(value: object, field: str) -> None:
    _nonempty_string(value, field)
    try:
        parsed = UUID(value)
    except ValueError:
        raise ValueError(f"{field} must be a canonical UUID4 string") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID4 string")


def _unexpected_reason(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = ["run_direct_samples", "run_session_samples"]
