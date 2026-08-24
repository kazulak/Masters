from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import quantum_bench.experiment as experiment
from quantum_bench.evidence import sample_id
from quantum_bench.experiment import run_direct_samples, run_session_samples
from quantum_bench.results import (
    ExecutionFailed,
    ExecutionSample,
    Measurement,
    UnsupportedExecution,
)


IDENTITIES = {
    "problem_id": "1" * 64,
    "tensor_network_structure_id": "2" * 64,
    "logical_plan_id": "3" * 64,
    "physical_plan_id": None,
    "executable_id": None,
    "environment_id": "4" * 64,
    "validation_policy_id": "5" * 64,
}
RUN_ID = "12345678-1234-4234-8234-1234567890ab"
EXPERIMENT_ID = "6" * 64


def _sample(
    *, output: np.ndarray | None = None, measurement: Measurement | None = None
) -> ExecutionSample:
    return ExecutionSample(
        output=np.array([1, 2] if output is None else output),
        measurement=measurement
        or Measurement(scope_id="steady_execution_v1", total_wall_s=1.0),
        backend_facts={"backend": "test", "nested": (1, True)},
        numeric_facts={"value": 1},
    )


def _validation(
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
        "scientific_validation_passed": policy_passed
        and (not full_precision_applicable or full_precision_passed is True),
        "max_abs_error": 0.0,
        "relative_l2_error": 0.0,
    }


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _hash(array: np.ndarray) -> str:
    contiguous = np.asarray(array, dtype=array.dtype.newbyteorder("<"), order="C")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "domain": "quantum_bench.output_sha256",
                "version": 1,
                "dtype": contiguous.dtype.str,
                "shape": contiguous.shape,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def test_direct_order_continuation_and_measurement_nulls(tmp_path: Path) -> None:
    calls = 0

    def run_once() -> ExecutionSample:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ExecutionFailed("kernel", "no result", {"rank": 1})
        return _sample()

    path = tmp_path / "samples.jsonl"
    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=2,
        run_once=run_once,
        samples_path=path,
    )

    assert [(row["sample_kind"], row["sample_index"]) for row in rows] == [
        ("warmup", 0),
        ("measurement", 0),
        ("measurement", 1),
    ]
    assert rows[1]["status"] == "failed"
    assert rows[2]["status"] == "success"
    assert set(rows[0]["measurement"]) == {
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
    }
    assert all(
        value is None
        for key, value in rows[0]["measurement"].items()
        if key != "scope_id" and key != "total_wall_s"
    )


def test_validation_runs_for_warmups_and_measurements_after_execution(
    tmp_path: Path,
) -> None:
    seen: list[ExecutionSample] = []

    def validate(sample: ExecutionSample) -> dict[str, object]:
        seen.append(sample)
        return _validation()

    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=1,
        run_once=_sample,
        samples_path=tmp_path / "samples.jsonl",
        validate=validate,
    )

    assert len(seen) == 2
    assert [row["status"] for row in rows] == ["success", "success"]
    assert all(row["validation"] == _validation() for row in rows)


def test_failed_validation_keeps_facts_but_not_measurement_or_output(
    tmp_path: Path,
) -> None:
    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=_sample,
        samples_path=tmp_path / "samples.jsonl",
        validate=lambda sample: _validation(policy_passed=False),
    )

    row = rows[0]
    assert row["status"] == "failed"
    assert row["measurement"] is None
    assert row["output_sha256"] is None
    assert row["backend_facts"] == {"backend": "test", "nested": [1, True]}
    assert row["numeric_facts"] == {"value": 1}
    assert row["validation"] == _validation(policy_passed=False)
    assert row["failure"] == {
        "stage": "validation",
        "reason": "scientific validation failed",
    }


def test_validator_exception_is_bounded_and_has_no_traceback(tmp_path: Path) -> None:
    def validate(sample: ExecutionSample) -> dict[str, object]:
        raise RuntimeError("bad\n" + ("x" * 1000))

    row = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=_sample,
        samples_path=tmp_path / "samples.jsonl",
        validate=validate,
    )[0]

    assert row["status"] == "failed"
    assert row["validation"] is None
    assert row["failure"]["stage"] == "validation"
    assert "traceback" not in row["failure"]["reason"].lower()
    assert len(row["failure"]["reason"]) <= 256


def test_session_validation_uses_the_same_hook_for_each_attempt(
    tmp_path: Path,
) -> None:
    seen: list[ExecutionSample] = []

    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    def validate(sample: ExecutionSample) -> dict[str, object]:
        seen.append(sample)
        return _validation()

    rows, _ = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=Session,
        inputs={},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
        validate=validate,
    )

    assert len(seen) == 2
    assert [row["status"] for row in rows] == ["success", "success"]


def test_direct_invalid_timing_scope_is_recorded_as_failure(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=lambda: _sample(
            measurement=Measurement(scope_id="invalid_scope", total_wall_s=1.0)
        ),
        samples_path=path,
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["failure"] == {
        "stage": "timing_contract",
        "reason": "samples require a frozen timing scope",
    }
    assert rows[0]["measurement"] is rows[0]["output_sha256"] is None
    assert _read(path) == [rows[0]]


def test_output_hash_binds_dtype_and_shape(tmp_path: Path) -> None:
    outputs = [
        np.array([1, 2], dtype=np.int8),
        np.array([[1, 2]], dtype=np.int8),
        np.array([1, 2], dtype=np.int16),
    ]
    rows = []
    for index, output in enumerate(outputs):
        rows.extend(
            run_direct_samples(
                run_id=f"12345678-1234-4234-8234-1234567890a{index}",
                experiment_id=EXPERIMENT_ID,
                case_id="case",
                route_id="route",
                identities=IDENTITIES,
                warmups=0,
                repetitions=1,
                run_once=lambda output=output: _sample(output=output),
                samples_path=tmp_path / f"samples-{index}.jsonl",
            )
        )
    assert [row["output_sha256"] for row in rows] == [
        _hash(output) for output in outputs
    ]
    assert len({row["output_sha256"] for row in rows}) == 3


def test_output_hash_normalizes_byte_order_and_rejects_invalid_outputs(
    tmp_path: Path,
) -> None:
    native = np.array([1.5, 2.5], dtype=np.float32)
    big_endian = native.astype(">f4")
    native_row = run_direct_samples(
        run_id="42345678-1234-4234-8234-1234567890ab",
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=lambda: _sample(output=native),
        samples_path=tmp_path / "native.jsonl",
    )[0]
    big_endian_row = run_direct_samples(
        run_id="52345678-1234-4234-8234-1234567890ab",
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=lambda: _sample(output=big_endian),
        samples_path=tmp_path / "big-endian.jsonl",
    )[0]
    assert native_row["output_sha256"] == big_endian_row["output_sha256"]

    for index, output in enumerate(
        (
            np.array([object()], dtype=object),
            np.array(["text"], dtype="U4"),
            np.array([(1,)], dtype=[("value", "i4")]),
            np.array([np.nan], dtype=np.float64),
        )
    ):
        row = run_direct_samples(
            run_id=f"62345678-1234-4234-8234-1234567890a{index}",
            experiment_id=EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=IDENTITIES,
            warmups=0,
            repetitions=1,
            run_once=lambda output=output: _sample(output=output),
            samples_path=tmp_path / f"invalid-output-{index}.jsonl",
        )[0]
        assert row["status"] == "failed"
        assert row["failure"]["stage"] == "execution"


def test_session_is_persistent_and_close_is_outside_measurements(
    tmp_path: Path,
) -> None:
    class Session:
        def __init__(self) -> None:
            self.runs = 0
            self.closes = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            assert inputs == {"input": "value"}
            self.runs += 1
            return _sample()

        def close(self) -> dict[str, object]:
            self.closes += 1
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    session = Session()
    rows, session_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: session,
        inputs={"input": "value"},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
    )
    assert len(rows) == 2
    assert session.runs == 2
    assert session.closes == 1
    assert session_row["status"] == "success"
    assert rows[0]["measurement"]["session_open_s"] is None
    assert session_row["open_s"] is not None
    assert session_row["session_close_s"] is not None


def test_session_stops_after_failure_and_open_failure_is_recorded(
    tmp_path: Path,
) -> None:
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
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=2,
        repetitions=2,
        session_protocol_id="protocol",
        open_session=lambda: Session(),
        inputs={},
        samples_path=samples_path,
        sessions_path=sessions_path,
    )
    assert len(rows) == 1
    assert calls == 1
    assert session_row["status"] == "failed"
    assert session_row["failure"] == {
        "stage": "preflight",
        "reason": "unsupported route",
    }

    _, open_row = run_session_samples(
        run_id="22345678-1234-4234-8234-1234567890ab",
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
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
    assert len(_read(samples_path)) == 1


def test_invalid_uuid_rejects_before_direct_or_session_side_effects(
    tmp_path: Path,
) -> None:
    direct_calls = 0
    open_calls = 0
    samples_path = tmp_path / "samples.jsonl"
    sessions_path = tmp_path / "sessions.jsonl"

    def run_once() -> ExecutionSample:
        nonlocal direct_calls
        direct_calls += 1
        return _sample()

    def open_session() -> object:
        nonlocal open_calls
        open_calls += 1
        return object()

    with pytest.raises(ValueError, match="canonical UUID4"):
        run_direct_samples(
            run_id="not-a-uuid",
            experiment_id=EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=IDENTITIES,
            warmups=0,
            repetitions=1,
            run_once=run_once,
            samples_path=samples_path,
        )
    with pytest.raises(ValueError, match="canonical UUID4"):
        run_session_samples(
            run_id="not-a-uuid",
            experiment_id=EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=IDENTITIES,
            warmups=0,
            repetitions=1,
            session_protocol_id="protocol",
            open_session=open_session,
            inputs={},
            samples_path=samples_path,
            sessions_path=sessions_path,
        )
    assert direct_calls == open_calls == 0
    assert not samples_path.exists()
    assert not sessions_path.exists()


def test_invalid_opened_session_is_closed_once_and_recorded(tmp_path: Path) -> None:
    class InvalidSession:
        closes = 0

        def close(self) -> dict[str, object]:
            self.closes += 1
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    session = InvalidSession()
    rows, session_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: session,
        inputs={},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
    )
    assert rows == ()
    assert session.closes == 1
    assert session_row["status"] == "failed"
    assert session_row["failure"]["stage"] == "session_open"


def test_session_timing_contract_failure_stops_later_attempts(tmp_path: Path) -> None:
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            self.calls += 1
            return _sample(
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

    session = Session()
    rows, session_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=2,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: session,
        inputs={},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
    )
    assert session.calls == len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["failure"]["stage"] == "timing_contract"
    assert rows[0]["measurement"] is rows[0]["output_sha256"] is None
    assert session_row["failure"] == {
        "stage": "timing_contract",
        "reason": "persistent session samples require steady_execution_v1 timing",
    }


@pytest.mark.parametrize(
    ("outcome", "failure"),
    [
        (
            ExecutionFailed("kernel", "failed", {"rank": 1}),
            {"stage": "kernel", "reason": "failed"},
        ),
        (RuntimeError("boom"), {"stage": "execution", "reason": "RuntimeError: boom"}),
    ],
)
def test_session_execution_failures_stop_and_propagate(
    tmp_path: Path, outcome: Exception, failure: dict[str, str]
) -> None:
    class Session:
        def __init__(self) -> None:
            self.calls = 0

        def run_once(self, inputs: object) -> ExecutionSample:
            self.calls += 1
            raise outcome

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    session = Session()
    rows, session_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=1,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: session,
        inputs={},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=tmp_path / "sessions.jsonl",
    )
    assert session.calls == len(rows) == 1
    assert rows[0]["failure"] == failure
    assert session_row["status"] == "failed"
    assert session_row["failure"] == failure


def test_argument_errors_do_not_write(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    with pytest.raises((TypeError, ValueError)):
        run_direct_samples(
            run_id="",
            experiment_id=EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities=IDENTITIES,
            warmups=0,
            repetitions=1,
            run_once=lambda: _sample(),
            samples_path=path,
        )
    with pytest.raises((TypeError, ValueError)):
        run_direct_samples(
            run_id=RUN_ID,
            experiment_id=EXPERIMENT_ID,
            case_id="case",
            route_id="route",
            identities={**IDENTITIES, "extra": "bad"},
            warmups=False,
            repetitions=1,
            run_once=lambda: _sample(),
            samples_path=path,
        )
    assert not path.exists()


def test_sample_ids_use_canonical_identity_fields(tmp_path: Path) -> None:
    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        run_once=lambda: _sample(),
        samples_path=tmp_path / "samples.jsonl",
    )
    assert rows[0]["sample_id"] == sample_id(RUN_ID, "case", "route", "measurement", 0)


def test_direct_sample_id_collision_rejects_before_execution(tmp_path: Path) -> None:
    calls = 0
    path = tmp_path / "samples.jsonl"

    def run_once() -> ExecutionSample:
        nonlocal calls
        calls += 1
        return _sample()

    kwargs = {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "case_id": "case",
        "route_id": "route",
        "identities": IDENTITIES,
        "warmups": 0,
        "repetitions": 1,
        "run_once": run_once,
        "samples_path": path,
    }
    run_direct_samples(**kwargs)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="planned sample IDs"):
        run_direct_samples(**kwargs)

    assert calls == 1
    assert path.read_bytes() == original
    run_direct_samples(**{**kwargs, "case_id": "other-case"})
    assert calls == 2
    assert len(_read(path)) == 2


def test_session_sample_id_collision_rejects_before_opening(tmp_path: Path) -> None:
    opens = 0
    samples_path = tmp_path / "samples.jsonl"
    sessions_path = tmp_path / "sessions.jsonl"

    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    def open_session() -> Session:
        nonlocal opens
        opens += 1
        return Session()

    kwargs = {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "case_id": "case",
        "route_id": "route",
        "identities": IDENTITIES,
        "warmups": 0,
        "repetitions": 1,
        "session_protocol_id": "protocol",
        "open_session": open_session,
        "inputs": {},
        "samples_path": samples_path,
        "sessions_path": sessions_path,
    }
    run_session_samples(**kwargs)
    original_samples = samples_path.read_bytes()
    original_sessions = sessions_path.read_bytes()

    with pytest.raises(ValueError, match="planned sample IDs"):
        run_session_samples(**kwargs)

    assert opens == 1
    assert samples_path.read_bytes() == original_samples
    assert sessions_path.read_bytes() == original_sessions
    run_session_samples(**{**kwargs, "route_id": "other-route"})
    assert opens == 2
    assert len(_read(samples_path)) == len(_read(sessions_path)) == 2


def test_session_instance_collision_rejects_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opens = 0
    samples_path = tmp_path / "samples.jsonl"
    sessions_path = tmp_path / "sessions.jsonl"

    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    def open_session() -> Session:
        nonlocal opens
        opens += 1
        return Session()

    kwargs = {
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "case_id": "case",
        "route_id": "route",
        "identities": IDENTITIES,
        "warmups": 0,
        "repetitions": 1,
        "session_protocol_id": "protocol",
        "open_session": open_session,
        "inputs": {},
        "samples_path": samples_path,
        "sessions_path": sessions_path,
    }
    _, first_session = run_session_samples(**kwargs)
    monkeypatch.setattr(
        experiment, "uuid4", lambda: first_session["session_instance_id"]
    )

    with pytest.raises(ValueError, match="session_instance_id already exists"):
        run_session_samples(
            **{**kwargs, "case_id": "other-case", "route_id": "other-route"}
        )

    assert opens == 1
    assert len(_read(samples_path)) == len(_read(sessions_path)) == 1


def test_direct_unsupported_and_unexpected_rows(tmp_path: Path) -> None:
    outcomes = iter(
        (
            UnsupportedExecution("preflight", "missing device", "accelerator"),
            RuntimeError("boom"),
        )
    )

    def run_once() -> ExecutionSample:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        raise outcome

    rows = run_direct_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
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
    assert rows[1]["failure"] == {"stage": "execution", "reason": "RuntimeError: boom"}
    assert rows[0]["backend_facts"] == rows[0]["numeric_facts"] == {}
    assert rows[1]["backend_facts"] == rows[1]["numeric_facts"] == {}


def test_session_close_and_release_failures_are_recorded(tmp_path: Path) -> None:
    class CloseFailure:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            raise ExecutionFailed(
                "release", "release failed", {"hardware_release_attempted": True}
            )

    _, close_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: CloseFailure(),
        inputs={},
        samples_path=tmp_path / "close-samples.jsonl",
        sessions_path=tmp_path / "close-sessions.jsonl",
    )
    assert close_row["status"] == "failed"
    assert close_row["failure"] == {"stage": "release", "reason": "release failed"}
    assert close_row["terminal_backend_facts"] == {"hardware_release_attempted": True}
    assert close_row["release_attempted"] is True
    assert close_row["release_succeeded"] is False
    assert close_row["release_verified"] is False

    class ReleaseFailure:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": True,
                "hardware_release_succeeded": False,
                "hardware_release_verified": False,
            }

    _, release_row = run_session_samples(
        run_id="32345678-1234-4234-8234-1234567890ab",
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: ReleaseFailure(),
        inputs={},
        samples_path=tmp_path / "release-samples.jsonl",
        sessions_path=tmp_path / "release-sessions.jsonl",
    )
    assert release_row["status"] == "failed"
    assert release_row["failure"] == {
        "stage": "session_close",
        "reason": "hardware release was not fully verified",
    }
    assert release_row["release_attempted"] is True
    assert release_row["release_succeeded"] is False
    assert release_row["release_verified"] is False


def test_contradictory_release_facts_are_preserved_and_normalized(
    tmp_path: Path,
) -> None:
    class Session:
        def run_once(self, inputs: object) -> ExecutionSample:
            return _sample()

        def close(self) -> dict[str, object]:
            return {
                "hardware_release_attempted": False,
                "hardware_release_succeeded": True,
                "hardware_release_verified": True,
            }

    sessions_path = tmp_path / "sessions.jsonl"
    _, session_row = run_session_samples(
        run_id=RUN_ID,
        experiment_id=EXPERIMENT_ID,
        case_id="case",
        route_id="route",
        identities=IDENTITIES,
        warmups=0,
        repetitions=1,
        session_protocol_id="protocol",
        open_session=lambda: Session(),
        inputs={},
        samples_path=tmp_path / "samples.jsonl",
        sessions_path=sessions_path,
    )
    assert session_row["status"] == "failed"
    assert session_row["failure"] == {
        "stage": "session_close",
        "reason": "hardware release facts are inconsistent",
    }
    assert session_row["terminal_backend_facts"] == {
        "hardware_release_attempted": False,
        "hardware_release_succeeded": True,
        "hardware_release_verified": True,
    }
    assert (
        session_row["release_attempted"],
        session_row["release_succeeded"],
        session_row["release_verified"],
    ) == (False, False, False)
    assert _read(sessions_path) == [session_row]
