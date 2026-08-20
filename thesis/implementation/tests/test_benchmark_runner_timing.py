from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.bench import runner
from quantum_bench.bench.config import validate_suite
from quantum_bench.bench.summary import write_summary
from quantum_bench.core.jsonio import append_jsonl
from quantum_bench.core.records import TIMING_SCHEMA_VERSION, BenchmarkCaseResult, ExecutionProfile, RouteEstimate, RouteIdentity, RouteOutput, RouteResult, TimingContract, TimingScope


class _Route:
    backend_family = "test"

    def __init__(
        self,
        route_id: str,
        validation_mode: str,
        profile: ExecutionProfile | None = None,
    ) -> None:
        self.identity = RouteIdentity(
            route_id=route_id,
            display_name=route_id,
            role="candidate",
            simulation_method="tensor_network",
            kernel_family="test",
            hardware_target="host",
            execution_mode="test",
            output_contract="tensor",
            validation_mode=validation_mode,
        )
        self.prepare_calls = 0
        self.execute_calls = 0
        self.events: list[str] | None = None
        self.fail_warmup_prepare = False
        self.fail_execute = False
        self.profile = profile or ExecutionProfile(reduction_s=0.125)

    def can_execute(self, graph: object, context: object) -> tuple[bool, None]:
        return True, None

    def estimate(self, graph: object, context: object) -> RouteEstimate:
        return RouteEstimate(self.identity.route_id, 0, 0, None)

    def prepare(self, graph: object, network: object, context: object) -> object:
        self.prepare_calls += 1
        if self.events is not None:
            self.events.append(f"prepare:{self.identity.route_id}")
        if self.fail_warmup_prepare and context.repeat_id < 0:
            raise RuntimeError("warmup prepare failed")
        return object()

    def execute(self, prepared: object, context: object) -> RouteResult:
        self.execute_calls += 1
        if self.events is not None:
            self.events.append(f"execute:{self.identity.route_id}")
        if self.fail_execute:
            raise RuntimeError("execute failed")
        return RouteResult(
            self.identity.route_id,
            self.backend_family,
            "completed",
            RouteOutput("tensor", array=np.asarray([1.0 + 0.0j])),
            self.profile,
            None,
            "unavailable",
        )


def _configure_run(tmp_path, monkeypatch, name: str, route_modes: list[tuple[str, str]], *, fail_fast: bool = False, warmups: int = 1, repeats: int = 2):
    root_dir = tmp_path / name
    run_dir = root_dir / "run"
    (run_dir / "config").mkdir(parents=True)
    suite = {
        "suite_id": "timing-cache",
        "cases": [{"case_id": "case-1", "circuit": {"name": "test"}}],
        "planner": {"engine": "opt_einsum", "optimize": "greedy"},
        "route_policy": {"routes": [route_id for route_id, _ in route_modes], "fail_fast": fail_fast},
        "warmups": warmups,
        "repeats": repeats,
        "tolerances": {},
    }
    graph = SimpleNamespace(
        path_summary=SimpleNamespace(planner="test", text="test path"),
        planning_time_s=0.0,
        tasks=(),
        tensor_network_hash="stable-network-hash",
    )
    generated = {
        "circuit": SimpleNamespace(n_qubits=1, operations=(), name="test"),
        "network": object(),
        "graph": graph,
        "generate_s": 0.0,
        "target_estimate_artifacts": {},
    }
    routes = {route_id: _Route(route_id, mode) for route_id, mode in route_modes}
    records: list[BenchmarkCaseResult] = []

    monkeypatch.setattr(runner, "load_suite", lambda path: suite)
    monkeypatch.setattr(runner, "create_run_dir", lambda *args, **kwargs: run_dir)
    monkeypatch.setattr(runner, "write_run_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "write_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "capture_environment", lambda root: {})
    monkeypatch.setattr(runner, "_generate_case", lambda *args: generated)
    monkeypatch.setattr(runner, "_persist_route_artifacts", lambda *args: None)
    monkeypatch.setattr(runner, "append_jsonl", lambda path, value: records.append(value) if isinstance(value, BenchmarkCaseResult) else None)
    monkeypatch.setattr(runner, "route_registry", lambda root: routes)
    return root_dir, run_dir, generated, routes, records


def test_mixed_reference_modes_are_cached_independently_of_route_order(tmp_path, monkeypatch) -> None:
    artifacts = []
    reference_ids = []
    for index, route_modes in enumerate(
        (
            [("tensor", "compare_output"), ("statevector", "compare_statevector")],
            [("statevector", "compare_statevector"), ("tensor", "compare_output")],
        )
    ):
        reference_calls = []
        adaptation_calls = []
        events: list[str] = []

        def fake_reference(network: object, optimize: str) -> tuple[np.ndarray, float]:
            events.append("tensor_reference")
            reference_calls.append((network, optimize))
            return np.asarray([1.0 + 0.0j]), 0.75

        def fake_adaptation(tensor: np.ndarray) -> np.ndarray:
            events.append("statevector_adaptation")
            adaptation_calls.append(tensor)
            return np.asarray(tensor)

        monkeypatch.setattr(runner, "compute_reference", fake_reference)
        monkeypatch.setattr(runner, "tensor_to_quest_statevector", fake_adaptation)
        root_dir, run_dir, generated, routes, records = _configure_run(tmp_path, monkeypatch, f"order-{index}", route_modes)
        for route in routes.values():
            route.events = events

        runner.run_suite(root_dir / "suite.yml", root_dir)

        artifact_paths = list((run_dir / "cases" / "case-1").glob("reference_*.json"))
        assert len(artifact_paths) == 1
        artifact_path = artifact_paths[0]
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifacts.append(artifact)
        reference_ids.append(artifact["reference_id"])
        assert events[:2] == ["tensor_reference", "statevector_adaptation"]
        assert reference_calls == [(generated["network"], "greedy")]
        assert len(adaptation_calls) == 1
        assert len(records) == 4
        assert {record.reference_id for record in records} == {artifact["reference_id"]}
        expected_artifact = artifact_path.relative_to(run_dir).as_posix()
        expected_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert {record.reference_artifact for record in records} == {expected_artifact}
        assert {record.reference_artifact_sha256 for record in records} == {expected_sha256}
        for record in records:
            assert "reference_s" not in record.timings
            assert record.timings["reduction_s"] == 0.125
            expected_component = "statevector_adaptation" if record.route == "statevector" else "tensor_reference"
            assert record.reference_component == expected_component
            assert record.timing_schema_version == TIMING_SCHEMA_VERSION
            assert record.timing_scope == TimingScope.ROUTE_TOTAL.value

    assert reference_ids[0] == reference_ids[1]
    for artifact in artifacts:
        assert artifact["schema_version"] == 1
        assert artifact["timing_schema_version"] == TIMING_SCHEMA_VERSION
        assert artifact["components"]["tensor_reference"] == {
            "scope": TimingScope.CASE_TENSOR_REFERENCE.value,
            "status": "completed",
            "timing_s": 0.75,
        }
        statevector = artifact["components"]["statevector_adaptation"]
        assert statevector["scope"] == TimingScope.CASE_STATEVECTOR_ADAPTATION.value
        assert statevector["status"] == "completed"
        assert statevector["timing_s"] >= 0.0


def test_reference_failure_records_repeats_and_continues_when_not_fail_fast(tmp_path, monkeypatch) -> None:
    calls = []

    def fail_reference(network: object, optimize: str):
        calls.append((network, optimize))
        raise ValueError("reference exploded")

    monkeypatch.setattr(runner, "compute_reference", fail_reference)
    modes = [("tensor", "compare_output"), ("statevector", "compare_statevector")]
    root_dir, run_dir, generated, routes, records = _configure_run(tmp_path, monkeypatch, "failure-continue", modes)

    runner.run_suite(root_dir / "suite.yml", root_dir)

    assert calls == [(generated["network"], "greedy")]
    assert len(records) == 4
    assert all(record.status == "failed" for record in records)
    assert all(record.route_metadata["failure_phase"] == "reference_generation" for record in records)
    assert all(record.route_metadata["reference_failure_component"] == "tensor_reference" for record in records)
    assert {record.reference_component for record in records} == {"tensor_reference", "statevector_adaptation"}
    assert len({record.reference_id for record in records}) == 1
    assert all(route.prepare_calls == 0 and route.execute_calls == 0 for route in routes.values())
    artifact_paths = list((run_dir / "cases" / "case-1").glob("reference_*.json"))
    assert len(artifact_paths) == 1
    artifact = json.loads(artifact_paths[0].read_text(encoding="utf-8"))
    assert artifact["components"]["tensor_reference"]["status"] == "failed"
    assert artifact["components"]["statevector_adaptation"]["status"] == "not_computed"


def test_reference_failure_records_repeats_before_fail_fast_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "compute_reference", lambda network, optimize: (_ for _ in ()).throw(ValueError("reference exploded")))
    root_dir, _, _, routes, records = _configure_run(
        tmp_path,
        monkeypatch,
        "failure-fast",
        [("tensor", "compare_output")],
        fail_fast=True,
    )

    with pytest.raises(runner._ReferenceGenerationError, match="reference exploded"):
        runner.run_suite(root_dir / "suite.yml", root_dir)

    assert len(records) == 2
    assert all(record.route_metadata["failure_phase"] == "reference_generation" for record in records)
    assert routes["tensor"].prepare_calls == 0


def test_route_total_has_explicit_field_and_legacy_total_alias(tmp_path, monkeypatch) -> None:
    generated = {
        "network": object(),
        "graph": SimpleNamespace(planning_time_s=0.0),
        "generate_s": 0.0,
    }
    ticks = iter((10.0, 14.0, 20.0, 23.0, 24.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))

    result = runner._run_repeat(
        _Route("tensor", "compare_output"),
        generated,
        {"planner": {"optimize": "greedy"}, "tolerances": {}},
        {"case_id": "case-1"},
        {},
        tmp_path,
        tmp_path,
        0,
        persist=False,
        reference=runner._RouteReference(
            "reference-id",
            "cases/case-1/reference_reference-id.json",
            "a" * 64,
            "tensor_reference",
            np.asarray([1.0 + 0.0j]),
        ),
    )

    assert result.profile.route_host_wall_s == 4.0
    assert result.profile.validation_s == 3.0
    assert result.profile.route_total_s == 14.0
    assert result.profile.total_s == result.profile.route_total_s
    assert result.profile.reduction_s == 0.125
    assert result.profile.timing_schema_version == TIMING_SCHEMA_VERSION
    assert result.profile.timing_contract.total_s_alias_of == "route_total_s"


def test_route_profile_preserves_nondefault_timing_contract(tmp_path, monkeypatch) -> None:
    generated = {
        "network": object(),
        "graph": SimpleNamespace(planning_time_s=0.0),
        "generate_s": 0.0,
    }
    contract = TimingContract(route_total_scope=TimingScope.ROUTE_HOST_WALL)
    route = _Route(
        "custom-timing",
        "benchmark_only",
        ExecutionProfile(timing_schema_version=99, timing_contract=contract),
    )
    ticks = iter((10.0, 14.0, 20.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))

    result = runner._run_repeat(
        route,
        generated,
        {"tolerances": {}},
        {"case_id": "case-1"},
        {},
        tmp_path,
        tmp_path,
        0,
        persist=False,
    )

    assert result.profile.timing_schema_version == 99
    assert result.profile.timing_contract is contract
    assert result.profile.timing_contract.route_total_scope is TimingScope.ROUTE_HOST_WALL


def test_warmup_failure_is_separate_and_measured_attempts_continue(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "compute_reference", lambda network, optimize: (np.asarray([1.0 + 0.0j]), 0.25))
    root_dir, run_dir, _, routes, records = _configure_run(
        tmp_path,
        monkeypatch,
        "warmup-continue",
        [("tensor", "compare_output")],
    )
    routes["tensor"].fail_warmup_prepare = True

    runner.run_suite(root_dir / "suite.yml", root_dir)

    assert len(records) == 2
    assert {record.repeat_id for record in records} == {0, 1}
    assert routes["tensor"].prepare_calls == 3
    assert routes["tensor"].execute_calls == 2
    failures = list((run_dir / "cases" / "case-1" / "warmup_failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["warmup_id"] == 0
    assert failure["failure_phase"] == "prepare"
    assert failure["timings"]["total_s"] == failure["timings"]["route_total_s"]


def test_warmup_failure_artifact_is_written_before_fail_fast_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "compute_reference", lambda network, optimize: (np.asarray([1.0 + 0.0j]), 0.25))
    root_dir, run_dir, _, routes, records = _configure_run(
        tmp_path,
        monkeypatch,
        "warmup-fast",
        [("tensor", "compare_output")],
        fail_fast=True,
    )
    routes["tensor"].fail_warmup_prepare = True

    with pytest.raises(runner._RouteAttemptError, match="warmup prepare failed"):
        runner.run_suite(root_dir / "suite.yml", root_dir)

    assert records == []
    assert len(list((run_dir / "cases" / "case-1" / "warmup_failures").glob("*.json"))) == 1


def test_measured_execute_failure_preserves_partial_timing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "compute_reference", lambda network, optimize: (np.asarray([1.0 + 0.0j]), 0.25))
    ticks = iter((0.0, 10.0, 14.0, 20.0, 25.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    root_dir, _, _, routes, records = _configure_run(
        tmp_path,
        monkeypatch,
        "measured-failure",
        [("tensor", "compare_output")],
        warmups=0,
    )
    routes["tensor"].fail_execute = True

    runner.run_suite(root_dir / "suite.yml", root_dir)

    assert len(records) == 2
    assert [record.timings["route_total_s"] for record in records] == [4.0, 5.0]
    for record in records:
        assert record.status == "failed"
        assert record.route_metadata["failure_phase"] == "execute"
        assert record.timings["route_host_wall_s"] == record.timings["route_total_s"]
        assert record.timings["total_s"] == record.timings["route_total_s"]
        assert record.total_time_s == record.timings["route_total_s"]


def test_validation_exception_is_wrapped_with_partial_timing(tmp_path, monkeypatch) -> None:
    generated = {"network": object(), "graph": SimpleNamespace(planning_time_s=0.0), "generate_s": 0.0}
    ticks = iter((10.0, 14.0, 20.0, 23.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(runner, "validate", lambda actual, reference, tolerances: (_ for _ in ()).throw(ValueError("validation failed")))
    reference = runner._RouteReference("reference-id", "cases/case/reference_id.json", "b" * 64, "tensor_reference", np.asarray([1.0 + 0.0j]))

    with pytest.raises(runner._RouteAttemptError) as failure:
        runner._run_repeat(
            _Route("tensor", "compare_output"),
            generated,
            {"tolerances": {}},
            {"case_id": "case-1"},
            {},
            tmp_path,
            tmp_path,
            0,
            persist=False,
            reference=reference,
        )

    assert failure.value.phase == "validation"
    assert failure.value.profile.route_host_wall_s == 4.0
    assert failure.value.profile.validation_s == 3.0
    assert failure.value.profile.route_total_s == 13.0
    assert failure.value.profile.total_s == 13.0


def test_summary_preserves_mixed_timing_contracts_as_separate_rows(tmp_path) -> None:
    base = {
        "case_id": "case-1",
        "route": "route-a",
        "status": "passed",
        "total_time_s": 1.0,
        "energy_joules": None,
        "energy_source": "unavailable",
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "timing_scope": TimingScope.ROUTE_TOTAL.value,
    }
    run_dir = tmp_path / "mixed-contracts"
    append_jsonl(run_dir / "raw" / "case.jsonl", base)
    append_jsonl(run_dir / "raw" / "case.jsonl", {**base, "route": "route-b", "timing_scope": "kernel_only"})
    append_jsonl(run_dir / "raw" / "case.jsonl", {**base, "route": "route-c", "timing_schema_version": TIMING_SCHEMA_VERSION + 1})
    summary = write_summary(run_dir)

    assert summary["timing_schema_version"] is None
    assert summary["timing_scope"] is None
    assert summary["timing_contracts"] == [
        {"timing_schema_version": TIMING_SCHEMA_VERSION, "timing_scope": "kernel_only"},
        {"timing_schema_version": TIMING_SCHEMA_VERSION, "timing_scope": TimingScope.ROUTE_TOTAL.value},
        {"timing_schema_version": TIMING_SCHEMA_VERSION + 1, "timing_scope": TimingScope.ROUTE_TOTAL.value},
    ]
    rows = {row["route"]: row for row in summary["rows"]}
    assert rows["route-a"]["timing_scope"] == TimingScope.ROUTE_TOTAL.value
    assert rows["route-b"]["timing_scope"] == "kernel_only"
    assert rows["route-c"]["timing_schema_version"] == TIMING_SCHEMA_VERSION + 1


def test_summary_rejects_inconsistent_contract_within_case_route_group(tmp_path) -> None:
    base = {
        "case_id": "case-1",
        "route": "route-a",
        "status": "passed",
        "total_time_s": 1.0,
        "energy_joules": None,
        "energy_source": "unavailable",
        "timing_schema_version": TIMING_SCHEMA_VERSION,
        "timing_scope": TimingScope.ROUTE_TOTAL.value,
    }
    append_jsonl(tmp_path / "raw" / "case.jsonl", base)
    append_jsonl(tmp_path / "raw" / "case.jsonl", {**base, "timing_scope": "kernel_only"})

    with pytest.raises(ValueError, match="Inconsistent timing contract"):
        write_summary(tmp_path)


def test_suite_validation_rejects_duplicate_case_ids() -> None:
    suite = {
        "cases": [
            {"case_id": "duplicate", "circuit": {"name": "a"}},
            {"case_id": "duplicate", "circuit": {"name": "b"}},
        ],
        "repeats": 1,
        "warmups": 0,
        "route_policy": {"routes": ["route-a"]},
    }

    with pytest.raises(ValueError, match="Duplicate case_id: duplicate"):
        validate_suite(suite)
