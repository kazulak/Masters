"""Focused tests for the direct tensor-network baselines."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import subprocess
from types import MappingProxyType

import numpy as np
import pytest

from quantum_bench import baselines
from quantum_bench.circuits import builtin_circuit, quest_compatible_circuit
from quantum_bench.model import CircuitOperation, CircuitSpec, make_simulation_job
from quantum_bench.results import ExecutionFailed, ExecutionSample, UnsupportedExecution


def _job(name: str = "bell_2q"):
    return make_simulation_job(builtin_circuit(name))


def _complex_job():
    circuit = CircuitSpec(
        "complex_fixture",
        1,
        (CircuitOperation("h", (0,)), CircuitOperation("s", (0,))),
        {"kind": "fixture"},
    )
    return make_simulation_job(circuit)


def _order_distinguishing_complex_job():
    circuit = CircuitSpec(
        "complex_order_fixture",
        2,
        (
            CircuitOperation("h", (0,)),
            CircuitOperation("s", (0,)),
            CircuitOperation("h", (1,)),
            CircuitOperation("t", (1,)),
            CircuitOperation("cx", (0, 1)),
        ),
        {"kind": "fixture", "purpose": "axis-order regression"},
    )
    return make_simulation_job(circuit)


def _quest_job(name: str, **params):
    source = {"kind": "quest_compatible", "name": name, **params}
    return make_simulation_job(quest_compatible_circuit(name, source))


def _fake_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "native" / "quest_cpu" / "bin" / "quest_runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    return runner


def _fake_quest_process(monkeypatch, job, *, payload_updates=None, dump_updates=None):
    calls = []
    payload_updates = payload_updates or {}
    dump_updates = dump_updates or {}

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["cwd"] == Path(command[0]).parent.parent
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        dump_path = Path(command[command.index("--dump-state-json") + 1])
        algorithm = command[command.index("--algo") + 1]
        if algorithm == "HS":
            input_qubits = int(command[command.index("--logical-qubits") + 1])
            allocated_qubits = input_qubits * 2
        else:
            input_qubits = int(command[command.index("--qubits") + 1])
            allocated_qubits = input_qubits
        repeat_layers = int(command[command.index("--repeat-layers") + 1])
        one_qubit = sum(
            len(operation.wires) == 1 for operation in job.circuit.operations
        )
        two_qubit = sum(
            len(operation.wires) == 2 for operation in job.circuit.operations
        )
        payload = {
            "status": "ok",
            "algo": algorithm,
            "input_qubits": input_qubits,
            "allocated_qubits": allocated_qubits,
            "depth": 0,
            "repeat_layers": repeat_layers,
            "state_dump_requested": True,
            "one_qubit_gates": one_qubit,
            "two_qubit_gates": two_qubit,
            "time_s": 1.25,
            "state_dump_time_s": 0.015,
            "quest_version": "QuEST-test",
            "energy_joules": None,
            "energy_source": "unavailable",
        }
        payload.update(payload_updates)
        dump = {
            "schema_version": "quest_state_dump_v1",
            "basis_order": "quest_little_endian_integer_index",
            "allocated_qubits": allocated_qubits,
            "amplitude_count": 1 << allocated_qubits,
            "quest_version": "QuEST-test",
            "real": [1.0] + [0.0] * ((1 << allocated_qubits) - 1),
            "imag": [0.0] * (1 << allocated_qubits),
        }
        dump.update(dump_updates)
        dump_path.write_text(json.dumps(dump), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(baselines.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize(
    ("name", "params", "expected_algorithm"),
    [
        ("qrng", {"n_qubits": 2, "repeat_layers": 2}, "QRNG"),
        ("bb_n", {"n_qubits": 2}, "BB84"),
        ("bernstein_vazirani", {"n_qubits": 3}, "BV"),
        ("dense_coding", {"n_qubits": 2}, "EDC"),
        ("hidden_shift", {"logical_qubits": 2, "allocated_qubits": 4}, "HS"),
        ("parity", {"n_qubits": 3}, "XOR"),
    ],
)
def test_quest_cpu_runs_all_canonical_aliases(
    monkeypatch, tmp_path, name, params, expected_algorithm
):
    job = _quest_job(name, **params)
    runner = _fake_runner(tmp_path)
    calls = _fake_quest_process(monkeypatch, job)

    sample = baselines.run_quest_cpu(job, runner=runner)

    assert sample.output.dtype == np.dtype("complex128")
    expected = np.zeros(1 << job.circuit.n_qubits, dtype=np.complex128)
    expected[0] = 1
    np.testing.assert_array_equal(sample.output, expected)
    assert not sample.output.flags.writeable
    assert sample.measurement.scope_id == "simulation_end_to_end_v1"
    assert sample.measurement.kernel_s == 1.25
    assert sample.measurement.decode_s is not None
    assert sample.measurement.energy_j is None
    facts = sample.backend_facts
    assert facts["backend_id"] == "quest_cpu_full_state_v1"
    assert facts["execution_class"] == "external_process_cpu"
    assert facts["hardware_execution"] is True
    assert facts["physical_upmem_execution"] is False
    assert facts["target_observed"] == "cpu"
    assert facts["native_depth"] == 0
    assert re.fullmatch(r"[0-9a-f]{64}", facts["runner_sha256"])
    assert facts["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    provenance_command = facts["command"]
    assert "<state_dump.json>" in provenance_command
    assert not any("qbench-quest-" in item for item in provenance_command)
    command, kwargs = calls[0]
    assert command[command.index("--algo") + 1] == expected_algorithm
    assert "--depth" not in command
    assert kwargs["cwd"] == runner.parent.parent
    if expected_algorithm == "HS":
        assert "--logical-qubits" in command
        assert "--qubits" not in command
    else:
        assert "--qubits" in command


def test_quest_cpu_resolves_relative_runner(monkeypatch, tmp_path):
    job = _quest_job("qrng", n_qubits=2)
    runner = _fake_runner(tmp_path)
    calls = _fake_quest_process(monkeypatch, job)
    monkeypatch.chdir(tmp_path)

    relative_runner = runner.relative_to(tmp_path)
    sample = baselines.run_quest_cpu(job, runner=relative_runner)

    assert sample.backend_facts["runner"] == str(runner.resolve())
    assert calls[0][0][0] == str(runner.resolve())


def test_quest_cpu_repeats_gate_counts_and_preserves_job(monkeypatch, tmp_path):
    job = _quest_job("qrng", n_qubits=2, repeat_layers=3)
    original = job
    calls = _fake_quest_process(monkeypatch, job)
    sample = baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    command = calls[0][0]
    assert command[command.index("--repeat-layers") + 1] == "3"
    assert sample.backend_facts["native_compute_time_s"] == 1.25
    assert job == original


@pytest.mark.parametrize(
    "mutator",
    [
        lambda job: make_simulation_job(builtin_circuit("qrng", {"n_qubits": 2})),
        lambda job: make_simulation_job(
            CircuitSpec("qasm", 1, (), {"kind": "qasm_file"})
        ),
        lambda job: make_simulation_job(
            CircuitSpec(
                "random",
                2,
                (),
                {
                    "kind": "quest_compatible",
                    "name": "RANDOM",
                    "deterministic_unitary": True,
                },
            )
        ),
        lambda job: make_simulation_job(
            quest_compatible_circuit(
                "qrng", {"kind": "quest_compatible", "name": "qrng", "n_qubits": 2}
            )
        ),
    ],
)
def test_quest_cpu_rejects_noncanonical_sources(mutator, tmp_path):
    job = mutator(_quest_job("qrng", n_qubits=2))
    if (
        job.circuit.source.get("kind") == "quest_compatible"
        and job.circuit.source.get("name") == "qrng"
    ):
        operations = tuple((*job.circuit.operations, CircuitOperation("x", (0,))))
        job = make_simulation_job(
            CircuitSpec("altered", 2, operations, dict(job.circuit.source))
        )
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    assert error.value.stage == "preflight"


@pytest.mark.parametrize(
    "job_factory",
    [
        lambda: make_simulation_job(
            quest_compatible_circuit(
                "qrng", {"kind": "quest_compatible", "name": "qrng", "n_qubits": 2}
            )
        ).__class__(
            circuit=quest_compatible_circuit(
                "qrng", {"kind": "quest_compatible", "name": "qrng", "n_qubits": 2}
            ),
            parameters=(("x", 1),),
        ),
        lambda: make_simulation_job(
            quest_compatible_circuit(
                "qrng", {"kind": "quest_compatible", "name": "qrng", "n_qubits": 2}
            ),
            seed=7,
        ),
    ],
)
def test_quest_cpu_rejects_seed_and_parameters(job_factory, tmp_path):
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quest_cpu(job_factory(), runner=_fake_runner(tmp_path))
    assert error.value.stage == "preflight"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_quest_cpu_rejects_invalid_timeout(timeout, tmp_path):
    with pytest.raises(UnsupportedExecution):
        baselines.run_quest_cpu(
            _quest_job("qrng", n_qubits=1),
            runner=_fake_runner(tmp_path),
            timeout_s=timeout,
        )


def test_quest_cpu_rejects_invalid_output_cap_and_runner(tmp_path):
    runner = _fake_runner(tmp_path)
    with pytest.raises(UnsupportedExecution):
        baselines.run_quest_cpu(
            _quest_job("qrng", n_qubits=2), runner=runner, max_output_amplitudes=3
        )
    missing = tmp_path / "missing"
    with pytest.raises(UnsupportedExecution):
        baselines.run_quest_cpu(_quest_job("qrng", n_qubits=2), runner=missing)


@pytest.mark.parametrize(
    "payload_updates",
    [
        {"status": "failed"},
        {"quest_version": ""},
        {"time_s": -1.0},
        {"one_qubit_gates": 999},
        {"state_dump_requested": False},
    ],
)
def test_quest_cpu_rejects_bad_native_stdout(monkeypatch, tmp_path, payload_updates):
    job = _quest_job("qrng", n_qubits=2)
    _fake_quest_process(monkeypatch, job, payload_updates=payload_updates)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    assert error.value.stage in {"stdout", "status"}


def test_quest_cpu_rejects_mismatched_native_depth(monkeypatch, tmp_path):
    job = _quest_job("qrng", n_qubits=2)
    _fake_quest_process(monkeypatch, job, payload_updates={"depth": 1})
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    assert error.value.stage == "stdout"


@pytest.mark.parametrize(
    ("energy_joules", "energy_source"),
    [
        (None, "rapl_measured"),
        (-1.0, "rapl_measured"),
        (float("nan"), "rapl_measured"),
        (1.0, "unavailable"),
        (1.0, ""),
        (1.0, 1),
    ],
)
def test_quest_cpu_rejects_bad_energy_metadata(
    monkeypatch, tmp_path, energy_joules, energy_source
):
    job = _quest_job("qrng", n_qubits=2)
    _fake_quest_process(
        monkeypatch,
        job,
        payload_updates={
            "energy_joules": energy_joules,
            "energy_source": energy_source,
        },
    )
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    assert error.value.stage == "stdout"


def test_quest_cpu_accepts_measured_energy_pair(monkeypatch, tmp_path):
    job = _quest_job("qrng", n_qubits=2)
    _fake_quest_process(
        monkeypatch,
        job,
        payload_updates={"energy_joules": 1.5, "energy_source": "rapl_measured"},
    )

    sample = baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))

    assert sample.backend_facts["native_compute_energy_j"] == 1.5
    assert sample.backend_facts["energy_source"] == "rapl_measured"


def test_quest_cpu_preflight_and_semantic_validation_timing_order(
    monkeypatch, tmp_path
):
    job = _quest_job("qrng", n_qubits=2)
    runner = _fake_runner(tmp_path)
    _fake_quest_process(monkeypatch, job)
    events = []

    original_preflight = baselines._preflight_quest_runner
    original_validate = baselines._validate_quest_cpu_request

    def preflight(value):
        events.append("preflight")
        return original_preflight(value)

    def validate(value, **kwargs):
        events.append("semantic")
        return original_validate(value, **kwargs)

    def clock():
        events.append("timer")
        return float(len(events))

    monkeypatch.setattr(baselines, "_preflight_quest_runner", preflight)
    monkeypatch.setattr(baselines, "_validate_quest_cpu_request", validate)
    monkeypatch.setattr(baselines.time, "perf_counter", clock)

    baselines.run_quest_cpu(job, runner=runner)

    assert events[:3] == ["preflight", "timer", "semantic"]


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected_stage"),
    [
        (2, "{}", "process"),
        (0, "not-json", "stdout"),
    ],
)
def test_quest_cpu_rejects_process_and_stdout_failures(
    monkeypatch, tmp_path, returncode, stdout, expected_stage
):
    job = _quest_job("qrng", n_qubits=2)
    runner = _fake_runner(tmp_path)

    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout, "native error")

    monkeypatch.setattr(baselines.subprocess, "run", fail)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=runner)
    assert error.value.stage == expected_stage


@pytest.mark.parametrize(
    "dump_updates",
    [
        {"schema_version": "wrong"},
        {"basis_order": "big_endian"},
        {"quest_version": ""},
        {"quest_version": "Other-QuEST"},
        {"amplitude_count": 3},
        {"real": [float("nan"), 0.0, 0.0, 0.0]},
        {"imag": [0.0]},
    ],
)
def test_quest_cpu_rejects_bad_state_dump(monkeypatch, tmp_path, dump_updates):
    job = _quest_job("qrng", n_qubits=2)
    _fake_quest_process(monkeypatch, job, dump_updates=dump_updates)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=_fake_runner(tmp_path))
    assert error.value.stage == "decode"


def test_quest_cpu_timeout_is_execution_failure(monkeypatch, tmp_path):
    job = _quest_job("qrng", n_qubits=2)
    runner = _fake_runner(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(baselines.subprocess, "run", timeout)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quest_cpu(job, runner=runner)
    assert error.value.stage == "process"
    assert error.value.backend_facts["timeout_s"] == 120.0


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_bell_has_canonical_statevector(runner):
    sample = runner(_job())
    expected = np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(sample.output, expected)


def test_nonzero_imaginary_amplitude_and_route_equality():
    job = _complex_job()
    quimb_sample = baselines.run_quimb(job)
    cotengra_sample = baselines.run_cotengra(job)
    expected = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    np.testing.assert_allclose(quimb_sample.output, expected)
    np.testing.assert_allclose(cotengra_sample.output, expected)
    np.testing.assert_allclose(quimb_sample.output, cotengra_sample.output)


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_multiqubit_complex_fixture_preserves_quest_basis_order(runner):
    sample = runner(_order_distinguishing_complex_job())
    phase = np.exp(1j * np.pi / 4)
    expected = np.array([1, 1j * phase, phase, 1j], dtype=np.complex128) / 2
    np.testing.assert_allclose(sample.output, expected)


def test_lowering_once_input_unchanged_and_output_deterministic(monkeypatch):
    job = _job()
    original_lower = baselines.lower_tensor_network
    calls = 0

    def counted_lower(value):
        nonlocal calls
        calls += 1
        return original_lower(value)

    monkeypatch.setattr(baselines, "lower_tensor_network", counted_lower)
    first = baselines.run_quimb(job)
    second = baselines.run_quimb(job)

    assert calls == 2
    assert job == _job()
    np.testing.assert_array_equal(first.output, second.output)
    assert first.backend_facts == second.backend_facts
    assert first.numeric_facts == second.numeric_facts


def test_timing_facts_and_result_are_immutable():
    sample = baselines.run_cotengra(_job(), methods="greedy", max_repeats=1)
    assert isinstance(sample, ExecutionSample)
    assert sample.output.dtype == np.dtype("complex128")
    assert not sample.output.flags.writeable
    assert sample.measurement.scope_id == "simulation_end_to_end_v1"
    assert sample.measurement.total_wall_s >= 0.0
    assert sample.measurement.lowering_s is not None
    assert sample.measurement.planning_s is not None
    assert sample.measurement.kernel_s is not None
    assert sample.measurement.decode_s is not None
    assert sample.backend_facts["backend_id"] == "cotengra_tn_v1"
    assert sample.backend_facts["methods"] == "greedy"
    assert sample.backend_facts["max_repeats"] == 1
    assert sample.backend_facts["hardware_execution"] is False
    assert sample.backend_facts["deterministic_planning_seed"] == 0
    assert sample.backend_facts["deterministic_planning_rngs"] == (
        "python_random",
        "numpy_legacy",
        "cotengra_hyperoptimizer",
    )
    assert "optimizer_seed" not in sample.backend_facts
    assert re.fullmatch(
        r"[0-9a-f]{64}", sample.backend_facts["contraction_path_fingerprint"]
    )
    assert sample.backend_facts["contraction_path_length"] > 0
    assert sample.numeric_facts["output_dtype"] == "complex128"
    assert isinstance(sample.backend_facts, MappingProxyType)
    with pytest.raises(ValueError):
        sample.output[0] = 0
    with pytest.raises(TypeError):
        sample.backend_facts["backend_id"] = "changed"


@pytest.mark.parametrize(
    "runner, kwargs",
    [
        (baselines.run_quimb, {"optimize": "invalid"}),
        (baselines.run_cotengra, {"methods": "invalid"}),
        (baselines.run_cotengra, {"max_repeats": 0}),
    ],
)
def test_invalid_configuration_is_unsupported(runner, kwargs):
    with pytest.raises(UnsupportedExecution) as error:
        runner(_job(), **kwargs)
    assert error.value.stage == "preflight"


def test_random_quimb_optimizer_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quimb(_job(), optimize="random-greedy")
    assert error.value.stage == "preflight"


def test_auto_quimb_optimizer_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_quimb(_job(), optimize="auto")
    assert error.value.stage == "preflight"


def test_cotengra_invalid_prefix_is_unsupported():
    with pytest.raises(UnsupportedExecution) as error:
        baselines.run_cotengra(_job(), methods="greedy-not-a-method")
    assert error.value.stage == "preflight"


def test_cotengra_labels_is_an_accepted_deterministic_method():
    first = baselines.run_cotengra(_job("ghz_4q"), methods="labels")
    second = baselines.run_cotengra(_job("ghz_4q"), methods="labels")
    assert first.backend_facts["methods"] == "labels"
    assert (
        first.backend_facts["contraction_path_fingerprint"]
        == second.backend_facts["contraction_path_fingerprint"]
    )


@pytest.mark.parametrize("runner", [baselines.run_quimb, baselines.run_cotengra])
def test_contraction_path_provenance_is_sha256_and_stable(runner):
    first = runner(_job("ghz_4q"))
    second = runner(_job("ghz_4q"))

    first_fingerprint = first.backend_facts["contraction_path_fingerprint"]
    second_fingerprint = second.backend_facts["contraction_path_fingerprint"]
    assert isinstance(first_fingerprint, str)
    assert re.fullmatch(r"[0-9a-f]{64}", first_fingerprint)
    assert first_fingerprint == second_fingerprint
    assert first.backend_facts["contraction_path_length"] > 0
    assert (
        first.backend_facts["contraction_path_length"]
        == second.backend_facts["contraction_path_length"]
    )


def test_cotengra_path_is_deterministic_across_external_rng_states():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(17)
        np.random.seed(19)
        first = baselines.run_cotengra(_job("ghz_4q"))

        random.seed(101)
        np.random.seed(103)
        second = baselines.run_cotengra(_job("ghz_4q"))
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)

    first_path = first.backend_facts["contraction_path_fingerprint"]
    second_path = second.backend_facts["contraction_path_fingerprint"]
    assert isinstance(first_path, str) and first_path
    assert first_path == second_path


def test_cotengra_restores_python_and_numpy_rng_states():
    original_python_state = random.getstate()
    original_numpy_state = np.random.get_state()
    try:
        random.seed(211)
        np.random.seed(223)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()

        baselines.run_cotengra(_job("ghz_4q"))

        assert random.getstate() == expected_python_state
        _assert_numpy_rng_state_equal(np.random.get_state(), expected_numpy_state)
    finally:
        random.setstate(original_python_state)
        np.random.set_state(original_numpy_state)


def test_nonfinite_decoded_output_is_a_decode_failure(monkeypatch):
    monkeypatch.setattr(
        baselines,
        "_tensor_to_quest_statevector",
        lambda tensor: np.array([np.nan + 0j], dtype=np.complex128),
    )
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quimb(_job())
    assert error.value.stage == "decode"


def test_unexpected_contraction_error_reports_kernel_stage(monkeypatch):
    original_contract = baselines._build_quimb_network

    def failing_network(qtn, network, inputs):
        tensor_network, output_inds = original_contract(qtn, network, inputs)

        def fail(*args, **kwargs):
            raise RuntimeError("fixture contraction failure")

        tensor_network.contract = fail
        return tensor_network, output_inds

    monkeypatch.setattr(baselines, "_build_quimb_network", failing_network)
    with pytest.raises(ExecutionFailed) as error:
        baselines.run_quimb(_job())
    assert error.value.stage == "planning"


def test_import_boundary_is_canonical_and_public_api_is_function_only():
    tree = ast.parse(inspect.getsource(baselines))
    forbidden = {
        "providers",
        "routing",
        "TaskGraph",
        "TensorNetworkValue",
        "ContractionDAG",
        "execution",
        "registry",
    }
    imports = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert all(not any(item in module for item in forbidden) for module in imports)
    assert set(baselines.__all__) == {"run_quimb", "run_cotengra", "run_quest_cpu"}
    assert all(callable(getattr(baselines, name)) for name in baselines.__all__)


def _assert_numpy_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]
