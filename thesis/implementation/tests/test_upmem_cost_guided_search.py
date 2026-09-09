from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import optuna
import cotengra

from quantum_bench.circuits import builtin_circuit
from quantum_bench.model import TensorNetwork, TensorSpec


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "upmem_cost_guided_search_test", ROOT / "scripts/upmem_cost_guided_path.py"
)
search = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def test_required_search_dependency_version():
    assert optuna.__version__ == "4.5.0"
    assert cotengra.__version__ == "0.7.5"


def test_search_tells_complete_production_plan_cost_not_tree_flops(monkeypatch):
    from quantum_bench.lowering import build_contraction_dag
    from quantum_bench.upmem.execution_features import extract_launch_cost_features
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, LaunchCostScales, upmem_launch_cost_v1
    from quantum_bench.upmem.plan import UpmemTopology, plan_upmem
    from quantum_bench.upmem import runtime

    network = _network()
    scales = LaunchCostScales(1, 1, 1, 1, 1)
    weights = (2, 2, 2, 2, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("offline path search attempted an execution session")

    monkeypatch.setattr(runtime, "open_upmem", forbidden)
    monkeypatch.setattr(runtime, "open_upmem_simulator", forbidden)

    def evaluate(path, flops):
        dag = build_contraction_dag(network, path)
        plan = plan_upmem(
            dag, numeric_policy="split_complex_float32_v1", schedule_policy="static_dag_waves_v1",
            topology=UpmemTopology(dpu_count=1, tasklets_per_dpu=8, rank_count=1),
        )
        features = extract_launch_cost_features(dag, plan)
        score = upmem_launch_cost_v1(LaunchCostFacts.from_mapping(features), weights, scales)
        return search.GuidedEvaluation(score=score, facts=features)

    result = search.run_cost_guided_search(
        network, workload_id="fixture", cell_id="fixture/1dpu_t8", circuit_id="fixture", stage="software_test",
        objective_id="upmem_launch_cost_v1", profile_id="unit_scales_fixture_only",
        trace_context={"executor_id": "frozen_fixture", "extractor_id": "launch_cost", "scales_id": "unit"},
        evaluation_callback=evaluate,
    )
    assert len(result["trace"]) == 128
    for record in result["trace"]:
        assert record["told_objective"] == upmem_launch_cost_v1(
            LaunchCostFacts.from_mapping(record["facts"]), weights, scales,
        )
        assert record["score"] != record["tree_flops"]


def _network() -> TensorNetwork:
    return TensorNetwork(
        circuit=builtin_circuit("bell_2q"),
        tensors=(
            TensorSpec("a", (0, 1), (2, 2), "dense"),
            TensorSpec("b", (1, 2), (2, 2), "dense"),
            TensorSpec("c", (2, 3), (2, 2), "dense"),
            TensorSpec("d", (3, 4), (2, 2), "dense"),
        ),
        output_labels=(0, 4),
        einsum_expression="ab,bc,cd,de->ae",
    )


class _FakeTree:
    def __init__(self, path: tuple[tuple[int, int], ...], flops: float) -> None:
        self._path = path
        self._flops = flops

    def get_path(self) -> tuple[tuple[int, int], ...]:
        return self._path

    def total_flops(self) -> float:
        return self._flops


class _StrictFakeOptimizer:
    def __init__(
        self,
        *,
        max_repeats: int,
        costmod: float,
        temperature: float,
        seed: int,
        simplify: bool,
        accel: bool,
        parallel: bool,
    ) -> None:
        assert max_repeats == 1
        assert simplify is True
        assert accel is False
        assert parallel is False
        assert isinstance(seed, int)
        self.costmod = costmod
        self.temperature = temperature

    def search(
        self,
        inputs: list[tuple[int, ...]],
        output: tuple[int, ...],
        size_dict: dict[int, int],
    ) -> _FakeTree:
        assert len(inputs) == 4
        assert output == (0, 4)
        assert size_dict == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}
        if self.costmod < 2.0:
            return _FakeTree(((0, 1), (0, 1), (0, 1)), 1.0)
        return _FakeTree(((2, 3), (1, 2), (0, 1)), 2.0)


def _use_fake_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_class: type = _StrictFakeOptimizer,
) -> None:
    monkeypatch.setattr(search, "_random_greedy_optimizer_class", lambda: optimizer_class)


class _SlowFakeOptimizer(_StrictFakeOptimizer):
    def search(self, inputs, output, size_dict):
        time.sleep(0.2)
        return super().search(inputs, output, size_dict)


class _DeadFakeOptimizer(_StrictFakeOptimizer):
    def search(self, inputs, output, size_dict):
        os._exit(17)


def _run_fake(
    monkeypatch: pytest.MonkeyPatch,
    callback,
    *,
    proposals: int = 4,
    startup_trials: int | None = None,
    stage: str = "training",
    objective_id: str = "objective",
    profile_id: str = "profile",
    trace_context: dict[str, object] | None = None,
    proposal_timeout_s: float = 5.0,
    lowering_timeout_s: float = 2.0,
    search_timeout_s: float = 30.0,
    optimizer_class: type = _StrictFakeOptimizer,
) -> dict[str, object]:
    _use_fake_optimizer(monkeypatch, optimizer_class)
    return search.run_cost_guided_search(
        _network(),
        workload_id="workload",
        cell_id="cell",
        circuit_id="circuit",
        stage=stage,
        objective_id=objective_id,
        profile_id=profile_id,
        trace_context=(
            trace_context
            if trace_context is not None
            else {
                "executor_id": "executor",
                "extractor_id": "extractor",
                "scales_id": "scales",
            }
        ),
        evaluation_callback=callback,
        proposals=proposals,
        startup_trials=startup_trials or min(4, proposals),
        proposal_timeout_s=proposal_timeout_s,
        lowering_timeout_s=lowering_timeout_s,
        search_timeout_s=search_timeout_s,
    )


def test_seed_payload_is_sorted_compact_and_objective_free() -> None:
    actual = search._seed_from_identity(
        master_seed=20260909,
        workload_id="workload",
        cell_id="cell",
        seed_domain="domain",
        stage="stage",
        proposal_ordinal=3,
    )
    payload = {
        "cell_id": "cell",
        "master_seed": 20260909,
        "proposal_ordinal": 3,
        "seed_domain": "domain",
        "stage": "stage",
        "workload_id": "workload",
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    expected = int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:4], "big")
    assert actual == expected
    assert "objective" not in payload


def test_trace_keeps_duplicates_and_known_infeasible_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[tuple[int, int], ...], float]] = []

    def evaluate(path, flops):
        calls.append((path, flops))
        return search.GuidedEvaluation(
            score=None,
            known_infeasible_reason="full-plan admission failed",
        )

    result = _run_fake(monkeypatch, evaluate, proposals=4)
    trace = result["trace"]
    assert len(trace) == 4
    assert len(calls) == 4
    assert [item["tell_order"] for item in trace] == [1, 2, 3, 4]
    assert all(item["status"] == "known_infeasible" for item in trace)
    assert all(item["score"] is None for item in trace)
    assert all(item["told_objective"] == "positive_infinity" for item in trace)
    assert all(item["rejection_reason"] == "full-plan admission failed" for item in trace)
    assert trace[0]["duplicate"] is False
    assert any(item["duplicate"] is True for item in trace[1:])
    assert result["best_path_id"] is None
    assert result["trace_context"]["caller_binding"]["executor_id"] == "executor"


def test_stage_changes_seeds_but_trace_objective_and_profile_do_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def callback(path, flops):
        return search.GuidedEvaluation(score=flops, facts={})

    first = _run_fake(
        monkeypatch,
        callback,
        proposals=2,
        stage="training",
        objective_id="objective-a",
        profile_id="profile-a",
    )
    second = _run_fake(
        monkeypatch,
        callback,
        proposals=2,
        stage="training",
        objective_id="objective-b",
        profile_id="profile-b",
    )
    third = _run_fake(monkeypatch, callback, proposals=2, stage="test")
    assert first["sampler_seed"] == second["sampler_seed"]
    assert [item["proposal_seed"] for item in first["trace"]] == [
        item["proposal_seed"] for item in second["trace"]
    ]
    assert first["objective_id"] != second["objective_id"]
    assert first["profile_id"] != second["profile_id"]
    assert first["sampler_seed"] != third["sampler_seed"]



def test_fake_study_ask_tell_is_strictly_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, int]] = []

    class Trial:
        def __init__(self, number: int) -> None:
            self.number = number

        def suggest_float(self, name, low, high, *, log):
            events.append((f"suggest:{name}", self.number))
            return 1.0 if name == "costmod" else 0.5

    class Study:
        def __init__(self) -> None:
            self.next_number = 0

        def ask(self):
            trial = Trial(self.next_number)
            self.next_number += 1
            events.append(("ask", trial.number))
            return trial

        def tell(self, trial, value):
            assert trial.number == len([event for event in events if event[0] == "ask"]) - 1
            events.append(("tell", trial.number))

    _use_fake_optimizer(monkeypatch)
    monkeypatch.setattr(search, "_new_tpe_study", lambda **_: Study())
    result = search.run_cost_guided_search(
        _network(),
        workload_id="workload",
        cell_id="cell",
        stage="training",
        circuit_id="circuit",
        objective_id="objective",
        profile_id="profile",
        trace_context={"executor_id": "executor", "extractor_id": "extractor", "scales_id": "scales"},
        evaluation_callback=lambda path, flops: search.GuidedEvaluation(
            score=flops, facts={"path_length": len(path)}
        ),
        proposals=3,
        startup_trials=1,
        proposal_timeout_s=5.0,
        lowering_timeout_s=2.0,
        search_timeout_s=30.0,
    )
    assert result["completed"] is True
    assert [event[0] for event in events] == [
        "ask", "suggest:costmod", "suggest:temperature", "tell",
        "ask", "suggest:costmod", "suggest:temperature", "tell",
        "ask", "suggest:costmod", "suggest:temperature", "tell",
    ]


def test_objective_feedback_changes_tpe_after_identical_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(sign: float) -> dict[str, object]:
        return _run_fake(
            monkeypatch,
            lambda path, flops: search.GuidedEvaluation(
                score=flops if sign > 0 else 3.0 - flops,
                facts={"flops": flops},
            ),
            proposals=24,
            startup_trials=16,
        )

    lower_is_better = run(1.0)
    higher_is_better = run(-1.0)
    first = lower_is_better["trace"]
    second = higher_is_better["trace"]
    assert [item["params"] for item in first[:16]] == [item["params"] for item in second[:16]]
    assert [item["params"] for item in first[16:]] != [item["params"] for item in second[16:]]


def test_nonfinite_score_aborts_and_retains_partial_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def evaluate(path, flops):
        return search.GuidedEvaluation(score=math.nan, facts={"flops": flops})

    with pytest.raises(search.CostGuidedSearchError) as caught:
        _run_fake(monkeypatch, evaluate, proposals=3)
    assert len(caught.value.trace) == 1
    assert caught.value.trace[0]["status"] == "aborted"
    assert "ValueError" in str(caught.value.trace[0]["rejection_reason"])
    assert caught.value.trace[0]["tell_order"] is None


def test_dead_worker_aborts_with_partial_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(search.CostGuidedSearchError) as caught:
        _run_fake(
            monkeypatch,
            lambda path, flops: search.GuidedEvaluation(score=1.0, facts={}),
            proposals=2,
            optimizer_class=_DeadFakeOptimizer,
        )
    assert len(caught.value.trace) == 1
    assert "returned no result" in str(caught.value.trace[0]["rejection_reason"])


def test_proposal_timeout_aborts_with_partial_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(search.CostGuidedSearchError) as caught:
        _run_fake(
            monkeypatch,
            lambda path, flops: search.GuidedEvaluation(score=1.0, facts={}),
            proposals=2,
            proposal_timeout_s=0.05,
            optimizer_class=_SlowFakeOptimizer,
        )
    assert len(caught.value.trace) == 1
    assert "exceeded 300-second guard" in str(caught.value.trace[0]["rejection_reason"])


def test_lowering_guard_restores_signal_state_on_callback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before_handler = signal.getsignal(signal.SIGALRM)
    before_timer = signal.getitimer(signal.ITIMER_REAL)
    assert before_timer == (0.0, 0.0)

    def evaluate(path, flops):
        raise RuntimeError("callback failure")

    with pytest.raises(search.CostGuidedSearchError):
        _run_fake(monkeypatch, evaluate, proposals=2)
    assert signal.getsignal(signal.SIGALRM) is before_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_lowering_guard_timeout_restores_signal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def evaluate(path, flops):
        time.sleep(0.2)
        return search.GuidedEvaluation(score=1.0, facts={})

    with pytest.raises(search.CostGuidedSearchError) as caught:
        _run_fake(monkeypatch, evaluate, proposals=2, lowering_timeout_s=0.05)
    assert len(caught.value.trace) == 1
    assert "LoweringTimeout" in str(caught.value.trace[0]["rejection_reason"])
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_caught_timeout_is_still_an_aborted_proposal(monkeypatch):
    def evaluate(path, flops):
        try:
            time.sleep(0.2)
        except TimeoutError:
            pass
        return search.GuidedEvaluation(score=1.0, facts={})

    with pytest.raises(search.CostGuidedSearchError, match="after its deadline"):
        _run_fake(monkeypatch, evaluate, proposals=2, lowering_timeout_s=0.02)


def test_each_trace_owns_its_nested_fact_snapshot(monkeypatch):
    scratch = {"nested": [0]}

    def evaluate(path, flops):
        scratch["nested"][0] += 1
        return search.GuidedEvaluation(score=1.0, facts=scratch)

    result = _run_fake(monkeypatch, evaluate, proposals=3)
    assert [row["facts"]["nested"][0] for row in result["trace"]] == [1, 2, 3]


def test_active_signal_timer_fails_without_mutating_it() -> None:
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            search._evaluate_with_guard(lambda path, flops: search.GuidedEvaluation(score=1.0, facts={}), (), 1.0, 0.1)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def test_real_optuna_cotengra_128_trace_is_fresh_process_deterministic() -> None:
    python = sys.executable
    program = r'''
import json
from quantum_bench.circuits import builtin_circuit
from quantum_bench.model import TensorNetwork, TensorSpec
from upmem_cost_guided_path import GuidedEvaluation, run_cost_guided_search

network = TensorNetwork(
    circuit=builtin_circuit("bell_2q"),
    tensors=(
        TensorSpec("a", (0, 1), (2, 2), "dense"),
        TensorSpec("b", (1, 2), (2, 2), "dense"),
        TensorSpec("c", (2, 3), (2, 2), "dense"),
        TensorSpec("d", (3, 4), (2, 2), "dense"),
    ),
    output_labels=(0, 4),
    einsum_expression="ab,bc,cd,de->ae",
)

def evaluate(path, flops):
    return GuidedEvaluation(score=flops, facts={"path_length": len(path)})

result = run_cost_guided_search(
    network,
    workload_id="workload",
    cell_id="cell",
    circuit_id="circuit",
    stage="training",
    objective_id="objective",
    profile_id="profile",
    trace_context={"executor_id": "executor", "extractor_id": "extractor", "scales_id": "scales"},
    evaluation_callback=evaluate,
)
print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}"
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    def invoke() -> dict[str, object]:
        completed = subprocess.run(
            [str(python), "-c", program],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    first = invoke()
    second = invoke()
    for result in (first, second):
        for row in result["trace"]:
            timing = row.pop("timings_s")
            assert set(timing) == {"adaptive_ask", "candidate_generation", "plan_evaluation", "adaptive_tell"}
            assert all(math.isfinite(value) and value >= 0 for value in timing.values())
    assert first == second
    assert first["completed"] is True
    assert first["proposal_count"] == 128
    assert len(first["trace"]) == 128
    assert all(item["tell_order"] == item["proposal_index"] + 1 for item in first["trace"])
