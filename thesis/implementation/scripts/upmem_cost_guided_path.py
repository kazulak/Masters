"""Private, bounded cost-guided path study over the frozen UPMEM executor."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import queue as queue_module
from pathlib import Path
import signal
import threading
import time

from quantum_bench.planning import (
    _network_preflight,
    _size_dict,
    _validate_pairwise_path,
)
from quantum_bench.upmem.path_heuristic import path_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "configs/upmem_cost_guided_path_study_v1.json"


@dataclass(frozen=True, slots=True)
class GuidedEvaluation:
    """The caller's complete-plan result for one generated contraction path."""

    score: float | None
    facts: Mapping[str, object] | None = None
    known_infeasible_reason: str | None = None

    def __post_init__(self) -> None:
        if self.score is None:
            if not isinstance(self.known_infeasible_reason, str) or not self.known_infeasible_reason.strip():
                raise ValueError("known-infeasible evaluations need a nonempty reason")
        else:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("eligible score must be numeric")
            score = float(self.score)
            if not math.isfinite(score) or score < 0:
                raise ValueError("eligible score must be finite and nonnegative")
            object.__setattr__(self, "score", score)
            if self.known_infeasible_reason is not None:
                raise ValueError("eligible evaluations cannot have an infeasibility reason")
            if not isinstance(self.facts, Mapping):
                raise ValueError("eligible evaluations need a facts mapping")

        if self.facts is not None:
            if not isinstance(self.facts, Mapping):
                raise ValueError("evaluation facts must be a mapping")
            try:
                json.dumps(self.facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError("evaluation facts must be JSON-safe") from exc

class CostGuidedSearchError(RuntimeError):
    """An aborted search carrying the trace accumulated before the abort."""

    def __init__(self, message: str, trace: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.trace = tuple(trace)
        self.partial_trace = self.trace


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _require_nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _bounded_parameter(value: object, low: float, high: float, name: str) -> float:
    value = _require_positive_finite(value, name)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside its declared search bounds")
    return value


def _seed_from_identity(
    *,
    master_seed: int,
    workload_id: str,
    cell_id: str,
    seed_domain: str,
    stage: str,
    proposal_ordinal: int | None,
) -> int:
    """Derive paired sampler/proposal seeds without objective-dependent state."""

    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise ValueError("master_seed must be an integer")
    for value, name in ((workload_id, "workload_id"), (cell_id, "cell_id"), (seed_domain, "seed_domain"), (stage, "stage")):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    if proposal_ordinal is not None and (
        isinstance(proposal_ordinal, bool)
        or not isinstance(proposal_ordinal, int)
        or proposal_ordinal < 0
    ):
        raise ValueError("proposal_ordinal must be a nonnegative integer or None")
    payload = {
        "cell_id": cell_id,
        "master_seed": master_seed,
        "proposal_ordinal": proposal_ordinal,
        "seed_domain": seed_domain,
        "stage": stage,
        "workload_id": workload_id,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big", signed=False)


def _random_greedy_optimizer_class() -> type:
    try:
        from cotengra import RandomGreedyOptimizer
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("cotengra==0.7.5 is required for cost-guided search") from exc
    return RandomGreedyOptimizer


def _random_greedy_worker(
    inputs: list[tuple[int, ...]],
    output: tuple[int, ...],
    size_dict: dict[int, int],
    costmod: float,
    temperature: float,
    proposal_seed: int,
    result_queue: object,
) -> None:
    try:
        optimizer = _random_greedy_optimizer_class()(
            max_repeats=1,
            costmod=costmod,
            temperature=temperature,
            seed=proposal_seed,
            simplify=True,
            accel=False,
            parallel=False,
        )
        tree = optimizer.search(inputs, output, size_dict)
        raw_path = tree.get_path()
        flops = _require_nonnegative_finite(tree.total_flops(), "tree FLOPs")
        result_queue.put((raw_path, flops, None))
    except BaseException as exc:  # The parent converts this into an aborted trace.
        result_queue.put((None, None, f"{type(exc).__name__}:{exc}"))


def _isolated_random_greedy(
    *,
    inputs: list[tuple[int, ...]],
    output: tuple[int, ...],
    size_dict: dict[int, int],
    tensor_count: int,
    costmod: float,
    temperature: float,
    proposal_seed: int,
    deadline: float,
) -> tuple[tuple[tuple[int, int], ...], float]:
    """Run exactly one fresh optimizer in the existing Linux fork guard."""

    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:  # pragma: no cover - supported research host is Linux
        raise RuntimeError("missing fork-based proposal guard") from exc
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_random_greedy_worker,
        args=(inputs, output, size_dict, costmod, temperature, proposal_seed, result_queue),
    )
    process.start()
    result = None
    try:
        while result is None:
            try:
                result = result_queue.get_nowait()
            except queue_module.Empty:
                if not process.is_alive():
                    process.join()
                    try:
                        result = result_queue.get(timeout=0.25)
                    except queue_module.Empty as exc:
                        raise RuntimeError(
                            "random-greedy proposal returned no result "
                            f"(exitcode={process.exitcode})"
                        ) from exc
                if time.monotonic() >= deadline:
                    process.kill()
                    process.join()
                    raise TimeoutError("random-greedy proposal exceeded 300-second guard")
                time.sleep(0.01)
    finally:
        if process.is_alive():
            process.join(timeout=0.25)
        if process.is_alive():
            process.kill()
        process.join()
        result_queue.close()

    raw_path, flops, error = result
    if error is not None:
        raise RuntimeError(f"random-greedy proposal failed: {error}")
    canonical_path = _validate_pairwise_path(raw_path, tensor_count)
    return canonical_path, _require_nonnegative_finite(flops, "tree FLOPs")


class _LoweringTimeout(TimeoutError):
    pass


def _evaluate_with_guard(
    callback: Callable[[tuple[tuple[int, int], ...], float], GuidedEvaluation],
    canonical_path: tuple[tuple[int, int], ...],
    tree_flops: float,
    timeout_s: float,
) -> GuidedEvaluation:
    """Apply the existing lowering/evaluation guard around the caller callback."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("missing 60-second lowering guard: main thread required")
    try:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        if previous_timer != (0.0, 0.0):
            raise RuntimeError("missing 60-second lowering guard: SIGALRM timer already active")
    except (AttributeError, OSError, ValueError) as exc:  # pragma: no cover - Linux main thread
        raise RuntimeError("missing 60-second lowering guard") from exc

    handler_installed = False
    timer_armed = False

    def alarm_handler(_signum: int, _frame: object) -> None:
        raise _LoweringTimeout("lowering/evaluation exceeded its guard")

    try:
        signal.signal(signal.SIGALRM, alarm_handler)
        handler_installed = True
        deadline = time.monotonic() + timeout_s
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        timer_armed = True
        evaluation = callback(canonical_path, tree_flops)
        if time.monotonic() > deadline:
            raise _LoweringTimeout("lowering/evaluation completed after its deadline")
    finally:
        try:
            if timer_armed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            if handler_installed:
                signal.signal(signal.SIGALRM, previous_handler)
    if not isinstance(evaluation, GuidedEvaluation):
        raise TypeError("evaluation callback must return GuidedEvaluation")
    return evaluation


def _new_tpe_study(*, sampler_seed: int, startup_trials: int) -> object:
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:  # pragma: no cover - dependency is environment-specific
        raise RuntimeError("optuna==4.5.0 is required for cost-guided search") from exc
    sampler = TPESampler(
        seed=sampler_seed,
        n_startup_trials=startup_trials,
        multivariate=False,
        group=False,
        constant_liar=False,
    )
    return optuna.create_study(direction="minimize", sampler=sampler)


def run_cost_guided_search(
    network: object,
    *,
    workload_id: str,
    cell_id: str,
    stage: str,
    objective_id: str,
    profile_id: str,
    trace_context: Mapping[str, object],
    evaluation_callback: Callable[[tuple[tuple[int, int], ...], float], GuidedEvaluation],
    master_seed: int = 20260909,
    proposals: int = 128,
    startup_trials: int = 16,
    costmod_low: float = 0.1,
    costmod_high: float = 4.0,
    temperature_low: float = 0.001,
    temperature_high: float = 1.0,
    proposal_timeout_s: float = 300.0,
    lowering_timeout_s: float = 60.0,
    search_timeout_s: float = 7200.0,
) -> dict[str, object]:
    """Run the private serial TPE/RandomGreedy proposal loop.

    The callback is the caller-owned full-plan admission and score boundary.  It
    receives only a validated complete path and the cotengra tree FLOPs; this
    function does not lower, execute, cache, or select a replacement path.
    """

    if not isinstance(evaluation_callback, Callable):
        raise TypeError("evaluation_callback must be callable")
    _require_nonempty_text(workload_id, "workload_id")
    _require_nonempty_text(cell_id, "cell_id")
    _require_nonempty_text(stage, "stage")
    _require_nonempty_text(objective_id, "objective_id")
    _require_nonempty_text(profile_id, "profile_id")
    if not isinstance(trace_context, Mapping):
        raise TypeError("trace_context must be a mapping")
    try:
        trace_context = json.loads(json.dumps(
            trace_context, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ))
    except (TypeError, ValueError) as exc:
        raise ValueError("trace_context must be JSON-safe") from exc
    if isinstance(proposals, bool) or not isinstance(proposals, int) or proposals < 1:
        raise ValueError("proposals must be a positive integer")
    if isinstance(startup_trials, bool) or not isinstance(startup_trials, int) or not 1 <= startup_trials <= proposals:
        raise ValueError("startup_trials must be between one and proposals")
    costmod_low = _require_positive_finite(costmod_low, "costmod_low")
    costmod_high = _require_positive_finite(costmod_high, "costmod_high")
    temperature_low = _require_positive_finite(temperature_low, "temperature_low")
    temperature_high = _require_positive_finite(temperature_high, "temperature_high")
    if costmod_low > costmod_high or temperature_low > temperature_high:
        raise ValueError("search parameter bounds must be ordered")
    proposal_timeout_s = _require_positive_finite(proposal_timeout_s, "proposal_timeout_s")
    lowering_timeout_s = _require_positive_finite(lowering_timeout_s, "lowering_timeout_s")
    search_timeout_s = _require_positive_finite(search_timeout_s, "search_timeout_s")

    tensor_count = _network_preflight(network)
    inputs = [tuple(int(label) for label in tensor.labels) for tensor in network.tensors]
    output = tuple(int(label) for label in network.output_labels)
    size_dict = _size_dict(network)
    _random_greedy_optimizer_class()

    sampler_seed = _seed_from_identity(
        master_seed=master_seed,
        workload_id=workload_id,
        cell_id=cell_id,
        seed_domain="optuna_tpe_sampler",
        stage=stage,
        proposal_ordinal=None,
    )
    study = _new_tpe_study(sampler_seed=sampler_seed, startup_trials=startup_trials)
    trace: list[dict[str, object]] = []
    seen_path_ids: set[str] = set()
    search_deadline = time.monotonic() + search_timeout_s

    for proposal_index in range(proposals):
        remaining = search_deadline - time.monotonic()
        if remaining <= 0:
            raise CostGuidedSearchError("search exceeded 7200-second guard", trace)
        record: dict[str, object] | None = None
        try:
            trial = study.ask()
            costmod = _bounded_parameter(
                trial.suggest_float("costmod", costmod_low, costmod_high, log=False),
                costmod_low,
                costmod_high,
                "costmod",
            )
            temperature = _bounded_parameter(
                trial.suggest_float("temperature", temperature_low, temperature_high, log=True),
                temperature_low,
                temperature_high,
                "temperature",
            )
            proposal_seed = _seed_from_identity(
                master_seed=master_seed,
                workload_id=workload_id,
                cell_id=cell_id,
                seed_domain="cotengra_random_greedy_proposal",
                stage=stage,
                proposal_ordinal=proposal_index,
            )
            record: dict[str, object] = {
                "proposal_index": proposal_index,
                "trial_number": int(trial.number),
                "params": {"costmod": costmod, "temperature": temperature},
                "sampler_seed": sampler_seed,
                "proposal_seed": proposal_seed,
                "stage": stage,
                "objective_id": objective_id,
                "profile_id": profile_id,
                "path": None,
                "path_id": None,
                "tree_flops": None,
                "score": None,
                "facts": None,
                "status": "aborted",
                "rejection_reason": None,
                "told_objective": None,
                "duplicate": False,
                "tell_order": None,
            }
            proposal_deadline = min(search_deadline, time.monotonic() + proposal_timeout_s)
            canonical_path, tree_flops = _isolated_random_greedy(
                inputs=inputs,
                output=output,
                size_dict=size_dict,
                tensor_count=tensor_count,
                costmod=costmod,
                temperature=temperature,
                proposal_seed=proposal_seed,
                deadline=proposal_deadline,
            )
            candidate_path_id = path_id(canonical_path, circuit_id=cell_id)
            record["path"] = [list(step) for step in canonical_path]
            record["path_id"] = candidate_path_id
            record["tree_flops"] = tree_flops
            record["duplicate"] = candidate_path_id in seen_path_ids
            callback_timeout = min(lowering_timeout_s, proposal_deadline - time.monotonic())
            if callback_timeout <= 0:
                raise TimeoutError("proposal exceeded 300-second guard before evaluation")
            evaluation = _evaluate_with_guard(
                evaluation_callback,
                canonical_path,
                tree_flops,
                callback_timeout,
            )
            if time.monotonic() > min(proposal_deadline, search_deadline):
                raise TimeoutError("proposal completed after its deadline")
            # A caller may reuse its scratch dictionaries on the next proposal.
            snapshot = json.loads(json.dumps(evaluation.facts, allow_nan=False))
            if evaluation.score is None:
                study.tell(trial, float("inf"))
                record["status"] = "known_infeasible"
                record["rejection_reason"] = evaluation.known_infeasible_reason
                record["facts"] = snapshot
                record["told_objective"] = "positive_infinity"
            else:
                study.tell(trial, evaluation.score)
                record["status"] = "eligible"
                record["score"] = evaluation.score
                record["facts"] = snapshot
                record["told_objective"] = evaluation.score
            record["tell_order"] = proposal_index + 1
            trace.append(record)
            seen_path_ids.add(candidate_path_id)
        except CostGuidedSearchError:
            raise
        except BaseException as exc:
            if record is not None and record.get("tell_order") is None:
                record["rejection_reason"] = f"{type(exc).__name__}:{exc}"
                trace.append(record)
            raise CostGuidedSearchError(
                f"cost-guided proposal {proposal_index} aborted: {type(exc).__name__}:{exc}",
                trace,
            ) from exc

    eligible = [item for item in trace if item["status"] == "eligible"]
    best = min(eligible, key=lambda item: (float(item["score"]), str(item["path_id"])), default=None)
    return {
        "schema": "upmem_cost_guided_search_trace_v1",
        "completed": True,
        "workload_id": workload_id,
        "cell_id": cell_id,
        "stage": stage,
        "objective_id": objective_id,
        "profile_id": profile_id,
        "trace_context": {
            "stage": stage,
            "objective_id": objective_id,
            "profile_id": profile_id,
            "caller_binding": dict(trace_context),
        },
        "master_seed": master_seed,
        "sampler_seed": sampler_seed,
        "proposal_count": proposals,
        "startup_trials": startup_trials,
        "generator": {
            "optimizer": "cotengra.RandomGreedyOptimizer",
            "max_repeats": 1,
            "accel": False,
            "parallel": False,
            "simplify": True,
        },
        "seed_derivation": "sha256_sorted_compact_utf8_json_first_four_bytes_big_endian",
        "best_path_id": best["path_id"] if best is not None else None,
        "trace": trace,
    }


def attempt_budget(development_cells: int, evaluation_cells: int, cap: int) -> dict:
    """Reserve both feedback rounds and four evaluation methods before round 0."""
    for value in (development_cells, evaluation_cells, cap):
        if type(value) is not int or value <= 0:
            raise ValueError("Cell counts and attempt cap must be positive integers")
    initial_paths = min(
        6, (cap - 24 * development_cells - 24 * evaluation_cells)
        // (4 * development_cells),
    )
    if initial_paths < 3:
        raise ValueError("Budget cannot admit the three mandatory initial roles")
    stages = {
        "initial": 4 * development_cells * initial_paths,
        "feedback_1": 12 * development_cells,
        "feedback_2": 12 * development_cells,
        "evaluation": 24 * evaluation_cells,
    }
    total = sum(stages.values())
    return {
        "development_cells": development_cells,
        "evaluation_cells": evaluation_cells,
        "initial_paths_per_cell": initial_paths,
        "stages": stages,
        "maximum_attempts": total,
        "unallocated_attempts": cap - total,
    }


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Study asset is outside the implementation root")
    return path


def load_study(path: Path = DEFAULT_STUDY, *, root: Path = ROOT) -> tuple[dict, dict, dict]:
    """Bind the unchanged manifest and byte assets, without lowering or execution."""
    study = json.loads(path.read_text(encoding="utf-8"))
    if study["study_id"] != "upmem_cost_guided_path_study_v1":
        raise ValueError("Wrong study identity")
    binding = study["workload"]
    workload_path = _contained_path(root, binding["path"])
    raw = workload_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Frozen workload hash mismatch")
    workload = json.loads(raw)
    instances = workload["instances"]
    ids = [item["instance_id"] for item in instances]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate workload instance")
    splits = Counter(item["split"] for item in instances)
    if set(splits) != {binding["development_split"], binding["evaluation_split"]}:
        raise ValueError("Unexpected workload split")
    assets = workload["provenance"]["qasm_assets"]
    if Counter(item["instance_id"] for item in assets) != Counter(ids):
        raise ValueError("QASM asset membership differs from workload")
    for asset in assets:
        data = _contained_path(root, asset["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError(f"QASM hash mismatch: {asset['instance_id']}")
    topologies = study["executor"]["topologies"]
    if topologies != [
        {"topology_id": "1dpu_t8", "dpu_count": 1, "tasklets_per_dpu": 8, "rank_count": 1},
        {"topology_id": "4dpu_t8", "dpu_count": 4, "tasklets_per_dpu": 8, "rank_count": 1},
    ]:
        raise ValueError("Study requires the two frozen one-rank topologies")
    campaign = study["campaign"]
    budget = attempt_budget(
        splits[binding["development_split"]] * len(topologies),
        splits[binding["evaluation_split"]] * len(topologies),
        campaign["authorized_attempt_cap"],
    )
    if budget["maximum_attempts"] != campaign["effective_attempt_cap"]:
        raise ValueError("Declared effective budget does not match workload")
    if budget["initial_paths_per_cell"] != campaign["initial_paths_per_cell"]:
        raise ValueError("Declared initial-path cap does not match budget")
    return study, workload, budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect",))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    study, workload, budget = load_study(args.study)
    print(json.dumps({
        "study_id": study["study_id"],
        "workload_id": workload["workload_id"],
        "executor": study["executor"]["source"],
        "budget": budget,
        "physical_admission": "not_performed",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
