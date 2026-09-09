"""Private, bounded cost-guided path study over the frozen UPMEM executor."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from importlib import metadata
import multiprocessing
import os
import queue as queue_module
from pathlib import Path
import signal
import subprocess
import threading
import time
import tempfile

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
    circuit_id: str,
    stage: str,
    objective_id: str,
    profile_id: str,
    trace_context: Mapping[str, object],
    evaluation_callback: Callable[[tuple[tuple[int, int], ...], float], GuidedEvaluation],
    record_callback: Callable[[dict[str, object]], None] | None = None,
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
    _require_nonempty_text(circuit_id, "circuit_id")
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
            ask_started = time.monotonic()
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
                "timings_s": {"adaptive_ask": time.monotonic() - ask_started},
            }
            proposal_deadline = min(search_deadline, time.monotonic() + proposal_timeout_s)
            generation_started = time.monotonic()
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
            record["timings_s"]["candidate_generation"] = time.monotonic() - generation_started
            candidate_path_id = path_id(canonical_path, circuit_id=circuit_id)
            record["path"] = [list(step) for step in canonical_path]
            record["path_id"] = candidate_path_id
            record["tree_flops"] = tree_flops
            record["duplicate"] = candidate_path_id in seen_path_ids
            callback_timeout = min(lowering_timeout_s, proposal_deadline - time.monotonic())
            if callback_timeout <= 0:
                raise TimeoutError("proposal exceeded 300-second guard before evaluation")
            evaluation_started = time.monotonic()
            evaluation = _evaluate_with_guard(
                evaluation_callback,
                canonical_path,
                tree_flops,
                callback_timeout,
            )
            record["timings_s"]["plan_evaluation"] = time.monotonic() - evaluation_started
            if time.monotonic() > min(proposal_deadline, search_deadline):
                raise TimeoutError("proposal completed after its deadline")
            # A caller may reuse its scratch dictionaries on the next proposal.
            snapshot = json.loads(json.dumps(evaluation.facts, allow_nan=False))
            tell_started = time.monotonic()
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
            record["timings_s"]["adaptive_tell"] = time.monotonic() - tell_started
            record["tell_order"] = proposal_index + 1
            trace.append(record)
            if record_callback is not None:
                record_callback(record)
            seen_path_ids.add(candidate_path_id)
        except CostGuidedSearchError:
            raise
        except BaseException as exc:
            if record is not None and record.get("tell_order") is None:
                record["rejection_reason"] = f"{type(exc).__name__}:{exc}"
                trace.append(record)
                if record_callback is not None:
                    record_callback(record)
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
        "circuit_id": circuit_id,
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
    if campaign["authorized_attempt_cap"] > 792 or campaign["feedback_rounds"] != 2:
        raise ValueError("Study exceeds the authorized attempt/round contract")
    required = {
        "search": {"cotengra_version": "0.7.5", "optuna_version": "4.5.0",
                   "sampler": "TPESampler", "generator": "fresh_RandomGreedyOptimizer",
                   "startup_trials": 16, "proposals": 128, "master_seed": 20260909,
                   "max_repeats": 1, "accel": False, "parallel": False,
                   "simplify": True, "serial": True,
                   "costmod": {"low": 0.1, "high": 4.0, "log": False},
                   "temperature": {"low": 0.001, "high": 1.0, "log": True},
                   "proposal_timeout_s": 300, "lowering_timeout_s": 60, "search_timeout_s": 7200,
                   "maximum_planned_work_units": 400, "maximum_semantic_identity_expansion_units": 1000000,
                   "host_memory_admission_bytes": 536870912},
        "score": {"model_id": "upmem_launch_cost_v1", "terms": ["H", "P", "N", "M", "W"],
                  "coefficient_denominator": 10, "grid_size": 1001,
                  "initial_integer_coefficients": [2, 2, 2, 2, 2],
                  "timing_primary": "session_open_s + steady_execution_v1.total_wall_s + session_close_s",
                  "historical_observations_allowed": False, "evaluation_observations_allowed_for_fit": False},
        "executor": {"source": "459935f586fdd16c82013838e6d27a12604c3093",
                     "schedule_policy": "static_dag_waves_v1", "request_transport": "packed_wave_v1",
                     "fuse_complex": True, "geometry_policy": "panel_only_v1",
                     "numeric_policy": "split_complex_float32_v1"},
        "campaign": {"feedback_paths_per_cell": 3, "calibration_warmup_blocks": 1,
                     "calibration_measurement_blocks": 3, "evaluation_methods": ["G", "F", "R", "U"],
                     "evaluation_warmup_blocks": 1, "evaluation_measurement_blocks": 5,
                     "retry_or_replacement": False, "unused_attempts_reallocated": False,
                     "acceptance_requires_verified_copies": 2,
                     "attempt_timeout_s": 120, "physical_stage_timeout_s": 86400},
    }
    for section, fields in required.items():
        for name, expected in fields.items():
            if study[section].get(name) != expected or type(study[section].get(name)) is not type(expected):
                raise ValueError(f"Frozen study contract mismatch: {section}.{name}")
    return study, workload, budget


def record_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def research_binding(study: dict) -> dict:
    """Require a clean source and the fully pinned research interpreter."""
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("Frozen preparation requires a clean worktree")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime_hash = hashlib.sha256((ROOT / "src/quantum_bench/upmem/runtime.py").read_bytes()).hexdigest()
    if runtime_hash != "b7168cd09f007978622346fd9954bdda54beb9ce48d870e4df8d158ed8681c02":
        raise ValueError("Frozen executor runtime source changed")
    lock = ROOT / "requirements-upmem-path-search.txt"
    versions = {}
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        actual = metadata.version(name)
        if actual != version:
            raise ValueError(f"Research dependency mismatch: {name}: {actual} != {version}")
        versions[name] = actual
    return {
        "source_sha": source,
        "executor_source": study["executor"]["source"],
        "runtime_source_hash": runtime_hash,
        "study_hash": record_hash(study),
        "workload_hash": study["workload"]["sha256"],
        "research_lock_hash": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "dependencies": versions,
        "source_hashes": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "src/quantum_bench/upmem/execution_features.py",
                "src/quantum_bench/upmem/path_heuristic.py",
                "scripts/upmem_cost_guided_path.py",
            )
        },
    }


def lower_candidate(network, path, topology: dict, study: dict) -> GuidedEvaluation:
    """Observe/admit the production plan; unexpected defects are never exclusions."""
    from quantum_bench.lowering import build_contraction_dag
    from quantum_bench.upmem.execution_features import extract_launch_cost_features
    from quantum_bench.upmem.path_heuristic import extract_conventional_features
    from quantum_bench.upmem.plan import UpmemTopology, collection_resource_admission, plan_upmem
    import upmem_path_heuristic as existing

    dag = build_contraction_dag(network, path)
    limits = study["search"]
    units = existing._estimated_work_unit_count(dag)
    if units > limits["maximum_planned_work_units"]:
        return GuidedEvaluation(None, {"work_unit_count": units}, "work_unit_bound")
    expansion = existing._semantic_identity_expansion_units(
        dag, stop_after=limits["maximum_semantic_identity_expansion_units"],
    )
    if expansion > limits["maximum_semantic_identity_expansion_units"]:
        return GuidedEvaluation(None, {"identity_expansion_units": expansion}, "identity_expansion_bound")
    executor = study["executor"]
    resources = UpmemTopology(**{key: topology[key] for key in (
        "dpu_count", "rank_count", "tasklets_per_dpu",
    )})
    plan = plan_upmem(dag, numeric_policy=executor["numeric_policy"], topology=resources,
                      schedule_policy=executor["schedule_policy"])
    try:
        existing._require_wave_execution_coverage(plan)
    except ValueError as exc:
        if not str(exc).startswith("planned_execution_resource_admission_failed:"):
            raise
        return GuidedEvaluation(None, {"work_unit_count": units}, str(exc))
    features = extract_launch_cost_features(
        dag, plan, fuse_complex=executor["fuse_complex"], geometry_policy=executor["geometry_policy"],
    )
    execution = features.pop("execution")
    memory = execution["host_buffers"]["declared_executor_memory_estimate_bytes"]
    if memory > limits["host_memory_admission_bytes"]:
        return GuidedEvaluation(None, {"declared_host_bytes": memory}, "host_memory_bound")
    admission = collection_resource_admission(plan)
    existing._wave_scaling_admission(admission, resources)
    features.update(
        physical_plan_id=execution["plan"]["physical_plan_id"],
        logical_plan_id=execution["plan"]["logical_plan_id"],
        resource_admission=admission,
        declared_host_bytes=memory,
        work_unit_count=units,
        conventional=extract_conventional_features(dag).as_mapping(),
    )
    return GuidedEvaluation(0.0, features)


def prepare_normalization(study: dict, workload: dict) -> dict:
    """Development greedy plans only; no tensor execution or test-cell scaling."""
    import upmem_path_heuristic as existing
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, development_greedy_scales

    cells, references = {}, {}
    for instance in sorted(workload["instances"], key=lambda item: item["instance_id"]):
        if instance["split"] != study["workload"]["development_split"]:
            continue
        circuit = existing._circuit_from_definition(instance["circuit"])
        network, _ = existing.lower_tensor_network(existing.make_simulation_job(circuit))
        path, _ = existing.plan_opt_einsum(network, optimize="greedy")
        for topology in study["executor"]["topologies"]:
            cell_id = f"{instance['instance_id']}/{topology['topology_id']}"
            evaluated = _evaluate_with_guard(
                lambda p, _: lower_candidate(network, p, topology, study), path, 0.0,
                study["search"]["lowering_timeout_s"],
            )
            if evaluated.score is None:
                raise ValueError(f"Required greedy cell is infeasible: {cell_id}: {evaluated.known_infeasible_reason}")
            facts = LaunchCostFacts.from_mapping(evaluated.facts)
            cells[cell_id] = ("training", facts)
            references[cell_id] = {
                "circuit_id": instance["instance_id"], "family": instance["family"],
                "topology_id": topology["topology_id"], "split": "training",
                "path_id": path_id(path, circuit_id=instance["instance_id"]),
                "path": path, "facts": evaluated.facts,
                "tree_flops": conventional_tree_flops(network, path),
            }
    expected = sum(i["split"] == study["workload"]["development_split"] for i in workload["instances"])
    if len(cells) != expected * len(study["executor"]["topologies"]):
        raise ValueError("Development normalization membership is incomplete")
    return {
        "model_id": "upmem_launch_cost_v1", "study_hash": record_hash(study),
        "workload_hash": study["workload"]["sha256"],
        "scales": development_greedy_scales(cells).as_mapping(), "greedy_cells": references,
    }


def conventional_tree_flops(network, path) -> float:
    """Put greedy on the identical cotengra FLOP scale used in search trials."""
    from cotengra import ContractionTree

    inputs = [tuple(tensor.labels) for tensor in network.tensors]
    tree = ContractionTree.from_path(inputs, tuple(network.output_labels), _size_dict(network), path=path)
    return _require_nonnegative_finite(tree.total_flops(), "greedy tree FLOPs")


def choose_round_paths(candidates: Mapping[str, dict], greedy_id: str, *, scales, weights,
                       initial_cap: int | None = None, previously_measured=frozenset()) -> dict:
    """Fixed initial G/F/R+diversity or feedback new-best+diversity+G selection."""
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, upmem_launch_cost_v1

    if greedy_id not in candidates:
        raise ValueError("A round requires its eligible greedy control")
    facts = {identifier: LaunchCostFacts.from_mapping(row["facts"]) for identifier, row in candidates.items()}
    scores = {identifier: upmem_launch_cost_v1(fact, weights, scales) for identifier, fact in facts.items()}
    vectors = {identifier: tuple(x / scale for x, scale in zip(fact.as_tuple(), scales.as_tuple(), strict=True))
               for identifier, fact in facts.items()}
    selected = {greedy_id}
    roles = {"G": greedy_id}
    if initial_cap is not None:
        if type(initial_cap) is not int or not 3 <= initial_cap <= 6:
            raise ValueError("Initial cap must admit G/F/R and stay within six paths")
        roles["F"] = min(candidates, key=lambda p: (candidates[p]["tree_flops"], p))
        roles["R"] = min(candidates, key=lambda p: (scores[p], p))
        selected.update(roles.values())
        remaining = set(candidates) - selected
        maximum = initial_cap
    else:
        remaining = set(candidates) - set(previously_measured) - {greedy_id}
        if not remaining:
            return {"roles": {}, "path_ids": [], "reason": "no_new_eligible_candidate"}
        best = min(remaining, key=lambda p: (scores[p], p))
        roles["new_best"] = best
        selected.add(best)
        remaining.remove(best)
        maximum = 3
    while remaining and len(selected) < maximum:
        distance = {
            p: min(sum(abs(x-y) for x, y in zip(vectors[p], vectors[a], strict=True)) for a in sorted(selected))
            for p in sorted(remaining)
        }
        chosen = min(remaining, key=lambda p: (-distance[p], p))
        roles[f"diverse_{len(selected)}"] = chosen
        selected.add(chosen)
        remaining.remove(chosen)
    return {"roles": roles, "path_ids": sorted(selected), "scores": scores}


def choose_evaluation_paths(greedy: dict, flop_trace: dict, upmem_trace: dict, *, profile: dict, normalization: dict) -> dict:
    """Four method labels with U isolated from the F/R candidate pool."""
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, LaunchCostScales, upmem_launch_cost_v1

    if profile.get("model_id") != "upmem_launch_cost_v1" or profile.get("normalization_hash") != record_hash(normalization):
        raise ValueError("Evaluation profile/normalization binding mismatch")
    weights = tuple(profile["integer_weights"])
    scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
    for trace in (flop_trace, upmem_trace):
        if trace.get("completed") is not True or trace.get("proposal_count") != 128 or len(trace["trace"]) != 128:
            raise ValueError("Evaluation requires two complete 128-proposal traces")
        context = trace.get("trace_context", {})
        if "caller_binding" not in context or any(context.get(key) != trace.get(key) for key in (
            "stage", "objective_id", "profile_id",
        )):
            raise ValueError("Evaluation trace binding mismatch: trace_context")
        if trace.get("profile_id") != record_hash(profile) or context["caller_binding"].get("normalization_hash") != record_hash(normalization):
            raise ValueError("Evaluation trace binding mismatch: supplied profile/normalization")
    for key in ("workload_id", "cell_id", "circuit_id", "stage", "master_seed", "sampler_seed", "profile_id"):
        if key not in flop_trace or flop_trace[key] != upmem_trace.get(key):
            raise ValueError(f"Evaluation trace binding mismatch: {key}")
    if flop_trace["trace_context"].get("caller_binding") != upmem_trace["trace_context"].get("caller_binding"):
        raise ValueError("Evaluation trace binding mismatch: caller_binding")
    if flop_trace.get("objective_id") != "cotengra_tree_flops_v1" or upmem_trace.get("objective_id") != "upmem_launch_cost_v1":
        raise ValueError("Evaluation must distinguish FLOP-guided and UPMEM-guided traces")
    for trace in (flop_trace, upmem_trace):
        for row in trace["trace"]:
            if row["status"] == "eligible":
                expected_score = row["tree_flops"] if trace is flop_trace else upmem_launch_cost_v1(
                    LaunchCostFacts.from_mapping(row["facts"]), weights, scales,
                )
                if row.get("score") != expected_score or row.get("told_objective") != expected_score:
                    raise ValueError("Evaluation trace score differs from its frozen objective")
    f_pool, u_pool = eligible_candidate_pool(flop_trace, greedy), eligible_candidate_pool(upmem_trace, greedy)
    def score(row):
        return upmem_launch_cost_v1(LaunchCostFacts.from_mapping(row["facts"]), weights, scales)
    return {
        "G": greedy["path_id"],
        "F": min(f_pool, key=lambda p: (f_pool[p]["tree_flops"], p)),
        "R": min(f_pool, key=lambda p: (score(f_pool[p]), p)),
        "U": min(u_pool, key=lambda p: (score(u_pool[p]), p)),
    }


def eligible_candidate_pool(trace: dict, greedy: dict) -> dict:
    """Deduplicate facts, not trial identities or objective-dependent scores."""
    result = {}
    for row in [*trace["trace"], greedy]:
        if row.get("status", "eligible") != "eligible":
            continue
        canonical = {key: row[key] for key in ("path_id", "tree_flops", "facts")}
        if "path" in row:
            canonical["path"] = [sorted(pair) for pair in row["path"]]
        identifier = row["path_id"]
        if identifier in result and record_hash(canonical) != record_hash(result[identifier]):
            raise ValueError(f"Conflicting deterministic facts for path {identifier}")
        result[identifier] = canonical
    return result


def validate_search_trace(trace: dict, study: dict, *, workload_id: str, cell_id: str,
                          circuit_id: str, stage: str, objective_id: str, profile_id: str,
                          context: dict) -> None:
    """Validate the retained engine trace against its already frozen invocation."""
    from quantum_bench.planning import normalize_frozen_path
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts

    settings = study["search"]
    sampler_seed = _seed_from_identity(
        master_seed=settings["master_seed"], workload_id=workload_id, cell_id=cell_id,
        seed_domain="optuna_tpe_sampler", stage=stage, proposal_ordinal=None,
    )
    expected = {"schema": "upmem_cost_guided_search_trace_v1", "completed": True,
                "workload_id": workload_id, "cell_id": cell_id, "circuit_id": circuit_id,
                "stage": stage, "objective_id": objective_id, "profile_id": profile_id,
                "proposal_count": settings["proposals"], "startup_trials": settings["startup_trials"],
                "master_seed": settings["master_seed"], "sampler_seed": sampler_seed,
                "seed_derivation": settings["seed_derivation"],
                "generator": {"optimizer": "cotengra.RandomGreedyOptimizer", "max_repeats": 1,
                              "accel": False, "parallel": False, "simplify": True},
                "trace_context": {"stage": stage, "objective_id": objective_id,
                                  "profile_id": profile_id, "caller_binding": context}}
    if any(record_hash(trace.get(k)) != record_hash(v) for k, v in expected.items()):
        raise ValueError("Incomplete or mismatched search invocation provenance")
    if len(trace["trace"]) != settings["proposals"]:
        raise ValueError("Incomplete search proposal set")
    seen = {}
    for index, row in enumerate(trace["trace"]):
        ordinal_fields = {"proposal_index": index, "trial_number": index, "tell_order": index + 1,
                          "stage": stage, "objective_id": objective_id, "profile_id": profile_id,
                          "sampler_seed": sampler_seed, "proposal_seed": _seed_from_identity(
                              master_seed=settings["master_seed"], workload_id=workload_id, cell_id=cell_id,
                              seed_domain="cotengra_random_greedy_proposal", stage=stage, proposal_ordinal=index)}
        if any(record_hash(row.get(k)) != record_hash(v) for k, v in ordinal_fields.items()):
            raise ValueError("Search row provenance or proposal/tell sequence mismatch")
        if set(row["params"]) != {"costmod", "temperature"}:
            raise ValueError("Unexpected search proposal parameters")
        for key in ("costmod", "temperature"):
            _bounded_parameter(row["params"][key], settings[key]["low"], settings[key]["high"], key)
        path = normalize_frozen_path(row["path"])
        if row["path_id"] != path_id(path, circuit_id=circuit_id):
            raise ValueError("Search canonical path identity mismatch")
        if row.get("duplicate") is not (row["path_id"] in seen):
            raise ValueError("Search duplicate flag mismatch")
        _require_nonnegative_finite(row["tree_flops"], "tree_flops")
        if row["status"] == "eligible":
            LaunchCostFacts.from_mapping(row["facts"])
            score = _require_nonnegative_finite(row["score"], "score")
            if row["told_objective"] != score or row.get("rejection_reason") is not None:
                raise ValueError("Eligible proposal tell/rejection mismatch")
            if objective_id == "cotengra_tree_flops_v1" and score != row["tree_flops"]:
                raise ValueError("Search did not use the declared FLOP objective")
        elif row["status"] == "known_infeasible":
            if row["score"] is not None or row["told_objective"] != "positive_infinity":
                raise ValueError("Infeasible proposal must tell positive infinity without a finite score")
            _require_nonempty_text(row["rejection_reason"], "rejection_reason")
        else:
            raise ValueError("Aborted proposal cannot enter a frozen round")
        deterministic = {k: row[k] for k in ("facts", "status", "tree_flops", "rejection_reason", "score")}
        if row["path_id"] in seen and record_hash(seen[row["path_id"]]) != record_hash(deterministic):
            raise ValueError("Conflicting duplicate proposal facts or score")
        seen[row["path_id"]] = deterministic
    eligible = [row for row in trace["trace"] if row["status"] == "eligible"]
    best = min(eligible, key=lambda row: (row["score"], row["path_id"]), default=None)
    if trace.get("best_path_id") != (best["path_id"] if best is not None else None):
        raise ValueError("Search best-path identity mismatch")


def _write_new_json(path: Path, value: object) -> None:
    """Exclusive durable write: interrupted artifacts are retained, never reused."""
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initialize_preparation(directory: Path, study: dict, workload: dict) -> dict:
    """Freeze software binding and development-only scales, without a search."""
    binding = research_binding(study)
    directory.mkdir(parents=True, exist_ok=False)
    _write_new_json(directory / "binding.json", binding)
    try:
        normalization = prepare_normalization(study, workload)
        normalization["binding_hash"] = record_hash(binding)
        _write_new_json(directory / "normalization.json", normalization)
        profile = {
            "model_id": study["score"]["model_id"], "stage": "initial",
            "integer_weights": study["score"]["initial_integer_coefficients"],
            "normalization_hash": record_hash(normalization), "binding_hash": record_hash(binding),
            "observation_rounds": [],
        }
        _write_new_json(directory / "initial_profile.json", profile)
        return {"binding_hash": record_hash(binding), "normalization_hash": record_hash(normalization),
                "profile_hash": record_hash(profile)}
    except BaseException as exc:
        _write_new_json(directory / "preparation_failed.json", {"reason": f"{type(exc).__name__}:{exc}"})
        raise


def _load_preparation(directory: Path, study: dict, workload: dict) -> tuple[dict, dict, dict]:
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, development_greedy_scales

    if (directory / "preparation_failed.json").exists():
        raise ValueError("Failed preparation cannot be resumed")
    binding, normalization, profile = (
        json.loads((directory / name).read_text())
        for name in ("binding.json", "normalization.json", "initial_profile.json")
    )
    if research_binding(study) != binding:
        raise ValueError("Preparation source, environment or study binding changed")
    if normalization.get("binding_hash") != record_hash(binding) or profile != {
        "model_id": study["score"]["model_id"], "stage": "initial",
        "integer_weights": study["score"]["initial_integer_coefficients"],
        "normalization_hash": record_hash(normalization), "binding_hash": record_hash(binding),
        "observation_rounds": [],
    }:
        raise ValueError("Preparation normalization/profile binding mismatch")
    expected = {f"{i['instance_id']}/{t['topology_id']}": (i, t)
                for i in workload["instances"] if i["split"] == study["workload"]["development_split"]
                for t in study["executor"]["topologies"]}
    if set(normalization["greedy_cells"]) != set(expected):
        raise ValueError("Normalization must contain the exact development membership")
    references = {}
    for cell_id, row in normalization["greedy_cells"].items():
        instance, topology = expected[cell_id]
        if (row["circuit_id"], row["family"], row["topology_id"], row["split"]) != (
            instance["instance_id"], instance["family"], topology["topology_id"], "training",
        ) or row["path_id"] != path_id(row["path"], circuit_id=instance["instance_id"]):
            raise ValueError("Normalization greedy cell identity mismatch")
        references[cell_id] = ("training", LaunchCostFacts.from_mapping(row["facts"]))
    if normalization["scales"] != development_greedy_scales(references).as_mapping():
        raise ValueError("Normalization scales differ from frozen development greedy facts")
    return binding, normalization, profile


def _cell_inputs(cell_id: str, study: dict, workload: dict):
    import upmem_path_heuristic as existing

    cells = {f"{instance['instance_id']}/{topology['topology_id']}": (instance, topology)
             for instance in workload["instances"] for topology in study["executor"]["topologies"]}
    if cell_id not in cells:
        raise ValueError("Unknown workload cell")
    instance, topology = cells[cell_id]
    circuit = existing._circuit_from_definition(instance["circuit"])
    network, _ = existing.lower_tensor_network(existing.make_simulation_job(circuit))
    return instance, topology, network


def initial_cell_search(directory: Path, cell_id: str, study: dict, workload: dict) -> dict:
    """One durable initial trace; interrupted cells never refill."""
    binding, normalization, profile = _load_preparation(directory, study, workload)
    if cell_id not in normalization["greedy_cells"]:
        raise ValueError("Initial search is restricted to frozen development cells")
    if (directory / "initial_round.json").exists():
        raise ValueError("Initial round is already frozen")
    instance, topology, network = _cell_inputs(cell_id, study, workload)
    if instance["split"] != study["workload"]["development_split"]:
        raise ValueError("Evaluation data cannot enter initial development search")
    return _run_cell_search(directory, cell_id, study, workload, binding, normalization, profile,
                            instance, topology, network, stage="initial", objective_id="cotengra_tree_flops_v1")


def _run_cell_search(directory, cell_id, study, workload, binding, normalization, profile,
                     instance, topology, network, *, stage, objective_id):
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, LaunchCostScales, upmem_launch_cost_v1

    scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
    cell_directory = directory / stage / record_hash(cell_id)
    if stage == "evaluation":
        cell_directory /= objective_id
    cell_directory.mkdir(parents=True, exist_ok=False)
    context = {"binding_hash": record_hash(binding), "normalization_hash": record_hash(normalization)}
    _write_new_json(cell_directory / "started.json", {"cell_id": cell_id, **context})
    started = time.monotonic()
    def evaluate(path, flops):
        result = lower_candidate(network, path, topology, study)
        if result.score is None:
            return result
        score = flops if objective_id == "cotengra_tree_flops_v1" else upmem_launch_cost_v1(
            LaunchCostFacts.from_mapping(result.facts), tuple(profile["integer_weights"]), scales,
        )
        return GuidedEvaluation(score, result.facts)
    try:
        with (cell_directory / "search_trace.jsonl").open("x", encoding="utf-8") as output:
            def retain(record):
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            limits = study["search"]
            trace = run_cost_guided_search(
                network, workload_id=workload["workload_id"], cell_id=cell_id,
                circuit_id=instance["instance_id"], stage=stage,
                objective_id=objective_id, profile_id=record_hash(profile),
                trace_context=context, evaluation_callback=evaluate, record_callback=retain,
                master_seed=limits["master_seed"], proposals=limits["proposals"],
                startup_trials=limits["startup_trials"], proposal_timeout_s=limits["proposal_timeout_s"],
                lowering_timeout_s=limits["lowering_timeout_s"], search_timeout_s=limits["search_timeout_s"],
            )
        trace["search_wall_s"] = time.monotonic() - started
        _write_new_json(cell_directory / "completed.json", trace)
        return {"cell_id": cell_id, "trace_hash": record_hash(trace), "proposal_count": trace["proposal_count"]}
    except BaseException as exc:
        _write_new_json(cell_directory / "failed.json", {"reason": f"{type(exc).__name__}:{exc}",
                                                       "search_wall_s": time.monotonic() - started})
        raise


def freeze_initial_round(directory: Path, study: dict, workload: dict, budget: dict) -> dict:
    """All development traces precede the immutable physical candidate manifest."""
    from quantum_bench.upmem.path_heuristic import LaunchCostScales

    binding, normalization, profile = _load_preparation(directory, study, workload)
    scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
    cells = {}
    for cell_id, greedy in sorted(normalization["greedy_cells"].items()):
        folder = directory / "initial" / record_hash(cell_id)
        if (folder / "failed.json").exists():
            raise ValueError(f"Failed cell cannot enter a frozen round: {cell_id}")
        trace = json.loads((folder / "completed.json").read_text())
        context = {
            "binding_hash": record_hash(binding), "normalization_hash": record_hash(normalization),
        }
        validate_search_trace(trace, study, workload_id=workload["workload_id"], cell_id=cell_id,
                              circuit_id=greedy["circuit_id"], stage="initial",
                              objective_id="cotengra_tree_flops_v1", profile_id=record_hash(profile), context=context)
        persisted = [json.loads(line) for line in (folder / "search_trace.jsonl").read_text().splitlines()]
        if persisted != trace["trace"]:
            raise ValueError("Incremental trace differs from completed trace")
        candidates = eligible_candidate_pool(trace, greedy)
        selection = choose_round_paths(candidates, greedy["path_id"], scales=scales,
                                       weights=tuple(profile["integer_weights"]),
                                       initial_cap=budget["initial_paths_per_cell"])
        cells[cell_id] = {"selection": selection, "trace_hash": record_hash(trace),
                          "candidates": {p: candidates[p] for p in selection["path_ids"]},
                          "circuit_id": greedy["circuit_id"], "family": greedy["family"],
                          "topology_id": greedy["topology_id"], "split": "training"}
    attempts = 4 * sum(len(cell["candidates"]) for cell in cells.values())
    if len(cells) != budget["development_cells"] or attempts > budget["stages"]["initial"]:
        raise ValueError("Initial membership or attempt budget mismatch")
    manifest = {"study_id": study["study_id"], "stage": "initial", "binding_hash": record_hash(binding),
                "profile_hash": record_hash(profile), "normalization_hash": record_hash(normalization),
                "cells": cells, "expected_attempts": attempts, "warmup_blocks": [0],
                "measurement_blocks": [1, 2, 3], "physical_admission": "not_performed"}
    _write_new_json(directory / "initial_round.json", manifest)
    return manifest


STAGES = ("initial", "feedback_1", "feedback_2", "evaluation")


def _stage_prefix(stage: str) -> tuple[str, ...]:
    if stage not in STAGES:
        raise ValueError("Unknown study stage; extra adaptation rounds are forbidden")
    return STAGES[:STAGES.index(stage)]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_round_archives(archives: list[Path], manifest: dict, workload: dict, binding: dict, study: dict) -> dict:
    """Reopen two durable copies and derive observations through canonical checks."""
    from freeze_packed_operation_transport import _safe_extract
    from qualify_quantized_upmem_execution import verify_checksums
    from upmem_cost_guided_evidence import extract_round_observations

    if len(archives) != 2:
        raise ValueError("Stage acceptance requires two archive copies")
    paths = [Path(path).resolve(strict=True) for path in archives]
    if paths[0].samefile(paths[1]):
        raise ValueError("Two paths to the same archive are not independent copies")
    if any(path.is_relative_to(root) for path in paths for root in (Path("/tmp"), Path("/dev/shm"), Path("/run"))):
        raise ValueError("Accepted archive copies must be in durable storage")
    copies, extracted = [], []
    for archive in paths:
        digest = _file_digest(archive)
        checksum = Path(str(archive) + ".sha256").read_text().strip().split()
        if checksum != [digest, archive.name]:
            raise ValueError("Outer archive checksum or name mismatch")
        with tempfile.TemporaryDirectory(prefix="cost-guided-verify-") as temporary:
            root = _safe_extract(archive, Path(temporary))
            entries = (root / "SHA256SUMS").read_text().splitlines()
            names = [line.partition("  ")[2] for line in entries]
            if len(set(names)) != len(names) or any(not name or Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise ValueError("Unsafe or duplicate stage checksum entries")
            verify_checksums(root)
            extracted.append(extract_round_observations(root / "raw", manifest, workload, binding, study))
        copies.append({"path": str(archive), "sha256": digest})
    if copies[0]["sha256"] != copies[1]["sha256"] or extracted[0]["rows"] != extracted[1]["rows"]:
        raise ValueError("Verified archive copies or observations differ")
    return {**extracted[0], "archives": copies}


def accept_round(directory: Path, stage: str, archives: list[Path], study: dict, workload: dict) -> dict:
    """Acceptance is evidence verification, never an asserted success flag."""
    binding, normalization, initial_profile = _load_preparation(directory, study, workload)
    predecessors = _stage_prefix(stage)
    for previous in predecessors:
        _accepted_round(directory, previous, study, workload)
    manifest = _read_json(directory / f"{stage}_round.json")
    profile = (initial_profile if stage == "initial" else
               _pretest_profile(directory, study, workload) if stage == "evaluation" else
               _fitted_profile(directory, predecessors[-1], study, workload))
    if (manifest.get("stage") != stage or manifest.get("study_id") != study["study_id"]
            or manifest["binding_hash"] != record_hash(binding) or manifest["normalization_hash"] != record_hash(normalization)
            or manifest["profile_hash"] != record_hash(profile)):
        raise ValueError("Round binding changed")
    target = directory / f"{stage}.accepted.json"
    if target.exists() or (directory / f"{stage}.failed.json").exists():
        raise ValueError("Accepted or failed rounds cannot be replaced")
    if not manifest["cells"]:
        if stage not in ("feedback_1", "feedback_2") or manifest["expected_attempts"] != 0 or archives:
            raise ValueError("Only a declared empty feedback round can skip physical execution")
        _verify_empty_feedback(directory, stage, manifest, study, workload, binding, normalization, profile)
        result = {"rows": [], "archives": [], "physical_stage_elapsed_s": 0.0}
    else:
        result = verify_round_archives(archives, manifest, workload, binding, study)
    elapsed = _require_nonnegative_finite(result["physical_stage_elapsed_s"], "physical stage elapsed")
    prior_elapsed = sum(_read_json(directory / f"{p}.accepted.json")["physical_stage_elapsed_s"] for p in predecessors)
    if prior_elapsed + elapsed > study["campaign"]["physical_stage_timeout_s"]:
        raise ValueError("Cumulative physical-stage time budget exceeded")
    record = {"stage": stage, "round_manifest_hash": record_hash(manifest),
              "binding_hash": record_hash(binding), "rows": result["rows"], "archives": result["archives"],
              "physical_stage_elapsed_s": elapsed}
    _write_new_json(target, record)
    return record


def _accepted_round(directory: Path, stage: str, study: dict, workload: dict) -> tuple[dict, dict]:
    binding = _read_json(directory / "binding.json")
    manifest = _read_json(directory / f"{stage}_round.json")
    accepted = _read_json(directory / f"{stage}.accepted.json")
    if (directory / f"{stage}.failed.json").exists() or accepted["stage"] != stage or accepted["round_manifest_hash"] != record_hash(manifest) or accepted["binding_hash"] != record_hash(binding):
        raise ValueError("Accepted round identity changed or stage failed")
    if manifest["cells"]:
        result = verify_round_archives([Path(item["path"]) for item in accepted["archives"]], manifest, workload, binding, study)
        if (result["rows"] != accepted["rows"] or result["archives"] != accepted["archives"]
                or result["physical_stage_elapsed_s"] != accepted["physical_stage_elapsed_s"]):
            raise ValueError("Accepted observations no longer match raw evidence")
    elif stage not in ("feedback_1", "feedback_2") or accepted["rows"] or accepted["archives"] or manifest["expected_attempts"] != 0:
        raise ValueError("Invalid empty accepted round")
    return manifest, accepted


def _verify_empty_feedback(directory, stage, manifest, study, workload, binding, normalization, profile):
    if set(manifest.get("skipped_cells", {})) != set(normalization["greedy_cells"]):
        raise ValueError("Empty feedback must account for every development cell")
    measured = {cell_id: set() for cell_id in normalization["greedy_cells"]}
    for previous in _stage_prefix(stage):
        prior, _ = _accepted_round(directory, previous, study, workload)
        for cell_id, cell in prior["cells"].items():
            measured[cell_id].update(cell["candidates"])
    for cell_id, greedy in normalization["greedy_cells"].items():
        trace = _completed_cell_trace(directory, stage, cell_id, greedy["circuit_id"], study, workload,
                                      binding, normalization, profile, "upmem_launch_cost_v1")
        skipped = manifest["skipped_cells"][cell_id]
        if skipped != {"reason": "no_new_eligible_candidate", "trace_hash": record_hash(trace)}:
            raise ValueError("Empty feedback skip is not trace-bound")
        if set(eligible_candidate_pool(trace, greedy)) - measured[cell_id]:
            raise ValueError("Empty feedback omitted a new eligible candidate")


def fit_accepted_rounds(directory: Path, through_stage: str, study: dict, workload: dict) -> dict:
    """Refit only newly verified development evidence; evaluation is never input."""
    from quantum_bench.upmem.path_heuristic import LaunchCostFacts, LaunchCostScales, fit_launch_cost_weights

    if through_stage not in STAGES[:3] or (directory / "pretest_profile.json").exists():
        raise ValueError("Fitting is forbidden after pretest freeze or for evaluation")
    binding, normalization, _ = _load_preparation(directory, study, workload)
    cells = {cell_id: {"family": row["family"], "split": "training", "greedy_path_id": row["path_id"],
                       "path_facts": {row["path_id"]: LaunchCostFacts.from_mapping(row["facts"])}}
             for cell_id, row in normalization["greedy_cells"].items()}
    rounds, rows, accepted_hashes = [], [], []
    identity = {"source_sha": binding["source_sha"], "execution_source": study["executor"]["source"],
                "policy_id": study["executor"]["numeric_policy"], "timing_scope": "steady_execution_v1"}
    for stage in (*_stage_prefix(through_stage), through_stage):
        manifest, accepted = _accepted_round(directory, stage, study, workload)
        accepted_hashes.append(record_hash(accepted))
        if not manifest["cells"]:
            continue
        for cell_id, cell in manifest["cells"].items():
            if cell_id not in cells or cell["split"] != "training":
                raise ValueError("Undeclared development or evaluation cell in fit")
            for path, candidate in cell["candidates"].items():
                fact = LaunchCostFacts.from_mapping(candidate["facts"])
                prior = cells[cell_id]["path_facts"].setdefault(path, fact)
                if prior != fact:
                    raise ValueError("Path facts changed between accepted rounds")
        rounds.append({"round_id": stage, "accepted": True, "identity": accepted["rows"][0]["identity"],
                       "cell_paths": {cell_id: list(cell["candidates"]) for cell_id, cell in manifest["cells"].items()},
                       "warmup_blocks": manifest["warmup_blocks"], "measurement_blocks": manifest["measurement_blocks"]})
        rows.extend(accepted["rows"])
    scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
    fitted, table = fit_launch_cost_weights(cells, rounds, rows, scales=scales, expected_identity=identity)
    profile = {"model_id": fitted["model_id"], "integer_weights": list(fitted["integer_weights"]),
               "through_stage": through_stage, "binding_hash": record_hash(binding),
               "normalization_hash": record_hash(normalization), "accepted_round_hashes": accepted_hashes,
               "fit_hash": record_hash(fitted), "grid_hash": record_hash(table)}
    target = directory / f"{through_stage}_fit"
    target.mkdir(exist_ok=False)
    _write_new_json(target / "fit.json", fitted)
    _write_new_json(target / "grid.json", table)
    _write_new_json(target / "profile.json", profile)
    return profile


def _fitted_profile(directory: Path, through_stage: str, study: dict, workload: dict) -> dict:
    binding, normalization, _ = _load_preparation(directory, study, workload)
    folder = directory / f"{through_stage}_fit"
    profile, fitted, grid = (_read_json(folder / name) for name in ("profile.json", "fit.json", "grid.json"))
    accepted_hashes = [record_hash(_accepted_round(directory, p, study, workload)[1])
                       for p in (*_stage_prefix(through_stage), through_stage)]
    if (profile["binding_hash"] != record_hash(binding) or profile["normalization_hash"] != record_hash(normalization)
            or profile["through_stage"] != through_stage or profile["accepted_round_hashes"] != accepted_hashes
            or profile["fit_hash"] != record_hash(fitted) or profile["grid_hash"] != record_hash(grid)
            or profile["integer_weights"] != fitted["integer_weights"] or len(grid) != 1001):
        raise ValueError("Fitted profile or accepted evidence binding changed")
    return profile


def feedback_cell_search(directory: Path, stage: str, cell_id: str, study: dict, workload: dict) -> dict:
    if stage not in ("feedback_1", "feedback_2") or (directory / "pretest_profile.json").exists():
        raise ValueError("No additional feedback or adaptation after pretest freeze")
    binding, normalization, _ = _load_preparation(directory, study, workload)
    if cell_id not in normalization["greedy_cells"] or (directory / f"{stage}_round.json").exists():
        raise ValueError("Feedback requires a development cell and an unfrozen round")
    profile = _fitted_profile(directory, _stage_prefix(stage)[-1], study, workload)
    instance, topology, network = _cell_inputs(cell_id, study, workload)
    return _run_cell_search(directory, cell_id, study, workload, binding, normalization, profile,
                            instance, topology, network, stage=stage, objective_id="upmem_launch_cost_v1")


def _completed_cell_trace(directory, stage, cell_id, circuit_id, study, workload, binding,
                          normalization, profile, objective_id):
    folder = directory / stage / record_hash(cell_id)
    if stage == "evaluation":
        folder /= objective_id
    if (folder / "failed.json").exists():
        raise ValueError("Failed search cannot enter a candidate round")
    trace = _read_json(folder / "completed.json")
    context = {"binding_hash": record_hash(binding), "normalization_hash": record_hash(normalization)}
    validate_search_trace(trace, study, workload_id=workload["workload_id"], cell_id=cell_id,
                          circuit_id=circuit_id, stage=stage, objective_id=objective_id,
                          profile_id=record_hash(profile), context=context)
    persisted = [json.loads(line) for line in (folder / "search_trace.jsonl").read_text().splitlines()]
    if persisted != trace["trace"]:
        raise ValueError("Incremental/completed trace mismatch")
    if objective_id == "upmem_launch_cost_v1":
        from quantum_bench.upmem.path_heuristic import LaunchCostFacts, LaunchCostScales, upmem_launch_cost_v1
        scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
        for row in persisted:
            if row["status"] == "eligible" and row["score"] != upmem_launch_cost_v1(
                LaunchCostFacts.from_mapping(row["facts"]), tuple(profile["integer_weights"]), scales,
            ):
                raise ValueError("UPMEM trace score differs from its frozen profile")
    return trace


def freeze_feedback_round(directory: Path, stage: str, study: dict, workload: dict, budget: dict) -> dict:
    from quantum_bench.upmem.path_heuristic import LaunchCostScales

    if stage not in ("feedback_1", "feedback_2") or (directory / "pretest_profile.json").exists():
        raise ValueError("Feedback freeze is limited to two rounds before pretest")
    binding, normalization, _ = _load_preparation(directory, study, workload)
    prior_stages = _stage_prefix(stage)
    profile = _fitted_profile(directory, prior_stages[-1], study, workload)
    measured = {cell: set() for cell in normalization["greedy_cells"]}
    for prior in prior_stages:
        manifest, _ = _accepted_round(directory, prior, study, workload)
        for cell, record in manifest["cells"].items():
            measured[cell].update(record["candidates"])
    scales = LaunchCostScales(**{k.lower(): v for k, v in normalization["scales"].items()})
    cells, skipped = {}, {}
    for cell_id, greedy in sorted(normalization["greedy_cells"].items()):
        trace = _completed_cell_trace(directory, stage, cell_id, greedy["circuit_id"], study, workload,
                                      binding, normalization, profile, "upmem_launch_cost_v1")
        candidates = eligible_candidate_pool(trace, greedy)
        selection = choose_round_paths(candidates, greedy["path_id"], scales=scales,
                                       weights=tuple(profile["integer_weights"]), previously_measured=measured[cell_id])
        if not selection["path_ids"]:
            skipped[cell_id] = {"reason": selection["reason"], "trace_hash": record_hash(trace)}
            continue
        cells[cell_id] = {"selection": selection, "trace_hash": record_hash(trace),
                          "candidates": {p: candidates[p] for p in selection["path_ids"]},
                          "circuit_id": greedy["circuit_id"], "family": greedy["family"],
                          "topology_id": greedy["topology_id"], "split": "training"}
    attempts = 4 * sum(len(cell["candidates"]) for cell in cells.values())
    if len(cells) + len(skipped) != budget["development_cells"] or attempts > budget["stages"][stage]:
        raise ValueError("Feedback membership or budget mismatch")
    manifest = {"study_id": study["study_id"], "stage": stage, "binding_hash": record_hash(binding),
                "normalization_hash": record_hash(normalization), "profile_hash": record_hash(profile),
                "prior_rounds": profile["accepted_round_hashes"], "cells": cells, "skipped_cells": skipped,
                "expected_attempts": attempts, "warmup_blocks": [0], "measurement_blocks": [1, 2, 3],
                "physical_admission": "not_performed"}
    _write_new_json(directory / f"{stage}_round.json", manifest)
    return manifest


def freeze_pretest(directory: Path, study: dict, workload: dict) -> dict:
    profile = _fitted_profile(directory, "feedback_2", study, workload)
    membership = sorted(f"{i['instance_id']}/{t['topology_id']}"
                        for i in workload["instances"] if i["split"] == study["workload"]["evaluation_split"]
                        for t in study["executor"]["topologies"])
    frozen = {**profile, "stage": "pretest", "evaluation_cells": membership,
              "search_settings_hash": record_hash(study["search"]), "workload_hash": study["workload"]["sha256"]}
    _write_new_json(directory / "pretest_profile.json", frozen)
    return frozen


def _pretest_profile(directory: Path, study: dict, workload: dict) -> dict:
    frozen = _read_json(directory / "pretest_profile.json")
    expected = _fitted_profile(directory, "feedback_2", study, workload)
    membership = sorted(f"{i['instance_id']}/{t['topology_id']}"
                        for i in workload["instances"] if i["split"] == study["workload"]["evaluation_split"]
                        for t in study["executor"]["topologies"])
    if frozen != {**expected, "stage": "pretest", "evaluation_cells": membership,
                  "search_settings_hash": record_hash(study["search"]), "workload_hash": study["workload"]["sha256"]}:
        raise ValueError("Pretest profile, membership or search settings changed")
    return frozen


def evaluation_cell_search(directory: Path, cell_id: str, objective_id: str, study: dict, workload: dict) -> dict:
    binding, normalization, _ = _load_preparation(directory, study, workload)
    profile = _pretest_profile(directory, study, workload)
    if cell_id not in profile["evaluation_cells"] or objective_id not in ("cotengra_tree_flops_v1", "upmem_launch_cost_v1"):
        raise ValueError("Evaluation requires a frozen test cell and F or U objective")
    if (directory / "evaluation_round.json").exists():
        raise ValueError("All evaluation paths are already frozen")
    instance, topology, network = _cell_inputs(cell_id, study, workload)
    return _run_cell_search(directory, cell_id, study, workload, binding, normalization, profile,
                            instance, topology, network, stage="evaluation", objective_id=objective_id)


def freeze_evaluation_round(directory: Path, study: dict, workload: dict, budget: dict) -> dict:
    from quantum_bench.planning import plan_opt_einsum

    binding, normalization, _ = _load_preparation(directory, study, workload)
    profile = _pretest_profile(directory, study, workload)
    cells = {}
    for cell_id in profile["evaluation_cells"]:
        instance, topology, network = _cell_inputs(cell_id, study, workload)
        path, _ = plan_opt_einsum(network, optimize="greedy")
        evaluated = _evaluate_with_guard(lambda p, _: lower_candidate(network, p, topology, study), path, 0.0,
                                         study["search"]["lowering_timeout_s"])
        if evaluated.score is None:
            raise ValueError("Required evaluation greedy path is infeasible")
        greedy = {"path_id": path_id(path, circuit_id=instance["instance_id"]), "path": path,
                  "facts": evaluated.facts, "tree_flops": conventional_tree_flops(network, path)}
        f_trace, u_trace = [_completed_cell_trace(directory, "evaluation", cell_id, instance["instance_id"], study,
                                                  workload, binding, normalization, profile, objective)
                            for objective in ("cotengra_tree_flops_v1", "upmem_launch_cost_v1")]
        roles = choose_evaluation_paths(greedy, f_trace, u_trace, profile=profile, normalization=normalization)
        # The union is used only after the two method-specific selections, for execution deduplication.
        pool = eligible_candidate_pool({"trace": [*f_trace["trace"], *u_trace["trace"]]}, greedy)
        selected = sorted(set(roles.values()))
        cells[cell_id] = {"selection": {"roles": roles, "path_ids": selected},
                          "trace_hashes": {"F": record_hash(f_trace), "U": record_hash(u_trace)},
                          "candidates": {p: pool[p] for p in selected}, "circuit_id": instance["instance_id"],
                          "topology_id": topology["topology_id"], "family": instance["family"], "split": "test"}
    attempts = 6 * sum(len(cell["candidates"]) for cell in cells.values())
    if len(cells) != budget["evaluation_cells"] or attempts > budget["stages"]["evaluation"]:
        raise ValueError("Evaluation membership or budget mismatch")
    manifest = {"study_id": study["study_id"], "stage": "evaluation", "binding_hash": record_hash(binding),
                "normalization_hash": record_hash(normalization), "profile_hash": record_hash(profile),
                "cells": cells, "expected_attempts": attempts, "warmup_blocks": [0],
                "measurement_blocks": [1, 2, 3, 4, 5], "physical_admission": "not_performed"}
    _write_new_json(directory / "evaluation_round.json", manifest)
    return manifest


def remaining_execution_budget(directory: Path, stage: str, manifest: dict, study: dict, workload: dict) -> dict:
    """Prelaunch budget from verified predecessors, never from a caller's counter."""
    previous_attempts, previous_elapsed = 0, 0.0
    for previous in _stage_prefix(stage):
        prior, accepted = _accepted_round(directory, previous, study, workload)
        previous_attempts += prior["expected_attempts"]
        previous_elapsed += accepted["physical_stage_elapsed_s"]
    for state in ("running", "accepted", "failed"):
        if (directory / f"{stage}.{state}.json").exists():
            raise ValueError("A running, accepted or failed physical stage cannot execute again")
    attempts = manifest["expected_attempts"]
    derived = sum(len(cell["candidates"]) for cell in manifest["cells"].values()) * (6 if stage == "evaluation" else 4)
    cap = {"initial": 192, "feedback_1": 144, "feedback_2": 144, "evaluation": 288}[stage]
    if type(attempts) is not int or attempts != derived or not 0 < attempts <= cap:
        raise ValueError("Physical attempt count is empty or outside the frozen stage budget")
    if previous_attempts + attempts > study["campaign"]["effective_attempt_cap"]:
        raise ValueError("Cumulative physical attempt budget exceeded")
    remaining = study["campaign"]["physical_stage_timeout_s"] - previous_elapsed
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("Cumulative physical time budget exhausted")
    return {"prior_attempts": previous_attempts, "stage_attempts": attempts,
            "cumulative_attempts": previous_attempts + attempts,
            "prior_physical_stage_elapsed_s": previous_elapsed, "remaining_physical_time_s": remaining}


def write_execution_packet(directory: Path, stage: str, output: Path, execution_root: Path,
                            study: dict, workload: dict, *, simulator: bool = False,
                            cpu_reference: bool = False) -> dict:
    """Stage the selected paths for the existing runner; never start execution."""
    import yaml
    from quantum_bench.evidence import canonical_json
    from quantum_bench.experiment import load_experiment_config
    from qualify_upmem_path_candidates import prepare_cost_guided_config
    from qualify_quantized_upmem_execution import write_checksums, verify_checksums
    from upmem_path_heuristic import _canonical_bytes

    binding, normalization, initial_profile = _load_preparation(directory, study, workload)
    predecessors = _stage_prefix(stage)
    profile = (initial_profile if stage == "initial" else
               _pretest_profile(directory, study, workload) if stage == "evaluation" else
               _fitted_profile(directory, predecessors[-1], study, workload))
    manifest = _read_json(directory / f"{stage}_round.json")
    if (manifest["stage"] != stage or manifest["binding_hash"] != record_hash(binding)
            or manifest["profile_hash"] != record_hash(profile)
            or manifest["normalization_hash"] != record_hash(normalization)):
        raise ValueError("Execution packet round/profile binding mismatch")
    if (directory / f"{stage}.accepted.json").exists() or (directory / f"{stage}.failed.json").exists():
        raise ValueError("Completed or failed stages cannot create replacement execution packets")
    target = "cpu" if cpu_reference else "sdk" if simulator else "physical"
    experiment_id = f"upmem-cost-guided-{stage}-{target}-{binding['source_sha'][:8]}-{record_hash(manifest)[:12]}"
    config, provenance = prepare_cost_guided_config(manifest, workload, execution_root=execution_root,
                                                     experiment_id=experiment_id, simulator=simulator,
                                                     cpu_reference=cpu_reference)
    if not simulator and not cpu_reference:
        provenance["execution_budget"] = remaining_execution_budget(directory, stage, manifest, study, workload)
    output.mkdir(parents=True, exist_ok=False)
    for source in provenance["qasm_source_bindings"]:
        data = Path(source["source_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != source["qasm_sha256"]:
            raise ValueError("QASM changed during packet staging")
        path = _contained_path(output, source["prepared_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    configuration = output / "physical.yml"
    with configuration.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=False)
    normalized = json.loads(canonical_json(load_experiment_config(configuration)))
    binary_manifest = {str(execution_root / "native/upmem/runtime/bin" / name): digest
                       for name, digest in study["executor"]["binaries"].items()}
    _write_new_json(output / "binary_sha256.json", binary_manifest)
    provenance.update(source_sha=binding["source_sha"], execution_source=study["executor"]["source"],
                      round_manifest_hash=record_hash(manifest), configuration_sha256=_file_digest(configuration),
                      normalized_configuration_sha256=hashlib.sha256(_canonical_bytes(normalized)).hexdigest(),
                      physical_admission="not_performed")
    with (output / "physical.yml.provenance.json").open("xb") as stream:
        stream.write(_canonical_bytes(provenance))
    _write_new_json(output / "round_manifest.json", manifest)
    _write_new_json(output / "binding.json", binding)
    _write_new_json(output / "profile.json", profile)
    _write_new_json(output / "normalization.json", normalization)
    write_checksums(output)
    verify_checksums(output)
    return {"experiment_id": experiment_id, "configuration_sha256": _file_digest(configuration),
            "packet_checksums_sha256": _file_digest(output / "SHA256SUMS"),
            "expected_attempts": provenance["expected_prepared_attempts"], "physical_admission": "not_performed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "inspect", "initialize", "initial-search", "freeze-initial", "accept", "fit",
        "feedback-search", "freeze-feedback", "freeze-pretest", "evaluation-search", "freeze-evaluation",
        "write-packet",
    ))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--objective", choices=("cotengra_tree_flops_v1", "upmem_launch_cost_v1"))
    parser.add_argument("--archive", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-root", type=Path)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--cpu-reference", action="store_true")
    args = parser.parse_args()
    study, workload, budget = load_study(args.study)
    if args.command != "inspect":
        if args.directory is None:
            parser.error("--directory is required for preparation commands")
        if args.command == "initialize":
            result = initialize_preparation(args.directory, study, workload)
        elif args.command == "initial-search":
            if args.cell is None:
                parser.error("--cell is required for initial-search")
            result = initial_cell_search(args.directory, args.cell, study, workload)
        elif args.command == "freeze-initial":
            result = freeze_initial_round(args.directory, study, workload, budget)
        elif args.command == "freeze-pretest":
            result = freeze_pretest(args.directory, study, workload)
        elif args.command == "freeze-evaluation":
            result = freeze_evaluation_round(args.directory, study, workload, budget)
        elif args.command == "evaluation-search":
            if args.cell is None or args.objective is None:
                parser.error("--cell and --objective are required for evaluation-search")
            result = evaluation_cell_search(args.directory, args.cell, args.objective, study, workload)
        else:
            if args.stage is None:
                parser.error("--stage is required")
            if args.command == "accept":
                result = accept_round(args.directory, args.stage, args.archive, study, workload)
            elif args.command == "write-packet":
                if args.output is None or args.execution_root is None:
                    parser.error("--output and --execution-root are required for write-packet")
                result = write_execution_packet(args.directory, args.stage, args.output, args.execution_root,
                                                study, workload, simulator=args.simulator,
                                                cpu_reference=args.cpu_reference)
            elif args.command == "fit":
                result = fit_accepted_rounds(args.directory, args.stage, study, workload)
            elif args.command == "freeze-feedback":
                result = freeze_feedback_round(args.directory, args.stage, study, workload, budget)
            else:
                if args.cell is None:
                    parser.error("--cell is required for feedback-search")
                result = feedback_cell_search(args.directory, args.stage, args.cell, study, workload)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(json.dumps({
        "study_id": study["study_id"],
        "workload_id": workload["workload_id"],
        "executor": study["executor"]["source"],
        "budget": budget,
        "physical_admission": "not_performed",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
