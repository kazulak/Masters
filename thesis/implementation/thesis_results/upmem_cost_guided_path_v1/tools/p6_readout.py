#!/usr/bin/env python3
"""Read-only P6 readout. Uses the repository's verifier; never executes hardware.

Place outside the tracked checkout. This is an operator-side report adapter,
not a new cost model, fitter, evidence-acceptance implementation, or runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = ("G", "F", "R", "U")
# Numerator/denominator. A ratio above one favors the denominator method.
CONTRASTS = (("R", "U"), ("F", "U"), ("G", "U"), ("F", "R"), ("G", "R"), ("G", "F"))
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 10_000
CORE_METRICS = ("session_inclusive_s", "total_wall_s")
OPTIONAL_METRICS = (
    "session_open_s", "session_close_s", "kernel_s", "h2d_s", "d2h_s",
    "request_build_sum_s", "request_wave_wall_sum_s",
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(manifest: dict, rows: list[dict]) -> dict:
    """Pure reporting math on rows already accepted by the canonical verifier.

    Point estimates are ratios of within-cell arm medians. Paired bootstrap
    draws resample the same complete block indices for every arm and cell.
    Families/cells are fixed; they are not bootstrap sampling units here.
    """
    require(manifest.get("stage") == "evaluation", "evaluation manifest required")
    blocks = tuple(manifest["measurement_blocks"])
    require(blocks == (1, 2, 3, 4, 5), "require the five frozen evaluation blocks")
    cells = manifest["cells"]
    require(bool(cells), "empty evaluation")
    lookup = {}
    for row in rows:
        if row["attempt_type"] == "warmup":
            continue
        require(row["attempt_type"] == "measurement", "unknown attempt kind")
        require(row["round_id"] == "evaluation" and row["split"] == "test", "wrong split/stage")
        require(row["status"] == "success" and row["validation"] == "pass"
                and row["fallback"] is False, "ineligible observation")
        key = (row["cell_id"], row["path_id"], row["block"])
        require(key not in lookup, "duplicate physical observation")
        inclusive = row["session_open_s"] + row["total_wall_s"] + row["session_close_s"]
        require(math.isclose(inclusive, row["session_inclusive_s"], rel_tol=1e-12, abs_tol=1e-12),
                "inconsistent session-inclusive boundary")
        for field in CORE_METRICS:
            require(type(row[field]) in (int, float) and math.isfinite(row[field]) and row[field] > 0,
                    f"invalid {field}")
        lookup[key] = row
    expected = {(c, p, b) for c, cell in cells.items() for p in cell["candidates"] for b in blocks}
    require(set(lookup) == expected, "measurement set differs from frozen selected paths")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(blocks), size=(BOOTSTRAP_DRAWS, len(blocks))
    )
    observations, method_rows, contrast_rows = [], [], []
    bootstrap_by_key = {}
    matrices = {}
    for cell_id, cell in sorted(cells.items()):
        require(cell["split"] == "test", "non-test cell")
        roles = cell["selection"]["roles"]
        require(set(roles) == set(METHODS), "G/F/R/U roles missing or unexpected")
        metadata = {"cell_id": cell_id, "family": cell["family"], "topology_id": cell["topology_id"]}
        for method in METHODS:
            selected = roles[method]
            chosen = [lookup[(cell_id, selected, b)] for b in blocks]
            base = {**metadata, "method": method, "path_id": selected, "n": len(blocks)}
            for block, row in zip(blocks, chosen):
                observations.append({**base, "block": block,
                                     "sample_id": row.get("sample_id"),
                                     "session_instance_id": row.get("session_instance_id"),
                                     **{f: row.get(f) for f in (*CORE_METRICS, *OPTIONAL_METRICS)}})
            for field in (*CORE_METRICS, *OPTIONAL_METRICS):
                present = all(row.get(field) is not None for row in chosen)
                if not present:
                    base[f"{field}_median"] = None
                    base[f"{field}_mad"] = None
                    continue
                values = np.asarray([row[field] for row in chosen], dtype=np.float64)
                require(np.all(np.isfinite(values)) and np.all(values >= 0), "invalid optional timing")
                med = float(np.median(values))
                base[f"{field}_median"] = med
                base[f"{field}_mad"] = float(np.median(np.abs(values - med)))
                if field in CORE_METRICS:
                    matrices[(cell_id, method, field)] = values
            method_rows.append(base)
        for metric in CORE_METRICS:
            for numerator, denominator in CONTRASTS:
                a, b = matrices[(cell_id, numerator, metric)], matrices[(cell_id, denominator, metric)]
                ratio = float(np.median(a) / np.median(b))
                boot = np.median(a[draws], axis=1) / np.median(b[draws], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                label = f"{numerator}/{denominator}"
                bootstrap_by_key[(cell_id, label, metric)] = boot
                contrast_rows.append({**metadata, "contrast": label, "metric": metric,
                                      "ratio_of_medians": ratio,
                                      "denominator_time_reduction_pct": 100 * (1 - 1 / ratio),
                                      "paired_bootstrap_low": float(lo), "paired_bootstrap_high": float(hi),
                                      "same_selected_path": roles[numerator] == roles[denominator]})
    aggregate_rows = []
    groups = [("all", sorted(cells))] + [
        (topology, sorted(c for c in cells if cells[c]["topology_id"] == topology))
        for topology in sorted({cell["topology_id"] for cell in cells.values()})
    ]
    for group, ids in groups:
        counts = Counter(cells[c]["family"] for c in ids)
        weights = np.asarray([1 / (len(counts) * counts[cells[c]["family"]]) for c in ids])
        require(np.isclose(weights.sum(), 1), "family weights do not sum to one")
        for metric in CORE_METRICS:
            for numerator, denominator in CONTRASTS:
                label = f"{numerator}/{denominator}"
                selected_rows = [next(row for row in contrast_rows if row["cell_id"] == c
                                     and row["contrast"] == label and row["metric"] == metric) for c in ids]
                point = np.asarray([row["ratio_of_medians"] for row in selected_rows])
                combined = np.stack([bootstrap_by_key[(c, label, metric)] for c in ids])
                ratio = float(np.exp(weights @ np.log(point)))
                boot = np.exp(weights @ np.log(combined))
                lo, hi = np.quantile(boot, [0.025, 0.975], method="linear")
                aggregate_rows.append({"group": group, "contrast": label, "metric": metric,
                                       "cells": len(ids), "families": len(counts),
                                       "family_balanced_geometric_ratio": ratio,
                                       "paired_bootstrap_low": float(lo), "paired_bootstrap_high": float(hi),
                                       "worst_cell_ratio": float(point.min()),
                                       "cells_denominator_slower": int(np.sum(point < 1)),
                                       "cells_same_path": sum(row["same_selected_path"] for row in selected_rows)})
    return {"observations": observations, "methods": method_rows,
            "contrasts": contrast_rows, "aggregates": aggregate_rows}


def trace_readout(directory: Path, manifest: dict, coordinator, study: dict, workload: dict) -> list[dict]:
    binding, normalization, _ = coordinator._load_preparation(directory, study, workload)
    profile = coordinator._pretest_profile(directory, study, workload)
    result = []
    for cell_id, cell in sorted(manifest["cells"].items()):
        traces = {}
        for method, objective in (("F", "cotengra_tree_flops_v1"), ("U", "upmem_launch_cost_v1")):
            trace = coordinator._completed_cell_trace(
                directory, "evaluation", cell_id, cell["circuit_id"], study, workload,
                binding, normalization, profile, objective,
            )
            require(coordinator.record_hash(trace) == cell["trace_hashes"][method], "trace hash mismatch")
            traces[method] = trace
            attempts = trace["trace"]
            result.append({"cell_id": cell_id, "method": method,
                           "proposals": len(attempts),
                           "unique_eligible_paths": len({r["path_id"] for r in attempts if r["status"] == "eligible"}),
                           "duplicate_proposals": sum(bool(r["duplicate"]) for r in attempts),
                           "known_infeasible": sum(r["status"] == "known_infeasible" for r in attempts),
                           "search_wall_s": trace["search_wall_s"],
                           **{name + "_sum_s": sum(float(r["timings_s"].get(name, 0)) for r in attempts)
                              for name in ("adaptive_ask", "adaptive_tell", "candidate_generation", "plan_evaluation")}})
        changes = sum(a["params"] != b["params"]
                      for a, b in zip(traces["F"]["trace"][16:], traces["U"]["trace"][16:]))
        for row in result[-2:]:
            row["F_U_parameter_differences_after_startup"] = changes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    impl, directory, output = args.implementation.resolve(), args.directory.resolve(), args.output.resolve()
    require(not output.exists(), "readout output already exists; do not overwrite")
    require(not output.is_relative_to(directory), "keep reports outside the immutable control directory")
    sys.path[:0] = [str(impl / "scripts"), str(impl / "src")]
    import upmem_cost_guided_path as coordinator

    study, workload, budget = coordinator.load_study(impl / "configs/upmem_cost_guided_path_study_v1.json", root=impl)
    coordinator._load_preparation(directory, study, workload)  # Exact source and research environment.
    profile = coordinator._pretest_profile(directory, study, workload)
    manifest, accepted = coordinator._accepted_round(directory, "evaluation", study, workload)
    require(manifest["profile_hash"] == coordinator.record_hash(profile), "evaluation profile mismatch")
    require(len(manifest["cells"]) == budget["evaluation_cells"], "missing evaluation cells")
    summary = summarize(manifest, accepted["rows"])
    traces = trace_readout(directory, manifest, coordinator, study, workload)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in summary.items():
        write_csv(output / f"{name}.csv", rows)
    write_csv(output / "search.csv", traces)
    write_json(output / "accepted_evaluation_rows.json", accepted["rows"])
    feature_rows = []
    for cell_id, cell in sorted(manifest["cells"].items()):
        for method in METHODS:
            p = cell["selection"]["roles"][method]
            f = cell["candidates"][p]["facts"]
            feature_rows.append({"cell_id": cell_id, "method": method, "path_id": p,
                                 **{k: f[k] for k in ("H", "P", "N")},
                                 "M_total": sum(sum(l["M"]) for l in f["launches"]),
                                 "W_total": sum(sum(l["W"]) for l in f["launches"]),
                                 "launch_count": len(f["launches"])})
    write_csv(output / "selected_plan_facts.csv", feature_rows)
    # Descriptive diagnostics only; these never change coefficients or selection.
    measured = {}
    for stage in ("initial", "feedback_1", "feedback_2"):
        development = read_json(directory / f"{stage}_round.json")
        for cell_id, cell in development["cells"].items():
            for path_id, candidate in cell["candidates"].items():
                f = candidate["facts"]
                vector = (f["H"], f["P"], f["N"],
                          sum(sum(l["M"]) for l in f["launches"]),
                          sum(sum(l["W"]) for l in f["launches"]))
                prior = measured.setdefault((cell_id, path_id), vector)
                require(prior == vector, "development facts changed")
    names = ("H", "P", "N", "M", "W")
    x = np.asarray(list(measured.values()), dtype=np.float64)
    constant = [bool(np.all(x[:, k] == x[0, k])) for k in range(5)]
    correlations = [[None if constant[a] or constant[b]
                     else float(np.corrcoef(x[:, a], x[:, b])[0, 1])
                     for b in range(5)] for a in range(5)]
    grid = read_json(directory / "feedback_2_fit/grid.json")
    decisions = {tuple((cell_id, value["path_id"]) for cell_id, value in sorted(row["cells"].items()))
                 for row in grid}
    best_j = max(row["rounded_J"] for row in grid)
    write_json(output / "model_diagnostics.json", {
        "features": names, "pooled_measured_development_pairs": len(measured),
        "constant_features": [name for name, yes in zip(names, constant) if yes],
        "pooled_total_feature_correlations": correlations,
        "grid_size": len(grid), "distinct_grid_selection_vectors": len(decisions),
        "tuples_tied_at_best_rounded_J": sum(row["rounded_J"] == best_j for row in grid),
        "interpretation": "Pooled aggregate feature totals, not per-cell identifiability or a proof of unique physical coefficients; launch maxima remain in the actual model.",
    })
    write_json(output / "provenance.json", {
        "study_hash": coordinator.record_hash(study), "profile": profile,
        "accepted_evaluation_hash": coordinator.record_hash(accepted),
        "evaluation_manifest_hash": coordinator.record_hash(manifest),
        "verified_archives": accepted["archives"], "report_script_sha256": digest(Path(__file__)),
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "point_estimator": "ratio_of_arm_medians_within_cell_then_family_balanced_geometric_aggregation",
        "bootstrap_unit": "five complete block IDs, common resample across cells and methods",
        "primary_contrast": "R/U on session_inclusive_s; >1 favors U",
        "interpretation": "Conditional descriptive intervals, fixed six families/two topologies, one search seed; not population-generalization intervals.",
    })
    primary = next(r for r in summary["aggregates"] if r["group"] == "all"
                   and r["contrast"] == "R/U" and r["metric"] == "session_inclusive_s")
    lines = ["# Frozen P6 evaluation readout", "",
             "Primary comparison: R/U; a ratio above one favors UPMEM-guided generation.", "",
             f"Family-balanced geometric ratio: **{primary['family_balanced_geometric_ratio']:.6f}**.",
             f"Descriptive paired-block 95% bootstrap interval: [{primary['paired_bootstrap_low']:.6f}, {primary['paired_bootstrap_high']:.6f}].", "",
             "## Interpretation boundaries", "",
             "All figures are derived from reverified accepted raw evidence. Warmups are excluded.",
             "Method aliases share a physical sample; they are not additional independent observations.",
             "Session-inclusive time is not complete job time. Search cost is reported separately.",
             "Five timing blocks and one paired search-seed schedule do not establish broad population or optimizer-seed robustness.",
             "A neutral or negative result does not authorize further tuning.", "",
             "See methods.csv, contrasts.csv, aggregates.csv, search.csv and selected_plan_facts.csv for the complete numerical result."]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    with (output / "SHA256SUMS").open("x", encoding="ascii") as stream:
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS":
                stream.write(f"{digest(path)}  {path.name}\n")
    print(json.dumps({"output": str(output), "primary_R_over_U": primary}, indent=2))


if __name__ == "__main__":
    main()
