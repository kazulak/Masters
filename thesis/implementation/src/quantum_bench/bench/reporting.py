from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quantum_bench.bench.result_artifacts import RESULT_FIELDS, load_result_records
from quantum_bench.core.jsonio import read_jsonl, write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable


RUN_MANIFEST_SCHEMA_VERSION = "run_manifest_v1"
ARTIFACT_REFERENCE_SCHEMA_VERSION = "artifact_reference_v1"
ARTIFACT_RETENTION_SCHEMA_VERSION = "artifact_retention_v1"
REPORT_RUN_SCHEMA_VERSION = "report_run_v1"
COMPARE_RUNS_SCHEMA_VERSION = "compare_runs_v1"
NORMALIZED_RECORDS_SCHEMA_VERSION = "normalized_records_v1"

RETENTION_MODES = ("full", "compact")
COMPACT_PRUNE_PATTERNS = (
    "runner_work",
    ".bin",
    "operands",
    "references",
    "outputs",
)

REPORT_RESULT_FIELDS = [
    "case_id",
    "workload_id",
    "route_id",
    "backend_family",
    "execution_model",
    "output_kind",
    "policy",
    "quantization_mode",
    "kernel_family",
    "status",
    "validation_status",
    "contraction_execution_target",
    "upmem_execution_mode",
    "execution_scope",
    "task_count",
    "validated_task_count",
    "unsupported_task_count",
    "total_wall_time_s",
    "kernel_time_s",
    "build_time_s",
    "hardware_speedup",
]

TIMING_FIELDS = [
    "cpu_reference_time_s",
    "upmem_runtime_wall_time_s",
    "host_orchestration_time_s",
    "quantization_time_s",
    "bridge_prepare_time_s",
    "native_build_time_s",
    "dpu_program_wall_time_s",
    "dequantization_time_s",
    "validation_time_s",
]


@dataclass(frozen=True)
class ReportRunResult:
    run_dir: Path
    report_path: Path
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class PruneRunResult:
    run_dir: Path
    manifest_path: Path
    status: str
    pruned_file_count: int


@dataclass(frozen=True)
class CompareRunsResult:
    run_dir: Path
    artifact_path: Path
    summary_path: Path
    status: str


def validate_retention_mode(mode: str) -> None:
    if mode == "summary-only":
        raise ValueError("summary-only artifact retention is deferred for a later wave")
    if mode not in RETENTION_MODES:
        raise ValueError(f"unsupported artifact retention mode: {mode}")


def write_run_manifest(
    run_dir: Path,
    *,
    run_kind: str,
    suite_id: str | None,
    suite_path: str | None,
    policies: Iterable[str] = (),
    quantization_modes: Iterable[str] = (),
    upmem_execution_mode: str | None = None,
    artifact_retention: str = "full",
    command: str | None = None,
    root_dir: Path | None = None,
) -> JsonDict:
    validate_retention_mode(artifact_retention)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "timestamp": None,
        "git_commit": _git_commit(root_dir or run_dir),
        "dirty_tree": _git_dirty(root_dir or run_dir),
        "suite_id": suite_id,
        "suite_path": suite_path,
        "policies": tuple(policies),
        "quantization_modes": tuple(quantization_modes),
        "upmem_execution_mode": upmem_execution_mode,
        "artifact_retention": artifact_retention,
        "python_version": platform.python_version(),
        "upmem_sdk_available": "unknown",
        "hardware_available": "not_checked",
        "environment_hash": _environment_hash(),
        "command": command,
        "schema_versions": {
            "run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
            "artifact_reference": ARTIFACT_REFERENCE_SCHEMA_VERSION,
            "artifact_retention": ARTIFACT_RETENTION_SCHEMA_VERSION,
            "report_run": REPORT_RUN_SCHEMA_VERSION,
            "compare_runs": COMPARE_RUNS_SCHEMA_VERSION,
            "normalized_records": NORMALIZED_RECORDS_SCHEMA_VERSION,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return to_jsonable(manifest)


def artifact_ref(run_dir: Path, rel_path: str | Path | None, *, role: str) -> JsonDict | None:
    if rel_path is None:
        return None
    rel = Path(rel_path)
    payload = {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "role": role,
        "relative_path": rel.as_posix(),
        "retained": (run_dir / rel).exists(),
        "status": "retained" if (run_dir / rel).exists() else "missing_unexpectedly",
        "prune_reason": None,
        "metadata": _file_metadata(run_dir / rel),
    }
    return to_jsonable(payload)


def write_normalized_records(run_dir: Path, records: Iterable[JsonDict]) -> Path:
    path = run_dir / "normalized_records.jsonl"
    payloads = []
    for record in records:
        normalized = dict(record)
        normalized.setdefault("normalized_record_schema_version", NORMALIZED_RECORDS_SCHEMA_VERSION)
        payloads.append(to_jsonable(normalized))
    write_jsonl(path, payloads)
    return path


def load_normalized_records(run_dir: Path) -> list[JsonDict]:
    return read_jsonl(run_dir / "normalized_records.jsonl")


def report_run(run_dir: Path, *, output_plots: bool = True) -> ReportRunResult:
    run_dir = run_dir.resolve()
    records = _load_run_records(run_dir)
    if not records:
        raise ValueError("report-run found no normalized benchmark records")
    _write_report_artifacts(run_dir, records, output_plots=output_plots)
    payload = {
        "schema_version": REPORT_RUN_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "completed",
        "record_count": len(records),
        "non_destructive": True,
        "overwrites_derived_reports": True,
        "deletes_execution_artifacts": False,
    }
    report_path = run_dir / "report_run.json"
    write_json(report_path, payload)
    _cleanup_empty_report_dirs(run_dir)
    return ReportRunResult(run_dir=run_dir, report_path=report_path, status="completed")


def prune_run(run_dir: Path, *, artifact_retention: str = "compact") -> PruneRunResult:
    validate_retention_mode(artifact_retention)
    run_dir = run_dir.resolve()
    _require_new_run_layout(run_dir)
    if artifact_retention == "full":
        manifest = _retention_manifest(run_dir, mode="full", pruned=[], retained=_all_files(run_dir))
        write_json(run_dir / "artifact_retention_manifest.json", manifest)
        return PruneRunResult(run_dir, run_dir / "artifact_retention_manifest.json", "completed", 0)
    candidates = _compact_prune_candidates(run_dir)
    pruned_refs: list[JsonDict] = []
    for path in candidates:
        if not path.exists():
            continue
        rel = path.relative_to(run_dir)
        ref = _pruned_reference(run_dir, rel, "compact_retention")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        pruned_refs.append(ref)
    _mark_pruned_references(run_dir, pruned_refs)
    manifest = _retention_manifest(run_dir, mode="compact", pruned=pruned_refs, retained=_all_files(run_dir))
    write_json(run_dir / "artifact_retention_manifest.json", manifest)
    _cleanup_empty_report_dirs(run_dir)
    return PruneRunResult(run_dir, run_dir / "artifact_retention_manifest.json", "completed", len(pruned_refs))


def compare_runs(baseline: Path, candidate: Path, out_dir: Path) -> CompareRunsResult:
    baseline_records = _load_run_records(baseline.resolve())
    candidate_records = _load_run_records(candidate.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    final = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=(
            "case_id",
            "route_id",
            "execution_model",
            "policy",
            "quantization_mode",
            "contraction_execution_target",
            "upmem_execution_mode",
        ),
    )
    cpu = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=("case_id", "contraction_execution_target", "execution_scope"),
        predicate=lambda record: record.get("execution_target") == "cpu",
    )
    kernel = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=("case_id", "policy", "quantization_mode", "contraction_execution_target", "upmem_execution_mode", "kernel_family"),
    )
    payload = {
        "schema_version": COMPARE_RUNS_SCHEMA_VERSION,
        "status": "completed",
        "baseline_run": baseline.resolve().name,
        "candidate_run": candidate.resolve().name,
        "final_validation_accuracy_timing": final,
        "cpu_reference": cpu,
        "kernel_family_mix": kernel,
        "metadata": {
            "hardware_speedup_not_inferred_from_sdk_simulator": True,
            "comparison_keys": {
                "final": (
                    "case_id",
                    "route_id",
                    "execution_model",
                    "policy",
                    "quantization_mode",
                    "contraction_execution_target",
                    "upmem_execution_mode",
                ),
                "cpu": ("case_id", "contraction_execution_target", "execution_scope"),
                "kernel_family": (
                    "case_id",
                    "policy",
                    "quantization_mode",
                    "contraction_execution_target",
                    "upmem_execution_mode",
                    "kernel_family",
                ),
            },
        },
    }
    artifact_path = out_dir / "compare_runs.json"
    summary_path = out_dir / "compare_runs_summary.md"
    write_json(artifact_path, payload)
    summary_path.write_text(_compare_runs_markdown(payload), encoding="utf-8")
    return CompareRunsResult(out_dir, artifact_path, summary_path, "completed")


def _load_run_records(run_dir: Path) -> list[JsonDict]:
    normalized = run_dir / "normalized_records.jsonl"
    if normalized.exists():
        return read_jsonl(normalized)
    return load_result_records([run_dir])


def _write_report_artifacts(run_dir: Path, records: list[JsonDict], *, output_plots: bool) -> None:
    _write_csv(run_dir / "upmem_mvp_benchmark_results.csv", records, REPORT_RESULT_FIELDS)
    _write_csv(run_dir / "kernel_family_summary.csv", _kernel_family_summary(records), ["kernel_family", "record_count", "task_count", "validated_task_count", "unsupported_task_count"])
    _write_csv(run_dir / "quantization_accuracy_summary.csv", _quantization_rows(records), ["case_id", "policy", "quantization_mode", "validation_status", "max_abs_error", "l2_error"])
    _write_csv(run_dir / "unsupported_reasons.csv", _unsupported_rows(records), ["case_id", "policy", "quantization_mode", "reason", "count"])
    _write_csv(run_dir / "metrics" / "per_task_metrics.csv", _per_task_rows(run_dir), sorted(_per_task_fieldnames(run_dir)))
    _write_csv(run_dir / "metrics" / "per_case_metrics.csv", _per_case_rows(records), ["case_id", "policy", "quantization_mode", "task_count", "validated_task_count", "unsupported_task_count", "status"])
    _write_csv(run_dir / "metrics" / "timing_breakdown.csv", _timing_rows(records), ["case_id", "policy", "quantization_mode", *TIMING_FIELDS, "timing_status"])
    write_json(run_dir / "validation" / "validation_summary.json", _validation_summary(records))
    write_jsonl(run_dir / "validation" / "validation_failures.jsonl", _validation_failures(records))
    (run_dir / "comparison_summary.md").write_text(_report_markdown(records), encoding="utf-8")
    if output_plots:
        _write_plots(run_dir, records)
    else:
        write_json(run_dir / "plots" / "plot_manifest.json", {"schema_version": REPORT_RUN_SCHEMA_VERSION, "status": "skipped", "reason": "plot_generation_disabled"})


def _write_plots(run_dir: Path, records: list[JsonDict]) -> None:
    plots_dir = run_dir / "plots"
    skipped: list[JsonDict] = []
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots_dir / "plot_manifest.json", {"schema_version": REPORT_RUN_SCHEMA_VERSION, "status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc)})
        return
    required = {
        "kernel_family_counts.png": _plot_kernel_family_counts,
        "final_max_error_by_case.png": _plot_final_max_error,
        "runtime_breakdown_by_policy.png": _plot_runtime_breakdown,
        "unsupported_reasons.png": _plot_unsupported_reasons,
    }
    written: list[str] = []
    for name, fn in required.items():
        reason = fn(plt, plots_dir / name, records)
        if reason:
            skipped.append({"plot": name, "reason": reason})
        else:
            written.append(name)
    write_json(plots_dir / "plot_manifest.json", {"schema_version": REPORT_RUN_SCHEMA_VERSION, "status": "completed", "written": written, "skipped": skipped})


def _plot_kernel_family_counts(plt: Any, path: Path, records: list[JsonDict]) -> str | None:
    counts = Counter(str(record.get("kernel_family") or "unknown") for record in records)
    return _simple_bar(plt, path, counts, "Kernel Family Counts", "Records")


def _plot_final_max_error(plt: Any, path: Path, records: list[JsonDict]) -> str | None:
    values: dict[str, float] = {}
    for record in records:
        metrics = _json_value(record.get("validation_error_metrics"))
        value = metrics.get("max_abs_error")
        if value is not None:
            values[_record_label(record)] = float(value)
    return _simple_bar(plt, path, values, "Final Max Error By Case", "Max abs error")


def _plot_runtime_breakdown(plt: Any, path: Path, records: list[JsonDict]) -> str | None:
    values: dict[str, float] = {}
    for record in records:
        values[_record_label(record)] = float(record.get("total_wall_time_s") or 0.0)
    return _simple_bar(plt, path, values, "Runtime By Backend/Policy", "Seconds")


def _plot_unsupported_reasons(plt: Any, path: Path, records: list[JsonDict]) -> str | None:
    counts: Counter[str] = Counter()
    for record in records:
        if int(record.get("unsupported_task_count", 0) or 0):
            counts[str(record.get("warnings") or record.get("status") or "unsupported")] += int(record.get("unsupported_task_count", 0) or 0)
    return _simple_bar(plt, path, counts, "Unsupported Reasons", "Tasks")


def _simple_bar(plt: Any, path: Path, values: dict[str, float] | Counter[str], title: str, ylabel: str) -> str | None:
    if not values:
        return "required_data_unavailable"
    labels = list(values.keys())
    data = [float(values[label]) for label in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(max(6.0, len(labels) * 0.6), 4.0))
    axis.bar(labels, data, color="#2563eb")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return None


def _record_label(record: JsonDict) -> str:
    parts = [str(record.get("case_id") or "case")]
    if record.get("route_id"):
        parts.append(str(record["route_id"]))
    if record.get("policy"):
        parts.append(str(record["policy"]))
    if record.get("kernel_family"):
        parts.append(str(record["kernel_family"]))
    return "/".join(parts)


def _retention_manifest(run_dir: Path, *, mode: str, pruned: list[JsonDict], retained: list[Path]) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": ARTIFACT_RETENTION_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "artifact_retention": mode,
            "status": "completed",
            "idempotent": True,
            "pruned_file_count": len(pruned),
            "pruned_byte_count": sum(int((ref.get("metadata") or {}).get("size_bytes", 0) or 0) for ref in pruned),
            "retained_file_count": len(retained),
            "pruned_artifacts": pruned,
            "compare_results_supported": (run_dir / "normalized_records.jsonl").exists(),
            "report_run_supported": (run_dir / "normalized_records.jsonl").exists(),
        }
    )


def _require_new_run_layout(run_dir: Path) -> None:
    if not (run_dir / "run_manifest.json").exists() or not (run_dir / "normalized_records.jsonl").exists():
        raise ValueError("unsupported_legacy_run_layout")


def _compact_prune_candidates(run_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    for path in run_dir.rglob("*"):
        rel = path.relative_to(run_dir).as_posix()
        if "runner_work/" in rel or rel.endswith("/runner_work") or path.name == "runner_work":
            candidates.add(path)
            continue
        if path.is_file() and path.suffix == ".bin":
            candidates.add(path)
            continue
        if any(part in {"operands", "references", "outputs"} for part in path.relative_to(run_dir).parts):
            candidates.add(path)
    return sorted(candidates, key=lambda item: (len(item.parts), item.as_posix()), reverse=True)


def _pruned_reference(run_dir: Path, rel_path: Path, reason: str) -> JsonDict:
    path = run_dir / rel_path
    return to_jsonable(
        {
            "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
            "role": "pruned_artifact",
            "relative_path": rel_path.as_posix(),
            "retained": False,
            "status": "intentionally_pruned",
            "prune_reason": reason,
            "metadata": _file_metadata(path),
        }
    )


def _mark_pruned_references(run_dir: Path, refs: list[JsonDict]) -> None:
    if not refs:
        return
    by_path = {str(ref["relative_path"]): ref for ref in refs}
    for json_path in list(run_dir.rglob("*.json")):
        if any(part in {"runner_work", "operands", "references", "outputs"} for part in json_path.relative_to(run_dir).parts):
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        updated = _replace_pruned_refs(payload, by_path, run_dir=run_dir, base_dir=json_path.parent)
        if updated != payload:
            write_json(json_path, updated)


def _replace_pruned_refs(value: Any, refs: dict[str, JsonDict], *, run_dir: Path, base_dir: Path) -> Any:
    if isinstance(value, dict):
        current = dict(value)
        rel = current.get("relative_path")
        resolved = _matching_reference_key(rel, refs, run_dir=run_dir, base_dir=base_dir) if isinstance(rel, str) else None
        if resolved in refs:
            return refs[resolved]
        for key, child in list(current.items()):
            resolved_child = _matching_reference_key(child, refs, run_dir=run_dir, base_dir=base_dir) if isinstance(child, str) else None
            if resolved_child in refs and (key.endswith("artifact") or key.endswith("path") or key == "relative_path"):
                current[key] = refs[resolved_child]
            else:
                current[key] = _replace_pruned_refs(child, refs, run_dir=run_dir, base_dir=base_dir)
        return current
    if isinstance(value, list):
        return [_replace_pruned_refs(item, refs, run_dir=run_dir, base_dir=base_dir) for item in value]
    return value


def _matching_reference_key(value: str | None, refs: dict[str, JsonDict], *, run_dir: Path, base_dir: Path) -> str | None:
    for key in _reference_keys(value, run_dir=run_dir, base_dir=base_dir):
        if key in refs:
            return key
    return None


def _reference_keys(value: str | None, *, run_dir: Path, base_dir: Path) -> list[str]:
    if not value:
        return []
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(run_dir / raw)
        candidates.append(base_dir / raw)
    keys: list[str] = []
    for candidate in candidates:
        try:
            keys.append(candidate.resolve().relative_to(run_dir.resolve()).as_posix())
        except ValueError:
            try:
                keys.append(candidate.relative_to(run_dir).as_posix())
            except ValueError:
                continue
    keys.append(value)
    return keys


def _file_metadata(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        return {"kind": "directory", "file_count": len(files), "size_bytes": sum(item.stat().st_size for item in files)}
    payload: JsonDict = {"kind": "file", "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    if path.suffix == ".npy":
        try:
            import numpy as np

            array = np.load(path, allow_pickle=False, mmap_mode="r")
            payload["dtype"] = str(array.dtype)
            payload["shape"] = tuple(int(dim) for dim in array.shape)
        except Exception:
            pass
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_files(run_dir: Path) -> list[Path]:
    return sorted(path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file())


def _cleanup_empty_report_dirs(run_dir: Path) -> None:
    for directory in sorted((p for p in run_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if directory == run_dir:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def _write_csv(path: Path, rows: list[JsonDict], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    tmp.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return value


def _kernel_family_summary(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("kernel_family") or "unsupported")].append(record)
    rows = []
    for family, group in sorted(grouped.items()):
        rows.append(
            {
                "kernel_family": family,
                "record_count": len(group),
                "task_count": sum(int(row.get("task_count", 0) or 0) for row in group),
                "validated_task_count": sum(int(row.get("validated_task_count", 0) or 0) for row in group),
                "unsupported_task_count": sum(int(row.get("unsupported_task_count", 0) or 0) for row in group),
            }
        )
    return rows


def _quantization_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows = []
    for record in records:
        if record.get("execution_target") != "upmem":
            continue
        metrics = _json_value(record.get("validation_error_metrics"))
        notes = _json_value(record.get("notes"))
        rows.append(
            {
                "case_id": record.get("case_id"),
                "policy": notes.get("policy"),
                "quantization_mode": notes.get("quantization_mode"),
                "validation_status": record.get("validation_status"),
                "max_abs_error": metrics.get("max_abs_error"),
                "l2_error": metrics.get("l2_error"),
            }
        )
    return rows


def _unsupported_rows(records: list[JsonDict]) -> list[JsonDict]:
    grouped: Counter[tuple[str, str, str, str]] = Counter()
    for record in records:
        count = int(record.get("unsupported_task_count", 0) or 0)
        if count <= 0:
            continue
        notes = _json_value(record.get("notes"))
        key = (
            str(record.get("case_id")),
            str(notes.get("policy")),
            str(notes.get("quantization_mode")),
            str(record.get("warnings") or record.get("status") or "unsupported"),
        )
        grouped[key] += count
    return [{"case_id": k[0], "policy": k[1], "quantization_mode": k[2], "reason": k[3], "count": v} for k, v in sorted(grouped.items())]


def _per_task_rows(run_dir: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for path in sorted(run_dir.rglob("upmem_taskgraph_task_metrics.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _per_task_fieldnames(run_dir: Path) -> set[str]:
    fields: set[str] = set()
    for row in _per_task_rows(run_dir):
        fields.update(row)
    return fields or {"status"}


def _per_case_rows(records: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "case_id": record.get("case_id"),
            "policy": record.get("policy") or _json_value(record.get("notes")).get("policy"),
            "quantization_mode": record.get("quantization_mode") or _json_value(record.get("notes")).get("quantization_mode"),
            "task_count": record.get("task_count"),
            "validated_task_count": record.get("validated_task_count"),
            "unsupported_task_count": record.get("unsupported_task_count"),
            "status": record.get("status"),
        }
        for record in records
    ]


def _timing_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows = []
    for record in records:
        notes = _json_value(record.get("notes"))
        rows.append(
            {
                "case_id": record.get("case_id"),
                "policy": notes.get("policy"),
                "quantization_mode": notes.get("quantization_mode"),
                "cpu_reference_time_s": record.get("total_wall_time_s") if record.get("execution_target") == "cpu" else None,
                "upmem_runtime_wall_time_s": record.get("total_wall_time_s") if record.get("execution_target") == "upmem" else None,
                "host_orchestration_time_s": None,
                "quantization_time_s": None,
                "bridge_prepare_time_s": None,
                "native_build_time_s": record.get("build_time_s") if record.get("execution_target") == "upmem" else None,
                "dpu_program_wall_time_s": record.get("kernel_time_s") if record.get("execution_target") == "upmem" else None,
                "dequantization_time_s": None,
                "validation_time_s": None,
                "timing_status": _timing_status(record),
            }
        )
    return rows


def _timing_status(record: JsonDict) -> str:
    if record.get("simulator_or_hardware") == "simulator":
        return "measured_sdk_simulator_wall_clock_not_hardware"
    if record.get("execution_target") == "cpu":
        return "measured_cpu_reference"
    return "not_measured"


def _validation_summary(records: list[JsonDict]) -> JsonDict:
    return {
        "schema_version": REPORT_RUN_SCHEMA_VERSION,
        "record_count": len(records),
        "passed_count": sum(1 for row in records if row.get("validation_status") in {"passed", "reference"}),
        "failed_count": sum(1 for row in records if row.get("validation_status") == "failed"),
        "hardware_speedup_applicable": False,
    }


def _validation_failures(records: list[JsonDict]) -> list[JsonDict]:
    return [row for row in records if row.get("validation_status") == "failed"]


def _report_markdown(records: list[JsonDict]) -> str:
    lines = [
        "# Benchmark Run Report",
        "",
        f"Records: {len(records)}",
        "",
        "SDK simulator timings, when present, are wall-clock development measurements, not hardware speedups.",
        "CPU full-state, CPU tensor-network, and future UPMEM tensor-network records are separated by execution model and target.",
        "",
        "## Execution Models",
        "",
        "| Execution model | Records |",
        "| --- | ---: |",
    ]
    for model, count in sorted(Counter(str(record.get("execution_model") or "unspecified") for record in records).items()):
        lines.append(f"| {model} | {count} |")
    lines.extend(
        [
            "",
            "## Kernel Families",
            "",
            "| Kernel family | Records | Tasks | Validated tasks | Unsupported tasks |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _kernel_family_summary(records):
        lines.append(f"| {row['kernel_family']} | {row['record_count']} | {row['task_count']} | {row['validated_task_count']} | {row['unsupported_task_count']} |")
    lines.append("")
    return "\n".join(lines)


def _compare_grouped(
    baseline: list[JsonDict],
    candidate: list[JsonDict],
    *,
    key_fields: tuple[str, ...],
    predicate: Any | None = None,
) -> JsonDict:
    def selected(records: list[JsonDict]) -> dict[tuple[Any, ...], JsonDict]:
        out = {}
        for record in records:
            if predicate is not None and not predicate(record):
                continue
            out[tuple(record.get(field) for field in key_fields)] = record
        return out

    old = selected(baseline)
    new = selected(candidate)
    keys = sorted(set(old) | set(new), key=lambda item: tuple(str(part) for part in item))
    rows = []
    for key in keys:
        before = old.get(key)
        after = new.get(key)
        rows.append(
            {
                "key": dict(zip(key_fields, key)),
                "baseline_present": before is not None,
                "candidate_present": after is not None,
                "validation_status_changed": (before or {}).get("validation_status") != (after or {}).get("validation_status"),
                "status_changed": (before or {}).get("status") != (after or {}).get("status"),
                "task_count_delta": int((after or {}).get("task_count", 0) or 0) - int((before or {}).get("task_count", 0) or 0),
                "unsupported_task_count_delta": int((after or {}).get("unsupported_task_count", 0) or 0) - int((before or {}).get("unsupported_task_count", 0) or 0),
                "total_wall_time_delta_s": float((after or {}).get("total_wall_time_s", 0.0) or 0.0) - float((before or {}).get("total_wall_time_s", 0.0) or 0.0),
                "max_abs_error_delta": _metric_delta(before, after, "max_abs_error"),
            }
        )
    return {
        "key_fields": key_fields,
        "row_count": len(rows),
        "newly_supported_count": sum(1 for row in rows if not row["baseline_present"] and row["candidate_present"]),
        "newly_unsupported_count": sum(1 for row in rows if row["baseline_present"] and not row["candidate_present"]),
        "validation_regression_count": sum(1 for row in rows if row["validation_status_changed"]),
        "rows": rows,
    }


def _metric_delta(before: JsonDict | None, after: JsonDict | None, name: str) -> float | None:
    old = _json_value((before or {}).get("validation_error_metrics")).get(name)
    new = _json_value((after or {}).get("validation_error_metrics")).get(name)
    if old is None or new is None:
        return None
    return float(new) - float(old)


def _compare_runs_markdown(payload: JsonDict) -> str:
    final = payload["final_validation_accuracy_timing"]
    kernel = payload["kernel_family_mix"]
    lines = [
        "# Benchmark Run Comparison",
        "",
        f"Baseline: `{payload['baseline_run']}`",
        f"Candidate: `{payload['candidate_run']}`",
        "",
        f"Final comparison rows: {final['row_count']}",
        f"Validation changed rows: {final['validation_regression_count']}",
        f"Kernel-family comparison rows: {kernel['row_count']}",
        "",
        "SDK simulator timing deltas are not hardware speedups.",
        "",
    ]
    return "\n".join(lines)


def _json_value(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _git_commit(root_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root_dir, check=False, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(root_dir: Path) -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=root_dir, check=False, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _environment_hash() -> str:
    keys = {key: value for key, value in os.environ.items() if key.startswith(("UPMEM", "SIMPLEPIM", "PID_COMM", "PYTHON"))}
    payload = json.dumps(keys, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
