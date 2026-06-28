from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from quantum_bench.bench.run_dirs import create_run_dir
from quantum_bench.core.jsonio import write_json
from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.environment import capture_environment
from quantum_bench.targets.upmem.external_libs import (
    CommandRunner,
    external_libs_report_rows,
    build_external_pim_libraries_report,
)
from quantum_bench.targets.upmem.environment import DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS, run_command


EXTERNAL_PIM_LIBRARIES_CSV_FIELDS = [
    "schema_version",
    "component",
    "candidate_id",
    "candidate_role",
    "status",
    "execution_implemented",
    "top_level_benchmark_route",
    "blocker_reason",
    "evidence_detected",
    "capability_proven",
    "evidence_paths",
]


def run_upmem_external_libs_check(
    root_dir: Path,
    *,
    simplepim_home: str | None = None,
    pid_comm_home: str | None = None,
    check_pid_comm_build: bool = False,
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = run_command,
) -> tuple[Path, Path, str]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    run_dir = create_run_dir(root_dir, "upmem_external_libs_check")
    write_json(run_dir / "environment.json", capture_environment(root_dir))
    report = build_external_pim_libraries_report(
        root_dir,
        simplepim_home=simplepim_home,
        pid_comm_home=pid_comm_home,
        check_pid_comm_build=check_pid_comm_build,
        timeout_seconds=timeout_seconds,
        env=env,
        command_runner=command_runner,
    )
    artifact_path = run_dir / "external_pim_libraries.json"
    write_json(artifact_path, report)
    _write_csv(run_dir / "external_pim_libraries.csv", external_libs_report_rows(report))
    (run_dir / "external_pim_libraries_summary.md").write_text(
        _summary_markdown(report.to_json_dict()),
        encoding="utf-8",
    )
    return run_dir, artifact_path, report.status


def _write_csv(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTERNAL_PIM_LIBRARIES_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in EXTERNAL_PIM_LIBRARIES_CSV_FIELDS})


def _summary_markdown(payload: JsonDict) -> str:
    native = dict(payload.get("native_sdk_control") or {})
    simplepim = dict(payload.get("simplepim") or {})
    pid_comm = dict(payload.get("pid_comm") or {})
    simplepim_candidate = dict(simplepim.get("candidate") or {})
    pid_candidate = dict(pid_comm.get("candidate") or {})
    lines = [
        "# External PIM Libraries Check",
        "",
        "This report is a feasibility and integration-boundary artifact. It does not execute SimplePIM kernels, PID-Comm collectives, providers, or UPMEM benchmark routes.",
        "",
        "Marker scans are evidence only. A detected source marker does not prove a working API or kernel capability.",
        "",
        "## Candidate Status",
        "",
        "| Candidate | Role | Status | Execution implemented | Top-level route | Blocker |",
        "|---|---|---|---:|---:|---|",
        _candidate_row("native_upmem_sdk_control", "control_baseline_and_fallback", native.get("status"), True, False, "" if native.get("status") == "available" else "upmem_sdk_unavailable"),
        _candidate_row(
            simplepim_candidate.get("candidate_id", "simplepim_dense_candidate"),
            simplepim_candidate.get("candidate_role", "internal_l1_l2_compute_candidate"),
            simplepim_candidate.get("status"),
            simplepim_candidate.get("execution_implemented"),
            simplepim_candidate.get("top_level_benchmark_route"),
            simplepim.get("simplepim_blocker_reason"),
        ),
        _candidate_row(
            pid_candidate.get("candidate_id", "pid_comm_candidate"),
            pid_candidate.get("candidate_role", "internal_l3_communication_candidate"),
            pid_candidate.get("status"),
            pid_candidate.get("execution_implemented"),
            pid_candidate.get("top_level_benchmark_route"),
            pid_comm.get("pid_comm_blocker_reason"),
        ),
        "",
        "## SimplePIM",
        "",
        f"- Source detected: {simplepim.get('simplepim_detected')}",
        f"- Source: {simplepim.get('simplepim_source')}",
        f"- Ready GEMM primitive detected: {simplepim.get('ready_gemm_primitive_detected')}",
        f"- Ready GEMM capability proven: {simplepim.get('ready_gemm_capability_proven')}",
        f"- Blocker: {simplepim.get('simplepim_blocker_reason')}",
        "",
        "## PID-Comm",
        "",
        f"- Source detected: {pid_comm.get('pid_comm_detected')}",
        f"- Source: {pid_comm.get('pid_comm_source')}",
        f"- Collective evidence detected: {dict(pid_comm.get('collective_api') or {}).get('evidence_detected')}",
        f"- Collective capability proven: {pid_comm.get('collective_capability_proven')}",
        f"- Build check status: {pid_comm.get('build_check_status')}",
        f"- Blocker: {pid_comm.get('pid_comm_blocker_reason')}",
        "",
        "## Recommendation",
        "",
        str(payload.get("recommended_next_backend_work", "")),
        "",
    ]
    return "\n".join(lines)


def _candidate_row(
    candidate_id: object,
    role: object,
    status: object,
    execution_implemented: object,
    top_level_route: object,
    blocker: object,
) -> str:
    return f"| {candidate_id} | {role} | {status} | {bool(execution_implemented)} | {bool(top_level_route)} | {blocker or ''} |"


def _csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value
