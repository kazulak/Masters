from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from quantum_bench.core.records import JsonDict, to_jsonable
from quantum_bench.targets.upmem.environment import (
    DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    CommandExecutionRecord,
    UpmemSdkDiscovery,
    discover_simplepim_source,
    discover_upmem_sdk,
    run_command,
)


EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION = "external_pim_libraries_v1"

CapabilityStatus = Literal["available", "unavailable", "build_failed", "not_checked", "unsupported", "blocked"]
PathLookup = Callable[[str], str | None]
CommandRunner = Callable[..., CommandExecutionRecord]

_TEXT_EXTENSIONS = {
    "",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hh",
    ".hpp",
    ".py",
    ".md",
    ".txt",
    ".mk",
    ".cmake",
}
_SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".h", ".hh", ".hpp", ".py"}
_SKIP_DIRS = {".git", "__pycache__", "build", "bin", "obj", ".cache"}
_MAX_SCAN_FILES = 400
_MAX_SCAN_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class CapabilityEvidence:
    name: str
    evidence_detected: bool
    capability_proven: bool
    evidence_paths: tuple[str, ...] = ()
    detection_method: str = "bounded_source_marker_scan"
    evidence_kind: str = "source_marker"

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class ExternalPimCandidate:
    candidate_id: str
    candidate_role: str
    candidate_execution_classes: tuple[str, ...]
    status: CapabilityStatus
    execution_implemented: bool
    top_level_benchmark_route: bool
    blocker_reason: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class NativeSdkControlReport:
    schema_version: str
    candidate_id: str
    status: CapabilityStatus
    upmem_sdk_detected: bool
    upmem_sdk_home: str | None
    control_baseline: bool
    fallback_candidate: bool
    implemented_scope: tuple[str, ...]
    top_level_benchmark_route: bool
    tools: JsonDict

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class SimplePimCapabilityReport:
    schema_version: str
    status: CapabilityStatus
    simplepim_detected: bool
    simplepim_home: str | None
    simplepim_source: str
    management_api: CapabilityEvidence
    communication_api: CapabilityEvidence
    map_zip_reduce_api: CapabilityEvidence
    int8: CapabilityEvidence
    int32_accumulation: CapabilityEvidence
    mram_wram_control: CapabilityEvidence
    ready_gemm_primitive_detected: bool
    ready_gemm_evidence_paths: tuple[str, ...]
    ready_gemm_capability_proven: bool
    simplepim_gemm_ready: bool
    simplepim_blocker_reason: str | None
    candidate: ExternalPimCandidate

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class PidCommCapabilityReport:
    schema_version: str
    status: CapabilityStatus
    pid_comm_detected: bool
    pid_comm_home: str | None
    pid_comm_source: str
    collective_api: CapabilityEvidence
    broadcast: CapabilityEvidence
    scatter: CapabilityEvidence
    gather: CapabilityEvidence
    reduce: CapabilityEvidence
    allreduce: CapabilityEvidence
    collective_capability_proven: bool
    build_check_status: CapabilityStatus
    build_check: JsonDict | None
    requires_hardware_status: str
    l3_candidate_operations: tuple[str, ...]
    pid_comm_blocker_reason: str | None
    candidate: ExternalPimCandidate

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


@dataclass(frozen=True)
class ExternalPimLibrariesReport:
    schema_version: str
    status: CapabilityStatus
    native_sdk_control: NativeSdkControlReport
    simplepim: SimplePimCapabilityReport
    pid_comm: PidCommCapabilityReport
    l1_l2_compute_backend_candidates: tuple[ExternalPimCandidate, ...]
    l3_communication_backend_candidates: tuple[ExternalPimCandidate, ...]
    recommended_next_backend_work: str
    metadata: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return to_jsonable(self)


def build_external_pim_libraries_report(
    root_dir: Path,
    *,
    simplepim_home: str | None = None,
    pid_comm_home: str | None = None,
    check_pid_comm_build: bool = False,
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    path_lookup: PathLookup | None = None,
    command_runner: CommandRunner = run_command,
) -> ExternalPimLibrariesReport:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    probe_env = env if env is not None else os.environ
    sdk = discover_upmem_sdk(
        env=probe_env,
        path_lookup=path_lookup,
        command_runner=command_runner,
        timeout_seconds=timeout_seconds,
    )
    native = _native_sdk_report(sdk)
    simplepim = inspect_simplepim_capability(root_dir, simplepim_home=simplepim_home, env=probe_env)
    pid_comm = inspect_pid_comm_capability(
        root_dir,
        pid_comm_home=pid_comm_home,
        check_build=check_pid_comm_build,
        timeout_seconds=timeout_seconds,
        env=probe_env,
        command_runner=command_runner,
    )
    status = _overall_status(native, simplepim, pid_comm)
    recommended = _recommend_next_work(native, simplepim, pid_comm)
    return ExternalPimLibrariesReport(
        schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
        status=status,
        native_sdk_control=native,
        simplepim=simplepim,
        pid_comm=pid_comm,
        l1_l2_compute_backend_candidates=(native_to_candidate(native), simplepim.candidate),
        l3_communication_backend_candidates=(
            ExternalPimCandidate(
                candidate_id="native_host_mediated_candidate",
                candidate_role="internal_l3_communication_candidate",
                candidate_execution_classes=("L3_MULTI_DPU",),
                status="not_checked",
                execution_implemented=False,
                top_level_benchmark_route=False,
                blocker_reason="host_mediated_l3_communication_not_implemented",
            ),
            pid_comm.candidate,
        ),
        recommended_next_backend_work=recommended,
        metadata={
            "providers_executed": False,
            "upmem_kernels_executed": False,
            "simplepim_kernel_executed": False,
            "pid_comm_collective_executed": False,
            "evidence_is_not_capability_proof": True,
        },
    )


def inspect_simplepim_capability(
    root_dir: Path,
    *,
    simplepim_home: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SimplePimCapabilityReport:
    discovery = discover_simplepim_source(root_dir, simplepim_home_override=simplepim_home, env=env)
    if not discovery.simplepim_detected or not discovery.simplepim_home:
        candidate = ExternalPimCandidate(
            candidate_id="simplepim_dense_candidate",
            candidate_role="internal_l1_l2_compute_candidate",
            candidate_execution_classes=("L1_WRAM", "L2_SINGLE_DPU_MRAM"),
            status="unavailable",
            execution_implemented=False,
            top_level_benchmark_route=False,
            blocker_reason="simplepim_source_unavailable",
        )
        return SimplePimCapabilityReport(
            schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
            status="unavailable",
            simplepim_detected=False,
            simplepim_home=None,
            simplepim_source=discovery.simplepim_source,
            management_api=_empty_evidence("management_api"),
            communication_api=_empty_evidence("communication_api"),
            map_zip_reduce_api=_empty_evidence("map_zip_reduce_api"),
            int8=_empty_evidence("int8"),
            int32_accumulation=_empty_evidence("int32_accumulation"),
            mram_wram_control=_empty_evidence("mram_wram_control"),
            ready_gemm_primitive_detected=False,
            ready_gemm_evidence_paths=(),
            ready_gemm_capability_proven=False,
            simplepim_gemm_ready=False,
            simplepim_blocker_reason="simplepim_source_unavailable",
            candidate=candidate,
        )

    home = Path(discovery.simplepim_home)
    indexed = _index_text_files(home)
    management = _path_evidence("management_api", home, ("lib/management",))
    communication = _path_evidence("communication_api", home, ("lib/communication",))
    map_zip_reduce = _path_evidence(
        "map_zip_reduce_api",
        home,
        ("lib/processing/map", "lib/processing/zip", "lib/processing/gen_red"),
    )
    int8 = _marker_evidence("int8", indexed, (r"\bint8_t\b", r"\buint8_t\b", r"\bint8\b"))
    int32 = _marker_evidence(
        "int32_accumulation",
        indexed,
        (r"\bint32_t\b", r"\buint32_t\b", r"\baccum(?:ulator|ulation)?\b"),
    )
    mram_wram = _marker_evidence(
        "mram_wram_control",
        indexed,
        (r"\bMRAM\b", r"\bWRAM\b", r"\bmram_read\s*\(", r"\bmram_write\s*\("),
    )
    ready_gemm_paths = _ready_gemm_evidence_paths(indexed)
    ready_gemm = bool(ready_gemm_paths)
    blocker = None if ready_gemm else "no_ready_gemm_primitive_detected"
    status: CapabilityStatus = "available" if discovery.simplepim_detected else "unavailable"
    candidate_status: CapabilityStatus = "blocked" if blocker else "available"
    candidate = ExternalPimCandidate(
        candidate_id="simplepim_dense_candidate",
        candidate_role="internal_l1_l2_compute_candidate",
        candidate_execution_classes=("L1_WRAM", "L2_SINGLE_DPU_MRAM"),
        status=candidate_status,
        execution_implemented=False,
        top_level_benchmark_route=False,
        blocker_reason=blocker,
        metadata={"ready_gemm_evidence_detected": ready_gemm},
    )
    return SimplePimCapabilityReport(
        schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
        status=status,
        simplepim_detected=True,
        simplepim_home=str(home),
        simplepim_source=discovery.simplepim_source,
        management_api=management,
        communication_api=communication,
        map_zip_reduce_api=map_zip_reduce,
        int8=int8,
        int32_accumulation=int32,
        mram_wram_control=mram_wram,
        ready_gemm_primitive_detected=ready_gemm,
        ready_gemm_evidence_paths=tuple(ready_gemm_paths),
        ready_gemm_capability_proven=False,
        simplepim_gemm_ready=False,
        simplepim_blocker_reason=blocker or "ready_gemm_primitive_detected_but_not_proven",
        candidate=candidate,
    )


def inspect_pid_comm_capability(
    root_dir: Path,
    *,
    pid_comm_home: str | None = None,
    check_build: bool = False,
    timeout_seconds: float = DEFAULT_UPMEM_ENV_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    command_runner: CommandRunner = run_command,
) -> PidCommCapabilityReport:
    discovery = discover_pid_comm_source(root_dir, pid_comm_home=pid_comm_home, env=env)
    if not discovery["detected"]:
        candidate = ExternalPimCandidate(
            candidate_id="pid_comm_candidate",
            candidate_role="internal_l3_communication_candidate",
            candidate_execution_classes=("L3_MULTI_DPU",),
            status="unavailable",
            execution_implemented=False,
            top_level_benchmark_route=False,
            blocker_reason="pid_comm_source_unavailable",
        )
        return PidCommCapabilityReport(
            schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
            status="unavailable",
            pid_comm_detected=False,
            pid_comm_home=None,
            pid_comm_source=discovery["source"],
            collective_api=_empty_evidence("collective_api"),
            broadcast=_empty_evidence("broadcast"),
            scatter=_empty_evidence("scatter"),
            gather=_empty_evidence("gather"),
            reduce=_empty_evidence("reduce"),
            allreduce=_empty_evidence("allreduce"),
            collective_capability_proven=False,
            build_check_status="not_checked",
            build_check=None,
            requires_hardware_status="unknown",
            l3_candidate_operations=(
                "broadcast_a_b_tiles",
                "scatter_task_partitions",
                "gather_c_tiles",
                "reduce_partial_c_tiles",
            ),
            pid_comm_blocker_reason="pid_comm_source_unavailable",
            candidate=candidate,
        )

    home = Path(str(discovery["home"]))
    indexed = _index_text_files(home)
    broadcast = _marker_evidence("broadcast", indexed, (r"\bbroadcast\b",))
    scatter = _marker_evidence("scatter", indexed, (r"\bscatter\b",))
    gather = _marker_evidence("gather", indexed, (r"\bgather\b",))
    reduce = _marker_evidence("reduce", indexed, (r"\breduce\b", r"\breduction\b"))
    allreduce = _marker_evidence("allreduce", indexed, (r"\ballreduce\b", r"\ball-reduce\b", r"\ball_reduce\b"))
    collective_paths = sorted(
        {
            *broadcast.evidence_paths,
            *scatter.evidence_paths,
            *gather.evidence_paths,
            *reduce.evidence_paths,
            *allreduce.evidence_paths,
        }
    )
    collective = CapabilityEvidence(
        name="collective_api",
        evidence_detected=bool(collective_paths),
        capability_proven=False,
        evidence_paths=tuple(collective_paths),
        detection_method="bounded_source_marker_scan",
        evidence_kind="source_marker",
    )
    build_status, build_record = _pid_comm_build_check(home, check_build, timeout_seconds, command_runner)
    blocker = _pid_comm_blocker(collective, build_status)
    candidate_status: CapabilityStatus = "blocked" if blocker else "available"
    candidate = ExternalPimCandidate(
        candidate_id="pid_comm_candidate",
        candidate_role="internal_l3_communication_candidate",
        candidate_execution_classes=("L3_MULTI_DPU",),
        status=candidate_status,
        execution_implemented=False,
        top_level_benchmark_route=False,
        blocker_reason=blocker,
        metadata={"collective_evidence_detected": collective.evidence_detected},
    )
    return PidCommCapabilityReport(
        schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
        status="available",
        pid_comm_detected=True,
        pid_comm_home=str(home),
        pid_comm_source=discovery["source"],
        collective_api=collective,
        broadcast=broadcast,
        scatter=scatter,
        gather=gather,
        reduce=reduce,
        allreduce=allreduce,
        collective_capability_proven=False,
        build_check_status=build_status,
        build_check=build_record,
        requires_hardware_status=_requires_hardware_status(indexed),
        l3_candidate_operations=(
            "broadcast_a_b_tiles",
            "scatter_task_partitions",
            "gather_c_tiles",
            "reduce_partial_c_tiles",
        ),
        pid_comm_blocker_reason=blocker,
        candidate=candidate,
    )


def discover_pid_comm_source(
    root_dir: Path,
    *,
    pid_comm_home: str | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    probe_env = env if env is not None else os.environ
    extern = root_dir.parent / "legacy" / "extern"
    candidates: tuple[tuple[str, str | None], ...] = (
        ("cli", pid_comm_home),
        ("environment", probe_env.get("PID_COMM_HOME")),
        ("repo_fallback", str(extern / "PID-Comm")),
        ("repo_fallback", str(extern / "PID_Comm")),
        ("repo_fallback", str(extern / "PIDComm")),
        ("repo_fallback", str(extern / "pid-comm")),
        ("repo_fallback", str(extern / "pid_comm")),
    )
    for source, raw_path in candidates:
        cleaned = _clean_path(raw_path)
        if cleaned and Path(cleaned).is_dir():
            return {"detected": True, "home": cleaned, "source": source}
    return {"detected": False, "home": None, "source": "none"}


def native_to_candidate(native: NativeSdkControlReport) -> ExternalPimCandidate:
    return ExternalPimCandidate(
        candidate_id="native_upmem_sdk_control",
        candidate_role="control_baseline_and_fallback",
        candidate_execution_classes=("L1_WRAM", "L2_SINGLE_DPU_MRAM"),
        status=native.status,
        execution_implemented=True,
        top_level_benchmark_route=False,
        blocker_reason=None if native.status == "available" else "upmem_sdk_unavailable",
        metadata={"implemented_scope": list(native.implemented_scope)},
    )


def external_libs_report_rows(report: ExternalPimLibrariesReport) -> list[JsonDict]:
    return [
        {
            "schema_version": report.schema_version,
            "component": "native_upmem_sdk_control",
            "candidate_id": "native_upmem_sdk_control",
            "candidate_role": "control_baseline_and_fallback",
            "status": report.native_sdk_control.status,
            "execution_implemented": True,
            "top_level_benchmark_route": False,
            "blocker_reason": "" if report.native_sdk_control.status == "available" else "upmem_sdk_unavailable",
            "evidence_detected": report.native_sdk_control.upmem_sdk_detected,
            "capability_proven": report.native_sdk_control.upmem_sdk_detected,
            "evidence_paths": (),
        },
        {
            "schema_version": report.schema_version,
            "component": "simplepim",
            "candidate_id": report.simplepim.candidate.candidate_id,
            "candidate_role": report.simplepim.candidate.candidate_role,
            "status": report.simplepim.candidate.status,
            "execution_implemented": False,
            "top_level_benchmark_route": False,
            "blocker_reason": report.simplepim.simplepim_blocker_reason,
            "evidence_detected": report.simplepim.simplepim_detected,
            "capability_proven": report.simplepim.ready_gemm_capability_proven,
            "evidence_paths": report.simplepim.ready_gemm_evidence_paths,
        },
        {
            "schema_version": report.schema_version,
            "component": "pid_comm",
            "candidate_id": report.pid_comm.candidate.candidate_id,
            "candidate_role": report.pid_comm.candidate.candidate_role,
            "status": report.pid_comm.candidate.status,
            "execution_implemented": False,
            "top_level_benchmark_route": False,
            "blocker_reason": report.pid_comm.pid_comm_blocker_reason,
            "evidence_detected": report.pid_comm.collective_api.evidence_detected,
            "capability_proven": report.pid_comm.collective_capability_proven,
            "evidence_paths": report.pid_comm.collective_api.evidence_paths,
        },
    ]


def candidate_status_payload_from_report(report: ExternalPimLibrariesReport | JsonDict | None) -> JsonDict:
    if report is None:
        return _not_checked_candidate_payload()
    payload = report.to_json_dict() if isinstance(report, ExternalPimLibrariesReport) else dict(report)
    native = dict(payload.get("native_sdk_control") or {})
    simplepim = dict(payload.get("simplepim") or {})
    pid_comm = dict(payload.get("pid_comm") or {})
    l1_l2_candidates = [
        dict(candidate).get("candidate_id")
        for candidate in payload.get("l1_l2_compute_backend_candidates", ())
        if isinstance(candidate, dict)
    ]
    l3_candidates = [
        dict(candidate).get("candidate_id")
        for candidate in payload.get("l3_communication_backend_candidates", ())
        if isinstance(candidate, dict)
    ]
    return {
        "l1_l2_compute_backend_candidates": tuple(candidate for candidate in l1_l2_candidates if candidate),
        "l3_communication_backend_candidates": tuple(candidate for candidate in l3_candidates if candidate),
        "simplepim_candidate_status": dict(simplepim.get("candidate") or {}).get("status", "not_checked"),
        "pid_comm_candidate_status": dict(pid_comm.get("candidate") or {}).get("status", "not_checked"),
        "native_sdk_control_status": native.get("status", "not_checked"),
        "recommended_next_backend_work": payload.get("recommended_next_backend_work", "not_checked"),
    }


def _not_checked_candidate_payload() -> JsonDict:
    return {
        "l1_l2_compute_backend_candidates": ("native_upmem_sdk_control", "simplepim_dense_candidate"),
        "l3_communication_backend_candidates": ("native_host_mediated_candidate", "pid_comm_candidate"),
        "simplepim_candidate_status": "not_checked",
        "pid_comm_candidate_status": "not_checked",
        "native_sdk_control_status": "not_checked",
        "recommended_next_backend_work": "run upmem-external-libs-check to populate external PIM candidate status",
    }


def _native_sdk_report(sdk: UpmemSdkDiscovery) -> NativeSdkControlReport:
    return NativeSdkControlReport(
        schema_version=EXTERNAL_PIM_LIBRARIES_SCHEMA_VERSION,
        candidate_id="native_upmem_sdk_control",
        status="available" if sdk.upmem_sdk_detected else "unavailable",
        upmem_sdk_detected=sdk.upmem_sdk_detected,
        upmem_sdk_home=sdk.upmem_sdk_home,
        control_baseline=True,
        fallback_candidate=True,
        implemented_scope=(
            "L1_WRAM task-level simulator dense bridge subset",
            "L2_SINGLE_DPU_MRAM task-level simulator real-valued dense bridge subset",
        ),
        top_level_benchmark_route=False,
        tools={tool.name: tool.to_json_dict() for tool in sdk.tools},
    )


def _index_text_files(root: Path) -> dict[str, str]:
    indexed: dict[str, str] = {}
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if path.suffix not in _TEXT_EXTENSIONS and path.name not in {"Makefile", "CMakeLists.txt"}:
            continue
        try:
            if path.stat().st_size > _MAX_SCAN_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        indexed[_as_posix(path.relative_to(root))] = text
        count += 1
        if count >= _MAX_SCAN_FILES:
            break
    return indexed


def _marker_evidence(name: str, indexed: dict[str, str], patterns: Sequence[str]) -> CapabilityEvidence:
    regexes = tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)
    paths = sorted(
        path
        for path, text in indexed.items()
        if _is_source_like(path) and any(regex.search(text) for regex in regexes)
    )
    return CapabilityEvidence(
        name=name,
        evidence_detected=bool(paths),
        capability_proven=False,
        evidence_paths=tuple(paths),
        detection_method="bounded_source_marker_scan",
        evidence_kind="source_marker",
    )


def _path_evidence(name: str, root: Path, rel_paths: Sequence[str]) -> CapabilityEvidence:
    paths = tuple(rel for rel in rel_paths if (root / rel).exists())
    return CapabilityEvidence(
        name=name,
        evidence_detected=bool(paths),
        capability_proven=False,
        evidence_paths=paths,
        detection_method="path_structure_scan",
        evidence_kind="source_tree_path",
    )


def _ready_gemm_evidence_paths(indexed: dict[str, str]) -> list[str]:
    function_pattern = re.compile(
        r"\b(?:gemm|matmul|matrix_multiply|matrix_multiplication)\s*\(",
        flags=re.IGNORECASE,
    )
    target_pattern = re.compile(
        r"^\s*(?:[A-Za-z0-9_.-]*?(?:gemm|matmul)[A-Za-z0-9_.-]*?)\s*:",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    paths: list[str] = []
    for path, text in indexed.items():
        if _is_source_like(path) and function_pattern.search(text):
            paths.append(path)
        elif Path(path).name == "Makefile" and target_pattern.search(text):
            paths.append(path)
        elif re.search(r"(?:^|/)(?:gemm|matmul)(?:/|\.|$)", path, flags=re.IGNORECASE):
            paths.append(path)
    return sorted(set(paths))


def _empty_evidence(name: str) -> CapabilityEvidence:
    return CapabilityEvidence(name=name, evidence_detected=False, capability_proven=False)


def _pid_comm_build_check(
    home: Path,
    check_build: bool,
    timeout_seconds: float,
    command_runner: CommandRunner,
) -> tuple[CapabilityStatus, JsonDict | None]:
    if not check_build:
        return "not_checked", None
    if not (home / "Makefile").exists():
        return "unsupported", {"attempted": False, "reason": "makefile_not_found"}
    result = command_runner(("make", "-n"), cwd=home, cwd_label=str(home), timeout_seconds=timeout_seconds)
    status: CapabilityStatus = "available" if result.status == "passed" else "build_failed"
    return status, {
        "attempted": True,
        "command": result.command,
        "return_code": result.return_code,
        "timed_out": result.timed_out,
        "stdout_snippet": result.stdout_snippet,
        "stderr_snippet": result.stderr_snippet,
        "status": result.status,
    }


def _pid_comm_blocker(collective: CapabilityEvidence, build_status: CapabilityStatus) -> str | None:
    if build_status == "build_failed":
        return "pid_comm_build_dry_run_failed"
    if build_status == "unsupported":
        return "pid_comm_build_system_not_supported"
    if not collective.evidence_detected:
        return "pid_comm_collective_evidence_not_detected"
    return "pid_comm_collective_capability_not_proven"


def _requires_hardware_status(indexed: dict[str, str]) -> str:
    text = "\n".join(indexed.values()).lower()
    if "dpu_backend" in text and "simulator" in text:
        return "simulator_evidence_detected"
    if "hardware" in text or "dimm" in text:
        return "hardware_evidence_detected"
    return "unknown"


def _overall_status(
    native: NativeSdkControlReport,
    simplepim: SimplePimCapabilityReport,
    pid_comm: PidCommCapabilityReport,
) -> CapabilityStatus:
    if native.status == "unavailable" and simplepim.status == "unavailable" and pid_comm.status == "unavailable":
        return "unavailable"
    if pid_comm.build_check_status == "build_failed":
        return "build_failed"
    if simplepim.candidate.status == "blocked" or pid_comm.candidate.status == "blocked":
        return "blocked"
    return "available"


def _recommend_next_work(
    native: NativeSdkControlReport,
    simplepim: SimplePimCapabilityReport,
    pid_comm: PidCommCapabilityReport,
) -> str:
    if native.status != "available":
        return "restore UPMEM SDK detection before external library integration"
    if not simplepim.ready_gemm_primitive_detected:
        if not pid_comm.pid_comm_detected:
            return "keep native SDK L1/L2 as control baseline; provide PID_COMM_HOME or plan native host-mediated L3 communication"
        return "keep native SDK L1/L2 as control baseline; evaluate PID-Comm L3 collective feasibility before SimplePIM GEMM work"
    if not simplepim.ready_gemm_capability_proven:
        return "prototype a bounded SimplePIM GEMM proof before replacing native SDK L1/L2 control paths"
    return "compare SimplePIM L1/L2 candidate against native SDK control on the dense bridge harness"


def _is_source_like(path: str) -> bool:
    suffix = Path(path).suffix
    return suffix in _SOURCE_EXTENSIONS or Path(path).name in {"Makefile", "CMakeLists.txt"}


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _clean_path(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())
