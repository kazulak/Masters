from __future__ import annotations

from quantum_bench.core.records import JsonDict, to_jsonable


def summary_dpu_invocations(summary: JsonDict) -> int:
    return int(summary.get("dpu_program_executed_task_count", 0) or 0)


def strict_upmem_runtime_assertions(summary: JsonDict) -> JsonDict:
    task_count = int(summary.get("total_tasks", 0) or 0)
    dpu_invocations = summary_dpu_invocations(summary)
    cpu_fallback_task_count = 1 if bool(summary.get("cpu_fallback_used", False)) else 0
    checks = {
        "task_count_positive": task_count > 0,
        "upmem_task_count_matches_task_count": dpu_invocations == task_count,
        "cpu_fallback_task_count_zero": cpu_fallback_task_count == 0,
        "dpu_program_invocations_positive": dpu_invocations > 0,
        "upmem_program_executed": bool(summary.get("dpu_program_executed_all_tasks", False)),
        "native_sdk_control_path": summary.get("native_sdk_control_path") is True,
        "simplepim_api_used_false": summary.get("simplepim_api_used") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return to_jsonable(
        {
            "status": "passed" if not failed else "failed",
            "reason": None if not failed else f"strict_upmem_runtime_assertion_failed:{','.join(failed)}",
            "task_count": task_count,
            "upmem_task_count": dpu_invocations,
            "cpu_fallback_task_count": cpu_fallback_task_count,
            "dpu_program_invocations": dpu_invocations,
            "upmem_program_executed": bool(summary.get("dpu_program_executed_all_tasks", False)),
            "checks": checks,
        }
    )


def upmem_sdk_simulator_preflight_payload(status: str, reason: str | None, *, summary: JsonDict | None = None) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": "upmem_sdk_simulator_preflight_v1",
            "status": status,
            "reason": reason,
            "required_conditions": {
                "upmem_sdk_present": status == "passed",
                "simulator_mode_available": status == "passed",
                "dpu_program_build_load_works": status == "passed",
                "real_sdk_simulator_dpu_program_executed": status == "passed",
            },
            "summary": summary or {},
        }
    )

