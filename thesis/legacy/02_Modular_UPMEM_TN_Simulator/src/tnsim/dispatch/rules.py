from __future__ import annotations


KNOWN_ROUTES = {
    "cpu_reference": "Implemented tensor-network CPU route using NumPy pairwise contractions.",
    "quest_exact_statevector": "External exact CPU state-vector baseline using QuEST.",
    "gpu_cupy": "Optional future tensor-network GPU route using CuPy.",
    "raw_upmem_dense": "V2 wrapper around the 01_MVP_DenseGEMM UPMEM simulator baseline.",
    "simplepim_default": "Future SimplePIM-first UPMEM provider.",
    "custom_dense": "Future optimized dense UPMEM provider.",
    "sparsep": "Future sparse route for density-threshold experiments.",
    "pidcomm_collective": "Future collective provider for sliced multi-DPU reductions.",
}


def dispatch_task(task: dict, config: dict) -> dict:
    routes = config["execution"]["routes"]
    enabled = list(routes.get("enabled", []))
    forced = routes.get("forced")
    if forced and forced not in enabled:
        raise ValueError(f"Forced route {forced} is not in execution.routes.enabled")

    selected = forced or ("cpu_reference" if "cpu_reference" in enabled else None)
    if selected not in {"cpu_reference", "quest_exact_statevector", "raw_upmem_dense"}:
        raise ValueError(
            f"Route {selected} is not implemented in Stage 1A. "
            "Use cpu_reference, quest_exact_statevector, raw_upmem_dense, "
            "or add the provider behind the shared interface."
        )

    rejected = []
    for route in KNOWN_ROUTES:
        if route == selected:
            continue
        if route not in enabled:
            reason = "disabled_by_config"
        elif route == "quest_exact_statevector":
            reason = "external_full_state_baseline_not_selected"
        elif route == "raw_upmem_dense":
            reason = "upmem_mvp_simulator_baseline_not_selected"
        elif route == "gpu_cupy":
            reason = "optional_gpu_route_not_enabled_in_stage_1a"
        elif route == "sparsep":
            reason = "task_classified_dense"
        elif route == "pidcomm_collective":
            reason = "task_is_not_collective"
        else:
            reason = "provider_not_implemented_in_stage_1a"
        rejected.append({"route": route, "reason": reason})

    reason = "forced_by_config" if forced else "default_cpu_reference_stage_1a"
    if selected == "quest_exact_statevector":
        reason = "forced_external_exact_statevector_baseline"
    if selected == "raw_upmem_dense":
        reason = "forced_raw_upmem_dense_mvp_simulator_baseline"

    return {
        "task_id": task["id"],
        "op_kind": task["op_kind"],
        "candidate_routes": enabled,
        "selected_route": selected,
        "selected_data_format": config["execution"]["data_format"],
        "reason": reason,
        "rejected_routes_with_reasons": rejected,
    }
