"""Thin M6a binding for one bounded physical two-DPU frontier study."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from quantum_bench.bench import upmem_simplepim_taskgraph as engine
from quantum_bench.targets.upmem.execution_plan_v1 import (
    ExecutionPlan,
    PLACEMENT_FRONTIER,
)


SUITE_ID = "upmem_hardware_frontier_m6a"
SCHEMA_VERSION = "upmem_hardware_frontier_m6a_v1"
PROFILE_VERSION = "hardware_frontier_m6a_v1"
ROUTE_ID = "upmem_tn_hardware_frontier_m6a"
BACKEND_ID = "upmem_hardware_frontier_m6a"
QASM_PATH = "configs/circuits/upmem_m6a/two_qubit_frontier_2_2_1.qasm"

M6A_CONTRACT = engine.RouteStudyContract(
    suite_id=SUITE_ID,
    schema_version=SCHEMA_VERSION,
    profile_version=PROFILE_VERSION,
    route_id=ROUTE_ID,
    backend_id=BACKEND_ID,
    claim_boundary="physical_frontier_functionality_only_no_scaling_claim",
    route_label="upmem_hw_frontier_m6a",
    execution_scope="bounded_two_dpu_frontier_taskgraph_m6a",
    benchmark_role="physical_two_dpu_frontier_taskgraph_functionality",
    placements=(PLACEMENT_FRONTIER,),
    case_id="two_qubit_frontier_2_2_1",
    qasm_path=QASM_PATH,
    warmups=1,
    repeats=5,
    tolerance=1.0e-6,
    max_logical_tasks=8,
    max_waves=8,
    max_frontier_width=2,
    expected_path=((0, 2), (0, 1), (0, 2), (0, 1), (0, 1)),
    expected_wave_widths=(2, 2, 1),
    expected_dpu_task_counts=(3, 2),
    expected_assignment_dpu_ids=(0, 1, 0, 1, 0),
    expected_transfer_edges=(
        engine.TransferEdgeExpectation(
            producer_task_id="task_3",
            consumer_task_id="task_4",
            producer_dpu_id=1,
            consumer_dpu_id=0,
            slot_id=9,
            element_count=2,
            transfer_bytes=8,
        ),
    ),
    require_rank_evidence=True,
    allow_slot_reuse=False,
    include_failures_in_normalized_records=True,
)


def load_upmem_hardware_frontier_m6a_suite(path: Path) -> dict[str, Any]:
    return engine.load_route_study_suite(path, M6A_CONTRACT)


def prepare_upmem_hardware_frontier_m6a(
    root_dir: Path,
    *,
    suite_path: Path,
    build: bool = False,
    environment: Mapping[str, str] | None = None,
    native_target: engine.Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    return engine.prepare_route_study(
        root_dir,
        suite_path=suite_path,
        contract=M6A_CONTRACT,
        build=build,
        environment=environment,
        native_target=native_target,
        plan_compiler=plan_compiler,
    )


def run_upmem_hardware_frontier_m6a(
    root_dir: Path,
    *,
    suite_path: Path,
    environment: Mapping[str, str] | None = None,
    native_target: engine.Block2NativeTarget | None = None,
    plan_compiler: Callable[..., ExecutionPlan] | None = None,
) -> dict[str, Any]:
    return engine.execute_route_study(
        root_dir,
        suite_path=suite_path,
        contract=M6A_CONTRACT,
        environment=environment,
        native_target=native_target,
        plan_compiler=plan_compiler,
    )


# CLI dispatchers use these standard short names; explicit long names remain
# available for targeted tests and other milestone modules.
prepare = prepare_upmem_hardware_frontier_m6a
execute = run_upmem_hardware_frontier_m6a


__all__ = [
    "BACKEND_ID",
    "M6A_CONTRACT",
    "PROFILE_VERSION",
    "QASM_PATH",
    "ROUTE_ID",
    "SCHEMA_VERSION",
    "SUITE_ID",
    "load_upmem_hardware_frontier_m6a_suite",
    "execute",
    "prepare",
    "prepare_upmem_hardware_frontier_m6a",
    "run_upmem_hardware_frontier_m6a",
]
