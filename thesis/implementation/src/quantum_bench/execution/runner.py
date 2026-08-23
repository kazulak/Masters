"""Explicit execution dispatch for the functional execution slice."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from quantum_bench.execution.contracts import (
    ExecutionFailure,
    ExecutionPlan,
    ExecutionResult,
    RunContext,
    Target,
)
from quantum_bench.execution.cpu import run_cpu
from quantum_bench.upmem.runtime import run_upmem
from quantum_bench.model import ContractionDAG
def execute(
    plan: ExecutionPlan,
    dag: ContractionDAG,
    inputs: Mapping[str, np.ndarray],
    context: RunContext,
) -> ExecutionResult | ExecutionFailure:
    """Dispatch one compiled plan without fallback.

    ``ExecutionFailure`` represents deterministic dispatch rejection only.
    Malformed requests and native/session failures raise unchanged; experiment
    orchestration records them as failure rows with their original stage.
    """

    match plan.target:
        case Target.CPU:
            return run_cpu(plan, dag, inputs, context)
        case Target.UPMEM:
            if (
                not getattr(plan.payload, "node_order", ())
                and context.target_resources is None
            ):
                return ExecutionFailure(
                    contraction_dag_hash=plan.contraction_dag_hash,
                    target=Target.UPMEM,
                    stage="execution_dispatch",
                    reason="UPMEM execution adapter is not implemented for legacy plans",
                )
            return run_upmem(plan, dag, inputs, context)
        case Target.GPU:
            return ExecutionFailure(
                contraction_dag_hash=plan.contraction_dag_hash,
                target=Target.GPU,
                stage="execution_dispatch",
                reason="GPU execution adapter is not implemented in this slice",
            )
        case _:
            raise TypeError(f"Unsupported execution target: {plan.target!r}")


__all__ = ["execute"]
