"""Small, target-neutral execution contracts for tensor-network runs."""

from quantum_bench.execution.contracts import (
    BackendFacts,
    CpuCompileRequest,
    CpuPlan,
    ExecutionFailure,
    ExecutionPlan,
    ExecutionResult,
    NumericMode,
    RunContext,
    Target,
    TimingBreakdown,
    UnsupportedExecution,
    UpmemCompileRequest,
    UpmemNodeWorkPlan,
    UpmemPlan,
    UpmemRuntimeResources,
    UpmemTopology,
    canonical_serialize,
    execution_plan_hash,
    validate_execution_plan,
    validate_execution_result,
    validate_transfer_bytes,
    validate_upmem_runtime_resources,
)
from quantum_bench.execution.compiler import compile_cpu, compile_execution, compile_upmem
from quantum_bench.execution.cpu import run_cpu
from quantum_bench.execution.runner import execute
from quantum_bench.execution.upmem import run_upmem
from quantum_bench.execution.numeric import contract_node, reduce_values

__all__ = [
    "CpuCompileRequest",
    "CpuPlan",
    "BackendFacts",
    "ExecutionFailure",
    "ExecutionPlan",
    "ExecutionResult",
    "NumericMode",
    "RunContext",
    "Target",
    "TimingBreakdown",
    "UnsupportedExecution",
    "UpmemCompileRequest",
    "UpmemNodeWorkPlan",
    "UpmemPlan",
    "UpmemRuntimeResources",
    "UpmemTopology",
    "canonical_serialize",
    "execution_plan_hash",
    "validate_execution_plan",
    "validate_execution_result",
    "validate_transfer_bytes",
    "validate_upmem_runtime_resources",
    "compile_cpu",
    "compile_execution",
    "compile_upmem",
    "run_cpu",
    "run_upmem",
    "execute",
    "contract_node",
    "reduce_values",
]
