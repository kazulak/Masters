# Audited & Approved Implementation Plan: Milestone M4.6c

> **Status:** AGREED - PLAN APPROVED BY AUDITOR & PLANNER AGENTS  
> **Target:** Milestone M4.6c (Physical & Simulator Multi-Tasklet Evidence Collection & Benchmark Scaling Study)  
> **Source Document:** [docs/m4_6c_implementation_plan.md](file:///home/tom/repos/Masters/thesis/implementation/docs/m4_6c_implementation_plan.md)

---

## 1. Executive Summary & Design Invariants

Micro-Step 4.6c completes the Multi-Tasklet execution milestone by building the evidence collection harness. It measures the physical scaling of UPMEM generic DPU contraction across different thread counts ($NR\_TASKLETS \in \{1, 2, 4, 8, 11, 16\}$) and calibrates the SLR Section 9.3 theoretical PIM Cost Model ($C_{\text{PIM}}$) using empirical cycles.

### Key Design Invariants:
1. **KISS Compliance**: Reuses the existing `benchmark_result_artifact_v1` schema without adding new complex logging frameworks. Uses existing pytest fixtures.
2. **Benchmark Sweep**: Sweeps `NR_TASKLETS` $\in \{1, 2, 4, 8, 11, 16\}$ across canonical quantum circuits (QRNG 4-qubit, Bell 2-qubit, GHZ 4-qubit).
3. **Derived Scaling Metrics**: Computes multi-tasklet speedup ratios $S(N) = T(1) / T(N)$ and parallel efficiency $E(N) = S(N) / N$.
4. **Hardware vs. Simulator Fallbacks**:
   - `dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None`
5. **Cost Model Calibration ($C_{\text{PIM}}$)**:
   - $C_{\text{PIM}} = (\text{empirical\_dpu\_cycles} / \text{modeled\_estimated\_flops}) \text{ if } (\text{empirical\_dpu\_cycles} > 0 \text{ and } \text{modeled\_estimated\_flops} > 0) \text{ else None}$

---

## 2. Component Diffs & File Changes

### Component 1: Execution Metadata & Summary Payload

#### [MODIFY] `src/quantum_bench/targets/upmem/taskgraph_runtime.py`
- Ensure `tasklets_per_dpu` is explicitly exposed in `executor_config`.
- Propagate aggregate `dpu_run_time_cycles` and transfer metrics (`host_to_dpu_transfer_seconds`, `dpu_to_host_transfer_seconds`) into the `summary` payload.
- Enforce fallback logic: `dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None`.
- Adhere strictly to `benchmark_result_artifact_v1`.

---

### Component 2: Multi-Tasklet Benchmark Execution Fixture

#### [MODIFY] `tests/test_upmem_simplepim_taskgraph_executor.py`
- Add parameterized test suite `test_multi_tasklet_benchmark_scaling`.
- Parameterize `tasklets_per_dpu` over `[1, 2, 4, 8, 11, 16]`.
- Execute taskgraphs across canonical circuits (`QRNG`, `Bell`, `GHZ`).
- Extract `dpu_run_time_cycles` and calculate $S(N) = T(1) / T(N)$ and $E(N) = S(N) / N$.
- Assert that $S(N)$ monotonically increases up to $N = 11$ and validation checks pass.

---

### Component 3: SLR Section 9.3 Cost Model Calibration

#### [MODIFY] `src/quantum_bench/tn/upmem_path_cost_v2.py`
- Collect `estimated_flops` and `mram_dma_window_bytes_model` for executed tasks.
- Calculate $C_{\text{PIM}}$ operational constant safely:
  `C_PIM = (empirical_dpu_cycles / modeled_estimated_flops) if (empirical_dpu_cycles > 0 and modeled_estimated_flops > 0) else None`
- Correlate theoretical network layout costs with empirical physical cycles.

---

## 3. Verification Plan

```bash
# 1. Run the Multi-Tasklet benchmark scaling test suite
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_upmem_simplepim_taskgraph_executor.py -k test_multi_tasklet_benchmark_scaling -v

# 2. Run full repository pytest regression suite
make test
```

All tests must pass cleanly without warning or `ZeroDivisionError`.
