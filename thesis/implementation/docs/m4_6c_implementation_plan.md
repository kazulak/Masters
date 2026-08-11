# Audited & Approved Implementation Plan: Milestone M4.6c

> **Historical plan:** M4.6c has since been implemented and physically
> accepted as a bounded development sweep. See
> [m4_m5_physical_acceptance.md](m4_m5_physical_acceptance.md) for the ETH
> commands, run IDs, and claim boundary. The active tasklet set is
> `1/2/4/8/16`; this document is retained as the original design record.

> **Status:** HISTORICAL PROPOSAL - SUPERSEDED BY PHYSICAL DEVELOPMENT ACCEPTANCE
> **Target:** Milestone M4.6c (Physical & Simulator Multi-Tasklet Evidence Collection & Benchmark Scaling Study)  
> **Source Document:** [docs/m4_6c_implementation_plan.md](file:///home/tom/repos/Masters/thesis/implementation/docs/m4_6c_implementation_plan.md)

---

## 1. Executive Summary & Design Invariants

This historical plan proposed an evidence harness for the multi-tasklet route.
The accepted development sweep uses `NR_TASKLETS in {1, 2, 4, 8, 16}` and
does not calibrate the SLR cost model; the observed ratios remain diagnostic.

### Historical Proposed Design Invariants
1. **KISS Compliance**: Reuses the existing `benchmark_result_artifact_v1` schema without adding new complex logging frameworks. Uses existing pytest fixtures.
2. **Benchmark Sweep (historical design)**: The original proposal swept six
   tasklet counts across small circuits; the accepted development sweep uses
   `1/2/4/8/16` across its fixed suite.
3. **Derived diagnostics**: Development-run cycle and host-observed ratios are
   diagnostic only; they are not final scaling or cost-model calibration
   evidence.
4. **Hardware vs. Simulator Fallbacks**:
   - `dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None`
5. **Cost Model Calibration (historical proposal):** The draft proposed
   $C_{\text{PIM}} = \text{cycles} / \text{estimated FLOPs}$. The accepted
   development run does not calibrate the cost model.

---

## 2. Historical Proposed Component Changes

### Component 1: Execution Metadata & Summary Payload

#### [MODIFY] `src/quantum_bench/targets/upmem/taskgraph_runtime.py`
- Ensure `tasklets_per_dpu` is explicitly exposed in `executor_config`.
- Propagate aggregate `dpu_run_time_cycles` and transfer metrics (`host_to_dpu_transfer_seconds`, `dpu_to_host_transfer_seconds`) into the `summary` payload.
- Enforce fallback logic: `dpu_compute_seconds = (dpu_cycles / dpu_freq_hz) if (dpu_cycles > 0 and dpu_freq_hz > 0) else None`.
- Adhere strictly to `benchmark_result_artifact_v1`.

---

### Component 2: Multi-Tasklet Benchmark Fixture (Historical Proposal)

#### [MODIFY] `tests/test_upmem_simplepim_taskgraph_executor.py`
- Add parameterized test suite `test_multi_tasklet_benchmark_scaling`.
- Historical proposal: parameterize `tasklets_per_dpu` over
  `[1, 2, 4, 8, 11, 16]`. The accepted active set is `[1, 2, 4, 8, 16]`.
- Execute taskgraphs across canonical circuits (`QRNG`, `Bell`, `GHZ`).
- Extract `dpu_run_time_cycles` and calculate $S(N) = T(1) / T(N)$ and $E(N) = S(N) / N$.
- Historical proposal: assert monotonic scaling through 11 tasklets. The
  accepted active development run was non-monotonic: the small workloads
  improved through 8 tasklets and declined at 16. This is diagnostic behavior,
  not a scaling guarantee.

---

### Component 3: SLR Section 9.3 Cost Model Calibration (Historical Proposal)

The following work was proposed but is not established by the accepted run;
no hardware cost-model calibration claim is made.

#### [MODIFY] `src/quantum_bench/tn/upmem_path_cost_v2.py`
- Collect `estimated_flops` and `mram_dma_window_bytes_model` for executed tasks.
- Calculate $C_{\text{PIM}}$ operational constant safely:
  `C_PIM = (empirical_dpu_cycles / modeled_estimated_flops) if (empirical_dpu_cycles > 0 and modeled_estimated_flops > 0) else None`
- Correlate theoretical network layout costs with empirical physical cycles.

---

## 3. Historical Verification Plan (Not Active)

The commands below are retained as the original proposal. The accepted active
set and observed ETH runs are documented in
[m4_m5_physical_acceptance.md](m4_m5_physical_acceptance.md).

```bash
# 1. Run the Multi-Tasklet benchmark scaling test suite
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_upmem_simplepim_taskgraph_executor.py -k test_multi_tasklet_benchmark_scaling -v

# 2. Run full repository pytest regression suite
make test
```

All tests must pass cleanly without warning or `ZeroDivisionError`.
