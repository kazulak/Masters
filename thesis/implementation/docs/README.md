# Documentation Index

This directory contains technical documentation, architectural specifications,
benchmark policies, and active hardware runbooks for the thesis implementation.

> **Stale snapshot warning:** tracked `thesis_results/` snapshots are historical
> evidence capsules. Regenerate the M5 report from the exact normalized input;
> do not treat a snapshot as current benchmark data without checking its
> provenance and report timestamp.

## Core Reference Index

* **Root Command / Milestone Entrypoint:** [README.md](../README.md) is the current command and milestone entrypoint.
* **Target Architecture:** [slr_architecture_implementation_roadmap.md](slr_architecture_implementation_roadmap.md) is the target architecture.
* **M4/M5 Development Record:** [m4_m5_physical_acceptance.md](m4_m5_physical_acceptance.md) separates historical M4/M5.1/M5.2 ETH acceptance from the pending M5 v3 lane.
* **M5 v3 Contract:** [upmem_m5_benchmark_contract.md](upmem_m5_benchmark_contract.md) defines admission, reporting, and claim limits for the additive execution-plan-v3 route.
* **Benchmark Policy:** [research_benchmark_methodology.md](research_benchmark_methodology.md) is the benchmark policy.
* **Evidence Workflow:** [evidence_workflow.md](evidence_workflow.md) is the evidence workflow.
* **Implementation Audit:** [thesis_implementation_audit.md](thesis_implementation_audit.md) tracks SLR compliance and implementation status.

## Active Hardware Runbooks

* [upmem_hardware_frontier_m3_1_runbook.md](upmem_hardware_frontier_m3_1_runbook.md): M3.1 Frontier Runbook
* [upmem_hardware_simplepim_rank1_m4_2_runbook.md](upmem_hardware_simplepim_rank1_m4_2_runbook.md): M4.2 SimplePIM rank-1 qualification runbook
* [upmem_hardware_simplepim_taskgraph_m4_3_runbook.md](upmem_hardware_simplepim_taskgraph_m4_3_runbook.md): M4.3 physical SimplePIM TaskGraph adapter runbook
* [upmem_hardware_sliced_resident_m2_3_runbook.md](upmem_hardware_sliced_resident_m2_3_runbook.md): M2.3 physical two-DPU runbook
* [upmem_hardware_sliced_resident_mvp_runbook.md](upmem_hardware_sliced_resident_mvp_runbook.md): M2 sliced-resident MVP runbook
* [upmem_hardware_taskgraph_m4_1_runbook.md](upmem_hardware_taskgraph_m4_1_runbook.md): M4.1 physical differential runbook
* [upmem_hardware_taskgraph_resident_runbook.md](upmem_hardware_taskgraph_resident_runbook.md): UPMEM MRAM-resident TaskGraph runbook
* [upmem_provider_qualification_runbook.md](upmem_provider_qualification_runbook.md): UPMEM provider qualification runbook

## M5 v3 Status

The additive `upmem-hw-m5-plan` lane is locally hardware-free validated. The
exact command below prepares its configured plan set, preserves unsupported
cases, reports failures explicitly, and performs no DPU allocation or launch.

```bash
UPMEM_HW_M5_DPU_COUNTS=3 UPMEM_HW_M5_TASKLETS=3 make upmem-hw-m5-plan
```

The active target is one-rank multi-DPU execution of one contraction. The route
supports output/contracted-axis partitioning, float32 and per-task resident
int8 requantization, real highest-work contractions, and synthetic strong/weak
diagnostics. Both numeric modes use float32 MRAM transport. Physical ETH
execution is pending. The future execution command is:

```bash
UPMEM_HW_RANK_PATH=/dev/dpu_rank1 \
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make upmem-hw-m5
```

No physical acceptance, performance, or scaling claim is allowed before that
lane is executed and reviewed. For M5 v3, SimplePIM is
`initialization_binary_and_management_state_only`; allocation, transfer, and
launch use raw synchronous UPMEM SDK calls. The thesis-owned C kernel and host
`float64` reduction are outside SimplePIM compute operators. This lane is not
full distributed TaskGraph execution.
