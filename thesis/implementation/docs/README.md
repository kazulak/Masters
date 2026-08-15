# Documentation Index

This index separates the documents needed to run and extend the current
research system from historical compatibility material. The main entry point is
the repository [README](../README.md).

## Active References

| Document | Use it for |
| --- | --- |
| [../ARCHITECTURE.md](../ARCHITECTURE.md) | Current ownership, route boundaries, and evidence classes. |
| [PIPELINE_CONTRACT.md](PIPELINE_CONTRACT.md) | Symbol-level stage contract, mutable state, parameters, and hashes. |
| [../THESIS_BENCHMARK_MATRIX.md](../THESIS_BENCHMARK_MATRIX.md) | Required thesis comparisons, measurements, and stop rules. |
| [MILESTONES.md](MILESTONES.md) | Generated truth ledger for milestone implementation and evidence status. |
| [upmem_m5_5_whole_circuit_runbook.md](upmem_m5_5_whole_circuit_runbook.md) | Current physical M5.5 planning, execution, reporting, and claim limits. |
| [research_benchmark_methodology.md](research_benchmark_methodology.md) | Methodology and allowed claims. |
| [evidence_workflow.md](evidence_workflow.md) | Copying ETH runs, report generation, promotion, and cleanup. |
| [slr_architecture_implementation_roadmap.md](slr_architecture_implementation_roadmap.md) | SLR-derived target architecture and remaining work. |

## Current Supporting References

| Document | Scope |
| --- | --- |
| [upmem_m5_benchmark_contract.md](upmem_m5_benchmark_contract.md) | M5 execution-plan compatibility and report-admission contract. |
| [m4_m5_physical_acceptance.md](m4_m5_physical_acceptance.md) | ETH development acceptance record for earlier M4/M5 lanes. |
| [thesis_implementation_audit.md](thesis_implementation_audit.md) | Periodic implementation audit and outstanding risks. |
| [upmem_provider_qualification_runbook.md](upmem_provider_qualification_runbook.md) | External-provider qualification procedure. |
| [gpu_tn_feasibility.md](gpu_tn_feasibility.md) | GPU TN feasibility boundary. |

## Historical Compatibility Runbooks

These documents preserve reproducibility for bounded qualification fixtures.
They are not the primary route for new whole-circuit work:

- [upmem_hardware_sliced_resident_mvp_runbook.md](history/upmem_hardware_sliced_resident_mvp_runbook.md)
- [upmem_hardware_sliced_resident_m2_3_runbook.md](history/upmem_hardware_sliced_resident_m2_3_runbook.md)
- [upmem_hardware_frontier_m3_1_runbook.md](history/upmem_hardware_frontier_m3_1_runbook.md)
- [upmem_hardware_taskgraph_resident_runbook.md](history/upmem_hardware_taskgraph_resident_runbook.md)
- [upmem_hardware_taskgraph_m4_1_runbook.md](history/upmem_hardware_taskgraph_m4_1_runbook.md)
- [upmem_hardware_simplepim_rank1_m4_2_runbook.md](history/upmem_hardware_simplepim_rank1_m4_2_runbook.md)
- [upmem_hardware_simplepim_taskgraph_m4_3_runbook.md](history/upmem_hardware_simplepim_taskgraph_m4_3_runbook.md)
- [upmem_m5_4_runbook.md](history/upmem_m5_4_runbook.md)
- [upmem_multi_dpu_scheduling_design.md](history/upmem_multi_dpu_scheduling_design.md)
- [upmem_m2_eth_evidence_analysis.md](history/upmem_m2_eth_evidence_analysis.md)
- [walkthrough.md](history/walkthrough.md)

Tracked `thesis_results/` directories are evidence capsules, not general
documentation. Always inspect the selected snapshot manifest and source run
before interpreting a historical figure.
