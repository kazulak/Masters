# Documentation Index

The research implementation is complete. This index separates final research records
from historical/superseded planning documents so that old “pending” language is not
mistaken for the current project state.

## Final/current documents

| Document | Role |
| --- | --- |
| [`../README.md`](../README.md) | Technical entry point and reproduction commands |
| [`../STATUS.md`](../STATUS.md) | Final capability and research status |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | Final execution and P6 architecture |
| [`RESEARCH_MILESTONES.md`](RESEARCH_MILESTONES.md) | Final milestone chronology and disposition |
| [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) | Thesis-safe interpretation of accepted results |
| [`repository_lineage.md`](repository_lineage.md) | Scientific source/tag lineage |
| [`identities.md`](identities.md) | Evidence identity definitions |
| [`timing.md`](timing.md) | Timing scopes and valid comparison rules |
| [`evidence_workflow.md`](evidence_workflow.md) | Evidence handling and verification rules |
| [`upmem_kernel_schedule_system_v1.md`](upmem_kernel_schedule_system_v1.md) | Detailed P1-P5 executor research record |
| [`upmem_cost_guided_path_v1.md`](upmem_cost_guided_path_v1.md) | Detailed P6 software/campaign checkpoint record |
| [`quantized_contraction_policy_v1.md`](quantized_contraction_policy_v1.md) | Shared-scale int8 mathematical policy |
| [`quantized_upmem_execution_diagnostic_v1.md`](quantized_upmem_execution_diagnostic_v1.md) | Physical int8 diagnostic |
| [`upmem_pimutation_workload_reconciliation_v1.md`](upmem_pimutation_workload_reconciliation_v1.md) | Final family-aligned workload provenance |
| [`upmem_resident_pair_measurement_boundary.md`](upmem_resident_pair_measurement_boundary.md) | Bounded residency experiment boundary |

## Historical/superseded documents retained for provenance

The following documents describe earlier states or superseded path-study designs. They are
valuable research records but are **not active instructions**:

- `ROADMAP.md` — replaced with a final completed-roadmap summary in the publication
  cleanup;
- `upmem_path_heuristic_v1.md` — historical pilot; original raw physical archives were
  lost and the result is noncanonical;
- `upmem_path_heuristic_generalization_v1.md` — superseded pre-final path work;
- `upmem_final_system_path_study_v2.md` — superseded workload/protocol allocation;
- `upmem_execution_preparation_v1.md` — predecessor integration preparation;
- files below `docs/archive/` — earlier milestone and planning material.

The authoritative final path result is the immutable package:

```text
../thesis_results/upmem_cost_guided_path_v1/
```

The authoritative final executor identity is
`thesis-upmem-kernel-schedule-system-v1`; the authoritative final P6 identities are
`thesis-upmem-cost-guided-software-v1` and `thesis-upmem-cost-guided-results-v1`.
