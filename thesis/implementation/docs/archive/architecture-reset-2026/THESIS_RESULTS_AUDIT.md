# Thesis Results Audit

Date: 2026-07-12

## Snapshot Status

This file is a historical audit of the tracked thesis snapshot, not a current
benchmark result. The evidence and derived report named below were generated
before the research-pack reporting and quantization-stress updates. The
tracked `thesis_results/current` snapshot remains stale until the research
suites are rerun and the snapshot is regenerated.

Historical inputs:

- Full-state CPU/GPU evidence: `thesis/implementation/thesis_results/current/evidence/full_state_performance/`
- CPU/TN evidence: `thesis/implementation/thesis_results/current/evidence/cpu_tn/`
- UPMEM evidence: `thesis/implementation/thesis_results/current/evidence/upmem_generic_boundary/`
- Tracked report: `thesis/implementation/thesis_results/current/`

No statement in this audit should be read as the current UPMEM boundary. In
particular, the old snapshot must not be used to assert that a specific QRNG
case is current or that tiling is absent.

## Historical Interpretation

The old snapshot established these bounded claims:

- QuEST CPU/GPU rows were full-state evidence, with GPU claims gated on verified
  native execution.
- Quimb/cotengra was the serious external CPU tensor-network baseline.
- Internal NumPy TaskGraph replay was a shared-plan CPU reference and diagnostic
  route, not a state-of-the-art TN baseline.
- Strict generic UPMEM rows represented SDK-simulator code-path evidence only.
- UPMEM simulator timing was not hardware timing or hardware speedup.
- Quantized execution validation and full-precision accuracy were not yet
  separated consistently in all derived tables.

The historical quantized CPU replay was diagnostic quantize/dequantize overhead
followed by CPU complex contraction. It was not evidence about native int8
UPMEM performance.

## Post-2E.65 Implementation State

The current source includes bounded tiling support and metadata. The planner
contains L1 direct and L2 single-DPU MRAM-resident/WRAM-tiled plans, and the
strict generic route uses bounded output tiling. These paths remain bounded by
rank, element-count, layout, dtype, and complex-operation contracts. Unsupported
records are the source of truth for the next boundary; this audit does not
substitute a new boundary from the stale snapshot.

The current research batch adds the deterministic
`quantization_stress` builtin at 4, 6, and 8 qubits and a manual strict
generic-only float32/int8 suite. Its CPU reference is the internal
`cpu_tn_einsum_exact` TaskGraph replay. The suite makes no hardware claim.

## Regeneration Requirement

Before using thesis results, rerun the research pack suites, regenerate the
derived CSVs and plots, and promote a new compact snapshot. The regenerated
pack must derive tiling status, highest supported qubit count, first
unsupported case and reason, and the next implementation target from its
loaded normalized records.

Until that happens, `thesis_results/current` is explicitly stale.
