# Sequential UPMEM Reference Baseline v1

## Scope

The baseline supports deterministic full pre-measurement statevector simulation
through an exact, untruncated tensor network. UPMEM execution uses finite-precision
split-complex float32. The result is a bounded reference baseline, not an optimized
TN implementation.

Unsupported scope includes sampling or mid-circuit measurement, noise, approximate
or truncated TN methods, distributed or multi-rank execution, DPU-resident graph
execution, parallel slice scheduling, energy claims, hardware-calibrated planning,
and general quantum-SDK compatibility. Slicing is secondary correctness evidence
only. It is not part of the primary Stress18 diagnostic characterization.

## Ownership And Roles

The thesis owns circuit-to-TN lowering, the logical `ContractionDAG`, the UPMEM
physical plan, ABI-v4 runtime, validation, canonical evidence, qualification, and
reports. External components have narrow roles:

- NumPy is the transparent same-DAG CPU execution control.
- opt_einsum supplies the deterministic greedy contraction path.
- Quimb is the exact direct-TN oracle and external TN context route.
- cotengra optimizes the path used by the Quimb external-context route.
- QuEST is a correctness-only oracle for the compatible QRNG and BV fixtures.
- Qiskit has no role in this baseline.

## Frozen Contract

The sequential UPMEM route uses one rank, one DPU, one tasklet, float32, fresh
physical sessions, and sequential real-product dispatch. Correctness contains
Bell2 unsliced plus Stress4 unsliced and sliced, with three successful and fully
released physical sessions. The primary powersave-conditioned diagnostic characterization contains only Stress18 and pairs
`numpy_same_dag` with `upmem_float32_1dpu_t1` in two warmup and 30 measured
complete randomized blocks. The external TN context contains only Stress18 with
`quimb_greedy` and `quimb_cotengra_path`, one warmup and five measurements each.

The four qualification inputs remain separate:

1. Software conformance establishes eight exact fixtures and oracle boundaries.
2. Physical correctness establishes UPMEM output, accuracy, release, and no-fallback provenance.
3. Same-DAG diagnostic characterization compares NumPy and UPMEM only within complete paired blocks.
4. External TN context reports Quimb routes separately and is not mixed statistically with same-DAG timing.

`steady_execution_v1` excludes planning and session lifecycle for NumPy/UPMEM.
`simulation_end_to_end_v1` covers route preparation through output extraction for
the external routes. Validation, reference calculation, hashing, and reporting are
outside both scopes. The powersave-conditioned diagnostic characterization requires
exact CPU affinity, the observed CPU 0 `powersave` governor, and the recorded
single-thread environment facts. It is claim-ineligible for optimized performance
or speedup claims. Every evidence manifest must bind to the
same exact clean source commit; bundle provenance records versions, input hashes,
and the three T1 binary hashes.

## Reproduction

Run from `thesis/implementation`. Preparation writes ignored machine-specific
copies and never edits tracked YAML:

```bash
make sequential-conformance CONFORMANCE_OUTPUT=runs/sequential-baseline/conformance.json
PYTHONPATH=src ../.venv/bin/python scripts/qualify_sequential_baseline.py prepare \
  --output-dir runs/configs/sequential-baseline \
  --rank-path /dev/dpu_rank0 \
  --correctness-session-root runs/upmem_sessions/sequential-correctness \
  --performance-session-root runs/upmem_sessions/sequential-performance \
  --expected-cpus 0
```

Correctness remains physical-only qualification. The powersave diagnostic is deliberately a
mixed generic run because it pairs NumPy and physical UPMEM:

```bash
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 make qualify \
  PHYSICAL_CONFIG=runs/configs/sequential-baseline/sequential-upmem-correctness.yml \
  OUTPUT=runs/evidence/sequential-correctness
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 PYTHONPATH=src ../.venv/bin/python \
  -m quantum_bench.cli run \
  --config runs/configs/sequential-baseline/sequential-upmem-performance.yml \
  --output runs/evidence/sequential-performance --allow-physical
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.cli run \
  --config configs/tn_benchmark_external_tn_context.yml \
  --output runs/evidence/external-tn-context
```

Generate one report per canonical evidence directory, then inspect all four
inputs. `inspect` requires the current worktree to be clean and exact-head:

```bash
make report INPUT=runs/evidence/sequential-correctness \
  REPORT_OUTPUT=runs/reports/sequential-correctness
make report INPUT=runs/evidence/sequential-performance \
  REPORT_OUTPUT=runs/reports/sequential-performance
make report INPUT=runs/evidence/external-tn-context \
  REPORT_OUTPUT=runs/reports/external-tn-context
PYTHONPATH=src ../.venv/bin/python scripts/qualify_sequential_baseline.py inspect \
  --conformance runs/sequential-baseline/conformance.json \
  --correctness runs/evidence/sequential-correctness \
  --performance runs/evidence/sequential-performance \
  --external-context runs/evidence/external-tn-context \
  --output runs/sequential-baseline/baseline-summary.json
make sequential-baseline
```

Bundle the already inspected inputs with explicit paths:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/qualify_sequential_baseline.py bundle \
  --summary runs/sequential-baseline/baseline-summary.json \
  --conformance runs/sequential-baseline/conformance.json \
  --correctness runs/evidence/sequential-correctness \
  --performance runs/evidence/sequential-performance \
  --external-context runs/evidence/external-tn-context \
  --correctness-config runs/configs/sequential-baseline/sequential-upmem-correctness.yml \
  --performance-config runs/configs/sequential-baseline/sequential-upmem-performance.yml \
  --correctness-report runs/reports/sequential-correctness \
  --performance-report runs/reports/sequential-performance \
  --external-context-report runs/reports/external-tn-context \
  --output runs/sequential-baseline/sequential-upmem-baseline-v1
```

`bundle` re-runs inspection, copies only the qualified inputs and compact
provenance, writes sorted relative `SHA256SUMS`, and produces a `.tar.gz` with an
adjacent outer SHA-256. It does not execute hardware, tag a commit, or publish a
release.

## Claim Boundary

Allowed claims are deterministic exact/untruncated pre-measurement statevector
correctness within the declared fixtures, physical float32 correctness when the
qualified artifact exists, and descriptive same-DAG paired control ratios for the
exact powersave Stress18 diagnostic. External TN timings are context only.

Prohibited claims include optimized performance or speedup claims from the
powersave diagnostic, general UPMEM acceleration, optimized TN performance,
end-to-end speedup across mismatched timing scopes, energy efficiency, scaling,
parallelism, approximate-TN quality, or broader circuit support. Any later
parallel branch must recollect the T1 control contemporaneously on the same
machine and must not reuse this frozen T1 timing as its performance baseline.
