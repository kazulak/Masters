# Physical SimplePIM TaskGraph M4.5 Evidence

This is a compact, tracked capsule of the successful ETH physical UPMEM
functionality run captured on 2026-08-09. It is not the selected thesis
snapshot and is not a benchmark report.

## What This Evidence Shows

- One fixed real-valued three-task contraction TaskGraph executed on physical
  UPMEM hardware with one and two DPUs.
- Both placements used the same resident package and source circuit/TN/plan
  identities, with distinct schedules and execution-plan hashes.
- The two-DPU placement executed its two independent initial contractions in
  one frontier wave, followed by one host-mediated dependency handoff.
- Allocation, native kernel execution, aggregate exact-once completion,
  final-output CPU-reference validation, release, and application-visible
  transfer accounting succeeded without simulator or CPU fallback.

The final session outputs passed the CPU reference with a maximum absolute
error of `3.787677971267556e-08` against a `1e-6` tolerance.

## Claim Boundary

This is physical functionality evidence only. It supports bounded execution,
dependency preservation, final-session validation, and transfer-accounting
claims. It does not support timing, speedup, scaling, overlap, energy,
tasklet-parallelism, multi-rank, multi-DIMM, PID-Comm, ATiM, SparseP, or
general quantum-TN performance claims.

The native session retained one final output and aggregate completion counters
for each placement. Consequently, `normalized_records.jsonl` contains
measured-run row metadata only: each row explicitly records
`repeat_output_validation_status=not_individually_collected` and
`repeat_completion_observation_status=not_individually_collected`. Validation
and exact-once completion are session-scoped, not per-repeat claims.

## Contents

- `normalized_records.jsonl`: six measured rows, three repeats for each
  placement.
- `warmup_records.jsonl`: two warmup rows, excluded from measured evidence.
- `run_manifest.json`, `environment.json`, and the summary: source provenance
  and run-level outcomes.
- `resolved_suite.yml`, `hardware_profile.json`, and the resident request:
  exact run configuration and package request.
- `session_failures.jsonl`: empty, preserving the source run's no-failure
  result.
- `capsule_manifest.json` and `checksums.json`: capsule provenance and
  integrity data.

This is a raw/normalized evidence capsule, so it deliberately has no report
manifest, derived tables, or plots. Those would imply a benchmark-report
surface that this timing-free bring-up run does not provide.

Source run: `eth-evidence/2026-08-09_22-19-27` (ignored development import).
Source commit: `c7bbf957d17346e819c52fc45ca592c3bcb691ca`.
