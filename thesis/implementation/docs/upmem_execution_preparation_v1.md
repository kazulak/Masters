# Execution Preparation v1

## Status and Boundary

This is speculative software-only preparation, not adoption of the integrated
runtime or physical qualification. The frozen Phase A execution source is
`b921b8804e324da75222354ee2f4df41e770b75c`; its unchanged seven-session physical
gate must pass before production adoption or downstream physical campaigns.
The preparation branch starts from documentation source
`5b93f87c1a034944859348c99e2fe263961a2114`.

Allowed changes are scripts, focused tests and documentation. No production
runtime, native code, ABI, transport format, numerical policy, frozen candidate
pool or physical configuration changes are authorized by this experiment.
The budget is three focused days within the existing ten-day roadmap cap.

## Fixed Workload

Use the retained generalization candidate pools for Stress16 (two repeat layers),
HS20 (depth one), and EDC14, plus the retained pilot BV18 candidate pool. Select
greedy, minimum conventional FLOPs and minimum peak intermediate, tie-breaking by
candidate-path ID. Deduplicate coincident paths before physical lowering. Never
regenerate candidates or substitute a path after a rejection or timing result.

Reconstruct each DAG and lower it under `split_complex_float32_v1` and
`complex_int8_shared_scale_v1` at one and four DPUs, T8, one rank. There are at
most 48 census cells and 16 greedy benchmark cells. Preserve the 512 MiB planning
memory limit, 400-work-unit limit and 60-second lowering timeout. Rejections are
explicit records, not omitted observations. GHZ16, GHZ14 and XOR18 are excluded.

Record source/dependency identities, input pool hashes, selected paths and roles,
logical/physical identities, geometry and work-unit facts. Static movement and
memory estimates are not hardware counters. The older heuristic feature named
`packed_operation_count` aliases waves; it is not the runtime submission count.
Report actual lane-envelope submissions separately without changing that frozen
feature definition.

Eligibility here means bounded host-only preparation, not physical scaling
admission. Keep legal underfilled waves and report their scaling-admission facts
separately. Envelope descriptors count requests; DPU records include idle output
paths, whereas actual output files count active work only. The retained BV18
minimum-peak selection has no logical identity and is already excluded in its
source pool for semantic-identity expansion. Preserve that exclusion without
substituting a path. All sixteen greedy cells remain available for the probe.

## One Prototype

The baseline constructs and writes four lane envelopes, in RR, II, RI, IR order.
The candidate constructs one envelope containing the same requests in that same
lane-major and wave order. Both use the existing request/template builders and
`pack_operation`, not a new request model or a new native host process.

Embedded manifests, sidecars, payload bytes, hashes, request sequences, topology
and relative output paths must agree. The envelope hashes and descriptor offsets
may differ. Request/output identities must remain unique. Four real-product
launches per wave, operand movement and output reconstruction are not eliminated
by this host-only comparison. No actual launch or SDK allocation is performed.

Use deterministic payloads at the actual planned operation geometry and the
existing numerical encoder. These are preparation fixtures, not physical tensor
observations or proof of complete physical output. Use small CPU replay fixtures
separately for numerical checks.

## Measurement Protocol

Benchmark every contraction of each eligible greedy cell, not one representative
tiny request. Use one warmup and seven measured complete paired blocks, arm order
randomized with seed `20260905`. Both arms run locally with one CPU thread, fresh
isolated roots on the same filesystem and identical transient-file durability
(no fsync). Preserve every raw observation and failure; no replacement samples.

Include materialization, encoding, request construction, packing and file writes
inside each arm's preparation measurement. Report setup, validation and cleanup
separately. Do not precompute away common expensive work in only one arm. Use
identical validation outside the timed preparation, with no extra validator
subprocess in only one arm. Record monotonic wall and CPU time, file/byte/request
counts, and per-arm peak memory with a fresh-process measurement boundary.

Report per-operation values, complete-cell totals, paired ratios/differences,
median, raw MAD and range. Local timings cannot establish ETH, SDK, DPU kernel,
steady-execution or session-inclusive speedup. Any Amdahl projection requires a
separately justified affected physical fraction; do not invent one from the
unmeasured interval or a different numerical policy.

## Qualification and Delivery

Tests cover deterministic selection/deduplication, both-policy arithmetic,
rejections, identities, byte/hash equivalence, lane/wave order, unique output
paths, partial waves, idle slots, odd dimensions, split-K, and corrupt envelopes.
Run focused tests, the full pinned software suite, Ruff, diff checks and hosted
CI on the preparation branch. Do not rerun the unchanged 14-cell baseline SDK
gate or start physical execution as part of this host-only experiment.

Retain census JSON/CSV, raw paired benchmark JSON/CSV, equivalence and memory/count
reports, provenance, and sorted relative SHA256SUMS in durable ignored storage.
Create and verify a portable archive and a second retained copy. A negative or
neutral preparation result completes this bounded experiment; positive results
authorize proposing a production experiment, not merging this probe into runtime.
