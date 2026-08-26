# Evidence Workflow

Generated runs are local artifacts. They are not source changes and are not
automatically thesis evidence.

## Locations

| Location | Git status | Purpose |
|---|---|---|
| `runs/inbox/eth/<id>/` | ignored | Untouched archives copied from ETH |
| `runs/evidence/<id>/` | ignored | Extracted canonical run consumed by verification/reporting |
| `runs/comparisons/<id>/` | ignored | Derived tables, plots and summaries |
| `thesis_results/<id>/` | tracked only after review | Deliberately promoted compact evidence |

Create ignored directories as needed:

```bash
mkdir -p runs/inbox/eth runs/evidence runs/comparisons
```

## Run Locally

The active interface is:

```bash
make plan CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/plan
make run  CONFIG=configs/tn_benchmark_reset.yml OUTPUT=runs/evidence/local-run
make verify INPUT=runs/evidence/local-run
make report INPUT=runs/evidence/local-run REPORT_OUTPUT=runs/comparisons/local-run
```

The CLI creates exactly three canonical files in the run directory:

```text
manifest.json
samples.jsonl
sessions.jsonl
```

`manifest.json` binds schema, run/experiment/environment/validation identities,
source commit and dirty-tree state, configuration, expected counts, file names,
and terminal status. `samples.jsonl` has one row per warmup or measurement
attempt. Each row contains case/route/plan identity, attempt kind/index, block
and deterministic order, status, measurement, backend facts, numeric facts,
validation, output hash, and failure when applicable. `sessions.jsonl` records
opened or attempted sessions,
protocol identity, terminal facts, close time, release attempts and release
verification.

An unsupported row is a preflight capability rejection. A failed row records a
runtime attempt and failure stage. Fatal external termination may leave an
incomplete artifact; verification must reject it as incomplete.

Manifests use `evidence_manifest_v2`, samples use `evidence_sample_v3`,
sessions use `evidence_session_v1`, and reports use `evidence_report_v4`.
Earlier sample evidence is unsupported by the active verifier unless the
generation is listed below. Sample `status`
describes whether the complete attempt finished: a validator exception creates
a failed sample, while a policy-reference mismatch or accuracy qualification
miss remains a successful sample with its measurement, output hash, and facts
retained. Policy-reference correctness is reported separately from
`accuracy_qualified`.

## Artifact Compatibility

Evidence is verified with the toolchain that owns its schema generation. The
active implementation does not carry a generic migration framework.

| Artifact generation | Manifest / sample / session / report schemas | Source tag or release | Active verifier support | Reviewer command |
|---|---|---|---|---|
| M6 software qualification | `v1` / `v2` / `v1` / `v2` | `thesis-m6-software-ready-v1` release | No | Use the M6 tag/release toolchain: `make verify INPUT=<run>` |
| M7A WRAM-kernel qualification | `v2` / `v3` / `v1` / `v3` | `thesis-m7a-wram-kernel-software-ready-v1` release | Yes | `make verify INPUT=<run>` |
| M7B pre-physical evidence | `v2` / `v3` / `v1` / `v4` | M7B qualified tag when created | Defined by that tag | Use the matching qualified tag's `make verify` command |

The M6 release exists; it is tag-pinned rather than unsupported because of a
missing release. M7A evidence remains readable by the active verifier so that
later qualification can re-check its immutable bundle.

## Copy ETH Results

On ETH, archive the completed run directory without changing its contents:

```bash
RUN=$(readlink -f ~/work/Masters/thesis/implementation/runs/evidence/<run-id>)
tar -C "$(dirname "$RUN")" -czf ~/qbench_$(date -u +%Y-%m-%d_%H-%M-%S).tar.gz "$(basename "$RUN")"
```

On the local machine:

```bash
mkdir -p runs/inbox/eth runs/evidence
scp safari-baguette1:~/qbench_<timestamp>.tar.gz runs/inbox/eth/
tar -tzf runs/inbox/eth/qbench_<timestamp>.tar.gz | head
tar -xzf runs/inbox/eth/qbench_<timestamp>.tar.gz -C runs/evidence
make verify INPUT=runs/evidence/<timestamped-run>
```

Keep the original archive. Do not copy raw results into tracked source
directories. The extracted run must contain the three canonical files and must
retain the source commit and environment facts recorded by the run.

## Verification and Reporting

`verify` checks schemas, IDs, exact matrix-route identity bindings,
sample/session links, expected counts, scopes, statuses, output/validation
fields, session release and terminal manifest state. A completed artifact may
neither omit a configured route nor introduce an undeclared route. `report`
reads an already verified run; it never executes workloads.
Failed artifacts may omit routes that were never attempted, but every observed
sample or session must still match a persisted identity binding and a route in
the configured matrix.
Derived reports belong under `runs/comparisons/` and must be regenerated from
the canonical JSONL records. `speedups.csv` is reserved for matched NumPy
versus physical-UPMEM comparisons; `scaling.csv` holds matched UPMEM tasklet
and DPU comparisons. Both require the persisted campaign admission facts,
not geometry recomputed by the report. Plots must facet by route, plan, numeric
policy, topology and timing scope where those dimensions differ, and must
reject duplicate series/x-value keys.

The manifest's collection policy fixes deterministic warmup/measurement block
order and lifecycle. Reports retain attempted, successful, failed, and
unsupported measurement counts; calculate median, raw MAD, and deterministic
percentile-bootstrap intervals from successful measurements; and visibly label
accuracy-unqualified series. No post-hoc outlier exclusion or replacement
attempt is performed. Block-paired bootstrap speedup intervals are available
only after the physical provenance and complete-measurement claim gates pass.

## Promotion and Claims

Promotion to `thesis_results/` is a deliberate review action: copy only the
selected manifest, records, checksums, tables and plots, then verify the
promoted tree before tracking it. Never add `runs/`, inbox archives, build
outputs or unreviewed reports to Git.

Claim guards are mandatory. Model and SDK-simulator rows support diagnostic
correctness/protocol checks only, not physical timing, scaling, speedup or
energy. A speedup candidate must have an applicable and passed policy reference,
`accuracy_qualified=true`, qualified physical provenance, and matching scope
and identities. Its matching CPU same-plan baseline must have
`accuracy_qualified=true` and pass its policy reference when applicable.
All planned measurement attempts must complete successfully, clean linked
artifacts and a non-bring-up scope remain required. Energy requires measured
energy with boundary, sensor/counter
identity, interval and provenance. A rejected claim must be reported with its
reasons.

This workflow records capability and evidence; it does not claim that any
physical UPMEM qualification has been completed.
