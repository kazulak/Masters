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
attempt. Each row contains case/route/plan identity, sample kind/index, status,
measurement, backend facts, numeric facts, validation, output hash, and failure
when applicable. `sessions.jsonl` records opened or attempted sessions,
protocol identity, terminal facts, close time, release attempts and release
verification.

An unsupported row is a preflight capability rejection. A failed row records a
runtime attempt and failure stage. Fatal external termination may leave an
incomplete artifact; verification must reject it as incomplete.

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
the canonical JSONL records. Plots must facet by route, plan, numeric policy,
topology and timing scope where those dimensions differ, and must reject
duplicate series/x-value keys.

## Promotion and Claims

Promotion to `thesis_results/` is a deliberate review action: copy only the
selected manifest, records, checksums, tables and plots, then verify the
promoted tree before tracking it. Never add `runs/`, inbox archives, build
outputs or unreviewed reports to Git.

Claim guards are mandatory. Model and SDK-simulator rows support diagnostic
correctness/protocol checks only, not physical timing, scaling, speedup or
energy. Physical execution requires physical backend facts and validated
release. Speedup additionally requires a validated CPU same-plan baseline,
matching plan and timing identities, repeated measured samples, clean linked
artifacts, and a non-bring-up scope. Energy requires measured energy with
boundary, sensor/counter identity, interval and provenance. A rejected claim
must be reported with its reasons.

This workflow records capability and evidence; it does not claim that any
physical UPMEM qualification has been completed.
