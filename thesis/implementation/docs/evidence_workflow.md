# ETH Evidence Workflow

This project has three distinct artifact locations. Keeping them separate
prevents raw hardware output from appearing as source changes and prevents
unreviewed output from becoming thesis evidence.

| Location | Git status | Purpose |
| --- | --- | --- |
| `runs/inbox/eth/<experiment-id>/` | Ignored | Raw archives copied from ETH. Preserve these unchanged. |
| `runs/evidence/<suite>/<route>/<timestamp>/` | Ignored | Exact unpacked benchmark run consumed by report commands. |
| `runs/comparisons/<label>/<timestamp>/` | Ignored | Derived CSVs, plots, and benchmark summary regenerated from normalized records. |
| `thesis_results/<snapshot-name>/` | Tracked after review | Compact, checksum-verified evidence selected for the thesis. |

The inbox is intentionally not created by Git. Create it after every fresh
clone:

```bash
make evidence-inbox
```

## Copy a Physical Run

On ETH, archive the *completed evidence run directory*, not a source checkout.
For the physical TaskGraph route, the source directory is normally:

```bash
RUN=$(readlink -f ~/work/Masters/thesis/implementation/runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph/latest)
tar -C "$(dirname "$RUN")" -czf ~/upmem_taskgraph_$(date -u +%Y-%m-%d_%H-%M-%S).tar.gz "$(basename "$RUN")"
```

On the local machine, from `thesis/implementation`:

```bash
make evidence-inbox
scp safari-baguette1:~/upmem_taskgraph_<timestamp>.tar.gz runs/inbox/eth/
tar -tzf runs/inbox/eth/upmem_taskgraph_<timestamp>.tar.gz | head

mkdir -p runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph
tar -xzf runs/inbox/eth/upmem_taskgraph_<timestamp>.tar.gz \
  -C runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph
```

The archive must unpack to one timestamped run directory containing at least:

```text
run_manifest.json
environment.json
normalized_records.jsonl
```

For an existing raw tree accidentally copied outside `implementation`, move it
without deleting data. For example, from `thesis/implementation`:

```bash
make evidence-inbox
mv ../upmem-investigation "runs/inbox/eth/physical-mvp-raw-2026-07-16"
```

Do this only after checking that the source directory is the copied raw
evidence you intend to keep. Git will then stop reporting it as an untracked
repository change.

## Audit and Report

Report from the exact extracted run. Do not rerun the hardware benchmark just
to make plots:

```bash
RUN=runs/evidence/upmem_hardware_taskgraph_correctness/upmem_hw_taskgraph/<timestamp>
make upmem-hw-taskgraph-report UPMEM_HW_TASKGRAPH_RUN="$RUN"
```

The command prints a generated comparison directory under
`runs/comparisons/research_pack/upmem_hw_taskgraph/<timestamp>/`. Inspect its
`benchmark_summary.md`, `plot_manifest.json`, `tables/`, and `plots/`.

For the first physical TaskGraph route, the transfer, error, and validation
plots are evidence. The runtime plot remains a TODO unless the record states a
non-bring-up timing scope. It must not be used as a speedup figure.

## Promote a Reviewed Snapshot

Promotion is deliberately manual. Before it, the local `thesis/implementation`
source must be at the same clean commit recorded by the ETH run. Commit and
push implementation changes before running ETH; do not use `--allow-dirty` for
thesis results.

```bash
PACK=runs/comparisons/research_pack/upmem_hw_taskgraph/<timestamp>
../.venv/bin/python scripts/thesis_snapshot.py promote \
  --pack "$PACK" \
  --out thesis_results/physical_hardware_taskgraph_v1
../.venv/bin/python scripts/thesis_snapshot.py verify \
  --snapshot thesis_results/physical_hardware_taskgraph_v1
```

Review the staged snapshot, then add only that named `thesis_results/` tree to
Git. Do not add `runs/`, `build/`, or `runs/inbox/`.

## Cleanup

`make thesis-clean` considers `runs/evidence/` and `runs/comparisons/`; it does
not delete the inbox. Keep the original archive until its extracted evidence,
report, and any promoted snapshot have been verified. Use `archive-evidence`
only for an explicit generated run you no longer need inside the repository.
