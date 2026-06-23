# State-Vector Simulation Baseline: QuEST CPU

This directory contains a reproducible CPU baseline for PIMutation-shaped quantum
circuit workloads using the local QuEST v4 build. The ASP-DAC 2025 PIMutation
paper used QuEST v3.7.0, so results from this tree are a local SOTA CPU baseline,
not an exact paper-version reproduction unless QuEST v3.7.0 is also built and
measured.

## Build And Verify

```bash
make clean
make
./bin/quest_runner --verify FULL
```

Invalid verification selections are errors:

```bash
./bin/quest_runner --verify NOT_A_SUITE
```

## Manual Runner

Human-readable run:

```bash
./bin/quest_runner --algo BV --qubits 26
```

Machine-readable run:

```bash
./bin/quest_runner --algo BV --qubits 26 --json
```

`--json` prints one JSON object with `algo`, `input_qubits`,
`allocated_qubits`, `depth`, `threads`, `time_s`, `energy_joules`,
`energy_source`, `status`, and `error`. Energy is reported as
`energy_source=rapl_measured` only when Linux RAPL is readable; otherwise
`energy_joules` is `null` and `energy_source=unavailable`.

Algorithm names are `BB84`, `BV`, `EDC`, `HS`, `QRNG`, `XOR`, and optional
`RANDOM`. `BB` is accepted only as a compatibility alias for `BB84`; prefer
`BB84` in scripts and source-facing CLI.

For `HS`, the paper workload is `HS_2n`: logical `n` qubits and `2n` allocated
qubits. Use either:

```bash
./bin/quest_runner --algo HS --logical-qubits 13
./bin/quest_runner --algo HS --qubits 26
```

## Reproducible Suites

Suites are YAML files under `suites/`. Outputs are never overwritten; each run
gets a new timestamped directory:

```text
runs/YYYYMMDD_HHMMSS_<suite_id>/
  environment.json
  raw/repeats.jsonl
  summary.csv
  summary.json
  plots/*.png
```

Run the quick local suite:

```bash
python3 src/profiling/run_experiments.py --preset local_quick
```

For energy measurements, run the suite with privileges but keep plotting on the
thesis virtualenv interpreter so matplotlib is used instead of the fallback PNG
plotter:

```bash
sudo python3 src/profiling/run_experiments.py \
  --preset local_plot \
  --plot-python ../../.venv/bin/python
```

Equivalently:

```bash
sudo env QUEST_PLOT_PYTHON="$(realpath ../../.venv/bin/python)" \
  python3 src/profiling/run_experiments.py --preset local_plot
```

Other bundled presets:

- `local_plot`: lightweight visualization sweep with readable matplotlib plots.
- `local_energy`: RAPL-oriented local sweep using larger state vectors to avoid
  zero-energy medians.
- `paper_16_32`: PIMutation-shaped sweep, guarded at 32 allocated qubits.
- `bb84_pc_limit`: automatic largest fair `BB84/BB_n` selector above 16 qubits.

Every repeat is written to `raw/repeats.jsonl` before summaries are derived.
Skipped and failed repeats remain visible in both raw JSONL and summary files.
The suite runner captures CPU, RAM, OS, compiler flags, QuEST version/path,
OpenMP variables, git commit, and RAPL availability in `environment.json`.

Plot an existing run:

```bash
python3 src/profiling/plot_results.py runs/latest
python3 src/profiling/plot_results.py runs/current_run --baseline runs/baseline_run
```

## Circuit Manifest

The code manifest lives in `src/circuits/circuit_manifest.*`; verification checks
the implementation gate counts against it before semantic checks.

| CLI | Paper label | Qubits | Paper 1Q | Paper 2Q | Implementation 1Q | Implementation 2Q | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BB84` | `BB_n` | `n` allocated | `2n` | `0` | `2n` | `0` | PIMutation workload-shape reproduction |
| `BV` | `BV_n` | `n` allocated | `2n` | `n-1` | `2n` | `n-1` | Workload-shape reproduction, not textbook phase-kickback |
| `EDC` | `EDC_n` | `n` allocated | `2n` | `2n-2` | `2n` | `2n-2` | PIMutation workload-shape reproduction |
| `HS` | `HS_2n` | logical `n`, allocated `2n` | `6n` | `2n` | `6n` | `2n` | Identity-preserving workload-shape reproduction |
| `QRNG` | `QRNG_n` | `n` allocated | `n` | `0` | `n` | `0` | Textbook QRNG |
| `XOR` | `XOR_n` | `n` allocated | `0` | `n-1` | `0` | `n-1` | PIMutation workload-shape reproduction |

`RANDOM` is a configurable stress test and is not one of the six paper circuits.
