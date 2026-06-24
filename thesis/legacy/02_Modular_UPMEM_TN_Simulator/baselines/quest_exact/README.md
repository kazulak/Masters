# QuEST Exact State-Vector Baseline

This baseline is the V2-runtime adapter for exact CPU state-vector simulation
with QuEST. It is separate from `../../PIMutation_reconstruction` so it can use
the same benchmark YAML, output records, and validation workflow as the
TaskGraph tensor-network route.

The runner supports the small shared circuits used in Stage 1A:

- `bell_2q`
- `ghz_4q`
- `ghz_chain --qubits N`
- OpenQASM 2.0 files with the small gate subset used by the MVP fixtures

Build:

```bash
make
```

Manual run:

```bash
./bin/quest_exact_runner --circuit bell_2q --output /tmp/bell_quest.bin
```

Energy:

- If Linux RAPL is readable, the runner reports measured CPU package Joules.
- Otherwise the Python pipeline records an explicit static-power estimate using
  `measurement.energy.cpu_watts` from the benchmark YAML.

