# Quantum Bench Implementation

Active implementation for the Master's thesis runtime described in
`../CODEX_IMPLEMENTATION_DIRECTION.md` and
`../CODEX_UPMEM_ARCHITECTURE_DIRECTION.md`.

The runtime is route-aware: it builds exact tensor networks from quantum
circuits, plans contractions, asks each provider route whether it can execute,
records skip reasons, validates numerical output, and writes benchmark artifacts
under timestamped `runs/` directories.

## Quick Start

```bash
cd thesis/implementation
PYTHONPATH=src ../.venv/bin/python -m pytest -q
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/smoke.yml
PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench plot runs/latest
```

For RAPL energy measurement, run the suite with the helper so sudo still uses
the thesis virtual environment:

```bash
scripts/run_energy_suite.sh configs/suites/local_energy.yml
```

## Active Boundaries

- `src/quantum_bench/` contains the Python implementation.
- `configs/suites/` contains the reproducible benchmark suite definitions.
- `native/quest_cpu/` contains only the C QuEST runner used by the
  `quest_cpu_full_state_benchmark` provider.
- `external/QuEST/` contains the local QuEST dependency used by that runner.
- `src/quantum_bench/targets/upmem/` contains host-side UPMEM WRAM, traffic,
  and schedule groundwork shared by future UPMEM providers.
- `native/upmem/` is reserved for future UPMEM native code.
- `../legacy/` contains old prototypes and generated sudo-owned run folders kept
  out of the active implementation.
