# QuEST CPU Native Baseline

This directory contains the small C runner used by `quantum_bench` as the CPU
full-state-vector benchmark provider. It is not a separate benchmark pipeline.
Run suites through the project-level command:

```bash
cd thesis/implementation
../.venv/bin/python -m quantum_bench.bench run --suite configs/suites/local_energy.yml
```

## Build And Verify

The runner links against the local QuEST source under `../../external/QuEST`.
Generated QuEST build files are written outside the submodule by default, under
`../../build/external/QuEST`, so `external/QuEST` stays clean.

```bash
make clean
make
./bin/quest_runner --verify FULL
```

`make` builds the QuEST library first via the `quest-lib` target. Override the
generated-library path only when needed:

```bash
make QUEST_BUILD=/tmp/quest-build
```

`../../build/external/QuEST`, `build/`, and `bin/` are generated artifacts and
should not be committed. Use `make clean-all` to remove both the native runner
build and the implementation-local QuEST build.

Machine-readable benchmark call:

```bash
./bin/quest_runner --algo BV --qubits 26 --json
```

Algorithm names are canonical only: `BB84`, `BV`, `EDC`, `HS`, `QRNG`, `XOR`,
and optional `RANDOM`.

For `HS`, the paper workload is `HS_2n`: logical `n` qubits and `2n` allocated
qubits.

```bash
./bin/quest_runner --algo HS --logical-qubits 13 --json
./bin/quest_runner --algo HS --qubits 26 --json
```

Energy is reported as `energy_source=rapl_measured` only when Linux RAPL is
readable; otherwise `energy_joules` is `null` and `energy_source=unavailable`.

## Circuit Manifest

The code manifest lives in `src/circuits/circuit_manifest.*`; verification
checks implementation gate counts against it before semantic checks.

| CLI | Paper label | Qubits | Paper 1Q | Paper 2Q | Implementation 1Q | Implementation 2Q | Kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `BB84` | `BB_n` | `n` allocated | `2n` | `0` | `2n` | `0` | PIMutation workload-shape reproduction |
| `BV` | `BV_n` | `n` allocated | `2n` | `n-1` | `2n` | `n-1` | Workload-shape reproduction, not textbook phase-kickback |
| `EDC` | `EDC_n` | `n` allocated | `2n` | `2n-2` | `2n` | `2n-2` | PIMutation workload-shape reproduction |
| `HS` | `HS_2n` | logical `n`, allocated `2n` | `6n` | `2n` | `6n` | `2n` | Identity-preserving workload-shape reproduction |
| `QRNG` | `QRNG_n` | `n` allocated | `n` | `0` | `n` | `0` | Textbook QRNG |
| `XOR` | `XOR_n` | `n` allocated | `0` | `n-1` | `0` | `n-1` | PIMutation workload-shape reproduction |

`RANDOM` is a configurable stress test and is not one of the six paper circuits.
