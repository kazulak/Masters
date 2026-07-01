# QuEST GPU Verification Runner

This directory is a developer-only build wrapper for verifying a real QuEST GPU
full-state execution path. It reuses the CPU runner sources from
`../quest_cpu/src` and links them against a generated GPU-enabled QuEST build.

Generated artifacts stay outside the `external/QuEST` submodule:

- `../../build/external/QuEST-hip`
- `../../build/external/QuEST-cuda`
- `build/`
- `bin/`

The benchmark route `quest_gpu_full_state_exact` is optional and gated. It is
registered for suite compatibility, but it cannot execute until
`simulation-backend-probe --verify-gpu ...` has written a verified artifact under
`../../build/gpu_verification/`.

## AMD ROCm/HIP

```bash
make clean-all
make GPU_BACKEND=hip
./bin/quest_gpu_runner --algo QRNG --qubits 2 --json
```

## NVIDIA CUDA

```bash
make clean-all
make GPU_BACKEND=cuda
./bin/quest_gpu_runner --algo QRNG --qubits 2 --json
```

The preferred project entrypoint is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench simulation-backend-probe --verify-gpu auto
```

`auto` chooses QuEST HIP on AMD hardware and QuEST CUDA on NVIDIA hardware. It
does not try CUDA on AMD, and it does not promote generic PyTorch/ROCm tensor
execution to the tailored quantum GPU route.
