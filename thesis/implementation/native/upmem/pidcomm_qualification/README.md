# PID-Comm qualification

This is an isolated physical qualification lane for the PID-Comm public
all-reduce API. It is not an executor integration and does not share source,
build products, or run directories with resident/M4.6/M5 routes.

The prepare command records the CPU flags, installed system SDK/compiler,
external commit, thesis commit/dirty state, hashes for every thesis-owned and
staged external input, candidate DPU counts, and payload contract without
building or allocating a DPU. The compatibility-only command stages and
compile/links the host against the installed SDK without allocating a DPU.
The execution command requires `UPMEM_ALLOW_PHYSICAL_HARDWARE=1`, performs
that compatibility preflight before any physical allocation, and writes
candidate logs and manifests under `runs/`.

This is a mixed system SDK plus staged PID-Comm source/prebuilt-binary stack.
The staged build links the installed SDK found through `dpu-pkg-config` and
stages the pinned PID-Comm `commlib.c`, headers, and communication binaries.
The PID-Comm checkout and its bundled modified SDK are read-only inputs; the
bundled SDK is never put on `PATH` or `LD_LIBRARY_PATH`.
