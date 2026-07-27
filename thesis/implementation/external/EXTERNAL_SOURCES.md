# External Submodules

`thesis/implementation/external` is the canonical location for active external
dependencies and target architecture providers. SimplePIM, PID-Comm, ATiM, and
SparseP are central target providers, each limited to its task-specific role;
none is a universal replacement for the explicit SDK control. Checked-out
entries are Git submodules, not vendored source files. The repository-level
`.gitmodules` file lives at the main Git root and points to these paths.

Historical copies under `../legacy/extern/` are fallback/provenance material
only. Active code must not require them.

## Setup

From the repository root:

```bash
git submodule update --init --recursive thesis/implementation/external/QuEST thesis/implementation/external/SimplePIM thesis/implementation/external/PID-Comm
```

From `thesis/implementation`:

```bash
git submodule update --init --recursive external/QuEST external/SimplePIM external/PID-Comm
```

## Registry

| Path | URL | Pinned commit | Role | Status |
|---|---|---:|---|---|
| `QuEST/` | `https://github.com/quest-kit/QuEST.git` | `9d7618d7263e3bfba433b88cf1eac0647f08fa0a` | CPU full-state benchmark dependency | Submodule |
| `SimplePIM/` | `https://github.com/CMU-SAFARI/SimplePIM.git` | `1d639c53532555f01e9f71d872e7712b166d6cba` | Task-specific target for UPMEM management, distribution, and bounded array/map/zip/reduce primitives | Submodule; physical qualification is M1; not claimed as current executor integration |
| `PID-Comm/` | `https://github.com/AIS-SNU/PID-Comm.git` | `cecc39e29e6576ced73b2041db6e357769a6531a` | Task-specific target for multi-DPU relocation and collective reduction | Submodule; physical qualification is M1; not claimed as current executor integration |
| `ATiM/` | `https://zenodo.org/records/15379025` | Not pinned | Task-specific target for generated/autotuned dense local tensor kernels | Official artifact identified; physical qualification and source pinning are M1 prerequisites |
| `SparseP/` | `https://github.com/CMU-SAFARI/SparseP.git` | Not pinned | Task-specific target for sparse formats, kernels, partitioning, and load balancing | Official repository identified; physical qualification is an M1 prerequisite |

Additional SLR-derived references such as PRISM, PyGim, PIM-LLM GEMM, and
TransPimLib remain future references in `../ARCHITECTURE.md` until they have an
explicit responsibility in the target architecture.

## Qualification And Update Policy

M1 qualifies all four central providers independently on physical UPMEM using
small task-specific probes. Qualification records are development evidence:
they remain ignored and are not promoted while M0--M8 are in progress. A
qualified provider is still selected only for eligible work in later milestones.

- Pin submodules to explicit commits; do not track floating branches.
- Update one submodule at a time and record why the commit changed.
- Keep build outputs out of submodules. Build in `runs/`, temporary workspaces,
  or dedicated ignored build directories.
- Do not set submodules to ignore dirty state. Dirty external sources should be
  visible during development.
- Do not add copied third-party source trees directly to the parent repository.
