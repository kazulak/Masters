# External Submodules

`thesis/implementation/external` is the canonical location for active external
dependencies and candidate libraries. These entries are Git submodules, not
vendored source files. The repository-level `.gitmodules` file lives at the
main Git root and points to these paths.

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
| `SimplePIM/` | `https://github.com/CMU-SAFARI/SimplePIM.git` | `1d639c53532555f01e9f71d872e7712b166d6cba` | Target UPMEM compute/runtime abstraction for L1/L2 and local tile compute inside L3 | Submodule; not currently used for GEMM execution |
| `PID-Comm/` | `https://github.com/AIS-SNU/PID-Comm.git` | `cecc39e29e6576ced73b2041db6e357769a6531a` | Communication/orchestration substrate across L1/L2/L3, strongest for L3 distributed contraction | Submodule; not currently integrated into execution |
| `ATiM/` | Not confirmed | Not applicable | SLR-derived tensor-kernel autotuning candidate | Planned only; add a submodule only after the authoritative URL is known |

Additional SLR-derived candidates such as SparseP, PRISM, PyGim, PIM-LLM GEMM,
and TransPimLib stay documented in `docs/runtime_architecture_map.md` until
they become implementation-local submodules.

## Update Policy

- Pin submodules to explicit commits; do not track floating branches.
- Update one submodule at a time and record why the commit changed.
- Keep build outputs out of submodules. Build in `runs/`, temporary workspaces,
  or dedicated ignored build directories.
- Do not set submodules to ignore dirty state. Dirty external sources should be
  visible during development.
- Do not add copied third-party source trees directly to the parent repository.
