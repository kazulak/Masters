# UPMEM-Aware Contraction-Path Heuristic v1

## Purpose

This milestone tests whether an SLR-derived UPMEM-aware candidate-path score
can select physically faster contraction paths than the conventional
`opt_einsum` greedy path. Cotengra generates complete candidate paths; it is
not replaced or modified. Every candidate is lowered through the canonical
`ContractionDAG` and `UpmemPlan` before it is scored.

The experiment is preregistered in
`configs/upmem_path_heuristic_v1.json`. Circuit splits, candidate-generation
seeds, resource topologies, calibration size, fitting objective, and physical
collection sizes are frozen before candidate timing is observed.

## Cost Model

The score identifier is `upmem_slr_cost_v1`:

```text
C = alpha * B_host_dpu
  + beta  * B_mram_wram
  + gamma * I_dpu
  + delta * N_sync
  + eta   * E_num
  + theta * P_wram
```

Each term is normalized relative to the greedy path for the same circuit and
topology:

```text
z_i = log((x_i + 1) / (x_i_greedy + 1))
```

Weights are non-negative and sum to one. Lower scores are preferred.

Candidate physical lowering is bounded to 60 seconds per complete path. A
path exceeding that preregistered software-planning bound is retained in the
candidate pool with explicit infeasibility facts and is never submitted to
the SDK simulator or physical hardware.

Before materializing tiles, the generator derives their exact count from the
canonical B/M/N/K geometry and fixed tile limits. Candidates requiring more
than 500 planned work units are retained as explicitly infeasible. This is
approximately 2.25 times the 222-unit accepted Stress18 greedy plan while
bounding the finite calibration
campaign. The time bound is therefore a failure guard rather than the primary admission
rule.

- `B_host_dpu` is planned H2D plus D2H traffic. The components are retained.
- `B_mram_wram` is the existing aligned-transfer estimate derived from tile
  geometry. It is not a hardware counter.
- `I_dpu` is exact real-MAC work for the four real products used by the
  split-complex float32 route.
- `N_sync` exposes packed operations, waves, launches, host reductions, and
  modeled barrier events before applying one documented aggregate.
- `E_num` is inactive in v1 when float32 representation overhead is already
  represented by movement and arithmetic terms.
- WRAM admission is a hard feasibility condition. `P_wram` records only
  existing modeled buffer and tiling facts and is inactive when it does not
  discriminate candidates.

Transfer savings are not also credited to `E_num`; four-real-product work is
not counted twice; and movement caused by WRAM tiling is not automatically
penalized again through `P_wram`.

## Data Separation

Training uses Stress18, HS18, GHZ18, and BV16. EDC16 is validation-only and
BV18 is held out for the final test. Both resource topologies for one circuit
remain in the same split.

For each training circuit and topology, no more than six paths are physically
calibrated: greedy, minimum FLOPs, minimum peak intermediate, minimum writes,
equal-weight score best, and deterministic feature-diverse candidates needed
to fill coincident selections. The calibration set is selected without
runtime observations.

Weights are fitted offline only against measured calibration paths. Validation
does not authorize retuning. The held-out test is evaluated only after the
weight profile and all model decisions are frozen.

## Physical Scope

Physical v1 uses one rank, packed-operation transport, split-complex float32,
and two explicit resource topologies: one DPU with eight tasklets and four
DPUs with eight tasklets each. Simulator timings are never used for fitting.

Calibration uses one warmup and three measurements. Validation and test use
one warmup and five measurements for each unique greedy, minimum-FLOP, and
UPMEM-selected path. Raw observations are retained. Paths are never replaced
or automatically retried after a failed physical attempt.

## Claim Boundary

The milestone can report a physically calibrated, interpretable
UPMEM-aware candidate-path heuristic and diagnostic held-out results for the
frozen circuit instances. It cannot claim global path optimality, universal
circuit generality, optimal resource selection, final thesis performance, or
CPU/GPU competitiveness. A negative held-out result is a valid outcome.
