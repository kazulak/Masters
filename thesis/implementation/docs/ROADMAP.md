# Completed Research Roadmap and Final Scope

This file replaces the pre-freeze forward-looking roadmap. The thesis implementation
research is complete; entries below record final disposition rather than future work.

| Research area | Final disposition |
| --- | --- |
| Correct circuit -> TN -> path -> DAG pipeline | Complete |
| CPU/TN and full-state reference adapters | Complete for declared thesis roles |
| Physical UPMEM execution | Complete for the retained one-rank profile |
| WRAM-panel kernel | Retained |
| Tasklet parallelism | Implemented, qualified, and physically studied |
| Multi-DPU contraction | Implemented, qualified, and physically studied |
| Static DAG-wave concurrency | Implemented, qualified, and retained |
| Four-product complex fusion | Implemented, physically confirmed, retained when admitted |
| Outer-K1 specialization | Implemented and physically evaluated; rejected for adoption |
| Exact slicing | Implemented and bounded physically evaluated; not automatic |
| Resident intermediate pair | Bounded prototype physically evaluated; rejected for adoption |
| Shared-scale complex int8 | Implemented and physically characterized; accuracy separate |
| Hardware-aware path score | Implemented and physically calibrated |
| Hardware-aware reranking | Implemented and physically evaluated |
| Hardware-aware adaptive generation | Implemented and physically evaluated |
| Multi-rank execution | Explicitly outside final thesis scope |
| Async overlap | Explicitly outside final thesis scope |
| Active PID-Comm provider | Explicitly outside final thesis scope |
| ATiM production kernel | Explicitly outside final thesis scope |
| Energy measurement | Not implemented; no energy claim |
| Automatic heterogeneous placement | Explicitly outside final thesis scope |

## Final experimental sequence

```text
correct UPMEM execution
  -> transport/runtime integration
  -> tasklet and DPU parallelism
  -> kernel/complex-launch experiments
  -> DAG concurrency
  -> bounded locality/slicing experiments
  -> frozen composed executor
  -> launch-aware path cost
  -> bounded physical calibration
  -> frozen G/F/R/U evaluation
  -> accepted result package
```

## Final P6 answer

The P6 experiment supports two different conclusions:

1. Hardware-aware **selection** matters in the tested domain. UPMEM-aware R/U methods
   improve on the conventional FLOP-selected path F, particularly under 4-DPU
   execution.
2. Hardware-aware **adaptive candidate generation** adds no resolved execution benefit
   over hardware-aware reranking under the tested 128-proposal protocol. The primary
   R/U session-inclusive ratio is 1.001343x with descriptive interval
   [0.994836, 1.008509], and the same path is selected in 8/12 cells.

This closes the roadmap. Any new mechanism, topology, numerical policy, or path-search
strategy is a new research study and must not be appended to the accepted thesis
campaign merely because the final result is neutral in some comparison.
