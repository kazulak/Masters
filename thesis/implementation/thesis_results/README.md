# Tracked Thesis Results

This directory contains research evidence packages with different scientific roles.
Do not combine them as though they were one experiment.

## Final thesis result package

### `upmem_cost_guided_path_v1/`

This is the canonical final P6 result and audit package.

It contains:

- accepted initial, feedback-1, feedback-2, and evaluation records;
- the development-only normalization;
- all three 1,001-row coefficient grids and fitted profiles;
- the frozen pretest profile;
- the frozen evaluation method mapping;
- final P6 readout tables;
- model diagnostics;
- archive hashes and attempt accounting;
- the bounded operator/readout tools used for the campaign.

Canonical identities:

```text
qualified P6 software:
2beea27411c16e90ed76988613ddb00bcc09f942

accepted P6 results package:
8df2ebac61bacd08309ea490309be5a8dcb943b2
```

Verify the package with:

```bash
cd upmem_cost_guided_path_v1
sha256sum -c SHA256SUMS
python3 tools/test_p6_readout.py
```

The package is immutable. Publication corrections and thesis-safe interpretation belong
outside it.

## Supporting research packages

| Package | Role |
| --- | --- |
| `upmem_execution_integration_v1/` | Physical execution-integration evidence preceding the final kernel/DAG system |
| `quantized_contraction_policy_v1/` | CPU numerical characterization of shared-scale complex int8 |
| `physical_hardware_mvp_v1/` | Early physical UPMEM functionality evidence |
| `physical_simplepim_taskgraph_m4_5/` | Early physical TaskGraph functionality evidence |

## Historical/superseded packages

| Package | Status |
| --- | --- |
| `current/` | Older mixed research snapshot; not the final thesis result index |
| `planner_v2/` | Model-only planner hypothesis evidence |
| `upmem_path_heuristic_v1/` | Historical path pilot; original raw physical archives were lost |
| `upmem_path_heuristic_generalization_v1/` | Superseded pre-final path generalization work |

Historical packages remain useful for provenance and development chronology. They must not
override the final executor or P6 result identities.

## Physical executor evidence not duplicated here

The detailed accepted P1-P5 physical executor record is
`../docs/upmem_kernel_schedule_system_v1.md`, including archive hashes and evidence
locations for fusion, geometry, DAG scheduling, locality/slicing, composition scaling,
and the scalar microablation.

The final executor is frozen at
`459935f586fdd16c82013838e6d27a12604c3093`.
