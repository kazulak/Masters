# Final Results Interpretation

This document provides thesis-safe wording for the accepted research results. It does not
change the frozen experiment packages.

## P6 coefficient interpretation

The final integer coefficient vector is:

```text
[1, 2, 1, 1, 5]
```

for `(H, P, N, M, W)`.

Do **not** describe this as “10% host transfer, 20% host preparation, 10% launches,
10% local movement, and 50% compute time.” The coefficients are ranking parameters in a
normalized surrogate model.

The final diagnostics show strong feature correlation (for example H/P about 0.9995 and
H/M about 0.9881), only 47 distinct measured-pool selection vectors across the 1,001
coefficient tuples, and 16 tuples tied at the best rounded training objective. The vector
therefore does not identify unique physical runtime shares or architectural constants.

Recommended thesis wording:

> The fitted coefficients parameterize the ranking surrogate used to select contraction
> paths. They should not be interpreted as percentages of measured execution time or as
> uniquely identified machine constants. Correlation among several feature totals and
> multiple coefficient tuples producing equivalent best rounded training objectives limit
> coefficient identifiability.

## P6 primary comparison

The primary comparison is R/U on session-inclusive physical execution:

```text
ratio = 1.0013425605931643
descriptive paired-block 95% interval =
[0.9948355448727421, 1.0085087497299605]
same selected path = 8/12 cells
```

Do **not** call this “statistical equivalence” or “statistical parity.” No equivalence
test or optimizer-seed population experiment was performed.

Recommended wording:

> Under the bounded 128-proposal protocol, no additional execution advantage of
> UPMEM-guided adaptive candidate generation over UPMEM-aware reranking was resolved.
> The point estimate slightly favored U, but the descriptive paired-block interval
> spanned one, and R and U selected the same physical path in 8 of 12 evaluation cells.

## UPMEM-aware selection

The relevant overall session-inclusive contrasts are:

```text
F/R = 1.0390296080428785x
F/U = 1.040424568249768x
G/F = 1.2172992035970132x
G/U = 1.2665079983332086x
```

This supports the conclusion that most of the additional hardware-aware gain over F is
already obtained by reranking the conventional trace.

Recommended wording:

> Conventional path optimization already provided a large improvement over greedy
> selection. Applying the calibrated UPMEM objective to path selection provided a further
> physical improvement over the FLOP-selected candidate, while feeding the same objective
> back into adaptive candidate generation produced little additional execution benefit
> under the tested search budget.

## Speedup versus time reduction

If a reported ratio is:

```text
S = T_baseline / T_new
```

then:

```text
speedup = S x
time reduction = 100 * (1 - 1/S) %
```

Never use `(S - 1) * 100` as “percent less time.”

Examples:

| Contrast | Speedup | Equivalent execution-time reduction |
| --- | ---: | ---: |
| F/U overall session-inclusive | 1.040425x | 3.885% |
| F/U 4-DPU session-inclusive | 1.081841x | 7.565% |
| F/U 4-DPU steady wall | 1.115376x | 10.344% |
| G/U overall session-inclusive | 1.266508x | 21.043% |
| G/U 4-DPU steady wall | 1.431470x | 30.142% |

## Why evaluation used 198 rather than 288 attempts

The preregistered maximum was:

```text
12 cells * 4 method roles * 6 blocks = 288
```

but method-role coincidences were deduplicated before physical execution. The frozen
evaluation contained 33 distinct cell/path executions per block:

```text
33 distinct cell/path executions * 6 blocks = 198 physical attempts
```

Method aliases retain their labels but share the same physical observation. Therefore 198
is the complete frozen evaluation, not missing data.

## Generalization wording

The six families are represented in both development and test with distinct instances and
sizes.

Recommended wording:

> The final evaluation tests instance/size transfer within six represented circuit
> families. It is not a family-held-out generalization experiment.

## Search cost wording

Session-inclusive physical timing does not include offline path search. Search wall time
varies substantially by circuit, with some HS18 searches taking on the order of minutes.

Recommended wording:

> Path search is an offline planning cost and is reported separately from physical
> execution. The measured execution speedups therefore describe the benefit after a path
> has been selected; they are not one-shot end-to-end simulation speedups. The planning
> cost can be amortized when the same network structure/execution profile is reused.

## Raw archive reproducibility

The committed P6 audit package contains accepted observations, fits, manifests, final
readout, and hashes of the original physical stage archives. Full re-verification of the
raw physical archive bytes additionally requires the retained archive files themselves.

The standalone release should therefore attach one verified copy of each physical stage
archive as release assets.
