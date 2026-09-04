# Quantized Contraction Policy v1

`complex_int8_shared_scale_v1` is a CPU-only numerical-analysis policy for
the existing `ContractionDAG`. It is deliberately separate from the accepted
runtime policy identifiers and does not change the float32 route, packed
operation transport, ABI, evidence schema, or DPU kernel.

## Mathematical policy

For a finite complex tensor `X`, define

```text
a(X) = max(max_j |Re(X_j)|, max_j |Im(X_j)|)
s(X) = 1                 when a(X) = 0
       a(X) / 127        otherwise
q(v; s) = clip(round_to_nearest_even(v / s), -127, 127)
```

The encoded planes are

```text
q_real = q(Re(X); s(X))
q_imag = q(Im(X); s(X))
```

Both planes are `int8`, but `-128` is forbidden. The scale is the historical
Python-float/float64 representation. Non-finite inputs and scale underflow are
rejected. An all-zero tensor has scale exactly `1.0`.

One scale is shared by real and imaginary components because operands

```text
A ~= s_A (A_r + i A_i)
B ~= s_B (B_r + i B_i)
```

produce one product scale and four real integer contractions:

```text
Z_real = A_r @ B_r - A_i @ B_i
Z_imag = A_r @ B_i + A_i @ B_r
C_hat  = (s_A s_B) (Z_real + i Z_imag)
```

This minimizes scale metadata and keeps a later DPU implementation simple.
It can lose precision when real and imaginary dynamic ranges differ strongly;
v1 measures that effect rather than adding separate scales.

## Replay and references

At every binary contraction, current operands are materialized according to
the existing tensor views, independently quantized, contracted with explicit
int64 four-product accumulation, and dequantized to the historical `complex64`
intermediate. A produced intermediate is quantized again whenever a later
contraction consumes it. Explicit DAG reductions remain deterministic
host-side `complex64` reductions. Original complex128 tensor inputs retain
their source precision until their first int8 quantization; only produced
intermediates follow the dequantized complex64 storage policy.

The analysis records two different node errors:

- Local sensitivity quantizes only the corresponding same-DAG float32
  operands. It measures the node in isolation.
- Cumulative replay consumes prior quantized/dequantized intermediates. It
  measures propagation through the fixed contraction order.

Final outputs are compared, without phase alignment, to both the same-DAG
float32 replay and the existing complex128 DAG reference. The float32 versus
complex128 floor is retained separately.

For a contraction extent `K`, the conservative full-component accumulator
bound is

```text
2 K 127^2.
```

The software reference preflights that bound against int64. A pure helper also
reports whether the whole `K` extent fits int32; no arbitrary `K` limit is
used. This does not claim bit identity with a future tiled DPU accumulator
hierarchy.

For an unclipped tensor with `n` complex elements, the rounding model gives

```text
||E_X||_F <= sqrt(n / 2) s_X.
```

The node diagnostic applies the conservative matrixized bound

```text
||C - C_hat||_F
  <= ||E_A||_F ||B||_F + ||A_hat||_F ||E_B||_F.
```

The observed value paired with this bound is the pre-`complex64` Frobenius
error. Phase-sensitive post-cast max-absolute, relative-L2, and norm-drift
metrics remain separate. If a valid DAG contraction sums a label present in
only one operand, the implementation includes the corresponding broadcast
replication factor in the matrixized Frobenius bound.

## Logical cost facts

For `n` complex values, float32 complex storage is `8n` bytes. A shared-scale
int8 tensor is `2n + 8` bytes with the historical float64 scale, so nominal
compression approaches 4x for large tensors. These are logical encoded sizes,
not measured H2D bytes, MRAM traffic, WRAM traffic, or physical transfer
reductions.

The policy helper exposes scale reductions, quantization events, dequantized
outputs, integer multiply-accumulate counts, metadata bytes, and accumulator
requirements for future path scoring. It does not import planning code or
assign path weights. Numerical error is not the future `E_num` execution-cost
term.

## CPU-only analysis

Run from `thesis/implementation`:

```bash
PYTHONPATH=src ../.venv/bin/python scripts/analyze_quantized_contractions.py \
  --output thesis_results/quantized_contraction_policy_v1
```

The deterministic suite is Bell2, Stress4, GHZ18, HS18, and Stress18, each
with its existing greedy contraction path. The script writes only:

- `quantization_summary.json`
- `quantization_nodes.csv`
- `quantization_circuits.csv`

The characterization is accuracy-unqualified. It establishes no UPMEM
speedup, physical transfer reduction, tasklet/DPU scaling result, or claim
that int8 is an acceptable default.
