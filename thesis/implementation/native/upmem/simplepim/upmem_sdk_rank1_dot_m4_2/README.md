# M4.2 SimplePIM rank-1 contraction qualification

This isolated native route qualifies one real-valued rank-1 contraction:

```text
sum_i(int32_a[i] * int32_b[i]) -> int64
```

It uses the pinned SimplePIM APIs on exactly two physical DPUs:

```text
simplepim_scatter(a)
simplepim_scatter(b)
table_zip(a, b)          # virtual descriptor
table_map(pair, product) # int64 product
table_gen_red(product, scalar) # host-mediated partial reduction
```

The route owns a staged copy of SimplePIM and applies only
`simplepim_rank1_hardening.patch`. It is deliberately separate from M3/M4.1.
The executable performs one warmup and five measured repetitions while keeping
one management allocation alive. It uses unique table IDs because the pinned
management allocator does not reclaim table storage.

The pinned initialization kernel uses one tasklet; map and reduction operators
use twelve. Successful evidence requires two observed DPUs, operator API
completion plus registered table metadata checks, bounded table growth, exact
results, and a
confirmed release. Logical transfer bytes exclude SDK arguments, alignment,
control data, and runtime-internal traffic. The MRAM number is a conservative
layout high-water bound, not a verified device-capacity measurement.

Build or parser-only validation:

```sh
make -C native/upmem/simplepim/upmem_sdk_rank1_dot_m4_2 parser
```

Physical execution (after build, on a machine with the UPMEM SDK and hardware):

```sh
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
  make -C native/upmem/simplepim/upmem_sdk_rank1_dot_m4_2 execute
```

The generated response is under the route-local ignored `build/` directory.
It is qualification/functionality evidence only; it must not be used for
speedup, energy, or scaling claims.

Pinned SimplePIM still uses `DPU_ASSERT` around SDK operations. Structured JSON
is therefore guaranteed for host-controlled failures and successful runs; an
SDK abort can terminate without a response and must be classified by the
future bounded subprocess wrapper as a native failure.
