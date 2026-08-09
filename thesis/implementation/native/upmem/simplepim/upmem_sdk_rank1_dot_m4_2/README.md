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

## External operand file

The default `execute` target keeps the deterministic M4.2 operands unchanged.
For a bounded input-transport check, invoke the staged host directly with
`--operands-file PATH`. The file must be exactly 512 bytes:

```text
bytes 0..255:   left int8[256]
bytes 256..511: right int8[256]
```

The host converts these values to the existing int32 SimplePIM table contract,
computes the int64 CPU reference, hashes the exact 512 input bytes, and reports
`external_operand_transport`, `operand_input_length_bytes`, and
`operand_input_hash`. Missing, short, trailing, or unreadable files fail before
physical allocation; the host never falls back to the fixed operands when this
option is present.

After `make -C native/upmem/simplepim/upmem_sdk_rank1_dot_m4_2 build`, run on
physical hardware with the normal guards and an explicit response path:

```sh
UPMEM_ALLOW_PHYSICAL_HARDWARE=1 \
  build/simplepim_rank1_dot_m4_2/staged/benchmarks/rank1_dot/bin/rank1_dot_host \
  --mode execute \
  --operands-file /path/to/exactly-512-byte-operands.bin \
  --response build/simplepim_rank1_dot_m4_2/external_execute_response.json \
  --stage-manifest build/simplepim_rank1_dot_m4_2/staged/simplepim_stage_manifest.json
```

This remains the same two-DPU, one-operator-fixture qualification route; it
does not add general tensor shapes or TaskGraph transport.

Pinned SimplePIM still uses `DPU_ASSERT` around SDK operations. Structured JSON
is therefore guaranteed for host-controlled failures and successful runs; an
SDK abort can terminate without a response and must be classified by the
future bounded subprocess wrapper as a native failure.
