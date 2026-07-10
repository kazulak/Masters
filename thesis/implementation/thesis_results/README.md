# Tracked Thesis Results

`current/` is the selected compact evidence snapshot used for thesis writing.
It contains normalized records, resolved suites, source CSV tables, plots,
checksums, and a generated interpretation. It is created only from a successful
research pack on a clean Git commit:

```bash
make thesis-run
make thesis-promote
make thesis-verify
```

`releases/<name>/` may contain immutable named milestones created with:

```bash
make thesis-release NAME=<milestone>
```

Generated `runs/` remain ignored and may be pruned after promotion. Raw tensor
dumps and native build output are intentionally not tracked here.
