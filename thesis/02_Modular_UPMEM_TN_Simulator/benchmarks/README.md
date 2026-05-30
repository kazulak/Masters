# Benchmarks

Future home for benchmark definitions and run metadata.

Benchmark groups should include:

- small sanity circuits such as Bell and GHZ;
- PIMutation-style circuits used by the CPU baseline;
- random circuits with fixed seeds;
- entanglement-heavy circuits;
- sparse or diagonal-heavy circuits;
- permutation-heavy circuits for heuristic routing.

Every benchmark record should include:

- circuit name and parameters;
- seed;
- expected output type;
- baseline reference;
- allowed error metric;
- route/format configuration;
- hardware profile.
