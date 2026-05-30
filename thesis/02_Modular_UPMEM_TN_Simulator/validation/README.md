# Validation

Future home for correctness checks and regression tests.

Responsibilities:

- compare UPMEM results against host reference results;
- compute max absolute error, relative error, norm drift, and fidelity where
  applicable;
- validate task graph schema assumptions;
- test route fallback behavior;
- test forced-route failure messages;
- verify selected and rejected route records;
- verify unsupported route-format pairs fail before execution;
- keep small circuits fast enough for regular regression runs.

Correctness checks should be run before performance experiments. Performance
without an error metric is not useful for this thesis.
