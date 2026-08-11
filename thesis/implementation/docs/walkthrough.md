# Current Milestone Walkthrough

The authoritative milestone status is maintained in
[README.md](../README.md#current-milestone-status).

M4.1--M4.5 are physically accepted bounded milestones. M4.5 is the current
accepted SimplePIM-managed baseline: it provides the bounded descriptor-driven
shared runtime and one- and two-DPU functionality evidence.

M4.6 is implementation-ready locally but is not complete. Physical acceptance
requires the `1/2/4/8/16` ETH tasklet sweep to pass. The 336-row Aug-10 run was
one tasklet path/quantization evidence run, not a scaling run.

M5.1 output partitioning is under development and has not been physically
accepted. M5.2 host-contracted reduction and M5.3 PID-Comm are pending.
PID-Comm's pinned UPMEM SDK 2021.3 differs from the thesis ETH SDK 2023.1 and
requires qualification; no compatibility claim is made.

SimplePIM remains central: it supplies the accepted bounded management/operator
surface and the M4.5 managed baseline. These milestones do not claim speedup,
energy, general distributed execution, or completed scaling evidence.
