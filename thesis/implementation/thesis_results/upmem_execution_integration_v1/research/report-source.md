# ATiM Boundary Review for the Execution-System Plan

Audience: thesis implementation and independent-audit agents.
Date: 2026-09-05. Scope: source-only feasibility boundaries, not performance.
Pinned upstream: `SNU-CODElab/atim` artifact commit
`4d6dc5d8cce9647a5a44facd4d237bff7e8e56e8` (2025-05-21).

## Decision

Keep the optional ATiM probe isolated and nonblocking. A generated-C handoff is
visible in upstream code, but neither the thesis numerical policy nor current
SDK compatibility is established by the artifact. Do not run its default
tuning driver or install its system dependencies on the shared ETH server.

## Verified Facts

The artifact documents Ubuntu 20.04 and SDK 2021.3.0, with a much larger
recommended DPU installation than the thesis one-rank scope. Its installation
recipe is not evidence of compatibility with the current ETH environment.
[Pinned README](https://github.com/SNU-CODElab/atim/blob/4d6dc5d8cce9647a5a44facd4d237bff7e8e56e8/README.md)

The tuning driver constructs an int32 workload and defaults to 1000 trials
with 64 trials per iteration. Those defaults do not match the thesis's bounded
probe, and int32 test inputs do not establish shared-scale int8 semantics.
This is not proof that the compiler cannot support other types.
[Pinned driver](https://github.com/SNU-CODElab/atim/blob/4d6dc5d8cce9647a5a44facd4d237bff7e8e56e8/evaluation/atim_autotune.py)

Evaluation can write imported-module source to `upmem.c`. Its time evaluator
uses repeated executions and can zero the reported pre-kernel term when
`ignore_h2d` is enabled. Therefore, its reported total cannot substitute for
the thesis session-inclusive boundary. Generated C must be compiled and
evaluated through the thesis host, data layout, completion and replay checks.
[Pinned evaluation code](https://github.com/SNU-CODElab/atim/blob/4d6dc5d8cce9647a5a44facd4d237bff7e8e56e8/evaluation/base.py)

The repository root license identifies Apache-2.0. No code has been imported;
file-specific and third-party notices must be inspected before incorporation.
[Pinned license](https://github.com/SNU-CODElab/atim/blob/4d6dc5d8cce9647a5a44facd4d237bff7e8e56e8/LICENSE)

## Gap Matrix and Next Gate

| Question | Evidence and confidence | Remaining decision |
| --- | --- | --- |
| Can C source be emitted? | Direct source, high | Test one eligible generated workload after the shape census |
| Is the default experiment bounded appropriately? | Direct driver, high: no | Explicitly cap unique configurations and repeated attempts before any run |
| Does it implement the thesis float32/int8 contract? | Unproven; int32 default only | Exact arithmetic, layout, bounds and CPU replay qualification |
| Does it compile under the current SDK? | Unverified | Isolated build against recorded installed SDK; never replace the system SDK |
| Will it improve a complete route? | No evidence collected | Matched fixed-path, same-policy session-inclusive comparison |

No current evidence justifies bypassing integration or shape attribution.
The optional spike retains the plan's one-day and 32-configuration ceilings.
Repeated timing attempts count separately. A compile failure or absent
full-route benefit can produce a no-go; it must not lead to an SDK rewrite.

## Research Record

Discovery inspected the official README, driver and evaluation source.
Follow-up resolved the artifact commit through GitHub's API and inspected
pinned timing code and license. Web rendering omitted the evaluation source,
so direct first-party raw retrieval was used. A header-only license pipe ended
with curl's closed-output error after displaying the requested header; no
installation or upstream execution occurred.

The planning-progress API was unavailable; scope, follow-up and gap resolution
are recorded here. Research stops because remaining gaps require the later
bounded compile/numerical/physical experiment, not additional broad searching.
Markdown links and source boundaries were structurally checked; no rendered
document or physical-performance verification is claimed.
