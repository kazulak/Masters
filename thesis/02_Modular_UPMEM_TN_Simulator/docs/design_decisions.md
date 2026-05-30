# Design Decisions

These decisions define the intended architecture before implementation. They can
change, but only when experiments or integration constraints justify the change.

## DD-001: Preserve The MVP As A Baseline

Decision: keep `../01_MVP_DenseGEMM` as a reproducible baseline and build the next
architecture separately.

Reasoning: the thesis needs a stable before/after comparison. Refactoring the MVP
too early would make it harder to distinguish real architectural improvement from
baseline drift.

## DD-002: Host Owns Pathfinding And Slicing

Decision: contraction pathfinding, tensor slicing, route eligibility, and memory
planning run on the host CPU.

Reasoning: contraction path search is expensive and irregular. UPMEM DPUs have
tight WRAM limits and are a poor fit for dimension-tree search.

## DD-003: DPUs Execute Numeric Kernels Only

Decision: DPU kernels receive already-sized work units and execute a fixed numeric
operation.

Reasoning: this keeps DPU code small, measurable, and compatible with WRAM limits.
It also avoids implicit dynamic allocation or peer data dependencies.

## DD-004: Route Dispatch Is Mandatory

Decision: every operation goes through a dispatcher, even when only one route is
currently enabled.

Reasoning: the dispatcher is the experiment control point. It makes ablations,
forced routes, negative results, and route logs possible.

## DD-005: Routes Are Isolated Modules

Decision: heuristic, dense, sparse, prototype, and host-collective behavior must be
separate modules.

Reasoning: tensor-network simulation mixes operations with very different data
movement and arithmetic patterns. A single generalized kernel would hide those
differences and weaken the thesis analysis.

## DD-006: Data Format Is A First-Class Parameter

Decision: task graph and route logs include selected data format and scale metadata.

Reasoning: UPMEM DPUs are integer-native. The thesis should compare data formats
explicitly instead of treating int8 quantization as a hardcoded implementation
detail.

## DD-007: No Dynamic Inter-DPU Communication

Decision: algorithms cannot require one DPU to request data from another DPU during
execution.

Reasoning: UPMEM does not support direct DPU-to-DPU communication. Multi-DPU
contractions must use host-mediated collectives.

## DD-008: Every Performance Claim Needs A Correctness Metric

Decision: route and format experiments must report accuracy, not just runtime.

Reasoning: quantization, slicing, and format changes can silently alter amplitudes.
The thesis should report max error, relative error, norm drift, and fidelity where
those metrics apply.

## DD-009: Integration Claims Require Proof

Decision: papers such as SimplePIM, ATiM, SparseP, PRISM, TransPimLib, and PID-Comm
are treated as design influences until the code, license, build path, and benchmark
value are verified.

Reasoning: citing a system and integrating it into this repo are different claims.
The implementation plan should not depend on unverified external integration.
