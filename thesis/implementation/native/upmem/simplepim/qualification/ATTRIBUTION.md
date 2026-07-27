# SimplePIM qualification attribution

This qualification adapts the vector-addition example from the pinned
SimplePIM submodule at commit `1d639c53532555f01e9f71d872e7712b166d6cba`.
The adapted host keeps the upstream `scatter -> virtual zip -> map -> gather`
sequence, with fixed one-DPU inputs for a physical bring-up probe. The
upstream SimplePIM library sources are copied into the runner workdir before
building; the submodule is never modified.

Upstream project: SimplePIM, Jinfan Chen et al., PACT 2023.

Only the small VA parameter/map surface and an initialization shim are owned
by this qualification. The operational library implementation remains the
staged upstream source.

`patches/simplepim-map-unroll-rest.patch` is a thesis-owned, attributed local
correctness patch for two self-referential initializers in the pinned
`lib/processing/map/MapProcessing.h`. The runner applies it only to the staged
copy under the caller's `workdir/build`; the pinned submodule remains
unchanged.

SimplePIM's upstream host helpers use `DPU_ASSERT`, which can terminate the
process before control returns to this qualification's centralized cleanup.
The qualification does not replace or reimplement those upstream internals.
A nonzero native exit without a valid host result is therefore reported as a
failed probe with DPU release unconfirmed; it is never accepted as qualified.
