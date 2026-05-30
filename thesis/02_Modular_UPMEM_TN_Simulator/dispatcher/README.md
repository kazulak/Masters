# Dispatcher

Future home for route selection.

Responsibilities:

- register execution routes;
- ask each route whether it can execute a task;
- estimate route cost;
- call route `prepare` before route `execute`;
- select a route under ablation/configuration constraints;
- log selected and rejected routes with reasons;
- fail clearly when no legal route exists.

The dispatcher is the control point for thesis ablation studies. Every operation
should pass through it, even when the only enabled route is raw dense replay.

Initial dispatch should use explicit rules. Mature dispatch should use
`CostOracle` estimates that include transfer, preparation, conversion, reduction,
and numerical-error penalties.
