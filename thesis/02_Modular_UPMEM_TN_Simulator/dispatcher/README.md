# Dispatcher

Future home for route selection.

Responsibilities:

- register execution routes;
- ask each route whether it can execute a task;
- estimate route cost;
- select a route under ablation/configuration constraints;
- log selected and rejected routes with reasons;
- fail clearly when no legal route exists.

The dispatcher is the control point for thesis ablation studies. Every operation
should pass through it, even when the only enabled route is dense GEMM.
