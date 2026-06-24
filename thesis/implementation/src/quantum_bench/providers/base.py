from __future__ import annotations

from typing import Protocol

from quantum_bench.core.records import BenchmarkContext, RouteCapabilities, RouteEstimate, RouteIdentity, RouteProbe, RouteResult, TaskGraph
from quantum_bench.tn.network import TensorNetworkValue


class ExecutionRoute(Protocol):
    name: str
    backend_family: str
    identity: RouteIdentity

    def probe(self) -> RouteProbe:
        ...

    def capabilities(self) -> RouteCapabilities:
        ...

    def can_execute(self, graph: TaskGraph, context: BenchmarkContext) -> tuple[bool, str | None]:
        ...

    def estimate(self, graph: TaskGraph, context: BenchmarkContext) -> RouteEstimate:
        ...

    def prepare(self, graph: TaskGraph, network: TensorNetworkValue, context: BenchmarkContext) -> object:
        ...

    def execute(self, prepared: object, context: BenchmarkContext) -> RouteResult:
        ...
