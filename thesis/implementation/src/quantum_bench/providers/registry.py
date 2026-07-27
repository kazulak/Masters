from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from quantum_bench.providers.base import ExecutionRoute
from quantum_bench.providers.exact_tn import (
    CpuTnEinsumExactRoute,
    CpuTnPathReplayFloat64Route,
    CpuTnPathReplayInt8QuantizedRoute,
    QuimbTnExactRoute,
    QuimbTnSlicedExactRoute,
    UpmemTnSdkSimulatorQuantizedRoute,
)
from quantum_bench.providers.full_state import QuestCpuFullStateExactRoute, QuestGpuFullStateExactRoute


UPMEM_PROVIDER_DESCRIPTOR_SCHEMA_VERSION = "upmem_provider_descriptor_v1"
ProviderQualificationScope = Literal[
    "sdk_simulator_execution_contract",
    "resident_hardware_correctness_foundation",
    "not_integrated",
]
ProviderQualificationStatus = Literal["validated", "guarded", "planned"]
ProviderAvailability = Literal["available", "environment_dependent", "planned"]


@dataclass(frozen=True)
class UpmemProviderDescriptor:
    """Fixed metadata for an execution provider, not a route registration."""

    provider_id: str
    display_name: str
    route_id: str | None
    benchmark_surface_id: str | None
    backend_id: str
    qualification_scope: ProviderQualificationScope
    qualification_status: ProviderQualificationStatus
    availability_status: ProviderAvailability
    implementation: str
    schema_version: str = UPMEM_PROVIDER_DESCRIPTOR_SCHEMA_VERSION
    execution_modes: tuple[str, ...] = ()
    correctness_scope: str = "not_applicable"
    performance_claims: bool = False
    notes: tuple[str, ...] = ()

_UPMEM_PROVIDER_DESCRIPTORS = (
    UpmemProviderDescriptor(
        provider_id="upmem_sdk_simulator",
        display_name="UPMEM SDK simulator",
        route_id="upmem_tn_sdk_simulator_quantized",
        benchmark_surface_id=None,
        backend_id="upmem_sdk_simulator_generic_loop",
        qualification_scope="sdk_simulator_execution_contract",
        qualification_status="validated",
        availability_status="available",
        implementation="explicit_sdk",
        execution_modes=("sdk_simulator",),
        correctness_scope="bounded_taskgraph_policy_and_output",
        notes=("Current explicit SDK simulator route; external execution and preflight are required.",),
    ),
    UpmemProviderDescriptor(
        provider_id="upmem_resident_hardware",
        display_name="UPMEM resident hardware",
        route_id=None,
        benchmark_surface_id="upmem_tn_hardware_taskgraph_resident",
        backend_id="upmem_sdk_hardware_taskgraph_resident",
        qualification_scope="resident_hardware_correctness_foundation",
        qualification_status="guarded",
        availability_status="environment_dependent",
        implementation="explicit_sdk_resident",
        execution_modes=("physical_hardware",),
        correctness_scope="one_dpu_mram_resident_taskgraph_correctness",
        notes=("Guarded physical route; hardware and SDK availability are environment-dependent.",),
    ),
    UpmemProviderDescriptor(
        provider_id="simplepim",
        display_name="SimplePIM",
        route_id=None,
        benchmark_surface_id=None,
        backend_id="simplepim",
        qualification_scope="not_integrated",
        qualification_status="planned",
        availability_status="planned",
        implementation="external_adapter",
        execution_modes=("upmem_hardware",),
        notes=("Target adapter; not integrated into the current executor.",),
    ),
    UpmemProviderDescriptor(
        provider_id="pid_comm",
        display_name="PID-Comm",
        route_id=None,
        benchmark_surface_id=None,
        backend_id="pid_comm",
        qualification_scope="not_integrated",
        qualification_status="planned",
        availability_status="planned",
        implementation="external_adapter",
        execution_modes=("multi_dpu",),
        notes=("Target collective and relocation adapter; not currently executed.",),
    ),
    UpmemProviderDescriptor(
        provider_id="atim",
        display_name="ATiM",
        route_id=None,
        benchmark_surface_id=None,
        backend_id="atim",
        qualification_scope="not_integrated",
        qualification_status="planned",
        availability_status="planned",
        implementation="external_adapter",
        execution_modes=("generated_kernel",),
        notes=("Target generated/autotuned local-kernel provider; not integrated.",),
    ),
    UpmemProviderDescriptor(
        provider_id="sparsep",
        display_name="SparseP",
        route_id=None,
        benchmark_surface_id=None,
        backend_id="sparsep",
        qualification_scope="not_integrated",
        qualification_status="planned",
        availability_status="planned",
        implementation="external_adapter",
        execution_modes=("sparse_kernel",),
        notes=("Target sparse-kernel provider; not integrated.",),
    ),
)
UPMEM_PROVIDER_REGISTRY: Mapping[str, UpmemProviderDescriptor] = MappingProxyType(
    {descriptor.provider_id: descriptor for descriptor in _UPMEM_PROVIDER_DESCRIPTORS}
)


def route_registry(root_dir: Path) -> dict[str, ExecutionRoute]:
    routes: list[ExecutionRoute] = [
        CpuTnEinsumExactRoute(),
        CpuTnPathReplayFloat64Route(),
        CpuTnPathReplayInt8QuantizedRoute(),
        QuimbTnExactRoute(),
        QuimbTnSlicedExactRoute(),
        QuestCpuFullStateExactRoute(root_dir),
        QuestGpuFullStateExactRoute(root_dir),
        UpmemTnSdkSimulatorQuantizedRoute(),
    ]
    return {route.name: route for route in routes}
