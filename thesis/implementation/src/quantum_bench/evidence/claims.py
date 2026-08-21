"""Small, explicit claim admission rules for normalized evidence rows.

This module deliberately adapts the mapping-shaped M5 records rather than
introducing a second evidence schema.  It is the single admission point used
by the M5 report while broader evidence contracts are still being introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class Claim(str, Enum):
    FUNCTIONAL_CORRECTNESS = "functional_correctness"
    PHYSICAL_EXECUTION = "physical_execution"
    TIMING = "timing"
    SCALING = "scaling"
    SPEEDUP = "speedup"
    PATH_ABLATION = "path_ablation"
    NUMERIC_ABLATION = "numeric_ablation"
    ENERGY = "energy"


class ExecutionMode(str, Enum):
    MODEL = "model"
    SDK_SIMULATOR = "sdk_simulator"
    PHYSICAL_HARDWARE = "physical_hardware"
    CPU_HOST = "cpu_host"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()


class ClaimPolicy:
    """Admission policy for M5 normalized-row claims."""

    _DAG_V2_SCHEMA = "contraction_dag_v2"
    _M5_EXACT_ONCE_SCOPE = "host_dag_node_completion_per_route"

    def evaluate_row(self, claim: Claim, row: Mapping[str, Any]) -> ClaimDecision:
        if claim is Claim.FUNCTIONAL_CORRECTNESS:
            return self._functional(row)
        if claim is Claim.PHYSICAL_EXECUTION:
            functional = self._functional(row)
            reasons = list(functional.reasons)
            if self.execution_mode(row) is not ExecutionMode.PHYSICAL_HARDWARE:
                reasons.append("execution mode is not physical hardware")
            if self._engine_class(row) != "upmem":
                reasons.append("physical execution claim requires an UPMEM route")
            return self._decision(reasons)
        if claim is Claim.TIMING:
            functional = self._functional(row)
            reasons = list(functional.reasons)
            mode = self.execution_mode(row)
            if mode in {ExecutionMode.MODEL, ExecutionMode.SDK_SIMULATOR}:
                reasons.append("timing claim rejects modeled or simulator execution")
            origins = self._timing_origins(row)
            if origins and not all(
                self._measured_timing_origin(value) for value in origins
            ):
                reasons.append(
                    "timing metric origin is not a measured timer or counter"
                )
            if self._engine_class(row) == "upmem":
                if row.get("hardware_speedup_applicable") is not True:
                    reasons.append("hardware timing is not speedup-applicable")
                if row.get("timing_is_bringup_only") is not False:
                    reasons.append("timing is marked bringup-only")
            return self._decision(reasons)
        if claim is Claim.SCALING:
            return ClaimDecision(False, ("scaling requires a baseline/candidate pair",))
        if claim is Claim.ENERGY:
            physical = self.evaluate_row(Claim.PHYSICAL_EXECUTION, row)
            reasons = list(physical.reasons)
            energy = self._finite_positive(row.get("energy_joules"))
            if energy is None:
                reasons.append("measured energy_joules is missing or non-positive")
            source = str(row.get("energy_source") or "").strip().lower()
            if source not in {
                "rapl_measured",
                "external_meter_measured",
                "external_power_meter_measured",
                "hardware_counter_measured",
                "sensor_measured",
                "upmem_power_meter_measured",
            }:
                reasons.append("energy source is not an approved measured source")
            status = str(row.get("energy_measurement_status") or "").strip().lower()
            if status not in {
                "measured",
                "measured_valid",
                "measurement_completed",
            }:
                reasons.append("energy measurement status is not measured")
            if not self._energy_field(
                row,
                "energy_measurement_boundary",
                "energy_boundary",
                "energy_scope",
            ):
                reasons.append("energy measurement boundary or scope is missing")
            if not self._energy_field(
                row, "energy_sensor_id", "energy_counter_id", "energy_meter_id"
            ):
                reasons.append("energy sensor or counter identity is missing")
            if not self._positive_energy_interval(row):
                reasons.append("energy measurement interval is missing or non-positive")
            if not self._energy_counter_provenance(row):
                reasons.append(
                    "energy measurement lacks positive samples or counter readings"
                )
            return self._decision(reasons)
        if claim in {Claim.SPEEDUP, Claim.PATH_ABLATION, Claim.NUMERIC_ABLATION}:
            return ClaimDecision(
                False, (f"{claim.value} requires a baseline/candidate pair",)
            )
        return ClaimDecision(False, (f"unsupported claim: {claim.value}",))

    def evaluate_pair(
        self,
        claim: Claim,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        require_repeat_context: bool = True,
    ) -> ClaimDecision:
        if claim is Claim.SPEEDUP:
            return self._speedup_pair(
                baseline,
                candidate,
                require_repeat_context=require_repeat_context,
            )
        if claim is Claim.PATH_ABLATION:
            return self._path_pair(baseline, candidate)
        if claim is Claim.NUMERIC_ABLATION:
            return self._numeric_pair(baseline, candidate)
        if claim is Claim.SCALING:
            return self._scaling_pair(baseline, candidate)
        return ClaimDecision(
            False, (f"pair admission is not defined for {claim.value}",)
        )

    def _speedup_pair(
        self,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        require_repeat_context: bool,
    ) -> ClaimDecision:

        reasons: list[str] = []
        baseline_timing = self.evaluate_row(Claim.TIMING, baseline)
        candidate_timing = self.evaluate_row(Claim.TIMING, candidate)
        reasons.extend(f"baseline: {reason}" for reason in baseline_timing.reasons)
        reasons.extend(f"candidate: {reason}" for reason in candidate_timing.reasons)

        if self._engine_class(baseline) != "cpu":
            reasons.append("speedup baseline must be a CPU same-plan route")
        if self.execution_mode(baseline) is not ExecutionMode.CPU_HOST:
            reasons.append("speedup baseline must be a validated CPU host route")
        if self._engine_class(candidate) != "upmem":
            reasons.append("speedup candidate must be an UPMEM route")
        if self.execution_mode(candidate) is not ExecutionMode.PHYSICAL_HARDWARE:
            reasons.append("speedup candidate must use physical UPMEM hardware")

        baseline_scope = self._timing_scope(baseline)
        candidate_scope = self._timing_scope(candidate)
        if not baseline_scope or not candidate_scope:
            reasons.append("speedup requires a nonempty timing scope for both rows")
        elif baseline_scope != candidate_scope:
            reasons.append("speedup requires matching timing scopes")

        baseline_hashes = self._hashes(baseline)
        candidate_hashes = self._hashes(candidate)
        if baseline_hashes is None or candidate_hashes is None:
            reasons.append("speedup requires semantic, tensor-network, and plan hashes")
        elif baseline_hashes != candidate_hashes:
            reasons.append(
                "speedup requires matching semantic, tensor-network, and plan hashes"
            )
        self._require_dag_pair(reasons, "speedup", baseline, candidate)

        baseline_repeat = self._repeat_id(baseline)
        candidate_repeat = self._repeat_id(candidate)
        if require_repeat_context:
            if not baseline_repeat or not candidate_repeat:
                reasons.append("speedup aggregation requires repeat identifiers")
            elif baseline_repeat != candidate_repeat:
                reasons.append("speedup requires equal repeat identifiers")
        self._require_equal_nonempty(
            reasons,
            "speedup",
            "path",
            self._path(baseline),
            self._path(candidate),
        )
        self._require_equal_nonempty(
            reasons,
            "speedup",
            "numeric policy",
            self._numeric(baseline),
            self._numeric(candidate),
        )
        return self._decision(reasons)

    def _path_pair(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> ClaimDecision:
        reasons = self._comparison_timing_reasons("path ablation", baseline, candidate)
        self._require_equal_executor(reasons, baseline, candidate)
        baseline_hashes = self._hashes(baseline)
        candidate_hashes = self._hashes(candidate)
        if baseline_hashes is None or candidate_hashes is None:
            reasons.append("path ablation requires complete scientific hashes")
        elif baseline_hashes[:2] != candidate_hashes[:2]:
            reasons.append(
                "path ablation requires matching semantic and tensor-network hashes"
            )
        elif baseline_hashes[2] == candidate_hashes[2]:
            reasons.append("path ablation requires distinct contraction-plan hashes")
        self._require_dag_pair(
            reasons, "path ablation", baseline, candidate, require_distinct_v2=True
        )
        self._require_equal_nonempty(
            reasons,
            "path ablation",
            "numeric policy",
            self._numeric(baseline),
            self._numeric(candidate),
        )
        if not self._path(baseline) or not self._path(candidate):
            reasons.append("path ablation requires explicit planner/path identities")
        elif self._path(baseline) == self._path(candidate):
            reasons.append("path ablation requires distinct planner/path identities")
        reasons.extend(self._module_or_legacy_reasons(baseline, candidate, "planner"))
        return self._decision(reasons)

    def _numeric_pair(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> ClaimDecision:
        reasons = self._comparison_timing_reasons(
            "numeric ablation", baseline, candidate
        )
        self._require_equal_executor(reasons, baseline, candidate)
        baseline_hashes = self._hashes(baseline)
        candidate_hashes = self._hashes(candidate)
        if baseline_hashes is None or candidate_hashes is None:
            reasons.append("numeric ablation requires complete scientific hashes")
        elif baseline_hashes != candidate_hashes:
            reasons.append("numeric ablation requires matching full scientific hashes")
        self._require_dag_pair(reasons, "numeric ablation", baseline, candidate)
        self._require_equal_nonempty(
            reasons,
            "numeric ablation",
            "path",
            self._path(baseline),
            self._path(candidate),
        )
        if not self._is_float32_policy(baseline):
            reasons.append(
                "numeric ablation baseline must be an explicit float32 policy"
            )
        if not self._host_packed_int8(candidate):
            reasons.append(
                "numeric ablation candidate must be explicit host-packed Int8 with packed MRAM transport"
            )
        reasons.extend(self._module_or_legacy_reasons(baseline, candidate, "numeric"))
        return self._decision(reasons)

    def _scaling_pair(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> ClaimDecision:
        reasons = self._comparison_timing_reasons("scaling", baseline, candidate)
        self._require_equal_executor(reasons, baseline, candidate)
        for role, row in (("baseline", baseline), ("candidate", candidate)):
            if self._engine_class(row) != "upmem":
                reasons.append(f"scaling {role} must be an UPMEM route")
            if self.execution_mode(row) is not ExecutionMode.PHYSICAL_HARDWARE:
                reasons.append(f"scaling {role} must use physical hardware")
            if not self._fully_active_topology(row):
                reasons.append(f"scaling {role} topology is not fully active")
        baseline_hashes = self._hashes(baseline)
        candidate_hashes = self._hashes(candidate)
        if baseline_hashes is None or candidate_hashes is None:
            reasons.append("scaling requires complete scientific hashes")
        elif baseline_hashes != candidate_hashes:
            reasons.append("scaling requires matching full scientific hashes")
        self._require_dag_pair(reasons, "scaling", baseline, candidate)
        self._require_equal_nonempty(
            reasons, "scaling", "path", self._path(baseline), self._path(candidate)
        )
        self._require_equal_nonempty(
            reasons,
            "scaling",
            "numeric policy",
            self._numeric(baseline),
            self._numeric(candidate),
        )
        baseline_topology = self._topology_identity(baseline)
        candidate_topology = self._topology_identity(candidate)
        if baseline_topology is None or candidate_topology is None:
            reasons.append("scaling requires complete topology identities")
        elif baseline_topology == candidate_topology:
            reasons.append("scaling requires distinct topology identities")
        reasons.extend(self._module_or_legacy_reasons(baseline, candidate, "topology"))
        return self._decision(reasons)

    @staticmethod
    def execution_mode(row: Mapping[str, Any]) -> ExecutionMode:
        marker_mode = ClaimPolicy._marker_execution_mode(row)
        if marker_mode is not None:
            return marker_mode
        explicit = str(row.get("execution_mode") or "").strip().lower()
        if explicit in {"model", "modeled", "analytic_model"}:
            return ExecutionMode.MODEL
        if explicit in {"sdk_simulator", "simulator", "simulation"}:
            return ExecutionMode.SDK_SIMULATOR
        if explicit in {"physical", "physical_hardware", "hardware"}:
            return ExecutionMode.PHYSICAL_HARDWARE
        if explicit in {"cpu", "cpu_host", "host"}:
            return ExecutionMode.CPU_HOST
        target = str(row.get("target_observed") or "").strip().lower()
        if target in {"physical_hardware", "physical", "hardware"}:
            return ExecutionMode.PHYSICAL_HARDWARE
        if target in {"sdk_simulator", "simulator"}:
            return ExecutionMode.SDK_SIMULATOR
        engine = str(
            row.get("engine_id")
            or row.get("execution_engine")
            or row.get("engine")
            or row.get("backend_family")
            or row.get("backend_id")
            or row.get("route_id")
            or ""
        ).lower()
        if "model" in engine:
            return ExecutionMode.MODEL
        if "simulator" in engine:
            return ExecutionMode.SDK_SIMULATOR
        if "cpu" in engine or "numpy" in engine or row.get("execution_target") == "cpu":
            return ExecutionMode.CPU_HOST
        return ExecutionMode.UNKNOWN

    def _functional(self, row: Mapping[str, Any]) -> ClaimDecision:
        reasons: list[str] = []
        status = str(row.get("status", "completed")).lower()
        if status not in {"completed", "passed", "success", "verified"}:
            reasons.append("row status is not completed")
        if not self._has_runtime(row):
            reasons.append("runtime is missing or non-positive")
        engine_class = self._engine_class(row)
        if engine_class == "cross_algorithm":
            if str(row.get("validation_status") or "").lower() not in {
                "passed",
                "passed_native_status",
                "passed_runtime_only",
            }:
                reasons.append("cross-algorithm validation did not pass")
            return self._decision(reasons)
        if engine_class not in {"cpu", "upmem"}:
            reasons.append("engine class is not claim-admissible")
            return self._decision(reasons)
        if str(row.get("scientific_validation_status") or "").lower() != "passed":
            reasons.append("scientific validation did not pass")
        if self._is_current_m5_dag_v2(row):
            if row.get("host_dag_node_completion_coverage") is not True:
                reasons.append(
                    "host DAG completion coverage was not verified; "
                    "host DAG coverage is not native kernel exact-once evidence"
                )
            if row.get("exact_once_scope") != self._M5_EXACT_ONCE_SCOPE:
                reasons.append(
                    "exact_once_scope is not host_dag_node_completion_per_route; "
                    "host DAG coverage is not native kernel exact-once evidence"
                )
        elif row.get("exact_once") is not True:
            reasons.append("exact-once execution was not verified")
        if row.get("no_fallback_used") is not True:
            reasons.append("no-fallback contract was not verified")
        if engine_class == "upmem":
            if (
                self._is_current_m5_dag_v2(row)
                and self.execution_mode(row) is ExecutionMode.PHYSICAL_HARDWARE
                and row.get("native_identity_verified") is not True
            ):
                reasons.append(
                    "DAG-v2 physical UPMEM/M5 admission requires "
                    "native_identity_verified=True"
                )
            if row.get("target_observed") != "physical_hardware":
                reasons.append("UPMEM target was not observed as physical hardware")
            if row.get("hardware_allocation_verified") is not True:
                reasons.append("UPMEM hardware allocation was not verified")
            if row.get("native_kernel_executed") is not True:
                reasons.append("UPMEM native kernel execution was not verified")
            if row.get("hardware_kernel_executed") is not True:
                reasons.append("UPMEM hardware kernel execution was not verified")
            if not self._explicitly_false(
                row, "simulator", "simulator_kernel_executed"
            ):
                reasons.append("UPMEM simulator state is not explicitly false")
            if not self._explicitly_false(row, "cpu_fallback", "cpu_fallback_used"):
                reasons.append("UPMEM CPU fallback state is not explicitly false")
            if not self._release_succeeded(row):
                reasons.append("UPMEM resource release was not verified")
        return self._decision(reasons)

    @classmethod
    def _is_current_m5_dag_v2(cls, row: Mapping[str, Any]) -> bool:
        return row.get("contraction_dag_schema_version") == cls._DAG_V2_SCHEMA

    @staticmethod
    def _decision(reasons: list[str]) -> ClaimDecision:
        return ClaimDecision(not reasons, tuple(reasons))

    def _comparison_timing_reasons(
        self,
        label: str,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        baseline_timing = self.evaluate_row(Claim.TIMING, baseline)
        candidate_timing = self.evaluate_row(Claim.TIMING, candidate)
        reasons.extend(f"baseline: {reason}" for reason in baseline_timing.reasons)
        reasons.extend(f"candidate: {reason}" for reason in candidate_timing.reasons)
        self._require_equal_nonempty(
            reasons,
            label,
            "case",
            self._case_id(baseline),
            self._case_id(candidate),
        )
        self._require_equal_nonempty(
            reasons,
            label,
            "engine",
            self._engine_id(baseline),
            self._engine_id(candidate),
        )
        self._require_equal_nonempty(
            reasons,
            label,
            "timing scope",
            self._timing_scope(baseline),
            self._timing_scope(candidate),
        )
        self._require_equal_nonempty(
            reasons,
            label,
            "repeat identifier",
            self._repeat_id(baseline),
            self._repeat_id(candidate),
        )
        return reasons

    @staticmethod
    def _require_equal_nonempty(
        reasons: list[str], label: str, field: str, baseline: str, candidate: str
    ) -> None:
        if not baseline or not candidate:
            reasons.append(f"{label} requires complete {field} identities")
        elif baseline != candidate:
            reasons.append(f"{label} requires matching {field} identities")

    @staticmethod
    def _require_equal_executor(
        reasons: list[str],
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> None:
        baseline_hash = str(baseline.get("executor_config_hash") or "").strip()
        candidate_hash = str(candidate.get("executor_config_hash") or "").strip()
        if not baseline_hash or not candidate_hash:
            reasons.append(
                "comparison requires complete executor_config_hash identities"
            )
        elif baseline_hash != candidate_hash:
            reasons.append("comparison requires equal executor_config_hash identities")

    @staticmethod
    def _dag_identity(row: Mapping[str, Any]) -> tuple[str, str] | None:
        """Return a valid v2 or legacy DAG identity for pair admission."""

        schema = str(row.get("contraction_dag_schema_version") or "").strip()
        dag_hash = str(row.get("contraction_dag_hash") or "").strip()
        if schema == "contraction_dag_v2":
            return (schema, dag_hash) if dag_hash else None
        if not schema and not dag_hash:
            return ("legacy_unversioned", "")
        if schema.startswith("legacy") and not dag_hash:
            return (schema, "")
        return None

    def _require_dag_pair(
        self,
        reasons: list[str],
        label: str,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        require_distinct_v2: bool = False,
    ) -> None:
        baseline_identity = self._dag_identity(baseline)
        candidate_identity = self._dag_identity(candidate)
        if baseline_identity is None or candidate_identity is None:
            reasons.append(
                f"{label} requires complete valid contraction DAG identities"
            )
            return
        baseline_schema, baseline_hash = baseline_identity
        candidate_schema, candidate_hash = candidate_identity
        if baseline_schema != candidate_schema:
            reasons.append(
                f"{label} requires matching contraction DAG schema; legacy and v2 rows cannot mix"
            )
            return
        if baseline_schema != "contraction_dag_v2":
            return
        if require_distinct_v2:
            if baseline_hash == candidate_hash:
                reasons.append(f"{label} requires distinct contraction DAG hashes")
        elif baseline_hash != candidate_hash:
            reasons.append(f"{label} requires matching contraction DAG hashes")

    def _module_or_legacy_reasons(
        self,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        expected_role: str,
    ) -> list[str]:
        baseline_modules = baseline.get("route_modules")
        candidate_modules = candidate.get("route_modules")
        if not isinstance(baseline_modules, Mapping) or not isinstance(
            candidate_modules, Mapping
        ):
            return ["comparison requires complete route_modules on both rows"]
        required_roles = {
            "tensor_network",
            "planner",
            "numeric",
            "executor",
            "topology",
        }
        baseline_roles = set(baseline_modules)
        candidate_roles = set(candidate_modules)
        if not required_roles.issubset(baseline_roles) or not required_roles.issubset(
            candidate_roles
        ):
            return ["comparison route_modules are incomplete"]
        if baseline_roles != candidate_roles:
            return ["comparison route_modules define different role sets"]
        changed = {
            role
            for role in baseline_roles
            if baseline_modules.get(role) != candidate_modules.get(role)
        }
        if changed != {expected_role}:
            return [
                "comparison route_modules must change exactly "
                f"{expected_role}; observed={','.join(sorted(changed)) or 'none'}"
            ]
        return []

    @staticmethod
    def _case_id(row: Mapping[str, Any]) -> str:
        return str(
            row.get("case_id")
            or row.get("workload_id")
            or row.get("quantum_case")
            or ""
        ).strip()

    @staticmethod
    def _engine_id(row: Mapping[str, Any]) -> str:
        return str(
            row.get("engine_id")
            or row.get("execution_engine")
            or row.get("engine")
            or row.get("backend_family")
            or row.get("backend_id")
            or row.get("route_id")
            or ""
        ).strip()

    @staticmethod
    def _path(row: Mapping[str, Any]) -> str:
        return str(
            row.get("path_variant_id")
            or row.get("planner_id")
            or row.get("path_strategy")
            or row.get("path_id")
            or row.get("contraction_path")
            or ""
        ).strip()

    @staticmethod
    def _numeric(row: Mapping[str, Any]) -> str:
        return str(
            row.get("numeric_policy")
            or row.get("numeric_policy_id")
            or row.get("numeric_mode")
            or row.get("quantization_mode")
            or row.get("numeric_arithmetic")
            or ""
        ).strip()

    def _is_float32_policy(self, row: Mapping[str, Any]) -> bool:
        values = self._policy_identifiers(row)
        return any("float32" in value.lower() for value in values)

    def _host_packed_int8(self, row: Mapping[str, Any]) -> bool:
        policy_ids = self._policy_identifiers(row)
        explicit_policy = any(
            "host" in value.lower()
            and "packed" in value.lower()
            and "int8" in value.lower()
            for value in policy_ids
        )
        packed_values = self._nested_values(row, "packed_int8_transfer")
        transport_values = self._nested_values(row, "numeric_transport")
        return (
            explicit_policy
            and bool(packed_values)
            and all(value is True for value in packed_values)
            and bool(transport_values)
            and all(str(value) == "host_packed_int8_mram" for value in transport_values)
        )

    @staticmethod
    def _policy_identifiers(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(row[key]).strip()
            for key in (
                "numeric_policy_id",
                "numeric_policy",
                "numeric_mode",
                "quantization_mode",
            )
            if row.get(key) is not None and str(row[key]).strip()
        )

    @staticmethod
    def _nested_values(row: Mapping[str, Any], key: str) -> tuple[Any, ...]:
        values: list[Any] = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                if key in value:
                    values.append(value[key])
                for nested_key in (
                    "engine_metadata",
                    "numeric_metadata",
                    "task_metrics",
                    "transfer",
                    "metadata",
                ):
                    if nested_key in value:
                        visit(value[nested_key])
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(row)
        return tuple(values)

    @staticmethod
    def _topology_identity(row: Mapping[str, Any]) -> tuple[int, int, int] | None:
        rank_count = ClaimPolicy._positive_int(
            row.get("rank_count")
            or row.get("observed_rank_count")
            or row.get("requested_rank_count")
        )
        total = ClaimPolicy._positive_int(
            row.get("total_dpu_count")
            or row.get("allocated_total_dpu_count")
            or row.get("allocated_dpu_count")
            or row.get("requested_dpu_count")
        )
        local = ClaimPolicy._positive_int(
            row.get("local_dpu_count")
            or row.get("dpus_per_rank")
            or row.get("requested_dpus_per_rank")
            or row.get("allocated_dpus_per_rank")
        )
        if local is None and rank_count and total and total % rank_count == 0:
            local = total // rank_count
        if rank_count is None or total is None or local is None:
            return None
        return local, rank_count, total

    @staticmethod
    def _fully_active_topology(row: Mapping[str, Any]) -> bool:
        topology = ClaimPolicy._topology_identity(row)
        metadata = row.get("engine_metadata")
        if topology is None or not isinstance(metadata, Mapping):
            return False
        _local, ranks, total = topology
        dpu_ids = metadata.get("active_dpu_ids")
        rank_ids = metadata.get("active_rank_indices") or metadata.get(
            "active_rank_ids"
        )
        active_dpus = (
            len(set(dpu_ids))
            if isinstance(dpu_ids, (list, tuple, set, frozenset))
            else ClaimPolicy._positive_int(metadata.get("active_dpu_count"))
        )
        active_ranks = (
            len(set(rank_ids))
            if isinstance(rank_ids, (list, tuple, set, frozenset))
            else ClaimPolicy._positive_int(metadata.get("active_rank_count"))
        )
        return active_dpus == total and active_ranks == ranks

    @staticmethod
    def _positive_int(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _energy_field(row: Mapping[str, Any], *keys: str) -> Any:
        energy = row.get("energy")
        for key in keys:
            value = row.get(key)
            if value is not None and value != "":
                return value
            if isinstance(energy, Mapping):
                value = energy.get(key)
                if value is not None and value != "":
                    return value
        return None

    @staticmethod
    def _positive_energy_interval(row: Mapping[str, Any]) -> bool:
        interval = ClaimPolicy._energy_field(
            row,
            "energy_measurement_interval_s",
            "energy_interval_s",
            "energy_duration_s",
        )
        if ClaimPolicy._finite_positive(interval) is not None:
            return True
        for start_key, end_key in (
            ("energy_measurement_start_s", "energy_measurement_end_s"),
            ("energy_start_time_s", "energy_end_time_s"),
            ("energy_interval_start_s", "energy_interval_end_s"),
        ):
            start = ClaimPolicy._finite_number(
                ClaimPolicy._energy_field(row, start_key)
            )
            end = ClaimPolicy._finite_number(ClaimPolicy._energy_field(row, end_key))
            if start is not None and end is not None and end > start:
                return True
        return False

    @staticmethod
    def _energy_counter_provenance(row: Mapping[str, Any]) -> bool:
        sample_count = ClaimPolicy._energy_field(
            row,
            "energy_sample_count",
            "energy_measurement_sample_count",
            "energy_counter_sample_count",
            "energy_counter_count",
        )
        if ClaimPolicy._finite_positive(sample_count) is not None:
            return True
        for before_key, after_key in (
            ("energy_counter_before", "energy_counter_after"),
            ("energy_counter_before_uj", "energy_counter_after_uj"),
            ("energy_before_uj", "energy_after_uj"),
            ("energy_reading_before", "energy_reading_after"),
        ):
            before = ClaimPolicy._finite_number(
                ClaimPolicy._energy_field(row, before_key)
            )
            after = ClaimPolicy._finite_number(
                ClaimPolicy._energy_field(row, after_key)
            )
            if before is not None and after is not None and after > before:
                return True
        return False

    @staticmethod
    def _marker_execution_mode(row: Mapping[str, Any]) -> ExecutionMode | None:
        simulator_flags = (
            "simulator",
            "simulator_kernel_executed",
            "sdk_simulator_executed",
        )
        if any(
            value is True
            for key in simulator_flags
            for value in ClaimPolicy._nested_values(row, key)
        ):
            return ExecutionMode.SDK_SIMULATOR
        if any(
            value is True
            for key in ("modeled", "modeled_only")
            for value in ClaimPolicy._nested_values(row, key)
        ):
            return ExecutionMode.MODEL
        for key in (
            "simulation_method",
            "metric_origin",
            "timing_origin",
            "runtime_origin",
            "execution_origin",
            "simulation_origin",
        ):
            for value in ClaimPolicy._nested_values(row, key):
                normalized = str(value).strip().lower()
                tokens = normalized.replace("-", "_").replace(".", "_").split("_")
                if "simulator" in tokens or "simulation" in tokens:
                    return ExecutionMode.SDK_SIMULATOR
                if "model" in tokens or "modeled" in tokens:
                    return ExecutionMode.MODEL
        for value in ClaimPolicy._nested_values(row, "simulation_model_id"):
            normalized = str(value).strip().lower()
            tokens = normalized.replace("-", "_").replace(".", "_").split("_")
            if "simulator" in tokens or "simulation" in tokens:
                return ExecutionMode.SDK_SIMULATOR
        return None

    @staticmethod
    def _timing_origins(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(
            value
            for key in (
                "timing_origin",
                "runtime_origin",
                "metric_origin",
                "runtime_metric_origin",
                "timer_source",
                "timing_source",
            )
            for value in ClaimPolicy._nested_values(row, key)
        )

    @staticmethod
    def _measured_timing_origin(value: Any) -> bool:
        normalized = str(value).strip().lower().split(".")[-1]
        return normalized in {
            "host_timer",
            "host_wall_timer",
            "host_wall_clock",
            "runtime_counter",
            "device_counter",
            "device_cycles",
            "hardware_counter",
            "perf_counter",
            "python_perf_counter",
            "monotonic_clock",
            "wall_clock",
            "measured",
            "measured_timer",
        }

    @staticmethod
    def _engine_class(row: Mapping[str, Any]) -> str:
        value = str(
            row.get("engine_id")
            or row.get("execution_engine")
            or row.get("engine")
            or row.get("backend_family")
            or row.get("backend_id")
            or row.get("route_id")
            or ""
        ).lower()
        if "quimb" in value or "quest" in value:
            return "cross_algorithm"
        if "upmem" in value or "dpu" in value:
            return "upmem"
        if "cpu" in value or "numpy" in value or row.get("execution_target") == "cpu":
            return "cpu"
        return "other"

    @staticmethod
    def _timing_scope(row: Mapping[str, Any]) -> str:
        value = row.get("timing_scope") or row.get("timing_contract") or ""
        return str(value).strip()

    @staticmethod
    def _repeat_id(row: Mapping[str, Any]) -> str:
        for key in ("repeat_id", "repetition", "measurement_id"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _hashes(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
        hashes = tuple(
            str(row.get(key) or "").strip()
            for key in (
                "circuit_semantics_hash",
                "tensor_network_hash",
                "contraction_plan_hash",
            )
        )
        return hashes if all(hashes) else None

    @staticmethod
    def _has_runtime(row: Mapping[str, Any]) -> bool:
        for key in (
            "total_route_time_s",
            "total_wall_time_s",
            "timing_s",
            "runtime_s",
            "execution_time_s",
            "steady_state_graph_execution_s",
            "route_time_s",
        ):
            if ClaimPolicy._finite_positive(row.get(key)) is not None:
                return True
        for container_key in ("timing", "timings"):
            container = row.get(container_key)
            if isinstance(container, Mapping):
                for key in ("total_time_s", "total_s"):
                    if ClaimPolicy._finite_positive(container.get(key)) is not None:
                        return True
        return False

    @staticmethod
    def _finite_positive(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _explicitly_false(row: Mapping[str, Any], *keys: str) -> bool:
        values = [row[key] for key in keys if key in row]
        return bool(values) and all(value is False for value in values)

    @staticmethod
    def _release_succeeded(row: Mapping[str, Any]) -> bool:
        if any(
            row.get(key) is True
            for key in (
                "release_succeeded",
                "hardware_release_verified",
                "release_verified",
                "release_confirmed",
                "session_release_verified",
            )
        ):
            return True
        return str(row.get("resource_release_status") or "").lower() in {
            "released",
            "passed",
            "verified",
            "clean",
        }
