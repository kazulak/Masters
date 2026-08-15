"""Immutable target-estimate sidecars for scientific plans.

The scientific TaskGraph deliberately does not own target-specific estimates.
This module keeps those estimates keyed by the scientific plan identity and
task ID while making their metric provenance explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION = "target_estimate_sidecar_v1"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw(item[1]) for item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class TargetMetricSpec:
    name: str
    unit: str
    origin: str
    scope: str

    def to_json_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "unit": self.unit,
            "origin": self.origin,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class TargetEstimateRow:
    task_id: str
    input_tensor_ids: tuple[str, str]
    output_tensor_id: str
    values: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TargetEstimateRow":
        reserved = {
            "task_id",
            "input_tensor_ids",
            "output_tensor_id",
            "scientific_plan_hash",
            "target_id",
            "model_id",
            "schema_version",
            "sidecar_metadata",
            "metric_provenance",
        }
        values = tuple(
            (str(key), _freeze(value))
            for key, value in sorted(row.items(), key=lambda pair: str(pair[0]))
            if key not in reserved
        )
        return cls(
            task_id=str(row["task_id"]),
            input_tensor_ids=tuple(str(item) for item in row["input_tensor_ids"]),
            output_tensor_id=str(row["output_tensor_id"]),
            values=values,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "input_tensor_ids": list(self.input_tensor_ids),
            "output_tensor_id": self.output_tensor_id,
            **{key: _thaw(value) for key, value in self.values},
        }


@dataclass(frozen=True)
class TargetEstimateSet:
    """Target estimates keyed by one scientific plan and its task IDs."""

    scientific_plan_hash: str
    target_id: str
    model_id: str
    metric_specs: tuple[TargetMetricSpec, ...]
    rows: tuple[TargetEstimateRow, ...]
    metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: str = TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION

    @classmethod
    def from_rows(
        cls,
        scientific_plan_hash: str,
        target_id: str,
        model_id: str,
        rows: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
        metric_specs: tuple[TargetMetricSpec, ...] | list[TargetMetricSpec],
        metadata: Mapping[str, Any] | None = None,
        schema_version: str = TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION,
    ) -> "TargetEstimateSet":
        for row in rows:
            for key, expected in (
                ("scientific_plan_hash", scientific_plan_hash),
                ("target_id", target_id),
                ("model_id", model_id),
                ("schema_version", schema_version),
            ):
                if key in row and str(row[key]) != str(expected):
                    raise ValueError(
                        f"Target estimate row {key} does not match sidecar metadata"
                    )
            if "sidecar_metadata" in row and _thaw(
                _freeze(row["sidecar_metadata"])
            ) != dict(metadata or {}):
                raise ValueError(
                    "Target estimate row sidecar metadata does not match set metadata"
                )
            if "metric_provenance" in row and row["metric_provenance"] != [
                spec.to_json_dict() for spec in metric_specs
            ]:
                raise ValueError(
                    "Target estimate row metric provenance does not match set metadata"
                )
        normalized_rows = tuple(
            sorted(
                (TargetEstimateRow.from_mapping(row) for row in rows),
                key=lambda row: row.task_id,
            )
        )
        task_ids = [row.task_id for row in normalized_rows]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Target estimate sidecar contains duplicate task IDs")
        normalized_specs = tuple(sorted(metric_specs, key=lambda spec: spec.name))
        spec_names = [spec.name for spec in normalized_specs]
        if len(spec_names) != len(set(spec_names)):
            raise ValueError("Target estimate sidecar contains duplicate metric specs")
        expected_metric_names = set(spec_names)
        for row in normalized_rows:
            row_metric_names = {key for key, _ in row.values}
            if row_metric_names != expected_metric_names:
                missing = sorted(expected_metric_names - row_metric_names)
                extra = sorted(row_metric_names - expected_metric_names)
                raise ValueError(
                    "Target estimate metric specs do not cover every row; "
                    f"task_id={row.task_id}, missing={missing}, extra={extra}"
                )
        return cls(
            scientific_plan_hash=str(scientific_plan_hash),
            target_id=str(target_id),
            model_id=str(model_id),
            metric_specs=normalized_specs,
            rows=normalized_rows,
            metadata=tuple(
                (str(key), _freeze(value))
                for key, value in sorted(
                    (metadata or {}).items(), key=lambda pair: str(pair[0])
                )
            ),
            schema_version=schema_version,
        )

    @classmethod
    def from_jsonl_rows(
        cls,
        rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
        *,
        expected_target_id: str | None = None,
        expected_model_id: str | None = None,
    ) -> "TargetEstimateSet":
        """Reconstruct a sidecar using only persisted row metadata."""

        if not rows:
            raise ValueError(
                "Cannot reconstruct a target estimate sidecar from empty JSONL"
            )
        first = rows[0]
        schema_version = str(first.get("schema_version", ""))
        if schema_version != TARGET_ESTIMATE_SIDECAR_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported target estimate sidecar schema: {schema_version}"
            )
        target_id = str(first.get("target_id", ""))
        model_id = str(first.get("model_id", ""))
        if expected_target_id is not None and target_id != expected_target_id:
            raise ValueError(
                "Persisted target estimate sidecar has an unexpected target ID"
            )
        if expected_model_id is not None and model_id != expected_model_id:
            raise ValueError(
                "Persisted target estimate sidecar has an unexpected model ID"
            )
        provenance = tuple(
            TargetMetricSpec(**item) for item in first.get("metric_provenance", ())
        )
        metadata = first.get("sidecar_metadata", {})
        for row in rows:
            for key in (
                "schema_version",
                "scientific_plan_hash",
                "target_id",
                "model_id",
            ):
                if row.get(key) != first.get(key):
                    raise ValueError(
                        f"Persisted target estimate rows disagree on {key}"
                    )
        return cls.from_rows(
            scientific_plan_hash=str(first.get("scientific_plan_hash", "")),
            target_id=target_id,
            model_id=model_id,
            rows=list(rows),
            metric_specs=provenance,
            metadata=metadata,
            schema_version=schema_version,
        )

    def row(self, task_id: str) -> TargetEstimateRow | None:
        for row in self.rows:
            if row.task_id == task_id:
                return row
        return None

    def values_for(self, task_id: str) -> dict[str, Any] | None:
        row = self.row(task_id)
        return row.to_json_dict() if row is not None else None

    def validate_keys(
        self, scientific_plan_hash: str, task_ids: tuple[str, ...] | list[str]
    ) -> None:
        if self.scientific_plan_hash != scientific_plan_hash:
            raise ValueError(
                "Target estimate sidecar scientific plan hash does not match the graph"
            )
        expected = tuple(sorted(str(task_id) for task_id in task_ids))
        actual = tuple(sorted(row.task_id for row in self.rows))
        if actual != expected:
            raise ValueError("Target estimate sidecar task IDs do not match the graph")

    def validate_graph(
        self,
        graph: Any,
        *,
        expected_target_id: str | None = None,
        expected_model_id: str | None = None,
    ) -> None:
        """Validate plan, task IDs, and tensor descriptors against a graph."""

        if expected_target_id is not None and self.target_id != expected_target_id:
            raise ValueError(
                "Target estimate sidecar target ID does not match the requested target"
            )
        if expected_model_id is not None and self.model_id != expected_model_id:
            raise ValueError(
                "Target estimate sidecar model ID does not match the requested model"
            )
        self.validate_keys(
            graph.contraction_plan_hash, tuple(task.id for task in graph.tasks)
        )
        for task in graph.tasks:
            row = self.row(task.id)
            assert row is not None
            if row.input_tensor_ids != tuple(task.input_tensor_ids):
                raise ValueError(
                    f"Target estimate sidecar input tensors do not match task {task.id}"
                )
            if row.output_tensor_id != task.output_tensor_id:
                raise ValueError(
                    f"Target estimate sidecar output tensor does not match task {task.id}"
                )

    def metadata_dict(self) -> dict[str, Any]:
        return {key: _thaw(value) for key, value in self.metadata}

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scientific_plan_hash": self.scientific_plan_hash,
            "target_id": self.target_id,
            "model_id": self.model_id,
            "metadata": self.metadata_dict(),
            "metric_specs": [spec.to_json_dict() for spec in self.metric_specs],
            "rows": [row.to_json_dict() for row in self.rows],
        }

    def jsonl_rows(self) -> list[dict[str, Any]]:
        provenance = [spec.to_json_dict() for spec in self.metric_specs]
        return [
            {
                **row.to_json_dict(),
                "scientific_plan_hash": self.scientific_plan_hash,
                "target_id": self.target_id,
                "model_id": self.model_id,
                "schema_version": self.schema_version,
                "sidecar_metadata": self.metadata_dict(),
                "metric_provenance": provenance,
            }
            for row in self.rows
        ]
