#!/usr/bin/env python3
"""Audit which v4 request-record fields remain stable under controlled changes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import tempfile
from typing import Any

from quantum_bench.upmem.protocol import (
    NUMERIC_FLOAT32,
    NUMERIC_HOST_PACKED_INT8,
    V4Profile,
    V4WorkUnit,
    V4WorkUnitRecord,
    build_v4_request,
)


TASK_HASH = "ab" * 32
CELL_DEFINITIONS = (
    ("ghz_chain_18q", 1),
    ("ghz_chain_18q", 4),
    ("hs_18q_d1", 1),
    ("hs_18q_d1", 4),
    ("quantization_stress_18q_l2", 1),
    ("quantization_stress_18q_l2", 4),
)
VARIATION_NAMES = (
    "session_root",
    "request_sequence",
    "payload_contents",
    "geometry",
    "numeric_policy",
    "tasklet_topology",
)
FIELD_CLASSES = {
    "local_dpu_id": "plan_static",
    "flags": "plan_static",
    "tile_id": "operation_static",
    "batch_index": "operation_static",
    "m_offset": "operation_static",
    "n_offset": "operation_static",
    "k_offset": "operation_static",
    "m_elements": "operation_static",
    "n_elements": "operation_static",
    "k_elements": "operation_static",
    "a_transfer_bytes": "operation_static",
    "b_transfer_bytes": "operation_static",
    "c_transfer_bytes": "operation_static",
    "a_offset_bytes": "operation_static",
    "b_offset_bytes": "operation_static",
    "c_offset_bytes": "operation_static",
    "a_path": "invocation_specific",
    "b_path": "invocation_specific",
    "c_path": "invocation_specific",
    "a_sha256": "lane_specific",
    "b_sha256": "lane_specific",
}
ABI_BYTES = {
    "local_dpu_id": 4,
    "flags": 4,
    "tile_id": 8,
    "batch_index": 8,
    "m_offset": 8,
    "n_offset": 8,
    "k_offset": 8,
    "m_elements": 4,
    "n_elements": 4,
    "k_elements": 4,
    "a_transfer_bytes": 4,
    "b_transfer_bytes": 4,
    "c_transfer_bytes": 4,
    "a_offset_bytes": 4,
    "b_offset_bytes": 4,
    "c_offset_bytes": 4,
}
MANIFEST_FIELDS = frozenset(
    {"local_dpu_id", "tile_id", "a_path", "b_path", "c_path", "a_sha256", "b_sha256"}
)
CSV_FIELDS = (
    "case_id",
    "dpu_count",
    "field",
    "declared_class",
    "changed_under",
    "baseline_distinct_values",
    "abi_bytes",
    "manifest_bytes",
    "candidate_reuse_scope",
)


def _payload(length: int, seed: int) -> bytes:
    return bytes((seed + index) % 251 for index in range(length))


def _work_units(
    dpu_count: int,
    *,
    k: int = 3,
    numeric_mode: str = NUMERIC_FLOAT32,
    payload_seed: int = 1,
) -> list[V4WorkUnit]:
    element_bytes = 4 if numeric_mode == NUMERIC_FLOAT32 else 1
    payload = _payload(k * element_bytes, payload_seed)
    return [
        V4WorkUnit(
            local_dpu_id=dpu_id,
            tile_id=1000 + dpu_id,
            batch_index=0,
            m_offset=dpu_id,
            n_offset=0,
            k_offset=0,
            m_elements=1,
            n_elements=1,
            k_elements=k,
            a_payload=payload,
            b_payload=payload,
        )
        for dpu_id in range(dpu_count)
    ]


def _build(
    root: Path,
    *,
    dpu_count: int,
    sequence: int = 0,
    k: int = 3,
    numeric_mode: str = NUMERIC_FLOAT32,
    payload_seed: int = 1,
    tasklets: int = 8,
) -> tuple[V4WorkUnitRecord, ...]:
    return _build_artifact(
        root,
        dpu_count=dpu_count,
        sequence=sequence,
        k=k,
        numeric_mode=numeric_mode,
        payload_seed=payload_seed,
        tasklets=tasklets,
    ).work_units


def _build_artifact(
    root: Path,
    *,
    dpu_count: int,
    sequence: int = 0,
    k: int = 3,
    numeric_mode: str = NUMERIC_FLOAT32,
    payload_seed: int = 1,
    tasklets: int = 8,
) -> Any:
    return build_v4_request(
        root,
        profile=V4Profile(
            dpu_count=dpu_count,
            tasklets_per_dpu=tasklets,
            numeric_mode=numeric_mode,
        ),
        canonical_batch_count=1,
        canonical_m=dpu_count + (1 if k != 3 else 0),
        canonical_n=1,
        canonical_k=k,
        work_units=_work_units(
            dpu_count,
            k=k,
            numeric_mode=numeric_mode,
            payload_seed=payload_seed,
        ),
        task_contract_sha256=TASK_HASH,
        request_sequence=sequence,
    )


def _relative_artifact_files(artifact: Any) -> dict[str, bytes]:
    return {
        path.relative_to(artifact.request_dir).as_posix(): path.read_bytes()
        for path in sorted(artifact.request_dir.rglob("*"))
        if path.is_file()
    }


def _artifact_accounting(artifact: Any) -> dict[str, int | bool]:
    payload_files = sorted((artifact.request_dir / "payloads").glob("*.bin"))
    staged_bytes = sum(path.stat().st_size for path in payload_files)
    return {
        "payload_file_count_matches": len(payload_files)
        == artifact.payload_files_created,
        "payload_byte_count_matches": staged_bytes == artifact.payload_bytes_staged,
        "hash_byte_count_matches": artifact.payload_bytes_hashed
        == artifact.payload_bytes_staged,
        "payload_file_count": len(payload_files),
        "payload_bytes": staged_bytes,
    }


def _manifest_bytes(record: V4WorkUnitRecord, field: str) -> int:
    values = {
        "local_dpu_id": record.local_dpu_id,
        "tile_id": record.tile_id,
        "a_path": record.a_path,
        "b_path": record.b_path,
        "c_path": record.c_path,
        "a_sha256": record.a_sha256,
        "b_sha256": record.b_sha256,
    }
    return len(str(values[field]).encode("utf-8")) + 1


def _validate_source_fields() -> tuple[str, ...]:
    actual = tuple(field.name for field in fields(V4WorkUnitRecord))
    expected = tuple(FIELD_CLASSES)
    if set(actual) != set(expected):
        raise ValueError(
            "V4WorkUnitRecord fields changed; update the invariance audit explicitly"
        )
    return actual


def _variation_records(root: Path, dpu_count: int) -> dict[str, tuple[V4WorkUnitRecord, ...]]:
    return {
        "baseline": _build(root / "baseline", dpu_count=dpu_count),
        "session_root": _build(root / "alternate-root", dpu_count=dpu_count),
        "request_sequence": _build(root / "sequence", dpu_count=dpu_count, sequence=7),
        "payload_contents": _build(
            root / "payload", dpu_count=dpu_count, payload_seed=97
        ),
        "geometry": _build(root / "geometry", dpu_count=dpu_count, k=5),
        "numeric_policy": _build(
            root / "numeric", dpu_count=dpu_count, numeric_mode=NUMERIC_HOST_PACKED_INT8
        ),
        "tasklet_topology": _build(
            root / "topology", dpu_count=dpu_count, tasklets=5
        ),
    }


def _field_value(record: V4WorkUnitRecord, field: str) -> object:
    return getattr(record, field)


def _field_rows(
    baseline: tuple[V4WorkUnitRecord, ...],
    variants: dict[str, tuple[V4WorkUnitRecord, ...]],
    *,
    case_id: str,
    dpu_count: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in FIELD_CLASSES:
        changed_under = []
        for variation in VARIATION_NAMES:
            candidate = variants[variation]
            if len(candidate) != len(baseline) or any(
                _field_value(base, field) != _field_value(other, field)
                for base, other in zip(baseline, candidate, strict=True)
            ):
                changed_under.append(variation)
        sample = baseline[0]
        if field in {"a_path", "b_path", "c_path"}:
            scope = "one request sequence and session root"
        elif field in {"a_sha256", "b_sha256"}:
            scope = "one payload and invocation"
        else:
            scope = "one exact operation geometry and physical plan"
        rows.append(
            {
                "case_id": case_id,
                "dpu_count": dpu_count,
                "field": field,
                "declared_class": FIELD_CLASSES[field],
                "changed_under": ",".join(changed_under),
                "baseline_distinct_values": len(
                    {_field_value(record, field) for record in baseline}
                ),
                "abi_bytes": ABI_BYTES.get(field, 0),
                "manifest_bytes": _manifest_bytes(sample, field)
                if field in MANIFEST_FIELDS
                else 0,
                "candidate_reuse_scope": scope,
            }
        )
    return rows


def analyze(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")
    field_order = _validate_source_fields()
    all_rows: list[dict[str, object]] = []
    variation_rows: list[dict[str, object]] = []
    artifact_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="request-record-invariance-") as temporary:
        root = Path(temporary)
        for case_id, dpu_count in CELL_DEFINITIONS:
            variants = _variation_records(root / case_id / str(dpu_count), dpu_count)
            baseline = variants["baseline"]
            all_rows.extend(
                _field_rows(
                    baseline,
                    variants,
                    case_id=case_id,
                    dpu_count=dpu_count,
                )
            )
            for variation in VARIATION_NAMES:
                candidate = variants[variation]
                changed = [
                    field
                    for field in field_order
                    if len(candidate) != len(baseline)
                    or any(
                        _field_value(base, field) != _field_value(other, field)
                        for base, other in zip(baseline, candidate, strict=True)
                    )
                ]
                variation_rows.append(
                    {
                        "case_id": case_id,
                        "dpu_count": dpu_count,
                        "variation": variation,
                        "changed_fields": changed,
                    }
                )
            baseline_artifact = _build_artifact(
                root / case_id / str(dpu_count) / "tree-baseline",
                dpu_count=dpu_count,
            )
            alternate_artifact = _build_artifact(
                root / case_id / str(dpu_count) / "tree-alternate-root",
                dpu_count=dpu_count,
            )
            baseline_files = _relative_artifact_files(baseline_artifact)
            alternate_files = _relative_artifact_files(alternate_artifact)
            baseline_accounting = _artifact_accounting(baseline_artifact)
            alternate_accounting = _artifact_accounting(alternate_artifact)
            artifact_rows.append(
                {
                    "case_id": case_id,
                    "dpu_count": dpu_count,
                    "same_relative_artifact_bytes": baseline_files
                    == alternate_files,
                    "relative_file_count": len(baseline_files),
                    "baseline_accounting": baseline_accounting,
                    "alternate_accounting": alternate_accounting,
                }
            )
    result: dict[str, Any] = {
        "analysis_version": "request_record_invariance_v1",
        "record_schema": "V4WorkUnitRecord",
        "record_field_order": field_order,
        "fixtures": [
            {"case_id": case_id, "dpu_count": dpu_count}
            for case_id, dpu_count in CELL_DEFINITIONS
        ],
        "variations": variation_rows,
        "fields": all_rows,
        "artifact_equivalence": artifact_rows,
        "interpretation": (
            "A field is reusable only within its declared scope and only when "
            "it remains unchanged under every variation outside that scope."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request_record_invariance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "request_record_invariance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
