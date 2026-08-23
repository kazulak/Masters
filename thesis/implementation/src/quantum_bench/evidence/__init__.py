"""Evidence admission rules for thesis-facing reports."""

from .claims import Claim, ClaimDecision, ClaimPolicy, ExecutionMode
from .canonical import (
    append_sample,
    append_session,
    canonical_json,
    finalize_artifacts,
    identity_hash,
    new_run_id,
    require_matching_scope,
    sample_id,
    validate_artifact_set,
    validate_manifest,
    validate_sample,
    validate_session,
    write_manifest,
)

__all__ = [
    "Claim",
    "ClaimDecision",
    "ClaimPolicy",
    "ExecutionMode",
    "append_sample",
    "append_session",
    "canonical_json",
    "finalize_artifacts",
    "identity_hash",
    "new_run_id",
    "require_matching_scope",
    "sample_id",
    "validate_artifact_set",
    "validate_manifest",
    "validate_sample",
    "validate_session",
    "write_manifest",
]
