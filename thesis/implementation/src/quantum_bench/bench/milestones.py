"""Milestone truth ledger loading, verification, and rendering."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
GENERATED_DOCUMENT = Path("docs/MILESTONES.md")
STATUSES = (
    "planned",
    "implemented",
    "development_observed",
    "development_validated_external",
    "tracked_verified",
    "blocked",
    "superseded",
)
_MILESTONE_KEYS = {
    "id",
    "title",
    "status",
    "summary",
    "source_commit",
    "record_commit",
    "source_commands",
    "replay_commands",
    "evidence_origin",
    "evidence",
    "external_hashes",
    "allowed_claims",
    "prohibited_claims",
    "superseded_by",
    "sources",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def load_ledger(path: Path) -> dict[str, Any]:
    """Load a milestone ledger YAML mapping without accepting implicit schemas."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read milestone ledger {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Milestone ledger must contain a YAML mapping")
    return value


def verify_ledger(path: Path, root_dir: Path) -> dict[str, Any]:
    """Verify provenance, tracked evidence, and the generated document."""
    milestones = _verified_milestones(path, root_dir)
    expected = _render_markdown(milestones)
    generated_path = root_dir / GENERATED_DOCUMENT
    try:
        actual = generated_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read generated milestone document: {exc}") from exc
    if actual != expected:
        raise ValueError(
            f"Generated milestone document is stale: {generated_path}; "
            "run make milestones-render"
        )
    return {
        "ledger": str(path),
        "generated_document": str(generated_path),
        "milestone_count": len(milestones),
        "verified_tracked_evidence": sum(
            item["status"] == "tracked_verified" for item in milestones
        ),
        "status": "verified",
    }


def render_ledger(path: Path, root_dir: Path) -> str:
    """Return deterministic Markdown after applying provenance gates."""
    return _render_markdown(_verified_milestones(path, root_dir))


def write_rendered_ledger(path: Path, output: Path, root_dir: Path) -> dict[str, Any]:
    output.write_text(render_ledger(path, root_dir), encoding="utf-8")
    return {"ledger": str(path), "output": str(output), "status": "rendered"}


def _verified_milestones(path: Path, root_dir: Path) -> list[dict[str, Any]]:
    milestones = _validate_schema(load_ledger(path), root_dir)
    repository_root = _repository_root(root_dir)
    _require_full_history(repository_root)
    project_path = root_dir.resolve().relative_to(repository_root).as_posix()
    historical_makefile = f"{project_path}/Makefile"

    for milestone in milestones:
        source_commit = milestone["source_commit"]
        _verify_commit(source_commit, repository_root, "source_commit")
        _verify_source_commands(milestone, repository_root, historical_makefile)
        _verify_make_commands(
            milestone["replay_commands"],
            _make_targets((root_dir / "Makefile").read_text(encoding="utf-8")),
            f"{milestone['id']}.replay_commands",
        )
        if record_commit := milestone.get("record_commit"):
            _verify_commit(record_commit, repository_root, "record_commit")
        if milestone["status"] == "tracked_verified":
            _verify_tracked_evidence(milestone, root_dir, repository_root)
    return milestones


def _validate_schema(ledger: dict[str, Any], root_dir: Path) -> list[dict[str, Any]]:
    if set(ledger) != {"schema_version", "milestones"}:
        raise ValueError("Milestone ledger must contain only schema_version and milestones")
    if ledger["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Milestone ledger must use schema_version: {SCHEMA_VERSION}")
    milestones = ledger["milestones"]
    if not isinstance(milestones, list) or not milestones:
        raise ValueError("Milestone ledger must define a non-empty milestones list")

    seen_ids: set[str] = set()
    for milestone in milestones:
        if not isinstance(milestone, dict):
            raise ValueError("Every milestone must be a mapping")
        unknown = sorted(set(milestone) - _MILESTONE_KEYS)
        if unknown:
            raise ValueError(f"Unknown milestone field(s): {', '.join(unknown)}")
        _required_strings(milestone, "id", "title", "status", "summary", "source_commit")
        milestone_id = milestone["id"]
        if milestone_id in seen_ids:
            raise ValueError(f"Duplicate milestone id: {milestone_id}")
        seen_ids.add(milestone_id)
        if milestone["status"] not in STATUSES:
            raise ValueError(f"{milestone_id}.status must be one of: {', '.join(STATUSES)}")
        _validate_commit(milestone_id, "source_commit", milestone["source_commit"])
        if "record_commit" in milestone:
            _validate_commit(milestone_id, "record_commit", milestone["record_commit"])
        _string_list(milestone, "source_commands")
        _string_list(milestone, "replay_commands")
        _string_list(milestone, "allowed_claims")
        _string_list(milestone, "prohibited_claims")
        _validate_sources(milestone, root_dir)
        _validate_evidence(milestone)

    for milestone in milestones:
        superseded_by = milestone.get("superseded_by")
        if milestone["status"] == "superseded":
            values = _string_list(milestone, "superseded_by")
            unknown = sorted(set(values) - seen_ids)
            if unknown:
                raise ValueError(
                    f"{milestone['id']}.superseded_by references unknown milestone(s): "
                    f"{', '.join(unknown)}"
                )
        elif superseded_by is not None:
            raise ValueError(f"{milestone['id']}.superseded_by requires superseded status")
    return milestones


def _required_strings(milestone: dict[str, Any], *keys: str) -> None:
    for key in keys:
        if not isinstance(milestone.get(key), str) or not milestone[key].strip():
            raise ValueError(f"Milestone field {key} must be a non-empty string")


def _string_list(milestone: dict[str, Any], key: str) -> list[str]:
    values = milestone.get(key)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"{milestone['id']}.{key} must be a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError(f"{milestone['id']}.{key} must not contain duplicates")
    return values


def _validate_commit(milestone_id: str, field: str, value: Any) -> None:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise ValueError(f"{milestone_id}.{field} must be a full 40-character Git SHA")


def _validate_sources(milestone: dict[str, Any], root_dir: Path) -> None:
    for source in _string_list(milestone, "sources"):
        path = _project_path(root_dir, source, f"{milestone['id']}.sources")
        if not path.is_file():
            raise ValueError(f"Missing {milestone['id']}.sources file: {source}")


def _validate_evidence(milestone: dict[str, Any]) -> None:
    status = milestone["status"]
    origin = milestone.get("evidence_origin")
    evidence = milestone.get("evidence")
    hashes = milestone.get("external_hashes")
    record_commit = milestone.get("record_commit")
    if status in {"planned", "implemented", "blocked"}:
        if origin != "none" or evidence is not None or hashes is not None:
            raise ValueError(
                f"{milestone['id']} {status} evidence_origin must be none without evidence"
            )
        if record_commit is not None:
            raise ValueError(f"{milestone['id']} {status} cannot define record_commit")
        return
    if status == "tracked_verified":
        if origin != "tracked":
            raise ValueError(f"{milestone['id']} tracked_verified evidence_origin must be tracked")
        if record_commit is None:
            raise ValueError(f"{milestone['id']} tracked_verified requires record_commit")
        expected_keys = {"root", "checksums", "capsule_manifest"}
        if not isinstance(evidence, dict) or set(evidence) != expected_keys:
            raise ValueError(
                f"{milestone['id']} tracked evidence must contain root, checksums, "
                "and capsule_manifest"
            )
        if not all(isinstance(evidence[key], str) and evidence[key] for key in evidence):
            raise ValueError(f"{milestone['id']} tracked evidence paths must be strings")
        if hashes is not None:
            raise ValueError(f"{milestone['id']} tracked evidence cannot define external_hashes")
        return
    if record_commit is not None:
        raise ValueError(f"{milestone['id']} non-tracked evidence cannot define record_commit")
    if origin != "ignored_external":
        raise ValueError(
            f"{milestone['id']} non-tracked evidence_origin must be ignored_external"
        )
    if evidence is not None:
        raise ValueError(
            f"{milestone['id']} ignored external evidence cannot be checksum-verified locally"
        )
    if status == "development_validated_external":
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(
                f"{milestone['id']} development_validated_external requires external_hashes"
            )
        if not all(
            isinstance(name, str)
            and name.strip()
            and isinstance(value, str)
            and _SHA256.fullmatch(value)
            for name, value in hashes.items()
        ):
            raise ValueError(f"{milestone['id']} external_hashes must contain SHA-256 values")
    elif hashes is not None:
        raise ValueError(
            f"{milestone['id']} external_hashes require development_validated_external"
        )


def _repository_root(root_dir: Path) -> Path:
    result = _git(root_dir, "rev-parse", "--show-toplevel")
    if result.returncode:
        raise ValueError(f"Cannot locate Git repository: {result.stderr.strip()}")
    repository_root = Path(result.stdout.strip()).resolve()
    try:
        root_dir.resolve().relative_to(repository_root)
    except ValueError as exc:
        raise ValueError("Implementation root must be inside the Git repository") from exc
    return repository_root


def _require_full_history(repository_root: Path) -> None:
    result = _git(repository_root, "rev-parse", "--is-shallow-repository")
    if result.returncode:
        raise ValueError(f"Cannot inspect Git history: {result.stderr.strip()}")
    if result.stdout.strip() == "true":
        raise ValueError(
            "Milestone verification requires full Git history; shallow clone detected. "
            "Fetch full history with: git fetch --unshallow"
        )


def _verify_commit(commit: str, repository_root: Path, field: str) -> None:
    result = _git(repository_root, "cat-file", "-e", f"{commit}^{{commit}}")
    if result.returncode:
        raise ValueError(f"{field} does not exist in local Git history: {commit}")


def _verify_source_commands(
    milestone: dict[str, Any], repository_root: Path, makefile_path: str
) -> None:
    source_commit = milestone["source_commit"]
    result = _git(repository_root, "show", f"{source_commit}:{makefile_path}")
    if result.returncode:
        raise ValueError(
            f"Cannot read Makefile at {milestone['id']} source_commit {source_commit}: "
            f"{result.stderr.strip()}"
        )
    _verify_make_commands(
        milestone["source_commands"],
        _make_targets(result.stdout),
        f"{milestone['id']}.source_commands at {source_commit}",
    )


def _verify_make_commands(commands: list[str], targets: set[str], label: str) -> None:
    for command in commands:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} command {command!r}: {exc}") from exc
        if len(parts) != 2 or parts[0] != "make":
            raise ValueError(f"{label} command must be exactly 'make <target>': {command}")
        if parts[1] not in targets:
            raise ValueError(f"{label} references unknown Make target: {parts[1]}")


def _make_targets(makefile: str) -> set[str]:
    return {
        match.group(1)
        for line in makefile.splitlines()
        if (match := re.match(r"^([A-Za-z0-9][A-Za-z0-9_-]*):", line))
    }


def _verify_tracked_evidence(
    milestone: dict[str, Any], root_dir: Path, repository_root: Path
) -> None:
    evidence = milestone["evidence"]
    evidence_root = _project_path(root_dir, evidence["root"], "tracked evidence root")
    if not evidence_root.is_dir():
        raise ValueError(f"Tracked evidence root is not a directory: {evidence_root}")

    repository_path = evidence_root.relative_to(repository_root).as_posix()
    record_commit = milestone["record_commit"]
    recorded_tree = _git(repository_root, "cat-file", "-e", f"{record_commit}:{repository_path}")
    if recorded_tree.returncode:
        raise ValueError(
            f"Tracked evidence tree is absent from record_commit {record_commit}: "
            f"{repository_path}"
        )
    diff = _git(repository_root, "diff", "--quiet", record_commit, "--", repository_path)
    if diff.returncode == 1:
        raise ValueError(
            f"Tracked evidence tree differs from record_commit {record_commit}: "
            f"{repository_path}"
        )
    if diff.returncode:
        raise ValueError(f"Cannot compare tracked evidence tree: {diff.stderr.strip()}")
    _verify_capsule_contents(milestone, evidence_root)


def _verify_capsule_contents(milestone: dict[str, Any], evidence_root: Path) -> None:
    evidence = milestone["evidence"]
    checksums_name = _safe_relative_path(evidence["checksums"], "checksums path")
    capsule_name = _safe_relative_path(
        evidence["capsule_manifest"], "capsule manifest path"
    )
    checksums_path = _project_path(evidence_root, checksums_name, "checksums path")
    capsule_path = _project_path(evidence_root, capsule_name, "capsule manifest path")
    checksums = _load_json_mapping(checksums_path, "checksum manifest")
    capsule = _load_json_mapping(capsule_path, "capsule manifest")

    if capsule.get("source_commit") != milestone["source_commit"]:
        raise ValueError(
            f"Capsule source_commit does not match {milestone['id']}.source_commit"
        )
    files = capsule.get("files")
    if not isinstance(files, list) or not files or not all(
        isinstance(path, str) and path for path in files
    ):
        raise ValueError("Capsule manifest files must be a non-empty string list")
    if len(files) != len(set(files)):
        raise ValueError("Capsule manifest files must not contain duplicates")

    manifest_files = {
        _safe_relative_path(path, "capsule manifest file") for path in files
    }
    checksum_files = {
        _safe_relative_path(path, "checksum manifest file") for path in checksums
    }
    reserved = {"README.md", capsule_name, checksums_name}
    if manifest_files & reserved:
        raise ValueError("Capsule manifest files must exclude README and manifest files")
    if checksums_name in checksum_files:
        raise ValueError("Checksum manifest must exclude itself")
    expected_checksums = manifest_files | {"README.md", capsule_name}
    if checksum_files != expected_checksums:
        missing = sorted(expected_checksums - checksum_files)
        extra = sorted(checksum_files - expected_checksums)
        raise ValueError(
            "Checksum coverage does not exactly match capsule files plus "
            f"README/capsule_manifest; missing={missing}, extra={extra}"
        )

    expected_tree = checksum_files | {checksums_name}
    actual_tree = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file()
    }
    if actual_tree != expected_tree:
        raise ValueError("Tracked evidence tree contains undeclared or missing files")

    for relative_path, expected_hash in checksums.items():
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"Invalid SHA-256 for {relative_path} in {checksums_path}")
        artifact = _project_path(
            evidence_root,
            _safe_relative_path(relative_path, "checksum manifest file"),
            "tracked evidence artifact",
        )
        if not artifact.is_file():
            raise ValueError(f"Missing tracked evidence artifact: {artifact}")
        actual_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for tracked evidence artifact: {artifact}")


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot load {label} {path}: {exc}") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label.capitalize()} must be a non-empty mapping: {path}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _safe_relative_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a safe relative POSIX path: {value!r}")
    parts = value.split("/")
    if value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a safe relative POSIX path: {value!r}")
    return value


def _project_path(root_dir: Path, relative_path: str, label: str) -> Path:
    safe_path = _safe_relative_path(relative_path, label)
    candidate = (root_dir / safe_path).resolve()
    try:
        candidate.relative_to(root_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside its root: {relative_path}") from exc
    return candidate


def _render_markdown(milestones: list[dict[str, Any]]) -> str:
    lines = [
        "# Milestone Truth Ledger",
        "",
        "Generated from `configs/milestones.yml` by `make milestones-render`. Do not edit this file directly.",
        "",
        "This ledger separates tracked, checksum-verifiable evidence from documented development observations. "
        "Ignored external evidence is never a clean-clone verification claim.",
        "",
        "| Status | Meaning |",
        "| --- | --- |",
        "| `planned` | Intended work with no implementation claim. |",
        "| `implemented` | Code exists; no development-evidence claim is implied. |",
        "| `development_observed` | Documented development observation; required retained hashes are unavailable. |",
        "| `development_validated_external` | External development evidence has retained hashes; it is not clean-clone verified. |",
        "| `tracked_verified` | The recorded evidence tree and all tracked checksums verify locally. |",
        "| `blocked` | Work cannot proceed because its stated prerequisite is unavailable. |",
        "| `superseded` | Replaced by the listed later milestone or route. |",
        "",
        "## Current Records",
        "",
    ]
    for milestone in milestones:
        lines.extend(
            [
                f"### {milestone['id']}: {milestone['title']}",
                "",
                f"**Status:** `{milestone['status']}`",
                "",
                milestone["summary"],
                "",
                f"**Source commit:** `{milestone['source_commit']}`",
            ]
        )
        if record_commit := milestone.get("record_commit"):
            lines.extend(["", f"**Evidence record commit:** `{record_commit}`"])
        lines.extend(
            [
                "",
                "**Historical source commands:** "
                + ", ".join(f"`{item}`" for item in milestone["source_commands"]),
                "",
                "**Current replay commands:** "
                + ", ".join(f"`{item}`" for item in milestone["replay_commands"]),
                "",
                "**Evidence origin:** " + _evidence_origin_text(milestone),
            ]
        )
        if superseded_by := milestone.get("superseded_by"):
            lines.extend(["", "**Superseded by:** " + ", ".join(superseded_by)])
        lines.extend(["", "**Allowed claims:**"])
        lines.extend(f"- {claim}" for claim in milestone["allowed_claims"])
        lines.extend(["", "**Prohibited claims:**"])
        lines.extend(f"- {claim}" for claim in milestone["prohibited_claims"])
        lines.extend(
            [
                "",
                "**Tracked sources:** "
                + ", ".join(
                    f"[`{item}`](../{item})" for item in milestone["sources"]
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _evidence_origin_text(milestone: dict[str, Any]) -> str:
    if milestone["status"] == "tracked_verified":
        evidence = milestone["evidence"]
        return (
            "tracked and bound to the evidence record commit; checksums verified "
            f"from `{evidence['root']}/{evidence['checksums']}`."
        )
    if milestone["evidence_origin"] == "none":
        return "none."
    return "ignored external development evidence; not clean-clone verified."


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"Cannot execute Git: {exc}") from exc
