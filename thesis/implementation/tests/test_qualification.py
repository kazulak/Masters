from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qualify_m7b.py"


def _qualifier():
    specification = importlib.util.spec_from_file_location("qualify_m7b", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _archive(path: Path, member_name: str, *, kind: str = "file") -> None:
    source = path.parent / "payload.txt"
    source.write_text("payload\n", encoding="utf-8")
    with tarfile.open(path, "w:gz") as bundle:
        if kind == "file":
            bundle.add(source, arcname=member_name)
            return
        member = tarfile.TarInfo(member_name)
        member.type = tarfile.SYMTYPE
        member.linkname = "payload.txt"
        bundle.addfile(member)


@pytest.mark.parametrize("member_name,kind", [("../escape", "file"), ("link", "link")])
def test_qualifier_rejects_unsafe_release_archive(
    tmp_path: Path, member_name: str, kind: str
) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, member_name, kind=kind)

    with pytest.raises(ValueError, match="unsafe archive member"):
        _qualifier()._safe_extract_tar(archive, tmp_path / "output")


def test_qualifier_extracts_regular_relative_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.tar.gz"
    _archive(archive, "evidence/manifest.json")

    _qualifier()._safe_extract_tar(archive, tmp_path / "output")

    assert (tmp_path / "output" / "evidence" / "manifest.json").read_text(
        encoding="utf-8"
    ) == "payload\n"


def test_qualifier_verifies_bundled_hashes_and_records_external_provenance(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "m7a-bundle"
    evidence = bundle / "cpu-run" / "manifest.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    (bundle / "SHA256SUMS").write_text(
        f"{digest}  runs/{bundle.name}/cpu-run/manifest.json\n"
        + "0" * 64
        + "  native/upmem/runtime/bin/dpu\n",
        encoding="utf-8",
    )

    external = _qualifier()._verify_internal_hashes(tmp_path)

    assert external == ("native/upmem/runtime/bin/dpu",)
