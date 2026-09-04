from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "freeze_quantized_upmem_execution.py"
SPEC = importlib.util.spec_from_file_location("freeze_quantized_upmem_execution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_freeze_identity_contract_is_fixed() -> None:
    assert module.POLICY == "complex_int8_shared_scale_v1"
    assert module.POLICY_SOURCE == "6c9ca849a5ccc246dc645b63598ee391da75c599"
    assert module.PHYSICAL_SOURCE == "c0ec6c76439e418e537a953a6b768ce2e1ea0dc6"
    assert module.EXPECTED_TAG == "thesis-upmem-quantized-execution-diagnostic-v1"
    assert len(module.BINARY_SHA256) == 9


def test_bundle_checksum_inventory_is_sorted_and_exact(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "z.txt").write_bytes(b"z\n")
    (root / "a.txt").write_bytes(b"a\n")
    (root / "nested").mkdir()
    (root / "nested" / "value.bin").write_bytes(b"value\n")

    module._write_checksums(root)
    module._verify_checksums(root)
    listed = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert [line.split("  ", 1)[1] for line in listed] == sorted(
        line.split("  ", 1)[1] for line in listed
    )

    (root / "unlisted.txt").write_bytes(b"unlisted\n")
    with pytest.raises(ValueError, match="inventory"):
        module._verify_checksums(root)


def _write_tar_member(archive: Path, member: tarfile.TarInfo) -> None:
    with tarfile.open(archive, "w:gz") as stream:
        stream.addfile(member)


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "root/./file"])
def test_safe_extract_rejects_unsafe_paths(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    member = tarfile.TarInfo(name)
    member.size = 0
    _write_tar_member(archive, member)

    with pytest.raises(ValueError, match="unsafe archive member"):
        module._safe_extract(archive, tmp_path / "extract")


@pytest.mark.parametrize(
    "member_type",
    [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE],
)
def test_safe_extract_rejects_links_and_special_files(
    tmp_path: Path, member_type: bytes
) -> None:
    archive = tmp_path / "special.tar.gz"
    member = tarfile.TarInfo("root/member")
    member.type = member_type
    member.linkname = "target"
    _write_tar_member(archive, member)

    with pytest.raises(ValueError, match="unsafe archive member"):
        module._safe_extract(archive, tmp_path / "extract")


def test_safe_extract_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for payload in (b"one", b"two"):
            member = tarfile.TarInfo("root/value")
            member.size = len(payload)
            stream.addfile(member, __import__("io").BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        module._safe_extract(archive, tmp_path / "extract")


def test_safe_extract_accepts_one_regular_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("value\n", encoding="utf-8")
    archive = tmp_path / "safe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="root")

    destination = tmp_path / "extract"
    destination.mkdir()
    root = module._safe_extract(archive, destination)
    assert root.name == "root"
    assert (root / "value.txt").read_text(encoding="utf-8") == "value\n"


def test_descendant_allowlist_rejects_execution_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_git",
        lambda *_args: "thesis/implementation/native/changed.c\n",
    )
    with pytest.raises(ValueError, match="forbidden paths"):
        module._check_descendant_paths("f" * 40)
