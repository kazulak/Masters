import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile

import pytest

from tests.test_upmem_cost_guided_search import search
from scripts.qualify_quantized_upmem_execution import write_checksums


def test_two_copies_are_extracted_verified_and_not_same_inode(monkeypatch):
    import upmem_cost_guided_evidence as evidence

    # Accepted copies may not reside under /tmp, including during this safety test.
    parent = search.ROOT / "runs"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="archive-copy-test-", dir=parent) as directory:
        root = Path(directory)
        stage = root / "stage"
        (stage / "raw").mkdir(parents=True)
        (stage / "raw" / "fixture.txt").write_text("synthetic evidence fixture")
        write_checksums(stage)
        first, second = root / "first.tar.gz", root / "second.tar.gz"
        with tarfile.open(first, "x:gz") as archive:
            archive.add(stage, arcname="stage")
        shutil.copyfile(first, second)
        digest = hashlib.sha256(first.read_bytes()).hexdigest()
        for archive in (first, second):
            Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n")
        calls = []
        def extract(raw, *args):
            calls.append(raw)
            assert (raw / "fixture.txt").read_text() == "synthetic evidence fixture"
            return {"rows": [{"fixture": True}], "physical_stage_elapsed_s": 1.0}
        monkeypatch.setattr(evidence, "extract_round_observations", extract)
        result = search.verify_round_archives([first, second], {}, {}, {}, {})
        assert len(calls) == 2 and calls[0] != calls[1]
        assert result["archives"] == [{"path": str(p.resolve()), "sha256": digest} for p in (first, second)]
        with pytest.raises(ValueError, match="independent copies"):
            search.verify_round_archives([first, first], {}, {}, {}, {})
        Path(str(second) + ".sha256").write_text(f"{'0' * 64}  {second.name}\n")
        with pytest.raises(ValueError, match="Outer archive"):
            search.verify_round_archives([first, second], {}, {}, {}, {})


def test_volatile_or_missing_archive_copies_are_not_accepted(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"not raw evidence")
    second.write_bytes(b"not raw evidence")
    with pytest.raises(ValueError, match="durable storage"):
        search.verify_round_archives([first, second], {}, {}, {}, {})
    with pytest.raises(ValueError, match="two archive copies"):
        search.verify_round_archives([first], {}, {}, {}, {})
