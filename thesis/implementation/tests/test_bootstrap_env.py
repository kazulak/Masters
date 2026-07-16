from __future__ import annotations

from pathlib import Path

import pytest

from scripts import bootstrap_env


def test_bootstrap_requires_uv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv is required"):
        bootstrap_env.bootstrap(root=tmp_path, venv=tmp_path.parent / ".venv")


def test_bootstrap_refuses_incompatible_existing_environment(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap_env, "_python_version", lambda path: (3, 10))
    with pytest.raises(RuntimeError, match="refusing to replace"):
        bootstrap_env.bootstrap(root=root, venv=venv)


def test_bootstrap_reuses_compatible_environment_without_creating_it(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    (root / ".python-version").write_text("3.10\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap_env, "_python_version", lambda path: (3, 10))
    monkeypatch.setattr(bootstrap_env.subprocess, "run", lambda command, **kwargs: calls.append(command))

    bootstrap_env.bootstrap(root=root, venv=venv)

    assert len(calls) == 1
    assert calls[0][:3] == ["/usr/bin/uv", "pip", "install"]
    assert "--constraint" in calls[0]
    assert "-e" in calls[0] and ".[dev]" in calls[0]


def test_bootstrap_plan_does_not_run_commands(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap_env.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ran command")))

    bootstrap_env.bootstrap(root=root, venv=tmp_path / ".venv", dry_run=True)


def test_bootstrap_plan_is_available_before_uv_is_installed(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "implementation"
    root.mkdir()
    monkeypatch.setattr(bootstrap_env.shutil, "which", lambda name: None)

    assert bootstrap_env.bootstrap(root=root, venv=tmp_path / ".venv", dry_run=True) == 0
