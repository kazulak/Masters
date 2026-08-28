import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT.parent / ".venv"


def repository_root(root: Path = ROOT) -> Path:
    for candidate in (root, *root.parents):
        if (candidate / ".git").exists():
            return candidate
    return root


def requested_python(root: Path = ROOT) -> Optional[str]:
    version_file = root / ".python-version"
    if not version_file.exists():
        return None
    lines = version_file.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    value = lines[0].strip()
    return value or None


def bootstrap(*, dry_run: bool = False, root: Path = ROOT, venv: Path = VENV) -> int:
    uv = shutil.which("uv")
    if uv is None:
        if not dry_run:
            raise RuntimeError("uv is required; install uv and rerun bootstrap_env.py")
        uv = "uv"
    requested = requested_python(root)
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if venv.exists() and not python.exists():
        raise RuntimeError(f"existing environment is not a usable uv venv: {venv}")
    if python.exists():
        version = _python_version(python)
        if not _compatible(version, requested):
            wanted = requested or ">=3.10"
            raise RuntimeError(f"existing {venv} uses Python {version}, incompatible with {wanted}; refusing to replace it")
    commands: List[List[str]] = []
    if not python.exists():
        command = [uv, "venv", str(venv)]
        if requested:
            command += ["--python", requested]
        commands.append(command)
    install_python = str(python)
    commands.append([uv, "pip", "install", "--python", install_python, "--constraint", str(root / "ci" / "constraints.txt"), "-e", ".[dev]"])
    repo_root = repository_root(root)
    if (repo_root / ".gitmodules").exists():
        commands.append(["git", "submodule", "update", "--init", "--recursive"])
    for command in commands:
        print("PLAN " + " ".join(command) if dry_run else "RUN " + " ".join(command))
        if not dry_run:
            cwd = repo_root if command[:2] == ["git", "submodule"] else root
            subprocess.run(command, cwd=cwd, check=True)
    return 0


def _python_version(python: Path) -> Tuple[int, int]:
    result = subprocess.run([str(python), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"], check=True, capture_output=True, text=True)
    major, minor = result.stdout.strip().split(".")[:2]
    return int(major), int(minor)


def _compatible(version: Tuple[int, int], requested: Optional[str]) -> bool:
    if version < (3, 10):
        return False
    if not requested or requested in {"system", "default"}:
        return True
    parts = requested.split(".")
    try:
        return version == (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap the thesis Python environment with uv.")
    parser.add_argument("--dry-run", "--plan", action="store_true", dest="dry_run")
    args = parser.parse_args(argv)
    return bootstrap(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
