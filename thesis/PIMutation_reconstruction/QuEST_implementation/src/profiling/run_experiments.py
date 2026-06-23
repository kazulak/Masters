#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
RUNS_DIR = ROOT_DIR / "runs"
SUITES_DIR = ROOT_DIR / "suites"
DEFAULT_RUNNER = ROOT_DIR / "bin" / "quest_runner"
RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
PAPER_ALGORITHMS = ["BB84", "BV", "EDC", "HS", "QRNG", "XOR"]


def strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def parse_scalar(value: str) -> Any:
    value = strip_comment(value).strip()
    if value == "":
        return ""
    if value in {"[]", "{}"}:
        return [] if value == "[]" else {}
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(text: str) -> Any:
    """Parse the small YAML subset used by the bundled benchmark suites."""
    raw_lines = []
    for raw in text.splitlines():
        if not raw.strip() or strip_comment(raw).strip() == "":
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        raw_lines.append((indent, strip_comment(raw).strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(raw_lines):
            return {}, index

        is_list = raw_lines[index][1].startswith("- ")
        if is_list:
            items = []
            while index < len(raw_lines):
                line_indent, line = raw_lines[index]
                if line_indent < indent or not line.startswith("- "):
                    break
                if line_indent != indent:
                    raise ValueError(f"Unexpected indentation in suite YAML: {line}")
                item = line[2:].strip()
                if item == "":
                    child, index = parse_block(index + 1, indent + 2)
                    items.append(child)
                else:
                    items.append(parse_scalar(item))
                    index += 1
            return items, index

        data = {}
        while index < len(raw_lines):
            line_indent, line = raw_lines[index]
            if line_indent < indent or line.startswith("- "):
                break
            if line_indent != indent:
                raise ValueError(f"Unexpected indentation in suite YAML: {line}")
            if ":" not in line:
                raise ValueError(f"Expected key: value in suite YAML: {line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = parse_scalar(value)
                index += 1
            else:
                if index + 1 >= len(raw_lines) or raw_lines[index + 1][0] <= line_indent:
                    data[key] = {}
                    index += 1
                else:
                    child, index = parse_block(index + 1, raw_lines[index + 1][0])
                    data[key] = child
        return data, index

    parsed, final_index = parse_block(0, raw_lines[0][0] if raw_lines else 0)
    if final_index != len(raw_lines):
        raise ValueError("Could not parse complete suite YAML.")
    return parsed


def load_suite(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = load_simple_yaml(text)

    if not isinstance(data, dict):
        raise ValueError(f"Suite file {path} must contain a mapping.")
    data.setdefault("suite_id", path.stem)
    data["_suite_path"] = str(path)
    return data


def suite_path_from_args(args: argparse.Namespace) -> Path:
    if args.suite:
        candidate = Path(args.suite)
        if candidate.exists():
            return candidate.resolve()
        preset_candidate = SUITES_DIR / f"{args.suite}.yaml"
        if preset_candidate.exists():
            return preset_candidate.resolve()
        raise FileNotFoundError(f"Suite file not found: {args.suite}")
    return (SUITES_DIR / f"{args.preset}.yaml").resolve()


def sanitize_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "suite"


def create_run_dir(suite_id: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stamp}_{sanitize_run_id(suite_id)}"
    run_dir = RUNS_DIR / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = RUNS_DIR / f"{base_name}_{suffix:02d}"
        suffix += 1
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "plots").mkdir()
    return run_dir


def update_latest_symlink(run_dir: Path) -> None:
    latest = RUNS_DIR / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            return
        latest.symlink_to(run_dir.name)
    except OSError:
        return


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout.strip() or completed.stderr.strip()
    except OSError as exc:
        return None, str(exc)


def first_line(command: list[str]) -> str | None:
    code, output = run_command(command)
    if code == 0 and output:
        return output.splitlines()[0]
    return None


def read_cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def read_mem_total_kib() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return None


def read_make_var(name: str) -> str | None:
    makefile = ROOT_DIR / "Makefile"
    if not makefile.exists():
        return None
    lines = makefile.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if not line.startswith(f"{name} "):
            continue
        value = line.split("=", 1)[1].strip().rstrip("\\").strip()
        next_idx = idx + 1
        while next_idx < len(lines) and lines[next_idx - 1].rstrip().endswith("\\"):
            value += " " + lines[next_idx].strip().rstrip("\\").strip()
            next_idx += 1
        return re.sub(r"\s+", " ", value).strip()
    return None


def read_quest_version(quest_dir: Path) -> str | None:
    for candidate in [
        quest_dir / "build" / "quest" / "include" / "config.h",
        quest_dir / "CMakeLists.txt",
    ]:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'QUEST_VERSION_STRING\s+"([^"]+)"', text)
        if match:
            return match.group(1)
        match = re.search(r"project\(QuEST\s+VERSION\s+([0-9.]+)", text, re.MULTILINE)
        if match:
            return match.group(1)
    return None


def capture_environment() -> dict[str, Any]:
    quest_dir = ROOT_DIR.parent.parent / "extern" / "QuEST"
    commit_code, commit_output = run_command(["git", "rev-parse", "HEAD"], ROOT_DIR)
    dirty_code, dirty_output = run_command(
        ["git", "status", "--porcelain", "--", str(ROOT_DIR.relative_to(ROOT_DIR.parents[2]))],
        ROOT_DIR.parents[2],
    )
    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_kib": read_mem_total_kib(),
        "compiler": first_line(["gcc", "--version"]),
        "make_cc": read_make_var("CC"),
        "cflags": read_make_var("CFLAGS"),
        "ldflags": read_make_var("LDFLAGS"),
        "quest_path": str(quest_dir),
        "quest_version": read_quest_version(quest_dir),
        "paper_quest_version": "3.7.0",
        "openmp_env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("OMP_", "GOMP_", "KMP_"))
        },
        "git_commit": commit_output if commit_code == 0 else None,
        "git_dirty_for_quest_implementation": bool(dirty_output.strip()) if dirty_code == 0 else None,
        "rapl": {
            "path": str(RAPL_PATH),
            "available": RAPL_PATH.exists(),
            "readable": os.access(RAPL_PATH, os.R_OK),
        },
    }


def expand_qubits(spec: Any) -> list[int]:
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(item) for item in spec]
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported qubit specification: {spec!r}")
    if "values" in spec:
        return [int(item) for item in spec["values"]]
    start = int(spec["start"])
    stop = int(spec["stop"])
    step = int(spec.get("step", 1))
    if step <= 0:
        raise ValueError("Qubit range step must be positive.")
    return list(range(start, stop + 1, step))


def qubits_for_algo(suite: dict[str, Any], algo: str) -> list[int]:
    qubits = suite.get("qubits", [])
    if isinstance(qubits, dict):
        if algo in qubits:
            return expand_qubits(qubits[algo])
        if "default" in qubits:
            return expand_qubits(qubits["default"])
    return expand_qubits(qubits)


def configured_algorithms(suite: dict[str, Any]) -> list[str]:
    algorithms = suite.get("algorithms", PAPER_ALGORITHMS)
    if algorithms is None:
        return []
    if isinstance(algorithms, str):
        return [algorithms]
    return [str(algo) for algo in algorithms]


def allocated_qubits_for(suite: dict[str, Any], algo: str, input_qubits: int) -> int:
    if algo.upper() == "HS" and not bool(suite.get("hs_qubits_are_allocated", False)):
        return input_qubits * 2
    return input_qubits


def build_runner_command(
    runner: Path,
    suite: dict[str, Any],
    algo: str,
    input_qubits: int,
) -> list[str]:
    cmd = [str(runner), "--algo", algo]
    if algo.upper() == "HS" and not bool(suite.get("hs_qubits_are_allocated", False)):
        cmd.extend(["--logical-qubits", str(input_qubits)])
    else:
        cmd.extend(["--qubits", str(input_qubits)])
    if algo.upper() == "RANDOM":
        cmd.extend(["--depth", str(int(suite.get("random_depth", 10)))])
    cmd.append("--json")
    return cmd


def parse_runner_json(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        raise ValueError("Runner produced empty stdout.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        raise


def base_raw_record(
    suite_id: str,
    run_id: str,
    algo: str,
    input_qubits: int,
    allocated_qubits: int,
    repeat_index: int,
    thread_count: int | None,
    command: list[str],
) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "run_id": run_id,
        "phase": "repeat",
        "repeat_index": repeat_index,
        "algo_config": algo,
        "input_qubits_config": input_qubits,
        "allocated_qubits_config": allocated_qubits,
        "thread_count_config": thread_count,
        "command": command,
    }


def execute_runner(
    cmd: list[str],
    timeout_s: float,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, int | None, str, str, str | None]:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, None, exc.stdout or "", exc.stderr or "", f"Timed out after {timeout_s}s."
    except OSError as exc:
        return None, None, "", "", str(exc)

    try:
        payload = parse_runner_json(completed.stdout)
    except Exception as exc:  # noqa: BLE001 - raw record should preserve parser failures.
        return None, completed.returncode, completed.stdout, completed.stderr, f"Could not parse runner JSON: {exc}"

    return payload, completed.returncode, completed.stdout, completed.stderr, None


def run_warmups(
    cmd: list[str],
    warmups: int,
    timeout_s: float,
    env: dict[str, str],
) -> None:
    for _ in range(warmups):
        execute_runner(cmd, timeout_s, env)


def write_raw_record(raw_file: Any, records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    raw_file.write(json.dumps(record, sort_keys=True) + "\n")
    raw_file.flush()
    records.append(record)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        algo = record.get("algo") or record.get("algo_config")
        key = (
            algo,
            record.get("paper_algo"),
            record.get("result_label"),
            record.get("input_qubits", record.get("input_qubits_config")),
            record.get("allocated_qubits", record.get("allocated_qubits_config")),
            record.get("depth", 0),
            record.get("threads", record.get("thread_count_config")),
        )
        groups.setdefault(key, []).append(record)

    summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: (str(item[0][0]), item[0][3] or 0, item[0][6] or 0)):
        successes = [
            row for row in rows
            if row.get("status") == "ok" and row.get("returncode") == 0 and row.get("time_s") is not None
        ]
        skipped = [row for row in rows if row.get("status") == "skipped"]
        failed = [row for row in rows if row not in successes and row not in skipped]
        times = [float(row["time_s"]) for row in successes]
        energies = [
            float(row["energy_joules"])
            for row in successes
            if row.get("energy_source") == "rapl_measured" and row.get("energy_joules") is not None
        ]

        def stat_or_none(fn: Any, values: list[float]) -> float | None:
            return fn(values) if values else None

        summary = {
            "algo": key[0],
            "paper_algo": key[1],
            "result_label": key[2],
            "input_qubits": key[3],
            "allocated_qubits": key[4],
            "depth": key[5],
            "threads": key[6],
            "status": "ok" if successes and not failed and not skipped else (
                "partial" if successes else ("skipped" if skipped and not failed else "failed")
            ),
            "success_count": len(successes),
            "fail_count": len(failed),
            "skipped_count": len(skipped),
            "repeat_count": len(rows),
            "time_median_s": stat_or_none(statistics.median, times),
            "time_mean_s": stat_or_none(statistics.mean, times),
            "time_min_s": min(times) if times else None,
            "time_max_s": max(times) if times else None,
            "time_stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0 if times else None,
            "energy_median_j": stat_or_none(statistics.median, energies),
            "energy_mean_j": stat_or_none(statistics.mean, energies),
            "energy_min_j": min(energies) if energies else None,
            "energy_max_j": max(energies) if energies else None,
            "energy_stdev_j": statistics.stdev(energies) if len(energies) > 1 else 0.0 if energies else None,
            "energy_source": "rapl_measured" if energies else "unavailable",
            "one_qubit_gates": successes[0].get("one_qubit_gates") if successes else None,
            "two_qubit_gates": successes[0].get("two_qubit_gates") if successes else None,
        }
        summaries.append(summary)
    return summaries


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "algo",
        "paper_algo",
        "result_label",
        "input_qubits",
        "allocated_qubits",
        "depth",
        "threads",
        "status",
        "success_count",
        "fail_count",
        "skipped_count",
        "repeat_count",
        "time_median_s",
        "time_mean_s",
        "time_min_s",
        "time_max_s",
        "time_stdev_s",
        "energy_median_j",
        "energy_mean_j",
        "energy_min_j",
        "energy_max_j",
        "energy_stdev_j",
        "energy_source",
        "one_qubit_gates",
        "two_qubit_gates",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def append_skip_records(
    raw_file: Any,
    records: list[dict[str, Any]],
    suite_id: str,
    run_id: str,
    algo: str,
    input_qubits: int,
    allocated_qubits: int,
    repeats: int,
    thread_count: int | None,
    command: list[str],
    reason: str,
) -> list[dict[str, Any]]:
    skipped_records = []
    for repeat_index in range(repeats):
        record = base_raw_record(
            suite_id,
            run_id,
            algo,
            input_qubits,
            allocated_qubits,
            repeat_index,
            thread_count,
            command,
        )
        record.update(
            {
                "algo": "BB84" if algo.upper() == "BB" else algo.upper(),
                "input_qubits": input_qubits,
                "allocated_qubits": allocated_qubits,
                "depth": 0,
                "threads": thread_count,
                "returncode": None,
                "status": "skipped",
                "error": reason,
                "energy_source": "unavailable",
                "energy_joules": None,
                "time_s": None,
            }
        )
        write_raw_record(raw_file, records, record)
        skipped_records.append(record)
    return skipped_records


def run_repeats(
    raw_file: Any,
    records: list[dict[str, Any]],
    suite: dict[str, Any],
    suite_id: str,
    run_id: str,
    runner: Path,
    algo: str,
    input_qubits: int,
    repeats: int,
    warmups: int,
    timeout_s: float,
    thread_count: int | None,
) -> list[dict[str, Any]]:
    allocated_qubits = allocated_qubits_for(suite, algo, input_qubits)
    cmd = build_runner_command(runner, suite, algo, input_qubits)
    memory_guard = suite.get("memory_guard", {}) or {}
    max_allocated = memory_guard.get("max_allocated_qubits")
    group_records: list[dict[str, Any]] = []

    if max_allocated is not None and allocated_qubits > int(max_allocated):
        return append_skip_records(
            raw_file,
            records,
            suite_id,
            run_id,
            algo,
            input_qubits,
            allocated_qubits,
            repeats,
            thread_count,
            cmd,
            f"Requires {allocated_qubits} allocated qubits; guard is {max_allocated}.",
        )

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (suite.get("openmp_env", {}) or {}).items()})
    if thread_count is not None:
        env["OMP_NUM_THREADS"] = str(thread_count)

    if warmups > 0:
        run_warmups(cmd, warmups, timeout_s, env)

    for repeat_index in range(repeats):
        record = base_raw_record(
            suite_id,
            run_id,
            algo,
            input_qubits,
            allocated_qubits,
            repeat_index,
            thread_count,
            cmd,
        )
        record["started_at"] = datetime.now().isoformat(timespec="seconds")
        start = time.monotonic()
        payload, returncode, stdout, stderr, error = execute_runner(cmd, timeout_s, env)
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        record["wall_time_s"] = time.monotonic() - start
        record["returncode"] = returncode
        if payload:
            record.update(payload)
        if error:
            record.update(
                {
                    "algo": "BB84" if algo.upper() == "BB" else algo.upper(),
                    "input_qubits": input_qubits,
                    "allocated_qubits": allocated_qubits,
                    "depth": int(suite.get("random_depth", 10)) if algo.upper() == "RANDOM" else 0,
                    "threads": thread_count,
                    "status": "failed",
                    "error": error,
                    "time_s": None,
                    "energy_joules": None,
                    "energy_source": "unavailable",
                }
            )
        elif returncode != 0 and record.get("status") == "ok":
            record["status"] = "failed"
            record["error"] = f"Runner exited with code {returncode}."
        if stderr.strip():
            record["stderr"] = stderr.strip()
        if stdout.strip() and not payload:
            record["stdout"] = stdout.strip()
        write_raw_record(raw_file, records, record)
        group_records.append(record)

    return group_records


def run_largest_fair_bb84(
    raw_file: Any,
    records: list[dict[str, Any]],
    suite: dict[str, Any],
    suite_id: str,
    run_id: str,
    runner: Path,
    thread_count: int | None,
) -> dict[str, Any] | None:
    config = suite.get("largest_fair_bb84", {}) or {}
    if not config.get("enabled", False):
        return None

    start_qubits = int(config.get("start_qubits", 18))
    max_qubits = int(config.get("max_qubits", start_qubits))
    step = int(config.get("step", 1))
    repeats = int(config.get("repeats", suite.get("repeats", 1)))
    warmups = int(config.get("warmups", suite.get("warmups", 0)))
    timeout_s = float(config.get("timeout_s", suite.get("timeout_s", 60)))
    threshold = float(config.get("runtime_threshold_s", 5.0))
    selected: dict[str, Any] | None = None
    attempts = []

    for qubits in range(start_qubits, max_qubits + 1, step):
        group = run_repeats(
            raw_file,
            records,
            suite,
            suite_id,
            run_id,
            runner,
            "BB84",
            qubits,
            repeats,
            warmups,
            timeout_s,
            thread_count,
        )
        successes = [row for row in group if row.get("status") == "ok" and row.get("time_s") is not None]
        skipped = [row for row in group if row.get("status") == "skipped"]
        median = statistics.median([float(row["time_s"]) for row in successes]) if successes else None
        attempt = {
            "input_qubits": qubits,
            "allocated_qubits": qubits,
            "median_time_s": median,
            "success_count": len(successes),
            "skipped_count": len(skipped),
        }
        attempts.append(attempt)
        if skipped or not successes:
            break
        if median is not None and median <= threshold:
            selected = attempt
            continue
        break

    return {
        "threshold_s": threshold,
        "selected": selected,
        "attempts": attempts,
    }


def python_has_matplotlib(python: Path) -> bool:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/quest-mpl")
    completed = subprocess.run(
        [str(python), "-c", "import matplotlib"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return completed.returncode == 0


def default_plot_python() -> Path:
    env_python = os.environ.get("QUEST_PLOT_PYTHON")
    if env_python:
        return Path(env_python).absolute()

    thesis_venv_python = ROOT_DIR.parents[1] / ".venv" / "bin" / "python"
    if thesis_venv_python.exists() and python_has_matplotlib(thesis_venv_python):
        return thesis_venv_python.absolute()

    return Path(sys.executable).absolute()


def run_plots(run_dir: Path, plot_python: Path) -> dict[str, Any]:
    plot_script = SCRIPT_DIR / "plot_results.py"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/quest-mpl")
    completed = subprocess.run(
        [str(plot_python), str(plot_script), str(run_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return {
        "python": str(plot_python),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_suite(suite: dict[str, Any], runner: Path, make_plots: bool, plot_python: Path) -> Path:
    if not runner.exists():
        raise FileNotFoundError(f"Executable not found at {runner}. Run make first.")

    suite_id = str(suite["suite_id"])
    run_dir = create_run_dir(suite_id)
    run_id = run_dir.name
    raw_path = run_dir / "raw" / "repeats.jsonl"
    records: list[dict[str, Any]] = []
    repeats = int(suite.get("repeats", 1))
    warmups = int(suite.get("warmups", 0))
    timeout_s = float(suite.get("timeout_s", 60))
    thread_counts = suite.get("thread_counts", [None])
    if thread_counts in (None, []):
        thread_counts = [None]
    largest_results = []

    environment = capture_environment()
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for thread_count in thread_counts:
            thread_value = int(thread_count) if thread_count is not None else None
            for algo in configured_algorithms(suite):
                for input_qubits in qubits_for_algo(suite, algo):
                    print(f"[{run_id}] {algo} n={input_qubits} threads={thread_value or 'default'}")
                    run_repeats(
                        raw_file,
                        records,
                        suite,
                        suite_id,
                        run_id,
                        runner,
                        algo,
                        int(input_qubits),
                        repeats,
                        warmups,
                        timeout_s,
                        thread_value,
                    )
            largest_result = run_largest_fair_bb84(
                raw_file,
                records,
                suite,
                suite_id,
                run_id,
                runner,
                thread_value,
            )
            if largest_result:
                largest_result["thread_count_config"] = thread_value
                largest_results.append(largest_result)

    summaries = summarize_records(records)
    write_summary_csv(run_dir / "summary.csv", summaries)

    summary_json = {
        "run_id": run_id,
        "suite_id": suite_id,
        "suite": suite,
        "environment": environment,
        "summaries": summaries,
        "largest_fair_bb84": (
            largest_results[0]
            if len(largest_results) == 1
            else largest_results
            if largest_results
            else None
        ),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if make_plots:
        summary_json["plot_results"] = run_plots(run_dir, plot_python)

    (run_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_latest_symlink(run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible QuEST benchmark suites.")
    parser.add_argument("--suite", help="Path to a suite YAML file, or a suite name under suites/.")
    parser.add_argument(
        "--preset",
        default="local_quick",
        choices=["local_quick", "local_plot", "local_energy", "paper_16_32", "bb84_pc_limit"],
    )
    parser.add_argument("--runner", default=str(DEFAULT_RUNNER), help="Path to bin/quest_runner.")
    parser.add_argument(
        "--plot-python",
        default=None,
        help=(
            "Python interpreter for plot_results.py. Defaults to QUEST_PLOT_PYTHON, "
            "then thesis/.venv/bin/python when it has matplotlib, then sys.executable."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not invoke plot_results.py after summarizing.")
    args = parser.parse_args()

    suite_path = suite_path_from_args(args)
    suite = load_suite(suite_path)
    runner = Path(args.runner).resolve()
    plot_python = Path(args.plot_python).absolute() if args.plot_python else default_plot_python()
    run_dir = run_suite(
        suite,
        runner,
        make_plots=not args.no_plots and bool(suite.get("plot", True)),
        plot_python=plot_python,
    )
    print(f"Run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
