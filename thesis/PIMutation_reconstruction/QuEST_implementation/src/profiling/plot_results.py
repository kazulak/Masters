#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore  # noqa: E402
except ImportError:
    plt = None


MARKERS = {
    "BB84": "o",
    "BV": "s",
    "EDC": "^",
    "HS": "D",
    "QRNG": "P",
    "XOR": "v",
    "RANDOM": "X",
}
COLORS = [
    (31, 119, 180),
    (214, 39, 40),
    (44, 160, 44),
    (148, 103, 189),
    (255, 127, 14),
    (23, 190, 207),
    (127, 127, 127),
]


def parse_number(value: str) -> int | float | None:
    if value in {"", "None", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value  # type: ignore[return-value]


def load_summary(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {key: parse_number(value) for key, value in row.items()}
            rows.append(parsed)
        return rows


def successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row.get("success_count") or 0) > 0 and row.get("time_median_s") is not None
    ]


def display_label(row: dict[str, Any], include_threads: bool = False) -> str:
    algo = str(row.get("algo") or "")
    paper = row.get("paper_algo")
    if paper and str(paper) not in {algo, "RANDOM"}:
        label = f"{algo} ({paper})"
    else:
        label = algo
    if include_threads and row.get("threads") is not None:
        label = f"{label}, {row['threads']} thread(s)"
    return label


def label_for(row: dict[str, Any]) -> str:
    return display_label(row)


def has_multiple_thread_counts(rows: list[dict[str, Any]]) -> bool:
    return len({row.get("threads") for row in rows if row.get("threads") is not None}) > 1


def group_by_algo(rows: list[dict[str, Any]], x_key: str = "input_qubits") -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    include_threads = has_multiple_thread_counts(rows)
    for row in rows:
        groups.setdefault(display_label(row, include_threads), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: (item.get(x_key) or 0, item.get("threads") or 0))
    return groups


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"run_id": run_dir.name, "suite_id": "unknown"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"run_id": run_dir.name, "suite_id": "unknown"}
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "suite_id": summary.get("suite_id", "unknown"),
        "quest_version": (summary.get("environment") or {}).get("quest_version"),
        "paper_quest_version": (summary.get("environment") or {}).get("paper_quest_version"),
    }


def title_prefix(metadata: dict[str, Any]) -> str:
    suite_id = metadata.get("suite_id") or "unknown suite"
    quest_version = metadata.get("quest_version")
    if quest_version:
        return f"QuEST v{quest_version} CPU Benchmark - {suite_id}"
    return f"QuEST CPU Benchmark - {suite_id}"


def add_run_note(fig: Any, metadata: dict[str, Any]) -> None:
    run_id = metadata.get("run_id")
    paper_version = metadata.get("paper_quest_version")
    note = f"Run: {run_id}"
    if paper_version:
        note += f" | PIMutation paper used QuEST v{paper_version}"
    fig.text(0.01, 0.01, note, fontsize=8, color="#555555")


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset:offset + 3] = bytes(color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                set_pixel(pixels, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) +
            kind +
            data +
            struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride:(y + 1) * stride])

    png = (
        b"\x89PNG\r\n\x1a\n" +
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
        chunk(b"IDAT", zlib.compress(bytes(raw), 9)) +
        chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def fallback_plot(
    rows: list[dict[str, Any]],
    y_key: str,
    output_path: Path,
    x_key: str = "input_qubits",
    log_y: bool = False,
    horizontal_line: float | None = None,
) -> None:
    width, height = 1000, 650
    left, right, top, bottom = 80, 40, 45, 70
    pixels = bytearray([255] * width * height * 3)
    plot_w = width - left - right
    plot_h = height - top - bottom
    groups = group_by_algo(rows, x_key=x_key)
    points = [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(y_key) is not None and (not log_y or float(row[y_key]) > 0)
    ]
    if not points:
        write_png(output_path, width, height, pixels)
        return

    xs = [point[0] for point in points]
    ys = [math.log10(point[1]) if log_y else point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 0.5
        max_y += 0.5

    axis = (40, 40, 40)
    grid = (220, 220, 220)
    draw_line(pixels, width, height, left, top, left, top + plot_h, axis)
    draw_line(pixels, width, height, left, top + plot_h, left + plot_w, top + plot_h, axis)
    for idx in range(1, 5):
        y = top + int(plot_h * idx / 5)
        draw_line(pixels, width, height, left, y, left + plot_w, y, grid)
        x = left + int(plot_w * idx / 5)
        draw_line(pixels, width, height, x, top, x, top + plot_h, grid)

    def to_xy(x_value: float, y_value: float) -> tuple[int, int]:
        scaled_y = math.log10(y_value) if log_y else y_value
        x = left + int((x_value - min_x) / (max_x - min_x) * plot_w)
        y = top + plot_h - int((scaled_y - min_y) / (max_y - min_y) * plot_h)
        return x, y

    if horizontal_line is not None:
        y_value = math.log10(horizontal_line) if log_y else horizontal_line
        if min_y <= y_value <= max_y:
            y = top + plot_h - int((y_value - min_y) / (max_y - min_y) * plot_h)
            draw_line(pixels, width, height, left, y, left + plot_w, y, (0, 0, 0))

    for group_idx, group_rows in enumerate(groups.values()):
        color = COLORS[group_idx % len(COLORS)]
        usable = [
            row for row in group_rows
            if row.get(y_key) is not None and (not log_y or float(row[y_key]) > 0)
        ]
        if not usable:
            continue
        coords = [to_xy(float(row[x_key]), float(row[y_key])) for row in usable]
        for start, end in zip(coords, coords[1:]):
            draw_line(pixels, width, height, start[0], start[1], end[0], end[1], color)
        for x, y in coords:
            for ox in range(-4, 5):
                for oy in range(-4, 5):
                    set_pixel(pixels, width, height, x + ox, y + oy, color)

    write_png(output_path, width, height, pixels)


def plot_time(rows: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]) -> None:
    if plt is None:
        fallback_plot(rows, "time_median_s", output_path, log_y=True)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group_rows in group_by_algo(rows).items():
        algo = str(group_rows[0].get("algo"))
        ax.plot(
            [row["input_qubits"] for row in group_rows],
            [row["time_median_s"] for row in group_rows],
            marker=MARKERS.get(algo, "o"),
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax.set_title(f"{title_prefix(metadata)}\nExecution Time vs. Qubits")
    ax.set_xlabel("Input qubits (logical n)")
    ax.set_ylabel("Median execution time (s)")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="-", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)
    ax.legend(title="Circuit", fontsize=9, title_fontsize=9, ncols=2)
    add_run_note(fig, metadata)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def positive_energy_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    energy_rows = [
        row for row in rows
        if (
            row.get("energy_source") == "rapl_measured" and
            row.get("energy_median_j") is not None
        )
    ]
    positive_rows = [
        row for row in energy_rows
        if float(row.get("energy_median_j") or 0.0) > 0.0
    ]
    return positive_rows, len(energy_rows) - len(positive_rows)


def plot_energy(rows: list[dict[str, Any]], output_path: Path, metadata: dict[str, Any]) -> bool:
    energy_rows, omitted_nonpositive = positive_energy_rows(rows)
    if plt is None:
        if not energy_rows:
            return False
        fallback_plot(energy_rows, "energy_median_j", output_path, x_key="allocated_qubits", log_y=True)
        return True

    fig, ax = plt.subplots(figsize=(10, 6))
    if not energy_rows:
        ax.axis("off")
        ax.set_title(f"{title_prefix(metadata)}\nEnergy vs. Qubits")
        ax.text(
            0.5,
            0.55,
            "Energy telemetry unavailable",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.45,
            "Run with readable Linux RAPL counters to produce measured Joule curves.",
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )
        add_run_note(fig, metadata)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return True

    for label, group_rows in group_by_algo(energy_rows, x_key="allocated_qubits").items():
        algo = str(group_rows[0].get("algo"))
        ax.plot(
            [row["allocated_qubits"] for row in group_rows],
            [row["energy_median_j"] for row in group_rows],
            marker=MARKERS.get(algo, "o"),
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax.set_title(f"{title_prefix(metadata)}\nEnergy vs. Qubits")
    ax.set_xlabel("Allocated qubits (state-vector size)")
    ax.set_ylabel("Median measured CPU energy (J)")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="-", alpha=0.28)
    ax.grid(True, which="minor", linestyle=":", alpha=0.18)
    if omitted_nonpositive:
        ax.text(
            0.01,
            0.02,
            f"Omitted {omitted_nonpositive} non-positive median-energy points below RAPL resolution.",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
    ax.legend(title="Circuit", fontsize=9, title_fontsize=9, ncols=2)
    add_run_note(fig, metadata)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def speedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("algo"),
        row.get("input_qubits"),
        row.get("allocated_qubits"),
        row.get("depth"),
        row.get("threads"),
    )


def plot_speedup(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    output_path: Path,
    metadata: dict[str, Any],
) -> bool:
    baseline = {
        speedup_key(row): float(row["time_median_s"])
        for row in successful_rows(baseline_rows)
        if row.get("time_median_s") is not None
    }
    speedup_rows = []
    for row in rows:
        base_time = baseline.get(speedup_key(row))
        current_time = row.get("time_median_s")
        if base_time and current_time:
            copy = dict(row)
            copy["speedup"] = base_time / float(current_time)
            speedup_rows.append(copy)
    if not speedup_rows:
        return False
    if plt is None:
        fallback_plot(speedup_rows, "speedup", output_path, horizontal_line=1.0)
        return True

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group_rows in group_by_algo(speedup_rows).items():
        algo = str(group_rows[0].get("algo"))
        ax.plot(
            [row["input_qubits"] for row in group_rows],
            [row["speedup"] for row in group_rows],
            marker=MARKERS.get(algo, "o"),
            linewidth=2,
            markersize=6,
            label=label,
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{title_prefix(metadata)}\nSpeedup Relative to Baseline")
    ax.set_xlabel("Input qubits (logical n)")
    ax.set_ylabel("Baseline median time / current median time")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Circuit", fontsize=9, title_fontsize=9, ncols=2)
    add_run_note(fig, metadata)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a QuEST benchmark run directory.")
    parser.add_argument("run_dir", help="Run directory containing summary.csv.")
    parser.add_argument("--baseline", help="Optional baseline run directory for speedup plots.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    metadata = load_run_metadata(run_dir)

    rows = successful_rows(load_summary(run_dir))
    if not rows:
        raise SystemExit("No successful summary rows to plot.")

    plot_time(rows, plots_dir / "time_vs_qubits.png", metadata)
    energy_written = plot_energy(rows, plots_dir / "energy_vs_qubits.png", metadata)

    speedup_written = False
    if args.baseline:
        baseline_rows = load_summary(Path(args.baseline).resolve())
        speedup_written = plot_speedup(rows, baseline_rows, plots_dir / "speedup_vs_baseline.png", metadata)

    print(f"Wrote {plots_dir / 'time_vs_qubits.png'}")
    if energy_written:
        print(f"Wrote {plots_dir / 'energy_vs_qubits.png'}")
    if speedup_written:
        print(f"Wrote {plots_dir / 'speedup_vs_baseline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
