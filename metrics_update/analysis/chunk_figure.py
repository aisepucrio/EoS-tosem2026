"""
Illustration of the areas of interest used in the eye-tracking analysis: the inside/outside
distinction and the chunks into which the target code is divided.

Usage (from the repository root):

    python metrics_update/analysis/chunk_figure.py [--snippet 2] [--before 1] [--after 2]

The line -> chunk mapping is read from the fixation data (``chunk_num`` of the fixations on each
line; the most frequent value is used). Lines between the START/END markers are "inside"; every
other line of the displayed file is "outside". Lines without fixations take the chunk of the
previous line.

Output: ``metrics_update/outputs/snippets/chunk_example_snippet<id>.pdf``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from snippet_characteristics import FIXATIONS_PATH, OUTPUT_DIR, ROOT_DIR, displayed_region, snippet_file

CHUNK_COLORS = ["#cfe3f5", "#fbe0c3", "#d4eed1", "#f6d0d6", "#e2d8f0", "#f9f1bd", "#d0ece9", "#ead7c8"]
OUTSIDE_COLOR = "#ececec"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"


def line_chunks(snippet_id: int) -> pd.Series:
    fixations = pd.read_csv(FIXATIONS_PATH, sep="\t", usecols=["fixation_target", "source_file_line", "chunk_num"])
    fixations = fixations.loc[fixations["fixation_target"].eq(snippet_id)]

    return fixations.groupby("source_file_line")["chunk_num"].agg(lambda s: s.mode().iloc[0]).astype(int)


FONT_SIZE = 10.5
MAX_CHARS = 68  # longer lines are truncated with "…"


def draw(snippet_id: int, before: int, after: int, output: Path) -> None:
    lines = snippet_file(snippet_id).read_text(encoding="utf-8", errors="replace").splitlines()
    first, last = displayed_region(snippet_id)
    observed = line_chunks(snippet_id)

    start = max(1, first - 1 - before)
    end = min(len(lines), last + 1 + after)
    shown = range(start, end + 1)

    chunks = {}
    current = None
    for number in range(first, last + 1):
        current = int(observed.get(number, current if current is not None else 1))
        chunks[number] = current

    rows = len(shown) + (start > 1) + (end < len(lines))
    height = 0.29 * rows + 0.3
    fig, ax = plt.subplots(figsize=(9.0, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(rows - 0.4, -0.6)
    ax.axis("off")

    code_x, number_x, band_x0, band_x1 = 9.0, 7.6, 3.2, 82.0
    y = 0

    def ellipsis(text: str) -> None:
        nonlocal y
        ax.add_patch(Rectangle((band_x0, y - 0.5), band_x1 - band_x0, 1, color=OUTSIDE_COLOR, lw=0))
        ax.text(code_x, y, text, family="monospace", fontsize=FONT_SIZE, color=MUTED_COLOR, va="center", style="italic")
        y += 1

    if start > 1:
        ellipsis(f"...  lines 1–{start - 1} (license, imports, fields, other methods)")

    chunk_spans: dict[int, list[int]] = {}
    for number in shown:
        inside = first <= number <= last
        color = CHUNK_COLORS[(chunks[number] - 1) % len(CHUNK_COLORS)] if inside else OUTSIDE_COLOR
        ax.add_patch(Rectangle((band_x0, y - 0.5), band_x1 - band_x0, 1, color=color, lw=0))
        ax.text(number_x, y, str(number), family="monospace", fontsize=FONT_SIZE, color=MUTED_COLOR, ha="right", va="center")
        text = lines[number - 1].replace("\t", "    ")
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 1] + "…"
        ax.text(code_x, y, text, family="monospace", fontsize=FONT_SIZE, color=TEXT_COLOR, va="center")
        if inside:
            chunk_spans.setdefault(chunks[number], []).append(y)
        y += 1

    if end < len(lines):
        ellipsis(f"...  lines {end + 1}–{len(lines)}")

    # chunk brackets
    for chunk, ys in chunk_spans.items():
        top, bottom = min(ys) - 0.4, max(ys) + 0.4
        x = band_x1 + 0.8
        ax.plot([x, x + 0.7, x + 0.7, x], [top, top, bottom, bottom], color=TEXT_COLOR, lw=0.7)
        ax.text(x + 1.8, (top + bottom) / 2, f"Chunk {chunk}", fontsize=FONT_SIZE + 0.5, va="center", color=TEXT_COLOR)

    # inside / outside rail on the left
    inside_top = (first - start) + (start > 1) - 0.5
    inside_bottom = inside_top + (last - first + 1)
    rail_x = 2.1
    ax.plot([rail_x, rail_x], [-0.5, inside_top], color="#9a9a9a", lw=2.2, solid_capstyle="butt")
    ax.plot([rail_x, rail_x], [inside_top, inside_bottom], color="#2b6cb0", lw=2.2, solid_capstyle="butt")
    ax.plot([rail_x, rail_x], [inside_bottom, rows - 0.5], color="#9a9a9a", lw=2.2, solid_capstyle="butt")
    for label, (y0, y1), color in (
        ("Outside", (-0.5, inside_top), "#6b6b6b"),
        ("Inside (target code)", (inside_top, inside_bottom), "#2b6cb0"),
        ("Outside", (inside_bottom, rows - 0.5), "#6b6b6b"),
    ):
        ax.text(rail_x - 0.9, (y0 + y1) / 2, label, rotation=90, fontsize=FONT_SIZE, ha="right", va="center", color=color)

    fig.tight_layout(pad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snippet", type=int, default=2)
    parser.add_argument("--before", type=int, default=1, help="outside lines shown above the target code")
    parser.add_argument("--after", type=int, default=2, help="outside lines shown below the target code")
    args = parser.parse_args()

    output = OUTPUT_DIR / f"chunk_example_snippet{args.snippet}.pdf"
    draw(args.snippet, args.before, args.after, output)
    print("Saved:", output.relative_to(ROOT_DIR))


if __name__ == "__main__":
    main()
