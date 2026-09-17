"""Recover the data points from a vector plot in a PDF, exactly rather than by digitisation."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

# U+2212 MINUS SIGN, not the ASCII hyphen: matplotlib writes negative tick labels with it, so an ASCII-only.
NUMERIC = re.compile(r"^[-\u2212]?\d+(\.\d+)?$")


def to_float(token: str) -> float:
    """Parse a tick label, accepting the Unicode minus matplotlib emits."""
    return float(token.replace("\u2212", "-"))


def extract_marks(page: object) -> list[dict]:
    """Collect the small filled paths that represent data points."""
    marks = []
    for drawing in page.get_drawings():  # ty: ignore[unresolved-attribute]
        rect, fill = drawing["rect"], drawing.get("fill")
        if fill and rect.width < 6 and rect.height < 6 and rect.width > 0:
            marks.append({"x": (rect.x0 + rect.x1) / 2, "y": (rect.y0 + rect.y1) / 2,
                          "colour": tuple(round(c, 3) for c in fill)})
    return marks


def cluster(values: list[float], gap: float) -> list[list[float]]:
    """Group sorted values wherever consecutive ones differ by more than ``gap``."""
    groups: list[list[float]] = []
    for v in sorted(values):
        if groups and v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def _descendants(root: int, parent: dict[int, int]) -> set[int]:
    """Every index whose parent chain ends at ``root``."""
    out = set()
    for child in parent:
        node = child
        while node in parent:
            node = parent[node]
        if node == root:
            out.add(child)
    return out


def method_labels(words: list[tuple[str, float, float]], x_lo: float, x_hi: float,
                  y_bottom: float, depth: float = 34.0) -> tuple[list[str], set[int]]:
    """Read the rotated x-axis tick labels under a panel, left to right."""
    # Measured on page 9: a label's first line starts 2.4 pt left of its marker column, and the two share a.
    band = [(i, t, x, y) for i, (t, x, y) in enumerate(words)
            if y_bottom < y < y_bottom + depth and x_lo - 6 < x < x_hi + 16
            and not NUMERIC.match(t)]
    parent: dict[int, int] = {}
    for i, _, x, y in band:
        # Nearest word above-left at ~45 degrees; 5.0 pt separates the real offsets (|dx - dy| of
        # 0.4 to 2.0) from every wrong pairing (4.5 and up).
        best = min(((abs((x - px) - (y - py)), pi) for pi, _, px, py in band
                    if x - px > 5 and y - py > 5), default=(None, None))
        if best[0] is not None and best[0] < 5.0:
            parent[i] = best[1]
    lines: dict[int, list[tuple[float, str]]] = {}
    for i, t, _, y in band:
        root = i
        while root in parent:
            root = parent[root]
        lines.setdefault(root, []).append((y, t))
    by_x = {i: x for i, _, x, _ in band}
    # Continuations may reach past the last column; heads may not. Dropping the overshoot here
    # rather than in the band keeps a two-line last label whole.
    roots = [r for r in lines if by_x[r] < x_hi + 6]
    labels = [" ".join(t for _, t in sorted(lines[root]))
              for root in sorted(roots, key=lambda r: by_x[r])]
    consumed = {i for root in roots for i in _descendants(root, parent)} | set(roots)
    return labels, consumed


def _residual(ticks: list[tuple[float, float]], fit: tuple[float, float]) -> float:
    """Mean squared error of a log10 fit against its ticks, used only to compare two candidates."""
    import math

    a, b = fit
    return sum((a * y + b - math.log10(v)) ** 2 for y, v in ticks) / len(ticks)


def restore_log_decades(ticks: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Rebuild the decade a log axis's repeated minor-tick labels lost."""
    values = [v for _, v in ticks]
    if len(values) < 3 or not any(values.count(v) > 1 for v in values) or not all(v > 0 for v in values):
        return None
    # Device y grows downward, so ascending the axis means descending y.
    climbing = sorted(ticks, key=lambda t: -t[0])
    restored: list[tuple[float, float]] = []
    last = 0.0
    for y, label in climbing:
        scaled = label
        while scaled <= last:
            scaled *= 10.0
        restored.append((y, scaled))
        last = scaled
    return restored


def calibrate_axis(ticks: list[tuple[float, float]]) -> tuple[str, float, float] | None:
    """Choose a linear or log10 mapping from device y to data value, whichever fits the ticks."""
    import math


    linear = calibrate(ticks)
    if linear is None:
        return None
    values = [v for _, v in ticks]
    # From a property of the TICKS, not by comparing residuals in incommensurable units -- that comparison.
    restored = restore_log_decades(ticks)
    if restored is not None:
        raw_fit = calibrate([(y, math.log10(v)) for y, v in ticks])
        new_fit = calibrate([(y, math.log10(v)) for y, v in restored])
        if new_fit is not None and (raw_fit is None or _residual(restored, new_fit) < _residual(ticks, raw_fit)):
            return "log10", new_fit[0], new_fit[1]

    if all(v > 0 for v in values) and max(values) / min(values) >= 10:
        logged = [(y, math.log10(v)) for y, v in ticks]
        fit = calibrate(logged)
        if fit is not None:
            return "log10", fit[0], fit[1]
    return "linear", linear[0], linear[1]


def calibrate(ticks: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Fit ``value = a * device_y + b`` from tick (device_y, value) pairs by least squares."""
    if len(ticks) < 2:
        return None
    n = len(ticks)
    sx = sum(t[0] for t in ticks)
    sy = sum(t[1] for t in ticks)
    sxx = sum(t[0] * t[0] for t in ticks)
    sxy = sum(t[0] * t[1] for t in ticks)
    denominator = n * sxx - sx * sx
    if abs(denominator) < 1e-9:
        return None
    a = (n * sxy - sx * sy) / denominator
    return a, (sy - a * sx) / n


def main(
    pdf: Annotated[Path, typer.Option(help="PDF containing the figure")] = Path("paper.pdf"),
    page_number: Annotated[int, typer.Option(help="1-based page holding the figure")] = 9,
    output: Annotated[Path, typer.Option()] = Path("metrics/evoflows_figure3.json"),
) -> None:
    """Recover Figure 3's points and write them as JSON."""
    import pymupdf

    if not pdf.exists():
        raise typer.BadParameter(f"{pdf} does not exist")
    page = pymupdf.open(pdf)[page_number - 1]
    marks = extract_marks(page)
    if not marks:
        raise typer.BadParameter(
            f"no vector markers on page {page_number}; if the figure is a raster image use a "
            f"digitiser (WebPlotDigitizer / plotdigitizer) instead"
        )
    logger.info(f"{len(marks)} markers, {len({m['colour'] for m in marks})} colours")

    # Panels: two rows of five. Cluster on y first (rows), then x within a row (columns).
    words = [(w[4], (w[0] + w[2]) / 2, (w[1] + w[3]) / 2) for w in page.get_text("words")]
    # Rotated tick labels are anchored by their LEFT edge, not their centre: the bounding box of rotated text.
    words_left = [(w[4], w[0], (w[1] + w[3]) / 2) for w in page.get_text("words")]
    rows = cluster([m["y"] for m in marks], gap=40)
    logger.info(f"marker rows: {[len(r) for r in rows]}")

    panels = []
    for row in rows:
        lo, hi = min(row), max(row)
        in_row = [m for m in marks if lo - 1 <= m["y"] <= hi + 1]
        # 15 is measured: the threshold must sit between this figure's 22.0 pt panel-to-panel gap and its 7.9.
        for column in cluster([m["x"] for m in in_row], gap=15):
            x_lo, x_hi = min(column), max(column)
            panels.append([m for m in in_row if x_lo - 1 <= m["x"] <= x_hi + 1])
    logger.info(f"{len(panels)} panels with sizes {[len(p) for p in panels]}")

    # Labels first, for every panel, because a title window reaches back far enough to catch the method labels.
    geometry = [(min(m["x"] for m in p), max(m["x"] for m in p),
                 min(m["y"] for m in p), max(m["y"] for m in p)) for p in panels]
    per_panel_labels = []
    claimed: set[int] = set()
    for x_lo, x_hi, _, y_hi in geometry:
        labels, consumed = method_labels(words_left, x_lo, x_hi, y_hi)
        per_panel_labels.append(labels)
        claimed |= consumed

    report = {}
    for index, panel in enumerate(panels):
        x_lo, x_hi, y_lo, y_hi = geometry[index]
        labels = per_panel_labels[index]

        # y tick labels: numeric words just left of the panel, vertically within its span.
        ticks = [(wy, to_float(t)) for t, wx, wy in words
                 if NUMERIC.match(t) and x_lo - 60 < wx < x_lo - 2 and y_lo - 25 < wy < y_hi + 25]
        fit = calibrate_axis(ticks)
        # Panel title: words above the panel, minus every word any panel claimed as a tick label, minus bare.
        title = " ".join(t for word_index, (t, wx, wy) in enumerate(words)
                         if x_lo - 20 < wx < x_hi + 20 and y_lo - 42 < wy < y_lo - 6
                         and word_index not in claimed and not NUMERIC.match(t))

        # Methods occupy distinct x positions; six columns of six datasets each.
        by_method = defaultdict(list)
        for m in panel:
            by_method[m["colour"]].append(m)
        # Report the ticks the fit was actually built from.
        restored = restore_log_decades(ticks)
        entry: dict = {"title": title or f"panel {index}", "n_points": len(panel),
                       "y_ticks": sorted({v for _, v in (restored or ticks)}),
                       "y_tick_labels_as_printed": sorted({v for _, v in ticks}),
                       "log_decades_restored": restored is not None,
                       "calibrated": fit is not None, "methods": labels}
        if fit:
            kind, a, b = fit
            entry["axis_scale"] = kind

            def value(mark: dict, kind: str = kind, a: float = a, b: float = b) -> float:
                raw = a * mark["y"] + b
                return round(10 ** raw if kind == "log10" else raw, 5)

            ordered = sorted(by_method.items(), key=lambda kv: min(p["x"] for p in kv[1]))
            entry["by_colour"] = {
                str(colour): sorted(value(m) for m in group) for colour, group in ordered
            }
            # The same values keyed by method rather than by colour, which is what makes the figure citable.
            if len(labels) == len(ordered):
                entry["by_method"] = {
                    label: sorted(value(m) for m in group)
                    for label, (_, group) in zip(labels, ordered, strict=True)
                }
        report[f"panel_{index}"] = entry
        logger.info(f"  {entry['title'][:34]:<34} n={len(panel):<3} ticks={entry['y_ticks']} "
                    f"methods={len(labels)} {'OK' if fit else 'NOT CALIBRATED'}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    logger.info(f"wrote {output}")


if __name__ == "__main__":
    typer.run(main)
