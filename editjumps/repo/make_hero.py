"""Generate the README hero SVG in a light and a dark variant."""

from pathlib import Path

# GitHub's own light and dark palettes, so the hero sits in the page rather than on it.
PALETTES = {
    "light": {"ink": "#1f2328", "muted": "#59636e", "tile": "#f6f8fa", "edge": "#d1d9e0",
              "sub": "#8250df", "ins": "#1a7f37", "dele": "#cf222e", "rule": "#d1d9e0"},
    "dark": {"ink": "#e6edf3", "muted": "#9198a1", "tile": "#161b22", "edge": "#30363d",
             "sub": "#a371f7", "ins": "#3fb950", "dele": "#f85149", "rule": "#30363d"},
}

#: The figure is an ALIGNMENT, not two independent strings. A deleted residue leaves a gap beneath
#: it and an inserted one a gap above it - the same epsilon the edit-flow path carries internally.
#: Without those gaps a delete and an insert drawn in the same column are just a substitution,
#: which is what the figure used to show.
#:
#: Columns are ``(input residue, edited residue, operation)``; ``None`` is the gap.
COLUMNS: tuple[tuple[str | None, str | None, str | None], ...] = (
    ("Q", "Q", None),
    ("V", "V", None),
    ("Q", "Q", None),
    ("L", "L", None),
    (None, "Y", "insert"),        # no counterpart above: the variant is longer here
    ("V", "V", None),
    ("E", "Q", "substitute"),     # same column in both rows: length is unchanged
    ("S", "S", None),
    ("G", None, "delete"),        # no counterpart below: the variant is shorter here
)

#: Operation -> palette key.
OP_COLOUR = {"substitute": "sub", "insert": "ins", "delete": "dele"}

INPUT = "".join(residue for residue, _, _ in COLUMNS if residue)
OUTPUT = "".join(residue for _, residue, _ in COLUMNS if residue)

W, H = 680, 372
TILE, GAP = 44, 8
ROW_IN, ROW_OUT = 168, 278
#: The tile block is CENTRED, not indented from a fixed left margin.
ROW_WIDTH = len(COLUMNS) * (TILE + GAP) - GAP
LEFT = (W - ROW_WIDTH) // 2
MID = W // 2
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def tile_x(index: int) -> int:
    """Left edge of the tile at ``index``."""
    return LEFT + index * (TILE + GAP)


def row(side: int, y: int, p: dict) -> list[str]:
    """Emit one row of residue tiles; ``side`` is 0 for the input row and 1 for the edited one."""
    out = []
    for index, column in enumerate(COLUMNS):
        residue, op = column[side], column[2]
        x = tile_x(index)
        colour = p[OP_COLOUR[op]] if op else None
        if residue is None:
            # The gap an indel leaves. Drawing the empty slot rather than closing it up is the whole
            # point: it shows this row is one residue shorter here, instead of letting the column
            # read as a substitution.
            out.append(f'<rect x="{x}" y="{y}" width="{TILE}" height="{TILE}" rx="8" fill="none" '
                       f'stroke="{colour}" stroke-width="2" stroke-dasharray="4 4" '
                       f'stroke-opacity="0.5"/>')
            continue
        edge = colour if colour else p["edge"]
        text = colour if colour else p["ink"]
        weight = "600" if colour else "400"
        # fill-opacity rather than an 8-digit hex: SVG 1.1 renderers ignore #rrggbbaa and paint the
        # tile solid, which hides the residue letter underneath it.
        out.append(f'<rect x="{x}" y="{y}" width="{TILE}" height="{TILE}" rx="8" '
                   f'fill="{colour or p["tile"]}" fill-opacity="{0.14 if colour else 1}" '
                   f'stroke="{edge}" stroke-width="{2 if colour else 1}"/>')
        out.append(f'<text x="{x + TILE // 2}" y="{y + 30}" font-family="{MONO}" font-size="20" '
                   f'font-weight="{weight}" fill="{text}" text-anchor="middle">{residue}</text>')
        if op == "delete" and side == 0:
            out.append(f'<line x1="{x + 8}" y1="{y + TILE // 2}" x2="{x + TILE - 8}" '
                       f'y2="{y + TILE // 2}" stroke="{colour}" stroke-width="2.5"/>')
    return out


def render(theme: str) -> str:
    """Build the whole SVG for one theme."""
    p = PALETTES[theme]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="EditJumps: one sequence in, edited variants out">',
        '<title>EditJumps - fine-grained editing of protein sequences</title>',
        # Wordmark. "Edit" muted, "Jumps" in ink, so the name reads as two parts without a second font.
        f'<text x="{MID}" y="72" font-family="{SANS}" font-size="46" font-weight="700" '
        f'text-anchor="end" fill="{p["muted"]}">Edit</text>',
        f'<text x="{MID}" y="72" font-family="{SANS}" font-size="46" font-weight="700" '
        f'text-anchor="start" fill="{p["ink"]}">Jumps</text>',
        f'<text x="{MID}" y="102" font-family="{SANS}" font-size="16" fill="{p["muted"]}" '
        f'text-anchor="middle">'
        f'Fine-grained protein sequence editing with learned generative jump edits.</text>',
        f'<line x1="60" y1="124" x2="{W - 60}" y2="124" stroke="{p["rule"]}" stroke-width="1"/>',
    ]
    parts += row(0, ROW_IN, p)
    parts += row(1, ROW_OUT, p)

    # Row captions, centred above the first row and below the second, so the composition stays
    # symmetric. Hanging them off the right edge was what made the centred image look off-axis.
    parts.append(f'<text x="{MID}" y="{ROW_IN - 12}" font-family="{SANS}" font-size="13" '
                 f'fill="{p["muted"]}" text-anchor="middle">your sequence</text>')
    parts.append(f'<text x="{MID}" y="{ROW_OUT + TILE + 24}" font-family="{SANS}" font-size="13" '
                 f'fill="{p["muted"]}" text-anchor="middle">an edited variant</text>')

    # One connector per operation, each spanning both rows. Every operation now relates a column to
    # the SAME column in the other row - a residue, or the gap where one is missing - so all three
    # are drawn the same way.
    for index, (_, _, op) in enumerate(COLUMNS):
        if op is None:
            continue
        cx, colour = tile_x(index) + TILE // 2, p[OP_COLOUR[op]]
        parts.append(f'<path d="M {cx} {ROW_IN + TILE + 4} L {cx} {ROW_OUT - 4}" '
                     f'stroke="{colour}" stroke-width="2" stroke-dasharray="3 3" fill="none"/>')
        parts.append(f'<text x="{cx + 10}" y="{ROW_IN + TILE + 36}" font-family="{SANS}" '
                     f'font-size="13" font-weight="600" fill="{colour}">{op}</text>')

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Write both variants into ``.github/images/``."""
    target = Path(__file__).resolve().parents[2] / ".github" / "images"
    target.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        path = target / f"hero-{theme}.svg"
        path.write_text(render(theme))
        print(f"wrote {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
