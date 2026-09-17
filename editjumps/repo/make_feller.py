"""Generate the pixelated Feller portrait used in the README."""

import argparse
import urllib.request
from pathlib import Path

SOURCE = "https://mathshistory.st-andrews.ac.uk/Biographies/Feller/Feller_3.jpeg"
COLUMNS = 58
TONES = 8

#: Tone ramps, darkest to lightest.
RAMPS = {
    "light": ["#1f2328", "#32383f", "#4c545d", "#6e7781", "#9198a1", "#b7bfc9", "#dde3ea", "#ffffff"],
    "dark": ["#010409", "#0d1117", "#21262d", "#3d444d", "#656c76", "#8d96a0", "#b7bfc9", "#e6edf3"],
}
assert all(len(ramp) == TONES for ramp in RAMPS.values()), "every ramp needs one colour per tone"


def load_pixels(path: Path) -> tuple[list[list[float]], int, int]:
    """Read the JPEG and return row-major luminance in 0..1, plus its size."""
    from PIL import Image

    image = Image.open(path).convert("L")
    width, height = image.size
    raw = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
    # Convert("L") guarantees one band, so every sample is a scalar.
    data: list[int] = [int(v[0]) if isinstance(v, tuple) else int(v) for v in raw]
    # Percentile stretch. The scan is low-contrast, and quantising it straight washed the face out
    # and lost the spectacle frames -- the one feature that makes the portrait recognisable at all.
    ordered = sorted(data)
    lo = ordered[int(0.02 * len(ordered))]
    hi = ordered[int(0.98 * len(ordered))]
    span = max(1, hi - lo)
    scaled = [min(1.0, max(0.0, (v - lo) / span)) for v in data]
    return [[scaled[y * width + x] for x in range(width)] for y in range(height)], width, height


def downsample(rows: list[list[float]], width: int, height: int, columns: int) -> list[list[int]]:
    """Box-average into ``columns`` blocks wide and quantise each block to a tone index."""
    step = width / columns
    lines = int(height / step)
    out = []
    for line in range(lines):
        row = []
        for column in range(columns):
            x0, x1 = int(column * step), max(int(column * step) + 1, int((column + 1) * step))
            y0, y1 = int(line * step), max(int(line * step) + 1, int((line + 1) * step))
            cells = [rows[y][x] for y in range(y0, min(y1, height)) for x in range(x0, min(x1, width))]
            mean = sum(cells) / len(cells) if cells else 0.0
            row.append(min(TONES - 1, int(mean * TONES)))
        out.append(row)
    return out


def render(blocks: list[list[int]], theme: str, block: int = 8) -> str:
    """Emit the quantised portrait as one SVG rect per block."""
    ramp = RAMPS[theme]
    w, h = len(blocks[0]) * block, len(blocks) * block
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
             f'height="{h}" shape-rendering="crispEdges" role="img" '
             f'aria-label="Pixelated portrait of William Feller">',
             "<title>William Feller (1906-1970)</title>",
             f'<rect width="{w}" height="{h}" fill="{ramp[0]}"/>']
    # Runs of equal tone collapse into one rect: same picture, a third of the file.
    for y, row in enumerate(blocks):
        x = 0
        while x < len(row):
            run = x
            while run + 1 < len(row) and row[run + 1] == row[x]:
                run += 1
            if row[x] != 0:
                parts.append(f'<rect x="{x * block}" y="{y * block}" '
                             f'width="{(run - x + 1) * block}" height="{block}" fill="{ramp[row[x]]}"/>')
            x = run + 1
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    """Download the source if needed, then write both themed portraits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="local copy of the photograph; downloaded from MacTutor if omitted")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    source = args.source
    if source is None:
        source = root / ".github" / "images" / ".feller-source.jpg"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SOURCE, source)  # noqa: S310 - fixed https URL above
            print(f"downloaded {SOURCE}")

    rows, width, height = load_pixels(source)
    blocks = downsample(rows, width, height, COLUMNS)
    for theme in RAMPS:
        path = root / ".github" / "images" / f"feller-{theme}.svg"
        path.write_text(render(blocks, theme))
        print(f"wrote {path.relative_to(root)}")


if __name__ == "__main__":
    main()
