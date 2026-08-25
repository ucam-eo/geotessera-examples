# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr>=3.0", "xarray>=2024.10", "numpy", "rasterio", "shapely"]
# ///
"""Step 3 of 5: draw the classification as a vector SVG.

    uv run 03_render_svg.py prediction.zarr

An SVG is a set of shapes, so it stays sharp at any zoom and can be
restyled in a vector editor. Producing one means tracing the outline of
every run of same-coloured pixels with rasterio.features.shapes.

A per-pixel classifier produces speckle, which is a problem for vector
output: the raw k-NN map of a 22km square traces to about 240,000
polygons and 12MB of SVG, and simplification barely helps because most
polygons are single pixels. rasterio.features.sieve absorbs any patch
smaller than a threshold into its surrounding neighbour; at the default
of 32 pixels the same map becomes about 4,000 polygons and under 1MB.

Sieving changes the map. At size 32 it relabels roughly a sixth of the
pixels. It is applied to all three panels equally, the amount is
printed, and prediction.zarr holds the original. Pass --sieve 1 to
disable it.
"""

import argparse
import json

import numpy as np
import xarray as xr
from rasterio.features import shapes, sieve
from shapely.geometry import shape

COLOURS = {
    "cropland": "#d8b365",
    "grassland": "#a6d96a",
    "woodland": "#1a9641",
    "water": "#2c7fb8",
    "built": "#bdbdbd",
    "road": "#404040",
}

PANELS = [
    ("osm_labels", "OSM labels", None),
    ("prediction_knn", "k-NN", "accuracy_knn"),
    ("prediction_logreg", "logistic regression", "accuracy_logreg"),
]

GAP, MARGIN, TITLES, LEGEND = 60, 40, 150, 130


def trace(grid, sieve_size, simplify):
    """Turn a grid of class ids into {class id: [polygon]}, in pixel units."""
    changed = 0.0
    if sieve_size > 1:
        cleaned = sieve(grid, size=sieve_size)
        changed = float((cleaned != grid).mean())
        grid = cleaned

    # With no transform given, shapes() reports pixel corners, which are whole
    # numbers and are already the coordinates we want to write into the SVG.
    # Simplifying only ever drops existing corners, so they stay whole numbers.
    polygons = {}
    for outline, class_id in shapes(grid, mask=grid > 0):
        polygon = shape(outline).simplify(simplify)
        if not polygon.is_empty:
            polygons.setdefault(int(class_id), []).append(polygon)
    return polygons, changed


def ring_to_path(points):
    """One closed outline as SVG path commands."""
    points = points[:-1]  # the last point repeats the first; Z does that job
    if len(points) < 3:
        return ""
    start = f"M{points[0][0]:g},{points[0][1]:g}"
    return start + "".join(f"L{x:g},{y:g}" for x, y in points[1:]) + "Z"


def draw(panels, width, height, classes, subtitle):
    """Build the whole SVG as one string."""
    total_width = 2 * MARGIN + 3 * width + 2 * GAP
    total_height = 2 * MARGIN + TITLES + height + LEGEND

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {total_width} {total_height}" '
           f'width="{total_width}" height="{total_height}" '
           f'font-family="Helvetica,Arial,sans-serif">',
           "<style>",
           "  .frame{fill:#fff;stroke:#333;stroke-width:3}",
           "  path{fill-rule:evenodd}",  # so holes in polygons show through
           *(f"  .{name}{{fill:{colour}}}" for name, colour in COLOURS.items()),
           "</style>",
           # SVG has no background of its own; without this the text is
           # invisible to anyone viewing on a dark background.
           f'<rect width="{total_width}" height="{total_height}" fill="#fff"/>',
           f'<text x="{total_width / 2:g}" y="{MARGIN + 38}" font-size="40" '
           f'text-anchor="middle">{subtitle}</text>']

    for column, (title, polygons) in enumerate(panels):
        left, top = MARGIN + column * (width + GAP), MARGIN + TITLES
        svg.append(f'<text x="{left + width / 2:g}" y="{top - 30}" font-size="44" '
                   f'text-anchor="middle">{title}</text>')
        svg.append(f'<g transform="translate({left},{top})">')
        svg.append(f'<rect class="frame" width="{width}" height="{height}"/>')
        for class_id, name in enumerate(classes, start=1):
            # Every polygon of one class goes into a single <path>, so the whole
            # figure is 18 shapes rather than several thousand.
            outlines = []
            for polygon in polygons.get(class_id, []):
                outlines.append(ring_to_path(list(polygon.exterior.coords)))
                outlines += [ring_to_path(list(h.coords)) for h in polygon.interiors]
            drawing = "".join(o for o in outlines if o)
            if drawing:
                svg.append(f'<path class="{name}" d="{drawing}"/>')
        svg.append("</g>")

    baseline = MARGIN + TITLES + height + 60
    step = total_width / (len(classes) + 1)
    for i, name in enumerate(classes):
        x = step * (i + 0.75)
        svg.append(f'<rect class="{name}" x="{x:g}" y="{baseline - 38:g}" '
                   f'width="46" height="46" stroke="#333" stroke-width="2"/>')
        svg.append(f'<text x="{x + 60:g}" y="{baseline:g}" font-size="38">{name}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", nargs="?", default="prediction.zarr")
    parser.add_argument("-o", "--out", default="prediction.svg")
    parser.add_argument("--sieve", type=int, default=32,
                        help="absorb patches smaller than this many pixels; 1 disables")
    parser.add_argument("--simplify", type=float, default=2.0,
                        help="outline simplification tolerance, in pixels")
    args = parser.parse_args()

    data = xr.open_zarr(args.store, zarr_format=3, consolidated=True).load()
    classes = json.loads(data.attrs["classes"])
    height, width = data["osm_labels"].shape

    panels = []
    for name, title, accuracy_key in PANELS:
        polygons, changed = trace(data[name].values, args.sieve, args.simplify)
        count = sum(len(v) for v in polygons.values())
        print(f"  {name:<18} {count:>7,} polygons, {changed * 100:4.1f}% of pixels "
              f"changed by sieving")
        if accuracy_key in data.attrs:
            title += f" ({data.attrs[accuracy_key]:.3f})"
        panels.append((title, polygons))

    west, south, east, north = data.attrs["bbox"]
    subtitle = (f"Tessera {data.attrs['year']} — {west:.2f},{south:.2f} to "
                f"{east:.2f},{north:.2f} · sieve={args.sieve} px")
    svg = draw(panels, width, height, classes, subtitle)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {args.out} ({len(svg.encode()) / 1e6:.2f} MB)")
    if args.sieve > 1:
        print("note: accuracies above were measured before sieving")


if __name__ == "__main__":
    main()
