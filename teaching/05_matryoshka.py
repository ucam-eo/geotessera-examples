# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "zarr>=3.0", "xarray>=2024.10", "dask", "numpy", "pyproj",
#     "fsspec", "aiohttp", "requests",
#     "osmnx>=2.0", "geopandas", "shapely", "rasterio", "scikit-learn",
#     "matplotlib",
# ]
# ///
"""Step 5 of 5: classify with the 16-dimension matryoshka prefix.

The v2 model orders its 128 dimensions by importance, so a prefix of an
embedding is itself a usable embedding. The v2 store carries
``embeddings_d4`` and ``embeddings_d16`` arrays holding the first 4 and
16 dimensions, dequantised by the same ``scales``, so a client can read
16 dimensions for an eighth of the bytes of 128.

    uv run 05_matryoshka.py --lon 0.12 --lat 52.20

This script reads both ``embeddings_d16`` and the full ``embeddings``
for one window, trains the same k-NN on each, and draws the OSM labels
and the two predictions side by side to show what the reduced prefix
costs in accuracy.

Depth arrays exist only in v2 stores; steps 1 to 4 use v1.
"""

import argparse
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

STORE = "https://data.source.coop/tessera/tessera/zarr/v2-2B-L~beta1"
TILE = 0.1

CLASSES = ["cropland", "grassland", "woodland", "water", "built", "road"]
COLOURS = {
    "cropland": "#d8b365", "grassland": "#a6d96a", "woodland": "#1a9641",
    "water": "#2c7fb8", "built": "#bdbdbd", "road": "#404040",
}

OSM_TAGS = {
    "natural": ["water", "wood", "grassland", "heath", "scrub"],
    "landuse": ["forest", "farmland", "orchard", "meadow", "grass",
                "reservoir", "residential", "industrial", "commercial"],
    "leisure": ["park", "pitch", "golf_course"],
    "building": True,
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                "residential", "unclassified"],
}


def class_of(feature):
    """Which class is this OSM feature?  First match wins."""
    def tag(key):
        value = feature.get(key)
        return value if isinstance(value, str) else None

    natural, landuse, leisure = tag("natural"), tag("landuse"), tag("leisure")

    if tag("highway"):
        return "road"
    if tag("building"):
        return "built"
    if natural == "water" or landuse == "reservoir":
        return "water"
    if natural == "wood" or landuse == "forest":
        return "woodland"
    if landuse in ("farmland", "orchard"):
        return "cropland"
    if natural in ("grassland", "heath", "scrub") or leisure or landuse in ("meadow", "grass"):
        return "grassland"
    if landuse in ("residential", "industrial", "commercial"):
        return "built"
    return None


def read_window(lon, lat, year, tiles):
    """Read scales plus both depth arrays for a block of tiles."""
    from pyproj import Transformer

    west = math.floor(lon / TILE) * TILE
    south = math.floor(lat / TILE) * TILE
    east, north = west + tiles * TILE, south + tiles * TILE
    zone = int((lon + 180) / 6) % 60 + 1
    print(f"window {west:.2f},{south:.2f} to {east:.2f},{north:.2f} in UTM zone {zone}")

    store = xr.open_zarr(STORE, group=f"utm{zone:02d}", zarr_format=3,
                         consolidated=True,
                         chunks={"time": 1, "band": 128, "y": 256, "x": 256})
    crs = store.attrs["proj:code"]

    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [to_utm.transform(x, y) for x in (west, east) for y in (south, north)]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]

    window = store.sel(time=year, x=slice(min(xs), max(xs)),
                       y=slice(max(ys), min(ys)))
    print(f"  {window.sizes['y']} x {window.sizes['x']} pixels, downloading...")
    scales = fetch(window["scales"])
    emb16 = fetch(window["embeddings_d16"])
    emb128 = fetch(window["embeddings"])
    print(f"  d16: {emb16.nbytes / 1e6:.0f} MB, full: {emb128.nbytes / 1e6:.0f} MB")
    return {16: emb16, 128: emb128}, scales, window, crs, (west, south, east, north)


def fetch(array, attempts=5):
    """Load one variable, retrying if the server cuts a response short."""
    for attempt in range(1, attempts + 1):
        try:
            return array.values
        except Exception as problem:
            if attempt == attempts:
                raise
            print(f"  read failed ({type(problem).__name__}), retrying "
                  f"{attempt}/{attempts - 1}")
            time.sleep(5 * attempt)


def read_osm(bbox, crs, window, resolution=10.0):
    """Draw OSM land cover onto exactly the same pixel grid as the window."""
    import osmnx as ox
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    transform = Affine(resolution, 0, float(window.x[0]) - resolution / 2,
                       0, -resolution, float(window.y[0]) + resolution / 2)

    print("asking OpenStreetMap (this can take a minute)")
    west, south, east, north = bbox
    features = ox.features.features_from_bbox(bbox=(west, south, east, north), tags=OSM_TAGS)
    features = features.to_crs(crs)
    print(f"  {len(features)} features")

    labels = np.zeros((window.sizes["y"], window.sizes["x"]), dtype=np.uint8)
    for class_id, name in enumerate(CLASSES, start=1):
        chosen = features[[class_of(f) == name for _, f in features.iterrows()]]
        if name == "road":
            shapes = chosen[chosen.geom_type.isin(["LineString", "MultiLineString"])]
            shapes = shapes.buffer(5.0)
        else:
            shapes = chosen[chosen.geom_type.isin(["Polygon", "MultiPolygon"])].geometry
        if len(shapes) == 0:
            continue
        rasterize([(g, class_id) for g in shapes], out=labels, transform=transform)
        print(f"  {name:<10} {len(shapes):>6} shapes")
    return labels


def sample(labels, usable, per_class, rng):
    """Pick training pixel positions, shared by both depths."""
    rows, cols, classes = [], [], []
    for class_id, name in enumerate(CLASSES, start=1):
        r, c = np.nonzero((labels == class_id) & usable)
        if len(r) == 0:
            print(f"  {name:<10} none here, skipping")
            continue
        if len(r) > per_class:
            keep = rng.choice(len(r), per_class, replace=False)
            r, c = r[keep], c[keep]
        print(f"  {name:<10} {len(r):>6} pixels")
        rows.append(r)
        cols.append(c)
        classes.append(np.full(len(r), class_id, dtype=np.uint8))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(classes)


def predict_map(model, embeddings, scales, rows_at_a_time=256):
    """Classify every pixel, dequantising a block of rows at a time."""
    bands = embeddings.shape[0]
    height, width = scales.shape
    out = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, rows_at_a_time):
        band = slice(top, min(top + rows_at_a_time, height))
        x = (embeddings[:, band, :] * scales[band, :]).reshape(bands, -1).T
        usable = np.isfinite(scales[band, :]).reshape(-1)
        block = np.zeros(len(x), dtype=np.uint8)
        if usable.any():
            block[usable] = model.predict(x[usable])
        out[band] = block.reshape(-1, width)
    return out


def draw(panels, labels, bbox, year, out_path):
    """OSM labels and both predictions side by side, like step 2 does."""
    colours = ListedColormap(["#ffffff"] + [COLOURS[c] for c in CLASSES])
    style = dict(cmap=colours, vmin=0, vmax=len(CLASSES), interpolation="nearest")

    figure, axes = plt.subplots(1, 1 + len(panels), figsize=(16, 6),
                                layout="constrained")
    axes[0].imshow(labels, **style)
    axes[0].set_title("OSM labels")
    for axis, (title, image) in zip(axes[1:], panels):
        axis.imshow(image, **style)
        axis.set_title(title)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    west, south, east, north = bbox
    figure.suptitle(f"Tessera v2 {year} — 16 of 128 dimensions — "
                    f"{west:.2f},{south:.2f} to {east:.2f},{north:.2f}")
    figure.legend(handles=[Patch(facecolor=COLOURS[c], label=c) for c in CLASSES],
                  loc="outside lower center", ncol=len(CLASSES))
    figure.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lon", type=float, default=0.12, help="longitude (default: Cambridge)")
    parser.add_argument("--lat", type=float, default=52.20, help="latitude")
    parser.add_argument("--year", type=int, default=2024, help="2017 to 2025")
    parser.add_argument("--tiles", type=int, default=1, help="block size")
    parser.add_argument("--per-class", type=int, default=1000, help="training pixels per class")
    parser.add_argument("--out", default="matryoshka.png")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    cubes, scales, window, crs, bbox = read_window(
        args.lon, args.lat, args.year, args.tiles)
    labels = read_osm(bbox, crs, window)

    print("sampling training pixels")
    usable = np.isfinite(scales)
    r, c, y = sample(labels, usable, args.per_class, rng)
    fit_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)

    panels = []
    for depth, cube in sorted(cubes.items()):
        x = cube[:, r, c].T * scales[r, c][:, None]
        knn = KNeighborsClassifier(n_neighbors=5).fit(x[fit_idx], y[fit_idx])
        accuracy = accuracy_score(y[test_idx], knn.predict(x[test_idx]))
        print(f"d{depth}: {accuracy:.3f} correct on held-out pixels, "
              f"classifying every pixel...")
        panels.append((f"k-NN, {depth} dims ({accuracy:.3f})",
                       predict_map(knn, cube, scales)))

    draw(panels, labels, bbox, args.year, args.out)


if __name__ == "__main__":
    main()
