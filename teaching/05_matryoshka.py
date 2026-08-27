# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geotessera>=0.10.1",
#     "matplotlib", "numpy", "osmnx>=2.0", "rasterio", "scikit-learn",
# ]
# ///
"""Step 5 of 5: classify with the 16-dimension matryoshka prefix.

The v2 model orders its 128 dimensions by importance, so a prefix of an
embedding is itself a usable embedding. The v2 store carries
``embeddings_d4`` and ``embeddings_d16`` prefix arrays, and
``GeoTesseraZarr`` reads them with ``depth=``: 16 dimensions arrive for
an eighth of the bytes of 128, dequantised as usual.

    uv run 05_matryoshka.py --lon 0.12 --lat 52.20

This script reads one window at depth 16 and in full, trains the same
k-NN on each, and draws the OSM labels and the two predictions side by
side to show what the reduced prefix costs in accuracy.

Depth arrays exist only in v2 stores; steps 1 to 4 use v1.
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url

# geotessera reports progress through the logging module; show its INFO lines
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("geotessera").setLevel(logging.INFO)

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


def read_osm(bbox, crs, transform, shape):
    """Draw OSM land cover onto exactly the same pixel grid as the window."""
    import osmnx as ox
    from rasterio.features import rasterize

    print("asking OpenStreetMap (this can take a minute)")
    west, south, east, north = bbox
    features = ox.features.features_from_bbox(bbox=(west, south, east, north), tags=OSM_TAGS)
    features = features.to_crs(crs)
    print(f"  {len(features)} features")

    labels = np.zeros(shape, dtype=np.uint8)
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


def sample(labels, valid, per_class, rng):
    """Pick training pixel positions, shared by both depths."""
    rows, cols, classes = [], [], []
    for class_id, name in enumerate(CLASSES, start=1):
        r, c = np.nonzero((labels == class_id) & valid)
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


def predict_map(model, mosaic, valid):
    """Classify every valid pixel, in batches to bound sklearn's memory."""
    height, width, bands = mosaic.shape
    flat = mosaic.reshape(-1, bands)
    out = np.zeros(height * width, dtype=np.uint8)
    idx = np.flatnonzero(valid.reshape(-1))
    for start in range(0, len(idx), 100_000):
        batch = idx[start : start + 100_000]
        out[batch] = model.predict(flat[batch])
    return out.reshape(height, width)


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
    figure.suptitle(f"Tessera v2 {year}: 16 of 128 dimensions, "
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
    gt = GeoTesseraZarr(
        zarr_store_url("v2"),
        cache_dir=Path(__file__).parent / "tessera-cache",
    )
    print(f"store {gt.url}, depths {sorted(gt.depths)}")

    west = math.floor(args.lon / TILE) * TILE
    south = math.floor(args.lat / TILE) * TILE
    bbox = (west, south, west + args.tiles * TILE, south + args.tiles * TILE)

    mosaics = {}
    for depth in (16, None):
        mosaics[depth], transform, crs = gt.read_region(bbox, args.year, depth=depth)
        label = depth or mosaics[depth].shape[2]
        print(f"  d{label}: {mosaics[depth].shape}, "
              f"{mosaics[depth].nbytes / 1e6:.0f} MB")

    valid = np.isfinite(mosaics[None][:, :, 0])
    labels = read_osm(bbox, crs, transform, valid.shape)

    print("sampling training pixels")
    r, c, y = sample(labels, valid, args.per_class, rng)
    fit_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)

    panels = []
    for depth, mosaic in sorted(mosaics.items(), key=lambda kv: kv[1].shape[2]):
        x = mosaic[r, c, :]
        knn = KNeighborsClassifier(n_neighbors=5).fit(x[fit_idx], y[fit_idx])
        accuracy = accuracy_score(y[test_idx], knn.predict(x[test_idx]))
        bands = mosaic.shape[2]
        print(f"d{bands}: {accuracy:.3f} correct on held-out pixels, "
              f"classifying every pixel...")
        panels.append((f"k-NN, {bands} dims ({accuracy:.3f})",
                       predict_map(knn, mosaic, valid)))

    draw(panels, labels, bbox, args.year, args.out)


if __name__ == "__main__":
    main()
