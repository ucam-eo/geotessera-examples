# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geotessera>=0.10.1",
#     "numpy", "osmnx>=2.0", "rasterio", "scikit-learn", "xarray", "zarr",
# ]
# ///
"""Step 1 of 5: classify Tessera embeddings against OpenStreetMap labels.

Tessera publishes a 128-dimensional embedding for every 10m pixel of
land, for every year since 2017. This script reads a window of them
through the geotessera package, rasterises OpenStreetMap features over
the same grid as labels, and trains two classifiers to predict the
labels from the embeddings.

    uv run 01_classify.py --lon 0.12 --lat 52.20

It writes prediction.zarr. Steps 2 and 3 render it, and step 4 reads
the store behind it with plain zarr.

``GeoTesseraZarr.read_region`` takes a lon/lat box and returns float32
embeddings on their native UTM grid, with NaN where there is no data.
Zone routing, dequantisation and chunking happen in the library.
``--version`` selects the embedding release.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url

# geotessera reports progress through the logging module; show its INFO lines
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("geotessera").setLevel(logging.INFO)

TILE = 0.1  # Tessera tiles are a tenth of a degree square

# Class ids are the position in this list, plus one.  0 means "unlabelled".
CLASSES = ["cropland", "grassland", "woodland", "water", "built", "road"]

# One Overpass query covering every tag we care about; features are sorted into
# classes afterwards.  Six separate queries would mean six round trips.
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


def read_window(gt, lon, lat, year, tiles):
    """Read a tiles x tiles block of Tessera tiles around (lon, lat)."""
    import math

    # Snap the point down onto the tile grid, then take a block from there.
    west = math.floor(lon / TILE) * TILE
    south = math.floor(lat / TILE) * TILE
    east, north = west + tiles * TILE, south + tiles * TILE
    print(f"window {west:.2f},{south:.2f} to {east:.2f},{north:.2f}, downloading...")

    mosaic, transform, crs = gt.read_region((west, south, east, north), year)
    print(f"  {mosaic.shape[0]} x {mosaic.shape[1]} pixels in {crs}, "
          f"{mosaic.nbytes / 1e6:.0f} MB")
    return mosaic, transform, crs, (west, south, east, north)


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
            # Roads are lines; give them a 5 m half-width so they cover pixels.
            shapes = chosen[chosen.geom_type.isin(["LineString", "MultiLineString"])]
            shapes = shapes.buffer(5.0)
        else:
            shapes = chosen[chosen.geom_type.isin(["Polygon", "MultiPolygon"])].geometry
        if len(shapes) == 0:
            continue
        # Painting into one array in class order means later classes (roads,
        # buildings) win where they overlap earlier ones (fields, grass).
        rasterize([(g, class_id) for g in shapes], out=labels, transform=transform)
        print(f"  {name:<10} {len(shapes):>6} shapes")
    return labels


def sample(mosaic, labels, valid, per_class, rng):
    """Collect up to per_class training pixels for each class."""
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

    r, c = np.concatenate(rows), np.concatenate(cols)
    return mosaic[r, c, :], np.concatenate(classes)


def predict_map(model, scaler, mosaic, valid):
    """Classify every valid pixel, in batches to bound sklearn's memory."""
    height, width, bands = mosaic.shape
    flat = mosaic.reshape(-1, bands)
    out = np.zeros(height * width, dtype=np.uint8)  # 0 = unlabelled: no data
    idx = np.flatnonzero(valid.reshape(-1))
    for start in range(0, len(idx), 100_000):
        batch = idx[start : start + 100_000]
        x = flat[batch]
        out[batch] = model.predict(x if scaler is None else scaler.transform(x))
    return out.reshape(height, width)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lon", type=float, default=0.12, help="longitude (default: Cambridge)")
    parser.add_argument("--lat", type=float, default=52.20, help="latitude")
    parser.add_argument("--year", type=int, default=2024, help="2017 to 2025")
    parser.add_argument("--tiles", type=int, default=2, help="block size; 1 is much quicker")
    parser.add_argument("--per-class", type=int, default=1000, help="training pixels per class")
    parser.add_argument("--out", default="prediction.zarr")
    parser.add_argument("--version", default="v1",
                        help='embedding release: "v1" (default) or "v2"')
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    gt = GeoTesseraZarr(
        zarr_store_url(args.version),
        cache_dir=Path(__file__).parent / "tessera-cache",
    )
    print(f"store {gt.url}, model {gt.model_version}")

    mosaic, transform, crs, bbox = read_window(
        gt, args.lon, args.lat, args.year, args.tiles)
    valid = np.isfinite(mosaic[:, :, 0])
    labels = read_osm(bbox, crs, transform, mosaic.shape[:2])

    print("sampling training pixels")
    x, y = sample(mosaic, labels, valid, args.per_class, rng)

    x_fit, x_test, y_fit, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0, stratify=y)
    names = [CLASSES[i - 1] for i in np.unique(y)]
    print(f"training on {len(y_fit)} pixels, testing on {len(y_test)}")

    knn = KNeighborsClassifier(n_neighbors=5).fit(x_fit, y_fit)
    # Logistic regression wants comparably scaled inputs; k-NN does not care.
    scaler = StandardScaler().fit(x_fit)
    logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(x_fit), y_fit)

    accuracy = {}
    for label, model, prep in (("knn", knn, None), ("logreg", logreg, scaler)):
        guess = model.predict(x_test if prep is None else prep.transform(x_test))
        accuracy[label] = accuracy_score(y_test, guess)
        print(f"\n--- {label}: {accuracy[label]:.3f} correct on held-out pixels ---")
        print(classification_report(y_test, guess, target_names=names, zero_division=0))

    print("classifying every pixel")
    maps = {
        "prediction_knn": predict_map(knn, None, mosaic, valid),
        "prediction_logreg": predict_map(logreg, scaler, mosaic, valid),
    }

    height, width = mosaic.shape[:2]
    px = transform.a
    result = xr.Dataset(
        {"osm_labels": (("y", "x"), labels),
         **{k: (("y", "x"), v) for k, v in maps.items()}},
        coords={
            "x": transform.c + (np.arange(width) + 0.5) * px,
            "y": transform.f - (np.arange(height) + 0.5) * px,
        },
        attrs={
            "proj:code": crs,
            "spatial:transform": list(transform)[:6],
            "classes": json.dumps(CLASSES),
            "year": args.year,
            "bbox": list(bbox),
            "accuracy_knn": float(accuracy["knn"]),
            "accuracy_logreg": float(accuracy["logreg"]),
        },
    )
    result.to_zarr(args.out, mode="w", zarr_format=3, consolidated=True)
    print(f"\nwrote {args.out} — now run 02_render_png.py or 03_render_svg.py")


if __name__ == "__main__":
    main()
