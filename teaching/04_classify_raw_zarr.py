# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "zarr>=3.0", "xarray>=2024.10", "dask", "numpy", "pyproj",
#     "fsspec", "aiohttp", "requests",   # so zarr can read an https:// store
#     "osmnx>=2.0", "geopandas", "shapely", "rasterio", "scikit-learn",
# ]
# ///
"""Step 4 of 5: the classification from step 1, with plain zarr and xarray.

Step 1 called ``GeoTesseraZarr``. This step performs the same work
directly: select the UTM zone, open its group, tune the chunking, slice
the window, dequantise, and mask the sentinels. It is the path to take
when the library does not offer what is needed, such as custom chunking
or keeping the window quantised to fit in memory.

    uv run 04_classify_raw_zarr.py --lon 0.12 --lat 52.20

It writes prediction.zarr, the same as step 1, so steps 2 and 3 render it.

Three properties of the store matter here:

1. The store holds one array per UTM zone, and zone 31 alone is 81TB.
   Slicing by coordinate fetches only the chunks the window overlaps, so
   a 22km square costs a few hundred MB.

2. Embeddings are quantised as int8 with one float scale per pixel, and
   the real value is embeddings * scales. Dequantising quadruples the
   size, so it is done a block of rows at a time.

3. scales carries two sentinels: NaN means water and +inf means never
   written. Both mean no usable embedding, so np.isfinite is the mask.
"""

import argparse
import json
import math
import time

import numpy as np
import xarray as xr

# Releases sit side by side under .../zarr/; --store swaps them by URL.
STORE = "https://data.source.coop/tessera/tessera/zarr/v1"
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


def read_window(lon, lat, year, tiles, store_url=STORE):
    """Read a tiles x tiles block of Tessera tiles around (lon, lat)."""
    from pyproj import Transformer

    # Snap the point down onto the tile grid, then take a block from there.
    west = math.floor(lon / TILE) * TILE
    south = math.floor(lat / TILE) * TILE
    east, north = west + tiles * TILE, south + tiles * TILE
    zone = int((lon + 180) / 6) % 60 + 1
    print(f"window {west:.2f},{south:.2f} to {east:.2f},{north:.2f} in UTM zone {zone}")

    # Ask for the data in 256x256 tiles.  This matters: left to itself the
    # reader tries to pull the whole window in a few huge HTTP requests, which
    # time out.  Small tiles mean many small requests, fetched in parallel.
    store = xr.open_zarr(store_url, group=f"utm{zone:02d}", zarr_format=3,
                         consolidated=True,
                         chunks={"time": 1, "band": 128, "y": 256, "x": 256})
    crs = store.attrs["proj:code"]  # e.g. EPSG:32631

    # The window is a lat/lon rectangle, so in metres it is a slight trapezium.
    # Project all four corners and take the extremes, not just two corners.
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [to_utm.transform(x, y) for x in (west, east) for y in (south, north)]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]

    window = store.sel(
        time=year,
        x=slice(min(xs), max(xs)),
        y=slice(max(ys), min(ys)),  # y runs north to south, so descending
    )
    print(f"  {window.sizes['y']} x {window.sizes['x']} pixels, downloading...")
    embeddings, scales = download(window)
    print(f"  got {embeddings.nbytes / 1e6:.0f} MB")
    return embeddings, scales, window, crs, (west, south, east, north)


def download(window, attempts=5):
    """Fetch the window, retrying if the server cuts a response short.

    Public data servers drop or truncate responses when busy, and a big read is
    made of hundreds of requests, so over a few minutes the chance of one going
    wrong is real.  The data never changes, so simply asking again is safe.
    """
    for attempt in range(1, attempts + 1):
        try:
            return window["embeddings"].values, window["scales"].values
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

    # Coordinates are pixel centres, so step back half a pixel to the corner.
    # Building this from the window's own coordinates is what guarantees the
    # labels line up with the embeddings.
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
    return labels, transform


def sample(embeddings, scales, labels, per_class, rng):
    """Collect training pixels.  Only these get dequantised, not the whole cube."""
    usable = np.isfinite(scales)
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

    r, c = np.concatenate(rows), np.concatenate(cols)
    x = embeddings[:, r, c].T * scales[r, c][:, None]  # dequantise, (n, 128)
    return x, np.concatenate(classes)


def predict_map(model, scaler, embeddings, scales, rows_at_a_time=256):
    """Classify every pixel, a block of rows at a time to bound memory."""
    height, width = scales.shape
    out = np.zeros((height, width), dtype=np.uint8)
    for top in range(0, height, rows_at_a_time):
        band = slice(top, min(top + rows_at_a_time, height))
        # Dequantise just these rows: (128, rows, width) -> (rows * width, 128)
        x = (embeddings[:, band, :] * scales[band, :]).reshape(128, -1).T
        usable = np.isfinite(scales[band, :]).reshape(-1)
        block = np.zeros(len(x), dtype=np.uint8)  # 0 = unlabelled: water, or no data
        if usable.any():
            good = x[usable]
            block[usable] = model.predict(
                good if scaler is None else scaler.transform(good))
        out[band] = block.reshape(-1, width)
    return out


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
    parser.add_argument("--store", default=STORE,
                        help="zarr store URL; swap the trailing v1 for another "
                             "release, e.g. v2-2B-L~beta1, to trial a newer model")
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    embeddings, scales, window, crs, bbox = read_window(
        args.lon, args.lat, args.year, args.tiles, args.store)
    labels, transform = read_osm(bbox, crs, window)

    print("sampling training pixels")
    x, y = sample(embeddings, scales, labels, args.per_class, rng)

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
        "prediction_knn": predict_map(knn, None, embeddings, scales),
        "prediction_logreg": predict_map(logreg, scaler, embeddings, scales),
    }

    result = xr.Dataset(
        {"osm_labels": (("y", "x"), labels),
         **{k: (("y", "x"), v) for k, v in maps.items()}},
        coords={"y": window.y.values, "x": window.x.values},
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
