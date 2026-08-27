#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "geotessera>=0.10.1",
#   "numpy",
#   "rasterio",
#   "scikit-learn",
# ]
# ///
"""Pixel classification using GeoTesseraZarr.

Reads embeddings from the public zarr store on the native 10m UTM grid,
with zone routing and NaN masking in the library. Labels are GeoJSON
Points in lon/lat; ``fetch_osm.py`` makes them from OpenStreetMap and
``liverpool.geojson`` is a sample. Output is a colour-mapped GeoTIFF.

  uv run classify_zarr.py --labels liverpool.geojson -o classified.tif
  uv run classify_zarr.py --labels labels.geojson -o out.tif --year 2025 --k 7
  uv run classify_zarr.py --labels liverpool.geojson -o v2.tif --version v2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.neighbors import KNeighborsClassifier

from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url

# geotessera reports progress through the logging module; show its INFO lines
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("geotessera").setLevel(logging.INFO)

CLASS_COLORS = {
    "urban": (255, 99, 71), "water": (65, 105, 225), "forest": (34, 139, 34),
    "farmland": (218, 165, 32), "road": (169, 169, 169),
}


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return [
        (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"]["label"])
        for f in data["features"]
        if f["geometry"]["type"] == "Point" and f["properties"].get("label")
    ]


def classify(labels_path, output_path, year=2024, k=5, buffer=0.01, version="v1"):
    points = load_labels(labels_path)
    if not points:
        sys.exit("No labeled points found.")

    labels = sorted(set(p[2] for p in points))
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    print(f"{len(points)} points, {len(labels)} classes")

    gt = GeoTesseraZarr(
        zarr_store_url(version),
        cache_dir=Path(__file__).parent / "tessera-cache",
    )
    print(f"Store: {gt.url}")
    print(f"  years={gt.years}, model={gt.model_version}")

    # Sample training embeddings
    coords = [(x, y) for x, y, _ in points]
    y = np.array([l2i[l] for _, _, l in points])
    X = gt.sample_points(coords, year=year)

    valid = ~np.isnan(X).any(axis=1)
    if not valid.all():
        print(f"Dropping {(~valid).sum()} points outside coverage")
        X, y = X[valid], y[valid]

    # Train
    clf = KNeighborsClassifier(n_neighbors=min(k, len(X)), weights="distance")
    clf.fit(X, y)
    print(f"KNN: k={clf.n_neighbors}, accuracy={clf.score(X, y):.1%}")

    # Fetch region
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    bbox = (min(xs)-buffer, min(ys)-buffer, max(xs)+buffer, max(ys)+buffer)
    print(f"Fetching [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]...")
    mosaic, transform, out_crs = gt.read_region(bbox, year=year)
    h, w, nb = mosaic.shape
    print(f"  {h} x {w} px, {nb} bands")

    # Classify
    pixels = mosaic.reshape(-1, nb)
    mask = ~np.isnan(pixels).any(axis=1)
    preds = np.full(len(pixels), -1, dtype=np.int8)
    preds[mask] = clf.predict(pixels[mask])
    preds = preds.reshape(h, w)

    # Write
    print(f"Writing {output_path}...")
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for ci, l in i2l.items():
        rgb[preds == ci] = CLASS_COLORS.get(l, (128, 128, 128))

    with rasterio.open(output_path, "w", driver="GTiff", height=h, width=w,
                        count=3, dtype="uint8", crs=out_crs,
                        transform=transform, compress="lzw") as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)

    total = h * w
    print(f"\nDone: {output_path}")
    for l in labels:
        print(f"  {l:12s}  {100*(preds==l2i[l]).sum()/total:5.1f}%")


def main():
    p = argparse.ArgumentParser(description="Classify using GeoTesseraZarr")
    p.add_argument("-l", "--labels", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--buffer", type=float, default=0.01)
    p.add_argument("--version", default="v1",
                   help='embedding release: "v1" (default) or "v2"')
    args = p.parse_args()
    classify(args.labels, args.output, args.year, args.k, args.buffer, args.version)


if __name__ == "__main__":
    main()
