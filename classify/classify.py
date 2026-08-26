#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "geotessera>=0.10.0",
#   "numpy",
#   "rasterio",
#   "scikit-learn",
# ]
# ///
"""
Pixel classification from locally downloaded tiles.

This is the tile-download variant of classify_zarr.py. Embeddings are
fetched as 0.1-degree tiles into embeddings/ and everything after that
works offline. Prefer classify_zarr.py unless the tiles are wanted on
disk for reuse.

Embeddings are never resampled. Training pixels are read directly from
the tiles, the tiles are placed on their shared UTM grid, and
classification happens on that grid. Reprojecting embeddings before
classification would blend neighbouring vectors into values the model
was never trained on; reproject the classified result instead.

Examples
--------

  uv run classify.py --labels liverpool.geojson -o classified.tif
  uv run classify.py --labels labels.geojson -o out.tif --k 7 --buffer 0.02

  # Trial a different embedding release
  uv run classify.py --labels labels.geojson -o out.tif \\
      --dataset-version v1.1

GeoJSON format: FeatureCollection of Points with "label" property.
Use fetch_osm.py to generate from OSM data.
"""

import argparse
import json
import sys

import numpy as np
import rasterio
from sklearn.neighbors import KNeighborsClassifier

from geotessera import GeoTessera

CLASS_COLORS = {
    "urban":    (255, 99, 71),
    "water":    (65, 105, 225),
    "forest":   (34, 139, 34),
    "farmland": (218, 165, 32),
    "road":     (169, 169, 169),
}
DEFAULT_COLOR = (128, 128, 128)


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return [
        (feat["geometry"]["coordinates"][0], feat["geometry"]["coordinates"][1], feat["properties"]["label"])
        for feat in data["features"]
        if feat["geometry"]["type"] == "Point" and feat["properties"].get("label")
    ]


def bbox_from_points(points, buffer=0.01):
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return (
        min(lons) - buffer,
        min(lats) - buffer,
        max(lons) + buffer,
        max(lats) + buffer,
    )


def native_mosaic(gt, bbox, year):
    """Place the region's tiles on their shared UTM grid — no resampling.

    Tiles within one UTM zone share a CRS, a 10 m pixel size, and pixel
    alignment, so a mosaic is pure array placement.  Returns
    (mosaic, transform, crs) with mosaic shaped (H, W, 128) float32 and
    NaN where no tile covers.
    """
    tiles = gt.registry.load_blocks_for_region(bbox, year)
    if not tiles:
        sys.exit(f"No tiles found for bbox {bbox} in year {year}")
    print(f"  {len(tiles)} tiles to fetch")

    loaded = list(gt.fetch_embeddings(tiles))
    crs_set = {str(crs) for _, _, _, _, crs, _ in loaded}
    if len(crs_set) > 1:
        sys.exit(
            f"Region spans UTM zones ({', '.join(sorted(crs_set))}); "
            "use classify_zarr.py, which handles zone routing."
        )
    crs = loaded[0][4]
    px = loaded[0][5].a  # pixel size in metres

    # Combined extent, then each tile's offset in whole pixels
    min_e = min(t.c for _, _, _, _, _, t in loaded)
    max_n = max(t.f for _, _, _, _, _, t in loaded)
    max_e = max(t.c + e.shape[1] * px for _, _, _, e, _, t in loaded)
    min_n = min(t.f - e.shape[0] * px for _, _, _, e, _, t in loaded)
    width = round((max_e - min_e) / px)
    height = round((max_n - min_n) / px)

    bands = loaded[0][3].shape[2]
    mosaic = np.full((height, width, bands), np.nan, dtype=np.float32)
    for _, _, _, emb, _, t in loaded:
        col = round((t.c - min_e) / px)
        row = round((max_n - t.f) / px)
        mosaic[row : row + emb.shape[0], col : col + emb.shape[1], :] = emb

    transform = rasterio.transform.Affine(px, 0, min_e, 0, -px, max_n)
    return mosaic, transform, crs


def classify(labels_path, output_path, year=2024, k=5, buffer=0.01,
             dataset_version="v1", dataset_variant=None):
    # Step 1: labels
    print("Step 1/5  Loading labels...")
    points = load_labels(labels_path)
    if not points:
        sys.exit("Error: no labeled Point features found in the GeoJSON file.")

    unique_labels = sorted(set(p[2] for p in points))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    print(f"  {len(points)} points across {len(unique_labels)} classes")

    # Step 2: sample embeddings at the labelled points (read straight from
    # the tiles — native values, no interpolation)
    print("Step 2/5  Sampling embeddings at labeled points...")
    gt = GeoTessera(dataset_version=dataset_version,
                    dataset_variant=dataset_variant,
                    embeddings_dir="embeddings")
    coords = [(lon, lat) for lon, lat, _ in points]
    y_train = np.array([label_to_id[lbl] for _, _, lbl in points])
    X_train = gt.sample_embeddings_at_points(coords, year=year)

    valid = ~np.isnan(X_train).any(axis=1)
    if not valid.all():
        print(f"  Dropping {int((~valid).sum())} points outside coverage")
        X_train, y_train = X_train[valid], y_train[valid]
    print(f"  Training samples: {len(X_train)}")

    # Step 3: train
    effective_k = min(k, len(X_train))
    print(f"Step 3/5  Training KNN classifier (k={effective_k})...")
    clf = KNeighborsClassifier(n_neighbors=effective_k, weights="distance")
    clf.fit(X_train, y_train)
    print(f"  Training accuracy: {clf.score(X_train, y_train):.1%}")

    # Step 4: fetch the region's tiles and classify on the native grid
    bbox = bbox_from_points(points, buffer=buffer)
    print(f"Step 4/5  Fetching tiles for region "
          f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]...")
    mosaic, transform, crs = native_mosaic(gt, bbox, year)
    height, width, n_bands = mosaic.shape
    print(f"  Mosaic: {height} x {width} pixels ({crs}), {n_bands} bands")

    pixels = mosaic.reshape(-1, n_bands)
    valid_mask = ~np.isnan(pixels).any(axis=1)
    predictions = np.full(len(pixels), -1, dtype=np.int8)
    valid_indices = np.where(valid_mask)[0]
    print(f"  Classifying {len(valid_indices):,} of {height * width:,} pixels...")
    for start in range(0, len(valid_indices), 50_000):
        batch = valid_indices[start : start + 50_000]
        predictions[batch] = clf.predict(pixels[batch])
    predictions = predictions.reshape(height, width)

    # Step 5: colour-mapped GeoTIFF, still in the tiles' own UTM
    print(f"Step 5/5  Writing {output_path}...")
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for label, class_id in label_to_id.items():
        rgb[predictions == class_id] = CLASS_COLORS.get(label, DEFAULT_COLOR)

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=3,
        dtype="uint8", crs=crs, transform=transform, compress="lzw",
    ) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)

    total_pixels = height * width
    print(f"\nDone! Classification written to: {output_path}")
    print("\nLegend:")
    for label in unique_labels:
        count = int((predictions == label_to_id[label]).sum())
        print(f"  {label:12s}  RGB{CLASS_COLORS.get(label, DEFAULT_COLOR)!s:20s}"
              f"  {100 * count / total_pixels:5.1f}%")
    nodata = int((predictions == -1).sum())
    if nodata:
        print(f"  {'nodata':12s}  {'':20s}  {100 * nodata / total_pixels:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Pixel classification using downloaded GeoTessera tiles and KNN")
    parser.add_argument("-l", "--labels", required=True,
                        help="GeoJSON file with labeled Point features")
    parser.add_argument("-o", "--output", required=True,
                        help="Output GeoTIFF file path")
    parser.add_argument("--year", type=int, default=2024,
                        help="Embedding year (default: 2024)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of neighbours for KNN (default: 5)")
    parser.add_argument("--buffer", type=float, default=0.01,
                        help="Buffer around training points in degrees (default: 0.01)")
    parser.add_argument("--dataset-version", default="v1",
                        help="Embedding release to use, e.g. v1 or v1.1 (default: v1)")
    parser.add_argument("--dataset-variant", default=None,
                        help="Model run within the release (default: the published one)")
    args = parser.parse_args()
    classify(args.labels, args.output, year=args.year, k=args.k,
             buffer=args.buffer, dataset_version=args.dataset_version,
             dataset_variant=args.dataset_variant)


if __name__ == "__main__":
    main()
