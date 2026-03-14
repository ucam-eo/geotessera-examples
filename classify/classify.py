#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geotessera",
#   "numpy",
#   "rasterio",
#   "scikit-learn",
# ]
# ///
"""
Pixel Classification Tutorial with GeoTessera

Classify spatial pixels using TESSERA's 128-dimensional embedding tiles
and a K-Nearest Neighbour classifier.

Reads a GeoJSON of labeled Point features (each with a "label" property)
(which you can generate using fetch_osm.py if you're feeling lazy),
trains a distance-weighted KNN on embeddings sampled at those
points, classifies every pixel in the surrounding region, and writes a
colour-mapped GeoTIFF.

Examples
--------

  # Classify using a GeoJSON of labeled points:
  uv run classify.py --labels labels.geojson -o classified.tif

  # With custom KNN parameters:
  uv run classify.py --labels labels.geojson -o classified.tif --k 7 --buffer 0.02

GeoJSON format
--------------

The input GeoJSON should be a FeatureCollection of Point features, each
with a "label" property naming its class:

  {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [0.12, 52.20]},
        "properties": {"label": "water"}
      },
      ...
    ]
  }

Use fetch_osm.py to generate this file from OpenStreetMap data.
"""

import argparse
import json
import sys

import numpy as np
import rasterio
from sklearn.neighbors import KNeighborsClassifier

from geotessera import GeoTessera

# ---------------------------------------------------------------------------
# Class definitions: name -> RGB colour
#
# These five land-cover classes work well as a starting point. You can
# add your own classes here — any label string found in the GeoJSON that
# is not listed gets a default grey colour.
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "urban":    (255, 99, 71),    # tomato red
    "water":    (65, 105, 225),   # royal blue
    "forest":   (34, 139, 34),    # forest green
    "farmland": (218, 165, 32),   # goldenrod
    "road":     (169, 169, 169),  # dark grey
}

DEFAULT_COLOR = (128, 128, 128)


def load_labels(geojson_path):
    """Load labeled points from a standard GeoJSON file.

    Returns a list of (lon, lat, label) tuples.
    """
    with open(geojson_path) as f:
        data = json.load(f)

    points = []
    for feature in data["features"]:
        geom = feature["geometry"]
        if geom["type"] != "Point":
            continue
        lon, lat = geom["coordinates"][:2]
        label = feature["properties"].get("label")
        if label is None:
            continue
        points.append((lon, lat, label))
    return points


def bbox_from_points(points, buffer=0.01):
    """Compute a bounding box from labeled points, adding a buffer in degrees."""
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return (
        min(lons) - buffer,
        min(lats) - buffer,
        max(lons) + buffer,
        max(lats) + buffer,
    )


def classify(labels_path, output_path, year=2024, k=5, buffer=0.01):
    """Run the full classification pipeline.

    Steps:
      1. Load labeled points from a GeoJSON file
      2. Sample GeoTessera embeddings at those point locations
      3. Train a distance-weighted KNN classifier
      4. Fetch a seamless mosaic of embeddings covering the region
      5. Classify every pixel in the mosaic
      6. Write a colour-mapped GeoTIFF
    """
    # -- Step 1: Load labels ------------------------------------------------
    print("Step 1/6  Loading labels...")
    labeled_points = load_labels(labels_path)
    if not labeled_points:
        sys.exit("Error: no labeled Point features found in the GeoJSON file.")

    # Build label <-> numeric id mappings
    unique_labels = sorted(set(p[2] for p in labeled_points))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print(f"  Found {len(labeled_points)} points across {len(unique_labels)} classes:")
    for label in unique_labels:
        n = sum(1 for p in labeled_points if p[2] == label)
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        print(f"    {label:12s}  {n:4d} points  RGB{color}")

    # -- Step 2: Sample TESSERA embeddings at label locations -----------------------
    print("Step 2/6  Sampling embeddings at labeled points...")
    gt = GeoTessera()
    coords = [(lon, lat) for lon, lat, _ in labeled_points]
    labels_numeric = np.array([label_to_id[lbl] for _, _, lbl in labeled_points])

    embeddings = gt.sample_embeddings_at_points(coords, year=year)
    X_train = embeddings
    y_train = labels_numeric
    print(f"  Training samples: {len(X_train)}")

    # -- Step 3: Train KNN classifier --------------------------------------
    effective_k = min(k, len(X_train))
    print(f"Step 3/6  Training KNN classifier (k={effective_k})...")
    clf = KNeighborsClassifier(n_neighbors=effective_k, weights="distance")
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    print(f"  Training accuracy: {train_acc:.1%}")

    # -- Step 4: Fetch embedding mosaic for the region ----------------------
    bbox = bbox_from_points(labeled_points, buffer=buffer)
    print(f"Step 4/6  Fetching mosaic for region "
          f"[{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]...")
    mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=year)
    height, width, n_bands = mosaic.shape
    total_pixels = height * width
    print(f"  Mosaic: {height} x {width} pixels, {n_bands} bands")

    # -- Step 5: Classify every pixel ---------------------------------------
    print(f"Step 5/6  Classifying {total_pixels:,} pixels...")
    pixels = mosaic.reshape(-1, n_bands)

    # Build a mask for valid (non-NaN) pixels
    valid_mask = ~np.isnan(pixels).any(axis=1)

    # Initialise with -1 (nodata)
    predictions = np.full(len(pixels), -1, dtype=np.int8)
    valid_indices = np.where(valid_mask)[0]

    # Classify in batches to keep memory manageable
    batch_size = 50_000
    for start in range(0, len(valid_indices), batch_size):
        batch_idx = valid_indices[start : start + batch_size]
        predictions[batch_idx] = clf.predict(pixels[batch_idx])

    predictions = predictions.reshape(height, width)
    n_classified = int((predictions >= 0).sum())
    print(f"  Classified {n_classified:,} / {total_pixels:,} pixels "
          f"({100 * n_classified / total_pixels:.1f}%)")

    # -- Step 6: Write colour-mapped GeoTIFF --------------------------------
    print(f"Step 6/6  Writing {output_path}...")

    # Map each class id to its RGB colour
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, label in id_to_label.items():
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        mask = predictions == class_id
        rgb[mask] = color

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)

    # -- Print summary / legend ---------------------------------------------
    print("\nDone! Classification written to:", output_path)
    print("\nLegend:")
    for label in unique_labels:
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        count = int((predictions == label_to_id[label]).sum())
        pct = 100 * count / total_pixels
        print(f"  {label:12s}  RGB{color!s:20s}  {pct:5.1f}%")
    nodata_count = int((predictions == -1).sum())
    if nodata_count > 0:
        pct = 100 * nodata_count / total_pixels
        print(f"  {'nodata':12s}  RGB{'(0, 0, 0)':20s}  {pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Pixel classification using GeoTessera embeddings and KNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-l", "--labels", required=True,
        help="GeoJSON file with labeled Point features",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output GeoTIFF file path",
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Embedding year (default: 2024)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of neighbours for KNN (default: 5)",
    )
    parser.add_argument(
        "--buffer", type=float, default=0.01,
        help="Buffer around training points in degrees (default: 0.01)",
    )

    args = parser.parse_args()
    classify(
        labels_path=args.labels,
        output_path=args.output,
        year=args.year,
        k=args.k,
        buffer=args.buffer,
    )


if __name__ == "__main__":
    main()
