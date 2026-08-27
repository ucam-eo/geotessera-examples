#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "geotessera>=0.10.1",
#   "scikit-learn",
#   "umap-learn",
#   "matplotlib",
#   "pyproj",
#   "rasterio",
#   "numpy",
#   "shapely",
# ]
# ///
"""Solar panel detection from Tessera embeddings.

``sample_points`` fetches the training and test embeddings and
``iter_region`` streams the region in row strips, so the ~34M pixels
are classified without being in memory at once. Output is a single
prediction GeoTIFF on the native 10m UTM grid, plus a 64x64 embedding
patch around each of the largest detections via ``read_patch``, the
fixed-size window a second-stage model would consume.

Usage:
    uv run main.py [--data-dir /path/to/data] [--version v2]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression

from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url

# geotessera reports progress through the logging module; show its INFO lines
logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("geotessera").setLevel(logging.INFO)

YEAR = 2024
STRIP_ROWS = 256

# The training points fall inside the region streamed afterwards;
# cache_dir persists store metadata across runs and caches chunk data
# for the session, so the strip pass reuses chunks the point sampling
# already fetched.
CACHE_BYTES = 2 * 1024**3

parser = argparse.ArgumentParser(description='Solar panel detection using GeoTessera')
parser.add_argument('--data-dir', type=Path,
                    help='Directory containing data files (default: script directory)')
parser.add_argument('--version', default='v1',
                    help='embedding release: "v1" (default) or "v2"')
args = parser.parse_args()

data_dir = args.data_dir if args.data_dir else Path(__file__).parent

# Add data_dir to path so we can import util
sys.path.insert(0, str(data_dir))
from util import load_fetch_collection, train_with_label_subset, visualize_embeddings

bbox_file = data_dir / 'bbox.json'
if not bbox_file.exists():
    print(f"Error: bbox.json not found at {bbox_file}")
    sys.exit(1)
bounding_box = json.load(open(bbox_file))['bbox']

# load training and test sets
train_positive = [(a, True) for a in load_fetch_collection(str(data_dir / 'train_positive.geojson'))]
train_negative = [(a, False) for a in load_fetch_collection(str(data_dir / 'train_negative.geojson'))]
test_positive = [(a, True) for a in load_fetch_collection(str(data_dir / 'test_positive.geojson'))]
test_negative = [(a, False) for a in load_fetch_collection(str(data_dir / 'test_negative.geojson'))]

train = train_positive + train_negative
test = test_positive + test_negative

gt = GeoTesseraZarr(
    zarr_store_url(args.version),
    cache_dir=Path(__file__).parent / "tessera-cache",
    cache_max_size=CACHE_BYTES,
)
print(f"Store {gt.url} (cached)")

print(f"Sampling {len(train)} training and {len(test)} test points...")
train_embeddings = gt.sample_points([coord for coord, _ in train], YEAR)
test_embeddings = gt.sample_points([coord for coord, _ in test], YEAR)

train_labels = np.array([label for _, label in train], dtype=np.bool_)
test_labels = np.array([label for _, label in test], dtype=np.bool_)

# Drop points outside coverage (NaN embeddings)
train_valid = ~np.any(np.isnan(train_embeddings), axis=1)
test_valid = ~np.any(np.isnan(test_embeddings), axis=1)
if not np.all(train_valid):
    print(f"Warning: {np.sum(~train_valid)} training points outside coverage")
    train_embeddings, train_labels = train_embeddings[train_valid], train_labels[train_valid]
if not np.all(test_valid):
    print(f"Warning: {np.sum(~test_valid)} test points outside coverage")
    test_embeddings, test_labels = test_embeddings[test_valid], test_labels[test_valid]

print(f"Found {len(train_embeddings)} training points and {len(test_embeddings)} test points.")

visualize_embeddings(train_embeddings, train_labels, output_path=data_dir / 'train_embeddings_umap.png')

model = LogisticRegression(max_iter=1000)

# analyze performance with different label subsets
label_subsets = [1, 5, 10, 20, 50, 100, len(train_embeddings) // 2]
train_with_label_subset(train_embeddings, train_labels, test_embeddings, test_labels, model, label_subsets, num_times=10)

# final training on all data
model.fit(train_embeddings, train_labels)
print("Training accuracy:", model.score(train_embeddings, train_labels))
print("Test accuracy:", model.score(test_embeddings, test_labels))

# Classify the whole region, streamed in strips so the dequantised
# float32 pixels never all exist at once.
strips = gt.iter_region(bounding_box, YEAR, strip_rows=STRIP_ROWS)
first, transform, crs = next(strips)
width = first.shape[1]
print(f"\nClassifying the region in strips of {STRIP_ROWS} rows ({crs})...")


def classify_strip(block):
    pixels = block.reshape(-1, block.shape[2])
    valid = ~np.isnan(pixels).any(axis=1)
    proba = np.zeros(len(pixels), dtype=np.float32)
    if valid.any():
        proba[valid] = model.predict_proba(pixels[valid])[:, 1]
    preds = proba > 0.5
    # 0 = solar panel (black), 255 = not (white); nodata stays white
    image = (255 - preds * 255).astype(np.uint8).reshape(-1, width)
    return image, proba.reshape(-1, width)


output_dir = data_dir / "output"
output_dir.mkdir(exist_ok=True)
out_filename = output_dir / "prediction.tif"
first_image, first_proba = classify_strip(first)
images, probas = [first_image], [first_proba]
for block, _, _ in strips:
    image, proba = classify_strip(block)
    images.append(image)
    probas.append(proba)
    print(f"  {sum(i.shape[0] for i in images)} rows classified")
prediction = np.concatenate(images, axis=0)
probability = np.concatenate(probas, axis=0)

with rasterio.open(
    out_filename, "w", driver="GTiff", height=prediction.shape[0], width=width,
    count=1, dtype=rasterio.uint8, crs=crs, transform=transform, compress="lzw",
) as dest:
    dest.write(prediction, 1)

print(f"\n✅ Prediction saved to {out_filename}")

from pyproj import Transformer
from scipy import ndimage

detected = prediction == 0
to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

# Connected components of the mask; each footprint's confidence is the
# mean P(solar panel) over its pixels. rasterio's shapes() below traces
# the same 4-connected components, so a point inside a polygon indexes
# its component in `labelled`.
labelled, n_found = ndimage.label(detected)
component_confidence = (
    ndimage.mean(probability, labelled, range(1, n_found + 1)) if n_found else [])

# Vectorise the mask into polygon footprints. Tracing happens in the
# raster's UTM CRS so areas are square metres; only the output ring is
# reprojected to lon/lat.
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from shapely.ops import transform as reproject


def footprint_confidence(poly):
    inside = poly.representative_point()
    col, row = ~transform * (inside.x, inside.y)
    return float(component_confidence[labelled[int(row), int(col)] - 1])


MIN_AREA_M2 = 2000  # drop detections under 20 pixels
footprints = sorted(
    (p for p in (shape(g) for g, _ in
                 shapes(detected.astype(np.uint8), mask=detected, transform=transform))
     if p.area >= MIN_AREA_M2),
    key=lambda p: p.area, reverse=True)
with open(output_dir / "polygons.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": [
        {
            "type": "Feature",
            "geometry": mapping(reproject(to_ll.transform, poly)),
            "properties": {"area_m2": round(poly.area),
                           "area_ha": round(poly.area / 10_000, 2),
                           "confidence": round(footprint_confidence(poly), 3)},
        }
        for poly in footprints
    ]}, f)
print(f"Wrote {output_dir / 'polygons.geojson'}: {len(footprints)} footprints "
      f"of at least {MIN_AREA_M2} m², "
      f"{sum(p.area for p in footprints) / 10_000:,.0f} ha total")

# Second stage: extract an embedding patch around each of the largest
# detections, rescore its pixels with the trained model, and report the
# detections with their confidence.
if n_found:
    sizes = ndimage.sum_labels(np.ones_like(labelled), labelled, range(1, n_found + 1))
    top = np.argsort(sizes)[::-1][:5] + 1
    centres = ndimage.center_of_mass(detected, labelled, top)
    print(f"\nScoring the {len(top)} largest of {n_found} detections...")

    detections = []
    for i, (row, col) in enumerate(centres, start=1):
        east, north = transform * (col + 0.5, row + 0.5)
        lon, lat = to_ll.transform(east, north)
        patch, patch_transform, patch_crs = gt.read_patch(lon, lat, YEAR, 64)

        # Probability of solar panel for every pixel of the patch.
        pixels = patch.reshape(-1, patch.shape[2])
        valid = ~np.isnan(pixels).any(axis=1)
        proba = np.full(len(pixels), np.nan, dtype=np.float32)
        if valid.any():
            proba[valid] = model.predict_proba(pixels[valid])[:, 1]
        proba = proba.reshape(64, 64)
        score = float(np.nanmean(np.where(proba > 0.5, proba, np.nan)))

        patch_path = output_dir / f"patch_{i:02d}.tif"
        with rasterio.open(
            patch_path, "w", driver="GTiff", height=64, width=64,
            count=patch.shape[2], dtype="float32", crs=patch_crs,
            transform=patch_transform, compress="lzw",
        ) as dest:
            dest.write(patch.transpose(2, 0, 1))
        area_px = float(sizes[top[i - 1] - 1])
        detections.append((lon, lat, area_px, score, proba))
        print(f"  {area_px:5.0f} px at {lon:.4f},{lat:.4f}  "
              f"confidence {score:.2f}  -> {patch_path.name}")

    # A GeoJSON of the detections, for the QGIS project.
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"area_px": area, "confidence": round(score, 3)},
        }
        for lon, lat, area, score, _ in detections
    ]
    with open(output_dir / "detections.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)

    # A review card: each patch's probability map, for a human to check.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(detections), figsize=(3 * len(detections), 3.4),
                                layout="constrained")
    for axis, (lon, lat, area, score, proba) in zip(np.atleast_1d(axes), detections):
        shown = axis.imshow(proba, vmin=0, vmax=1, cmap="inferno")
        axis.set_title(f"{lon:.3f},{lat:.3f}\n{area:.0f} px, p={score:.2f}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(shown, ax=np.atleast_1d(axes).tolist(), shrink=0.8,
                    label="P(solar panel)")
    figure.savefig(output_dir / "detections.png", dpi=130)
    print(f"\nWrote {output_dir / 'detections.geojson'} and "
          f"{output_dir / 'detections.png'}")
