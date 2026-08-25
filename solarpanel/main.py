#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "geotessera",
#   "scikit-learn",
#   "umap-learn",
#   "matplotlib",
#   "pyproj",
#   "rasterio",
#   "numpy",
#   "zarr>=3.3",
# ]
#
# [tool.uv.sources]
# geotessera = { path = "../../geotessera", editable = true }
# ///
"""Solar panel detection from Tessera embeddings.

``sample_points`` fetches the training and test embeddings and
``iter_region`` streams the region in row strips, so the ~34M pixels
are classified without being in memory at once. Output is a single
prediction GeoTIFF on the native 10m UTM grid.

Usage:
    uv run main.py [--data-dir /path/to/data]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression

from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore

from geotessera import GeoTesseraZarr
from geotessera.store import DEFAULT_STORE, zarr_store

YEAR = 2024
STRIP_ROWS = 256

# The training points fall inside the region streamed afterwards; a
# session cache lets the strip pass reuse chunks the point sampling
# already fetched.
CACHE_BYTES = 2 * 1024**3

parser = argparse.ArgumentParser(description='Solar panel detection using GeoTessera')
parser.add_argument('--data-dir', type=Path,
                    help='Directory containing data files (default: script directory)')
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
    CacheStore(
        zarr_store(DEFAULT_STORE), cache_store=MemoryStore(), max_size=CACHE_BYTES
    )
)
print(f"Store {DEFAULT_STORE} (cached)")

print(f"Sampling {len(train)} training and {len(test)} test points...")
train_embeddings = gt.sample_points([coord for coord, _ in train], YEAR, progress=False)
test_embeddings = gt.sample_points([coord for coord, _ in test], YEAR, progress=False)

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
    preds = np.zeros(len(pixels), dtype=np.uint8)
    if valid.any():
        preds[valid] = model.predict(pixels[valid])
    # 0 = solar panel (black), 255 = not (white); nodata stays white
    return (255 - preds * 255).astype(np.uint8).reshape(-1, width)


output_dir = data_dir / "output"
output_dir.mkdir(exist_ok=True)
out_filename = output_dir / "prediction.tif"
images = [classify_strip(first)]
for block, _, _ in strips:
    images.append(classify_strip(block))
    print(f"  {sum(i.shape[0] for i in images)} rows classified")
prediction = np.concatenate(images, axis=0)

with rasterio.open(
    out_filename, "w", driver="GTiff", height=prediction.shape[0], width=width,
    count=1, dtype=rasterio.uint8, crs=crs, transform=transform, compress="lzw",
) as dest:
    dest.write(prediction, 1)

print(f"\n✅ Prediction saved to {out_filename}")
