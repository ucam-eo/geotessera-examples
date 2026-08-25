# /// script
# dependencies = [
#     "rasterio",
#     "numpy",
#     "zarr>=3.3",
#     "umap-learn",
#     "scikit-learn",
#     "tensorflow",
#     "joblib",
#     "geotessera"
# ]
# requires-python = ">=3.12,<3.13"
#
# [tool.uv.sources]
# geotessera = { path = "../../geotessera", editable = true }
# ///
"""
RGB visualisation of Tessera embeddings via parametric UMAP.

Streams the region from the public zarr store, trains a parametric UMAP
on a sample of pixels to project 128 dimensions to 3, and streams the
region again to write one RGB GeoTIFF on the native UTM grid.
Checkpoints the sample and the UMAP model for resuming.

Usage:
    uv run umap_visualization.py --bbox 0.0 52.0 0.3 52.3 --year 2024 --output output.tif --checkpoint-dir cache/
    uv run umap_visualization.py --region region.geojson --year 2024 --output output.tif
    uv run umap_visualization.py --country "United Kingdom" --year 2024 --output output.tif
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path
from typing import Tuple
import numpy as np
import rasterio
import rasterio.windows
import umap
from sklearn.preprocessing import StandardScaler
import warnings
import joblib

from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore

from geotessera import GeoTesseraZarr
from geotessera.store import DEFAULT_STORE, zarr_store
from geotessera.visualization import calculate_bbox_from_file
from geotessera.country import get_country_bbox

# The region is streamed twice, once to sample pixels for UMAP training
# and once to render.  A session cache serves the second pass from
# memory while the region fits within max_size.
CACHE_BYTES = 2 * 1024**3

warnings.filterwarnings("ignore")


def get_file_hash(file_path: Path) -> str:
    """Generate a hash for a file to detect changes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_checkpoint_metadata(checkpoint_dir: Path) -> dict:
    """Load checkpoint metadata if it exists."""
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint_metadata(checkpoint_dir: Path, metadata: dict):
    """Save checkpoint metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_embeddings_from_geotessera(
    gt: GeoTesseraZarr,
    bbox: Tuple[float, float, float, float],
    year: int,
    sample_rate: float,
    checkpoint_dir: Path = None,
):
    """Sample a fraction of the region's pixels, streamed strip by strip.

    Returns a ``(n_samples, 128)`` float32 array for UMAP training.
    """
    print(f"Sampling embeddings for bbox {bbox}, year {year}")
    input_hash = hashlib.md5(f"{bbox}:{year}:{sample_rate}".encode()).hexdigest()

    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        sampled_data_path = checkpoint_dir / "sampled_data.npy"
        if (
            sampled_data_path.exists()
            and metadata.get("sampling_complete")
            and metadata.get("input_hash") == input_hash
        ):
            sampled_data = np.load(sampled_data_path)
            print(f"Loaded cached sample: {sampled_data.shape}")
            return sampled_data

    all_data = []
    for block, _, _ in gt.iter_region(bbox, year, progress=True):
        pixels = block.reshape(-1, block.shape[2])
        valid = pixels[~np.isnan(pixels).any(axis=1)]
        n_samples = int(len(valid) * sample_rate)
        if n_samples:
            keep = np.random.choice(len(valid), size=n_samples, replace=False)
            all_data.append(valid[keep])

    if not all_data:
        raise ValueError(f"No valid pixels in bbox {bbox} for year {year}")
    combined_data = np.vstack(all_data)
    print(f"Sampled {len(combined_data)} pixels")

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_dir / "sampled_data.npy", combined_data)
        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata.update(
            sampling_complete=True,
            input_hash=input_hash,
            sample_rate=sample_rate,
            sampled_pixels=len(combined_data),
            bbox=bbox,
            year=year,
        )
        save_checkpoint_metadata(checkpoint_dir, metadata)

    return combined_data


def apply_umap_projection(
    data: np.ndarray,
    checkpoint_dir: Path = None,
    n_components: int = 3,
    random_state: int = 42,
):
    """Apply parametric UMAP dimensionality reduction to project data to RGB space."""

    # Check for cached UMAP model
    if checkpoint_dir:
        reducer_path = checkpoint_dir / "umap_reducer.pkl"
        scaler_path = checkpoint_dir / "scaler.pkl"
        embedding_path = checkpoint_dir / "embedding.npy"
        metadata = load_checkpoint_metadata(checkpoint_dir)

        if (
            reducer_path.exists()
            and scaler_path.exists()
            and embedding_path.exists()
            and metadata.get("umap_complete")
        ):
            print("Loading cached UMAP model and embedding...")
            reducer = joblib.load(reducer_path)
            scaler = joblib.load(scaler_path)
            embedding = np.load(embedding_path)
            print(f"Loaded UMAP embedding shape: {embedding.shape}")
            return embedding, reducer, scaler

    print(f"Applying parametric UMAP projection to {n_components} dimensions")

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply parametric UMAP with correct parameters
    reducer = umap.ParametricUMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        verbose=True,
        n_epochs=500,
    )

    embedding = reducer.fit_transform(data_scaled)

    print(f"Parametric UMAP embedding shape: {embedding.shape}")
    print(f"Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")

    # Save checkpoint.  Keras 3 cannot pickle the parametric model, so
    # the model cache is best-effort; the sample cache still saves a rerun
    # most of its time.
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, checkpoint_dir / "scaler.pkl")
        np.save(checkpoint_dir / "embedding.npy", embedding)
        try:
            joblib.dump(reducer, checkpoint_dir / "umap_reducer.pkl")
        except Exception as problem:
            print(f"Not caching the UMAP model ({type(problem).__name__}); "
                  "it will retrain on resume")

        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["umap_complete"] = True
        metadata["n_components"] = n_components
        metadata["random_state"] = random_state
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved UMAP model checkpoint to {checkpoint_dir}")

    return embedding, reducer, scaler


def normalize_to_rgb_global(embedding: np.ndarray, global_norm_params: list):
    """Normalize UMAP embedding to 0-255 RGB range using global parameters."""
    rgb_normalized = np.zeros_like(embedding)

    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        p_low, p_high = global_norm_params[i]

        # Clip and normalize using global parameters
        component_clipped = np.clip(component, p_low, p_high)

        if p_high > p_low:
            rgb_normalized[:, i] = (component_clipped - p_low) / (p_high - p_low)
        else:
            rgb_normalized[:, i] = 0.5

    # Apply a slight contrast enhancement to make colors more vivid
    rgb_enhanced = rgb_normalized
    rgb_enhanced = np.clip(rgb_enhanced * 1.2 - 0.1, 0, 1)  # Boost contrast slightly

    # Scale to 0-255 and convert to uint8
    rgb_255 = (rgb_enhanced * 255).astype(np.uint8)

    return rgb_255


def create_rgb_mosaic(
    gt: GeoTesseraZarr,
    bbox: Tuple[float, float, float, float],
    year: int,
    reducer,
    scaler,
    output_path: Path,
    checkpoint_dir: Path = None,
):
    """Project every pixel to RGB and write one native-UTM GeoTIFF.

    Streams the region once; global 2-98 percentile normalisation across
    all strips keeps colours consistent over the whole mosaic.
    """
    print("Projecting the region to RGB, strip by strip")
    strips = []
    crs = None
    for block, transform, crs in gt.iter_region(bbox, year, progress=True):
        h, w, c = block.shape
        pixels = block.reshape(-1, c)
        valid = ~np.isnan(pixels).any(axis=1)
        projected = None
        if valid.any():
            projected = reducer.transform(scaler.transform(pixels[valid]))
        strips.append((projected, valid, (h, w), transform))

    all_projected = np.vstack([p for p, *_ in strips if p is not None])
    global_norm_params = [
        (np.percentile(all_projected[:, i], 2), np.percentile(all_projected[:, i], 98))
        for i in range(3)
    ]
    print(f"Global normalisation from {len(all_projected)} pixels")

    height = sum(shape[0] for _, _, shape, _ in strips)
    width = strips[0][2][1]
    with rasterio.open(
        output_path, "w", driver="GTiff", height=height, width=width,
        count=3, dtype="uint8", crs=crs, transform=strips[0][3],
        compress="lzw", nodata=0,
    ) as dst:
        row = 0
        for projected, valid, (h, w), _ in strips:
            rgb = np.zeros((h * w, 3), dtype=np.uint8)
            if projected is not None:
                rgb[valid] = normalize_to_rgb_global(projected, global_norm_params)
            dst.write(
                rgb.reshape(h, w, 3).transpose(2, 0, 1),
                window=rasterio.windows.Window(0, row, w, h),
            )
            row += h

    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["mosaic_complete"] = True
        metadata["output_path"] = str(output_path)
        save_checkpoint_metadata(checkpoint_dir, metadata)

    print(f"RGB mosaic saved to {output_path} ({crs})")


def main():
    parser = argparse.ArgumentParser(
        description="Create RGB visualizations from GeoTessera embeddings using UMAP"
    )

    # Region specification (mutually exclusive)
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates (min_lon min_lat max_lon max_lat)",
    )
    region_group.add_argument(
        "--region",
        type=Path,
        help="Path to region file (GeoJSON, shapefile, etc.)",
    )
    region_group.add_argument(
        "--country",
        type=str,
        help="Country name (e.g., 'United Kingdom')",
    )

    # Required arguments
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for RGB visualization (e.g., output.tif)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year of embeddings to process (default: 2024)",
    )

    # Optional arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for storing UMAP checkpoints (delete to reset)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.05,
        help="Percentage of pixels to sample for UMAP training (default: 0.05)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear checkpoint cache before running",
    )

    args = parser.parse_args()

    # Determine bounding box from arguments
    if args.bbox:
        bbox = tuple(args.bbox)
        print(f"Using bounding box: {bbox}")
    elif args.region:
        if not args.region.exists():
            print(f"Error: Region file {args.region} does not exist")
            sys.exit(1)
        bbox = calculate_bbox_from_file(args.region)
        print(f"Calculated bbox from region file: {bbox}")
    elif args.country:
        bbox = get_country_bbox(args.country)
        print(f"Using bbox for {args.country}: {bbox}")
    else:
        print("Error: Must specify --bbox, --region, or --country")
        sys.exit(1)

    # Handle checkpoint directory
    if args.checkpoint_dir:
        if args.clear_cache and args.checkpoint_dir.exists():
            print(f"Clearing checkpoint directory: {args.checkpoint_dir}")
            import shutil
            shutil.rmtree(args.checkpoint_dir)

        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using checkpoint directory: {args.checkpoint_dir}")

        # Show current checkpoint status
        metadata = load_checkpoint_metadata(args.checkpoint_dir)
        if metadata:
            print("Checkpoint status:")
            if metadata.get("sampling_complete"):
                print(
                    f"  ✓ Sampling complete ({metadata.get('sampled_pixels', 0)} pixels)"
                )
            if metadata.get("umap_complete"):
                print("  ✓ UMAP training complete")
            if metadata.get("mosaic_complete"):
                print("  ✓ Previous mosaic complete")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Set random seed
        np.random.seed(args.random_seed)

        gt = GeoTesseraZarr(
            CacheStore(
                zarr_store(DEFAULT_STORE),
                cache_store=MemoryStore(),
                max_size=CACHE_BYTES,
            )
        )

        # Load and sample embedding data
        sampled_data = load_embeddings_from_geotessera(
            gt, bbox, args.year, args.sample_rate, args.checkpoint_dir
        )

        # Apply UMAP projection
        embedding, reducer, scaler = apply_umap_projection(
            sampled_data, args.checkpoint_dir, random_state=args.random_seed
        )

        # Create RGB mosaic
        create_rgb_mosaic(
            gt, bbox, args.year, reducer, scaler, args.output, args.checkpoint_dir
        )

        print(f"Successfully created RGB visualization: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
