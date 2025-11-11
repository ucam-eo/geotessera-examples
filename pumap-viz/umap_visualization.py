# /// script
# dependencies = [
#     "rasterio",
#     "numpy>=1.24.0,<2.0",
#     "umap-learn",
#     "scikit-learn",
#     "tqdm",
#     "tensorflow-macos; sys_platform == 'darwin'",
#     "tensorflow-metal; sys_platform == 'darwin'",
#     "tensorflow; sys_platform != 'darwin'",
#     "joblib",
#     "geotessera>=0.7.0"
# ]
# requires-python = ">=3.11"
# ///
"""
Script for creating RGB visualizations from GeoTessera embeddings using UMAP.

This script:
1. Fetches GeoTessera embedding tiles for a specified region
2. Samples a percentage of pixels from the embeddings
3. Uses UMAP to project 128-dimensional embeddings to 3D RGB space
4. Normalizes values to 0-255 range
5. Creates a mosaic and outputs the final RGB visualization as GeoTIFF
6. Supports checkpointing for resuming interrupted runs

Usage:
    # Using bounding box
    uv run --with=. umap_visualization.py --bbox -180 -90 180 90 --year 2024 --output output.tif --checkpoint-dir cache/

    # Using region file
    uv run --with=. umap_visualization.py --region region.geojson --year 2024 --output output.tif

    # Using country name
    uv run --with=. umap_visualization.py --country "United Kingdom" --year 2024 --output output.tif
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import umap
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import joblib

from geotessera.core import GeoTessera
from geotessera.visualization import calculate_bbox_from_file
from geotessera.country import get_country_bbox

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
    gt: GeoTessera,
    bbox: Tuple[float, float, float, float],
    year: int,
    sample_rate: float,
    checkpoint_dir: Path = None,
    embeddings_dir: Optional[Path] = None,
):
    """Load GeoTessera embedding tiles and sample pixels for UMAP processing.

    Args:
        gt: GeoTessera instance
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
        year: Year of embeddings
        sample_rate: Percentage of pixels to sample (e.g., 0.05 = 5%)
        checkpoint_dir: Directory for caching intermediate results
        embeddings_dir: Directory where embeddings are stored

    Returns:
        Tuple of (sampled_data, tiles_list) where:
        - sampled_data: numpy array of sampled pixels, shape (n_samples, 128)
        - tiles_list: list of (year, lon, lat) tuples for all tiles in region
    """
    print(f"Loading GeoTessera embeddings for bbox {bbox}, year {year}")

    # Get list of tiles in the bounding box
    tiles_to_fetch = gt.registry.load_blocks_for_region(bbox, year)

    if not tiles_to_fetch:
        raise ValueError(f"No embedding tiles found in bbox {bbox} for year {year}")

    print(f"Found {len(tiles_to_fetch)} embedding tiles")

    # Check if we have cached sampled data
    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        sampled_data_path = checkpoint_dir / "sampled_data.npy"
        tiles_list_path = checkpoint_dir / "tiles_list.json"

        # Create a hash of input parameters
        input_hash = hashlib.md5(
            f"{bbox}:{year}:{sample_rate}:{len(tiles_to_fetch)}".encode()
        ).hexdigest()

        if (
            sampled_data_path.exists()
            and tiles_list_path.exists()
            and metadata.get("sampling_complete")
            and metadata.get("input_hash") == input_hash
            and metadata.get("sample_rate") == sample_rate
        ):
            print("Loading cached sampled data...")
            sampled_data = np.load(sampled_data_path)
            with open(tiles_list_path, "r") as f:
                tiles_list = [tuple(t) for t in json.load(f)]
            print(f"Loaded cached data: {sampled_data.shape}")
            return sampled_data, tiles_list

    all_data = []

    # Fetch and sample embeddings from each tile
    for year_val, tile_lon, tile_lat in tqdm(tiles_to_fetch, desc="Loading embeddings"):
        # Fetch embedding tile (already dequantized)
        embedding, crs, transform = gt.fetch_embedding(tile_lon, tile_lat, year_val)

        # embedding shape: (height, width, 128)
        height, width, channels = embedding.shape

        # Reshape to (height*width, 128)
        data_reshaped = embedding.reshape(-1, channels)

        # Remove NaN/invalid values
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)
        valid_data = data_reshaped[valid_mask]

        if len(valid_data) == 0:
            continue

        # Sample pixels
        n_samples = int(len(valid_data) * sample_rate)
        if n_samples > 0:
            indices = np.random.choice(
                len(valid_data), size=n_samples, replace=False
            )
            sampled_data = valid_data[indices]
            all_data.append(sampled_data)

    if not all_data:
        raise ValueError("No valid data found in embedding tiles")

    # Combine all sampled data
    combined_data = np.vstack(all_data)
    print(f"Sampled {len(combined_data)} pixels from {len(all_data)} tiles")
    print(f"Data shape: {combined_data.shape}")

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.save(checkpoint_dir / "sampled_data.npy", combined_data)
        with open(checkpoint_dir / "tiles_list.json", "w") as f:
            json.dump([[y, lon, lat] for y, lon, lat in tiles_to_fetch], f, indent=2)

        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["sampling_complete"] = True
        metadata["input_hash"] = hashlib.md5(
            f"{bbox}:{year}:{sample_rate}:{len(tiles_to_fetch)}".encode()
        ).hexdigest()
        metadata["sample_rate"] = sample_rate
        metadata["num_tiles"] = len(tiles_to_fetch)
        metadata["sampled_pixels"] = len(combined_data)
        metadata["bbox"] = bbox
        metadata["year"] = year
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved sampled data checkpoint to {checkpoint_dir}")

    return combined_data, tiles_to_fetch


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

    # Save checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, checkpoint_dir / "umap_reducer.pkl")
        joblib.dump(scaler, checkpoint_dir / "scaler.pkl")
        np.save(checkpoint_dir / "embedding.npy", embedding)

        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["umap_complete"] = True
        metadata["n_components"] = n_components
        metadata["random_state"] = random_state
        save_checkpoint_metadata(checkpoint_dir, metadata)
        print(f"Saved UMAP model checkpoint to {checkpoint_dir}")

    return embedding, reducer, scaler


def normalize_to_rgb(embedding: np.ndarray):
    """Normalize UMAP embedding to 0-255 RGB range."""
    # Use percentile-based normalization for better color distribution
    # This prevents extreme values from dominating the color mapping
    percentile_low = 2
    percentile_high = 98

    rgb_normalized = np.zeros_like(embedding)

    for i in range(embedding.shape[1]):
        component = embedding[:, i]
        # Use percentiles to clip extreme values
        p_low = np.percentile(component, percentile_low)
        p_high = np.percentile(component, percentile_high)

        # Clip and normalize
        component_clipped = np.clip(component, p_low, p_high)

        if p_high > p_low:
            rgb_normalized[:, i] = (component_clipped - p_low) / (p_high - p_low)
        else:
            rgb_normalized[:, i] = 0.5

    # Apply a slight contrast enhancement to make colors more vivid
    # Using a sigmoid-like transformation
    rgb_enhanced = rgb_normalized
    rgb_enhanced = np.clip(rgb_enhanced * 1.2 - 0.1, 0, 1)  # Boost contrast slightly

    # Scale to 0-255 and convert to uint8
    rgb_255 = (rgb_enhanced * 255).astype(np.uint8)

    # Print statistics for debugging
    print("RGB channel statistics after normalization:")
    print(
        f"  R: min={rgb_255[:, 0].min()}, max={rgb_255[:, 0].max()}, mean={rgb_255[:, 0].mean():.1f}"
    )
    print(
        f"  G: min={rgb_255[:, 1].min()}, max={rgb_255[:, 1].max()}, mean={rgb_255[:, 1].mean():.1f}"
    )
    print(
        f"  B: min={rgb_255[:, 2].min()}, max={rgb_255[:, 2].max()}, mean={rgb_255[:, 2].mean():.1f}"
    )

    return rgb_255


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
    gt: GeoTessera,
    tiles_to_fetch: List[Tuple[int, float, float]],
    reducer,
    scaler,
    output_path: Path,
    checkpoint_dir: Path = None,
):
    """Create RGB mosaic by applying UMAP projection to all embedding tiles.

    Args:
        gt: GeoTessera instance
        tiles_to_fetch: List of (year, lon, lat) tuples for tiles to process
        reducer: Trained UMAP reducer
        scaler: Trained StandardScaler
        output_path: Path to save output mosaic
        checkpoint_dir: Optional directory for caching intermediate results
    """
    print("Creating RGB mosaic from embedding tiles")

    rgb_datasets = []
    # Use Web Mercator (EPSG:3857) as the common CRS for merging
    # This ensures all tiles can be properly aligned and matches web mapping standards
    target_crs = rasterio.crs.CRS.from_epsg(3857)

    # Create RGB cache directory
    rgb_cache_dir = None
    if checkpoint_dir:
        rgb_cache_dir = checkpoint_dir / "rgb_tiles"
        rgb_cache_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all embeddings to compute global statistics
    print("First pass: collecting all embeddings for global normalization")
    all_embeddings = []
    tile_embeddings = []  # Store embeddings per tile for second pass
    tile_metadata = []    # Store metadata per tile

    for year, tile_lon, tile_lat in tqdm(tiles_to_fetch, desc="Collecting embeddings"):
        # Check if we have a cached RGB version
        rgb_cache_path = None
        if rgb_cache_dir:
            tile_key = f"{year}_{tile_lon:.2f}_{tile_lat:.2f}"
            rgb_cache_path = rgb_cache_dir / f"rgb_{tile_key}.tif"

            if rgb_cache_path.exists():
                rgb_datasets.append(rgb_cache_path)
                tile_embeddings.append(None)  # Skip embedding collection for cached tiles
                tile_metadata.append(None)
                continue

        # Fetch embedding from GeoTessera
        embedding, crs, transform = gt.fetch_embedding(tile_lon, tile_lat, year)

        # embedding shape: (height, width, 128)
        height, width, channels = embedding.shape

        # Reshape for processing
        data_reshaped = embedding.reshape(-1, channels)

        # Handle NaN values
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)

        tile_embedding = None
        if np.any(valid_mask):
            valid_data = data_reshaped[valid_mask]

            # Apply same preprocessing as training
            valid_scaled = scaler.transform(valid_data)

            # Apply UMAP transformation
            valid_embedding = reducer.transform(valid_scaled)

            # Store embedding for this tile
            tile_embedding = valid_embedding
            all_embeddings.append(valid_embedding)

        # Store tile data for second pass
        tile_embeddings.append(tile_embedding)
        tile_metadata.append({
            'data_shape': (height, width),
            'valid_mask': valid_mask,
            'year': year,
            'lon': tile_lon,
            'lat': tile_lat,
            'crs': crs,
            'transform': transform,
            'cache_path': rgb_cache_path
        })

    # Compute global normalization parameters from all embeddings
    if all_embeddings:
        print("Computing global normalization parameters")
        combined_embeddings = np.vstack(all_embeddings)

        # Use percentile-based normalization for better color distribution
        percentile_low = 2
        percentile_high = 98

        global_norm_params = []
        for i in range(combined_embeddings.shape[1]):
            component = combined_embeddings[:, i]
            p_low = np.percentile(component, percentile_low)
            p_high = np.percentile(component, percentile_high)
            global_norm_params.append((p_low, p_high))

        print(f"Global normalization parameters computed from {len(combined_embeddings)} pixels across {len(all_embeddings)} tiles")
    else:
        global_norm_params = [(0, 1)] * 3  # Default if no embeddings

    # Second pass: apply global normalization and create RGB tiles
    print("Second pass: applying global normalization and creating tiles")
    for tile_embedding, metadata in zip(tile_embeddings, tile_metadata):
        if tile_embedding is None or metadata is None:
            continue  # Skip cached tiles

        height, width = metadata['data_shape']
        valid_mask = metadata['valid_mask']
        src_crs = metadata['crs']
        src_transform = metadata['transform']
        rgb_cache_path = metadata['cache_path']

        # Apply global normalization to RGB
        rgb_data = np.zeros((len(valid_mask), 3), dtype=np.uint8)

        if tile_embedding is not None:
            # Apply global normalization
            valid_rgb = normalize_to_rgb_global(tile_embedding, global_norm_params)
            rgb_data[valid_mask] = valid_rgb

        # Reshape back to image format
        rgb_image = rgb_data.reshape(height, width, 3).transpose(2, 0, 1)

        # Calculate bounds from transform
        from affine import Affine
        if isinstance(src_transform, Affine):
            left = src_transform.c
            top = src_transform.f
            right = left + src_transform.a * width
            bottom = top + src_transform.e * height
            bounds = (left, bottom, right, top)
        else:
            # Fallback if transform is not Affine
            bounds = None

        # Calculate target transform for reprojection to Web Mercator
        if bounds:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, width, height, *bounds
            )
        else:
            # Fallback: use identity transform
            dst_transform = src_transform
            dst_width = width
            dst_height = height

        # Create reprojected RGB dataset
        rgb_profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 3,
            "dtype": "uint8",
            "crs": target_crs,
            "transform": dst_transform,
            "compress": "lzw",
            "nodata": 0,
        }

        # Write to cache or temp file
        if rgb_cache_path:
            output_rgb_path = rgb_cache_path
        else:
            tile_key = f"{metadata['year']}_{metadata['lon']:.2f}_{metadata['lat']:.2f}"
            output_rgb_path = output_path.parent / f"temp_rgb_{tile_key}.tif"

        # Write and reproject RGB image to Web Mercator
        with rasterio.open(output_rgb_path, "w", **rgb_profile) as dst:
            # Reproject each RGB band
            for i in range(3):
                if bounds:
                    reproject(
                        source=rgb_image[i],
                        destination=rasterio.band(dst, i + 1),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                    )
                else:
                    # Write directly if no reprojection needed
                    dst.write(rgb_image[i], i + 1)

        rgb_datasets.append(output_rgb_path)

    # Merge all RGB datasets
    print("Merging RGB datasets into final mosaic")

    src_files_to_mosaic = []
    for rgb_file in rgb_datasets:
        src = rasterio.open(rgb_file)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)

    # Update profile for output
    out_profile = src_files_to_mosaic[0].profile.copy()
    out_profile.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": 3,
            "dtype": "uint8",
        }
    )

    # Write final mosaic
    with rasterio.open(output_path, "w", **out_profile) as dest:
        dest.write(mosaic)

    # Clean up
    for src in src_files_to_mosaic:
        src.close()

    # Clean up temporary files (but not cached ones)
    if not rgb_cache_dir:
        for temp_file in rgb_datasets:
            if Path(temp_file).name.startswith("temp_rgb_"):
                Path(temp_file).unlink()

    # Update metadata
    if checkpoint_dir:
        metadata = load_checkpoint_metadata(checkpoint_dir)
        metadata["mosaic_complete"] = True
        metadata["output_path"] = str(output_path)
        save_checkpoint_metadata(checkpoint_dir, metadata)

    print(f"RGB mosaic saved to {output_path}")


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
        "--embeddings-dir",
        type=Path,
        default="embeddings/",
        help="Directory containing pre-downloaded embedding tiles (default: embeddings/)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching registry files",
    )
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

        # Initialize GeoTessera
        print("Initializing GeoTessera...")
        gt = GeoTessera(
            embeddings_dir=args.embeddings_dir,
            cache_dir=args.cache_dir,
        )

        # Load and sample embedding data
        sampled_data, tiles_to_fetch = load_embeddings_from_geotessera(
            gt, bbox, args.year, args.sample_rate, args.checkpoint_dir
        )

        # Apply UMAP projection
        embedding, reducer, scaler = apply_umap_projection(
            sampled_data, args.checkpoint_dir, random_state=args.random_seed
        )

        # Create RGB mosaic
        create_rgb_mosaic(
            gt, tiles_to_fetch, reducer, scaler, args.output, args.checkpoint_dir
        )

        print(f"Successfully created RGB visualization: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
