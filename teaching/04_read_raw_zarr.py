# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "geotessera>=0.10.0",
#     "numpy", "xarray", "zarr", "dask", "pyproj",
# ]
# ///
"""Step 4 of 5: read the store with plain zarr and xarray.

Steps 1 to 3 let the geotessera package do the store access. This step
reads the same window directly, as a reference for reading the store
from any tool, and ends by checking the result matches ``read_region``
exactly.

    uv run 04_read_raw_zarr.py --lon 0.12 --lat 52.20

Three properties of the store are useful to understand here:

1. The store holds one array per UTM zone, and zone 31 fully is 81TB.
   Slicing by coordinate fetches only the chunks the window overlaps.

2. Embeddings are quantised as int8 with one float32 scale per pixel;
   the real value must be dequantized via embeddings * scales.

3. scales carries two sentinels. NaN means water and +inf means never
   written. Both mean no usable embedding, so np.isfinite is the mask.
"""

import argparse
import math

import numpy as np
import xarray as xr

STORE = "https://data.source.coop/tessera/tessera/zarr/v1"
TILE = 0.1  # Tessera tiles are a tenth of a degree square


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lon", type=float, default=0.12, help="longitude (default: Cambridge)")
    parser.add_argument("--lat", type=float, default=52.20, help="latitude")
    parser.add_argument("--year", type=int, default=2024, help="2017 to 2025")
    args = parser.parse_args()

    # A UTM zone spans 6 degrees of longitude; its group is named utmNN.
    zone = int((args.lon + 180) / 6) % 60 + 1
    ds = xr.open_zarr(STORE, group=f"utm{zone:02d}", zarr_format=3,
                      consolidated=True,
                      chunks={"time": 1, "band": 128, "y": 4096, "x": 4096})
    crs = ds.attrs["proj:code"]
    print(f"zone utm{zone:02d}: {crs}, {ds.sizes['y']} x {ds.sizes['x']} pixels, "
          f"years {[int(t) for t in ds.time.values]}")
    print(f"  embeddings {ds['embeddings'].dtype}, scales {ds['scales'].dtype}")

    from pyproj import Transformer

    west = math.floor(args.lon / TILE) * TILE
    south = math.floor(args.lat / TILE) * TILE
    bbox = (west, south, west + TILE, south + TILE)
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = [to_utm.transform(x, y) for x in bbox[::2] for y in bbox[1::2]]
    es = [c[0] for c in corners]
    ns = [c[1] for c in corners]
    window = ds.sel(time=args.year, x=slice(min(es), max(es)),
                    y=slice(max(ns), min(ns)))
    print(f"window {window.sizes['y']} x {window.sizes['x']} pixels, downloading...")

    emb_int8 = window["embeddings"].values  # (128, H, W) int8
    scales = window["scales"].values        # (H, W) float32
    print(f"  {emb_int8.nbytes / 1e6:.0f} MB quantised")

    # Dequantise, masking the sentinels.
    usable = np.isfinite(scales)
    mosaic = emb_int8.astype(np.float32) * np.where(usable, scales, np.nan)
    mosaic = mosaic.transpose(1, 2, 0)
    print(f"  {usable.mean():.0%} usable, {np.isnan(scales).mean():.0%} water, "
          f"{np.isinf(scales).mean():.0%} unwritten")

    # The library must produce these values exactly.
    from geotessera import GeoTesseraZarr

    reference, _, _ = GeoTesseraZarr().read_region(bbox, args.year)
    same = np.array_equal(mosaic, reference, equal_nan=True)
    print(f"matches GeoTesseraZarr.read_region exactly: {same}")
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
