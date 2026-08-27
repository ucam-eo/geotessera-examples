#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "geotessera>=0.10.1",
# ]
# ///
"""Read one embedding from the public zarr store.

Tessera publishes a 128-dimensional embedding for every 10m pixel of
land, for every year since 2017. GeoTesseraZarr routes a query to the
correct UTM zone and returns the values dequantised as float32.

  uv run quickstart.py
  uv run quickstart.py --lon -5.93 --lat 54.60 --year 2025 --version v2

The teaching/ examples read the same store with plain xarray and zarr.
"""

import argparse

import numpy as np

from geotessera import GeoTesseraZarr
from geotessera.registry import zarr_store_url


def main():
    p = argparse.ArgumentParser(description="GeoTessera quickstart — read one pixel")
    p.add_argument("--lon", type=float, default=-5.926, help="Longitude (default: Belfast)")
    p.add_argument("--lat", type=float, default=54.597, help="Latitude (default: Belfast)")
    p.add_argument("--year", type=int, default=2025, help="Year (default: 2025)")
    p.add_argument("--version", default="v1",
                   help='embedding release: "v1" (default) or "v2"')
    args = p.parse_args()

    gt = GeoTesseraZarr(zarr_store_url(args.version))
    print(f"Store: {gt.url}")
    print(f"  years={gt.years}, model={gt.model_version}, bands={gt.n_bands}")

    # probe() tells open water and gaps in coverage apart from real data
    embedding, status = gt.probe(args.lon, args.lat, year=args.year)
    print(f"Point ({args.lon}, {args.lat}), year {args.year}: {status}")
    if embedding is None:
        return

    print(f"Norm: {np.linalg.norm(embedding):.4f}")
    print(f"\nFirst 16 bands (of {len(embedding)}):")
    for i in range(16):
        print(f"  band {i:3d}  {embedding[i]:10.6f}")


if __name__ == "__main__":
    main()
