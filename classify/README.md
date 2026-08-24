# kNN classification from GeoJSON labels

Classify a region from labelled GeoJSON Points, via either interface:

```
uv run fetch_osm.py --bbox -2.969398 53.434288 -2.960644 53.439196 -o liverpool.geojson
uv run classify_zarr.py --labels liverpool.geojson -o classified.tif
```

- **`fetch_osm.py`** — labelled Points from OpenStreetMap;
  `liverpool.geojson` is a checked-in sample.
- **`classify_zarr.py`** — kNN classification straight from the zarr
  store; nothing lands on disk.
- **`classify.py`** — the same via downloaded tiles (`GeoTessera`), for
  embeddings cached on disk and offline reuse.

Both classifiers work on the native UTM grid with no resampling, and
take `--version` / `--dataset-version` to trial embedding releases
(`v1`, `v1.1`, `v2-2B-L~beta1`).
