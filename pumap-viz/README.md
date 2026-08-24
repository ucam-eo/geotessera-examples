# False-colour UMAP visualisation

Projects the 128-band embeddings to three dimensions with a parametric
UMAP and writes an RGB GeoTIFF of the region on its native UTM grid.

```
uv run umap_visualization.py --bbox 0.05 52.15 0.20 52.25 --year 2024 \
    --output cambridge.tif --checkpoint-dir cache/
```

`--country "United Kingdom"` and `--region file.geojson` select larger
areas.  `--checkpoint-dir` caches the pixel sample, so a rerun skips the
download.  Training needs tensorflow, installed automatically on
Python 3.12.
