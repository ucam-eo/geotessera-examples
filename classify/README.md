# Classification from GeoJSON labels

Classifies a region from labelled GeoJSON Points, through either
interface:

```
uv run fetch_osm.py --bbox -2.969398 53.434288 -2.960644 53.439196 -o liverpool.geojson
uv run classify_zarr.py --labels liverpool.geojson -o classified.tif
```

[fetch_osm.py](fetch_osm.py) fetches labelled Points from OpenStreetMap;
`liverpool.geojson` is a checked-in sample. [classify_zarr.py](classify_zarr.py)
trains a kNN on the labelled points and classifies the region straight
from the zarr store, with nothing written to disk.
[classify.py](classify.py) is the same pipeline via downloaded tiles,
for offline reuse.

Both classifiers work on the native 10m UTM grid with no resampling,
and take `--version` or `--dataset-version` to select the embedding
release.
