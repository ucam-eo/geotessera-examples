Simple kNN classifier using GeoTESSERA numpy embeddings.

The simplest way to run is:

```
uv run fetch_osm.py --bbox -2.969398 53.434288 -2.960644 53.439196 -o liverpool.geojson
uv run classify.py --labels liverpool.geojson -o classified.tif
```
