# Solar panel detection

Trains a solar-panel detector from labelled points and applies it to a
58km region, streamed from the public zarr store. The
[teaching tour](../teaching/) is a useful prerequisite: step 1
introduces reading embeddings through `GeoTesseraZarr`, and step 4
explains the store format underneath.

```
uv run main.py
uv run main.py --version v2   # trial a different embedding release
```

To browse the detections on an OpenStreetMap base layer, serve this
directory and open [map.html](map.html), which picks up every GeoJSON
in `output/`:

```
python3 -m http.server
# then open http://localhost:8000/map.html
```

This samples the training and test embeddings, trains and evaluates a
logistic regression, streams the region through the model in row
strips, and writes the prediction to `output/prediction.tif` on the
native 10m UTM grid, with every detection above 2000 m² traced as a
polygon footprint into `output/polygons.geojson`. It then extracts a
64x64 embedding patch around each of the five largest detections with
`read_patch`, rescores the patch pixels with the trained model, and
writes the patches as `patch_NN.tif`, the scored detections as
`detections.geojson` for the QGIS project, and a review card of the
probability maps as `detections.png`. It also writes
`train_embeddings_umap.png`, a 2D UMAP of the training embeddings, and
prints how test accuracy varies with the number of labels.

## Data

```
solarpanel/
├── main.py                      # Pipeline script
├── util.py                      # Helper functions
├── bbox.json                    # Region of interest
├── train_positive.geojson       # Training points with solar panels
├── train_negative.geojson       # Training points without solar panels
├── test_positive.geojson        # Test points with solar panels
├── test_negative.geojson        # Test points without solar panels
├── map.html                     # MapLibre viewer for output/*.geojson
├── solarpanel.qgz               # QGIS project for visualisation
└── output/                      # Prediction, patches, detections (auto-created)
```

## How it works

The store is opened with `cache_dir=`, so chunks fetched while sampling
the training points are reused when the same region streams through the
classifier, and store metadata persists across runs.

The labelled points are sampled in one call, which issues a bulk read
per UTM zone:

```python
train_embeddings = gt.sample_points(train_points, YEAR)
```

A logistic regression is trained on the sampled embeddings. The label
subset analysis retrains on 2 to 400 labels and shows that test
accuracy reaches 0.98 with 20 labels and 0.997 with all of them.

The region is then classified strip by strip, so the dequantised
float32 pixels never all exist at once:

```python
for block, transform, crs in gt.iter_region(bbox, YEAR, strip_rows=256):
    preds = model.predict(block.reshape(-1, 128))
```

The classified strips concatenate into a single-band GeoTIFF, where 0
marks a predicted solar panel and 255 marks the rest.

Finally, the largest detected clusters become fixed-size patches, the
window a second-stage model would consume:

```python
patch, transform, crs = gt.read_patch(lon, lat, YEAR, 64)
proba = model.predict_proba(patch.reshape(-1, 128))[:, 1]
```

Each patch is rescored pixel by pixel, giving the detection a
confidence and a probability map for the review card.

Before that, the whole mask is vectorised with
`rasterio.features.shapes`: tracing happens on the UTM grid so areas
come out in square metres, and every connected detection of at least
2000 m² (20 pixels) lands in `output/polygons.geojson` with its area
and confidence attached — the strip pass keeps `predict_proba`
alongside the class mask, and a footprint's confidence is the mean
P(solar panel) over its pixels.

## Exercise: the runway problem

Browse the polygons over the satellite base layer in `map.html` and
you will spot that several "solar farms" are in fact runways — the
region has a cluster of airfields, and their large paved surfaces look
enough like panel arrays to this classifier. They tend to sit in the
lower confidence range, but not all of them do.

Improving this is left as an exercise. The model has never been shown
a runway. Drop a handful of points on the offending airfields into
`train_negative.geojson` (geojson.io or QGIS on top of `map.html`'s
satellite view makes this quick) and re-run `main.py`.
The label-subset analysis the run prints will show how few points
are needed to move the boundary.

## Viewing the result

The QGIS project `solarpanel.qgz` overlays the training points and the
prediction layer on satellite imagery, and `output/detections.geojson`
adds the scored detections:

```
open solarpanel.qgz   # macOS
qgis solarpanel.qgz   # Linux
```
