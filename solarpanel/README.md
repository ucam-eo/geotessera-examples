# Solar panel detection

Trains a solar-panel detector from labelled points and applies it to a
58km region, streamed from the public zarr store.

```
uv run main.py
```

This samples the training and test embeddings, trains and evaluates a
logistic regression, streams the region through the model in row
strips, and writes the prediction to `output/prediction.tif` on the
native 10m UTM grid. It then extracts a 64x64 embedding patch around
each of the five largest detections with `read_patch`, rescores the
patch pixels with the trained model, and writes three artefacts to
`output/`: the patches as `patch_NN.tif`, the detections with their
confidence as `detections.geojson` for the QGIS project, and a review
card of the probability maps as `detections.png`. It also writes
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
├── solarpanel.qgz               # QGIS project for visualisation
└── output/                      # Prediction GeoTIFF (auto-created)
```

## How it works

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

Each strip is written into a single-band GeoTIFF, where 0 marks a
predicted solar panel and 255 marks the rest.

## Viewing the result

The QGIS project `solarpanel.qgz` overlays the training points and the
prediction layer on satellite imagery:

```
open solarpanel.qgz   # macOS
qgis solarpanel.qgz   # Linux
```
