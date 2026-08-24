# Solar Panel Detection with GeoTessera

This example demonstrates how to use [GeoTessera](https://github.com/ucam-eo/geotessera) embeddings for solar panel detection using a simple machine learning classifier.

## What is GeoTessera?

GeoTessera provides access to embeddings from the Tessera geospatial foundation model, which processes Sentinel-1 and Sentinel-2 satellite imagery to generate 128-channel representation maps at 10m resolution. These embeddings compress temporal-spectral features from a full year of satellite data into dense representations optimized for downstream analysis tasks like classification and detection.

## Overview

This example shows how geospatial embeddings can be used for solar panel detection with minimal labeled training data. The workflow demonstrates:

1. **Loading labeled training data** from GeoJSON files containing positive (solar panel) and negative (no solar panel) point locations
2. **Sampling embeddings** at labeled points, in one vectorised read from the public zarr store
3. **Training a classifier** with scikit-learn's logistic regression model
4. **Generating predictions** across the region, streamed in row strips and exported as a single GeoTIFF
5. **Visualizing results** in QGIS with both training data and prediction outputs

## Project Structure

```
solarpanel/
├── main.py                      # Main pipeline script
├── util.py                      # Helper functions
├── bbox.json                    # Bounding box defining region of interest
├── train_positive.geojson       # Training points with solar panels
├── train_negative.geojson       # Training points without solar panels
├── test_positive.geojson        # Test points with solar panels
├── test_negative.geojson        # Test points without solar panels
├── solarpanel.qgz               # QGIS project for visualization
└── output/                      # Prediction GeoTIFF (auto-created)
```

## How `main.py` Works

### 1. Load Training and Test Data

The script loads labeled point coordinates from GeoJSON files:
- **Training set**: Points labeled as having solar panels or not
- **Test set**: Separate points for validation

```python
train_positive = [(coord, True) for coord in load_fetch_collection('train_positive.geojson')]
train_negative = [(coord, False) for coord in load_fetch_collection('train_negative.geojson')]
```

### 2. Sample Embeddings at Points

One vectorised nearest-pixel read pulls the 128-dimensional embedding for every labeled point at once:

```python
sel = ds.sel(time=YEAR).sel(x=point_eastings, y=point_northings, method="nearest")
train_embeddings = sel["embeddings"].values.T * scales
```

Each point gets a 128-channel feature vector representing the spectral-temporal characteristics at that location.

### 3. Visualize Embeddings

A UMAP projection reduces the 128-dimensional embeddings to 2D for visualization, showing how well the embeddings separate solar panels from non-solar panels:

```python
visualize_embeddings(train_embeddings, train_labels,
                     output_path='train_embeddings_umap.png')
```

### 4. Train Classifier with Label Subset Analysis

The script analyzes how classification performance varies with different amounts of training data:

```python
label_subsets = [1, 5, 10, 20, 50, 100, len(train_embeddings) // 2]
train_with_label_subset(train_embeddings, train_labels,
                        test_embeddings, test_labels,
                        model, label_subsets, num_times=10)
```

This shows that embeddings can achieve good performance with relatively few labeled examples.

### 5. Generate Pixel-Level Predictions

The trained model is applied to every pixel in the region:

```python
# Slice the region from the open zone, a strip of rows at a time, so the
# dequantised float32 pixels never all exist at once
window = ds.sel(time=YEAR, x=slice(...), y=slice(...))
for top in range(0, height, STRIP_ROWS):
    strip = window.isel(y=slice(top, top + STRIP_ROWS))
    pixels = ds.tessera.dequantise(strip["embeddings"].values,
                                   strip["scales"].values)
    preds = model.predict(pixels.reshape(-1, 128))

    # Save as GeoTIFF with geospatial metadata
    with rasterio.open(output_file, "w", **metadata) as dest:
        dest.write(pred_image, 1)
```

Each output GeoTIFF contains a prediction map where pixel values indicate solar panel likelihood.

## Running the Example

To run the solar panel detection pipeline:

```bash
uv run main.py
```

This will:
1. Open the region's UTM zone in the public zarr store (nothing lands on disk)
2. Sample embeddings at training/test point locations
3. Train and evaluate the logistic regression classifier
4. Stream the region in strips and classify every pixel
5. Save the prediction to `output/prediction.tif`

**Expected output:**
- Training and test accuracy metrics
- Label subset performance analysis
- UMAP visualization saved as `train_embeddings_umap.png`
- A prediction GeoTIFF at `output/prediction.tif`

## Visualizing Results in QGIS

After running the pipeline, open the results in QGIS:

```bash
# Open the QGIS project file
open solarpanel.qgz  # macOS
# or
qgis solarpanel.qgz  # Linux
```

The QGIS project (`solarpanel.qgz`) includes:
- **Training data layers**: Positive and negative training points overlaid on imagery
- **Output prediction layer**: the GeoTIFF from `output/` showing predicted solar panel locations
- **Background imagery**: Satellite imagery for visual context

You can:
- Toggle layers on/off to compare training data with predictions
- Inspect individual pixels to see prediction confidence
- Overlay results on satellite imagery to validate detections
- Export maps or perform further spatial analysis

## Key Dependencies

- **geotessera**: Access to satellite imagery embeddings
- **scikit-learn**: Logistic regression classifier
- **rasterio**: GeoTIFF reading and writing
- **umap-learn**: Dimensionality reduction for visualization
- **matplotlib**: Plotting UMAP visualizations

All dependencies are managed automatically by `uv run`.

## Further Reading

- [GeoTessera documentation](../../geotessera/README.md)
- [Tessera foundation model](https://github.com/ucam-eo/tessera)
- [Sample embeddings API guide](../../geotessera/SAMPLE_EMBEDDINGS_USAGE.md)
