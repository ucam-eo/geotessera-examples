# Teaching: Tessera step by step

A five-step tour of the Tessera embeddings, written to be read as much
as run. It starts with the `geotessera` package doing the store access
and ends with the same work done against the zarr store directly.

```
uv run 01_classify.py --lon 0.12 --lat 52.20   # writes prediction.zarr
uv run 02_render_png.py prediction.zarr        # writes prediction.png
uv run 03_render_svg.py prediction.zarr        # writes prediction.svg
uv run 04_classify_raw_zarr.py --lon 0.12 --lat 52.20
uv run 05_matryoshka.py --lon 0.12 --lat 52.20 # writes matryoshka.png
```

1. [01_classify.py](01_classify.py) reads a window of embeddings via
   `GeoTesseraZarr`, labels it from OpenStreetMap, and trains two
   classifiers. `--version` selects the embedding release.
2. [02_render_png.py](02_render_png.py) draws the labels and both
   predictions side by side as a PNG.
3. [03_render_svg.py](03_render_svg.py) draws the same figure as a
   vector SVG.
4. [04_classify_raw_zarr.py](04_classify_raw_zarr.py) repeats step 1
   with plain `xarray` and `zarr`: zone selection, chunk tuning, int8
   dequantisation and sentinel masking by hand. This is the path to
   take when the library does not offer what you need.
5. [05_matryoshka.py](05_matryoshka.py) classifies with the v2 store's
   16-dimension `embeddings_d16` prefix and with all 128 dimensions,
   side by side. The prefix costs an eighth of the bytes.

The smallest possible use of the package is
[../quickstart.py](../quickstart.py).
