# Teaching: Tessera step by step

A five-step tour of the Tessera embeddings.  It starts the easy way —
the `geotessera` package does the store access — and ends the hard way,
reading the zarr store by hand, so the plumbing arrives only once you
have seen what it is for.

```
uv run 01_classify.py --lon 0.12 --lat 52.20   # writes prediction.zarr
uv run 02_render_png.py prediction.zarr        # writes prediction.png
uv run 03_render_svg.py prediction.zarr        # writes prediction.svg
uv run 04_classify_raw_zarr.py --lon 0.12 --lat 52.20
uv run 05_matryoshka.py --lon 0.12 --lat 52.20 # writes matryoshka.png
```

1. **`01_classify.py`** — reads a window of embeddings via
   `GeoTesseraZarr`, labels it from OpenStreetMap, and trains two
   classifiers; `--version` trials other embedding releases.
2. **`02_render_png.py`** — draws labels and predictions side by side.
3. **`03_render_svg.py`** — the same figure as a true vector SVG.
4. **`04_classify_raw_zarr.py`** — step 1 again with plain `xarray` and
   `zarr`, no `geotessera`: zone selection, chunk tuning, int8
   dequantisation and sentinel masking by hand.  More complicated, and
   the way in when you need what the wrapper does not offer.
5. **`05_matryoshka.py`** — the v2 model orders dimensions by
   importance, and its store carries `embeddings_d4`/`embeddings_d16`
   prefix arrays that cost a fraction of the bytes.  Classifies with 16
   dimensions and with all 128, side by side.

For the smallest possible use of the package, see
[`../quickstart.py`](../quickstart.py).
