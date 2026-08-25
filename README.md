# GeoTessera examples

Tessera publishes a 128-dimensional embedding for every 10m pixel of
land, for every year since 2017.
These examples show how to read and use them with
[GeoTessera](https://github.com/ucam-eo/geotessera).

## Setup

Every script declares its dependencies in a [PEP 723](https://peps.python.org/pep-0723/)
header, so [`uv`](https://docs.astral.sh/uv/) is a handy tool.

## Quickstart

This is a very basic way to get started with some numbers.

```
uv run quickstart.py
```

This reads one pixel's embedding straight from the public store.  Pass
`--lon`/`--lat` for a place you know, and `--version "v2"` for the v2 beta
model.

## Where next

| You want to... | Go to |
|---|---|
| Understand the embeddings and how to work with them | [teaching/](teaching/) |
| Classify a region from your own labelled points | [classify/](classify/) |
| Detect a feature over a large region | [solarpanel/](solarpanel/) |
| Make a false-colour map | [pumap-viz/](pumap-viz/) |

[teaching/](teaching/) is a five-step guided tour. It starts by
classifying land cover around a point, and renders the result as
PNG and SVG. It then repeats the classification with plain `xarray`/`zarr` to show what
the geotessera library does for you. It then demonstrates the v2 store's 16-dimension
matryoshka prefixes for faster 'sketch' analyses.

[classify/](classify/) converts a GeoJSON of labelled points into a
classified GeoTIFF. It streams from the zarr store, or via downloaded
tiles for offline reuse. It has a helper to grab labels from OpenStreetMap.

[solarpanel/](solarpanel/) trains a solar-panel detector from a
handful of labelled points, then streams a 58km region through it.
This is a good pattern for inference over regions larger than memory.

[pumap-viz/](pumap-viz/) projects the 128-bands to RGB with a
parametric UMAP for a false-colour view of any region or country.
This is for artwork purposes!

## Concepts

- `GeoTesseraZarr` streams from the public zarr store and is the way most
  examples work. `GeoTessera` downloads tiles to disk first, for offline
  reuse.
- Embeddings stay on their native 10m UTM grid where possible to avoid
  skew. Classify first and then reproject only the result.
- Embedding releases sit side by side in the store, so every pipeline
  takes a `--version`-style flag and trialling a new model is a one-flag
  change.
