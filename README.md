# GeoTessera examples

Tessera publishes a 128-number embedding for every 10 m pixel of land,
for every year since 2017.  These examples show how to read and use them
with [GeoTessera](https://github.com/ucam-eo/geotessera).

## Setup

Every script carries its dependencies in a
[PEP 723](https://peps.python.org/pep-0723/) header, so
[`uv`](https://docs.astral.sh/uv/) is the only tool you need.  The
scripts use a `geotessera` checkout in the sibling directory, so clone
the two repositories side by side:

```
git clone https://github.com/ucam-eo/geotessera
git clone https://github.com/ucam-eo/geotessera-examples
cd geotessera-examples
```

## Two minutes: read one embedding

```
uv run quickstart.py
```

reads one pixel's embedding straight from the public store — no account,
no download, nothing left on disk.  Pass `--lon`/`--lat` for a place you
know, and `--version "v2-2B-L~beta1"` for the v2 beta model.

## Where next

| You want to... | Go to |
|---|---|
| Understand the embeddings and how to work with them | [teaching/](teaching/) |
| Classify a region from your own labelled points | [classify/](classify/) |
| Detect a feature over a large region | [solarpanel/](solarpanel/) |
| Make a false-colour map | [pumap-viz/](pumap-viz/) |

**[teaching/](teaching/)** is a five-step guided tour, written to be read
as much as run.  It starts the easy way — classify land cover around a
point via the `geotessera` package, render the result as PNG and SVG —
then repeats the classification with plain `xarray`/`zarr` to show what
the library does for you, and closes with the v2 store's 16-dimension
matryoshka prefixes.

**[classify/](classify/)** turns a GeoJSON of labelled points into a
classified GeoTIFF — streaming from the zarr store, or via downloaded
tiles for offline reuse — with a helper that makes labels from
OpenStreetMap.

**[solarpanel/](solarpanel/)** trains a solar-panel detector from a
handful of labelled points, then streams a 58 km region through it: the
pattern for inference over regions larger than memory.

**[pumap-viz/](pumap-viz/)** projects the 128 bands to RGB with a
parametric UMAP for a false-colour view of any region or country.

## Habits the examples share

- `GeoTesseraZarr` streams from the public zarr store and is the way in
  for most work; `GeoTessera` downloads tiles to disk first, for offline
  reuse.
- Embeddings stay on their native 10 m UTM grid: classify first, and
  reproject only the result.
- Embedding releases sit side by side in the store, so every pipeline
  takes a `--version`-style flag and trialling a new model is a one-flag
  change.
