# /// script
# requires-python = ">=3.11"
# dependencies = ["zarr>=3.0", "xarray>=2024.10", "numpy", "matplotlib"]
# ///
"""Step 2 of 5 — draw the classification as a PNG.

    uv run 02_render_png.py prediction.zarr

Three panels side by side: what OpenStreetMap says is there, and what each of
the two classifiers thinks is there.  Reading them together is the point — the
places where the predictions agree with each other but disagree with OSM are
usually places where OSM is out of date or coarse, not where the model is wrong.

This is a picture of a grid of pixels, so a PNG is the honest format for it.
Step 3 makes an SVG instead, which is better for printing and editing but needs
a bit of work to avoid being enormous.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # no display needed; write straight to a file
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

COLOURS = {
    "cropland": "#d8b365",
    "grassland": "#a6d96a",
    "woodland": "#1a9641",
    "water": "#2c7fb8",
    "built": "#bdbdbd",
    "road": "#404040",
}

PANELS = [
    ("osm_labels", "OSM labels", None),
    ("prediction_knn", "k-NN", "accuracy_knn"),
    ("prediction_logreg", "logistic regression", "accuracy_logreg"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", nargs="?", default="prediction.zarr")
    parser.add_argument("-o", "--out", default="prediction.png")
    args = parser.parse_args()

    data = xr.open_zarr(args.store, zarr_format=3, consolidated=True).load()
    classes = json.loads(data.attrs["classes"])

    # Class ids are 1..n and 0 means unlabelled, so put white at the front and
    # the palette follows in the same order.  vmin/vmax pin ids to colours, so
    # a class missing from one panel still gets the same colour in the others.
    colours = ListedColormap(["#ffffff"] + [COLOURS[c] for c in classes])
    style = dict(cmap=colours, vmin=0, vmax=len(classes), interpolation="nearest")

    figure, axes = plt.subplots(1, 3, figsize=(16, 6), layout="constrained")
    for axis, (name, title, accuracy_key) in zip(axes, PANELS):
        if accuracy_key in data.attrs:
            title += f" ({data.attrs[accuracy_key]:.3f})"
        axis.imshow(data[name].values, **style)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    west, south, east, north = data.attrs["bbox"]
    figure.suptitle(f"Tessera {data.attrs['year']} — "
                    f"{west:.2f},{south:.2f} to {east:.2f},{north:.2f}")
    # "outside" stops constrained layout parking the legend on top of a panel.
    figure.legend(handles=[Patch(facecolor=COLOURS[c], label=c) for c in classes],
                  loc="outside lower center", ncol=len(classes))
    figure.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
