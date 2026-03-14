#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
# ]
# ///
"""
Fetch labels from OpenStreetMap as GeoJSON.

Queries the Overpass API for five land-cover classes (urban, water, forest,
farmland, road) within a bounding box and saves the results as a GeoJSON
file with a "label" property on each Point feature.

For polygon features (buildings, lakes, forests), multiple sample points
are generated inside the polygon rather than just using the centroid.
For linear features (roads), points are sampled along the line.
We try to get 5 points per class and discard extra ones.

  uv run fetch_osm.py --bbox -2.9693 53.4342 -2.9606 53.4391 -o liverpool.geojson
"""

import argparse
import json
import random
import sys
import time

import requests

CLASSES = {
    "urban":    ['["building"]'],
    "water":    ['["natural"="water"]', '["water"="lake"]', '["water"="river"]',
                 '["water"="reservoir"]', '["water"="pond"]', '["waterway"="riverbank"]'],
    "forest":   ['["landuse"="forest"]', '["natural"="wood"]'],
    "farmland": ['["landuse"="farmland"]', '["landuse"="meadow"]'],
    "road":     ['["highway"="primary"]', '["highway"="secondary"]',
                 '["highway"="tertiary"]', '["highway"="residential"]'],
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# How many elements to fetch from Overpass per class (we then sample points
# from within their geometries, so we don't need many).
OVERPASS_ELEMENT_LIMIT = 30

# How many sample points to generate from each polygon/line feature.
SAMPLES_PER_FEATURE = 5

# Minimum distance (degrees) between sampled points (~11m at mid-latitudes).
MIN_POINT_SPACING = 0.0001


# ---------------------------------------------------------------------------
# Geometry helpers (no shapely dependency)
# ---------------------------------------------------------------------------

def _point_in_polygon(x, y, polygon):
    """Ray-casting point-in-polygon test. polygon is list of (x, y)."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _is_closed(coords):
    """Check if a coordinate ring is a closed polygon (first == last)."""
    if len(coords) < 4:
        return False
    return (abs(coords[0][0] - coords[-1][0]) < 1e-9 and
            abs(coords[0][1] - coords[-1][1]) < 1e-9)


def _far_enough(lon, lat, existing, min_dist=MIN_POINT_SPACING):
    """Check that (lon, lat) is at least min_dist from all existing points."""
    for elon, elat in existing:
        if abs(lon - elon) < min_dist and abs(lat - elat) < min_dist:
            return False
    return True


def sample_points_in_polygon(polygon, n, max_attempts=2000):
    """Sample up to n random non-overlapping points inside a polygon."""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    points = []
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        if _point_in_polygon(lon, lat, polygon) and _far_enough(lon, lat, points):
            points.append((lon, lat))
    return points


def sample_points_along_line(coords, n):
    """Sample up to n non-overlapping points along a polyline."""
    if not coords:
        return []
    if len(coords) <= n:
        return list(coords)
    indices = sorted(random.sample(range(len(coords)), min(n, len(coords))))
    points = []
    for i in indices:
        lon, lat = coords[i]
        if _far_enough(lon, lat, points):
            points.append((lon, lat))
    return points


# ---------------------------------------------------------------------------
# Extract sample points from Overpass elements
# ---------------------------------------------------------------------------

def _coords_from_geometry(geometry_list):
    """Convert Overpass geometry [{lat, lon}, ...] to [(lon, lat), ...]."""
    return [(n["lon"], n["lat"]) for n in geometry_list]


def extract_points(el):
    """Extract sample points from a single Overpass element.

    - node  → the single point
    - way   → sample inside (polygon) or along (line)
    - relation → sample inside the first outer ring
    """
    if el["type"] == "node":
        return [(el["lon"], el["lat"])]

    if el["type"] == "way" and "geometry" in el:
        coords = _coords_from_geometry(el["geometry"])
        if _is_closed(coords):
            return sample_points_in_polygon(coords, SAMPLES_PER_FEATURE)
        else:
            return sample_points_along_line(coords, SAMPLES_PER_FEATURE)

    if el["type"] == "relation" and "members" in el:
        # Find the first outer ring with enough geometry to sample from
        for member in el["members"]:
            if member.get("role") == "outer" and "geometry" in member:
                coords = _coords_from_geometry(member["geometry"])
                if len(coords) >= 3:
                    return sample_points_in_polygon(coords, SAMPLES_PER_FEATURE)

    return []


# ---------------------------------------------------------------------------
# Overpass queries
# ---------------------------------------------------------------------------

def fetch_osm_labels(bbox, max_per_class=5):
    """Fetch training points from OpenStreetMap via the Overpass API.

    For each class, queries the corresponding OSM tags within the bounding
    box. Polygon/line features are sampled to produce multiple points per
    feature. The result is then balanced to max_per_class points per class.

    Args:
        bbox:          (min_lon, min_lat, max_lon, max_lat)
        max_per_class: maximum number of points to keep per class

    Returns:
        list of (lon, lat, label) tuples
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # Overpass uses [south,west,north,east] ordering
    ovp_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    random.seed(42)
    all_points = []

    for class_name, tag_filters in CLASSES.items():
        print(f"  Querying OSM for '{class_name}'...", end=" ", flush=True)

        statements = []
        for tag_filter in tag_filters:
            statements.append(f"    node{tag_filter}({ovp_bbox});")
            statements.append(f"    way{tag_filter}({ovp_bbox});")
            statements.append(f"    relation{tag_filter}({ovp_bbox});")

        query = (
            "[out:json][timeout:60];\n"
            "(\n"
            + "\n".join(statements)
            + "\n);\n"
            f"out geom {OVERPASS_ELEMENT_LIMIT};\n"
        )

        # POST with retry on 429 or 504 (Overpass rate limiting or gateway timeous)
        backoff = 5
        for attempt in range(5):
            resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=120)
            if resp.status_code in (429, 504):
                reason = "rate limited" if resp.status_code == 429 else "gateway timeout"
                wait = int(resp.headers.get("Retry-After", backoff))
                print(f"{reason}, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
            break
        else:
            sys.exit(f"Overpass API rate limited after 5 retries, try again later.")

        # Extract sample points from each element's geometry
        points = []
        for el in resp.json().get("elements", []):
            for lon, lat in extract_points(el):
                points.append((lon, lat, class_name))

        # Balance: subsample to max_per_class
        if len(points) > max_per_class:
            points = random.sample(points, max_per_class)

        print(f"{len(points)} points")
        all_points.extend(points)

        # Pause between requests to avoid Overpass rate limiting
        time.sleep(5)

    return all_points


# ---------------------------------------------------------------------------
# GeoJSON output
# ---------------------------------------------------------------------------

def save_geojson(points, path):
    """Save labeled points as a GeoJSON FeatureCollection."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"label": label},
        }
        for lon, lat, label in points
    ]
    collection = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(collection, f, indent=2)
    print(f"Saved {len(features)} labeled points to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch labeled training points from OpenStreetMap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The output GeoJSON can be used directly with classify.py:\n"
            "  uv run classify.py --labels labels.geojson -o classified.tif"
        ),
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box for the OSM query",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output GeoJSON file path",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=5,
        help="Maximum points per class (default: 5)",
    )

    args = parser.parse_args()

    print("Fetching training points from OpenStreetMap...")
    points = fetch_osm_labels(args.bbox, args.max_per_class)
    if not points:
        sys.exit("No points found. Try a larger bounding box or a different region.")
    save_geojson(points, args.output)

    print(f"\nNext step — classify the region:\n"
          f"  uv run classify.py --labels {args.output} -o classified.tif")


if __name__ == "__main__":
    main()
