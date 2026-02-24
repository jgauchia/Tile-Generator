# Ocean Water Polygons

OpenStreetMap does not store ocean surfaces as polygons. Coastlines are tagged as `natural=coastline` linestrings, and it is up to consumers to derive water areas from them.

The [osmdata.openstreetmap.de](https://osmdata.openstreetmap.de/data/water-polygons.html) project pre-computes water polygons from OSM coastlines daily. The generator can load these shapefiles to render oceans and seas.

## Why

Without water polygons, coastal areas show no ocean.

The `--water-shp` option is **optional** -- inland users can skip it entirely to save space.

## Setup

Download the WGS84 shapefile (once, ~540 MB):
```bash
wget https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip
unzip water-polygons-split-4326.zip
```

This creates `water-polygons-split-4326/water_polygons.shp` and associated files.

## Usage

```bash
./nav_generator input.pbf output features.json --zoom 6-17 \
    --water-shp path/to/water-polygons-split-4326/water_polygons.shp
```

The generator reads the PBF bounding box and uses OGR spatial filtering to only load water polygons that intersect the extract area. A regional extract like Languedoc-Roussillon loads ~4 polygons, not the entire world.

## How it works

1. The PBF header provides the bounding box of the extract
2. GDAL/OGR opens the shapefile and applies a spatial filter (`SetSpatialFilterRect`)
3. Matching polygons are injected into the feature store as `layer="water"` with the standard water color (`#aad2df`)
4. The tile processor clips, simplifies, and renders them like any other water feature

## Dependencies

Requires GDAL (`libgdal-dev` on Debian/Ubuntu). Already included in the CMake build.

## Notes

- Use `water-polygons-split-4326` (WGS84), not the 3857 (Mercator) variant
- The shapefile is ~540 MB unzipped but is gitignored
- Lakes, rivers, and reservoirs come from the PBF data, not from this shapefile
- Island holes in ocean polygons are handled by the existing `GEOSDifference` logic in the tile processor
