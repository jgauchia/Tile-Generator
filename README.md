# Tile-Generator (v0.5.0)

C++ toolset for generating optimized vector map tiles from OpenStreetMap PBF files, targeting ESP32-based GPS navigators.

## Features

- **High-Performance C++ Engine**: OSM PBF parsing and tile generation using GEOS, Osmium, and GDAL.
- **Four-Pass Rendering Pipeline**: Road casings, bridge decks, and layered text labels.
- **Smart Text Labels (GEOM_TEXT)**: Collision detection and population-based filtering for place names and road labels.
- **Packed Binary Containers (NPK2)**: Tiles consolidated into single `Zxx.nav` files per zoom level with Y-table index for O(1) row lookup.
- **Ocean Water Polygons**: Optional loading of pre-computed water polygons from [osmdata.openstreetmap.de](https://osmdata.openstreetmap.de) shapefiles, with spatial filtering on the PBF bounding box.
- **Memory Optimized**: POSIX `mmap` feature storage for country-scale processing with low RAM footprint.
- **Advanced Aesthetics**:
    - **Multi-Level Boundaries**: International, regional, and municipal borders.
    - **Smart Filtering**: Automatic removal of tunnels, subway lines, and underground polygons.
    - **Dynamic Styles**: Line widths (0.5px units) and 16-level priorities via `features.json`.
- **ESP32 Ready**: VarInt/ZigZag delta encoding and streaming-ready structures.
- **PC Simulator**: Pygame-based viewer with 4-pass rendering simulation.

---

## Getting Started

### 1. Requirements
Install the necessary libraries (Ubuntu/Debian):
```bash
sudo apt-get install libosmium2-dev libgeos-dev nlohmann-json3-dev libgdal-dev libboost-dev
```

### 2. Building the Generator
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Generating Maps
```bash
./nav_generator <input.pbf> <output_dir> features.json [--zoom 6-17] [--water-shp <path>]
```

### 4. Ocean Water Polygons (optional)

Download the pre-computed water polygons shapefile (once, ~540 MB):
```bash
wget https://osmdata.openstreetmap.de/download/water-polygons-split-4326.zip
unzip water-polygons-split-4326.zip
```

Generate with oceans:
```bash
./nav_generator input.pbf output features.json --zoom 6-17 \
    --water-shp water-polygons-split-4326/water_polygons.shp
```

The generator reads the PBF bounding box and only loads water polygons intersecting the extract area. See [WATER.md](WATER.md) for details.

### 5. Viewing Maps (PC)
```bash
python tile_viewer.py <output_dir> --lat <latitude> --lon <longitude> [--zoom <level>] --config features.json
```

---

## Processing Pipeline

```
1. Config loading (features.json)
2. Pass 1: RelationScanner (boundaries, water multipolygon relations)
3. Pass 2: PBF feature extraction (ways, areas → MappedStore)
4. Pass 3: Water polygon loading from shapefile (optional, --water-shp)
5. TileProcessor: clip, merge, simplify, emit for all zoom levels → NPK2 packs
```

---

## Technical Standards

- **Container Format**: NPK2 with Y-table index for O(1) row lookup + binary search within rows.
- **Tile Format**: NAV1, 13-byte feature header with casing flags and 0.5px width units.
- **Z-Order**: 16 priority levels (0-15) mapped to a 4-pass rendering pipeline.
- **Coordinate System**: Web Mercator, 12-bit tile-relative space (0-4096), delta VarInt/ZigZag encoding.

For detailed binary format specifications, see [`docs/bin_tile_format.md`](docs/bin_tile_format.md).

---

## Author
**Jordi Gauchía** (jgauchia@jgauchia.com)
