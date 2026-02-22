# IceNav Tile-Generator (v0.4.0)

Industrial-grade C++ toolset for generating optimized vector map tiles from OpenStreetMap PBF files. Specifically designed for the [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **High-Performance C++ Engine**: Optimized OSM PBF parsing and tile generation using GEOS and Osmium.
- **Four-Pass Rendering Pipeline**: Professional map aesthetics with road casings (borders), bridge decks, and layered text labels.
- **Smart Text Labels (GEOM_TEXT)**: Collision detection and population-based filtering for place names and road labels.
- **Packed Binary Containers (NPK1)**: Tiles are consolidated into single `Zxx.nav` files per zoom level, eliminating SD card file system overhead and cluster waste.
- **Casing & Bridge Support**: Automatic generation of bridge underlays and casing flags for two-pass line rendering.
- **Memory Optimized**: Uses POSIX `mmap` for feature storage, allowing country-scale processing with a low RAM footprint (~1GB for Catalonia).
- **Advanced Aesthetics**:
    - **Multi-Level Boundaries**: Two-pass extraction of international, regional, and municipal borders.
    - **Smart Filtering**: Automatic removal of tunnels and subway lines for superior urban map clarity.
    - **Dynamic Styles**: Total control over line widths (0.5px units) and 16-level priorities via `features.json`.
- **ESP32 Ready**: Produces hardware-friendly data using VarInt/ZigZag delta encoding and streaming-ready structures.
- **PC Simulator**: Pygame-based viewer with 4-pass rendering simulation and instant offset-based loading.

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
Generates packed containers in the output directory:
```bash
./nav_generator <input.pbf> <output_dir> features.json [--zoom 6-17]
```

### 4. Viewing Maps (PC)
```bash
python tile_viewer.py <output_dir> --lat <latitude> --lon <longitude> [--zoom <level>] --config features.json
```

---

## Technical Standards

- **Binary Format**: NPK1 (Container) + NAV1 (Tile) using Delta Encoding.
- **Header v0.4**: 13-byte feature header with casing flags and 0.5px width units.
- **Z-Order**: 16 priority levels (0-15) mapped to a 4-pass rendering pipeline.
- **Safety**: 2000 points per feature limit for ESP32 stability.

For detailed specifications, see the `docs/` directory.

---

## Author
**Jordi Gauchía** (jgauchia@jgauchia.com)
