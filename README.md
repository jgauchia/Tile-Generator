# IceNav Tile-Generator

Industrial-grade C++ toolset for generating optimized vector map tiles from OpenStreetMap PBF files. Specifically designed for the [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **High-Performance C++ Engine**: Optimized OSM PBF parsing and tile generation using GEOS and Osmium.
- **Noise Reduction (Culling)**: Automatically discards features too small for the zoom level (e.g., < 16px² or < 2px). Administrative boundaries are protected for full context.
- **Dynamic Simplification**: Uses zoom-dependent Douglas-Peucker tolerance for smooth curves at high zooms and efficient storage at low zooms.
- **Packed Binary Containers**: Tiles are consolidated into single `Zxx.nav` files per zoom level, eliminating SD card file system overhead and cluster waste.
- **Memory Optimized**: Uses POSIX `mmap` for feature storage, allowing country-scale processing with a low RAM footprint (~1GB for Catalonia).
- **Professional Aesthetics**:
    - **Multi-Level Boundaries**: Advanced two-pass extraction of international, regional, and municipal borders.
    - **Smart Filtering**: Automatic removal of tunnels and subway lines for superior urban map clarity.
    - **Dynamic Styles**: Total control over line widths and layer-based priorities via `features.json`.
- **ESP32 Ready**: Produces hardware-friendly data using VarInt/ZigZag delta encoding.
- **PC Simulator**: Pygame-based viewer with instant offset-based loading for rapid map validation.

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
python tile_viewer.py <output_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

---

## Technical Standards

- **Binary Format**: NPK1 (Container) + NAV1 (Tile) using Delta Encoding.
- **Precision**: 12-bit tile-relative coordinate space (0-4096).
- **Safety**: 2000 points per feature limit for ESP32 stability.
- **Style**: Manual Width Scaling and Layer-based priority system.

For detailed specifications, see the `docs/` directory.

---

## Author
**Jordi Gauchía** (jgauchia@jgauchia.com)
