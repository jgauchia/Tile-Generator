# IceNav Tile-Generator

High-performance C++ toolset for generating custom binary map tiles (NAV format) from OpenStreetMap PBF files. Designed specifically for the [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Key Components

- **NAV Generator (C++)**: An industrial-grade engine that converts OSM PBF files into optimized binary tiles. Features multi-core processing, GEOS-based geometry merging/clipping, and hardware-aware simplification.
- **Tile Viewer (Python)**: A Pygame-based desktop application to preview and validate the generated NAV tiles on a PC.
- **features.json**: Centralized configuration for styling, zoom levels, and rendering priorities.

## Technologies Used

- **C++17**: Core generator implementation.
- **libosmium**: Fast OSM PBF parsing.
- **GEOS**: Robust geometry operations (merging, clipping, simplification).
- **nlohmann-json**: Configuration management.
- **Python 3.8+ & Pygame**: PC tile viewer.

---

## Getting Started

### 1. Requirements

Install the necessary development libraries (Ubuntu/Debian example):
```bash
sudo apt-get install libosmium2-dev libgeos-dev nlohmann-json3-dev libgdal-dev libboost-dev
```
For detailed requirements, see [DEPENDENCIES.md](DEPENDENCIES.md).

### 2. Building the Generator

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 3. Generating Tiles

```bash
./nav_generator <input.pbf> <output_dir> features.json [--zoom 6-17]
```

### 4. Viewing Tiles (PC)

Install Python dependencies:
```bash
pip install pygame
```
Run the viewer:
```bash
python tile_viewer.py <nav_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

---

## Development Standards

- **Binary Format**: The NAV format uses Delta Encoding with VarInt/ZigZag compression for maximum SD card efficiency.
- **Simplification**: 2000 points per feature limit to ensure stability on ESP32 streaming renderers.
- **Styling**: Absolute priority given to the `widths` table in `features.json` for precise visual control.

For more details on the binary spec, see [docs/bin_tile_format.md](docs/bin_tile_format.md).

---

## Author
**Jordi Gauchía** (jgauchia@jgauchia.com)
