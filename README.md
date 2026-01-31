# OSM Tile Generator for IceNav

Converts OpenStreetMap PBF files to NAV tiles for [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **Direct PBF processing** - No intermediate formats (GOL, Docker, etc.)
- **Tile-based structure** - Standard z/x/y tile layout
- **Pre-calculated projection** - Mercator projection done on PC for zero-CPU rendering on ESP32
- **Optimized Storage** - int16 relative coordinates (~50% smaller than previous versions)
- **BBox-based Culling** - 4-byte object bounding box for ultra-fast visibility checks
- **Seamless borders** - Features stored with safety margin to avoid edge artifacts
- **Feature filtering** - Configurable via `features.json`

## Requirements

- Python 3.8+
- Virtual environment (recommended)

## Installation

```bash
# Clone repository
git clone https://github.com/jgauchia/Tile-Generator.git
cd Tile-Generator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install shapely pygame osmium
```

---

## Generate NAV Tiles

```bash
python tile_generator.py <input.pbf> <output_dir> features.json [--zoom 6-17]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `input.pbf` | OpenStreetMap PBF file | Required |
| `output_dir` | Output directory | Required |
| `features.json` | Feature configuration | Required |
| `--zoom` | Zoom range (e.g., `6-17`, `10-14`, `12`) | `6-17` |

---

## View NAV Tiles

```bash
python tile_viewer.py <nav_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

**Viewer Controls:**

| Key | Action |
|-----|--------|
| Arrow keys | Pan map |
| Mouse drag | Pan map |
| `[` / `]` or Mouse wheel | Zoom in/out |
| `B` | Toggle background (white/black) |
| `F` | Toggle polygon fill |
| `G` | Toggle tile grid |
| `Right Click` | Identify feature info |
| `Q` / `ESC` | Quit |

---

## NAV Binary Format Specification

NAV is a proprietary binary format designed as a lightweight alternative to FlatGeobuf for embedded devices with limited resources. Optimized for ESP32's SD card access and low memory usage.

**Key Features:**
- Pre-projected Mercator coordinates relative to tile (0-4096 range).
- int16 coordinates for minimal memory footprint.
- Sequential structure for streaming reads.

**File Header (22 bytes):**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | Magic | `NAV1` (0x4E, 0x41, 0x56, 0x31) |
| 4 | 2 | Feature count | Number of features (little-endian) |
| 6 | 16 | Tile BBox | lon_min, lat_min, lon_max, lat_max (4 x int32 scaled 1e7) |

**Feature Record:**

| Size | Field | Description |
|------|-------|-------------|
| 1 | Geometry type | 1=Point, 2=LineString, 3=Polygon |
| 2 | Color | RGB565 color (little-endian) |
| 1 | Zoom/Priority | High nibble = min_zoom, low nibble = priority |
| 1 | Width | Line width in pixels (1-15) |
| 4 | Object BBox | Normalized relative BBox [x1, y1, x2, y2] (4 x uint8) |
| 2 | Coord count | Number of coordinates (little-endian) |
| 4×N | Coordinates | x(int16) + y(int16) relative to tile origin |
| 1 | Ring count | (Polygons only) Number of rings (default 1) |
| 2×R | Ring ends | (Polygons only) End index of each ring |

**Coordinate Mapping:**

Coordinates are pre-projected to a 12-bit tile space (0-4096).
The ESP32 renderer converts these to screen pixels using a simple bit-shift:
`pixel = (coord * tile_size) >> 12`

---

## Output Structure

Standard z/x/y tile structure used by both IceNav and the viewer:
`output_dir/<zoom>/<x>/<y>.nav`

---

## SD Card Structure

Copy the output directory to your SD card: `/sdcard/NAVMAP/`

---

## License

MIT License

## Related Projects

- [IceNav-v3](https://github.com/jgauchia/IceNav-v3) - ESP32-based GPS navigator
