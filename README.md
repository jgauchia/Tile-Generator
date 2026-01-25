# OSM Tile Generator for IceNav

Converts OpenStreetMap PBF files to NAV tiles for [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **Direct PBF processing** - No intermediate formats (GOL, Docker, etc.)
- **Tile-based structure** - Standard z/x/y tile layout (like PNG/OSM tiles)
- **No clipping artifacts** - Features stored complete (no visible seams at tile edges)
- **Feature filtering** - Configurable via `features.json`
- **ESP32 optimized** - Small tiles, efficient for SD card access
- **Progress bar** - Visual progress per zoom level during generation

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
pip install geopandas pyogrio shapely pygame osmium
```

---

## Generate NAV Tiles

```bash
source venv/bin/activate
python tile_generator.py <input.pbf> <output_dir> features.json [--zoom 6-17]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `input.pbf` | OpenStreetMap PBF file | Required |
| `output_dir` | Output directory | Required |
| `features.json` | Feature configuration | Required |
| `--zoom` | Zoom range (e.g., `6-17`, `10-14`, `12`) | `6-17` |

**Example:**

```bash
python tile_generator.py andorra.osm.pbf ./nav_output features.json --zoom 6-17
```

---

## View NAV Tiles

```bash
python nav_viewer.py <nav_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

**Example:**

```bash
python nav_viewer.py ./nav_output --lat 42.5063 --lon 1.5218 --zoom 14
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
| `Q` / `ESC` | Quit |

---

## NAV Binary Format Specification

NAV is a proprietary binary format designed as a lightweight alternative to FlatGeobuf for embedded devices with limited resources. Unlike FlatGeobuf (which uses FlatBuffers serialization and R-Tree spatial indexing), NAV uses a minimal sequential structure optimized for ESP32's SD card access patterns.

**Key differences from FlatGeobuf:**
- No FlatBuffers dependency - pure binary format
- No R-Tree index - tiles are small enough for sequential reading
- int32 scaled coordinates instead of float64 (~50% smaller)
- Minimal header overhead

**File Header (22 bytes):**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | Magic | `NAV1` (0x4E, 0x41, 0x56, 0x31) |
| 4 | 2 | Feature count | Number of features (little-endian) |
| 6 | 4 | Min Lon | Bounding box min longitude (int32 scaled) |
| 10 | 4 | Min Lat | Bounding box min latitude (int32 scaled) |
| 14 | 4 | Max Lon | Bounding box max longitude (int32 scaled) |
| 18 | 4 | Max Lat | Bounding box max latitude (int32 scaled) |

**Feature Record:**

| Size | Field | Description |
|------|-------|-------------|
| 1 | Geometry type | 1=Point, 2=LineString, 3=Polygon |
| 2 | Color | RGB565 color (little-endian) |
| 1 | Zoom/Priority | High nibble = min_zoom, low nibble = priority/7 |
| 1 | Width | Line width in pixels (1-15, from OSM `width`/`lanes` tags) |
| 2 | Coord count | Number of coordinates (little-endian) |
| 8×N | Coordinates | lon(int32) + lat(int32) pairs |
| 1 | Ring count | (Polygons only) Number of rings |
| 2×R | Ring ends | (Polygons only) End index of each ring |

**Width Calculation:**

Width is derived from OSM tags and converted to pixels at the tile's zoom level:
- `width=*` tag: meters converted to pixels
- `lanes=*` tag: lanes × 3.5m converted to pixels
- Default: 1 pixel if no width tag present

Formula: `pixels = width_meters / (156543 × cos(lat) / 2^zoom)`

**Coordinate Scaling:**

Coordinates are stored as int32 scaled by 10,000,000 (1e7):
- `int32_value = (int32_t)(float_coord * 10000000)`
- `float_coord = (double)int32_value / 10000000.0`

This provides ~1cm precision while using half the space of float64.

---

## Output Structure

```
output/
├── 6/
│   ├── 32/
│   │   ├── 23.nav
│   │   └── 24.nav
│   └── 33/
│       └── ...
├── 13/
│   └── ...
└── 17/
    └── ...
```

Standard z/x/y tile structure:
- First level: zoom level
- Second level: tile X coordinate
- Third level: tile Y coordinate (`.nav` file)

**Note:** Features are NOT clipped to tile boundaries. Each feature is stored complete in every tile it intersects.

---

## SD Card Structure

Copy the output directory to your SD card:

```
/sdcard/NAVMAP/
├── 6/
│   └── 32/
│       └── 23.nav
├── 13/
│   └── 4123/
│       └── 2456.nav
└── 17/
    └── ...
```

---

## Feature Configuration

The `features.json` file defines which OSM features to include:

```json
{
  "highway=motorway": {
    "zoom": 6,
    "color": "#ff9999",
    "priority": 60
  },
  "highway=primary": {
    "zoom": 6,
    "color": "#ffcc99",
    "priority": 62
  },
  "building": {
    "zoom": 15,
    "color": "#dddddd",
    "priority": 80
  }
}
```

| Field | Description |
|-------|-------------|
| `zoom` | Minimum zoom level for feature visibility |
| `color` | Hex color (converted to RGB565 internally) |
| `priority` | Render order (lower = background, higher = foreground) |

---

## Download PBF Files

Get OSM extracts from [Geofabrik](https://download.geofabrik.de/)

## License

MIT License

## Related Projects

- [IceNav](https://github.com/jgauchia/IceNav-v3) - ESP32-based GPS navigator
