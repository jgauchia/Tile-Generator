# OSM to FlatGeobuf Tile Generator

Converts OpenStreetMap PBF files to FlatGeobuf (.fgb) format with R-Tree spatial indexing for [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **Direct PBF processing** - No intermediate formats (GOL, Docker, etc.)
- **Tile-based structure** - Standard z/x/y tile layout (like PNG/OSM tiles)
- **R-Tree spatial index** - Fast bounding box queries per tile
- **No clipping artifacts** - Features stored complete (no visible seams at tile edges)
- **Feature filtering** - Configurable via `features.json`
- **ESP32 optimized** - Small tiles (~100KB-1MB), efficient for SD card access
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

## Usage

### 1. Convert PBF to FlatGeobuf Tiles

```bash
source venv/bin/activate
python pbf_to_fgb.py <input.pbf> <output_dir> features.json [--zoom 6-17]
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
python pbf_to_fgb.py catalonia.osm.pbf ./fgb_output features.json --zoom 6-17
```

**Output:**

```
Processing OSM data (this may take a while for large files)...
  Progress: 1,234,567 ways, 456,789 features [2m 34s]
Processing completed in 154.23s
Statistics:
  Ways processed: 1,234,567
  Features extracted: 456,789
  Features filtered: 777,778
Generating tile-based FlatGeobuf files...
  Zoom  6: [██████████████████████████████] 7/7 tiles
  Zoom  6: 7 tiles written
  Zoom  7: [██████████████████████████████] 20/20 tiles
  Zoom  7: 20 tiles written
  ...
```

### 2. View Generated Tiles

The viewer simulates ESP32 rendering behavior - loads a 3x3 tile grid (768x768 pixels) centered on the given coordinates.

```bash
python fgb_viewer.py <fgb_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `fgb_dir` | Directory with FGB tiles | Required |
| `--lat` | Center latitude | Required |
| `--lon` | Center longitude | Required |
| `--zoom` | Zoom level (1-18) | `14` |

**Example:**

```bash
# Andorra (small test)
python fgb_viewer.py ./fgb_andorra --lat 42.5063 --lon 1.5218 --zoom 14

# Barcelona
python fgb_viewer.py ./fgb_output --lat 41.3851 --lon 2.1734 --zoom 14
```

**Viewer Controls:**

| Key | Action |
|-----|--------|
| Arrow keys | Pan map |
| Mouse drag | Pan map |
| `[` / `]` or Mouse wheel | Zoom in/out |
| `B` | Toggle background (white/black) |
| `F` | Toggle polygon fill |
| `G` | Toggle tile grid (shows tile x/y coordinates) |
| `Q` / `ESC` | Quit |

**Viewer Features:**

- **3x3 tile grid**: Loads exactly 9 tiles like ESP32
- **Viewport aligned to tiles**: No gaps or overlaps at tile boundaries
- **Tile grid overlay**: Press `G` to see tile coordinates (green=exists, red=missing)
- **Query stats**: Shows tiles loaded, features count, and load time

## Output Structure

```
fgb_output/
├── 6/
│   ├── 32/
│   │   ├── 23.fgb
│   │   └── 24.fgb
│   └── 33/
│       └── ...
├── 13/
│   ├── 4123/
│   │   ├── 2456.fgb
│   │   ├── 2457.fgb
│   │   └── ...
│   └── 4124/
│       └── ...
└── 17/
    └── ...
```

Standard z/x/y tile structure where:
- First level: zoom level
- Second level: tile X coordinate
- Third level: tile Y coordinate (.fgb file)

Each tile contains ALL layers (water, roads, buildings, etc.) combined with properties for filtering and rendering.

**Note:** Features are NOT clipped to tile boundaries. Each feature is stored complete in every tile it intersects. This avoids visible seams at tile edges - the renderer clips naturally when drawing.

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

**Fields:**

| Field | Description |
|-------|-------------|
| `zoom` | Minimum zoom level for feature visibility |
| `color` | Hex color (converted to RGB332 internally) |
| `priority` | Render order (lower = background, higher = foreground) |

## FlatGeobuf Properties

Each feature in the FGB tiles contains:

| Property | Type | Description |
|----------|------|-------------|
| `color_rgb332` | int | 8-bit color (RGB332 format) |
| `min_zoom` | int | Minimum zoom level |
| `priority` | int | Rendering priority |
| `feature_type` | string | OSM tag (e.g., `highway=primary`) |
| `osm_id` | int | Original OSM ID |
| `layer` | string | Layer name |

## ESP32 Implementation

The generated FGB tiles are optimized for ESP32:

1. **Calculate center tile**: Use Web Mercator projection to convert lat/lon/zoom to tile x,y
2. **Calculate viewport bbox**: Based on exact 3x3 tile boundaries (not center ± offset)
3. **Load 3x3 grid**: Load 9 tiles around center position
4. **Read features**: Each tile is small, read sequentially without random seeks
5. **Sort by priority**: Lower priority = rendered first (background)
6. **Render features**: Use `priority` for layer ordering, `color_rgb332` for colors

### Tile Coordinate Calculation

```cpp
// Web Mercator projection
double latRad = centerLat * M_PI / 180.0;
double n = pow(2.0, zoom);
int centerTileX = (int)((centerLon + 180.0) / 360.0 * n);
int centerTileY = (int)((1.0 - log(tan(latRad) + 1.0/cos(latRad)) / M_PI) / 2.0 * n);

// Viewport bbox aligned to tile boundaries
int minTileX = centerTileX - 1;
int maxTileX = centerTileX + 2;  // +2 for right edge
int minTileY = centerTileY - 1;
int maxTileY = centerTileY + 2;  // +2 for bottom edge
```

### Advantages of tile-based structure

- Small files (~100KB-1MB per tile)
- Sequential SD card reads (no random seeks = no timeout errors)
- Standard tile naming (compatible with OSM tools)
- Easy to update specific areas
- Familiar structure for map developers

### SD Card Structure

Copy the output directory to your SD card:

```
/sdcard/FGBMAP/
├── 6/
│   └── ...
├── 13/
│   └── ...
└── 17/
    └── ...
```

## Download PBF Files

Get OSM extracts from [Geofabrik](https://download.geofabrik.de/)

## License

MIT License

## Related Projects

- [IceNav](https://github.com/jgauchia/IceNav-v3) - ESP32-based GPS navigator
- [FlatGeobuf](https://flatgeobuf.org/) - Geospatial format specification
