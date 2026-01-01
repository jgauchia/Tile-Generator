# OSM to FlatGeobuf Tile Generator

Converts OpenStreetMap PBF files to FlatGeobuf (.fgb) format with R-Tree spatial indexing for [IceNav](https://github.com/jgauchia/IceNav-v3) ESP32-based GPS navigator.

## Features

- **Direct PBF processing** - No intermediate formats (GOL, Docker, etc.)
- **R-Tree spatial index** - Fast bounding box queries
- **Zoom-level organization** - Separate files per zoom level for optimal file size
- **Feature filtering** - Configurable via `features.json`
- **ESP32 optimized** - Small files, efficient for SD card access

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

### 1. Convert PBF to FlatGeobuf

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

### 2. View Generated Files

```bash
python fgb_viewer.py <fgb_dir> --lat <latitude> --lon <longitude> [--zoom <level>]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `fgb_dir` | Directory with FGB files | Required |
| `--lat` | Center latitude | Required |
| `--lon` | Center longitude | Required |
| `--zoom` | Zoom level (1-18) | `14` |

**Example:**

```bash
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
| `G` | Toggle tile grid |
| `Q` / `ESC` | Quit |

## Output Structure

```
fgb_output/
├── 6/
│   ├── water.fgb
│   ├── roads.fgb
│   └── landuse.fgb
├── 7/
│   └── ...
├── 10/
│   ├── roads.fgb      (+ secondary roads)
│   └── railways.fgb
├── 14/
│   ├── roads.fgb      (+ residential)
│   ├── buildings.fgb
│   └── ...
└── 17/
    └── ... (all features)
```

Each zoom level directory contains only features visible at that zoom level, resulting in smaller files optimized for R-Tree queries.

### Layer Files

| Layer | Contents |
|-------|----------|
| `water.fgb` | Coastlines, water bodies, rivers |
| `landuse.fgb` | Forests, parks, residential areas |
| `roads.fgb` | All road types |
| `railways.fgb` | Rail lines |
| `buildings.fgb` | Building footprints |
| `amenities.fgb` | Hospitals, schools, parking |
| `infrastructure.fgb` | Bridges, tunnels, airports |
| `terrain.fgb` | Peaks, cliffs |
| `places.fgb` | Towns, villages |

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

Each feature in the FGB files contains:

| Property | Type | Description |
|----------|------|-------------|
| `color_rgb332` | int | 8-bit color (RGB332 format) |
| `min_zoom` | int | Minimum zoom level |
| `priority` | int | Rendering priority |
| `feature_type` | string | OSM tag (e.g., `highway=primary`) |
| `osm_id` | int | Original OSM ID |
| `layer` | string | Layer name |

## ESP32 Implementation

The generated FGB files are optimized for ESP32:

1. **Directory selection**: Use current zoom to select folder
2. **R-Tree query**: Read index, seek to matching features
3. **Partial reads**: Use `fseek()`/`fread()` for bbox queries
4. **Properties**: Read `color_rgb332`, `priority` for rendering

### Advantages over tile-based formats

- No tile coordinate calculations
- Query any bounding box directly
- Smaller total file size
- Efficient SD card access with R-Tree seeks

## Download PBF Files

Get OSM extracts from [Geofabrik](https://download.geofabrik.de/)

## License

MIT License

## Related Projects

- [IceNav](https://github.com/jgauchia/IceNav-v3) - ESP32-based GPS navigator
- [FlatGeobuf](https://flatgeobuf.org/) - Geospatial format specification
