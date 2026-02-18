# Changes — Tile Generator v4 Rewrite

## Overview

Major rewrite of the tile generation engine. This version introduces modular architecture, rich OSM feature coverage (144 types), and rendering features designed for the IceNav ESP32 navigator.

## What Changed

### Modular Architecture

The code is organized into 5 focused modules:

| Module | Role |
|--------|------|
| `tile_generator.py` | Main orchestrator: CLI, PBF processing pipeline, parallel tile generation |
| `constants.py` | Pure data: NAV format constants, OSM Carto color palette, nibble mapping tables |
| `geo_utils.py` | Geographic utilities: coordinate transforms, RGB565 conversion, Douglas-Peucker simplification |
| `osm_handlers.py` | OSM extraction: Pyosmium handlers for nodes, ways, relations and multipolygons |
| `tile_writer.py` | Binary writer: polygon merging, geometry clipping, NAV serialization |

### New Rendering Features

- **Multipolygon support** — Correct handling of complex OSM relations (lakes with islands, forests with clearings, admin boundaries with exclaves)
- **Z-ordering via nibbles** — Proper draw order using the priority nibble: landuse (20-27) < water (40-45) < roads (50-58) < buildings (70) < labels (90-95)
- **Bridge rendering** — Bridges detected via `bridge=yes` tag and rendered on a dedicated upper layer
- **Building outlines** — Building footprint flag (bit 7 of feature properties) for zoom levels where individual buildings are visible
- **Label placement with collision detection** — City/town/village names placed avoiding overlaps, with font size scaled by place importance
- **Population-based filtering** — Cities appear progressively: capitals from z6, cities >100k from z8, towns from z10, villages from z12
- **Area filtering** — Polygons smaller than 4px² at target zoom are discarded to reduce tile size and visual noise
- **Polygon merging** — Adjacent same-type polygons merged via Shapely `unary_union` to reduce feature count per tile

### Feature Configuration

New `features_z8-16_v4.json` with 144 OSM feature types, organized by rendering layer:

- Landuse/landcover: forest, farmland, grass, meadow, vineyard, orchard, cemetery, residential, industrial, commercial...
- Natural: water, coastline, wetland, beach, sand, bare_rock, scree, glacier...
- Water: rivers, streams, canals, lakes, reservoirs
- Transport: motorway through track, railway, bridges
- Buildings: yes, residential, commercial, industrial, church...
- Amenities: hospital, school, university, parking
- Places: city, town, village labels with population

Each entry defines: minimum zoom, hex color (converted to RGB565), draw priority (nibble), and line width.

### Tile Viewer Rewrite

`tile_viewer.py` has been substantially rewritten. It is the essential companion to the generator — without it, there is no way to visually validate the generated `.nav` tiles on desktop before deploying to ESP32.

Changes:
- Renders all 144 feature types with correct z-ordering
- Labels rendered with proper font sizing by place type
- OSM Carto-inspired color palette matching the generator output
- Feature identification on right-click (shows OSM tags, priority, geometry type)
- Legend overlay (`L` key) showing color-to-feature mapping

### Binary Format Updates

The NAV1 binary format is updated (see `docs/bin_tile_format.md`):
- Feature header now 12 bytes (was 11) with alignment padding
- Width field bit 7 reserved for building outline flag
- Priority nibble used for z-ordering (was unused)

### Supporting Material

| Directory | Contents |
|-----------|----------|
| `debug/` | Diagnostic scripts: tile inspection, feature scanning, multipolygon debugging |
| `notes/` | Design investigation notes: nibble system rationale, road/rail analysis, multipolygon edge cases |
| `reference/` | OSM Carto SQL/YAML extracts used as rendering reference |
| `mass_copy/` | Batch tile copy utility for SD card deployment |
| `archive/` | Previous versions of generator, viewer, and feature config |

## Requirements

- Python 3.8+
- Virtual environment (recommended)

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install shapely pygame osmium
```

## Generate NAV Tiles

```bash
python tile_generator.py <input.pbf> <output_dir> features_z8-16_v4.json [--zoom 6-17]
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `input.pbf` | OpenStreetMap PBF file ([Geofabrik](https://download.geofabrik.de/)) | Required |
| `output_dir` | Output directory for tile tree | Required |
| `features_z8-16_v4.json` | Feature configuration | Required |
| `--zoom` | Zoom range (e.g., `8-16`, `10-14`, `12`) | `6-17` |
| `-v` | Verbose logging | Off |

## View NAV Tiles

```bash
python tile_viewer.py <nav_dir> --lat <latitude> --lon <longitude> [--zoom <level>] [--config <features.json>]
```
