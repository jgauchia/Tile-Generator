# OSM Vector Tile Generator

Highly optimized Python script for generating vector map tiles from OpenStreetMap (OSM) data using a custom binary format.

The generated tiles are extremely compact and optimized for fast rendering in custom map applications, featuring advanced compression techniques, dynamic color palette optimization, and intelligent line width handling.

---

## What Does the Script Do?

- **GOL format support** using gol CLI for efficient OSM data processing
- **Streaming GeoJSON processing** with incremental JSON parsing via ijson
- **Dynamic resource allocation** based on available system memory
- **Generates compact binary tiles** with efficient coordinate encoding
- **Global color palette optimization** with compact indices
- **Hybrid line width handling**: OSM tags + CartoDB-style zoom-based defaults
- **Geometry smoothing** at high zoom levels (≥16) for smooth curves
- **Geometry clipping** to tile boundaries using Cohen-Sutherland and Sutherland-Hodgman algorithms
- **Layer-based priority system** for correct rendering order
- **Adaptive batch sizing** based on zoom level and available memory
- **Real-time progress tracking** with performance metrics

---

## Drawing Command Set

The script implements these drawing commands for tile generation:

### Geometry Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `POLYLINE` | 2 (0x02) | Multi-point line with width | varint(width) + varint(num_points) + coordinates |
| `STROKE_POLYGON` | 3 (0x03) | Polygon outline with width | varint(width) + varint(num_points) + coordinates |

### State Management Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `SET_COLOR` | 128 (0x80) | Set RGB332 color directly | uint8(rgb332) |
| `SET_COLOR_INDEX` | 129 (0x81) | Set color from palette | varint(palette_index) |

**Note**: All commands are defined in the `DRAW_COMMANDS` dictionary in the script. Only these 4 commands are actively used in tile generation.

---

## Line Width System

The script uses a sophisticated hybrid approach for determining line widths:

### 1. Physical Width from OSM Tags
When available, the script uses physical width tags from OSM data:
- `width`: Explicit width measurement
- `maxwidth`: Maximum width
- `est_width`: Estimated width
- `diameter`: For circular features
- `gauge`: For railway tracks

Physical widths are converted from various units (meters, feet, inches) to pixels based on zoom level and latitude, then clamped to reasonable min/max constraints.

### 2. CartoDB-Style Zoom Defaults
When no physical tags exist, the script applies zoom-based styling optimized for small screens:

**Major Roads** (visible but not overwhelming at low zoom):
- Motorway: 1px@z6 → 16px@z18
- Trunk: 1px@z6 → 14px@z18  
- Primary: 1px@z8 → 14px@z18

**Connecting Roads** (thin until zoomed in):
- Secondary: 1px@z11 → 12px@z18
- Tertiary: 1px@z12 → 10px@z18

**Minor Roads** (hairline until very close):
- Residential: 0.5px@z13 → 8px@z18
- Service: 1px@z15 → 6px@z18

**Waterways** (drastically reduced):
- River: 1px@z8 → 10px@z18
- Stream: 1px@z14 → 3px@z18

**Transport**:
- Railway: 1px@z10 → 5px@z18
- Runway: 1px@z10 → 24px@z18

### 3. Width Constraints
All calculated widths are clamped with strict min/max values:
- Low zoom (≤10): Maximum 3px for any feature
- Medium zoom (≤12): Maximum 4px for any feature
- High zoom: Feature-specific maximums (e.g., motorway max 18px)

This ensures clean rendering on small screens without features blocking each other.

---

## Geometry Processing Pipeline

### 1. Smoothing (Zoom ≥ 16)
At high zoom levels, geometries are smoothed by interpolating additional points:
- Roundabouts get extra smoothing
- Maximum segment distance decreases with zoom
- Preserves closed rings for polygons

### 2. Clipping
Geometries are clipped to tile boundaries using industry-standard algorithms:
- **Lines**: Cohen-Sutherland line clipping
- **Polygons**: Sutherland-Hodgman polygon clipping
- Tolerance increased to 1e-5 for better continuity across tile edges

### 3. Coordinate Conversion
Geographic coordinates (lat/lon) are converted to tile pixel coordinates:
- Float pixels (0-256) scaled to uint16 (0-65536) for precision
- Delta encoding for compact storage
- Zigzag encoding for signed deltas

---

## Performance Optimizations

### Memory Management
- **Adaptive batch sizes**: Automatically adjusted based on zoom level and available memory
- **Dynamic worker allocation**: Based on CPU cores and available RAM
- **Garbage collection**: Triggered after processing batches
- **Memory monitoring**: System memory detected and utilized efficiently

### Processing Speed
- **Streaming JSON parsing**: Uses ijson to avoid loading entire dataset
- **Batch processing**: Features processed in configurable batch sizes
- **Thread pool execution**: Parallel tile writing with optimal worker count
- **Progress tracking**: Real-time feedback with features/second metrics

### Storage Efficiency
- **Variable-length encoding**: Varint and zigzag encoding for compact coordinates
- **Global color palette**: Colors indexed once, referenced by index
- **Delta encoding**: Coordinate differences stored instead of absolutes
- **RGB332 format**: 8-bit color (3R, 3G, 2B) reduces palette storage

---

## Usage

### Basic Usage
```bash
python tile_generator.py input.gol output_dir features.json --zoom 6-17
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `gol_file` | Path to .gol file | Required |
| `output_dir` | Output directory for tiles | Required |
| `config_file` | Features JSON configuration | Required |
| `--zoom` | Zoom level(s) (e.g. `12` or `6-17`) | `6-17` |
| `--max-file-size` | Max tile size in KB | 128 |
| `--batch-size` | Base batch size (auto-adjusted) | 10000 |

### Examples

**Process specific zoom level:**
```bash
python tile_generator.py london.gol tiles/ features.json --zoom 12
```

**Process zoom range with custom batch size:**
```bash
python tile_generator.py region.gol tiles/ features.json --zoom 10-15 --batch-size 5000
```

**Large region with smaller tile size:**
```bash
python tile_generator.py planet.gol tiles/ features.json --zoom 6-17 --max-file-size 64
```

---

## Configuration

### Features JSON Format
The script uses a JSON configuration file to define feature styling:

```json
{
  "highway=motorway": {
    "color": "#E892A2",
    "priority": 12,
    "zoom": 6
  },
  "highway=primary": {
    "color": "#FCD6A4", 
    "priority": 10,
    "zoom": 8
  },
  "building": {
    "color": "#D9D0C9",
    "priority": 14,
    "zoom": 13
  },
  "waterway=river": {
    "color": "#A0C8F0",
    "priority": 3,
    "zoom": 8
  }
}
```

**Configuration Features:**
- **Tag matching**: Supports both `key=value` and `key` only patterns
- **Color**: Hex color code (converted to RGB332 internally)
- **Priority**: Rendering order (higher = drawn later/on top)
- **Zoom**: Minimum zoom level for feature visibility
- **Automatic palette**: Colors automatically indexed into optimal palette

### Priority System
The script uses a sophisticated priority calculation:
1. **Base priority** from configuration
2. **Layer priority** from OSM `layer` tag (×1000 multiplier)
3. **Feature type priorities**:
   - Water features: 100-300
   - Land use: 200
   - Underground (tunnels): 500
   - Railways: 600
   - Roads: 700-1200 (by importance)
   - Bridges: 1300
   - Buildings: 1400
   - Amenities: 1500

---

## Binary Tile Format

### File Structure
```
[varint: num_commands]
[command_1]
[command_2]
...
[command_n]
```

### Command Structure

**SET_COLOR (0x80)**
```
[0x80][rgb332_byte]
```

**SET_COLOR_INDEX (0x81)**
```
[0x81][varint: palette_index]
```

**POLYLINE (0x02)**
```
[0x02][varint: width][varint: num_points]
[zigzag: x1][zigzag: y1]
[zigzag: dx2][zigzag: dy2]
...
```

**STROKE_POLYGON (0x03)**
```
[0x03][varint: width][varint: num_points]
[zigzag: x1][zigzag: y1]
[zigzag: dx2][zigzag: dy2]
...
```

### Palette File Format
```
palette.bin:
[uint32: num_colors]
[r1][g1][b1]  // RGB888 for color 0
[r2][g2][b2]  // RGB888 for color 1
...
```

---

## Output Structure

```
output_dir/
├── palette.bin          # Global color palette (RGB888)
├── 6/                   # Zoom level 6
│   └── 32/
│       └── 21.bin       # Tile (x=32, y=21, z=6)
├── 12/                  # Zoom level 12
│   ├── 2048/           
│   │   ├── 1536.bin    
│   │   └── 1537.bin
│   └── 2049/
└── 17/                  # Zoom level 17
    └── ...
```

---

## System Requirements

### Required
- **Python 3.7+**
- **gol CLI** ([download](https://www.geodesk.com/download/))
- **Python packages**: `ijson`

### Recommended
- **Multi-core CPU**: Script utilizes all available cores
- **4-8GB RAM**: For processing large regions
- **SSD storage**: For better I/O performance

### Installation
```bash
# Install Python dependencies
pip install ijson

# Install gol CLI
# Download from: https://www.geodesk.com/download/
```

---

## Performance Statistics

The script provides comprehensive processing reports:

```
System: 8 CPU cores, 16384MB RAM -> Using 6 workers
Resource settings: 6 workers, base batch size: 10000
Building color palette...
Palette: 39 colors
Palette written

Processing zoom 12 (batch size: 4000, workers: 6)...
Zoom 12: 54482 features processed, 139 tiles written
Zoom 12 completed in 2m 45.32s

Total tiles written: 139
Total size: 1.2MB
Average tile size: 8847 bytes
Palette file: tiles/palette.bin
Total processing time: 2m 45.32s
```

---

## Advanced Usage

### Processing Large Regions
For planet-scale or large extracts, process in smaller zoom ranges:
```bash
# Low zoom levels (broader features)
python tile_generator.py planet.gol tiles/ features.json --zoom 6-10

# Medium zoom levels
python tile_generator.py planet.gol tiles/ features.json --zoom 11-14

# High zoom levels (detailed features)
python tile_generator.py planet.gol tiles/ features.json --zoom 15-17
```

### Memory-Constrained Systems
Reduce batch size for systems with limited RAM:
```bash
python tile_generator.py input.gol tiles/ features.json --batch-size 2000
```

### Custom Tile Size Limits
Adjust maximum tile file size:
```bash
# Smaller tiles (64KB) for bandwidth-constrained environments
python tile_generator.py input.gol tiles/ features.json --max-file-size 64

# Larger tiles (512KB) for high-detail regions
python tile_generator.py input.gol tiles/ features.json --max-file-size 512
```

---

## Troubleshooting

### Common Issues

**gol CLI not found:**
```bash
# Download and install gol from https://www.geodesk.com/download/
# Add to PATH or use absolute path
```

**Memory issues:**
- Script automatically adjusts based on available memory
- Reduce `--batch-size` for very constrained systems
- Process zoom ranges separately

**Slow processing:**
- Verify GOL file is not corrupted
- Check system resources (CPU, RAM, disk I/O)
- Use SSD for better performance
- Script automatically optimizes worker count

**Missing features in tiles:**
- Check `zoom` setting in features.json
- Verify feature tags match configuration
- Review gol query output for filtering issues

---

## Technical Details

### Coordinate Precision
- **Input**: Geographic coordinates (lat/lon, decimal degrees)
- **Internal**: Float pixels (0-256 per tile)
- **Storage**: Uint16 (0-65536) for sub-pixel precision
- **Delta encoding**: Reduces storage by ~60-70%

### Color System
- **Input**: Hex colors (#RRGGBB)
- **Conversion**: RGB888 → RGB332 (8-bit)
- **Storage**: 3 bytes per palette color (RGB888)
- **Usage**: 1 byte per palette index in commands

### Width Calculation
1. Check OSM tags for physical width
2. Convert to pixels using zoom and latitude
3. Apply feature-specific constraints (min/max)
4. Fall back to CartoDB-style defaults if no tags
5. Interpolate between zoom breakpoints

### Clipping Algorithms
- **Cohen-Sutherland**: Fast line segment clipping with outcodes
- **Sutherland-Hodgman**: Polygon clipping against rectangular viewport
- **Tolerance**: 1e-5 for continuity across tile edges

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **OpenStreetMap**: For providing open map data
- **Geodesk**: For GOL format and gol CLI tool
- **ijson**: For incremental JSON parsing