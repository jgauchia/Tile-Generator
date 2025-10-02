# OSM Vector Tile Generator

This repository contains a Python script for generating vector map tiles from OpenStreetMap (OSM) data in a highly optimized custom binary format.

The generated tiles are extremely compact and optimized for fast rendering in custom map applications, featuring advanced compression techniques, dynamic color palette optimization, feature-specific optimizations, and state-based command architecture.

---

## What Does the Script Do?

- Extracts relevant geometries and attributes from OSM PBF files using `ogr2ogr` with parallel processing
- Filters and merges data into simplified GeoJSON with streaming processing
- **Builds dynamic color palette** automatically from `features.json` configuration
- **Memory-optimized processing** with single-pass feature reading for all zoom levels
- **SQLite database storage** for efficient feature management and retrieval
- Assigns features to tiles for zoom levels with boundary optimization
- **Generates compact binary tiles** with efficient coordinate encoding
- Uses comprehensive memory management for large-scale processing

---

## Key Optimizations

The script implements multiple optimization layers for maximum compression and performance:

### Memory Optimization
- **Single feature processing**: Features are read once for all zoom levels
- **Disk-based storage**: Uses SQLite database for temporary feature storage
- **Batch processing**: Processes features in configurable batches to control memory usage
- **Automatic cleanup**: Garbage collection and memory management throughout processing

### Dynamic Color Palette System
- **Automatic palette generation**: Analyzes `features.json` and creates optimal color palette
- **Indexed colors**: Each unique color gets compact index (0-N) for maximum compression
- **Smart state management**: SET_COLOR_INDEX and SET_COLOR commands minimize redundancy
- **Adaptive encoding**: Most frequent colors get optimal indices

### Compression Techniques
- **Variable-length encoding**: Efficient coordinate compression with varint/zigzag
- **Delta encoding**: Coordinate differences for compact storage
- **RGB332 color format**: 8-bit color encoding for efficient storage
- **Boundary optimization**: Cross-tile geometries optimized to reduce duplication

---

## Drawing Command Set

The script implements a set of drawing commands for efficient tile generation:

### Basic Geometry Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `LINE` | 0x01 | Single line segment | x1, y1, x2, y2 (delta-encoded) |
| `POLYLINE` | 0x02 | Multi-point line | point_count + coordinate_deltas |
| `STROKE_POLYGON` | 0x03 | Polygon outline | point_count + coordinate_deltas |
| `HORIZONTAL_LINE` | 0x05 | Horizontal line optimization | x1, width, y |
| `VERTICAL_LINE` | 0x06 | Vertical line optimization | x, y1, height |

---


### Processing Performance
- **Multi-core processing**: Utilizes all available CPU cores for PBF extraction
- **Streaming architecture**: Processes large PBF files without memory overflow
- **Memory optimization**: Constant memory usage regardless of zoom level count
- **Batch processing**: Configurable batch sizes for optimal memory management

---

## Script Features

### Memory-Optimized Processing
- **Single-pass feature processing**: Features are read once for all zoom levels
- **SQLite database storage**: Efficient disk-based feature storage and retrieval
- **Batch processing**: Configurable batch sizes to control memory usage
- **Automatic cleanup**: Garbage collection and memory management

### Dynamic Color Palette
```
Analyzing colors from features.json to build dynamic palette...
Dynamic color palette created:
  - Total unique colors: 24
  - Palette indices: 0-23
  - Memory saving potential: 24 colors -> compact indices
```

### Comprehensive Statistics
The script provides detailed processing reports:
```
Processing features and storing in database...
Processing 125000 features for 12 zoom levels...
Zoom 6: 15420 features stored
Zoom 7: 28930 features stored
Zoom 8: 45670 features stored
...
```

### Performance Monitoring
- Real-time memory usage tracking
- Processing time measurement per zoom level
- Progress bars for all major processing steps
- Detailed statistics and reporting

---

## Usage

### Command Line
```bash
python tile_generator.py planet.osm.pbf tiles/ features.json --zoom 6-17 --max-file-size 128 --db-path features.db
```

### Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `pbf_file` | Path to OSM PBF file | Required |
| `output_dir` | Output directory for tiles | Required |
| `config_file` | Features JSON configuration | Required |
| `--zoom` | Zoom level or range (e.g. `12` or `6-17`) | `6-17` |
| `--max-file-size` | Max tile size in KB | 128 |
| `--db-path` | Path for temporary database | `features.db` |

### Memory Optimization Features

The optimized version provides significant memory improvements:

- **Single feature processing**: Features are read once for all zoom levels
- **Disk-based storage**: Uses SQLite database for temporary feature storage
- **Memory efficiency**: Constant memory usage regardless of zoom level count
- **Scalability**: Handles large datasets without memory overflow
- **Batch processing**: Consistent memory usage across multiple zoom levels

---

## Binary Tile Format

### Format Features
- **Basic geometry commands**: Lines, polygons, polylines
- **Variable-length encoding**: Efficient coordinate compression with varint/zigzag
- **Delta encoding**: Coordinate differences for compact storage
- **RGB332 color format**: 8-bit color encoding embedded in each command

### Command Types
1. **Basic Geometry** (0x01-0x06): Lines, polygons, polylines
2. **Optimized Lines**: Horizontal and vertical line optimizations

---

## Configuration

### Features JSON
The script automatically detects feature types from your configuration:

```json
{
  "highway=primary": {
    "color": "#DC143C", 
    "priority": 6,
    "zoom": 8
  },
  "building": {
    "color": "#8B4513",
    "priority": 5, 
    "zoom": 12
  }
}
```

**Configuration Features:**
- Colors automatically indexed into optimal palette
- Zoom-based filtering reduces processing overhead
- Priority-based rendering order optimization

---

## Dependencies

### Required Packages
```
shapely  
ijson
tqdm
```

### System Requirements
- **GDAL/OGR**: `ogr2ogr` command-line tool for PBF processing
- **Multi-core CPU**: Script utilizes all available cores for optimal performance
- **RAM**: Streaming architecture works with modest RAM (4-8GB recommended for planet files)

### Installation
```bash
# Python packages
pip install shapely ijson tqdm

# GDAL (Ubuntu/Debian)
sudo apt-get install gdal-bin

# GDAL (macOS)
brew install gdal
```

---

## Advanced Usage

### Large-Scale Processing
For planet-scale or large regional extracts:
```bash
# Process in smaller zoom ranges for memory efficiency
python tile_generator.py region.osm.pbf tiles/ features.json --zoom 6-12
python tile_generator.py region.osm.pbf tiles/ features.json --zoom 13-17
```

### Custom Optimization Tuning
- **Color palette optimization**: Script automatically creates optimal color palette from your `features.json`
- **Memory optimization**: Object pools and caches automatically managed for optimal performance
- **Boundary optimization**: Cross-tile geometries automatically optimized to reduce duplication

### Performance Monitoring
The script provides comprehensive monitoring:
- Memory usage tracking during processing
- Optimization statistics per zoom level  
- Detailed savings reports with byte-level analysis
- Processing time measurements for performance tuning

---

## Output Structure

```
output_dir/
├── palette.bin          # Optimized color palette
├── 12/                  # Zoom level 12
│   ├── 2048/           
│   │   ├── 1536.bin    # Tile (x=2048, y=1536)
│   │   └── 1537.bin
│   └── 2049/
└── 13/                  # Zoom level 13
    └── ...
```

Each `.bin` file contains optimized drawing commands with variable-length encoding and embedded RGB332 colors.

---

## Documentation

- [Binary Tile File Format](/docs/bin_tile_format.md) - Complete specification of drawing commands
- [Features JSON Format](/docs/features_json_format.md) - Configuration format details
- [Tile Viewer Documentation](/docs/tile_viewer.md) - Tile viewer with command support

---

## Technical Highlights

### Memory Management
- **SQLite database**: Efficient disk-based storage for features
- **Batch processing**: Configurable batch sizes for memory control
- **Streaming processing**: Handles unlimited file sizes with constant memory usage
- **Garbage collection optimization**: Strategic GC calls prevent memory bloat

### Processing Optimization
- **Single-pass processing**: Features read once for all zoom levels
- **Multi-core processing**: Parallel PBF extraction and tile generation
- **Variable-length encoding**: Efficient coordinate compression
- **Boundary optimization**: Cross-tile geometries optimized to reduce duplication

### Rendering Optimization
- **State-based commands**: Minimizes GPU state changes for better rendering performance
- **Priority sorting**: Commands ordered for optimal rendering pipeline utilization
- **TFT display optimization**: Reduced color register updates for embedded displays
- **Cache-friendly layout**: Data structures optimized for CPU cache efficiency

---
