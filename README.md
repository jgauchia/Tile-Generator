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

### Advanced Optimization Suite
- **Database optimization**: Composite indexes, WAL mode, batch operations
- **Geometry optimization**: Cached operations, pre-filtering, smart simplification
- **Memory optimization**: Object pooling, smart GC, streaming with chunking
- **Parallelization optimization**: Intelligent work distribution, adaptive workers
- **I/O optimization**: Buffered operations, directory pre-creation, error handling
- **Algorithm optimization**: Mathematical caching, coordinate transformation caching

### Dynamic Color Palette
```
Analyzing colors from features.json to build dynamic palette
Dynamic color palette created:
  - Total unique colors: 39
  - Palette indices: 0-38
  - Memory saving potential: 39 colors -> compact indices
Dynamic palette ready with 39 colors from your features.json
Writing palette to TEST/palette.bin (39 colors)
Palette written successfully
```

### Comprehensive Statistics
The script provides detailed processing reports:
```
Processing PBF directly to database (minimal temporary files)
Available layers in PBF: ['points', 'lines', 'multilinestrings', 'multipolygons', 'other_relations']
Config requires these fields: amenity, building, highway, landuse, leisure, natural, place, railway, waterway
[1/5] Processing layer: points directly to database
Processed 1328 features from layer points
Layer points: 1328 features processed
[2/5] Processing layer: lines directly to database
Processed 8568 features from layer lines
Layer lines: 8568 features processed
[3/5] Processing layer: multilinestrings directly to database
Layer multilinestrings: 0 features processed
[4/5] Processing layer: multipolygons directly to database
Processed 11759 features from layer multipolygons
Layer multipolygons: 11759 features processed
[5/5] Processing layer: other_relations directly to database
Layer other_relations: 0 features processed
Total processed: 21655 features directly from PBF
Zoom 6: 870 features stored
Processing zoom level 6 from database
Found 1 tiles for zoom 6
Writing tiles (zoom 6): 100%|█████████████████████| 1/1 [00:00<00:00,  1.51it/s]
Zoom 6: 1 tiles, average size = 7831.00 bytes
Process completed successfully
Cleaning up temporary files and database...
Cleaned up database file: features.db
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

### Advanced Memory Management

The optimized version provides significant memory improvements:

- **Object pooling**: Reusable coordinate tuples and feature data dictionaries
- **Smart garbage collection**: Memory-aware GC with pressure monitoring
- **Cache management**: Automatic cache clearing (geometry, algorithm, object pools)
- **Streaming with chunking**: Process features in optimized 1000-feature chunks
- **Memory monitoring**: Real-time memory usage tracking and optimization
- **Scalability**: Handles large datasets without memory overflow

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
- **Geometry optimization**: Automatic caching and pre-filtering for optimal geometry processing
- **Parallelization optimization**: Automatic tile complexity classification and worker allocation
- **I/O optimization**: Automatic buffering and directory pre-creation for optimal file operations
- **Algorithm optimization**: Automatic mathematical operation caching for optimal performance

### Performance Monitoring
The script provides comprehensive monitoring:
- **Memory usage tracking**: Real-time memory usage monitoring during processing
- **Cache statistics**: Geometry, algorithm, and object pool cache performance
- **Worker allocation**: Adaptive worker allocation based on tile complexity
- **I/O optimization**: File operation buffering and directory creation statistics
- **Processing time measurements**: Detailed timing for each optimization layer
- **Optimization statistics**: Per-zoom-level performance metrics and savings reports

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

### Database Optimization
- **Composite indexes**: Multi-column indexes for optimal query performance
- **WAL mode**: Write-Ahead Logging for better concurrency and crash recovery
- **Batch operations**: Efficient bulk inserts and updates with executemany
- **Connection optimization**: Optimized SQLite connection settings

### Geometry Optimization
- **Geometry caching**: Cached simplified geometries by zoom level
- **Pre-filtering**: Bounding box checks to avoid expensive intersection operations
- **Mathematical caching**: Cached sin, cos, tan, log, atan operations
- **Tile bounds caching**: Cached tile boundary calculations

### Memory Optimization
- **Object pooling**: Reusable coordinate tuples and feature data dictionaries
- **Smart garbage collection**: Memory-aware GC with pressure monitoring
- **Cache management**: Automatic clearing of geometry, algorithm, and object caches
- **Streaming with chunking**: Process features in optimized chunks

### Parallelization Optimization
- **Intelligent work distribution**: Tiles classified by complexity (simple/medium/complex)
- **Adaptive worker allocation**: More workers for simple tiles, fewer for complex ones
- **Memory-aware scheduling**: Prevents memory overload in worker processes
- **Optimized batch creation**: Different batch sizes based on tile complexity

### I/O Optimization
- **Buffered operations**: Configurable buffer sizes for file operations
- **Directory pre-creation**: Batch directory creation to minimize system calls
- **Error handling**: Robust I/O error handling and recovery
- **Optimized file writes**: Pre-built data for efficient file writing

### Algorithm Optimization
- **Mathematical operation caching**: Cached trigonometric and logarithmic operations
- **Coordinate transformation caching**: Cached pixel coordinate calculations
- **Precision optimization**: Configurable coordinate precision for optimal performance
- **Tile bounds caching**: Cached tile boundary calculations

---
