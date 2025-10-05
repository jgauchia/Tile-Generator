# OSM Vector Tile Generator

This repository contains a Python script for generating vector map tiles from OpenStreetMap (OSM) data in a highly optimized custom binary format.

The generated tiles are extremely compact and optimized for fast rendering in custom map applications, featuring advanced compression techniques, dynamic color palette optimization, feature-specific optimizations, and state-based command architecture.

---

## What Does the Script Do?

- **Direct PBF processing** using Pyosmium for maximum performance (no temporary files)
- **Streaming OSM feature processing** with real-time geometry extraction and filtering
- **Builds dynamic color palette** automatically from `features.json` configuration
- **Memory-optimized processing** with single-pass feature reading for all zoom levels
- **SQLite database storage** for efficient feature management and retrieval
- **Zoom-accumulative rendering** - features appear in all zoom levels above their minimum
- **Generates compact binary tiles** with efficient coordinate encoding
- **Advanced polygon rendering** with intelligent border detection and styling

---

## Drawing Command Set

The script implements a comprehensive set of drawing commands for efficient tile generation:

### Basic Geometry Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `LINE` | 0x01 | Single line segment | x1, y1, x2, y2 (delta-encoded) |
| `POLYLINE` | 0x02 | Multi-point line | point_count + coordinate_deltas |
| `STROKE_POLYGON` | 0x03 | Polygon outline | point_count + coordinate_deltas |
| `HORIZONTAL_LINE` | 0x05 | Horizontal line optimization | x1, width, y |
| `VERTICAL_LINE` | 0x06 | Vertical line optimization | x, y1, height |

### State Management Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `SET_COLOR` | 0x80 | Set RGB color | r, g, b (8-bit values) |
| `SET_COLOR_INDEX` | 0x81 | Set color from palette | palette_index |
| `SET_LAYER` | 0x88 | Set rendering layer | layer_number |

### Advanced Pattern Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `GRID_PATTERN` | 0x85 | Grid lines for tile borders | pattern_data |
| `COMPRESSED_POLYLINE` | 0x8B | Compressed polyline | compressed_data |

### Optimized Geometry Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `OPTIMIZED_POLYGON` | 0x8C | Optimized polygon | optimized_data |
| `HOLLOW_POLYGON` | 0x8D | Hollow polygon | polygon_data |
| `OPTIMIZED_TRIANGLE` | 0x8E | Optimized triangle | triangle_data |
| `OPTIMIZED_RECTANGLE` | 0x8F | Optimized rectangle | rectangle_data |
| `OPTIMIZED_CIRCLE` | 0x90 | Optimized circle | circle_data |

### Simple Shape Commands
| Command | Code | Purpose | Data Format |
|---------|------|---------|-------------|
| `SIMPLE_RECTANGLE` | 0x96 | Simple rectangle | x, y, width, height |
| `SIMPLE_CIRCLE` | 0x97 | Simple circle | x, y, radius |
| `SIMPLE_TRIANGLE` | 0x98 | Simple triangle | x1, y1, x2, y2, x3, y3 |
| `DASHED_LINE` | 0x99 | Dashed line | x1, y1, x2, y2, dash_length |
| `DOTTED_LINE` | 0x9A | Dotted line | x1, y1, x2, y2, dot_spacing |

---

## Processing Performance

- **Pyosmium streaming**: Direct PBF processing without temporary files
- **Multi-core processing**: Utilizes all available CPU cores for feature extraction
- **Memory optimization**: Constant memory usage regardless of zoom level count
- **Batch processing**: Configurable batch sizes for optimal memory management
- **Zoom-accumulative logic**: Features appear in all zoom levels above their minimum

---

## Script Features

### Advanced Optimization Suite
- **Database optimization**: Composite indexes, WAL mode, batch operations
- **Geometry optimization**: Real-time simplification, pre-filtering, smart clipping
- **Memory optimization**: Streaming processing, smart GC, efficient data structures
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
Processing PBF directly to database using Pyosmium (maximum performance)
Config requires these fields: amenity, background_colors, building, config_file, fill_polygons, fps_limit, highway, landuse, leisure, log_level, max_cache_size, natural, place, railway, statusbar_height, thread_pool_size, tile_size, toolbar_width, viewport_size, waterway
Compiled 79 tag patterns for Pyosmium processing
Starting Pyosmium processing...
Pyosmium processing completed in 165.52s
Total processed: 54482 features directly from PBF
Zoom 6: 870 features stored
Zoom 7: 870 features stored
Zoom 8: 1627 features stored
Zoom 9: 1648 features stored
Zoom 10: 1649 features stored
Zoom 11: 1661 features stored
Zoom 12: 5483 features stored
Zoom 13: 5774 features stored
Zoom 14: 9482 features stored
Zoom 15: 25418 features stored
Processing zoom level 6 from database
Found 1 tiles for zoom 6
Writing tiles (zoom 6): 100%|█████████████████████| 1/1 [00:00<00:00,  1.61s/it]
Zoom 6: 1 tiles, average size = 3778.00 bytes
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

- **Streaming processing**: Direct PBF processing without temporary files
- **Smart garbage collection**: Memory-aware GC with pressure monitoring
- **Cache management**: Automatic cache clearing and optimization
- **Memory monitoring**: Real-time memory usage tracking and optimization
- **Scalability**: Handles large datasets without memory overflow

---

## Binary Tile Format

### Format Features
- **Comprehensive command set**: Lines, polygons, polylines, shapes, patterns
- **Variable-length encoding**: Efficient coordinate compression with varint/zigzag
- **Delta encoding**: Coordinate differences for compact storage
- **RGB332 color format**: 8-bit color encoding embedded in each command
- **Layer-based rendering**: Commands ordered by rendering layer for optimal display

### Command Types
1. **Basic Geometry** (0x01-0x06): Lines, polygons, polylines
2. **State Management** (0x80-0x81, 0x88): Color and layer control
3. **Advanced Patterns** (0x85, 0x8B): Grid patterns, compressed data
4. **Optimized Geometry** (0x8C-0x90): Optimized shapes and polygons
5. **Simple Shapes** (0x96-0x9A): Basic geometric shapes

---

## Configuration

### Features JSON
The script automatically detects feature types from your configuration:

```json
{
  "tile_size": 256,
  "viewport_size": 768,
  "fill_polygons": true,
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
- **System parameters**: Control tile generation and viewer behavior
- **Feature definitions**: Colors, priorities, and zoom levels
- **Polygon rendering**: Control fill behavior and border styling
- **Colors automatically indexed** into optimal palette
- **Zoom-based filtering** reduces processing overhead
- **Priority-based rendering order** optimization

---

## Dependencies

### Required Packages
```
shapely  
ijson
tqdm
osmium
```

### System Requirements
- **Pyosmium**: High-performance OSM PBF processing library
- **Multi-core CPU**: Script utilizes all available cores for optimal performance
- **RAM**: Streaming architecture works with modest RAM (4-8GB recommended for planet files)

### Installation
```bash
# Python packages
pip install shapely ijson tqdm osmium

# Pyosmium system dependencies (Ubuntu/Debian)
sudo apt-get install libosmium-dev python3-dev

# Pyosmium system dependencies (macOS)
brew install osmium-tool
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
- **Memory optimization**: Streaming processing and efficient data structures
- **Geometry optimization**: Real-time simplification and pre-filtering
- **Parallelization optimization**: Automatic worker allocation based on system capabilities
- **I/O optimization**: Automatic buffering and directory pre-creation
- **Algorithm optimization**: Mathematical operation caching for optimal performance

### Performance Monitoring
The script provides comprehensive monitoring:
- **Memory usage tracking**: Real-time memory usage monitoring during processing
- **Processing time measurements**: Detailed timing for each optimization layer
- **Feature statistics**: Per-zoom-level feature counts and processing metrics
- **Optimization statistics**: Performance metrics and efficiency reports

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

Each `.bin` file contains optimized drawing commands with variable-length encoding, embedded RGB332 colors, and layer-based rendering order.

---

## Documentation

- [Binary Tile File Format](bin_tile_format.md) - Complete specification of drawing commands
- [Features JSON Format](features_json_format.md) - Configuration format details
- [Tile Viewer Documentation](tile_viewer.md) - Tile viewer with command support

---

## Technical Highlights

### Pyosmium Integration
- **Direct PBF processing**: No temporary files, maximum performance
- **Streaming architecture**: Process large files without memory overflow
- **Real-time filtering**: Tag pattern matching during processing
- **Geometry extraction**: Direct conversion from OSM ways to Shapely geometries

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
- **Real-time simplification**: Douglas-Peucker algorithm with zoom-appropriate tolerances
- **Pre-filtering**: Bounding box checks to avoid expensive intersection operations
- **Tile clipping**: Geometries clipped to tile boundaries for optimal rendering
- **Zoom-accumulative logic**: Features appear in all zoom levels above their minimum

### Memory Optimization
- **Streaming processing**: Direct PBF processing without temporary files
- **Smart garbage collection**: Memory-aware GC with pressure monitoring
- **Efficient data structures**: Optimized for memory usage and performance
- **Batch processing**: Process features in optimized chunks

### Polygon Rendering
- **Intelligent border detection**: Distinguishes between tile border and interior segments
- **Conditional styling**: Different colors for border vs interior segments
- **Fill control**: Configurable polygon filling with `fill_polygons` parameter
- **Visual optimization**: Enhanced visual appearance with smart border handling

---

## Recent Updates

### Pyosmium Integration
- **Replaced ogr2ogr**: Direct PBF processing using Pyosmium for maximum performance
- **Eliminated temporary files**: No more GeoJSON intermediate files
- **Improved performance**: Significantly faster processing of large PBF files
- **Better memory usage**: Streaming processing without memory overflow

### Enhanced Polygon Rendering
- **Smart border detection**: Automatic detection of polygon segments on tile borders
- **Conditional styling**: Different rendering for border vs interior segments
- **Configurable filling**: Control polygon fill behavior via `features.json`
- **Visual improvements**: Better visual distinction between tile borders and polygon interiors

### Zoom Accumulative Logic
- **Cumulative rendering**: Features appear in all zoom levels above their minimum
- **Efficient processing**: Single-pass processing for all zoom levels
- **Memory optimization**: Shared geometry data across zoom levels
- **Performance improvement**: Reduced processing time and memory usage

---

## Performance Comparison

### Before Pyosmium Integration
- **Processing method**: ogr2ogr with temporary GeoJSON files
- **Memory usage**: High due to temporary files
- **Processing time**: Slower due to file I/O overhead
- **Scalability**: Limited by temporary file size

### After Pyosmium Integration
- **Processing method**: Direct PBF streaming with Pyosmium
- **Memory usage**: Low and constant regardless of file size
- **Processing time**: Significantly faster due to direct processing
- **Scalability**: Excellent, handles planet-scale files efficiently

---

## Troubleshooting

### Common Issues

1. **Pyosmium not available**:
   ```bash
   pip install osmium
   sudo apt-get install libosmium-dev python3-dev  # Ubuntu/Debian
   ```

2. **Memory issues with large files**:
   - Use smaller zoom ranges: `--zoom 6-12`
   - Reduce batch size in configuration
   - Ensure sufficient RAM (4-8GB recommended)

3. **Slow processing**:
   - Ensure Pyosmium is properly installed
   - Check system resources (CPU, RAM)
   - Verify PBF file is not corrupted

### Performance Tips

- **Use SSD storage** for better I/O performance
- **Allocate sufficient RAM** for large datasets
- **Process in zoom ranges** for very large files
- **Monitor memory usage** during processing
- **Use appropriate simplification tolerances** for your use case

---

## Contributing

This project is actively maintained and optimized for performance. Contributions are welcome, especially:

- **Performance optimizations**: Memory usage, processing speed
- **New drawing commands**: Additional geometric primitives
- **Format improvements**: Better compression, encoding
- **Documentation**: Examples, tutorials, best practices

---

## License

This project is open source and available under the MIT License.

---

## Acknowledgments

- **OpenStreetMap**: For providing the open map data
- **Pyosmium**: For the excellent PBF processing library
- **Shapely**: For robust geometric operations
- **SQLite**: For efficient data storage and retrieval

---

*Last updated: December 2024*