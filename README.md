# OSM Vector Tile Generator

This repository contains a highly optimized Python script for generating vector map tiles from OpenStreetMap (OSM) data using a custom binary format.

The generated tiles are extremely compact and optimized for fast rendering in custom map applications, featuring advanced compression techniques, dynamic color palette optimization, memory pooling, streaming database operations, and intelligent resource allocation.

---

## What Does the Script Do?

- **Direct PBF processing** using Pyosmium for maximum performance (no temporary files)
- **Dynamic resource allocation** based on 70% of total system memory
- **Memory pooling** for object reuse and reduced garbage collection
- **Streaming database operations** with LZ4 compression for optimal I/O
- **Batch geometry processing** for improved performance
- **Real-time progress tracking** with feature counts and processing statistics
- **Intelligent worker allocation** based on system capabilities and memory pressure
- **Generates compact binary tiles** with efficient coordinate encoding and palette optimization

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
- **Dynamic resource allocation**: Automatically adjusts workers and batch sizes based on system memory
- **Memory pooling**: Reuses objects to minimize garbage collection overhead
- **LZ4 compression**: Reduces database storage and I/O operations
- **Streaming operations**: Processes data in chunks to maintain constant memory usage
- **Real-time monitoring**: Tracks memory pressure and adjusts processing parameters dynamically

---

## Script Features

### Advanced Optimization Suite
- **Database optimization**: Composite indexes, WAL mode, LZ4 compression, streaming operations
- **Memory pooling**: Object reuse for points, commands, coordinates, and features
- **Dynamic resource allocation**: Automatic adjustment based on 70% of total system memory
- **Batch processing**: Intelligent grouping of geometries by type for efficient processing
- **Memory pressure monitoring**: Real-time adjustment of workers and batch sizes
- **Streaming I/O**: Processes data in chunks to maintain constant memory usage

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
The script provides detailed processing reports with real-time progress tracking:
```
Validating input parameters...
All input parameters validated successfully
System information:
  CPU cores: 8
  Max workers: 8
  DB batch size: 210,000
  Tile batch size: 70,000
  Total memory: 16384.0MB
  Allocated memory: 11468.8MB (70% of total)
  Worker memory limit: 1433.6MB per worker
  Memory pressure: low

Analyzing colors from features.json to build dynamic palette
Dynamic color palette created:
  - Total unique colors: 39
  - Palette indices: 0-38
  - Memory saving potential: 39 colors -> compact indices
Dynamic palette ready with 39 colors from your features.json
Writing palette to tiles/palette.bin (39 colors)
Palette written successfully

Processing PBF directly to database using Pyosmium (maximum performance)
Config requires these fields: amenity, building, highway, landuse, leisure, natural, place, railway, waterway
Compiled 79 tag patterns for Pyosmium processing
Starting Pyosmium processing...
📊 Progress: 125,430 scanned, 54,482 filtered, 1,250 features/s, 100.3s elapsed
Pyosmium processing completed in 100.3s
Total processed: 54482 features directly from PBF

┌──────┬─────────┬─────────────┐
│ Zoom │ Features│ Description │
├──────┼─────────┼─────────────┤
│    6 │     870 │ Level 6     │
│    7 │     870 │ Level 7     │
│    8 │   1,627 │ Level 8     │
│    9 │   1,648 │ Level 9     │
│   10 │   1,649 │ Level 10    │
│   11 │   1,661 │ Level 11    │
│   12 │   5,483 │ Level 12    │
│   13 │   5,774 │ Level 13    │
│   14 │   9,482 │ Level 14    │
│   15 │  25,418 │ Level 15    │
│TOTAL │  52,872 │ All levels  │
└──────┴─────────┴─────────────┘

Processing zoom level 6 from database
Found 1 tiles for zoom 6
Writing tiles (zoom 6): 100%|█████████████████████| 1/1 [00:00<00:00,  1.61s/it]

┌──────┬───────┬─────────────────┐
│ Zoom │ Tiles │ Avg Size (bytes)│
├──────┼───────┼─────────────────┤
│    6 │     1 │         3,778.00│
│    7 │     1 │         3,778.00│
│    8 │     4 │         4,125.50│
│    9 │     4 │         4,125.50│
│   10 │     4 │         4,125.50│
│   11 │     4 │         4,125.50│
│   12 │    16 │         4,250.00│
│   13 │    16 │         4,250.00│
│   14 │    25 │         4,300.00│
│   15 │    64 │         4,500.00│
│TOTAL │   139 │               - │
└──────┴───────┴─────────────────┘

Process completed successfully
Cleaning up temporary files and database...
Cleaned up database file: features.db

⏱️  Total execution time: 00d 00:02:45
```

### Performance Monitoring
- **Real-time progress tracking**: Shows scanned/filtered features with processing speed
- **Memory pressure monitoring**: Automatically adjusts workers and batch sizes
- **System resource display**: Shows CPU cores, memory allocation, and worker limits
- **Tabulated statistics**: Clean tables for features per zoom and tile generation stats
- **Execution time tracking**: Total time displayed in dd hh:mm:ss format

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

- **Dynamic resource allocation**: Uses 70% of total system memory for optimal performance
- **Memory pooling**: Reuses objects (points, commands, coordinates, features) to reduce GC overhead
- **LZ4 compression**: Reduces database storage footprint and I/O operations
- **Streaming operations**: Processes data in chunks to maintain constant memory usage
- **Memory pressure monitoring**: Real-time adjustment of workers and batch sizes based on available memory

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
  "highway=primary": {
    "color": "#DC143C", 
    "priority": 6,
    "zoom": 8
  },
  "building": {
    "color": "#8B4513",
    "priority": 5, 
    "zoom": 12
  },
  "waterway=river": {
    "color": "#4169E1",
    "priority": 3,
    "zoom": 6
  }
}
```

**Configuration Features:**
- **Feature definitions**: Colors, priorities, and zoom levels for OSM features
- **Dynamic palette generation**: Colors automatically indexed into optimal palette
- **Priority-based rendering**: Features rendered in order based on priority values
- **Zoom-based filtering**: Features only appear at their minimum zoom level and above
- **Tag pattern matching**: Supports both exact matches (`key=value`) and key-only matches

---

## Dependencies

### Required Packages
```
shapely
tqdm
osmium
tabulate
lz4
psutil
```

### System Requirements
- **Pyosmium**: High-performance OSM PBF processing library
- **Multi-core CPU**: Script utilizes all available cores for optimal performance
- **RAM**: Streaming architecture works with modest RAM (4-8GB recommended for planet files)

### Installation
```bash
# Python packages
pip install shapely tqdm osmium tabulate lz4 psutil

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
- **Dynamic resource allocation**: Automatically adjusts based on 70% of total system memory
- **Memory pooling**: Reuses objects to minimize garbage collection overhead
- **LZ4 compression**: Reduces database storage and improves I/O performance
- **Batch geometry processing**: Groups similar geometries for efficient processing
- **Memory pressure monitoring**: Real-time adjustment of workers and batch sizes
- **Streaming operations**: Processes data in chunks to maintain constant memory usage

### Performance Monitoring
The script provides comprehensive monitoring:
- **Real-time progress tracking**: Shows scanned/filtered features with processing speed
- **Memory pressure monitoring**: Automatically adjusts workers and batch sizes
- **System resource display**: Shows CPU cores, memory allocation, and worker limits
- **Tabulated statistics**: Clean tables for features per zoom and tile generation stats
- **Execution time tracking**: Total time displayed in dd hh:mm:ss format

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

- [Binary Tile File Format](docs/bin_tile_format.md) - Complete specification of drawing commands
- [Features JSON Format](docs/features_json_format.md) - Configuration format details
- [Tile Viewer Documentation](docs/tile_viewer.md) - Tile viewer with command support

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
- **RGB888 palette format**: 3-byte colors with 4-byte header for tile viewer compatibility

### Memory Pool Optimization
- **Object reuse**: Reuses points, commands, coordinates, and features to reduce GC overhead
- **Memory pooling**: Maintains pools of reusable objects with configurable size limits
- **Periodic cleanup**: Clears pools every 100 tiles to prevent memory buildup
- **Efficient allocation**: Reduces object creation and destruction overhead

### Database Optimization
- **LZ4 compression**: Compresses feature data to reduce storage and I/O operations
- **Composite indexes**: Multi-column indexes for optimal query performance
- **WAL mode**: Write-Ahead Logging for better concurrency and crash recovery
- **Streaming operations**: Generator-based data retrieval for memory efficiency
- **Batch operations**: Efficient bulk inserts and updates with executemany

### Dynamic Resource Allocation
- **Memory-based calculation**: Uses 70% of total system memory for optimal performance
- **Adaptive workers**: Automatically adjusts worker count based on available memory
- **Dynamic batch sizes**: Calculates optimal batch sizes based on system resources
- **Memory pressure monitoring**: Real-time adjustment of processing parameters

### Batch Geometry Processing
- **Type-based grouping**: Groups polygons, linestrings, and other geometries separately
- **Efficient processing**: Processes similar geometries in batches for better performance
- **Memory optimization**: Reduces function call overhead and improves cache locality
- **Layer ordering**: Maintains correct rendering order within each batch

---

## Recent Updates

### Memory Pool Optimization
- **Object reuse system**: Reuses points, commands, coordinates, and features to reduce GC overhead
- **Configurable pool sizes**: Adjustable pool limits based on system memory
- **Periodic cleanup**: Automatic pool clearing every 100 tiles to prevent memory buildup
- **Performance improvement**: Significant reduction in object allocation overhead

### Dynamic Resource Allocation
- **Memory-based calculation**: Uses 70% of total system memory for optimal performance
- **Adaptive worker allocation**: Automatically adjusts workers based on available memory
- **Dynamic batch sizing**: Calculates optimal batch sizes based on system resources
- **Memory pressure monitoring**: Real-time adjustment of processing parameters

### LZ4 Database Compression
- **Feature data compression**: Compresses stored feature data using LZ4 for better I/O
- **Storage optimization**: Reduces database file size and improves read/write performance
- **Transparent compression**: Automatic compression/decompression during database operations
- **Memory efficiency**: Reduces memory usage during database operations

### Batch Geometry Processing
- **Type-based grouping**: Groups similar geometries for efficient processing
- **Memory optimization**: Reduces function call overhead and improves cache locality
- **Layer ordering**: Maintains correct rendering order within each batch
- **Performance improvement**: Faster processing of large numbers of geometries

---

## Performance Comparison

### Before Optimization
- **Memory usage**: High due to object allocation overhead
- **Database I/O**: Uncompressed data storage and retrieval
- **Processing**: Individual geometry processing without batching
- **Resource allocation**: Fixed worker and batch sizes

### After Optimization
- **Memory usage**: Reduced through object pooling and reuse
- **Database I/O**: LZ4 compression reduces storage and improves performance
- **Processing**: Batch geometry processing for improved efficiency
- **Resource allocation**: Dynamic adjustment based on 70% of system memory

---

## Troubleshooting

### Common Issues

1. **Pyosmium not available**:
   ```bash
   pip install osmium
   sudo apt-get install libosmium-dev python3-dev  # Ubuntu/Debian
   ```

2. **Memory issues with large files**:
   - The script automatically adjusts based on available memory
   - Use smaller zoom ranges: `--zoom 6-12`
   - Ensure sufficient RAM (4-8GB recommended)

3. **Slow processing**:
   - Ensure Pyosmium is properly installed
   - Check system resources (CPU, RAM)
   - Verify PBF file is not corrupted
   - The script automatically optimizes based on system capabilities

### Performance Tips

- **Use SSD storage** for better I/O performance
- **The script automatically optimizes** based on available system resources
- **Process in zoom ranges** for very large files if needed
- **Memory monitoring is automatic** - the script adjusts parameters dynamically
- **System requirements**: 4-8GB RAM recommended for planet files

---

## Contributing

This project is actively maintained and optimized for performance. Contributions are welcome, especially:

- **Memory optimizations**: Further improvements to memory pooling and resource allocation
- **New drawing commands**: Additional geometric primitives for tile rendering
- **Database optimizations**: Better compression, indexing, or query performance
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
- **LZ4**: For fast compression and decompression
- **Tabulate**: For beautiful table formatting in console output

---

