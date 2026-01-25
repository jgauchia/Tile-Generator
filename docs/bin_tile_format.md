# NAV Tile Format Specification

This document describes the NAV binary format produced by the tile generation script (`tile_generator.py`) for vector map tiles. The format is optimized for ESP32 embedded systems with compact storage and efficient rendering of map data.

---

## File Overview

Each file represents a single tile for a given zoom level (`z`), x coordinate (`x`), and y coordinate (`y`).  
The filename and directory structure is:  
```
{output_dir}/{z}/{x}/{y}.nav
```

The file contains a collection of vector features with geometry data and rendering information.  
All coordinates are encoded as `int32` values scaled by `COORD_SCALE` (10,000,000) for ~1cm precision.

The format uses **self-contained feature records** with individual colors and coordinates for maximum compatibility with embedded rendering systems.

---

## File Structure

The file is a single binary blob with the following structure:

| Field          | Type      | Size  | Description                               |
|----------------|-----------|--------|-------------------------------------------|
| magic          | bytes[4]  | 4      | Format identifier ("NAV1")               |
| feature_count  | uint16    | 2      | Number of features in the tile              |
| min_lon        | int32     | 4      | Minimum longitude (scaled by COORD_SCALE) |
| min_lat        | int32     | 4      | Minimum latitude (scaled by COORD_SCALE)  |
| max_lon        | int32     | 4      | Maximum longitude (scaled by COORD_SCALE) |
| max_lat        | int32     | 4      | Maximum latitude (scaled by COORD_SCALE)  |
| features[]     | variable  | —      | Sequence of feature records               |

**Total Header Size**: 22 bytes

- All coordinates are scaled by `COORD_SCALE = 10,000,000` for ~1cm precision
- Coordinates are absolute geographic coordinates, not tile-relative
- All multi-byte values use little-endian byte order

---

## Feature Record Structure

Each feature contains a header followed by coordinate data:

| Field            | Type   | Size | Description                                  |
|------------------|--------|-------|----------------------------------------------|
| geom_type        | uint8  | 1     | Geometry type (Point/LineString/Polygon)   |
| color_rgb565     | uint16  | 2     | RGB565 color value                          |
| zoom_priority     | uint8  | 1     | Packed zoom and priority information        |
| width_pixels      | uint8  | 1     | Line width in pixels (NAV v2)             |
| coord_count      | uint16  | 2     | Number of coordinate pairs                    |
| coordinates[]    | int32[] | 8×N   | Longitude/latitude pairs for each point      |

**Feature Header Size**: 7 bytes + (8 × coord_count) bytes

---

## Geometry Types

| Type  | Value | Description                              |
|--------|-------|------------------------------------------|
| POINT      | 1      | Single point feature                      |
| LINESTRING | 2      | Polyline with multiple points             |
| POLYGON    | 3      | Closed polygon with outline or fill       |

---

## Color Encoding

### RGB565 Format
Colors are stored in 16-bit RGB565 format for efficient display on 16-bit systems:

| Component | Bits | Range   | Description                          |
|-----------|-------|--------|--------------------------------------|
| Red       | 5     | 0-31   | Red intensity                         |
| Green     | 6     | 0-63   | Green intensity (higher precision)      |
| Blue      | 5     | 0-31   | Blue intensity                        |

**Bit Layout**: `RRRRRGGGGGGBBBBB`

**Conversion to RGB888:**
```python
r = (rgb565 >> 11) & 0x1F
g = (rgb565 >> 5) & 0x3F  
b = rgb565 & 0x1F
r8 = (r << 3) | (r >> 2)
g8 = (g << 2) | (g >> 4)
b8 = (b << 3) | (b >> 2)
rgb888 = (r8 << 16) | (g8 << 8) | b8
```

---

## Zoom Priority Encoding

The `zoom_priority` byte combines minimum zoom level and rendering priority:

| Bits    | Range  | Description                           |
|----------|--------|---------------------------------------|
| 7-4      | 0-15   | Minimum zoom level (inclusive)         |
| 3-0      | 0-15   | Rendering priority (scaled by 7)     |

**Priority Calculation:**
```python
min_zoom = (zoom_priority >> 4) & 0x0F      # High 4 bits
priority = (zoom_priority & 0x0F) * 7        # Low 4 bits, scaled
```

---

## Width Calculation

### From OSM Width Tags
If a feature has a `width` tag (in meters), it's converted to pixels using Mercator projection:

```python
def meters_to_pixels(width_meters: float, zoom: int, lat: float = 45.0) -> int:
    """Convert width in meters to pixels at given zoom level.
    
    Uses approximation for given latitude (default 45° for Europe).
    Formula: meters_per_pixel ≈ 156543 * cos(lat) / 2^zoom
    """
    meters_per_pixel = 156543.0 * math.cos(math.radians(lat)) / (2 ** zoom)
    pixels = int(width_meters / meters_per_pixel + 0.5)
    return max(1, min(15, pixels))  # Clamp to 1-15
```

### Default Widths
When no width tag is present, defaults are applied:
- **Motorway/Trunk**: 3-15 pixels (zoom-dependent)
- **Primary**: 2-12 pixels
- **Secondary**: 2-10 pixels  
- **Residential**: 1-7 pixels
- **Other roads**: 1-5 pixels
- **Default**: 1 pixel

---

## Coordinate System

### Absolute Geographic Coordinates
All coordinates are stored as absolute geographic coordinates:

- **Scale Factor**: `COORD_SCALE = 10,000,000`
- **Precision**: ~1cm per unit
- **Range**: ±180° longitude, ±90° latitude
- **Format**: int32 (4 bytes per coordinate)

**Coordinate Conversion:**
```python
# Storing coordinates
lon_int = int(lon * COORD_SCALE)
lat_int = int(lat * COORD_SCALE)

# Reading coordinates  
lon = lon_int / COORD_SCALE
lat = lat_int / COORD_SCALE
```

### Tile Bounding Box
The header contains the minimum and maximum coordinates of all features in the tile:

```
min_lon, min_lat ------------ max_lon
       |                     |
       |   Tile Extent      |
       |                     |
max_lon, max_lat ------------ max_lon
```

---

## Polygon Ring Information

For polygon features, additional ring data follows the coordinates:

| Field          | Type   | Size | Description                    |
|----------------|--------|-------|--------------------------------|
| ring_count     | uint8  | 1     | Number of rings (always 1)    |
| ring_ends[]    | uint16 | 2×N   | End index of each ring          |

**Note**: Current implementation uses a single ring per polygon. Ring information is included but not used by most parsers.

---

## Geometry Simplification

The tile generator includes Douglas-Peucker simplification for zoom levels ≥ 16:

### Tolerance by Zoom Level
- **Zoom 16**: 0.000015 degrees (~1.7m)
- **Zoom 17**: 0.000012 degrees (~1.3m) 
- **Zoom 18**: 0.000010 degrees (~1.1m)
- **Zoom 19**: 0.000008 degrees (~0.9m)

### Simplification Rules
- Applied only to features with ≥50 vertices
- LineStrings with <10 points are not simplified
- Polygons maintain their closed ring structure
- Preserves topology and feature accuracy

---

## File Size Optimization

### Header Compression
- **Fixed 22-byte header** vs variable-length command headers
- **Bounding box** enables early tile rejection
- **Feature count** allows memory pre-allocation

### Data Compression
- **RGB565 format**: 50% reduction vs RGB888
- **Scaled coordinates**: 4 bytes per coordinate vs 8 bytes for double precision
- **No redundant data**: Each feature is self-contained

### Typical File Sizes
- **Urban tiles**: 2-15 KB (high feature density)
- **Rural tiles**: 0.5-3 KB (low feature density)  
- **Water tiles**: 0.1-0.5 KB (minimal features)

---

## ESP32 Optimization

### Memory Efficiency
- **Small tiles**: Typically <16KB fits easily in RAM
- **Streaming capability**: Features can be processed sequentially
- **Fixed record size**: Predictable memory allocation
- **No dynamic parsing**: Simple binary structure

### Rendering Performance
- **Priority-based**: Features pre-sorted by rendering order
- **Width pre-calculated**: No runtime calculations needed
- **RGB565 native**: Direct display without conversion
- **Coordinate scaling**: Pre-scaled for pixel-perfect rendering

---

## Implementation Guidelines

### Basic Parser Implementation
1. **Read header**: 22 bytes to get magic, feature count, and bbox
2. **Validate magic**: Check for "NAV1" identifier
3. **Allocate memory**: Pre-allocate based on feature count and bbox
4. **Process features**: Read each feature record sequentially
5. **Parse coordinates**: Apply COORD_SCALE to convert to geographic coordinates

### Error Handling
- **Invalid magic**: File is not NAV format
- **Corrupted data**: Handle gracefully with bounds checking
- **Coordinate overflow**: Clamp to valid geographic ranges
- **Feature count mismatch**: Validate against actual file size

### Rendering Integration
- **Sort by priority**: Use packed zoom_priority field
- **Convert colors**: RGB565 to display format
- **Scale coordinates**: Apply display scaling and projection
- **Width handling**: Use width_pixels for line thickness

---

## Compatibility

### Version Information
- **Current Version**: NAV v1 with width support (NAV v2 compatibility)
- **Backward Compatibility**: Readers can ignore width field
- **Forward Compatibility**: Additional fields can be added after existing structure

### File Extensions
- **Standard**: `.nav` for NAV binary tiles
- **Identifier**: "NAV1" magic bytes for format recognition

---

## Usage Examples

### Command Line Generation
```bash
python tile_generator.py input.osm.pbf output_directory features.json --zoom 6-17
```

### Expected Output Structure
```
output_directory/
├── 6/
│   ├── 0/
│   │   ├── 0.nav
│   │   ├── 1.nav
│   │   └── ...
│   ├── 1/
│   │   └── 0.nav
│   └── ...
├── 7/
│   └── ...
└── ...
```

### Integration with ESP32
The format is specifically designed for embedded navigation systems:
- **Direct memory mapping**: Can be loaded directly into ESP32 RAM
- **Efficient parsing**: Simple binary structure with minimal computation
- **Display-ready**: Colors and coordinates in native formats
- **Tile-based**: Supports incremental loading for large maps

---

## References

- **RGB565 Color Format**: 16-bit color encoding
- **Mercator Projection**: Geographic coordinate system
- **Douglas-Peucker Algorithm**: Line simplification technique
- **ESP32 Technical Specifications**: Memory and display constraints

---

## Notes

- This format is optimized for **embedded systems** with limited memory
- Coordinates use **absolute geographic positions**, not tile-relative coordinates
- Each feature is **self-contained** with its own color and geometry
- No compression beyond **data type optimization** (RGB565, scaled coordinates)
- The format prioritizes **simplicity and speed** over maximum compression