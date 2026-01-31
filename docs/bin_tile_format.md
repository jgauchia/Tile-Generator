# NAV Tile Format Specification

This document describes the optimized NAV binary format produced by `tile_generator.py`. The format is specifically designed for ultra-fast vector map rendering on ESP32 embedded systems by pre-calculating projections and using compact relative coordinates.

---

## File Overview

Each file represents a single tile for a given zoom level (`z`), x coordinate (`x`), and y coordinate (`y`).  
The filename and directory structure is:  
```
{output_dir}/{z}/{x}/{y}.nav
```

The format uses **tile-relative coordinates** (0-4096 range) with a pre-applied Mercator projection, allowing the ESP32 to draw features using simple bit-shifts and additions.

---

## File Structure

The file is a single binary blob with the following structure:

### 1. Tile Header (22 bytes)

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | Format identifier ("NAV1")               |
| 4      | feature_count  | uint16    | 2      | Number of features in the tile (Little-Endian) |
| 6      | min_lon        | int32     | 4      | Tile min longitude (scaled 1e7)           |
| 10     | min_lat        | int32     | 4      | Tile min latitude (scaled 1e7)            |
| 14     | max_lon        | int32     | 4      | Tile max longitude (scaled 1e7)           |
| 18     | max_lat        | int32     | 4      | Tile max latitude (scaled 1e7)            |

### 2. Feature Records (Sequential)

Each feature contains an 11-byte header followed by coordinate data:

| Field            | Type   | Size | Description                                  |
|------------------|--------|-------|----------------------------------------------|
| geom_type        | uint8  | 1     | 1=Point, 2=LineString, 3=Polygon             |
| color_rgb565     | uint16 | 2     | RGB565 color (Little-Endian)                 |
| zoom_priority    | uint8  | 1     | High nibble: min_zoom, Low nibble: priority  |
| width_pixels     | uint8  | 1     | Line width in pixels (1-15)                  |
| bbox             | uint8[4]| 4     | Object BBox [x1, y1, x2, y2] normalized 0-255|
| coord_count      | uint16 | 2     | Number of coordinate pairs                   |
| coordinates[]    | int16[]| 4×N   | x, y pairs relative to tile origin (0-4096)  |

**Feature Header Size**: 11 bytes + (4 × coord_count) bytes

---

## Coordinate System

### Pre-Calculated Projection
Unlike standard formats, NAV stores coordinates **already projected** using the Web Mercator projection. The values are mapped to a 12-bit space (0-4096) relative to the tile's top-left corner.

- **Range**: Coordinates usually fall between 0-4096, but values outside this range (e.g., -128 to 4224) are allowed to ensure seamless rendering across tile borders.
- **Format**: Signed `int16` (2 bytes per component).

### ESP32 Rendering Math
To convert a NAV coordinate to a screen pixel, the ESP32 only needs to perform:
```cpp
pixel_x = (nav_x * screen_tile_size) >> 12;
pixel_y = (nav_y * screen_tile_size) >> 12;
```
This eliminates all floating-point math and trigonometric functions during rendering.

---

## Object BBox Culling

The 4-byte `bbox` field in the feature header allows for ultra-fast visibility checks before processing any points:
- The coordinates `x1, y1, x2, y2` are the object's extent normalized to 0-255 (by shifting the 12-bit tile coordinates right by 4).
- Reconstruction: `extent_px = bbox_val << 4`.
- This allows the renderer to skip entire objects that are outside the current viewport with a single comparison.

---

## Geometry Types

| Type  | Value | Description                              |
|--------|-------|------------------------------------------|
| POINT      | 1      | Single point feature                      |
| LINESTRING | 2      | Polyline with multiple points             |
| POLYGON    | 3      | Closed polygon. Followed by Ring info.    |

### Polygon Ring Information
For polygons, the following data follows the coordinate array:
| Field          | Type   | Size | Description                    |
|----------------|--------|-------|--------------------------------|
| ring_count     | uint8  | 1     | Number of rings (default 1)    |
| ring_ends[]    | uint16 | 2×R   | End index of each ring          |

---

## ESP32 Optimization Summary

1.  **Memory**: `int16` coordinates reduce the memory footprint by 50% compared to `int32`.
2.  **CPU**: No projection math required. All features are display-ready.
3.  **IO**: Sequential structure and small tile headers are optimized for SD card streaming.
4.  **Culling**: Header-based BBox allows discarding hidden features without reading their points.

---

## Versioning & Compatibility

- **Identifier**: `NAV1` magic bytes.
- **Current Version**: Optimized relative format (Jan 2026 update).
- **Note**: This version is NOT backward compatible with readers expecting absolute `int32` coordinates.

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