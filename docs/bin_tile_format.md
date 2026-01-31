# NAV Tile Format Specification

This document describes the optimized NAV binary format produced by `tile_generator.py`. The format is specifically designed for ultra-fast vector map rendering on ESP32 embedded systems by using pre-projected, compact relative coordinates and hardware-friendly data alignment.

---

## File Overview

Each file represents a single tile for a given zoom level (`z`), x coordinate (`x`), and y coordinate (`y`).  
The directory structure follows the standard XYZ tile scheme:  
```
{output_dir}/{z}/{x}/{y}.nav
```

The format uses **tile-relative coordinates** (mapped to a 0-4096 range) with a pre-applied Mercator projection.

---

## File Structure

The file is a single binary blob consisting of a global header followed by sequential feature records.

### 1. Tile Header (22 bytes)

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | Format identifier ("NAV1")               |
| 4      | feature_count  | uint16    | 2      | Number of features in the tile (LE)      |
| 6      | min_lon        | int32     | 4      | Tile min longitude (scaled 1e7)           |
| 10     | min_lat        | int32     | 4      | Tile min latitude (scaled 1e7)            |
| 14     | max_lon        | int32     | 4      | Tile max longitude (scaled 1e7)           |
| 18     | max_lat        | int32     | 4      | Tile max latitude (scaled 1e7)            |

### 2. Feature Records (Sequential)

Each feature consists of a **12-byte aligned header**, followed by coordinate data, and optional ring information for polygons.

#### Feature Header (12 bytes)

| Offset | Field            | Type   | Size | Description                                  |
|--------|------------------|--------|-------|----------------------------------------------|
| 0      | geom_type        | uint8  | 1     | 1=Point, 2=LineString, 3=Polygon             |
| 1      | color_rgb565     | uint16 | 2     | RGB565 color (Little-Endian)                 |
| 3      | zoom_priority    | uint8  | 1     | High nibble: min_zoom, Low nibble: priority  |
| 4      | width_pixels     | uint8  | 1     | Rendered line width in pixels (1-15)         |
| 5      | bbox             | uint8[4]| 4     | Object BBox [x1, y1, x2, y2] normalized 0-255|
| 9      | coord_count      | uint16 | 2     | Total number of points (across all rings)    |
| 11     | padding          | uint8  | 1     | Alignment padding (always 0x00)              |

#### Coordinate Data

Follows the header immediately. Each point is a pair of signed 16-bit integers.

| Field            | Type   | Size | Description                                  |
|------------------|--------|-------|----------------------------------------------|
| coordinates[]    | int16[]| 4×N   | x, y pairs relative to tile origin (0-4096)  |

#### Polygon Ring Information (Polygons Only)

Only present if `geom_type == 3`. Follows the coordinate data.

| Field          | Type   | Size | Description                    |
|----------------|--------|-------|--------------------------------|
| ring_count     | uint8  | 1     | Total number of rings (exterior + holes) |
| ring_ends[]    | uint16 | 2×R   | Cumulative end index for each ring |

---

## Coordinate System

### Tile-Relative Projection
NAV stores coordinates already projected using the Web Mercator projection. Values are mapped to a 12-bit space (0-4096) relative to the tile's top-left corner.

- **Range**: The standard tile extent is 0-4096. To ensure seamless rendering across tiles, features are clipped with a **10% safety margin**, allowing coordinates to range from approximately -410 to 4506.
- **Precision**: 1 unit = 1/16th of a pixel (assuming a 256px tile).
- **Format**: Signed `int16`.

### ESP32 Rendering Math
Conversion to screen pixels is extremely efficient:
```cpp
pixel_x = (nav_x * screen_tile_size) >> 12;
pixel_y = (nav_y * screen_tile_size) >> 12;
```

---

## Optimizations

### BBox Culling
The 4-byte `bbox` field stores normalized extents (`tile_coord >> 4`). The renderer can reconstruct the pixel bounds (`bbox[i] << 4`) and skip features that do not intersect the current viewport before reading any coordinate data.

### Geometry Clipping
All features are clipped using `shapely` during generation. This reduces the number of points processed by the ESP32 and ensures that only relevant geometry is loaded from the SD card.

### Minimum Area Filter
Polygons whose bounding box area is smaller than 4 pixels squared at the target zoom level are discarded. This significantly reduces file size and visual noise in dense areas.

### Data Alignment
The 12-byte feature header ensures efficient memory access and simplified struct mapping in C/C++ environments.

---

## Color & Priority Encoding

### RGB565
Colors are stored in the native 16-bit format used by most ESP32 displays (`RRRRRGGGGGGBBBBB`), allowing direct buffer injection.

### Zoom Priority
Packed into a single byte:
- `min_zoom`: `zoom_priority >> 4` (High 4 bits).
- `priority`: `zoom_priority & 0x0F` (Low 4 bits). 
Features are pre-sorted by this priority during generation to ensure correct draw order.

---

## Implementation Example (C++)

### 1. Data Structures

The 12-byte aligned header allows direct mapping to C structs:

```cpp
#pragma pack(push, 1)
struct NavTileHeader {
    char magic[4];          // "NAV1"
    uint16_t feature_count; // Little-Endian
    int32_t min_lon;        // scaled 1e7
    int32_t min_lat;
    int32_t max_lon;
    int32_t max_lat;
};

struct NavFeatureHeader {
    uint8_t geom_type;      // 1=Point, 2=LineString, 3=Polygon
    uint16_t color_rgb565;  // Little-Endian
    uint8_t zoom_priority;  // High: min_zoom, Low: priority
    uint8_t width_pixels;   // 1-15
    uint8_t bbox[4];        // [x1, y1, x2, y2] normalized 0-255
    uint16_t coord_count;   // Total points
    uint8_t padding;        // Always 0x00
};
#pragma pack(pop)
```

### 2. Rendering Pseudo-code

```cpp
// 1. Read Tile Header
NavTileHeader tile;
file.read(&tile, sizeof(tile));

// 2. Iterate Features
for (int i = 0; i < tile.feature_count; i++) {
    NavFeatureHeader feat;
    file.read(&feat, sizeof(feat));

    // Fast BBox Culling
    // Reconstruct tile coords (0-4096) from normalized bbox (0-255)
    Rect object_rect = { feat.bbox[0] << 4, feat.bbox[1] << 4, 
                         feat.bbox[2] << 4, feat.bbox[3] << 4 };
                         
    if (!viewport.intersects(object_rect)) {
        file.seek(feat.coord_count * 4, SEEK_CUR); // Skip points
        if (feat.geom_type == 3) {
            uint8_t rings = file.readUint8();
            file.seek(rings * 2, SEEK_CUR); // Skip ring ends
        }
        continue;
    }

    // Read and Project Coordinates
    Point pts[feat.coord_count];
    for (int j = 0; j < feat.coord_count; j++) {
        int16_t nx, ny;
        file.read(&nx, 2); file.read(&ny, 2);
        // Zero-CPU projection using bit-shifts
        pts[j].x = (nx * screen_tile_size) >> 12;
        pts[j].y = (ny * screen_tile_size) >> 12;
    }

    // Render Geometry
    if (feat.geom_type == 2) {
        drawLines(pts, feat.coord_count, feat.width_pixels, feat.color_rgb565);
    } 
    else if (feat.geom_type == 3) {
        uint8_t ring_count = file.readUint8();
        uint16_t ring_ends[ring_count];
        file.read(ring_ends, ring_count * 2);
        fillPolygon(pts, ring_count, ring_ends, feat.color_rgb565);
    }
}
```
