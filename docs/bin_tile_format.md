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

Each feature consists of a **13-byte header**, followed by coordinate data, and optional ring information for polygons.

#### Feature Header (13 bytes)

| Offset | Field            | Type   | Size | Description                                  |
|--------|------------------|--------|-------|----------------------------------------------|
| 0      | geom_type        | uint8  | 1     | 1=Point, 2=LineString, 3=Polygon             |
| 1      | color_rgb565     | uint16 | 2     | RGB565 color (Little-Endian)                 |
| 3      | zoom_priority    | uint8  | 1     | High nibble: min_zoom, Low nibble: priority  |
| 4      | width_pixels     | uint8  | 1     | Rendered line width in pixels (1-15)         |
| 5      | bbox             | uint8[4]| 4     | Object BBox [x1, y1, x2, y2] normalized 0-255|
| 9      | coord_count      | uint16 | 2     | Total number of points (across all rings)    |
| 11     | payload_size     | uint16 | 2     | Size of coordinates + ring data in bytes     |

#### Coordinate Data

Follows the header immediately. Coordinates are stored using **Delta Encoding** with **VarInt** and **ZigZag** compression.

- **Delta Encoding**: Each point $(x_n, y_n)$ is stored as a difference from the previous point: $dx = x_n - x_{n-1}$, $dy = y_n - y_{n-1}$. The first point of each feature uses $(0,0)$ as the reference.
- **ZigZag**: Signed integers (deltas) are mapped to unsigned integers ($0 \to 0, -1 \to 1, 1 \to 2, -2 \to 3 \dots$).
- **VarInt**: Unsigned integers are stored using a variable number of bytes (7 bits per byte + 1 continuation bit).

| Field            | Type   | Size | Description                                  |
|------------------|--------|-------|----------------------------------------------|
| coordinates[]    | VarInt[]| Variable | Pairs of (dx, dy) encoded as VarInts        |

#### Polygon Ring Information (Polygons Only)

Only present if `geom_type == 3`. Located at the end of the coordinate data block (within the `payload_size`).

| Field          | Type   | Size | Description                    |
|----------------|--------|-------|--------------------------------|
| ring_count     | uint16 | 2     | Total number of rings (exterior + holes) |
| ring_ends[]    | uint16 | 2×R   | Cumulative end index for each ring |

---

## Coordinate System

### Tile-Relative Projection
NAV stores coordinates already projected using the Web Mercator projection. Values are mapped to a 12-bit space (0-4096) relative to the tile's top-left corner.

- **Range**: The standard tile extent is 0-4096. To ensure seamless rendering across tiles, features are clipped with a **20% safety margin**, allowing coordinates to range from approximately -820 to 4916.
- **Precision**: 1 unit = 1/16th of a pixel (assuming a 256px tile).
- **Complexity**: While the format supports up to 65535 points per feature, the standard v0.4.0 generator limits features to **2000 points** to ensure stability on ESP32 hardware.

---

## Rendering Rules

### Line Width Damping
To prevent roads from obscuring the map at high detail levels, the generator applies a **damping factor of 0.7x** to the calculated pixel width for all features at **Zoom 13 and above**.

### Delta Reset Logic
- **Linestrings**: The delta accumulator (lastX, lastY) is reset to (0,0) at the start of each feature.
- **Polygons**: The delta accumulator is reset to (0,0) at the start of the feature **and at the start of every interior ring (hole)**. This ensures that errors don't propagate between separate geometric rings.

---

## Implementation Example (C++)

### 1. Decoding Helpers

```cpp
int32_t zigzag_decode(uint32_t n) {
    return (n >> 1) ^ -(int32_t)(n & 1);
}

uint32_t read_varint(uint8_t** ptr) {
    uint32_t result = 0;
    int shift = 0;
    while (true) {
        uint8_t byte = **ptr;
        (*ptr)++;
        result |= (byte & 0x7F) << shift;
        if (!(byte & 0x80)) return result;
        shift += 7;
    }
}
```

### 2. Rendering Loop

```cpp
NavFeatureHeader feat;
file.read(&feat, sizeof(feat));

// Fast BBox Culling
if (!viewport.intersects(feat.bbox)) {
    file.seek(feat.payload_size, SEEK_CUR); // Fast skip
    continue;
}

// Read payload into buffer
uint8_t* buffer = (uint8_t*)malloc(feat.payload_size);
file.read(buffer, feat.payload_size);
uint8_t* ptr = buffer;

int32_t last_x = 0, last_y = 0;
Point pts[feat.coord_count];

for (int j = 0; j < feat.coord_count; j++) {
    last_x += zigzag_decode(read_varint(&ptr));
    last_y += zigzag_decode(read_varint(&ptr));
    
    pts[j].x = (last_x * screen_tile_size) >> 12;
    pts[j].y = (last_y * screen_tile_size) >> 12;
}

// Render...
if (feat.geom_type == 3) {
    uint16_t ring_count = *((uint16_t*)ptr); ptr += 2;
    uint16_t* ring_ends = (uint16_t*)ptr;
    fillPolygon(pts, ring_count, ring_ends, feat.color);
}

free(buffer);
```
