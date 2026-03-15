# NAV-PACK Format Specification (v2)

This document describes the **NPK2** container format, a high-performance binary storage system for vector map tiles designed for ESP32-based GPS navigators.

NPK2 eliminates the overhead of traditional file systems on SD cards by consolidating tiles into single files per zoom level, with a Y-table index for O(1) row lookup optimized for 20MHz SPI access.

---

## 1. Storage Architecture

Map data is organized into consolidated binary files, one for each zoom level:
`Z{zoom}.nav` (e.g., `Z14.nav`, `Z17.nav`).

Each Pack file consists of: **Tile Data** → **Y-Table** → **Index Table** → **Global Header** (written last).

---

## 2. File Structure

### 2.1. Global Pack Header (21 bytes)

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | "NPK2" (Nav Pack version 2)               |
| 4      | zoom           | uint8     | 1      | Zoom level of all tiles in this pack       |
| 5      | tile_count     | uint32    | 4      | Total number of tiles (LE)                 |
| 9      | y_min          | uint32    | 4      | Minimum Y tile coordinate                  |
| 13     | y_max          | uint32    | 4      | Maximum Y tile coordinate                  |
| 17     | ytable_offset  | uint32    | 4      | File offset to Y-table                     |

The header is located at offset 0 of the file.

### 2.2. Y-Table ((y_max - y_min + 1) * 8 bytes)

The Y-table provides O(1) lookup by tile Y coordinate. Each entry maps a Y row to its range in the index table.

| Offset | Field      | Type   | Size | Description                              |
|--------|------------|--------|------|------------------------------------------|
| 0      | idx_start  | uint32 | 4    | First index entry for this Y row         |
| 4      | idx_count  | uint32 | 4    | Number of tiles in this Y row            |

To find a tile at (x, y):
1. Read `ytable[y - y_min]` to get `idx_start` and `idx_count`
2. Binary search within `idx_count` index entries starting at `idx_start`

### 2.3. Index Table (tile_count * 16 bytes)

Entries are sorted by Y then X (row-major).

| Offset | Field  | Type   | Size | Description                          |
|--------|--------|--------|------|--------------------------------------|
| 0      | x      | uint32 | 4    | Tile X coordinate (LE)               |
| 4      | y      | uint32 | 4    | Tile Y coordinate (LE)               |
| 8      | offset | uint32 | 4    | Absolute byte offset of tile data    |
| 12     | size   | uint32 | 4    | Size of tile data in bytes           |

---

## 3. Internal Tile Format (NAV1)

Each tile in the data block contains:

### 3.1. Tile Header (22 bytes)

| Offset | Field         | Type     | Size | Description                          |
|--------|---------------|----------|------|--------------------------------------|
| 0      | magic         | bytes[4] | 4    | "NAV1"                               |
| 4      | feature_count | uint16   | 2    | Number of features                   |
| 6      | min_lon       | int32    | 4    | Tile min longitude (scaled 1e7)      |
| 10     | min_lat       | int32    | 4    | Tile min latitude (scaled 1e7)       |
| 14     | max_lon       | int32    | 4    | Tile max longitude (scaled 1e7)      |
| 18     | max_lat       | int32    | 4    | Tile max latitude (scaled 1e7)       |

### 3.2. Feature Records

Each feature has a **13-byte header** followed by compressed coordinates or text payload.

| Offset | Field         | Type   | Size | Description                                        |
|--------|---------------|--------|------|----------------------------------------------------|
| 0      | geom_type     | uint8  | 1    | 1=Point, 2=Line, 3=Polygon, 4=Text                |
| 1      | color         | uint16 | 2    | Color in RGB565 (LE)                               |
| 3      | zoom_priority | uint8  | 1    | `(min_zoom << 4) \| (priority & 0x0F)`            |
| 4      | width_flags   | uint8  | 1    | Bit 7=Casing, Bits 0-6=Width (0.5px units)        |
| 5      | min_x         | uint8  | 1    | BBox min X (coords/16)                             |
| 6      | min_y         | uint8  | 1    | BBox min Y (coords/16)                             |
| 7      | max_x         | uint8  | 1    | BBox max X (coords/16)                             |
| 8      | max_y         | uint8  | 1    | BBox max Y (coords/16)                             |
| 9      | coord_count   | uint16 | 2    | Number of vertices (or words for text)             |
| 11     | payload_size  | uint16 | 2    | Total bytes of data following the header           |

---

## 4. Coordinate System & Compression

- **Projection**: Web Mercator mapped to 12-bit tile-relative space (0-4096).
- **Clipping Margin**: Polygons 10%, lines 100% — ensures seamless tile edges.
- **Encoding**: Delta VarInt/ZigZag. Accumulators reset at the start of each feature and each polygon ring.
- **Text Encoding**: Anchor coordinates (int16), length, and UTF-8 string.

---

## 5. Rendering Pipeline

Four-pass rendering for road casings and layered text:

1. **Pass 1**: Polygons and at-grade lines (casing=0).
2. **Pass 2**: Bridge/Tunnel casings (casing=1, darkened color, width+1px).
3. **Pass 3**: Bridge/Tunnel cores (casing=1, original color, width).
4. **Pass 4**: Text labels.

---

## 6. Tile Lookup (ESP32)

```cpp
struct Npk2Header {
    char magic[4];       // "NPK2"
    uint8_t zoom;
    uint32_t tile_count;
    uint32_t y_min, y_max;
    uint32_t ytable_offset;
};

struct YTableEntry {
    uint32_t idx_start;
    uint32_t idx_count;
};

struct IndexEntry {
    uint32_t x, y, offset, size;
};

// Y-table loaded in PSRAM at pack open time
// Lookup: O(1) Y-table access + O(log N) binary search within row
bool findTile(File& f, YTableEntry* ytable, Npk2Header& hdr,
              uint32_t x, uint32_t y, IndexEntry& out)
{
    if (y < hdr.y_min || y > hdr.y_max) return false;
    YTableEntry& ye = ytable[y - hdr.y_min];
    if (ye.idx_count == 0) return false;

    // Index offset = after header (21 bytes) + tile data
    uint32_t base = hdr.ytable_offset
                  + (hdr.y_max - hdr.y_min + 1) * sizeof(YTableEntry)
                  + ye.idx_start * sizeof(IndexEntry);

    int low = 0, high = (int)ye.idx_count - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        IndexEntry entry;
        f.seek(base + mid * sizeof(IndexEntry));
        f.read((uint8_t*)&entry, sizeof(IndexEntry));

        if (entry.x == x && entry.y == y) {
            out = entry;
            return true;
        }
        if (entry.x < x) low = mid + 1;
        else high = mid - 1;
    }
    return false;
}
```

---

## 7. Priority Levels (Z-Order)

```
 0: Background land
 1: Large zones (aerodrome, residential, commercial, industrial)
 2: Landuse base (parking, heath, scrub), islands
 3: Boundaries, retail, cemetery, leisure
 4: Surfaces (pitch, beach, wetland, farmland)
 5: Forest/wood, apron
 6: Infrastructure (runway, taxiway, helipad)
 7: Buildings, water
 8-14: Roads (by class)
15: Rail, bridges, labels
```
