# NAV-PACK Format Specification (v1)

This document describes the **NAV-PACK** container format, a high-performance binary storage system for vector map tiles designed specifically for the IceNav ESP32 navigator. 

NAV-PACK eliminates the overhead of traditional file systems on SD cards by consolidating millions of tiles into single files per zoom level, optimized for 20MHz SPI access and sub-millisecond lookups.

---

## 1. Storage Architecture

Map data is organized into consolidated binary files, one for each zoom level:  
`Z{zoom}.nav` (e.g., `Z14.nav`, `Z17.nav`).

Each Pack file consists of a **Global Header**, a **Fixed-Size Index Table**, and the **Tile Data Block**.

---

## 2. File Structure

### 2.1. Global Pack Header (9 bytes)

The header identifies the file and provides basic metadata.

| Offset | Field          | Type      | Size  | Value / Description                       |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | "NPK1" (Nav Pack version 1)               |
| 4      | zoom           | uint8     | 1      | The zoom level of all tiles in this pack |
| 5      | tile_count     | uint32    | 4      | Total number of tiles in the pack (LE)   |

### 2.2. Index Table (tile_count * 16 bytes)

The Index Table allows the renderer to find the exact byte offset of any tile in $O(\log N)$ time using binary search. Entries are **sorted by Y then X coordinate** (Row-Major) to optimize spatial locality and scroll performance.

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | tileX          | uint32    | 4      | Tile X coordinate (LE)                   |
| 4      | tileY          | uint32    | 4      | Tile Y coordinate (LE)                   |
| 8      | offset         | uint32    | 4      | Absolute byte offset of the tile data    |
| 12     | size           | uint32    | 4      | Size of the tile data in bytes           |

---

## 3. Internal Tile Format (NAV1)

Each entry in the Data Block starting at `offset` contains a single tile with the following structure:

### 3.1. Tile Header (22 bytes)

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | "NAV1" (Internal tile identifier)        |
| 4      | feature_count  | uint16    | 2      | Number of features in the tile           |
| 6      | min_lon        | int32     | 4      | Tile min longitude (scaled 1e7)           |
| 10     | min_lat        | int32     | 4      | Tile min latitude (scaled 1e7)            |
| 14     | max_lon        | int32     | 4      | Tile max longitude (scaled 1e7)           |
| 18     | max_lat        | int32     | 4      | Tile max latitude (scaled 1e7)            |

### 3.2. Feature Records

Features are stored sequentially. Each feature has a **13-byte header** followed by compressed coordinates.

---

## 4. Coordinate System & Compression

- **Projection**: Web Mercator coordinates mapped to a 12-bit tile-relative space (0-4096).
- **Safety Margin**: 20% clipping margin (approx. -820 to 4916) ensures seamless connections between tiles.
- **Encoding**: **Delta VarInt/ZigZag**. Each point is stored as a difference from the previous one. Accumulators are reset to (0,0) at the start of each feature and each polygon ring.

---

## 5. Implementation Example (C++)

The following code demonstrates how to perform an efficient binary search to find a tile offset without loading the entire index into RAM.

```cpp
struct IndexEntry {
    uint32_t x, y, offset, size;
};

// Returns offset and size of tile (x, y) from an open Pack file
bool findTile(FILE* f, uint32_t x, uint32_t y, uint32_t& out_offset, uint32_t& out_size) {
    uint32_t tileCount;
    fseek(f, 5, SEEK_SET);
    fread(&tileCount, 4, 1, f);

    int low = 0, high = tileCount - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        IndexEntry entry;
        fseek(f, 9 + (mid * 16), SEEK_SET);
        fread(&entry, 16, 1, f);

        if (entry.x == x && entry.y == y) {
            out_offset = entry.offset;
            out_size = entry.size;
            return true;
        }
        // Row-Major sort: Y is primary, X is secondary
        if (entry.y < y || (entry.y == y && entry.x < x)) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return false;
}
```
