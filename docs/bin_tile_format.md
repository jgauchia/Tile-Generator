# NAV-PACK Format Specification (v0.5.0)

This document describes the **NPK2** container format evolved with **Pure Hilbert Indexing**, a high-performance binary storage system for vector map tiles designed for ESP32-based GPS navigators.

NPK2-Hilbert optimizes SD card access by ordering tiles and indices along a space-filling curve, ensuring that geographically adjacent tiles are physically close in the binary file.

---

## 1. Storage Architecture

Map data is organized into consolidated binary files per zoom level: `Z{zoom}.nav`.

Each Pack file consists of: **Global Header** → **Hilbert Index Table** → **Tile Data Blocks**.

### 1.1. Key Features
- **Spatial Locality**: Data is ordered by Hilbert distance.
- **Binary Deduplication**: Multiple index entries can point to the same physical data block (e.g., empty ocean/land tiles).
- **Extensible Header**: Reserved fields for future metadata or checksums.

---

## 2. File Structure

### 2.1. Global Pack Header (37 bytes)

| Offset | Field          | Type      | Size  | Description                               |
|--------|----------------|-----------|--------|-------------------------------------------|
| 0      | magic          | bytes[4]  | 4      | "NPK2"                                    |
| 4      | zoom           | uint8     | 1      | Zoom level                                |
| 5      | tile_count     | uint32    | 4      | Total number of tiles (LE)                |
| 9      | index_offset   | uint32    | 4      | File offset to the Hilbert Index Table    |
| 13     | reserved[4]    | uint32[4] | 16     | Reserved for future use (alignment)       |

### 2.2. Hilbert Index Table (tile_count * 16 bytes)

The index table is sorted by the `h` (Hilbert Index) field to allow $O(\log N)$ binary search.

| Offset | Field      | Type   | Size | Description                              |
|--------|------------|--------|------|------------------------------------------|
| 0      | h          | uint64 | 8    | Hilbert distance index (calculated from x, y) |
| 8      | offset     | uint32 | 4    | Absolute byte offset of tile data        |
| 12     | size       | uint32 | 4    | Size of tile data in bytes               |

---

## 3. Internal Tile Format (NAV1)

Each tile block is byte-for-byte identical for deduplication if geometry matches.

### 3.1. Tile Header (22 bytes)

| Offset | Field         | Type     | Size | Description                          |
|--------|---------------|----------|------|--------------------------------------|
| 0      | magic         | bytes[4] | 4    | "NAV1"                               |
| 4      | feature_count | uint16   | 2    | Number of features                   |
| 6      | reserved      | uint8[16]| 16   | Set to 0 to enable binary deduplication |

---

## 4. Feature Records

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

## 5. Tile Lookup (ESP32 Algorithm)

1. **Calculate Hilbert Index**:
   Convert target `(x, y)` to Hilbert index `h` using the `xy_to_hilbert(x, y, zoom)` function.
2. **Binary Search**:
   Perform a binary search on the Index Table (starting at `index_offset`) to find the entry where `entry.h == h`.
3. **Fetch Data**:
   Read `entry.size` bytes starting at `entry.offset`.

```cpp
// Example C++ lookup logic
uint64_t h = xy_to_hilbert(targetX, targetY, zoom);
int low = 0, high = hdr.tile_count - 1;
while (low <= high) {
    int mid = low + (high - low) / 2;
    IndexEntry entry;
    f.seek(hdr.index_offset + mid * sizeof(IndexEntry));
    f.read((uint8_t*)&entry, sizeof(IndexEntry));
    if (entry.h == h) return entry; // Found!
    if (entry.h < h) low = mid + 1;
    else high = mid - 1;
}
```

---

## 6. Priority Levels (Z-Order)

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
