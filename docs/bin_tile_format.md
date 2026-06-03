# NAV-PACK Format Specification (NPK2)

This document describes the **NPK2** container format used by the NAV tile generator.
NPK2 uses a flat 2D array index over a rectangular bounding box, providing **O(1)**
tile lookup: one seek to the index entry, one read of 8 bytes, no search required.

---

## 1. Storage Architecture

Map data is organized into one binary file per zoom level: `Z{zoom}.nav`.

Each file consists of: **Map Header** → **Flat Index Table** → **Tile Data Blocks**.

The index is a contiguous array of `tiles_wide × tiles_high` entries stored row-major
(Y outer, X inner). The position of a tile `(x, y)` in the index is:

```
flat_index = (y - bottom_left[1]) * tiles_wide + (x - bottom_left[0])
```

Empty slots (gaps inside the bounding box) have `offset = 0` and `size = 0`.

---

## 2. File Structure

### 2.1. Map Header (21 bytes, `#pragma pack(push,1)`)

| Offset | Field          | Type     | Size | Description                           |
|--------|----------------|----------|------|---------------------------------------|
| 0      | magic          | bytes[4] | 4    | `"NPK2"`                              |
| 4      | zoom           | uint8    | 1    | Zoom level                            |
| 5      | tiles_wide     | uint32   | 4    | Width of the bounding box in tiles    |
| 9      | tiles_high     | uint32   | 4    | Height of the bounding box in tiles   |
| 13     | bottom_left[0] | uint32   | 4    | Absolute tile X of origin (min_x)     |
| 17     | bottom_left[1] | uint32   | 4    | Absolute tile Y of origin (min_y)     |

### 2.2. Flat Index Table (`tiles_wide × tiles_high × 8` bytes)

| Offset | Field  | Type   | Size | Description                         |
|--------|--------|--------|------|-------------------------------------|
| 0      | offset | uint32 | 4    | Absolute byte offset of tile data   |
| 4      | size   | uint32 | 4    | Size of tile data in bytes (0=empty)|

---

## 3. Internal Tile Format (NAV1)

### 3.1. Tile Header (22 bytes)

| Offset | Field         | Type      | Size | Description        |
|--------|---------------|-----------|------|--------------------|
| 0      | magic         | bytes[4]  | 4    | `"NAV1"`           |
| 4      | feature_count | uint16    | 2    | Number of features |
| 6      | reserved      | uint8[16] | 16   | Set to 0           |

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

## 5. Tile Lookup Algorithm (O(1))

```cpp
// 1. Read MapHeader at file offset 0
MapHeader hdr;
file.read(&hdr, sizeof(MapHeader));

// 2. Compute relative offsets
int32_t x_off = target_x - (int32_t)hdr.bottom_left[0];
int32_t y_off = target_y - (int32_t)hdr.bottom_left[1];

// 3. Bounds check — outside bounding box means tile does not exist
if (x_off < 0 || y_off < 0 ||
    (uint32_t)x_off >= hdr.tiles_wide ||
    (uint32_t)y_off >= hdr.tiles_high)
    return false;

// 4. Seek directly to the index entry
uint32_t flat_idx = (uint32_t)y_off * hdr.tiles_wide + (uint32_t)x_off;
uint32_t entry_pos = sizeof(MapHeader) + flat_idx * sizeof(IndexEntry);
file.seek(entry_pos);

// 5. Read 8-byte entry — size == 0 means empty slot
IndexEntry entry;
file.read(&entry, sizeof(IndexEntry));
if (entry.size == 0)
    return false;

offset = entry.offset;
size   = entry.size;
return true;
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
