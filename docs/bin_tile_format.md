# NAV-PACK Format Specification (NPK2)

This document describes the **NPK2** container format used by the NAV tile generator.
NPK2 uses a **sparse index** (coverage bitmap + popcount rank table + compact 8-byte
entries) over a rectangular bounding box, providing **O(1)** tile lookup with 2-3 small
random reads: a 64-byte bitmap block, a 4-byte rank value and an 8-byte entry. Empty
cells (sea, gaps) resolve with a single small read instead of wasting a full 8-byte
slot per cell.

---

## 1. Storage Architecture

Map data is organized into one binary file per zoom level: `Z{zoom}.nav`.

Each file consists of: **Map Header** → **Sparse Index** → **Color Palette** → **Tile Data Blocks**.

The grid is defined by `tiles_wide × tiles_high` cells (row-major, Y outer, X inner).
The position of a tile `(x, y)` in the grid is:

```
flat_index = (y - bottom_left[1]) * tiles_wide + (x - bottom_left[0])
```

The sparse index stores **only the cells that contain data** (compact entries in ascending
`flat_index` order) plus a 1-bit-per-cell coverage bitmap. A popcount rank table (one
`uint32` per 64 bitmap bytes = 512 cells) maps a cell to its entry position in O(1).

---

## 2. File Structure

### 2.1. Map Header (23 bytes, `#pragma pack(push,1)`)

| Offset | Field          | Type     | Size | Description                           |
|--------|----------------|----------|------|---------------------------------------|
| 0      | magic          | bytes[4] | 4    | `"NPK2"`                              |
| 4      | zoom           | uint8    | 1    | Zoom level                            |
| 5      | tiles_wide     | uint32   | 4    | Width of the bounding box in tiles    |
| 9      | tiles_high     | uint32   | 4    | Height of the bounding box in tiles   |
| 13     | bottom_left[0] | uint32   | 4    | Absolute tile X of origin (min_x)     |
| 17     | bottom_left[1] | uint32   | 4    | Absolute tile Y of origin (min_y)     |
| 21     | color_count    | uint16   | 2    | Number of RGB565 entries in palette   |

### 2.2. Sparse Index

Immediately after the header:

| Offset        | Field         | Type     | Size                    | Description                              |
|---------------|---------------|----------|-------------------------|------------------------------------------|
| 0             | index_count   | uint32   | 4                       | Number of cells with data (compact entries) |
| 4             | coverage      | bytes    | `ceil(tiles_wide × tiles_high / 8)` | 1 bit per cell; bit `f` set = cell `f` has data (LSB-first, byte `f >> 3`) |
| after bitmap  | rank          | uint32[] | `ceil(bitmap_bytes / 64) × 4` | `rank[i]` = number of set bits in `bitmap[0 .. i*64)` (cumulative popcount) |
| after rank    | entries       | —        | `index_count × 8`       | `IndexEntry { offset u32, size u32 }` in ascending `flat_index` order |

#### Compact entry lookup

```
f        = flat_index of the target cell
block    = bitmap[f >> 13] .. bitmap[(f >> 13) + 63]   // 512 cells, clamped at the end of the bitmap
bit f&7  of byte f>>3 is 0  →  cell is empty, no data
rank     = rank[f >> 9]                                // set bits before this 512-cell block
within   = popcount(bitmap bytes of the block before byte f>>3)
e        = rank + within                               // position in the compact entries
offset,size = entries[e]
```

### 2.3. Color Palette (`color_count × 2` bytes)

Placed immediately after the compact entries and before the first tile block. A contiguous
array of `color_count` RGB565 values (LE). Feature headers reference colors by 1-byte
index into this table instead of storing the full 16-bit color. Tile data offsets in the
entries already account for the palette size, so reading is still a single seek.

| Offset      | Field   | Type   | Size | Description           |
|-------------|---------|--------|------|-----------------------|
| `i × 2`     | color   | uint16 | 2    | RGB565 color (LE)     |

---

## 3. Internal Tile Format (NAV1)

### 3.1. Tile Header (6 bytes)

| Offset | Field         | Type      | Size | Description        |
|--------|---------------|-----------|------|--------------------|
| 0      | magic         | bytes[4]  | 4    | `"NAV1"`           |
| 4      | feature_count | uint16    | 2    | Number of features |

---

## 4. Feature Records

Each feature has a **variable-length header** followed by compressed coordinates or text
payload. The fixed part is 8 bytes; `coord_count` and `payload_size` are LEB128 varints.
Text payloads are **not 4-byte aligned**: `coord_count` equals the payload size in bytes.

| Field         | Type    | Size   | Description                                        |
|---------------|---------|--------|----------------------------------------------------|
| geom_type     | uint8   | 1      | 1=Point, 2=Line, 3=Polygon, 4=Text                 |
| color_index   | uint8   | 1      | Index into the pack color palette                  |
| zoom_priority | uint8   | 1      | `(min_zoom << 4) \| (priority & 0x0F)`             |
| width_flags   | uint8   | 1      | Bit 7=Casing, Bits 0-6=Width (0.5px units)         |
| min_x         | uint8   | 1      | BBox min X (coords/16)                             |
| min_y         | uint8   | 1      | BBox min Y (coords/16)                             |
| max_x         | uint8   | 1      | BBox max X (coords/16)                             |
| max_y         | uint8   | 1      | BBox max Y (coords/16)                             |
| coord_count   | varint  | 1–3    | Number of vertices (or payload bytes for text)     |
| payload_size  | varint  | 1–3    | Total bytes of data following the header           |

> **Note:** the previous format used a fixed 13-byte header with a 2-byte inline color
> and 2-byte `coord_count`/`payload_size`. The current format moves color to a global
> palette (1-byte index) and varint-encodes the two counts, shrinking the typical feature
> header from 13 to ~10 bytes and the tile header from 22 to 6 bytes. The `"NPK2"` /
> `"NAV1"` magics are unchanged.

---

## 5. Tile Lookup Algorithm (O(1))

```cpp
// 1. Read MapHeader at file offset 0
MapHeader hdr;
file.read(&hdr, sizeof(MapHeader));

// 2. Read the sparse index layout
uint32_t index_count = read_u32();
uint32_t bitmap_bytes = (hdr.tiles_wide * hdr.tiles_high + 7) / 8;
uint32_t rank_count = (bitmap_bytes + 63) / 64;
uint32_t bitmap_base  = sizeof(MapHeader) + 4;
uint32_t rank_base    = bitmap_base + bitmap_bytes;
uint32_t entries_base = rank_base + rank_count * 4;

// 3. Compute relative offsets and bounds check
int32_t x_off = target_x - (int32_t)hdr.bottom_left[0];
int32_t y_off = target_y - (int32_t)hdr.bottom_left[1];
if (x_off < 0 || y_off < 0 ||
    (uint32_t)x_off >= hdr.tiles_wide ||
    (uint32_t)y_off >= hdr.tiles_high)
    return false;

uint32_t flat = (uint32_t)y_off * hdr.tiles_wide + (uint32_t)x_off;

// 4. Read the 64-byte coverage block and test the cell bit
uint32_t block_start = (flat >> 9) * 64;
uint8_t block[64];
file.seek(bitmap_base + block_start);
file.read(block, min(64, bitmap_bytes - block_start));
if (!(block[(flat >> 3) - block_start] & (1u << (flat & 7))))
    return false;   // empty cell

// 5. Rank: cumulative popcount up to this cell (block rank + in-block popcount)
uint32_t rank = read_u32_at(rank_base + (flat >> 9) * 4);
for (uint32_t i = 0; i < (flat >> 3) - block_start; ++i)
    rank += popcount(block[i]);

// 6. Read the compact 8-byte entry
IndexEntry entry;
file.seek(entries_base + (uint64_t)rank * sizeof(IndexEntry));
file.read(&entry, sizeof(IndexEntry));

offset = entry.offset;
size   = entry.size;
return true;
```

The lookup is unaffected by the palette: `entry.offset` is an absolute file offset that
already includes `sizeof(MapHeader) + sparse_index + palette`. The palette
(`color_count × 2` bytes, located right after the compact entries) is read once when the
pack is opened and kept in memory; each feature's `color_index` is then resolved to RGB565
via `palette[color_index]`.

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
