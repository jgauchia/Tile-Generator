# Route Generator

`route_generator` builds a single A\* routing graph file (`ROUTE/ROUTE.bin`) from an OSM PBF extract. It is independent from the tile generator — you can use it with vector tiles, PNG tiles, or any other map source.

---

## Dependencies

Same as `nav_generator`:

```bash
sudo apt-get install -y build-essential cmake \
    libosmium2-dev libgeos-dev libgdal-dev \
    libbz2-dev zlib1g-dev libexpat1-dev
```

## Compilation

Built alongside `nav_generator` from the same CMake project:

```bash
mkdir build && cd build
cmake ..
make -j$(nproc) route_generator
```

The binary is produced at `build/route_generator`.

---

## Usage

```bash
route_generator <input.pbf> <output_dir>
```

| Argument | Description |
|---|---|
| `input.pbf` | OSM PBF extract (any region) |
| `output_dir` | Directory where `ROUTE/` will be created |

### Example

```bash
# Generate routing graph for Andorra
./route_generator andorra-251227.osm.pbf .
# Output: ./ROUTE/ROUTE.bin

# Generate for a larger region
./route_generator catalonia-latest.osm.pbf /data/maps
# Output: /data/maps/ROUTE/ROUTE.bin  (9 cells for Catalonia)
```

The graph is internally partitioned into **1°×1° cells** and stored in a single `ROUTE.bin` file with a cell index. Readers load only the cells that cover the route endpoints.

---

## Output: ROUTE.bin format

### File layout

```
FileHeader     (32 bytes)
CellIndex[]    (24 bytes × cell_count)
Node[]         (12 bytes × total_node_count)   — all cells concatenated
Edge[]         (14 bytes × total_edge_count)   — all cells concatenated
```

`dst_node` in each Edge is a **global index** into the flat Node array — no remapping needed when merging cells.
`edge_offset` in each Node is relative to the cell's own edge block; add `CellIndex.edge_offset` to get the absolute position.

### FileHeader (32 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | char[4] | magic | `"ROUT"` |
| 4 | uint8 | version | `2` |
| 5 | uint8[3] | reserved | padding |
| 8 | uint32 | cell_count | number of 1°×1° cells in the index |
| 12 | uint32[4] | reserved2 | padding to 32 bytes |

### CellIndex entry (24 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | int16 | lat_floor | floor(latitude) of cell south-west corner |
| 2 | int16 | lon_floor | floor(longitude) of cell south-west corner |
| 4 | uint32 | node_offset | global index of first node (`dst_node` base for this cell) |
| 8 | uint32 | node_count | number of nodes in this cell |
| 12 | uint32 | edge_offset | index of first edge in the flat Edge array |
| 16 | uint32 | edge_count | number of edges in this cell |
| 20 | uint32 | reserved | padding to 24 bytes |

### Node (12 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | float | lat | latitude |
| 4 | float | lon | longitude |
| 8 | uint32 | edge_offset | index of first outgoing edge (relative to cell edge block) |

Edges for node `i` span `edge[node[i].edge_offset .. node[i+1].edge_offset - 1]` within the cell's edge block.

### Edge (14 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | uint32 | dst_node | destination node index (relative to cell node block) |
| 4 | uint32 | cost | travel time in tenths of second |
| 8 | uint16 | dist_m | segment length in metres (max 65535) |
| 10 | uint8 | flags | `bit0` = oneway, `bits1-3` = highway class |
| 11 | uint16 | name_idx | reserved, always 0 |
| 13 | uint8 | reserved | |

### Highway classes (bits 1-3 of flags)

| Value | `highway=` | Base speed |
|---|---|---|
| 0 | other / service / track | 20 km/h |
| 1 | living_street / residential | 30 km/h |
| 2 | unclassified / tertiary | 50 km/h |
| 3 | secondary | 70 km/h |
| 4 | primary | 90 km/h |
| 5 | trunk | 110 km/h |
| 6 | motorway | 130 km/h |

---

## Console output

The generator prints per-cell statistics to stdout:

```
[GRAPH] Cell R42_1: 6244 nodes, 12766 edges
[GRAPH]   Giant component : 6015 / 6244 (96.3%)
[GRAPH]   Oneway edges    : 1846 (14.5%)
[GRAPH]   Classes (0-6)   : 5551 2659 904 1063 1759 830 0
[GRAPH] Written: ROUTE/ROUTE.bin  (1 cells, 6244 nodes, 12766 edges)
```

A warning is printed if the giant connected component is below 95% — this usually indicates a problem with intersection detection.

---

## Using with tile_viewer.py

```bash
# Generate tiles and routing graph separately
./nav_generator andorra.pbf NAVMAP features.json
./route_generator andorra.pbf .

# Open viewer with both
python tile_viewer.py NAVMAP --lat 42.5069 --lon 1.5218 --route-dir ROUTE
```

See [`docs/tile_viewer.md`](tile_viewer.md) for routing interaction details.

---

## Using with IceNav-v3 (ESP32)

Copy `ROUTE/ROUTE.bin` to the SD card alongside `NAVMAP/`:

```
/sdcard/
  NAVMAP/   ← vector tiles
  ROUTE/    ← routing graphs
  TRK/
  WPT/
```
