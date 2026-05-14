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
# Output: /data/maps/ROUTE/ROUTE.bin
```

The graph is internally partitioned into **0.1°×0.1° subcells** and stored in a single `ROUTE.bin` file with a cell index. Readers load only the cells that cover the route bounding box.

---

## Output: ROUTE.bin format

### File layout

```
FileHeader      (32 bytes)
CellIndex[]     (20 bytes × cell_count)
--- data block ---
  Cell 0: Node[node_count_0] + Edge[edge_count_0]
  Cell 1: Node[node_count_1] + Edge[edge_count_1]
  ...
  Cell N: Node[node_count_N] + Edge[edge_count_N]
```

Each cell's nodes and edges are stored **contiguously** in the data block. The `data_offset` field in the cell index gives the byte offset from the start of the data block to the beginning of that cell's node array. This layout allows the firmware loader to read each cell with a **single seek + read** instead of two separate operations.

`dst_node` in each Edge is a **global node index** — absolute across all cells, no remapping needed.  
`edge_offset` in each Node is relative to that cell's own edge block (i.e., index 0 = first edge of this cell).

### FileHeader (32 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | char[4] | magic | `"ROUT"` |
| 4 | uint32 | sub_step_e4 | grid step × 10000; `1000` = 0.1° cells |
| 8 | uint32 | cell_count | number of cells in the index |
| 12 | uint32[5] | reserved | padding to 32 bytes |

### CellIndex entry (20 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | int32 | lat_e4 | SW corner latitude × 10000, snapped to 0.1° grid |
| 4 | int32 | lon_e4 | SW corner longitude × 10000, snapped to 0.1° grid |
| 8 | uint32 | node_offset | global index of first node (used for nearest-node lookup) |
| 12 | uint16 | node_count | number of nodes in this cell |
| 14 | uint32 | data_offset | byte offset from start of data block to this cell's `Node[0]` |
| 18 | uint16 | edge_count | number of edges in this cell |

### Node (12 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | float | lat | latitude in degrees |
| 4 | float | lon | longitude in degrees |
| 8 | uint32 | edge_offset | index of first outgoing edge within this cell's edge block |

Edges for node `i` span `edge[node[i].edge_offset .. node[i+1].edge_offset - 1]` within the cell's edge block. For the last node, the range ends at `edge_count`.

### Edge (12 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | uint32 | dst_node | destination global node index |
| 4 | uint32 | cost | travel time in tenths of second |
| 8 | uint16 | dist_m | segment length in metres (capped at 65535) |
| 10 | uint8 | flags | `bit0` = oneway, `bits1-3` = highway class (0–6) |
| 11 | uint8 | reserved | always 0 |

### Highway classes (bits 1–3 of flags)

| Value | `highway=` tags | Default speed |
|---|---|---|
| 0 | service / track / other | 20 km/h |
| 1 | living_street / residential | 30 km/h |
| 2 | unclassified / tertiary / tertiary_link | 50 km/h |
| 3 | secondary / secondary_link | 70 km/h |
| 4 | primary / primary_link | 90 km/h |
| 5 | trunk / trunk_link | 110 km/h |
| 6 | motorway / motorway_link | 130 km/h |

If `maxspeed` is set in OSM and is in the range 1–254 km/h, it overrides the default speed for cost calculation.

---

## Console output

The generator prints per-cell statistics to stdout:

```
[GRAPH] Cell E4:415000_10000: 6244 nodes, 12766 edges
[GRAPH]   Giant component : 6015 / 6244 (96.3%)
[GRAPH]   Oneway edges    : 1846 (14.5%)
[GRAPH]   Classes (0-6)   : 5551 2659 904 1063 1759 830 0
[GRAPH] Written: ROUTE/ROUTE.bin  (1 cells, 6244 nodes, 12766 edges)
```

A warning is printed to stderr if the giant connected component is below 95% — this usually indicates a problem with intersection detection.

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
  ROUTE/    ← routing graph
  TRK/
  WPT/
```
