# Route Generator

`route_generator` builds A\* routing graph files from an OSM PBF extract. It generates **three independent ROUTE.bin files** — one per routing profile — in separate subdirectories. It is independent from the tile generator and can be used with vector tiles, PNG tiles, or any other map source.

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
| `output_dir` | Directory where `ROUTE/` subdirectories will be created |

The generator always produces all three profiles in a single run:

```
<output_dir>/ROUTE/CAR/ROUTE.bin
<output_dir>/ROUTE/BIKE/ROUTE.bin
<output_dir>/ROUTE/WALK/ROUTE.bin
```

### Example

```bash
./route_generator andorra-251227.osm.pbf .
# Output:
#   ./ROUTE/CAR/ROUTE.bin
#   ./ROUTE/BIKE/ROUTE.bin
#   ./ROUTE/WALK/ROUTE.bin

./route_generator catalonia-latest.osm.pbf /data/maps
# Output:
#   /data/maps/ROUTE/CAR/ROUTE.bin
#   /data/maps/ROUTE/BIKE/ROUTE.bin
#   /data/maps/ROUTE/WALK/ROUTE.bin
```

The graph is internally partitioned into **0.05°×0.05° subcells** and stored in a single `ROUTE.bin` file per profile with a cell index. Readers load only the cells needed for the route on-demand.

---

## Routing profiles

Each profile controls which roads are included and at what speed. The `cost` field in each edge encodes travel time in tenths of second: `cost = (dist_m / speed_ms) × 10`.

| hw_class | Car | Walk | Bike |
|---|---|---|---|
| 0 — service / track | 20 km/h | 5 km/h | 10 km/h |
| 1 — residential / living\_street | 25 km/h | 5 km/h | 15 km/h |
| 2 — tertiary / unclassified | 50 km/h | 5 km/h | 18 km/h |
| 3 — secondary | 70 km/h | 5 km/h | 20 km/h |
| 4 — primary | 90 km/h | 5 km/h | 22 km/h |
| 5 — trunk | 110 km/h | — | — |
| 6 — motorway | 130 km/h | — | — |

`—` means the road type is inaccessible for that profile; edges are not generated.

**Car**: uses OSM `maxspeed` tag when present to override the base speed. Residential roads are penalised (25 km/h) vs arterials to prefer main roads. Footways, cycleways, pedestrian zones, paths and steps are excluded.

**Walk**: constant 5 km/h on all accessible roads. Motorway and trunk are excluded. Includes footway, path, pedestrian, cycleway.

**Bike**: 10–22 km/h depending on road class. Motorway, trunk, footway and steps are excluded.

---

## Using with IceNav-v3 (ESP32)

Copy all three profile subdirectories to the SD card alongside `NAVMAP/`:

```
/sdcard/
  NAVMAP/          ← vector tiles
  ROUTE/
    CAR/
      ROUTE.bin    ← car routing graph
    BIKE/
      ROUTE.bin    ← bike routing graph
    WALK/
      ROUTE.bin    ← pedestrian routing graph
  TRK/
  WPT/
```

The active profile is selected in **Device Settings → Routing Profile** (Car / Bike / Pedestrian). IceNav-v3 loads the corresponding `ROUTE.bin` automatically. Changing the profile takes effect on the next route calculation.

The A\* heuristic speed is derived from the selected profile — no manual configuration needed and no mismatch between graph costs and heuristic is possible.

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
| 4 | uint32 | sub_step_e4 | grid step × 10000; `500` = 0.05° cells |
| 8 | uint32 | cell_count | number of cells in the index |
| 12 | uint32[5] | reserved | padding to 32 bytes |

### CellIndex entry (20 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | int32 | lat_e4 | SW corner latitude × 10000, snapped to 0.05° grid |
| 4 | int32 | lon_e4 | SW corner longitude × 10000, snapped to 0.05° grid |
| 8 | uint32 | node_offset | global index of first node (used for nearest-node lookup) |
| 12 | uint16 | node_count | number of nodes in this cell |
| 14 | uint32 | data_offset | byte offset from start of data block to this cell's `Node[0]` |
| 18 | uint16 | edge_count | number of edges in this cell |

### Node (10 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | float | lat | latitude in degrees |
| 4 | float | lon | longitude in degrees |
| 8 | uint16 | edge_offset | index of first outgoing edge within this cell's edge block |

Edges for node `i` span `edge[node[i].edge_offset .. node[i+1].edge_offset - 1]` within the cell's edge block. For the last node, the range ends at `edge_count`.

### Edge (8 bytes)

| Offset | Type | Field | Description |
|---|---|---|---|
| 0 | uint32 | dst_node | destination global node index |
| 4 | uint32 | cost | travel time in tenths of second |

> **Note:** earlier versions stored an extra `dist_m` (uint16), `flags` (uint8, oneway +
> highway class) and a `reserved` byte per edge (12-byte edge). These were not read by the
> firmware router — A\* only uses `dst_node` and `cost` — so they were removed, shrinking the
> edge record from 12 to 8 bytes (the edge block is the dominant part of the file). The
> oneway constraint and per-class speed are already baked into the graph during generation
> (reverse edges are omitted for oneways; `cost` already encodes the class-dependent speed).

---

## Console output

The generator prints per-profile and per-cell statistics to stdout:

```
Route generator
Input  : andorra-251227.osm.pbf (38.3 MB)
Output : ./ROUTE/{CAR,BIKE,WALK}/ROUTE.bin

=== Profile: car ===
Pass 1: Scanning relations...
Pass 2: Extracting road ways...
Ways processed : 38412
Road ways found: 12853 (filtered: 25559)
Building routing graph...
[GRAPH] Cell E4:415000_10000: 6244 nodes, 12766 edges
[GRAPH]   Giant component : 6015 / 6244 (96.3%)
[GRAPH]   Oneway edges    : 1846 (14.5%)
[GRAPH]   Classes (0-6)   : 5551 2659 904 1063 1759 830 0
[GRAPH] Written: ./ROUTE/CAR/ROUTE.bin  (1 cells, 6244 nodes, 12766 edges)
Profile done in 12s

=== Profile: bike ===
...
[GRAPH] Written: ./ROUTE/BIKE/ROUTE.bin  (1 cells, 7102 nodes, 14330 edges)
Profile done in 11s

=== Profile: pedestrian ===
...
[GRAPH] Written: ./ROUTE/WALK/ROUTE.bin  (1 cells, 7580 nodes, 15210 edges)
Profile done in 11s

All profiles done in 34s
```

A warning is printed to stderr if the giant connected component is below 95% — this usually indicates a problem with intersection detection.
