# NAV Tile Viewer - ESP32 Map Simulator (v0.4.0)

`tile_viewer.py` is a specialized simulator for the **v0.4.0 packed binary format**. It mirrors the exact rendering logic of the IceNav ESP32 firmware, including the **Four-Pass Rendering Pipeline** and **0.5px width units**.

## New Features (v0.4.0)

- **Four-Pass Rendering Simulation**: Automatically draws layers in the correct order (Polygons → Road Casings → Road Cores → Text Labels) to ensure visual parity with the hardware.
- **Priority Filter Sliders**: Real-time sliders in the sidebar allow you to isolate specific priority levels (0-15) for map analysis and debugging.
- **Reverse Tag Mapping**: When launched with `--config features.json`, the viewer can identify the original OSM tags of an object based on its binary color.
- **Advanced Stats & Legend**: Dedicated panels to analyze the distribution of features by type, priority, and color in the current viewport.
- **NPK1 Container Support**: Directly reads consolidated `Zxx.nav` files, parsing their index tables for instant tile access via `fseek`.
- **Text Label Rendering**: Full support for `GEOM_TEXT` features, including multi-line text and anchor point calculation.

---

## Technical Specifications

### Rendering Pipeline
The simulator mirrors the IceNav-v3 firmware's four-pass logic:
1. **Pass 1**: Polygons and at-grade lines.
2. **Pass 2**: Road casings (darkened color, width+1px).
3. **Pass 3**: Road cores (original color and width).
4. **Pass 4**: Text labels on top of everything.

### Coordinate Mapping
Uses the same bit-shift pixel calculation and 12-bit tile-relative coordinate space (0-4096) as the ESP32:
```python
fx = (tile_x - tl_x) + (coord_x / 4096.0)
pixel_x = fx * TILE_SIZE
```

---

## Controls

### Keyboard Controls
- **Arrow Keys**: Pan map.
- **`[` / `]`**: Zoom out / zoom in.
- **`B`**: Toggle background color (White/Black).
- **`F`**: Toggle polygon fill.
- **`G`**: Toggle tile grid and coordinate labels.
- **`S`**: Open/Close **Statistics Panel**.
- **`L`**: Open/Close **Color Legend Panel**.
- **`R`**: Refresh current viewport (Clear Cache).
- **`Q` / `ESC`**: Quit application.

### Mouse Controls
- **Left Click + Drag**: Pan the map.
- **Mouse Wheel**: Zoom in/out.
- **Right Click**: Identify the feature under the cursor (Type, Color, Tags, Prio, BBox).
- **Sliders**: Drag the blue/red handles in the sidebar to filter priority levels.

---

## Sidebar Panels

### Query Statistics
Displays real-time performance data: tiles loaded from NPK1, total feature count, and query time in milliseconds.

### Feature Legend & Stats
The `L` key opens a color legend that maps RGB565 colors back to their OSM tags (requires `--config`). The `S` key shows a breakdown of features by geometry type and priority nibble.
