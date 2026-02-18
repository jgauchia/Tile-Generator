# NAV Tile Viewer - ESP32 Map Simulator (NAV-PACK Version)

`tile_viewer.py` is a specialized simulator for the **NAV-PACK** binary format. It is designed to preview and debug consolidated vector map data before deployment to the IceNav ESP32 navigator. It displays a 768x768 viewport using the same offset-based lookup and coordinate math as the hardware.

## Features

- **NAV-PACK Support**: Directly reads consolidated `Zxx.nav` files, parsing their index tables for instant tile access.
- **ESP32 Rendering Simulation**: Mirrors the bit-shift pixel calculation and tile-relative coordinate space used in the ESP32 firmware.
- **Adaptive Tile Loading**: Loads tiles surrounding the current position to ensure the 768x768 viewport is always fully covered.
- **Feature Identification**: Right-click any object to see its type, color, zoom level, and pre-calculated BBox.
- **Grid Overlay**: Visualizes tile boundaries and Pack coordinates (X/Y) for easy debugging.
- **Polygon Fill Toggle**: Switch between outlined and filled polygon rendering.

---

## What Does This Script Do?

- Loads and renders consolidated NAV packs for ESP32 navigation simulation.
- Displays a 768x768 viewport (exactly 3x3 tiles at 256px, but simulates wider coverage).
- Simulates the zero-CPU projection by using the pre-calculated 12-bit tile space (0-4096).
- Supports interactive panning and zooming across all available Packs in the directory.
- Provides real-time statistics on tile loading and feature counts.

---

## Technical Specifications

### NAV-PACK Loading
The viewer scans the directory for files named `Z*.nav`. It loads the **Index Table** of each pack into memory to perform fast `fseek` operations, exactly like the ESP32 firmware does.

### Rendering Pipeline
The simulator uses the same math as the IceNav-v3 firmware to map tile coordinates to screen pixels:
```python
fx = (tile_x - tl_x) + (coord_x / 4096.0)
pixel_x = fx * TILE_SIZE
```
This ensures that what you see in the viewer is exactly what will be rendered on the ESP32 display.

---

## Controls

### Mouse Controls
- **Left Click + Drag**: Pan the map.
- **Mouse Wheel**: Zoom in/out (limited to available Packs).
- **Right Click**: Identify the feature under the cursor.

### Keyboard Controls
- **Arrow Keys**: Pan map.
- **`[` / `]`**: Zoom out / zoom in.
- **`B`**: Toggle background color (White/Black).
- **`F`**: Toggle polygon fill.
- **`G`**: Toggle tile grid and coordinate labels.
- **`Q` / `ESC`**: Quit application.

---

## Status and Sidebar Information

### Query Statistics
- **Tiles**: Number of tiles found in the Pack vs. total tiles required for the view.
- **Features**: Total number of vector objects currently being rendered.
- **Time**: Time taken to query the Index Table and parse tile data in milliseconds.

### Selected Feature Details
When you right-click an object, the sidebar displays:
- **Type**: Point, Line, or Polygon.
- **Color**: Hexadecimal representation of the RGB565 color.
- **Zoom**: The minimum zoom level configured for this object.
- **Pts**: Number of vertices in the geometry.
- **BBox**: Pre-calculated object extent normalized to 0-255 (used for ESP32 culling).
