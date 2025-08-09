# Tile Viewer

`tile_viewer.py` is a Python application for visualizing map tiles, supporting both custom vector tiles in `.bin` format and raster tiles in `.png` format.  
It provides an interactive map viewer with mouse and keyboard controls, allowing exploration of tiles rendered in either format.

---

## What Does This Script Do?

- Loads and renders tiles from a directory containing tiles in `.bin` (vector) **or** `.png` (raster) format.
- Supports multiple zoom levels and smooth panning.
- Displays tile boundaries, coordinates, and labels.
- Offers interactive controls via mouse and keyboard.
- Can show GPS cursor coordinates and fill polygons in vector tiles.

**Note:** If both `.bin` and `.png` files are present for a tile, `.bin` is preferred and rendered as vector graphics.  
This script is useful for visualizing both the output of the vector tile generator and other map tiles in PNG format.

---

## How It Works

### Controls

- **Mouse Drag:** Pan the viewport across the map.
- **Mouse Wheel / `[ ]` keys:** Zoom in and out between available zoom levels.
- **Arrow Keys:** Move viewport left, right, up, or down.
- **Buttons (on the right toolbar):**
    - **Background:** Toggle background color (black/white).
    - **Tile Labels:** Show/hide tile coordinates and file names.
    - **GPS Cursor:** Show/hide cursor latitude/longitude (in decimal and GMS).
    - **Fill Polygons:** Toggle filled polygons rendering (for vector `.bin` tiles).
- **`l` key:** Toggle tile labels.
- **Toolbar buttons:** Click to activate features.

### Status Bar

- Shows the current zoom level and progress bars for indexing/loading tiles.

### Startup

Start the script with the tile directory as argument:

```sh
python tile_viewer.py TILES_DIRECTORY
```

---

## Dependencies

You need the following Python package:

- `pygame`

### Install with:

```sh
pip install pygame
```

---

## Usage Example

```sh
python tile_viewer.py tiles/
```

- Replace `tiles/` with the path to your directory containing `{zoom}/{x}/{y}.bin` or `.png` tiles.

---

## Notes

- If the tile directory contains both `.bin` and `.png` files, `.bin` is preferred for vector rendering.
- The viewer supports multiple zoom levels, switching seamlessly with mouse wheel or bracket keys.
- The GPS tooltip shows coordinates at the mouse position, both in decimal degrees and degrees/minutes/seconds (GMS).
- You can visualize both vector tiles (`.bin`) and raster tiles (`.png`) in the same directory.

---

## License

MIT License

---