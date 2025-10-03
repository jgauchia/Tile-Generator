# Tile Viewer

`tile_viewer.py` is an advanced Python application for visualizing map tiles, supporting both custom vector tiles in `.bin` format and raster tiles in `.png` format.  
It provides an interactive map viewer with mouse and keyboard controls, allowing exploration of tiles rendered in either format.

## Features

- **High Performance**: LRU cache system with configurable memory limits
- **Multi-threading**: Persistent thread pool for efficient tile loading
- **Error Recovery**: Graceful handling of missing or corrupted tiles
- **Lazy Loading**: Only loads visible tiles to optimize memory usage
- **Advanced Rendering**: Supports advanced compression commands (GRID_PATTERN, CIRCLE, PREDICTED_LINE)
- **Dynamic Palette**: Loads color palettes from configuration files
- **Modern UI**: Beautiful icons and improved button text rendering

---

## What Does This Script Do?

- Loads and renders tiles from a directory containing tiles in `.bin` (vector) **or** `.png` (raster) format
- Supports multiple zoom levels and smooth panning
- Displays tile boundaries, coordinates, and labels
- Offers interactive controls via mouse and keyboard
- Can show GPS cursor coordinates and fill polygons in vector tiles
- Implements intelligent caching and memory management
- Provides detailed logging and error reporting

**Note:** If both `.bin` and `.png` files are present for a tile, `.bin` is preferred and rendered as vector graphics.  
This script is useful for visualizing both the output of the vector tile generator and other map tiles in PNG format.

---

## How It Works

### Controls

- **Mouse Drag:** Pan the viewport across the map
- **Mouse Wheel / `[ ]` keys:** Zoom in and out between available zoom levels
- **Arrow Keys:** Move viewport left, right, up, or down
- **Buttons (on the right toolbar):**
    - **Background:** Toggle background color (black/white)
    - **Tile Labels:** Show/hide tile coordinates and file names
    - **GPS Cursor:** Show/hide cursor latitude/longitude (in decimal and GMS)
    - **Fill Polygons:** Toggle filled polygons rendering (for vector `.bin` tiles)
- **`l` key:** Toggle tile labels
- **Toolbar buttons:** Click to activate features with improved multi-line text rendering

### Status Bar

- Shows the current zoom level and progress bars for indexing/loading tiles
- Displays cache statistics and performance metrics
- Real-time logging of operations and errors

### Startup

Start the script with the tile directory as argument:

```sh
python3 tile_viewer.py TILES_DIRECTORY
```

---

## Dependencies

You need the following Python package:

- `pygame` - Graphics and window management

### Install with:

```sh
pip install pygame
```

---

## Usage Example

### Basic Usage

```sh
python3 tile_viewer.py tiles/
```

- Replace `tiles/` with the path to your directory containing `{zoom}/{x}/{y}.bin` or `.png` tiles.

## Configuration

The application supports configuration through a `features.json` file placed in the current directory:

```json
{
    "tile_size": 256,
    "viewport_size": 768,
    "toolbar_width": 160,
    "statusbar_height": 40,
    "max_cache_size": 1000,
    "thread_pool_size": 4,
    "background_colors": [[0, 0, 0], [255, 255, 255]],
    "log_level": "INFO",
    "fps_limit": 30
}
```

### Configuration Options

- `tile_size`: Size of individual tiles in pixels (default: 256)
- `viewport_size`: Size of the viewport window (default: 768)
- `toolbar_width`: Width of the right toolbar in pixels (default: 160)
- `statusbar_height`: Height of the status bar in pixels (default: 40)
- `max_cache_size`: Maximum number of tiles to cache (default: 1000)
- `thread_pool_size`: Number of worker threads for tile loading (default: 4)
- `background_colors`: Array of background colors for toggle (default: black and white)
- `log_level`: Logging level (default: "INFO")
- `fps_limit`: Maximum frames per second (default: 30)

**Note:** The application will automatically load the configuration from `features.json` if it exists in the same directory as the script.

---

## Performance Features

- **LRU Cache**: Intelligent memory management with configurable limits
- **Thread Pool**: Persistent worker threads for efficient tile loading
- **Lazy Loading**: Only loads visible tiles to optimize memory usage
- **Coordinate Caching**: Cached coordinate conversions for better performance
- **Error Recovery**: Graceful handling of missing or corrupted tiles

---

## Notes

- If the tile directory contains both `.bin` and `.png` files, `.bin` is preferred for vector rendering
- The viewer supports multiple zoom levels, switching seamlessly with mouse wheel or bracket keys
- The GPS tooltip shows coordinates at the mouse position, both in decimal degrees and degrees/minutes/seconds (GMS)
- You can visualize both vector tiles (`.bin`) and raster tiles (`.png`) in the same directory
- The application includes comprehensive logging for debugging and monitoring
- Modern UI with beautiful icons and improved text rendering

---
