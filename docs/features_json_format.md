# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation.  
The file defines feature styling, priority, and minimum zoom level for OpenStreetMap (OSM) tags to be rendered in the generated tiles.

---

## File Overview

- The file is a JSON object containing both **system configuration** and **feature definitions**.
- **System configuration** parameters control tile generation and viewer behavior.
- **Feature definitions** specify styling, priority, and minimum zoom level for OSM tags.

---

## System Configuration Parameters

The following parameters control the tile generation and viewer system:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tile_size` | integer | Size of generated tiles in pixels | 256 |
| `viewport_size` | integer | Size of the viewer viewport in pixels | 768 |
| `toolbar_width` | integer | Width of the viewer toolbar in pixels | 160 |
| `statusbar_height` | integer | Height of the viewer status bar in pixels | 40 |
| `max_cache_size` | integer | Maximum number of tiles to cache in memory | 1000 |
| `thread_pool_size` | integer | Number of worker threads for tile generation | 4 |
| `background_colors` | array | Array of background colors [black, white] | `[[0,0,0], [255,255,255]]` |
| `log_level` | string | Logging level (DEBUG, INFO, WARNING, ERROR) | "INFO" |
| `config_file` | string | Path to this configuration file | "features.json" |
| `fps_limit` | integer | Maximum frames per second for the viewer | 30 |
| `fill_polygons` | boolean | Whether to fill polygons or show only outlines | true |

---

## Feature Definitions

Each feature definition is a key-value pair where:
- The key is a **feature selector** for OSM data, matching tags in the form `key=value` or just `key`.
- The value is an object specifying options for rendering that feature.

---

## Feature Selector

- The key is either:
    - `"key=value"`: Matches OSM features where the given tag equals the specified value.
    - `"key"`: Matches any OSM feature with the given key present (any value).

**Examples:**
```json
"natural=coastline": {...}
"building": {...}
```

---

## Feature Parameters

Each feature definition object can include:

| Parameter    | Type     | Description                                                                                       |
|--------------|----------|---------------------------------------------------------------------------------------------------|
| zoom         | integer  | Minimum zoom level at which the feature is rendered.                                              |
| color        | string   | Fill/stroke color in HTML hexadecimal format (`#rrggbb`).                                         |
| description  | string   | Human-readable description of the feature.                                                        |
| priority     | integer  | Rendering priority (lower numbers are rendered first/underneath; higher numbers are on top).      |

---

### Parameter Details

- **zoom**  
  - Specifies the lowest zoom at which the feature will be rendered.
  - Features are omitted from tiles with a zoom less than this value.

- **color**  
  - Specifies the color to render the feature.
  - Uses standard 6-character hex notation (e.g., `#ffbe00` for yellow).
  - This is mapped to a compact color encoding (RGB332) in the binary tiles.

- **description**  
  - Provides a short, human-readable label for the feature type.

- **priority**  
  - Controls draw order.  
  - Lower values are rendered below higher values in the tile.

---

## Complete Example

```json
{
  "tile_size": 256,
  "viewport_size": 768,
  "toolbar_width": 160,
  "statusbar_height": 40,
  "max_cache_size": 1000,
  "thread_pool_size": 4,
  "background_colors": [
    [0, 0, 0],
    [255, 255, 255]
  ],
  "log_level": "INFO",
  "config_file": "features.json",
  "fps_limit": 30,
  "fill_polygons": true,
  "natural=coastline": {
    "zoom": 6,
    "color": "#0077FF",
    "description": "Coastlines",
    "priority": 1
  },
  "natural=water": {
    "zoom": 11,
    "color": "#3399FF",
    "description": "Water bodies",
    "priority": 1
  },
  "building": {
    "zoom": 15,
    "color": "#BBBBBB",
    "description": "Buildings",
    "priority": 9
  }
}
```

- **System configuration**: Controls tile generation and viewer behavior
- **Feature definitions**: 
    - Coastlines appear from zoom level 6 upward, in blue color, with high priority
    - Water bodies are rendered from zoom 11 upward, in light blue
    - Buildings appear only at zoom 15 and higher, in gray

---

## How Matching Works

- During tile generation, each OSM feature is checked against the keys in the JSON:
    - If the feature has a matching tag (`key=value`), the corresponding entry is used for rendering.
    - If only the key matches (e.g., `"building"`), the entry applies for any value of that key.

- If a feature matches multiple entries, the most specific (key=value) takes precedence.

---

## Polygon Rendering

The `fill_polygons` parameter controls how polygons are rendered:

- **`true`**: Polygons are filled with their specified color and have outlines:
  - **Interior segments**: Outlines are 40% darker than the fill color
  - **Border segments**: Outlines use the original polygon color
- **`false`**: Polygons show only outlines:
  - **Interior segments**: Outlines use the original polygon color
  - **Border segments**: Outlines use the background color (making them invisible)

This creates a visual distinction between polygon interiors and tile borders.

## Usage Notes

- You can extend the file with new feature selectors and styling as needed.
- All parameters are optional, but `zoom` and `color` are recommended for correct rendering.
- This file drives both filtering (which features are extracted from the OSM source) and styling (how they appear in the output tiles).
- The `fill_polygons` parameter significantly affects the visual appearance of the generated tiles.

---

## References

- [OpenStreetMap Tagging](https://wiki.openstreetmap.org/wiki/Tags)
- [Hex Color Codes](https://www.w3schools.com/colors/colors_picker.asp)

---