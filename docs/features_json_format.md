# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation.  
The file defines feature styling, priority, and minimum zoom level for OpenStreetMap (OSM) tags to be rendered in the generated tiles.

---

## File Overview

- The file is a JSON object containing **feature definitions** for OpenStreetMap data styling and filtering.
- **Feature definitions** specify styling, priority, and minimum zoom level for OSM tags.

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

## Example

```json
{
  "natural=coastline": {
    "zoom": 6,
    "color": "#0077FF",
    "description": "Coastlines",
    "priority": 1
  },
  "natural=water": {
    "zoom": 12,
    "color": "#3399FF",
    "description": "Water bodies",
    "priority": 1
  },
  "building": {
    "zoom": 15,
    "color": "#BBBBBB",
    "description": "Buildings",
    "priority": 9
  },
  "highway=motorway": {
    "zoom": 6,
    "color": "#FFFFFF",
    "description": "Motorways",
    "priority": 10
  },
  "highway=primary": {
    "zoom": 6,
    "color": "#FFA500",
    "description": "Primary roads",
    "priority": 12
  },
  "amenity=hospital": {
    "zoom": 15,
    "color": "#FF0000",
    "description": "Hospitals",
    "priority": 8
  }
}
```

- **Feature definitions**: Examples showing different feature types and their properties
- **Complete configuration**: The actual `features.json` contains 497 feature definitions covering all major OSM categories

---

## How Matching Works

- During tile generation, each OSM feature is checked against the keys in the JSON:
    - If the feature has a matching tag (`key=value`), the corresponding entry is used for rendering.
    - If only the key matches (e.g., `"building"`), the entry applies for any value of that key.

- If a feature matches multiple entries, the most specific (key=value) takes precedence over generic key matches.

- Features are filtered by zoom level: only features with `zoom` parameter less than or equal to the current tile zoom level are included.

- Features are rendered in priority order: lower priority numbers are drawn first (underneath), higher priority numbers are drawn on top.

---

## Polygon Rendering

Polygons can be rendered in two modes (controlled by the tile viewer application):

- **Filled mode**: Polygons are filled with their specified color and have outlines:
  - **Interior segments**: Outlines are 40% darker than the fill color
  - **Border segments**: Outlines use the original polygon color
- **Outline mode**: Polygons show only outlines:
  - **Interior segments**: Outlines use the original polygon color
  - **Border segments**: Outlines use the background color (making them invisible)

This creates a visual distinction between polygon interiors and tile borders.

## Usage Notes

- You can extend the file with new feature selectors and styling as needed.
- All parameters are optional, but `zoom` and `color` are recommended for correct rendering.
- This file drives both filtering (which features are extracted from the OSM source) and styling (how they appear in the output tiles).
- Priority values range from 1-35 in the current configuration, with natural features having the lowest priorities and place labels having the highest.
- Zoom levels range from 6-17, with major features (coastlines, motorways) appearing at low zoom levels and detailed features (individual trees, street lamps) appearing at high zoom levels.
- Color values use standard HTML hex notation and are automatically converted to RGB332 format for efficient storage in binary tiles.

---

## References

- [OpenStreetMap Tagging](https://wiki.openstreetmap.org/wiki/Tags)
- [Hex Color Codes](https://www.w3schools.com/colors/colors_picker.asp)

---