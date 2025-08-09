# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation.  
The file defines feature styling, priority, and minimum zoom level for OpenStreetMap (OSM) tags to be rendered in the generated tiles.

---

## File Overview

- The file is a JSON object.
- Each key is a **feature selector** for OSM data, matching tags in the form `key=value` or just `key`.
- Each value is an object specifying options for rendering that feature.

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

## Example Features

```json
{
  "natural=coastline": {
    "zoom": 6,
    "color": "#b5d0d0",
    "description": "Coastlines",
    "priority": 1
  },
  "landuse=forest": {
    "zoom": 12,
    "color": "#aed18d",
    "description": "Forest areas",
    "priority": 2
  },
  "building": {
    "zoom": 15,
    "color": "#bbbbbb",
    "description": "Buildings",
    "priority": 9
  }
}
```

- In this example:
    - Coastlines appear from zoom level 6 upward, in blueish color, and with high priority.
    - Forests are rendered from zoom 12 upward, in green.
    - Buildings appear only at zoom 15 and higher.

---

## How Matching Works

- During tile generation, each OSM feature is checked against the keys in the JSON:
    - If the feature has a matching tag (`key=value`), the corresponding entry is used for rendering.
    - If only the key matches (e.g., `"building"`), the entry applies for any value of that key.

- If a feature matches multiple entries, the most specific (key=value) takes precedence.

---

## Usage Notes

- You can extend the file with new feature selectors and styling as needed.
- All parameters are optional, but `zoom` and `color` are recommended for correct rendering.
- This file drives both filtering (which features are extracted from the OSM source) and styling (how they appear in the output tiles).

---

## References

- [OpenStreetMap Tagging](https://wiki.openstreetmap.org/wiki/Tags)
- [Hex Color Codes](https://www.w3schools.com/colors/colors_picker.asp)

---