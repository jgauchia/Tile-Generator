# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation.  
The file defines feature styling, priority, and minimum zoom level for OpenStreetMap (OSM) tags to be rendered in the generated tiles.

---

## File Overview

- The file is a JSON object containing **feature definitions** for OpenStreetMap data styling and filtering.
- **Feature definitions** specify styling, priority, and minimum zoom level for OSM tags.
- Uses a **layer-based priority system** with 9 distinct categories for efficient rendering.

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
"natural=water": {...}
"building": {...}
```

---

## Feature Parameters

Each feature definition object includes:

| Parameter    | Type     | Description                                                                                       |
|--------------|----------|---------------------------------------------------------------------------------------------------|
| `zoom`       | integer  | Minimum zoom level at which the feature is rendered.                                             |
| `color`      | string   | Fill/stroke color in HTML hexadecimal format (`#rrggbb`).                                         |
| `priority`   | integer  | Rendering priority (higher numbers are drawn on top).                                             |
| `width`      | float    | Physical width in meters (used to calculate pixel width).                                         |
| `widths`     | object   | Dictionary of zoom level to fixed pixel width (dynamic styling).                                  |

### Width Calculation Rules

The final rendered width in pixels is determined by:
1. Calculating the pixel equivalent of the physical `width` (if provided).
2. Selecting the maximum value between that and any value in the `widths` table for the current zoom.
3. **Automatic Damping**: For Zoom 13 and above, the calculated width is multiplied by **0.7** to prevent roads from overcrowding the map display.
4. Clamping the result between **1 and 15 pixels**.

### Static vs. Dynamic Widths

You can define the line width in two ways:

1. **Physical Width (`width`):** A float value in meters. The generator will calculate the exact pixel size based on the zoom level and latitude.
   ```json
   "highway=primary": { "width": 10.5 }
   ```

2. **Dynamic Widths (`widths`):** A dictionary mapping zoom levels to fixed pixel values. The generator will select the maximum value between the physical calculation and this aesthetic table.
   ```json
   "highway=motorway": {
     "widths": {
       "6": 1,
       "10": 2,
       "12": 4,
       "15": 8
     }
   }
   ```


---

### Parameter Details

- **zoom**  
  - Specifies the lowest zoom at which the feature will be rendered.
  - Features are omitted from tiles with a zoom less than this value.

- **color**  
  - Specifies the color to render the feature.
  - Uses standard 6-character hex notation (e.g., `#aad3df`).
  - Colors are stored in RGB565 format in binary tiles.

- **priority**  
  - Controls draw order within each layer.
  - Final priority = `layer_base + (priority % 10)`.

---

## Layer System

Features are organized into 9 distinct layers for efficient rendering hierarchy:

| Layer          | Base Priority | Description                                    |
|----------------|----------------|------------------------------------------------|
| landuse        | 10             | Base terrain, parks, forests, urban areas      |
| terrain        | 20             | Natural terrain features, ridges, peaks        |
| water          | 30             | Water bodies, coastlines (Drawn above land)    |
| amenities      | 35             | Facility grounds (Schools, Hospitals)          |
| railways       | 40             | Railway lines and infrastructure               |
| roads          | 50             | All highway types and road features            |
| infrastructure | 60             | Bridges, tunnels, and utility lines            |
| buildings      | 70             | Building footprints and structures             |
| places         | 90             | Geographic labels and place names              |

---

## Special Metadata

The file includes two metadata fields for system configuration:

### Comment Field
```json
"_comment": "OSM-style colors in RGB565 format. Priority: layer_base + (priority % 10)."
```

### Layers Definition
```json
"_layers": "landuse:10, terrain:20, water:30, amenities:35, railways:40, roads:50, infrastructure:60, buildings:70, places:90"
```
- Documents the rendering hierarchy used by the generator.

---

## Example Structure

```json
{
  "_comment": "OSM-style colors in RGB565 format. Priority: layer_base + (priority % 10).",
  "_layers": "landuse:10, terrain:20, water:30, amenities:35, railways:40, roads:50, infrastructure:60, buildings:70, places:90",
  
  "natural=water": {
    "zoom": 12,
    "color": "#aad3df", 
    "priority": 1
  },
  "landuse=residential": {
    "zoom": 12,
    "color": "#e0dfdf",
    "priority": 0
  }
}
```

---

## How Matching Works

- During tile generation, each OSM feature is checked against the keys in the JSON:
    - If a feature has a matching tag (`key=value`), the corresponding entry is used.
    - If only the key matches (e.g., `"building"`), it applies for any value of that key.
- Specific matches (`key=value`) take precedence over generic key matches.
- Features are filtered by zoom level and then sorted by final priority.

---

## Rendering Order Summary

1. **Landuse** (10-19): Base background.
2. **Terrain** (20-29): Relief details.
3. **Water** (30-34): Rivers and lakes on top of land.
4. **Amenities** (35-39): School/Hospital areas.
5. **Railways** (40-49): Tracks.
6. **Roads** (50-59): Streets.
7. **Infrastructure** (60-69): Bridges/Tunnels.
8. **Buildings** (70-79): Footprints.
9. **Places** (90-99): Labels.
