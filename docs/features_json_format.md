# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation.  
The file defines feature styling, priority, and minimum zoom level for OpenStreetMap (OSM) tags to be rendered in the generated tiles.

---

## File Overview

- The file is a JSON object containing **feature definitions** for OpenStreetMap data styling and filtering.
- **Feature definitions** specify styling, priority, and minimum zoom level for OSM tags.
- Uses **layer-based priority system** with 8 distinct categories for efficient rendering.

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

Each feature definition object includes:

| Parameter    | Type     | Description                                                                                       |
|--------------|----------|---------------------------------------------------------------------------------------------------|
| zoom         | integer  | Minimum zoom level at which the feature is rendered.                                             |
| color        | string   | Fill/stroke color in HTML hexadecimal format (`#rrggbb`).                                         |
| priority     | integer  | Layer-based priority for rendering order (calculated as layer_base + (priority % 10)).      |

---

### Parameter Details

- **zoom**  
  - Specifies the lowest zoom at which the feature will be rendered.
  - Features are omitted from tiles with a zoom less than this value.

- **color**  
  - Specifies the color to render the feature.
  - Uses standard 6-character hex notation (e.g., `#aad3df` for water).
  - Colors are stored in RGB565 format in binary tiles.

- **priority**  
  - Controls draw order within each layer.
  - Uses layer-based system: `layer_base + (priority % 10)`.
  - Layer bases: water(10), landuse(20), terrain(30), railways(40), roads(50), infrastructure(60), buildings(70), amenities(80), places(90).
  - Priority within layer: 0-9, where higher numbers render on top.

---

## Layer System

Features are organized into 8 distinct layers for efficient rendering:

| Layer          | Base Priority | Range     | Description                                    |
|----------------|----------------|------------|------------------------------------------------|
| water          | 10             | 10-19      | Water bodies, coastlines, waterways                |
| landuse        | 20             | 20-29      | Land use areas, parks, natural terrain             |
| terrain        | 30             | 30-39      | Natural terrain features                        |
| railways       | 40             | 40-49      | Railway lines and infrastructure                 |
| roads          | 50             | 50-69      | All highway types and road features             |
| infrastructure | 60             | 60-69      | Infrastructure and utility lines                |
| buildings      | 70             | 70-79      | Building footprints and structures              |
| amenities      | 80             | 80-89      | Points of interest and facilities             |
| places         | 90             | 90-99      | Geographic places and labels                |

**Priority Calculation:**
```python
# Example for feature with priority=15 in roads layer:
final_priority = 50 + (15 % 10)  # = 65
```

---

## Color System

Colors are directly used in RGB565 format for efficient embedded display:

### RGB565 Format
- **Bit Layout**: `RRRRRGGGGGGBBBBB` (5-6-5 bits)
- **Red**: 5 bits (0-31)
- **Green**: 6 bits (0-63) 
- **Blue**: 5 bits (0-31)
- **Storage**: 16 bits per color (50% reduction vs RGB888)

### Color Examples
Colors used in the configuration include water tones, terrain colors, and infrastructure colors:
- Water: `#aad3df` (blue tones)
- Terrain: `#fff1ba` (sandy), `#f5e9c6` (tan)
- Infrastructure: `#d6d99f` (gray tones)
- Buildings: `#e0dfdf` (light gray)
- Roads: `#c8d7ab` (dark gray), `#f2dad9` (orange)

---

## Special Metadata

The file includes two metadata fields for system configuration:

### Comment Field
```json
"_comment": "OSM-style colors in RGB565 format. Priority: layer_base + (priority % 10)."
```
- Describes the color format and priority calculation system
- Documents the layer-based priority approach

### Layers Definition
```json
"_layers": "water:10, landuse:20, terrain:30, railways:40, roads:50, infrastructure:60, buildings:70, amenities:80, places:90"
```
- Defines the 8 layers and their base priority values
- Establishes the priority structure for the entire system

---

## Example Structure

```json
{
  "_comment": "OSM-style colors in RGB565 format. Priority: layer_base + (priority % 10).",
  "_layers": "water:10, landuse:20, terrain:30, railways:40, roads:50, infrastructure:60, buildings:70, amenities:80, places:90",
  
  "natural=coastline": {
    "zoom": 6,
    "color": "#aad3df",
    "priority": 10
  },
  "natural=water": {
    "zoom": 12,
    "color": "#aad3df", 
    "priority": 11
  },
  "waterway=river": {
    "zoom": 8,
    "color": "#aad3df",
    "priority": 15
  },
  "natural=beach": {
    "zoom": 12,
    "color": "#fff1ba",
    "priority": 20
  },
  "building=yes": {
    "zoom": 14,
    "color": "#e0dfdf",
    "priority": 71
  },
  "highway=motorway": {
    "zoom": 6,
    "color": "#c8d7ab",
    "priority": 65
  },
  "highway=primary": {
    "zoom": 6,
    "color": "#f2dad9",
    "priority": 62
  },
  "railway=rail": {
    "zoom": 10,
    "color": "#d6d99f",
    "priority": 40
  },
  "amenity=hospital": {
    "zoom": 15,
    "color": "#f2c8c6",
    "priority": 80
  }
}
```

---

## How Matching Works

- During tile generation, each OSM feature is checked against the keys in the JSON:
    - If a feature has a matching tag (`key=value`), the corresponding entry is used for rendering.
    - If only the key matches (e.g., `"building"`), the entry applies for any value of that key.
    
- If a feature matches multiple entries, the most specific (key=value) takes precedence over generic key matches.

- Features are filtered by zoom level: only features with `zoom` parameter less than or equal to the current tile zoom level are included.

- Features are rendered in priority order calculated using the layer-based system, ensuring proper visual layering.

---

## Rendering Order

Features are rendered according to their calculated priority:

1. **Water layer** (10-19): Coastlines, water bodies, waterways
2. **Landuse layer** (20-29): Parks, forests, urban areas
3. **Terrain layer** (30-39): Natural features, peaks, cliffs
4. **Railway layer** (40-49): Railway tracks and infrastructure
5. **Roads layer** (50-69): All highway types with varying priorities
6. **Infrastructure layer** (60-69): Power lines, pipelines, barriers
7. **Buildings layer** (70-79): Building footprints and structures
8. **Amenities layer** (80-89): Points of interest, facilities
9. **Places layer** (90-99): Geographic labels and place names

---

## Usage Notes

- You can extend the file with new feature selectors and styling as needed.
- All parameters are optional, but `zoom` and `color` are recommended for correct rendering.
- This file drives both filtering (which features are extracted from the OSM source) and styling (how they appear in output tiles).
- The layer-based priority system ensures consistent visual hierarchy across different zoom levels.
- Colors use standard HTML hex notation and are automatically converted to RGB565 format for efficient storage in binary tiles.
- The system supports 322 feature definitions optimized for embedded navigation display.

---

## References

- [OpenStreetMap Tagging](https://wiki.openstreetmap.org/wiki/Tags)
- [Hex Color Codes](https://www.w3schools.com/colors/colors_picker.asp)
- [RGB565 Color Format](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#RGB565)

---

## Notes

- This layer-based priority system replaces simple numerical priority schemes.
- The priority calculation ensures consistent ordering across the entire feature set.
- Zoom levels are optimized for embedded systems: not all levels 6-17 are used in practice.
- RGB565 format provides 50% storage savings compared to RGB888 while maintaining good color fidelity for navigation displays.