# Features JSON Format Specification

This document describes the structure and usage of the `features.json` configuration file for vector tile generation. The file defines styling, priority, and minimum zoom levels for OpenStreetMap (OSM) tags.

---

## 1. Feature Definitions

Each entry in the JSON file is a key-value pair:
- **Key**: An OSM tag selector, either `"key=value"` (exact match) or `"key"` (matches any value). Note: Compound selectors (separated by commas) are not supported.
- **Value**: An object specifying the rendering parameters for that feature.

---

## 2. Feature Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `zoom` | integer | Minimum zoom level at which the feature is included in the packed containers. |
| `color` | string | HTML hex color (`#rrggbb`). Converted to RGB565 in the binary format. |
| `priority`| integer | Local priority (0-15). Lower nibble of the binary zoom_priority. |
| `widths` | object | **Manual Width Scaling**: Map of zoom level to pixels (0.5px units in binary). |

---

## 3. Usage Example

### Forest (Polygon)
```json
"landuse=forest": {
  "zoom": 8,
  "color": "#add19e",
  "priority": 5
}
```
*   **zoom**: Visible starting from level 8.
*   **color**: Light green.
*   **priority**: Level 5 (standard for vegetation/surfaces).

### Motorway (Line)
```json
"highway=motorway": {
  "zoom": 6,
  "color": "#e892a2",
  "priority": 14
}
```
*   **zoom**: Visible from level 6.
*   **color**: Pinkish-red (OSM standard).
*   **priority**: High (Level 14, renders above most other features).

---

## 4. Layer Priority System (v0.4.0)

The generator uses a simplified 16-level priority system (0-15). This value determines the rendering order within each of the four rendering passes. If multiple features have the same priority, their order is determined by their sequence in the PBF file.

| Priority | Feature Type Examples |
| :--- | :--- |
| **0** | Background land. |
| **1-2** | Large areas (Aerodromes, large forests, terrain). |
| **3-5** | Smaller areas (Parks, gardens, grass, cemeteries). |
| **6-7** | Infrastructure and Buildings. |
| **8** | Water (Lakes, rivers). |
| **9-11** | Links and local roads. |
| **12-14** | Major roads and highways. |
| **15** | Railways, Bridges, and Text Labels. |

---

## 5. Special Handling: Administrative Boundaries

Administrative boundaries are extracted and assigned to specific priority levels to ensure they provide context without cluttering the map.
- **Regional Borders** (admin_level < 8): Render below transport layers (Priority 3).
- **Municipal Borders** (admin_level >= 8): Render above everything (Priority 15) but using subtle, low-contrast colors.

---

## 6. Special Handling: Text Labels

Labels for places and road references are generated with collision detection:
- **Place Names**: Priority 12-15 based on population.
- **Road Shields**: Priority 15, matching the bridge/railway layer.

The generator uses a 10-layer hierarchy. Final priority (0-15) is calculated as:  
`Final Priority = (Layer_Base / 10) + (priority_from_json / 7)` (clamped to 15).

| Layer | Base Priority | Description |
| :--- | :--- | :--- |
| **landuse** | 10 | Terrain, forests, urban areas. |
| **boundaries**| 35 | Countries, regions, and provinces (Under roads). |
| **water** | 30 | Lakes and rivers. |
| **railways** | 40 | Train tracks (surface only). |
| **roads** | 50 | Streets and highways. |
| **places** | 90 | City labels and **Municipal boundaries** (Above roads). |

---

## 4. Special Handling: Administrative Boundaries

Administrative boundaries are extracted using a two-pass scanner to ensure full fidelity.
- **Regional Borders (admin_level < 8)**: Assigned to the `boundaries` layer (Base 35). They render below transport layers to avoid visual clutter.
- **Municipal Borders (admin_level >= 8)**: Assigned to the `places` layer (Base 90). They render above everything using subtle colors to provide local context without obstructing navigation.

---

## 5. Rendering Order (Bottom to Top)

1. **Background** (Base 10-30): Land, Water, Terrain.
2. **Regional Borders** (Base 35): Countries and provinces.
3. **Transport** (Base 40-50): Railways and Roads.
4. **Information** (Base 90): Labels and Municipal limits.
