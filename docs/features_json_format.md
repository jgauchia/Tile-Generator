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
| `priority`| integer | Local priority (0-99). Used to calculate the final rendering layer. |
| `widths` | object | **Manual Width Scaling**: Map of zoom level to exact pixel width. |

---

## 3. Layer Priority System

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
