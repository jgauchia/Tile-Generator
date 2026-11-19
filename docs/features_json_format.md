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
| `priority`| integer | Rendering priority (0-15) for line and polygon features; it becomes the lower nibble of the binary zoom_priority. Higher values draw later. |
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

## 4. Layer Priority System

The generator uses a simplified 16-level priority system (0-15). For line and polygon features, this value determines the rendering order within each of the four rendering passes; features with the same priority share a z-order level.

| Priority | Feature Type Examples |
| :--- | :--- |
| **0** | Background land. |
| **1-2** | Large areas (Aerodromes, large forests, parks, terrain). |
| **3-5** | Smaller areas (Gardens, grass, cemeteries). |
| **6-7** | Infrastructure, rock overlays, buildings and water. |
| **8** | Paths, trails and footways. |
| **9-11** | Tracks, links and local roads. |
| **12-14** | Major roads and highways. |
| **15** | Railways, Bridges, and Text Labels. |

> Paths and tracks are solid strokes: the packed format has no dash pattern.
>
> Bare rock and scree polygons use priority 6 so they stay visible above grass and forest; rock polygons keep priority 2, matching OSM at z16.
>
> `highway=footway` is included at z16 in red (`#f88576`), priority 8, with a 1 px core (2 half-pixel units). It does not use a casing. Underground footways (`tunnel=yes`, `layer<0` or `level<0`) are deliberately excluded. These settings are tag-scoped: they do not change the style or priority of other features.
>
> Waterway lines with `layer<0` are retained unless they are tagged `tunnel=yes`/`culvert` or `level<0`; a negative layer alone only changes stacking, so a river passing below a bridge is not erased. This keeps the `Riu Aixec` stretches visible while the underground drain remains excluded. `waterway=river` uses 2 half-pixel units at z10 and 6 (3 px) at z16 so the line remains legible; `waterway=stream`/`canal` also use 6 (3 px) at z16 so newly visible tributaries match the river, while drains/ditches use 2 (1 px). Water line segments with `layer<0` (a river below a bridge/deck) get at least 8 (4 px) at z16, so restored below-grade stretches such as Riu Aixec do not disappear; at-grade rivers keep their configured width.
>
> `natural=coastline` is a line in a darker blue (`#5a8ca3`, priority 8), above the water fill (`#aad2df`, priority 7),
> so the shore remains visible instead of blending into the sea. A coastline way that is also an outer member of an
> administrative boundary is emitted as coastline, not as a boundary, so it is not lost under the boundary layer.

> `highway=track` uses brown (`#996600`) at priority 9. OSM Carto draws a white fill with a dashed brown casing; the packed format has no dash pattern, so a solid brown stroke is what keeps the track visible on a light background. At z16 the stroke is 2 px (4 half-pixel units), the same width OSM draws there.
>
> `man_made=storage_tank` is treated as a building: `buildings` layer, building colour and casing, and it merges into the building landcover below z16. The post-projection minimum area still applies.
>
> `tourism=picnic_site` and `tourism=camp_site` are rendered as `leisure` areas with OSM Carto's campsite fill (`#def6c0`) from z10. The packed format has no icons, so the picnic table and tent symbols are not drawn.
>
> `leisure=dog_park` uses the lighter OSM Carto leisure fill (`#d8fdda`) from z10, priority 3. It sits above the park fill so the enclosed dog area remains visible; the striped pattern and icon are not drawn.
>
> Parking polygons (`amenity=parking`/`parking_space`) use priority 6: above grass, forest and campsites (priorities 2-5) and below buildings, water and roads (7+). This keeps parking areas visible instead of being covered by the green landcover.
>
> `man_made=pier` uses priority 8 so marina fingers stay above the water fill (priority 7). Without this, the pier lines are present in the pack but disappear under the water polygon.
>
> `landuse=orchard` and `landuse=vineyard` use priority 0: above the tile background but below residential, commercial and industrial landuse, so urban landuse covers them where the polygons overlap, as in OSM.
>
> Golf details use a dedicated `golf` layer: `golf=bunker` uses the sand fill (`#f5e9c7`, z14, priority 6), `golf=green`/`golf=tee` use the pitch green (`#a9e0cb`, z13, priority 6), and `golf=rough` uses `#c9d7ac` (z12, priority 5); `golf=fairway`/`golf=driving_range` use the grass fill (`#ceecb1`, z12, priority 5). They render above `leisure=golf_course` (priority 3), whose exposed base uses `#def6c0`; that keeps the course base lighter than the fairways and the driving range, as in OSM. Generator-only; NAV1, viewer and firmware are unchanged.
>
> `highway=tertiary` uses priority 12, the same as residential and unclassified, so a `highway=secondary` (13) is painted on top where both share a road; OSM draws secondary above tertiary. Colour and width are unchanged.
>
> `amenity=kindergarten` uses the same `#ffffe5` school fill from z13 as `amenity=school`.
>
> A polygon tagged `building=*` takes the `building` fill even when it also has `amenity` or `leisure` tags; OSM Carto fills the building and draws the amenity/leisure icon on top. For example, `building=yes` + `leisure=sports_centre` renders as a building, not as a sports-centre area.
>
> **Pending width improvement (not implemented):** for path-like highways (`path`, `track`, `footway`, `bridleway`, `cycleway`), use the OSM `width` tag when present instead of always taking `line_width_per_zoom()`. The value is already parsed into `width_meters`; it must be converted to the packed half-pixel width and clamped. Most paths have no `width`, so the table remains the fallback. The decision point is in `src/tile_processor.hpp` at the line-width block.

---

## 5. Special Handling: Administrative Boundaries

Administrative boundaries are extracted using a two-pass scanner and assigned to specific priority levels to ensure they provide context without cluttering the map.
- **Regional Borders** (admin_level < 8): Render below transport layers (Priority 3).
- **Municipal Borders** (admin_level >= 8): Render above everything (Priority 15) but using subtle, low-contrast colors.

---

## 6. Special Handling: Text Labels

Labels for places and road references are generated with collision detection:
- **Place Names**: Priority 12-15 based on population.
- **Road Shields**: Priority 15, matching the bridge/railway layer.

The final z-order is the configured 0-15 priority nibble; there is no layer-base arithmetic. Special features override it: bridges use 15, place labels use 12-15 by population, and road shields use 15.

---

## 7. Rendering Order (Bottom to Top)

1. **Background**: priority 0.
2. **Polygons and lines**: their configured priority from `features.json`.
3. **Information**: labels and municipal limits on top.
