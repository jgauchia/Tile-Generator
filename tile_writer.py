"""
NAV tile binary writer.

Serializes features to .nav binary format: merge polygons, clip, project.
"""

import os
import struct
import logging
from typing import Dict, List
from collections import defaultdict

from constants import (
    NAV_MAGIC, COORD_SCALE, LAND_BG_COLOR,
    GEOM_LINESTRING, GEOM_POLYGON, GEOM_TEXT,
    K_VISIBILITY, K_HOLE_FACTOR, LINE_WIDTH_PER_ZOOM,
    CLIP_MARGIN_POLYGON, CLIP_MARGIN_LINE, BRIDGE_DECK_COLOR,
)
from geo_utils import (
    SHAPELY_AVAILABLE,
    hex_to_rgb565, pack_zoom_priority, meters_to_pixels,
    lat_to_mercator_y, tile_y_to_lat,
)

logger = logging.getLogger(__name__)


def _filter_and_group_polygons(features, zoom, tile_x, tile_y):
    """Separate polygons from other features, filter by area, group by style.

    Returns (other_features, merged_features, filtered_count).
    """
    if not SHAPELY_AVAILABLE:
        return features, [], 0

    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
    from shapely.ops import unary_union as shapely_unary_union

    min_area_deg2 = 0.0
    if zoom < 14:
        zres_prev = 360.0 / (2**(zoom - 1) * 256)
        if zoom <= 7:
            multiplier = 2.5
        elif zoom == 8:
            multiplier = 1.8
        elif zoom == 9:
            multiplier = 2.5
        else:
            multiplier = 3.0
        min_area_deg2 = (zres_prev ** 2) * multiplier

    polygons_by_style = defaultdict(list)
    other_features = []
    filtered_by_area = 0

    for feat in features:
        if feat['geom_type'] == GEOM_POLYGON:
            if min_area_deg2 > 0:
                inner = feat.get('inner_rings', [])
                if inner:
                    sp = ShapelyPolygon(feat['coords'], inner)
                else:
                    sp = ShapelyPolygon(feat['coords'])
                if sp.area < min_area_deg2:
                    filtered_by_area += 1
                    continue

            subclass = feat.get('subclass', '')
            style_key = (feat['color_rgb565'], feat['zoom_priority'], subclass)
            polygons_by_style[style_key].append(feat)
        else:
            other_features.append(feat)

    if filtered_by_area > 0:
        logger.debug(f"  Tile {tile_x},{tile_y}: Filtered {filtered_by_area} polygons by area at z{zoom}")

    merged_features = []
    merge_stats = {'holes_total': 0, 'holes_removed': 0, 'sharding_fallbacks': 0, 'groups_merged': 0}

    for (color, priority, subclass), poly_list in polygons_by_style.items():
        result = _merge_polygon_group(color, priority, subclass, poly_list, zoom,
                                      tile_x, tile_y, merge_stats)
        merged_features.extend(result)

    logger.debug(f"  Tile {tile_x},{tile_y}: Merge: {merge_stats['groups_merged']} groups merged, "
                 f"{merge_stats['holes_removed']}/{merge_stats['holes_total']} holes removed, "
                 f"{merge_stats['sharding_fallbacks']} sharding fallbacks")

    return other_features, merged_features, filtered_by_area


def _merge_polygon_group(color, priority, subclass, poly_list, zoom, tile_x, tile_y, merge_stats):
    """Merge a group of same-style polygons. Returns list of feature dicts."""
    if len(poly_list) < 2:
        return poly_list

    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
    from shapely.ops import unary_union as shapely_unary_union

    try:
        shapely_polys = []
        for p in poly_list:
            inner = p.get('inner_rings', [])
            if inner:
                sp = ShapelyPolygon(p['coords'], inner)
            else:
                sp = ShapelyPolygon(p['coords'])
            if not sp.is_valid:
                sp = sp.buffer(0)
            if not sp.is_empty:
                shapely_polys.append(sp)

        if not shapely_polys:
            return poly_list

        pixel_deg = 360.0 / (2**zoom * 256)

        priority_nibble = priority & 0x0F
        is_landcover = priority_nibble <= 3
        is_building_group = poly_list[0].get('is_building', False)

        merge_landcover = is_landcover and subclass in ('wood', 'forest')
        merge_buildings = is_building_group and zoom <= 15

        if merge_landcover:
            merged = shapely_unary_union(shapely_polys)
            merged = merged.simplify(pixel_deg * 0.5, preserve_topology=True)
        elif merge_buildings:
            if zoom <= 14:
                buf_deg = pixel_deg * 3
                simplify_factor = 1.0
            else:
                buf_deg = pixel_deg * 1.5
                simplify_factor = 0.5
            buffered = [sp.buffer(buf_deg) for sp in shapely_polys]
            merged = shapely_unary_union(buffered)
            merged = merged.buffer(-buf_deg * 0.7)
            merged = merged.simplify(pixel_deg * simplify_factor, preserve_topology=True)
            logger.debug(f"  Tile {tile_x},{tile_y}: Merged {len(shapely_polys)} buildings into blocks at z{zoom}")
        else:
            return poly_list

        if merged.is_empty:
            return poly_list

        parts = []
        if isinstance(merged, ShapelyMultiPolygon):
            parts = list(merged.geoms)
        elif isinstance(merged, ShapelyPolygon):
            parts = [merged]
        elif hasattr(merged, 'geoms'):
            parts = [g for g in merged.geoms if isinstance(g, ShapelyPolygon)]

        feature_layer = poly_list[0].get('layer', '')
        total_merged_points = 0
        candidate_features = []

        for part in parts:
            if not part.is_empty and part.exterior and len(part.exterior.coords) >= 4:
                if feature_layer == 'water':
                    inner_rings = [list(interior.coords) for interior in part.interiors if len(interior.coords) >= 4]
                    for interior in part.interiors:
                        merge_stats['holes_total'] += 1
                else:
                    inner_rings = []
                    for interior in part.interiors:
                        merge_stats['holes_total'] += 1
                        merge_stats['holes_removed'] += 1

                ext_coords = list(part.exterior.coords)
                total_merged_points += len(ext_coords)
                candidate_features.append({
                    'geom_type': GEOM_POLYGON,
                    'coords': ext_coords,
                    'inner_rings': inner_rings,
                    'color_rgb565': color,
                    'zoom_priority': priority,
                    'width_meters': 0.0,
                    'subclass': subclass,
                    'layer': feature_layer,
                    'is_building': feature_layer == 'buildings',
                    'name': '',
                    'id': 0,
                })

        if total_merged_points > 65535:
            merge_stats['sharding_fallbacks'] += 1
            return poly_list

        merge_stats['groups_merged'] += 1
        return candidate_features
    except Exception:
        return poly_list


def _generate_bridge_underlays(features, zoom):
    """Create grey deck polygons under bridge segments, grouped by category (road/rail)."""
    if not SHAPELY_AVAILABLE or zoom < 16:
        return []

    from shapely.geometry import (
        LineString as ShapelyLineString,
        Polygon as ShapelyPolygon,
        MultiPolygon as ShapelyMultiPolygon,
    )
    from shapely.ops import unary_union as shapely_unary_union

    BRIDGE_COLOR_RGB565 = hex_to_rgb565(BRIDGE_DECK_COLOR)
    ROAD_TYPES = {
        'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
        'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
        'residential', 'unclassified', 'living_street', 'pedestrian',
    }
    RAIL_TYPES = {'rail', 'narrow_gauge', 'funicular', 'tram', 'light_rail'}
    pixel_deg = 360.0 / (2**zoom * 256)

    groups = {'road': [], 'rail': []}
    for feature in features:
        if not feature.get('is_bridge') or feature['geom_type'] != GEOM_LINESTRING:
            continue
        hw_type = feature.get('highway_type', '')
        if hw_type in ROAD_TYPES:
            cat = 'road'
        elif hw_type in RAIL_TYPES:
            cat = 'rail'
        else:
            continue
        road_width = LINE_WIDTH_PER_ZOOM.get(hw_type, {}).get(zoom, 1)
        rail_extra = 2.0 if cat == 'rail' else 1.0
        tight_buf = pixel_deg * (road_width * 0.4 + rail_extra)
        extra = 5.0 * (1 << (zoom - 16))
        generous_buf = pixel_deg * (road_width / 4.0 + extra)
        try:
            line = ShapelyLineString(feature['coords'])
            groups[cat].append({
                'tight': line.buffer(tight_buf, cap_style='flat'),
                'generous': line.buffer(generous_buf, cap_style='flat'),
            })
        except Exception:
            pass

    result = []
    for cat, bridge_buffers in groups.items():
        if not bridge_buffers:
            continue

        all_generous = shapely_unary_union([b['generous'] for b in bridge_buffers])
        all_tight = [b['tight'] for b in bridge_buffers]

        generous_parts = []
        if isinstance(all_generous, ShapelyPolygon):
            generous_parts = [all_generous]
        elif isinstance(all_generous, ShapelyMultiPolygon):
            generous_parts = list(all_generous.geoms)
        elif hasattr(all_generous, 'geoms'):
            generous_parts = [g for g in all_generous.geoms if isinstance(g, ShapelyPolygon)]

        for gp in generous_parts:
            group_tight = [t for t in all_tight if t.intersects(gp)]
            if not group_tight:
                continue
            tight_union = shapely_unary_union(group_tight)
            if isinstance(tight_union, ShapelyMultiPolygon) or (hasattr(tight_union, 'geoms') and len(list(tight_union.geoms)) > 1):
                deck = tight_union.convex_hull
            elif isinstance(tight_union, ShapelyPolygon):
                deck = tight_union
            else:
                continue
            if deck.is_empty or not deck.exterior:
                continue
            result.append({
                'geom_type': GEOM_POLYGON,
                'coords': list(deck.exterior.coords),
                'inner_rings': [],
                'color_rgb565': BRIDGE_COLOR_RGB565,
                'zoom_priority': pack_zoom_priority(zoom, 14),
                'layer': 'infrastructure',
                '_bridge_underlay': True,
            })

    logger.debug(f"  Created {len(result)} bridge underlay polygons")
    return result


def _filter_holes(inner_rings, layer, zoom):
    """Filter small holes from polygon inner rings.

    Returns (filtered_rings, removed_count).
    """
    if not inner_rings or not SHAPELY_AVAILABLE:
        return inner_rings, 0

    if layer == 'water':
        return inner_rings, 0

    from shapely.geometry import Polygon
    pixel_deg = 360.0 / (2**zoom * 256)
    min_hole_pixels_sq = K_VISIBILITY * 1.5 if zoom >= 13 else K_VISIBILITY * 10.0
    min_hole_area_deg2 = (pixel_deg ** 2) * min_hole_pixels_sq

    filtered = []
    removed = 0
    for interior in inner_rings:
        try:
            hole_poly = Polygon(interior)
            if hole_poly.area >= min_hole_area_deg2:
                filtered.append(interior)
            else:
                removed += 1
        except Exception:
            removed += 1

    return filtered, removed


def _clip_geometry(orig_coords, inner_rings, is_polygon, feature_layer, clip_box, clip_box_line,
                   tolerance, is_bridge_deck, zoom):
    """Clip geometry to tile bounds. Returns list of ring-lists for each resulting part."""
    active_clip_box = clip_box_line if (not is_polygon or is_bridge_deck) and clip_box_line else clip_box

    if not active_clip_box:
        if is_polygon and inner_rings:
            return [[orig_coords] + inner_rings]
        return [[orig_coords]]

    try:
        from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection

        if is_polygon:
            if inner_rings:
                geom = Polygon(orig_coords, inner_rings)
            else:
                geom = Polygon(orig_coords)
            if not geom.is_valid:
                geom = geom.buffer(0)
        else:
            if len(orig_coords) < 2:
                return []
            geom = LineString(orig_coords)

        if geom is None or geom.is_empty:
            return []

        clipped = geom.intersection(active_clip_box)
        if clipped.is_empty:
            return []

        parts = list(clipped.geoms) if isinstance(clipped, GeometryCollection) else [clipped]
        result = []

        for part in parts:
            if is_polygon:
                if isinstance(part, (Polygon, MultiPolygon)):
                    polys = [part] if isinstance(part, Polygon) else list(part.geoms)
                    for p in polys:
                        if not p.is_empty and p.exterior and len(p.exterior.coords) >= 4:
                            if feature_layer in ('landuse', 'terrain') and zoom < 14:
                                simplified_poly = p.simplify(tolerance, preserve_topology=True)
                            else:
                                simplified_poly = p
                            if simplified_poly.is_empty or not simplified_poly.exterior:
                                continue
                            rings = [list(simplified_poly.exterior.coords)]
                            for interior in simplified_poly.interiors:
                                if len(interior.coords) >= 4:
                                    rings.append(list(interior.coords))
                            result.append(rings)
            else:
                lines = []
                if isinstance(part, LineString):
                    lines = [part]
                elif isinstance(part, MultiLineString):
                    lines = list(part.geoms)

                for l in lines:
                    if len(l.coords) < 2:
                        continue
                    if feature_layer in ('water', 'roads'):
                        simplified = l
                    elif feature_layer == 'infrastructure':
                        pixel_deg = 360.0 / (2**zoom * 256)
                        simplified = l.simplify(pixel_deg * 0.1, preserve_topology=True)
                    else:
                        simplified = l.simplify(tolerance, preserve_topology=True)
                    if len(simplified.coords) >= 2:
                        result.append([list(simplified.coords)])

        return result
    except Exception:
        if is_polygon:
            return [[orig_coords] + inner_rings]
        return [[orig_coords]]


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _to_varint(value: int) -> bytearray:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return out


def _project_and_write(f, feature, feature_rings, is_polygon, feature_layer,
                       tile_bounds, merc_bounds, zoom):
    """Project rings to tile coordinates and write binary data.

    Returns (written, filtered_by_size) counts.
    """
    tile_min_lon, tile_max_lon, tile_min_lat, tile_max_lat = tile_bounds
    t_max_merc, t_min_merc, merc_range = merc_bounds

    projected_rings = []
    total_points = 0
    f_min_x, f_min_y = 4096, 4096
    f_max_x, f_max_y = 0, 0

    for ring in feature_rings:
        projected_ring = []
        for lon, lat in ring:
            px = int((lon - tile_min_lon) / (tile_max_lon - tile_min_lon) * 4096)
            m_y = lat_to_mercator_y(lat)
            py = int((t_max_merc - m_y) / merc_range * 4096)
            projected_ring.append((px, py))
            c_px, c_py = max(0, min(4096, px)), max(0, min(4096, py))
            f_min_x, f_min_y = min(f_min_x, c_px), min(f_min_y, c_py)
            f_max_x, f_max_y = max(f_max_x, c_px), max(f_max_y, c_py)

        if len(projected_ring) >= (3 if is_polygon else 2):
            projected_rings.append(projected_ring)
            total_points += len(projected_ring)

    if is_polygon:
        pixel_area = (f_max_x - f_min_x) * (f_max_y - f_min_y) / (16 * 16)
        if feature_layer != 'water':
            if zoom <= 7:
                min_area = K_VISIBILITY * 8
            elif zoom == 8:
                min_area = K_VISIBILITY * 6
            elif zoom == 9:
                min_area = K_VISIBILITY * 8
            elif zoom <= 11:
                min_area = K_VISIBILITY * 8
            elif zoom == 12:
                min_area = K_VISIBILITY * 5
            elif zoom == 13:
                min_area = K_VISIBILITY * 2
            elif zoom == 14:
                min_area = K_VISIBILITY * 0.5
            else:
                min_area = K_VISIBILITY * 0.1
            if pixel_area < min_area:
                return 0, 1

    if total_points > 65535:
        logger.warning(f"  SKIPPING feature with {total_points} points (limit 65535). Type={feature.get('geom_type')}")
        return 0, 0

    if not projected_rings:
        return 0, 0

    width_pixels = feature.get('width_pixels', 0)
    if width_pixels == 0:
        hw_type = feature.get('highway_type', '')
        if hw_type and hw_type in LINE_WIDTH_PER_ZOOM:
            width_pixels = LINE_WIDTH_PER_ZOOM[hw_type].get(zoom, 1)
        else:
            width_meters = feature.get('width_meters', 0.0)
            width_pixels = meters_to_pixels(width_meters, zoom) if width_meters > 0 else 1

    needs_casing = feature.get('is_bridge', False) and zoom >= 14

    width_byte = min(width_pixels, 127)
    if is_polygon:
        width_byte = 0
        if feature.get('is_building', False) and zoom >= 16:
            width_byte |= 0x80
    elif needs_casing:
        width_byte |= 0x80

    bx1, by1 = max(0, min(255, f_min_x >> 4)), max(0, min(255, f_min_y >> 4))
    bx2, by2 = max(0, min(255, f_max_x >> 4)), max(0, min(255, f_max_y >> 4))

    coord_buffer = bytearray()
    last_x, last_y = 0, 0
    for ring in projected_rings:
        for px, py in ring:
            px_clamped = max(-32768, min(32767, px))
            py_clamped = max(-32768, min(32767, py))
            dx = px_clamped - last_x
            dy = py_clamped - last_y
            coord_buffer.extend(_to_varint(_zigzag_encode(dx)))
            coord_buffer.extend(_to_varint(_zigzag_encode(dy)))
            last_x, last_y = px_clamped, py_clamped

    extra_payload = bytearray()
    if is_polygon:
        extra_payload.extend(struct.pack('<H', len(projected_rings)))
        current_end = 0
        for ring in projected_rings:
            current_end += len(ring)
            extra_payload.extend(struct.pack('<H', current_end))

    payload_size = len(coord_buffer) + len(extra_payload)

    f.write(struct.pack('<B', feature['geom_type']))
    f.write(struct.pack('<H', feature['color_rgb565']))
    f.write(struct.pack('<B', feature['zoom_priority']))
    f.write(struct.pack('<B', width_byte))
    f.write(struct.pack('<BBBB', bx1, by1, bx2, by2))
    f.write(struct.pack('<H', total_points))
    f.write(struct.pack('<H', payload_size))

    f.write(coord_buffer)
    f.write(extra_payload)

    return 1, 0


def write_nav_tile(features: List[Dict], output_path: str, zoom: int, tile_x: int, tile_y: int, tolerance: float) -> bool:
    """Write features to NAV binary tile format using relative coordinates."""
    n = 2.0 ** zoom
    lon_deg_per_tile = 360.0 / n
    tile_min_lon = -180.0 + tile_x * lon_deg_per_tile
    tile_max_lon = tile_min_lon + lon_deg_per_tile

    tile_max_lat = tile_y_to_lat(tile_y, zoom)
    tile_min_lat = tile_y_to_lat(tile_y + 1, zoom)

    t_max_merc = lat_to_mercator_y(tile_max_lat)
    t_min_merc = lat_to_mercator_y(tile_min_lat)
    merc_range = t_max_merc - t_min_merc

    tile_bounds = (tile_min_lon, tile_max_lon, tile_min_lat, tile_max_lat)
    merc_bounds = (t_max_merc, t_min_merc, merc_range)

    # Build clip boxes
    clip_box = None
    clip_box_line = None
    if SHAPELY_AVAILABLE:
        from shapely.geometry import box
        poly_lon_m = (tile_max_lon - tile_min_lon) * CLIP_MARGIN_POLYGON
        poly_lat_m = (tile_max_lat - tile_min_lat) * CLIP_MARGIN_POLYGON
        line_lon_m = (tile_max_lon - tile_min_lon) * CLIP_MARGIN_LINE
        line_lat_m = (tile_max_lat - tile_min_lat) * CLIP_MARGIN_LINE
        clip_box = box(tile_min_lon - poly_lon_m, tile_min_lat - poly_lat_m,
                       tile_max_lon + poly_lon_m, tile_max_lat + poly_lat_m)
        clip_box_line = box(tile_min_lon - line_lon_m, tile_min_lat - line_lat_m,
                            tile_max_lon + line_lon_m, tile_max_lat + line_lat_m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Filter/group/merge polygons
    other_features, merged_features, _ = _filter_and_group_polygons(features, zoom, tile_x, tile_y)
    features = other_features + merged_features

    # 2. Bridge underlays
    bridge_features = _generate_bridge_underlays(features, zoom)
    features.extend(bridge_features)

    # 3. Sort by priority nibble
    features.sort(key=lambda f: f['zoom_priority'] & 0x0F)

    # 4. Write binary file
    written_features = 0
    filtered_by_size = 0
    filtered_holes_write = 0
    total_holes_write = 0

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<4sHiiii', NAV_MAGIC, 0,
                           int(tile_min_lon * COORD_SCALE),
                           int(tile_min_lat * COORD_SCALE),
                           int(tile_max_lon * COORD_SCALE),
                           int(tile_max_lat * COORD_SCALE)))

        # Background land polygon
        bg_points = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]
        bg_coord_buf = bytearray()
        bg_last_x, bg_last_y = 0, 0
        for px, py in bg_points:
            dx = px - bg_last_x
            dy = py - bg_last_y
            bg_coord_buf.extend(_to_varint(_zigzag_encode(dx)))
            bg_coord_buf.extend(_to_varint(_zigzag_encode(dy)))
            bg_last_x, bg_last_y = px, py
        bg_extra = struct.pack('<H', 1) + struct.pack('<H', 5)
        bg_payload_size = len(bg_coord_buf) + len(bg_extra)
        f.write(struct.pack('<B', GEOM_POLYGON))
        f.write(struct.pack('<H', hex_to_rgb565(LAND_BG_COLOR)))
        f.write(struct.pack('<B', pack_zoom_priority(0, 0)))
        f.write(struct.pack('<B', 1))
        f.write(struct.pack('<BBBB', 0, 0, 255, 255))
        f.write(struct.pack('<H', 5))
        f.write(struct.pack('<H', bg_payload_size))
        f.write(bg_coord_buf)
        f.write(bg_extra)
        written_features += 1

        for feature in features:
            if written_features >= 65534:
                logger.warning(f"  Tile {tile_x},{tile_y} z{zoom}: HIT FEATURE LIMIT (65534)! Truncating.")
                break

            # Text features
            if feature['geom_type'] == GEOM_TEXT:
                lon, lat = feature['coords'][0]
                px = int((lon - tile_min_lon) / (tile_max_lon - tile_min_lon) * 4096)
                m_y = lat_to_mercator_y(lat)
                py = int((t_max_merc - m_y) / merc_range * 4096)

                if not (-8192 < px < 12288 and -8192 < py < 12288):
                    continue

                text_bytes = feature['text']
                text_len = len(text_bytes)
                has_shield = 'bg_color_rgb565' in feature
                data_size = 4 + 1 + text_len + (4 if has_shield else 0)
                coord_count = (data_size + 3) // 4
                padded_size = coord_count * 4

                bx = max(0, min(255, px >> 4))
                by = max(0, min(255, py >> 4))

                text_payload = bytearray()
                text_payload.extend(struct.pack('<hh', px, py))
                text_payload.extend(struct.pack('<B', text_len))
                text_payload.extend(text_bytes)
                if has_shield:
                    text_payload.extend(struct.pack('<H', feature['bg_color_rgb565']))
                    text_payload.extend(struct.pack('<H', feature['border_color_rgb565']))
                padding = padded_size - data_size
                if padding > 0:
                    text_payload.extend(b'\x00' * padding)

                f.write(struct.pack('<B', GEOM_TEXT))
                f.write(struct.pack('<H', feature['color_rgb565']))
                f.write(struct.pack('<B', feature['zoom_priority']))
                f.write(struct.pack('<B', feature.get('font_size', 0)))
                f.write(struct.pack('<BBBB', bx, by, bx, by))
                f.write(struct.pack('<H', coord_count))
                f.write(struct.pack('<H', len(text_payload)))
                f.write(text_payload)

                written_features += 1
                continue

            # Geometry features (polygons, lines)
            orig_coords = feature['coords']
            inner_rings = feature.get('inner_rings', [])
            is_polygon = feature['geom_type'] == GEOM_POLYGON
            feature_layer = feature.get('layer', '')

            if is_polygon and inner_rings and SHAPELY_AVAILABLE:
                total_holes_write += len(inner_rings)
                inner_rings, removed = _filter_holes(inner_rings, feature_layer, zoom)
                filtered_holes_write += removed

            is_bridge_deck = feature.get('_bridge_underlay', False)
            ring_lists = _clip_geometry(orig_coords, inner_rings, is_polygon, feature_layer,
                                        clip_box, clip_box_line, tolerance, is_bridge_deck, zoom)

            for feature_rings in ring_lists:
                if written_features >= 65534:
                    break
                w, filt = _project_and_write(f, feature, feature_rings, is_polygon, feature_layer,
                                             tile_bounds, merc_bounds, zoom)
                written_features += w
                filtered_by_size += filt

        f.seek(4)
        f.write(struct.pack('<H', written_features))

    logger.debug(f"  Tile {tile_x},{tile_y}: Write: {written_features} features, "
                 f"{filtered_by_size} filtered by area (<{K_VISIBILITY}px2), "
                 f"{filtered_holes_write}/{total_holes_write} holes removed (<{K_VISIBILITY * K_HOLE_FACTOR}px2)")

    return True
