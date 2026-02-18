"""
NAV tile binary writer.

Serializes features to .nav binary format: merge polygons, clip, project.
"""

import os
import math
import struct
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

from constants import (
    NAV_MAGIC, COORD_SCALE, LAND_BG_COLOR,
    GEOM_POINT, GEOM_LINESTRING, GEOM_POLYGON, GEOM_TEXT,
    K_VISIBILITY, K_HOLE_FACTOR, LINE_WIDTH_PER_ZOOM,
)
from geo_utils import (
    SHAPELY_AVAILABLE,
    hex_to_rgb565, pack_zoom_priority, meters_to_pixels,
)

logger = logging.getLogger(__name__)


def write_nav_tile(features: List[Dict], output_path: str, zoom: int, tile_x: int, tile_y: int, tolerance: float) -> bool:
    """
    Write features to NAV binary tile format using relative coordinates.
    """
    # Calculate tile bounds
    n = 2.0 ** zoom
    lon_deg_per_tile = 360.0 / n
    tile_min_lon = -180.0 + tile_x * lon_deg_per_tile
    tile_max_lon = tile_min_lon + lon_deg_per_tile

    def lat_to_merc(l):
        r = math.radians(l)
        r = max(-0.999 * math.pi / 2, min(0.999 * math.pi / 2, r))
        return math.log(math.tan(r) + (1.0 / math.cos(r)))

    def lat_from_tile_y(y, z):
        n = 2.0 ** z
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad)

    tile_max_lat = lat_from_tile_y(tile_y, zoom)
    tile_min_lat = lat_from_tile_y(tile_y + 1, zoom)

    t_max_merc = lat_to_merc(tile_max_lat)
    t_min_merc = lat_to_merc(tile_min_lat)
    merc_range = t_max_merc - t_min_merc

    # Clipping box with margins: 10% for polygons, 100% for linestrings (long runways)
    poly_margin = 0.10  # Small margin for polygons to avoid artifacts
    line_margin = 1.0   # 100% = 1 full tile margin for runways spanning 8-12 tiles

    poly_lon_margin = (tile_max_lon - tile_min_lon) * poly_margin
    poly_lat_margin = (tile_max_lat - tile_min_lat) * poly_margin
    line_lon_margin = (tile_max_lon - tile_min_lon) * line_margin
    line_lat_margin = (tile_max_lat - tile_min_lat) * line_margin

    clip_box = None
    clip_box_line = None
    if SHAPELY_AVAILABLE:
        from shapely.geometry import box, Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection
        clip_box = box(tile_min_lon - poly_lon_margin, tile_min_lat - poly_lat_margin,
                       tile_max_lon + poly_lon_margin, tile_max_lat + poly_lat_margin)
        clip_box_line = box(tile_min_lon - line_lon_margin, tile_min_lat - line_lat_margin,
                            tile_max_lon + line_lon_margin, tile_max_lat + line_lat_margin)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Merge polygons of the same style to reduce feature count
    if SHAPELY_AVAILABLE:
        from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
        from shapely.ops import unary_union as shapely_unary_union

        # Area filter thresholds (applied to ALL polygons before grouping)
        # OpenMapTiles formula with zoom-adapted multipliers
        min_area_deg2 = 0.0
        if zoom < 14:
            zres_prev = 360.0 / (2**(zoom - 1) * 256)
            if zoom <= 7:
                multiplier = 2.5
            elif zoom == 8:
                multiplier = 1.8  # z8 : tres permissif pour voir plus de landuse
            elif zoom == 9:
                multiplier = 2.5  # z9 : garde le bon niveau actuel
            else:
                multiplier = 3.0
            min_area_deg2 = (zres_prev ** 2) * multiplier

        polygons_by_style = defaultdict(list)
        other_features = []
        filtered_by_area = 0

        for feat in features:
            if feat['geom_type'] == GEOM_POLYGON:
                # Apply area filter to ALL polygons (not just grouped ones)
                if min_area_deg2 > 0:
                    inner = feat.get('inner_rings', [])
                    if inner:
                        sp = ShapelyPolygon(feat['coords'], inner)
                    else:
                        sp = ShapelyPolygon(feat['coords'])
                    if sp.area < min_area_deg2:
                        filtered_by_area += 1
                        continue  # Skip small polygons

                # Group by color, priority AND subclass to separate wood/forest from farmland
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
            if len(poly_list) < 2:
                merged_features.extend(poly_list)
                continue
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
                    merged_features.extend(poly_list)
                    continue

                pixel_deg = 360.0 / (2**zoom * 256)

                # Extract priority nibble from packed byte (zoom_priority = zoom<<4 | prio)
                priority_nibble = priority & 0x0F
                # landuse(1-2), terrain(2-3) only — NOT water(4-5) to avoid flooding
                is_landcover = priority_nibble <= 3

                # OpenMapTiles-style merge: only wood/forest, keep farmland/grass individual
                should_merge = is_landcover and subclass in ('wood', 'forest')

                if should_merge:
                    # Merge wood/forest to reduce fragmentation
                    merged = shapely_unary_union(shapely_polys)
                    # Simplify merged result (merge creates complex polygons with too many vertices)
                    merged = merged.simplify(pixel_deg * 0.5, preserve_topology=True)
                else:
                    # Keep individual: farmland, grass, water, roads
                    merged_features.extend(poly_list)
                    continue

                parts = []
                if isinstance(merged, ShapelyMultiPolygon):
                    parts = list(merged.geoms)
                elif isinstance(merged, ShapelyPolygon):
                    parts = [merged]

                min_hole_deg2 = (pixel_deg ** 2) * K_VISIBILITY * K_HOLE_FACTOR
                total_merged_points = 0

                # Get layer from first feature in group
                feature_layer = poly_list[0].get('layer', '')

                candidate_features = []
                for part in parts:
                    if not part.is_empty and part.exterior and len(part.exterior.coords) >= 4:
                        # Keep inner_rings for water (islands), strip for landcover (pitting)
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
                        pt_count = len(ext_coords)
                        total_merged_points += pt_count
                        candidate_features.append({
                            'geom_type': GEOM_POLYGON,
                            'coords': ext_coords,
                            'inner_rings': inner_rings,
                            'color_rgb565': color,
                            'zoom_priority': priority,
                            'width_meters': 0.0,
                            'subclass': subclass,  # Preserve subclass after merge
                            'layer': feature_layer,  # Preserve layer
                            'is_building': feature_layer == 'buildings',
                            'name': '',
                            'id': 0,
                        })

                if total_merged_points > 65535:
                    # Merge too complex, keep original separate polygons
                    merge_stats['sharding_fallbacks'] += 1
                    merged_features.extend(poly_list)
                else:
                    merge_stats['groups_merged'] += 1
                    merged_features.extend(candidate_features)
            except Exception:
                merged_features.extend(poly_list)

        features = other_features + merged_features
        logger.debug(f"  Tile {tile_x},{tile_y}: Merge: {merge_stats['groups_merged']} groups merged, "
                     f"{merge_stats['holes_removed']}/{merge_stats['holes_total']} holes removed, "
                     f"{merge_stats['sharding_fallbacks']} sharding fallbacks")

    # Final sort by priority nibble to ensure strict rendering order on device.
    # This is the most critical step for correct Z-ordering.
    features.sort(key=lambda f: f['zoom_priority'] & 0x0F)

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

        # Background land polygon covering the entire tile
        bg_points = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]
        f.write(struct.pack('<B', GEOM_POLYGON))       # type
        f.write(struct.pack('<H', hex_to_rgb565(LAND_BG_COLOR)))  # color
        f.write(struct.pack('<B', pack_zoom_priority(0, 0)))  # lowest priority (Z=0)
        f.write(struct.pack('<B', 1))                   # width
        f.write(struct.pack('<BBBB', 0, 0, 255, 255))  # bbox = full tile
        f.write(struct.pack('<H', 5))                   # 5 points
        f.write(b'\x00')                                # reserved
        for px, py in bg_points:
            f.write(struct.pack('<hh', px, py))
        f.write(struct.pack('<H', 1))                   # 1 ring
        f.write(struct.pack('<H', 5))                   # ring end at point 5
        written_features += 1

        MAX_TILE_BYTES = 256 * 1024  # 256 KB max per tile for ESP32 read performance

        for feature in features:
            if written_features >= 65534:
                logger.warning(f"  Tile {tile_x},{tile_y} z{zoom}: HIT FEATURE LIMIT (65534)! Truncating.")
                break
            if f.tell() >= MAX_TILE_BYTES:
                logger.info(f"  Tile {tile_x},{tile_y} z{zoom}: Size cap ({MAX_TILE_BYTES//1024} KB) at {written_features} features.")
                break
            # Handle text features separately
            if feature['geom_type'] == GEOM_TEXT:
                lon, lat = feature['coords'][0]
                px = int((lon - tile_min_lon) / (tile_max_lon - tile_min_lon) * 4096)
                m_y = lat_to_merc(lat)
                py = int((t_max_merc - m_y) / merc_range * 4096)

                if not (-8192 < px < 12288 and -8192 < py < 12288):
                    continue

                text_bytes = feature['text']
                text_len = len(text_bytes)
                has_shield = 'bg_color_rgb565' in feature
                # data_size: x,y + text_len + text + (shield colors if present)
                data_size = 4 + 1 + text_len + (4 if has_shield else 0)
                coord_count = (data_size + 3) // 4
                padded_size = coord_count * 4

                bx = max(0, min(255, px >> 4))
                by = max(0, min(255, py >> 4))

                # Header
                f.write(struct.pack('<B', GEOM_TEXT))
                f.write(struct.pack('<H', feature['color_rgb565']))
                f.write(struct.pack('<B', feature['zoom_priority']))
                f.write(struct.pack('<B', feature.get('font_size', 0)))
                f.write(struct.pack('<BBBB', bx, by, bx, by))
                f.write(struct.pack('<H', coord_count))
                f.write(struct.pack('<B', 1 if has_shield else 0))  # Shield flag

                # Data: position + text + shield colors
                f.write(struct.pack('<hh', px, py))
                f.write(struct.pack('<B', text_len))
                f.write(text_bytes)
                if has_shield:
                    f.write(struct.pack('<H', feature['bg_color_rgb565']))
                    f.write(struct.pack('<H', feature['border_color_rgb565']))
                # Pad
                padding = padded_size - data_size
                if padding > 0:
                    f.write(b'\x00' * padding)

                written_features += 1
                continue

            orig_coords = feature['coords']
            inner_rings = feature.get('inner_rings', [])
            is_polygon = feature['geom_type'] == GEOM_POLYGON

            feature_layer = feature.get('layer', '')

            if is_polygon and inner_rings and SHAPELY_AVAILABLE:
                total_holes_write += len(inner_rings)

                # For water, always keep holes (islands)
                if feature_layer != 'water':
                    # For other layers, filter holes by visible area at the current zoom
                    pixel_deg = 360.0 / (2**zoom * 256)

                    # More permissive for z13+ to avoid "blob" effect on residential areas
                    min_hole_pixels_sq = K_VISIBILITY * 1.5 if zoom >= 13 else K_VISIBILITY * 10.0
                    min_hole_area_deg2 = (pixel_deg ** 2) * min_hole_pixels_sq

                    filtered_inner_rings = []
                    for interior in inner_rings:
                        try:
                            from shapely.geometry import Polygon
                            hole_poly = Polygon(interior)
                            if hole_poly.area >= min_hole_area_deg2:
                                filtered_inner_rings.append(interior)
                            else:
                                filtered_holes_write += 1
                        except Exception:
                            # Invalid hole geometry, discard
                            filtered_holes_write += 1

                    inner_rings = filtered_inner_rings

            # Each entry will be a list of rings: [ [ext_pts], [hole1_pts], ... ]
            final_features_data = []

            # Clip geometry (polygons with small margin, linestrings with large margin)
            active_clip_box = clip_box_line if (not is_polygon and clip_box_line) else clip_box
            if active_clip_box:
                try:
                    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, GeometryCollection

                    # 1. Create the appropriate Shapely geometry
                    if is_polygon:
                        if inner_rings:
                            geom = Polygon(orig_coords, inner_rings)
                        else:
                            geom = Polygon(orig_coords)
                        # Only repair Polygons (buffer(0) fixes self-intersections)
                        if not geom.is_valid:
                            geom = geom.buffer(0)
                    else:
                        # For roads/lines, NEVER use buffer(0) as it destroys the geometry
                        if len(orig_coords) < 2:
                            continue
                        geom = LineString(orig_coords)

                    if geom is None or geom.is_empty:
                        continue

                    # 2. Perform clipping (intersection with the tile bounding box)
                    clipped = geom.intersection(active_clip_box)

                    if clipped.is_empty:
                        continue

                    # 3. Extract parts from the result (handles MultiLineStrings and GeometryCollections)
                    parts = []
                    if isinstance(clipped, GeometryCollection):
                        parts = list(clipped.geoms)
                    else:
                        parts = [clipped]

                    for part in parts:
                        if is_polygon:
                            # Process Polygon results
                            if isinstance(part, (Polygon, MultiPolygon)):
                                polys = [part] if isinstance(part, Polygon) else list(part.geoms)
                                for p in polys:
                                    if not p.is_empty and p.exterior and len(p.exterior.coords) >= 4:
                                        # Simplify polygons depending on zoom level
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
                                        final_features_data.append(rings)
                        else:
                            # Process LineString results (Roads, Rivers, etc.)
                            lines = []
                            if isinstance(part, LineString):
                                lines = [part]
                            elif isinstance(part, MultiLineString):
                                lines = list(part.geoms)

                            for l in lines:
                                if len(l.coords) < 2:
                                    continue

                                # FIX: Removed hardcoded 0.25 (which meant 27km in degrees)
                                # We keep full detail for roads and water, or use tile-relative tolerance
                                if feature_layer in ('water', 'roads'):
                                    simplified = l
                                elif feature_layer == 'infrastructure':
                                    # Use a microscopic tolerance for noise removal at high zooms
                                    pixel_deg = 360.0 / (2**zoom * 256)
                                    simplified = l.simplify(pixel_deg * 0.1, preserve_topology=True)
                                else:
                                    simplified = l.simplify(tolerance, preserve_topology=True)

                                if len(simplified.coords) >= 2:
                                    final_features_data.append([list(simplified.coords)])
                except Exception as e:
                    # Fallback: if clipping fails, use original coordinates to avoid data loss
                    if is_polygon:
                        final_features_data.append([orig_coords] + inner_rings)
                    else:
                        final_features_data.append([orig_coords])
            else:
                # No clipping active: use the original geometry
                if is_polygon and inner_rings:
                    final_features_data = [[orig_coords] + inner_rings]
                else:
                    final_features_data = [[orig_coords]]

            # Project and write the features
            for feature_rings in final_features_data:
                # Project all rings for this feature part
                projected_rings = []
                total_points = 0
                f_min_x, f_min_y = 4096, 4096
                f_max_x, f_max_y = 0, 0

                for ring in feature_rings:
                    projected_ring = []
                    for lon, lat in ring:
                        px = int((lon - tile_min_lon) / (tile_max_lon - tile_min_lon) * 4096)
                        m_y = lat_to_merc(lat)
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

                    # Do NOT filter water by pixel area - keep all river segments
                    if feature_layer != 'water':
                        if zoom <= 7:
                            min_area = K_VISIBILITY * 8
                        elif zoom == 8:
                            min_area = K_VISIBILITY * 6  # z8 : tres permissif (12 pixels2)
                        elif zoom == 9:
                            min_area = K_VISIBILITY * 8  # z9 : garde le bon niveau actuel
                        elif zoom <= 11:
                            min_area = K_VISIBILITY * 8
                        elif zoom == 12:
                            min_area = K_VISIBILITY * 5
                        elif zoom == 13:
                            min_area = K_VISIBILITY * 2
                        elif zoom == 14:
                            min_area = K_VISIBILITY * 0.5
                        else:  # z15-16
                            min_area = K_VISIBILITY * 0.1  # 0.2 px2 - capture everything
                        if pixel_area < min_area:
                            filtered_by_size += 1
                            continue

                # Hard limit: skip features exceeding uint16 capacity
                # Impossible to render on ESP32 and would corrupt binary format
                if total_points > 65535:
                    logger.warning(f"  Tile {tile_x},{tile_y} z{zoom}: SKIPPING feature with {total_points} points (limit 65535). Type={feature.get('geom_type')}")
                    continue

                width_pixels = feature.get('width_pixels', 0)
                if width_pixels == 0:
                    # Use fixed road width table for highways
                    hw_type = feature.get('highway_type', '')
                    if hw_type and hw_type in LINE_WIDTH_PER_ZOOM:
                        width_pixels = LINE_WIDTH_PER_ZOOM[hw_type].get(zoom, 1)
                    else:
                        width_meters = feature.get('width_meters', 0.0)
                        width_pixels = meters_to_pixels(width_meters, zoom) if width_meters > 0 else 1

                # Mark roads that need casing (border rendering) based on priority nibble
                priority_nibble = feature['zoom_priority'] & 0x0F
                needs_casing = priority_nibble in (13, 14) or feature.get('is_bridge', False)

                # Encode width/flags byte (fp[4]):
                # Lines: bits 0-6 = width in half-pixels (firmware divides by 2.0f)
                # Polygons: bit 7 = hasOutline (buildings)
                width_byte = min(width_pixels, 127)  # Clamp to 7 bits (0-63.5px range)
                if is_polygon:
                    width_byte = 0
                    if feature.get('is_building', False):
                        width_byte |= 0x80  # Set bit 7 = hasOutline
                elif needs_casing:
                    width_byte |= 0x80  # Set bit 7 = hasCasing

                bx1, by1 = max(0, min(255, f_min_x >> 4)), max(0, min(255, f_min_y >> 4))
                bx2, by2 = max(0, min(255, f_max_x >> 4)), max(0, min(255, f_max_y >> 4))

                # Feature Header
                f.write(struct.pack('<B', feature['geom_type']))
                f.write(struct.pack('<H', feature['color_rgb565']))
                f.write(struct.pack('<B', feature['zoom_priority']))
                f.write(struct.pack('<B', width_byte))  # Width + casing flag
                f.write(struct.pack('<BBBB', bx1, by1, bx2, by2))
                f.write(struct.pack('<H', total_points))
                f.write(b'\x00')

                # Points for all rings (clamp to int16 range for long runways)
                for ring in projected_rings:
                    for px, py in ring:
                        # Clamp coordinates to fit in signed 16-bit integer range
                        px_clamped = max(-32768, min(32767, px))
                        py_clamped = max(-32768, min(32767, py))
                        f.write(struct.pack('<hh', px_clamped, py_clamped))

                if is_polygon:
                    # Write ring ends (using uint16 to support > 255 rings in complex merged areas)
                    f.write(struct.pack('<H', len(projected_rings)))
                    current_end = 0
                    for ring in projected_rings:
                        current_end += len(ring)
                        f.write(struct.pack('<H', current_end))

                written_features += 1

        f.seek(4)
        f.write(struct.pack('<H', written_features))

    logger.debug(f"  Tile {tile_x},{tile_y}: Write: {written_features} features, "
                 f"{filtered_by_size} filtered by area (<{K_VISIBILITY}px2), "
                 f"{filtered_holes_write}/{total_holes_write} holes removed (<{K_VISIBILITY * K_HOLE_FACTOR}px2)")

    return True
