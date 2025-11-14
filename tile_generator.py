#!/usr/bin/env python3
"""
Optimized tile generator - NO SHAPELY VERSION
Processes GeoJSON coordinates directly without heavy geometry libraries.
"""

import math
import struct
import json
import os
import gc
import subprocess
import ijson
from decimal import Decimal
from tqdm import tqdm
import argparse
import sys
import logging
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

TILE_SIZE = 256
UINT16_TILE_SIZE = 65536

DRAW_COMMANDS = {
    'LINE': 1,
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,
    'SET_COLOR': 0x80,
    'SET_COLOR_INDEX': 0x81,
    'SET_LAYER': 0x88,
}

GLOBAL_COLOR_PALETTE = {}
GLOBAL_INDEX_TO_RGB332 = {}

def precompute_global_color_palette(config: Dict[str, Any]) -> int:
    global GLOBAL_COLOR_PALETTE, GLOBAL_INDEX_TO_RGB332
    logger.info("Building color palette...")
    unique_colors = set()
    for feature_config in config.values():
        if isinstance(feature_config, dict) and 'color' in feature_config:
            hex_color = feature_config['color']
            if hex_color and isinstance(hex_color, str) and hex_color.startswith("#"):
                unique_colors.add(hex_color)
    sorted_colors = sorted(list(unique_colors))
    GLOBAL_COLOR_PALETTE.clear()
    GLOBAL_INDEX_TO_RGB332.clear()
    for index, hex_color in enumerate(sorted_colors):
        rgb332_value = hex_to_rgb332(hex_color)
        GLOBAL_COLOR_PALETTE[hex_color] = index
        GLOBAL_INDEX_TO_RGB332[index] = rgb332_value
    logger.info("Palette: " + str(len(unique_colors)) + " colors")
    return len(unique_colors)

def write_palette_bin(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    palette_path = os.path.join(output_dir, "palette.bin")
    num_colors = len(GLOBAL_INDEX_TO_RGB332)
    palette_data = bytearray()
    palette_data.extend(struct.pack('<I', num_colors))
    for index in range(num_colors):
        if index in GLOBAL_INDEX_TO_RGB332:
            rgb332 = GLOBAL_INDEX_TO_RGB332[index]
            r = (rgb332 & 0xE0) | ((rgb332 & 0xE0) >> 3) | ((rgb332 & 0xE0) >> 6)
            g = ((rgb332 & 0x1C) << 3) | ((rgb332 & 0x1C) >> 2) | ((rgb332 & 0x1C) << 1)
            b = ((rgb332 & 0x03) << 6) | ((rgb332 & 0x03) << 4) | ((rgb332 & 0x03) << 2) | (rgb332 & 0x03)
            palette_data.extend([r, g, b])
        else:
            palette_data.extend([255, 255, 255])
    with open(palette_path, 'wb') as f:
        f.write(bytes(palette_data))
    logger.info("Palette written")

def hex_to_color_index(hex_color: str) -> Optional[int]:
    return GLOBAL_COLOR_PALETTE.get(hex_color)

def hex_to_rgb332(hex_color: str) -> int:
    try:
        if not hex_color or not hex_color.startswith("#"):
            return 0xFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r & 0xE0) | ((g & 0xE0) >> 3) | (b >> 6))
    except:
        return 0xFF

def geojson_bounds(coordinates, geom_type: str) -> Tuple[float, float, float, float]:
    def flatten_coords(coords):
        if not coords:
            return []
        if isinstance(coords, (int, float, Decimal)):
            return [float(coords)]
        if len(coords) == 2:
            try:
                lon = float(coords[0]) if isinstance(coords[0], Decimal) else coords[0]
                lat = float(coords[1]) if isinstance(coords[1], Decimal) else coords[1]
                if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                    return [[lon, lat]]
            except (TypeError, ValueError, IndexError):
                pass
        result = []
        for item in coords:
            result.extend(flatten_coords(item))
        return result
    flat = flatten_coords(coordinates)
    if not flat:
        return (0, 0, 0, 0)
    lons = [c[0] for c in flat]
    lats = [c[1] for c in flat]
    return (min(lons), min(lats), max(lons), max(lats))

def coords_intersect_tile(coordinates, geom_type: str, tile_bbox: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = geojson_bounds(coordinates, geom_type)
    tile_minx, tile_miny, tile_maxx, tile_maxy = tile_bbox
    if maxx < tile_minx or minx > tile_maxx or maxy < tile_miny or miny > tile_maxy:
        return False
    return True

def normalize_coordinates(coordinates):
    if isinstance(coordinates, (int, float)):
        return float(coordinates)
    if isinstance(coordinates, Decimal):
        return float(coordinates)
    if isinstance(coordinates, (list, tuple)):
        if len(coordinates) == 2:
            try:
                lon = float(coordinates[0]) if isinstance(coordinates[0], Decimal) else coordinates[0]
                lat = float(coordinates[1]) if isinstance(coordinates[1], Decimal) else coordinates[1]
                if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                    return [lon, lat]
            except (TypeError, ValueError, IndexError):
                pass
        return [normalize_coordinates(item) for item in coordinates]
    return coordinates

def clip_polygon_sutherland_hodgman(vertices, minx, miny, maxx, maxy):
    if not vertices or len(vertices) < 3:
        return []
    def inside(p, edge):
        if edge == 'left': return p[0] >= minx
        elif edge == 'right': return p[0] <= maxx
        elif edge == 'bottom': return p[1] >= miny
        elif edge == 'top': return p[1] <= maxy
        return False
    def compute_intersection(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        if edge == 'left':
            x = minx
            y = y1 + (y2 - y1) * (minx - x1) / (x2 - x1) if x2 != x1 else y1
        elif edge == 'right':
            x = maxx
            y = y1 + (y2 - y1) * (maxx - x1) / (x2 - x1) if x2 != x1 else y1
        elif edge == 'bottom':
            y = miny
            x = x1 + (x2 - x1) * (miny - y1) / (y2 - y1) if y2 != y1 else x1
        elif edge == 'top':
            y = maxy
            x = x1 + (x2 - x1) * (maxy - y1) / (y2 - y1) if y2 != y1 else x1
        return [x, y]
    output = list(vertices)
    for edge in ['left', 'right', 'bottom', 'top']:
        if not output:
            break
        input_list = output
        output = []
        if not input_list:
            continue
        prev_vertex = input_list[-1]
        for curr_vertex in input_list:
            curr_inside = inside(curr_vertex, edge)
            prev_inside = inside(prev_vertex, edge)
            if curr_inside:
                if not prev_inside:
                    intersection = compute_intersection(prev_vertex, curr_vertex, edge)
                    output.append(intersection)
                output.append(curr_vertex)
            elif prev_inside:
                intersection = compute_intersection(prev_vertex, curr_vertex, edge)
                output.append(intersection)
            prev_vertex = curr_vertex
    if output and len(output) >= 3:
        if output[0] != output[-1]:
            output.append(output[0])
    return output if len(output) >= 4 else []

def clip_line_to_bbox(p1, p2, bbox):
    minx, miny, maxx, maxy = bbox
    x1, y1 = p1
    x2, y2 = p2
    def compute_outcode(x, y):
        code = 0
        if x < minx: code |= 1
        if x > maxx: code |= 2
        if y < miny: code |= 4
        if y > maxy: code |= 8
        return code
    outcode1 = compute_outcode(x1, y1)
    outcode2 = compute_outcode(x2, y2)
    while True:
        if outcode1 == 0 and outcode2 == 0:
            return ([x1, y1], [x2, y2])
        if (outcode1 & outcode2) != 0:
            return None
        outcode = outcode1 if outcode1 != 0 else outcode2
        if outcode & 8:
            x = x1 + (x2 - x1) * (maxy - y1) / (y2 - y1)
            y = maxy
        elif outcode & 4:
            x = x1 + (x2 - x1) * (miny - y1) / (y2 - y1)
            y = miny
        elif outcode & 2:
            y = y1 + (y2 - y1) * (maxx - x1) / (x2 - x1)
            x = maxx
        elif outcode & 1:
            y = y1 + (y2 - y1) * (minx - x1) / (x2 - x1)
            x = minx
        if outcode == outcode1:
            x1, y1 = x, y
            outcode1 = compute_outcode(x1, y1)
        else:
            x2, y2 = x, y
            outcode2 = compute_outcode(x2, y2)

def clip_coordinates_to_tile(coordinates, geom_type: str, tile_bbox: Tuple[float, float, float, float]):
    minx, miny, maxx, maxy = tile_bbox
    def clip_linestring(coords):
        if not coords or len(coords) < 2:
            return []
        clipped = []
        for i in range(len(coords) - 1):
            result = clip_line_to_bbox(coords[i], coords[i + 1], tile_bbox)
            if result:
                p1, p2 = result
                if not clipped or clipped[-1] != p1:
                    clipped.append(p1)
                clipped.append(p2)
        return clipped if len(clipped) >= 2 else []
    if geom_type == "Point":
        lon, lat = float(coordinates[0]), float(coordinates[1])
        if minx <= lon <= maxx and miny <= lat <= maxy:
            return coordinates
        return None
    elif geom_type == "LineString":
        clipped = clip_linestring(coordinates)
        return clipped if clipped else None
    elif geom_type == "Polygon":
        if not coordinates or not coordinates[0]:
            return None
        exterior = coordinates[0]
        clipped_exterior = clip_polygon_sutherland_hodgman(exterior, minx, miny, maxx, maxy)
        if not clipped_exterior or len(clipped_exterior) < 4:
            return None
        return [clipped_exterior]
    elif geom_type == "MultiLineString":
        clipped_lines = []
        for line in coordinates:
            clipped = clip_linestring(line)
            if clipped:
                clipped_lines.append(clipped)
        return clipped_lines if clipped_lines else None
    elif geom_type == "MultiPolygon":
        clipped_polys = []
        for poly in coordinates:
            if poly and poly[0]:
                clipped_exterior = clip_polygon_sutherland_hodgman(poly[0], minx, miny, maxx, maxy)
                if clipped_exterior and len(clipped_exterior) >= 4:
                    clipped_polys.append([clipped_exterior])
        return clipped_polys if clipped_polys else None
    return None

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_bbox(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
    return (lon_min, lat_min, lon_max, lat_max)

def coords_to_pixels(coordinates, zoom: int, tile_x: int, tile_y: int) -> List[Tuple[int, int]]:
    n = 2.0 ** zoom
    pixels = []
    for coord in coordinates:
        lon = float(coord[0]) if isinstance(coord[0], Decimal) else coord[0]
        lat = float(coord[1]) if isinstance(coord[1], Decimal) else coord[1]
        x = ((lon + 180.0) / 360.0 * n - tile_x) * TILE_SIZE
        lat_rad = math.radians(lat)
        y = ((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n - tile_y) * TILE_SIZE
        x = max(0, min(TILE_SIZE - 1, int(x)))
        y = max(0, min(TILE_SIZE - 1, int(y)))
        x_uint16 = int((x * (UINT16_TILE_SIZE - 1)) / (TILE_SIZE - 1))
        y_uint16 = int((y * (UINT16_TILE_SIZE - 1)) / (TILE_SIZE - 1))
        x_uint16 = max(0, min(UINT16_TILE_SIZE - 1, x_uint16))
        y_uint16 = max(0, min(UINT16_TILE_SIZE - 1, y_uint16))
        pixels.append((x_uint16, y_uint16))
    return pixels

def pack_varint(n):
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return out

def pack_zigzag(n):
    return pack_varint((n << 1) ^ (n >> 31))

def get_layer_from_tags(tags: Dict) -> int:
    if 'natural' in tags and tags['natural'] in ['water', 'coastline', 'bay']:
        return 0
    if 'waterway' in tags and tags.get('waterway') in ['riverbank', 'dock', 'boatyard']:
        return 0
    if 'landuse' in tags and tags['landuse'] in ['forest', 'farmland', 'meadow', 'grass', 'orchard', 'vineyard', 'residential', 'commercial', 'retail', 'industrial']:
        return 1
    if 'natural' in tags and tags['natural'] in ['wood', 'forest', 'scrub', 'heath', 'grassland', 'beach', 'sand', 'wetland']:
        return 1
    if 'leisure' in tags and tags['leisure'] in ['park', 'pitch', 'golf_course', 'stadium', 'sports_centre', 'playground']:
        return 1
    if 'waterway' in tags and tags['waterway'] not in ['riverbank', 'dock', 'boatyard']:
        return 2
    if 'building' in tags:
        return 3
    if 'highway' in tags:
        return 4
    if 'railway' in tags:
        return 4
    if 'amenity' in tags:
        return 5
    return 1

def geometry_to_commands(geom_type: str, coordinates, color: int, zoom: int, tile_x: int, tile_y: int, tags: Dict, hex_color: str = None) -> List[Dict]:
    commands = []
    layer = get_layer_from_tags(tags)
    commands.append({'type': DRAW_COMMANDS['SET_LAYER'], 'layer': layer})
    if hex_color:
        color_index = hex_to_color_index(hex_color)
        if color_index is not None:
            commands.append({'type': DRAW_COMMANDS['SET_COLOR_INDEX'], 'color_index': color_index})
        else:
            commands.append({'type': DRAW_COMMANDS['SET_COLOR'], 'color': color})
    else:
        commands.append({'type': DRAW_COMMANDS['SET_COLOR'], 'color': color})
    def process_linestring(coords):
        if len(coords) < 2:
            return []
        pixels = coords_to_pixels(coords, zoom, tile_x, tile_y)
        unique_pixels = []
        for i, pixel in enumerate(pixels):
            if i == 0 or pixel != pixels[i-1]:
                unique_pixels.append(pixel)
        if len(unique_pixels) < 2:
            return []
        if len(unique_pixels) == 2:
            x1, y1 = unique_pixels[0]
            x2, y2 = unique_pixels[1]
            return [{'type': DRAW_COMMANDS['LINE'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}]
        else:
            return [{'type': DRAW_COMMANDS['POLYLINE'], 'points': unique_pixels}]
    def process_polygon(coords):
        if not coords or not coords[0] or len(coords[0]) < 3:
            return []
        exterior = coords[0]
        pixels = coords_to_pixels(exterior, zoom, tile_x, tile_y)
        unique_pixels = []
        for i, pixel in enumerate(pixels):
            if i == 0 or pixel != pixels[i-1]:
                unique_pixels.append(pixel)
        if unique_pixels and unique_pixels[0] != unique_pixels[-1]:
            unique_pixels.append(unique_pixels[0])
        if len(set(unique_pixels)) < 3:
            return []
        return [{'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': unique_pixels}]
    if geom_type == "LineString":
        commands.extend(process_linestring(coordinates))
    elif geom_type == "Polygon":
        commands.extend(process_polygon(coordinates))
    elif geom_type == "MultiLineString":
        for line in coordinates:
            line_commands = process_linestring(line)
            if line_commands:
                commands.extend(line_commands)
    elif geom_type == "MultiPolygon":
        for poly in coordinates:
            poly_commands = process_polygon(poly)
            if poly_commands:
                commands.extend(poly_commands)
    return commands

def pack_draw_commands(commands: List[Dict]) -> bytes:
    out = bytearray()
    out += pack_varint(len(commands))
    for cmd in commands:
        cmd_type = cmd['type']
        out += pack_varint(cmd_type)
        if cmd_type == DRAW_COMMANDS['SET_COLOR']:
            out += struct.pack("B", cmd.get('color', 0xFF))
        elif cmd_type == DRAW_COMMANDS['SET_COLOR_INDEX']:
            out += pack_varint(cmd.get('color_index', 0))
        elif cmd_type == DRAW_COMMANDS['SET_LAYER']:
            out += pack_varint(cmd.get('layer', 0))
        elif cmd_type == DRAW_COMMANDS['LINE']:
            x1, y1, x2, y2 = cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif cmd_type in [DRAW_COMMANDS['POLYLINE'], DRAW_COMMANDS['STROKE_POLYGON']]:
            points = cmd['points']
            out += pack_varint(len(points))
            prev_x, prev_y = 0, 0
            for i, (x, y) in enumerate(points):
                if i == 0:
                    out += pack_zigzag(x)
                    out += pack_zigzag(y)
                else:
                    out += pack_zigzag(x - prev_x)
                    out += pack_zigzag(y - prev_y)
                prev_x, prev_y = x, y
    return bytes(out)

def compress_goql_queries(config: Dict) -> str:
    META = {'tile_size', 'viewport_size', 'toolbar_width', 'statusbar_height', 'max_cache_size', 'thread_pool_size', 'background_colors', 'log_level', 'config_file', 'fps_limit', 'fill_polygons'}
    LINEAR_TAGS = {'highway', 'railway'}
    LINEAR_VALUES = {'natural': {'coastline', 'tree_row'}, 'waterway': {'river', 'stream', 'canal'}}
    AREA_VALUES = {'waterway': {'riverbank', 'dock', 'boatyard'}}
    way_q = []
    area_q = []
    for key in config:
        if key in META or not isinstance(config[key], dict):
            continue
        if '=' in key:
            tag, val = key.split('=', 1)
            is_lin = False
            if tag in LINEAR_VALUES and val in LINEAR_VALUES[tag]:
                is_lin = True
            elif tag in LINEAR_TAGS:
                is_lin = True
            is_area = False
            if tag in AREA_VALUES and val in AREA_VALUES[tag]:
                is_area = True
            if is_lin and not is_area:
                way_q.append(tag + "=" + val)
            else:
                area_q.append(tag + "=" + val)
        else:
            area_q.append(key)
    def group(queries):
        grp = {}
        single = []
        for q in queries:
            if '=' in q:
                t, v = q.split('=', 1)
                if t not in grp:
                    grp[t] = []
                grp[t].append(v)
            else:
                single.append(q)
        return grp, single
    parts = []
    way_grp, _ = group(way_q)
    for tag, vals in sorted(way_grp.items()):
        if len(vals) == 1:
            parts.append("w[" + tag + "=" + vals[0] + "]")
        else:
            pat = '|'.join(vals)
            parts.append("w[" + tag + "~'^(" + pat + ")$']")
    area_grp, area_single = group(area_q)
    for tag, vals in sorted(area_grp.items()):
        if len(vals) == 1:
            parts.append("nwa[" + tag + "=" + vals[0] + "]")
        else:
            pat = '|'.join(vals)
            parts.append("nwa[" + tag + "~'^(" + pat + ")$']")
    for tag in sorted(area_single):
        parts.append("nwa[" + tag + "]")
    return ", ".join(parts) if parts else "*"

def process_feature(feature: Dict, config: Dict, zoom: int, tiles_data: Dict):
    geom = feature.get('geometry')
    props = feature.get('properties', {})
    if not geom or geom['type'] not in ['LineString', 'Polygon', 'MultiLineString', 'MultiPolygon']:
        return
    geom_type = geom['type']
    coordinates = geom['coordinates']
    coordinates = normalize_coordinates(coordinates)
    style = get_style_for_tags(props, config)
    if not style:
        return
    hex_color = style.get('color', '#FFFFFF')
    color = hex_to_rgb332(hex_color)
    zoom_filter = style.get('zoom', 6)
    priority = style.get('priority', 50)
    if zoom < zoom_filter:
        return
    minx, miny, maxx, maxy = geojson_bounds(coordinates, geom_type)
    tile_x_min, tile_y_min = deg2num(miny, minx, zoom)
    tile_x_max, tile_y_max = deg2num(maxy, maxx, zoom)
    for tx in range(min(tile_x_min, tile_x_max), max(tile_x_min, tile_x_max) + 1):
        for ty in range(min(tile_y_min, tile_y_max), max(tile_y_min, tile_y_max) + 1):
            tile_bounds = tile_bbox(tx, ty, zoom)
            if not coords_intersect_tile(coordinates, geom_type, tile_bounds):
                continue
            clipped = clip_coordinates_to_tile(coordinates, geom_type, tile_bounds)
            if not clipped:
                continue
            commands = geometry_to_commands(geom_type, clipped, color, zoom, tx, ty, props, hex_color)
            if commands:
                tile_key = (tx, ty)
                if tile_key not in tiles_data:
                    tiles_data[tile_key] = []
                tiles_data[tile_key].append({'commands': commands, 'priority': priority, 'layer': get_layer_from_tags(props)})

def get_style_for_tags(tags: Dict, config: Dict) -> Optional[Dict]:
    for k, v in tags.items():
        keyval = k + "=" + str(v)
        if keyval in config and isinstance(config[keyval], dict):
            return config[keyval]
    for k in tags:
        if k in config and isinstance(config[k], dict):
            return config[k]
    return None

def sort_and_flatten_commands(tile_entries: List[Dict]) -> List[Dict]:
    sorted_entries = sorted(tile_entries, key=lambda x: (x['layer'], -x['priority']))
    all_commands = []
    for entry in sorted_entries:
        all_commands.extend(entry['commands'])
    return all_commands

def write_single_tile(job):
    tx, ty, tile_entries, zoom, output_dir, max_file_size = job
    commands = sort_and_flatten_commands(tile_entries)
    tile_dir = os.path.join(output_dir, str(zoom), str(tx))
    os.makedirs(tile_dir, exist_ok=True)
    filepath = os.path.join(tile_dir, str(ty) + ".bin")
    data = pack_draw_commands(commands)
    if len(data) > max_file_size:
        truncated_data = bytearray()
        for i in range(len(commands)):
            test_data = pack_draw_commands(commands[:i+1])
            if len(test_data) > max_file_size:
                break
            truncated_data = test_data
        data = truncated_data if truncated_data else pack_draw_commands([])
    with open(filepath, 'wb') as f:
        f.write(data)

def generate_tiles(gol_file: str, output_dir: str, config_file: str, zoom_levels: List[int], max_file_size: int = 65536):
    with open(config_file) as f:
        config = json.load(f)
    precompute_global_color_palette(config)
    write_palette_bin(output_dir)
    query = compress_goql_queries(config)
    cpu_count = os.cpu_count() or 4
    max_workers = min(cpu_count, 8)
    for zoom in zoom_levels:
        logger.info("Processing zoom " + str(zoom) + "...")
        gol_cmd = "/gol" if os.path.exists("/gol") else "gol"
        process = subprocess.Popen([gol_cmd, "query", gol_file, query, "-f", "geojson"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=8192)
        tiles_data = {}
        feature_count = 0
        try:
            for feature in ijson.items(process.stdout, "features.item"):
                feature_count += 1
                process_feature(feature, config, zoom, tiles_data)
                if feature_count % 1000 == 0:
                    print("\rZoom " + str(zoom) + ": " + str(feature_count) + " features...", end='', flush=True)
            print("\rZoom " + str(zoom) + ": " + str(feature_count) + " features processed", end='')
            sys.stdout.flush()
        except ijson.common.IncompleteJSONError:
            pass
        process.wait()
        if process.returncode != 0:
            stderr = process.stderr.read()
            raise RuntimeError("gol query failed: " + stderr)
        if tiles_data:
            print(" - Writing " + str(len(tiles_data)) + " tiles")
            tile_jobs = [(tx, ty, entries, zoom, output_dir, max_file_size) for (tx, ty), entries in tiles_data.items()]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                desc_text = "Zoom " + str(zoom)
                with tqdm(total=len(tile_jobs), desc=desc_text, unit="tiles", leave=False) as pbar:
                    futures = [executor.submit(write_single_tile, job) for job in tile_jobs]
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)
        else:
            print()
        del tiles_data
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Generate map tiles from GoL file using features configuration', formatter_class=argparse.RawDescriptionHelpFormatter, epilog="""
Examples:
  ./tile_generator.py map.gol output_dir features.json --zoom 6-17
  ./tile_generator.py map.gol output_dir features.json --zoom 12 --max-file-size 256
    """)
    parser.add_argument("gol_file", help="Input GoL file path")
    parser.add_argument("output_dir", help="Output directory for tiles")
    parser.add_argument("config_file", help="JSON config file with feature definitions")
    parser.add_argument("--zoom", default="6-17", help="Zoom level(s) to generate")
    parser.add_argument("--max-file-size", type=int, default=128, help="Maximum tile file size in KB")
    args = parser.parse_args()
    if '-' in args.zoom:
        start, end = map(int, args.zoom.split('-'))
        zoom_levels = list(range(start, end + 1))
    else:
        zoom_levels = [int(args.zoom)]
    max_file_size_bytes = args.max_file_size * 1024
    logger.info("Input: " + args.gol_file)
    logger.info("Output: " + args.output_dir)
    logger.info("Config: " + args.config_file)
    logger.info("Zoom levels: " + str(zoom_levels))
    logger.info("Max tile size: " + str(args.max_file_size) + "KB")
    
    generate_tiles(args.gol_file, args.output_dir, args.config_file, zoom_levels, max_file_size_bytes)
    
    # Count actual tiles and calculate total size
    total_count = 0
    total_size = 0
    for zoom in zoom_levels:
        zoom_dir = os.path.join(args.output_dir, str(zoom))
        if os.path.exists(zoom_dir):
            for x_dir in os.listdir(zoom_dir):
                x_path = os.path.join(zoom_dir, x_dir)
                if os.path.isdir(x_path):
                    for tile_file in os.listdir(x_path):
                        if tile_file.endswith('.bin'):
                            total_count += 1
                            tile_path = os.path.join(x_path, tile_file)
                            total_size += os.path.getsize(tile_path)
    
    # Format size nicely
    if total_size < 1024:
        size_str = str(total_size) + "B"
    elif total_size < 1024 * 1024:
        size_str = str(round(total_size / 1024, 1)) + "KB"
    elif total_size < 1024 * 1024 * 1024:
        size_str = str(round(total_size / (1024 * 1024), 1)) + "MB"
    else:
        size_str = str(round(total_size / (1024 * 1024 * 1024), 2)) + "GB"
    
    logger.info("Total tiles written: " + str(total_count))
    logger.info("Total size: " + size_str)
    logger.info("Average tile size: " + str(round(total_size / total_count)) + " bytes" if total_count > 0 else "N/A")
    logger.info("Palette file: " + os.path.join(args.output_dir, 'palette.bin'))

if __name__ == "__main__":
    main()