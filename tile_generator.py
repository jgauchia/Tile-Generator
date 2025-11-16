#!/usr/bin/env python3
"""
Optimized tile generator - NO SHAPELY VERSION
Processes GeoJSON coordinates directly without heavy geometry libraries.
Optimized for 8GB RAM systems with conservative resource usage.
"""

import math
import struct
import json
import os
import gc
import subprocess
import ijson
import time
from decimal import Decimal
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
}

GLOBAL_COLOR_PALETTE = {}
GLOBAL_INDEX_TO_RGB332 = {}

def get_available_memory_mb() -> int:
    """Get available memory in MB, with fallback for different platforms"""
    try:
        if os.name == 'posix':
            # Linux/Mac
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        return int(line.split()[1]) // 1024
        elif os.name == 'nt':
            # Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
            memory = ctypes.c_ulonglong()
            if kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(memory)):
                return memory.value // (1024 * 1024)
    except:
        pass
    return 4096  # Fallback: 4GB assumption

def get_optimal_workers(cpu_count: int, available_memory_mb: int) -> int:
    """
    Calculate optimal worker count considering both CPU and memory constraints.
    Conservative for 8GB RAM systems, scales for better systems.
    """
    # Memory-based constraint: ~1GB per worker baseline
    memory_workers = max(1, min(available_memory_mb // 1024, 6))
    
    # CPU-based constraint: leave one core free
    cpu_workers = max(1, cpu_count - 1)
    
    # Conservative cap for 8GB systems, higher for more RAM
    if available_memory_mb <= 8192:  # 8GB or less
        max_workers = min(3, memory_workers, cpu_workers)
    else:
        max_workers = min(6, memory_workers, cpu_workers)
    
    logger.info(f"System: {cpu_count} CPU cores, {available_memory_mb}MB RAM -> Using {max_workers} workers")
    return max_workers

def get_adaptive_batch_size(zoom: int, base_batch_size: int, available_memory_mb: int) -> int:
    """
    Calculate adaptive batch size based on zoom level and available memory.
    Conservative for limited RAM systems, scales for better hardware.
    """
    # Memory factor: normalize to 4GB baseline, cap at 2.0 for systems with more RAM
    memory_factor = min(2.0, available_memory_mb / 4096)
    
    # Base adjustments for zoom levels with memory consideration
    if zoom < 10:
        adjusted_size = max(2000, int(base_batch_size * 0.6 * memory_factor))
    elif zoom < 13:
        adjusted_size = max(1000, int(base_batch_size * 0.4 * memory_factor))
    elif zoom < 15:
        adjusted_size = max(500, int(base_batch_size * 0.3 * memory_factor))
    else:
        adjusted_size = max(250, int(base_batch_size * 0.2 * memory_factor))
    
    logger.debug(f"Zoom {zoom}: base={base_batch_size}, memory_factor={memory_factor:.2f} -> batch={adjusted_size}")
    return adjusted_size

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

def douglas_peucker_simplify(points: List, tolerance: float) -> List:
    if len(points) < 3:
        return points
    
    def perpendicular_distance(point, line_start, line_end):
        x0, y0 = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dx**2 + dy**2)
    
    def simplify_recursive(points_list, start_idx, end_idx, tolerance):
        max_dist = 0
        max_idx = start_idx
        
        for i in range(start_idx + 1, end_idx):
            dist = perpendicular_distance(points_list[i], points_list[start_idx], points_list[end_idx])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        if max_dist > tolerance:
            left = simplify_recursive(points_list, start_idx, max_idx, tolerance)
            right = simplify_recursive(points_list, max_idx, end_idx, tolerance)
            return left[:-1] + right
        else:
            return [points_list[start_idx], points_list[end_idx]]
    
    if len(points) < 3:
        return points
    
    simplified = simplify_recursive(points, 0, len(points) - 1, tolerance)
    return simplified

def get_simplification_tolerance(zoom: int) -> float:
    if zoom < 13:
        return 0.0001
    elif zoom == 13:
        return 0.00005
    elif zoom == 14:
        return 0.00003
    elif zoom == 15:
        return 0.00002
    elif zoom == 16:
        return 0.00001
    else:
        return 0.000007

def simplify_coordinates(coordinates, geom_type: str, zoom: int):
    if zoom < 16:
        return coordinates
    
    tolerance = get_simplification_tolerance(zoom)
    
    def simplify_ring(ring):
        if not ring or len(ring) < 3:
            return ring
        simplified = douglas_peucker_simplify(ring, tolerance)
        if simplified and simplified[0] != simplified[-1]:
            simplified.append(simplified[0])
        return simplified
    
    if geom_type == "LineString":
        if len(coordinates) < 3:
            return coordinates
        return douglas_peucker_simplify(coordinates, tolerance)
    
    elif geom_type == "Polygon":
        if not coordinates or not coordinates[0]:
            return coordinates
        simplified_exterior = simplify_ring(coordinates[0])
        return [simplified_exterior]
    
    elif geom_type == "MultiLineString":
        return [douglas_peucker_simplify(line, tolerance) if len(line) >= 3 else line for line in coordinates]
    
    elif geom_type == "MultiPolygon":
        result = []
        for poly in coordinates:
            if poly and poly[0]:
                simplified_exterior = simplify_ring(poly[0])
                result.append([simplified_exterior])
        return result
    
    return coordinates

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

def get_layer_priority(tags: Dict) -> int:
    if 'layer' in tags:
        try:
            layer_val = int(tags['layer'])
            return layer_val * 1000
        except (ValueError, TypeError):
            pass

    if 'natural' in tags and tags['natural'] in ['water', 'coastline', 'bay']:
        return 100
    if 'waterway' in tags and tags.get('waterway') in ['riverbank', 'dock', 'boatyard']:
        return 100

    if 'landuse' in tags:
        return 200
    if 'natural' in tags and tags['natural'] in ['wood', 'forest', 'scrub', 'heath', 'grassland', 'beach', 'sand', 'wetland']:
        return 200
    if 'leisure' in tags and tags['leisure'] in ['park', 'nature_reserve', 'garden']:
        return 200

    if 'waterway' in tags and tags['waterway'] in ['river', 'stream', 'canal']:
        return 300

    if 'natural' in tags and tags['natural'] in ['peak', 'ridge', 'volcano', 'cliff']:
        return 400

    if 'tunnel' in tags and tags['tunnel'] == 'yes':
        return 500

    if 'railway' in tags:
        return 600

    if 'highway' in tags and tags['highway'] in ['path', 'footway', 'cycleway', 'steps', 'pedestrian', 'track']:
        return 700

    if 'highway' in tags and tags['highway'] in ['tertiary', 'tertiary_link']:
        return 800

    if 'highway' in tags and tags['highway'] in ['secondary', 'secondary_link']:
        return 900

    if 'highway' in tags and tags['highway'] in ['primary', 'primary_link']:
        return 1000

    if 'highway' in tags and tags['highway'] in ['trunk', 'trunk_link']:
        return 1100

    if 'highway' in tags and tags['highway'] in ['motorway', 'motorway_link']:
        return 1200

    if 'bridge' in tags and tags['bridge'] == 'yes':
        return 1300
    if 'aeroway' in tags:
        return 1300

    if 'building' in tags:
        return 1400

    if 'amenity' in tags:
        return 1500

    return 200

def geometry_to_commands(geom_type: str, coordinates, color: int, zoom: int, tile_x: int, tile_y: int, color_index: Optional[int] = None) -> List[Dict]:
    commands = []
    
    if color_index is not None:
        commands.append({'type': DRAW_COMMANDS['SET_COLOR_INDEX'], 'color_index': color_index})
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
    query = ", ".join(parts) if parts else "*"
    return query

def process_feature(feature: Dict, config: Dict, zoom: int, tiles_data: Dict):
    geom = feature.get('geometry')
    props = feature.get('properties', {})
    if not geom or geom['type'] not in ['LineString', 'Polygon', 'MultiLineString', 'MultiPolygon']:
        return
    geom_type = geom['type']
    coordinates = geom['coordinates']
    
    style = get_style_for_tags(props, config)
    if not style:
        return
    
    zoom_filter = style.get('zoom', 6)
    if zoom < zoom_filter:
        return
    
    coordinates = normalize_coordinates(coordinates)
    
    if zoom >= 16:
        coordinates = simplify_coordinates(coordinates, geom_type, zoom)
    
    hex_color = style.get('color', '#FFFFFF')
    color = hex_to_rgb332(hex_color)
    color_index = hex_to_color_index(hex_color)
    priority = style.get('priority', 50)
    layer_priority = get_layer_priority(props)
    
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
            
            commands = geometry_to_commands(geom_type, clipped, color, zoom, tx, ty, color_index)
            if commands:
                tile_key = (tx, ty)
                if tile_key not in tiles_data:
                    tiles_data[tile_key] = []
                combined_priority = layer_priority + priority
                tiles_data[tile_key].append({'commands': commands, 'priority': combined_priority})

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
    sorted_entries = sorted(tile_entries, key=lambda x: x['priority'])
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
        data = data[:max_file_size]
    with open(filepath, 'wb') as f:
        f.write(data)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

def merge_tiles_data(target: Dict, source: Dict):
    for tile_key, entries in source.items():
        if tile_key in target:
            target[tile_key].extend(entries)
        else:
            target[tile_key] = entries

def write_tiles_batch(tiles_data: Dict, zoom: int, output_dir: str, max_file_size: int, max_workers: int, persistent_tiles: Dict):
    merge_tiles_data(persistent_tiles, tiles_data)
    return len(tiles_data)

def write_final_tiles(persistent_tiles: Dict, zoom: int, output_dir: str, max_file_size: int, max_workers: int):
    if not persistent_tiles:
        return 0
    
    tile_jobs = [(tx, ty, entries, zoom, output_dir, max_file_size) for (tx, ty), entries in persistent_tiles.items()]
    tiles_written = len(tile_jobs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(write_single_tile, job) for job in tile_jobs]
        for future in as_completed(futures):
            future.result()
    
    return tiles_written

def generate_tiles(gol_file: str, output_dir: str, config_file: str, zoom_levels: List[int], max_file_size: int = 65536, base_batch_size: int = 10000):
    with open(config_file) as f:
        config = json.load(f)
    
    # Detect system resources
    available_memory_mb = get_available_memory_mb()
    cpu_count = os.cpu_count() or 4
    max_workers = get_optimal_workers(cpu_count, available_memory_mb)
    
    logger.info(f"System detected: {cpu_count} CPU cores, {available_memory_mb}MB RAM")
    logger.info(f"Resource settings: {max_workers} workers, base batch size: {base_batch_size}")
    
    precompute_global_color_palette(config)
    write_palette_bin(output_dir)
    query = compress_goql_queries(config)
    
    gol_cmd = "/gol" if os.path.exists("/gol") else "gol"
    
    for zoom in zoom_levels:
        zoom_start = time.time()
        
        # Calculate adaptive batch size for this zoom level
        adaptive_batch = get_adaptive_batch_size(zoom, base_batch_size, available_memory_mb)
        logger.info(f"Processing zoom {zoom} (batch size: {adaptive_batch}, workers: {max_workers})...")
        
        process = subprocess.Popen([gol_cmd, "query", gol_file, query, "-f", "geojson"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=65536)
        tiles_data = {}
        persistent_tiles = {}
        feature_count = 0
        batch_count = 0
        
        try:
            for feature in ijson.items(process.stdout, "features.item"):
                feature_count += 1
                process_feature(feature, config, zoom, tiles_data)
                
                if feature_count % adaptive_batch == 0:
                    batch_count += 1
                    write_tiles_batch(tiles_data, zoom, output_dir, max_file_size, max_workers, persistent_tiles)
                    print(f"\rZoom {zoom}: {feature_count} features, {len(persistent_tiles)} unique tiles (batch {batch_count})...", end='', flush=True)
                    tiles_data.clear()
                    gc.collect()
                elif feature_count % 5000 == 0:
                    print(f"\rZoom {zoom}: {feature_count} features, {len(tiles_data)} tiles in buffer, {len(persistent_tiles)} persistent...", end='', flush=True)
            
            if tiles_data:
                write_tiles_batch(tiles_data, zoom, output_dir, max_file_size, max_workers, persistent_tiles)
            
            print(f"\rZoom {zoom}: {feature_count} features processed, writing {len(persistent_tiles)} tiles...{' ' * 20}")
            sys.stdout.flush()
            
            total_tiles_written = write_final_tiles(persistent_tiles, zoom, output_dir, max_file_size, max_workers)
            
            print(f"\rZoom {zoom}: {feature_count} features processed, {total_tiles_written} tiles written{' ' * 20}")
            sys.stdout.flush()
        except ijson.common.IncompleteJSONError:
            pass
        finally:
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                stderr_content = process.stderr.read() if process.stderr else ""
                if process.stderr:
                    process.stderr.close()
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("gol query failed: " + stderr_content)
        
        zoom_elapsed = time.time() - zoom_start
        logger.info(f"Zoom {zoom} completed in {format_time(zoom_elapsed)}")
        
        del tiles_data
        del persistent_tiles
        del process
        gc.collect()

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Generate map tiles from GoL file using features configuration')
    parser.add_argument("gol_file", help="Input GoL file path")
    parser.add_argument("output_dir", help="Output directory for tiles")
    parser.add_argument("config_file", help="JSON config file with feature definitions")
    parser.add_argument("--zoom", default="6-17", help="Zoom level(s) to generate")
    parser.add_argument("--max-file-size", type=int, default=128, help="Maximum tile file size in KB")
    parser.add_argument("--batch-size", type=int, default=10000, help="Base batch size (auto-adjusted per zoom level and system RAM)")
    args = parser.parse_args()
    
    if '-' in args.zoom:
        start, end = map(int, args.zoom.split('-'))
        zoom_levels = list(range(start, end + 1))
    else:
        zoom_levels = [int(args.zoom)]
    
    max_file_size_bytes = args.max_file_size * 1024
    
    logger.info(f"Input: {args.gol_file}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Zoom levels: {zoom_levels}")
    logger.info(f"Max tile size: {args.max_file_size}KB")
    logger.info(f"Base batch size: {args.batch_size} features (adaptive per zoom and system RAM)")
    
    generate_tiles(args.gol_file, args.output_dir, args.config_file, zoom_levels, max_file_size_bytes, args.batch_size)
    
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
    
    if total_size < 1024:
        size_str = f"{total_size}B"
    elif total_size < 1024 * 1024:
        size_str = f"{round(total_size / 1024, 1)}KB"
    elif total_size < 1024 * 1024 * 1024:
        size_str = f"{round(total_size / (1024 * 1024), 1)}MB"
    else:
        size_str = f"{round(total_size / (1024 * 1024 * 1024), 2)}GB"
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Total tiles written: {total_count}")
    logger.info(f"Total size: {size_str}")
    logger.info(f"Average tile size: {round(total_size / total_count) if total_count > 0 else 0} bytes")
    logger.info(f"Palette file: {os.path.join(args.output_dir, 'palette.bin')}")
    logger.info("=" * 50)
    logger.info(f"Total processing time: {format_time(elapsed_time)}")

if __name__ == "__main__":
    main()