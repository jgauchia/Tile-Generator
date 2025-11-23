#!/usr/bin/env python3

import math
import struct
import json
import os
import gc
import subprocess
import ijson
import time
import re
from decimal import Decimal
import argparse
import sys
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
TILE_SIZE = 256
UINT16_TILE_SIZE = 65536

DRAW_COMMANDS = {
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,
    'SET_COLOR': 0x80,
    'SET_COLOR_INDEX': 0x81,
}

# Global color palette
GLOBAL_COLOR_PALETTE = {}
GLOBAL_INDEX_TO_RGB332 = {}

# --- CARTO-LIKE STYLE DEFINITIONS (Pixels per Zoom) ---
# OPTIMIZED FOR SMALL SCREENS: Thinner lines at low zoom, reduced max width at high zoom.
CARTO_STYLES = {
    # Major roads: Visible but not overwhelming at low zoom
    'motorway':       {6: 1, 10: 2, 13: 3, 15: 6, 17: 10, 18: 16},
    'trunk':          {6: 1, 10: 2, 13: 3, 15: 5, 17: 9,  18: 14},
    'primary':        {8: 1, 11: 2, 13: 3, 15: 5, 17: 9,  18: 14},
    
    # Connecting roads: Keep thin until zoomed in
    'secondary':      {11: 1, 13: 2, 15: 4, 17: 8, 18: 12},
    'tertiary':       {12: 1, 14: 2, 16: 5, 18: 10},
    
    # Minor roads: Hairline (1px) until very close
    'residential':    {13: 0.5, 15: 1, 16: 3, 18: 8},
    'unclassified':   {13: 0.5, 15: 1, 16: 3, 18: 8},
    'service':        {15: 1, 17: 3, 18: 6},
    'track':          {14: 0.5, 16: 1, 18: 4},
    
    # Paths: Very subtle
    'footway':        {15: 0.5, 17: 1, 19: 3},
    'path':           {15: 0.5, 17: 1, 19: 3},
    'cycleway':       {15: 0.5, 17: 1, 19: 3},
    
    # Transport
    'railway':        {10: 1, 14: 2, 16: 3, 18: 5},
    'subway':         {12: 1, 14: 2, 16: 3},
    
    # Waterways: Drastically reduced to prevent blocking features
    'river':          {8: 1, 12: 2, 14: 4, 16: 7, 18: 10}, 
    'canal':          {10: 1, 13: 2, 15: 4, 17: 6},
    'stream':         {14: 1, 16: 2, 18: 3},
    'drain':          {15: 1, 18: 2},
    
    # Aero: Reduced so they don't cover the airport terminal buildings
    'runway':         {10: 1, 12: 3, 14: 8, 16: 16, 18: 24},
    'taxiway':        {13: 1, 15: 4, 17: 8},
}


def interpolate_width(styles: Dict[int, int], zoom: int) -> float:
    """Linearly interpolate width between defined zoom levels."""
    zooms = sorted(styles.keys())
    
    if not zooms:
        return 1.0
        
    if zoom <= zooms[0]:
        return styles[zooms[0]]
    if zoom >= zooms[-1]:
        return styles[zooms[-1]]
    
    # Find the interval [z1, z2] containing zoom
    for i in range(len(zooms) - 1):
        z1, z2 = zooms[i], zooms[i+1]
        if z1 <= zoom <= z2:
            w1, w2 = styles[z1], styles[z2]
            # Linear interpolation
            return w1 + (w2 - w1) * (zoom - z1) / (z2 - z1)
    
    return 1.0


def get_available_memory_mb() -> int:
    """Detect available system memory in MB."""
    try:
        if os.name == 'posix':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        return int(line.split()[1]) // 1024
        elif os.name == 'nt':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetPhysicallyInstalledSystemMemory.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
            memory = ctypes.c_ulonglong()
            if kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(memory)):
                return memory.value // (1024 * 1024)
    except:
        pass
    return 4096


def get_optimal_workers(cpu_count: int, available_memory_mb: int) -> int:
    """Calculate optimal number of worker threads based on system resources."""
    memory_workers = max(1, min(available_memory_mb // 1024, 6))
    cpu_workers = max(1, cpu_count - 1)
    
    if available_memory_mb <= 8192:
        max_workers = min(3, memory_workers, cpu_workers)
    else:
        max_workers = min(6, memory_workers, cpu_workers)
    
    logger.info(f"System: {cpu_count} CPU cores, {available_memory_mb}MB RAM -> Using {max_workers} workers")
    return max_workers


def get_adaptive_batch_size(zoom: int, base_batch_size: int, available_memory_mb: int) -> int:
    """Calculate adaptive batch size based on zoom level and available memory."""
    memory_factor = min(2.0, available_memory_mb / 4096)
    
    if zoom < 10:
        adjusted_size = max(2000, int(base_batch_size * 0.6 * memory_factor))
    elif zoom < 13:
        adjusted_size = max(1000, int(base_batch_size * 0.4 * memory_factor))
    elif zoom < 15:
        adjusted_size = max(500, int(base_batch_size * 0.3 * memory_factor))
    else:
        adjusted_size = max(250, int(base_batch_size * 0.2 * memory_factor))
    
    return adjusted_size


def precompute_global_color_palette(config: Dict) -> int:
    """Build global color palette from configuration."""
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
    
    logger.info(f"Palette: {len(unique_colors)} colors")
    return len(unique_colors)


def write_palette_bin(output_dir: str):
    """Write color palette to binary file."""
    os.makedirs(output_dir, exist_ok=True)
    palette_path = os.path.join(output_dir, "palette.bin")
    num_colors = len(GLOBAL_INDEX_TO_RGB332)
    
    palette_data = bytearray()
    palette_data.extend(struct.pack('<I', num_colors))
    
    for index in range(num_colors):
        if index in GLOBAL_INDEX_TO_RGB332:
            rgb332 = GLOBAL_INDEX_TO_RGB332[index]
            
            # El formato RGB332 (8 bits) se distribuye como RRRGGGYY (siendo Y los 2 bits de B)
            
            # Extrae el valor de 3 bits de R (bits 7, 6, 5) y lo expande a 8 bits (R888)
            red_3bit = (rgb332 >> 5) & 0x07
            # Expansión estándar de 3 a 8 bits
            r = (red_3bit << 5) | (red_3bit << 2) | (red_3bit >> 1)
            
            # Extrae el valor de 3 bits de G (bits 4, 3, 2) y lo expande a 8 bits (G888)
            green_3bit = (rgb332 >> 2) & 0x07
            # Expansión estándar de 3 a 8 bits
            g = (green_3bit << 5) | (green_3bit << 2) | (green_3bit >> 1)
            
            # Extrae el valor de 2 bits de B (bits 1, 0) y lo expande a 8 bits (B888)
            blue_2bit = rgb332 & 0x03
            # Expansión estándar de 2 a 8 bits
            b = (blue_2bit << 6) | (blue_2bit << 4) | (blue_2bit << 2) | blue_2bit
            
            palette_data.extend([r, g, b])
        else:
            palette_data.extend([255, 255, 255])
    
    with open(palette_path, 'wb') as f:
        f.write(bytes(palette_data))
    
    logger.info("Palette written")


def hex_to_color_index(hex_color: str) -> Optional[int]:
    """Convert hex color to palette index."""
    return GLOBAL_COLOR_PALETTE.get(hex_color)


def hex_to_rgb332(hex_color: str) -> int:
    """Convert hex color (RGB888) to RGB332 format (8-bit index)."""
    try:
        if not hex_color or not hex_color.startswith("#"):
            return 0xFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # RRR (3 bits), GGG (3 bits), YY (2 bits para B)
        return ((r & 0xE0) | ((g & 0xE0) >> 3) | (b >> 6))
    except:
        return 0xFF


def parse_width_value(width_str: str) -> Optional[float]:
    """
    Parse width value from OSM tags, handling various formats.
    Returns width in meters, or None if parsing fails.
    """
    if not width_str:
        return None
    
    width_str = str(width_str).lower().strip().replace(',', '.')
    
    pattern = r'([0-9]+\.?[0-9]*)\s*(m|meter|meters|metre|metres|ft|feet|foot|\'|"|in|inch|inches)?'
    match = re.match(pattern, width_str)
    
    if not match:
        return None
    
    try:
        value = float(match.group(1))
        unit = match.group(2) if match.group(2) else 'm'
        
        if unit in ['ft', 'feet', 'foot', '\'']:
            value = value * 0.3048
        elif unit in ['"', 'in', 'inch', 'inches']:
            value = value * 0.0254
        
        return value
    except (ValueError, AttributeError):
        return None


def meters_to_pixels_physical(meters: float, zoom: int, lat: float) -> float:
    """Convertir metros físicos a píxeles sin escalado agresivo (factor de escala eliminado)."""
    base_resolution = 156543.03392
    lat_rad = math.radians(lat)
    resolution = base_resolution * math.cos(lat_rad) / (2 ** zoom)
    pixels = meters / resolution
    return pixels


def apply_width_constraints(pixels: float, feature_type: str, zoom: int) -> int:
    """Apply STRICT min/max pixel constraints based on feature type for small screens."""
    
    # Constraints (min_pixels, max_pixels)
    # MAX values reduced by ~40-50% from original standard
    constraints = {
        'motorway': (1, 18),     'trunk': (1, 16),      'primary': (1, 16),
        'secondary': (1, 12),    'tertiary': (1, 10),
        'residential': (0.5, 10), 'unclassified': (0.5, 10),
        'service': (0.5, 6),     'track': (0.5, 5),
        'footway': (0.5, 3),     'path': (0.5, 3),      'cycleway': (0.5, 3),
        
        'railway': (1, 6),
        
        # Waterways clamped tightly
        'river': (1, 10),        
        'waterway': (0.5, 8), 
        
        'runway': (1, 24),       'taxiway': (1, 10),
        'pipeline': (0.5, 4),    'power': (0.5, 2),
    }
    
    min_width, max_width = constraints.get(feature_type, (0.5, 8))
    
    # Low zoom overrides to ensure clean rendering
    if zoom <= 12:
        max_width = min(max_width, 4) # Nothing should be thicker than 4px at zoom 12
    
    if zoom <= 10:
        max_width = min(max_width, 3) # Nothing thicker than 3px at zoom 10
        
    clamped = max(min_width, min(max_width, pixels))
    
    # Round to nearest integer but ensure at least 1px if calculation > 0.5
    if clamped < 1.0 and clamped >= 0.5:
        return 1
    return max(1, int(round(clamped)))


def get_feature_category(tags: Dict) -> str:
    """Determine the feature category for styling lookups."""
    if 'highway' in tags: return tags['highway']
    elif 'railway' in tags:
        r = tags['railway']
        return r if r in ['rail', 'subway', 'tram', 'light_rail'] else 'railway'
    elif 'waterway' in tags: return tags['waterway']
    elif 'aeroway' in tags: return tags['aeroway']
    elif 'power' in tags: return 'power'
    elif 'man_made' in tags and tags['man_made'] in ['pipeline', 'embankment']: return 'pipeline'
    return 'default'


def get_line_width_from_tags(tags: Dict, zoom: int, coordinates, geom_type: str) -> int:
    """
    Calculate line width in pixels based on:
    1. Explicit OSM tags (width, maxwidth) -> Physical Calculation
    2. If no tags -> CartoCSS-style Zoom-based Defaults
    """
    if not tags:
        return 1
    
    # Polygons usually don't have stroke width unless specified
    if geom_type in ["Polygon", "MultiPolygon"]:
        if not any(k in tags for k in ['width', 'maxwidth', 'est_width']):
            return 1
    
    feature_category = get_feature_category(tags)
    calculated_pixels = 0.0
    width_found = False

    # 1. Try explicit physical width tags
    width_tags = ['width', 'maxwidth', 'est_width', 'diameter', 'gauge']
    for width_tag in width_tags:
        if width_tag in tags:
            width_meters = parse_width_value(tags[width_tag])
            if width_meters is not None and width_meters > 0:
                # Calculate average latitude for projection
                avg_lat = 0.0
                try:
                    if geom_type in ["LineString", "MultiLineString"]:
                        coords = coordinates if geom_type == "LineString" else (coordinates[0] if coordinates else [])
                        valid_coords = [c for c in coords if len(c) >= 2]
                        if valid_coords:
                            avg_lat = sum(c[1] for c in valid_coords) / len(valid_coords)
                    elif geom_type in ["Polygon", "MultiPolygon"]:
                         # Use first point of exterior ring
                         c = coordinates[0] if geom_type == "Polygon" else coordinates[0][0]
                         if c and len(c) > 0: avg_lat = c[0][1]
                except:
                    avg_lat = 0.0
                
                calculated_pixels = meters_to_pixels_physical(width_meters, zoom, avg_lat)
                width_found = True
                break
    
    # 2. If no explicit tag, use Carto-style defaults (Pixels based on Zoom)
    if not width_found:
        if feature_category in CARTO_STYLES:
            calculated_pixels = interpolate_width(CARTO_STYLES[feature_category], zoom)
        else:
            # Fallback for unknown features
            calculated_pixels = 1.5 if zoom > 14 else 1.0
    
    # 3. Apply Clamping (Min/Max constraints) regardless of source
    # This ensures a 500m wide river doesn't cover the screen (clamped tightly)
    final_width = apply_width_constraints(calculated_pixels, feature_category, zoom)
    
    return final_width


def interpolate_curve_points(points: List, max_segment_distance: float = 0.00005) -> List:
    """Interpolate additional points to smooth curves - works on geographic coordinates."""
    if len(points) < 2:
        return points
    
    result = [points[0]]
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate number of segments needed
        if distance > max_segment_distance:
            num_segments = int(math.ceil(distance / max_segment_distance))
            # Add intermediate points
            for j in range(1, num_segments):
                t = j / num_segments
                result.append([
                    p1[0] + dx * t,
                    p1[1] + dy * t
                ])
        
        result.append(p2)
    
    return result


def smooth_geometry_coords(coordinates, geom_type: str, zoom: int, tags: Dict = None):
    """Smooth geometry by adding intermediate points - applies to geographic coordinates."""
    # Iniciar suavizado a partir de zoom 16
    if zoom < 16:
        return coordinates
    
    # Determine maximum segment distance based on zoom
    is_rbt = tags and tags.get('junction') == 'roundabout'
    
    if zoom >= 19:
        max_dist = 0.000008 if is_rbt else 0.00001
    elif zoom >= 18:
        max_dist = 0.00001 if is_rbt else 0.000015
    elif zoom >= 17:
        max_dist = 0.000015 if is_rbt else 0.00002
    else:  # zoom 16
        max_dist = 0.00002 if is_rbt else 0.00003
    
    def smooth_linestring(coords):
        if len(coords) < 2:
            return coords
        return interpolate_curve_points(coords, max_dist)
    
    def smooth_ring(ring):
        if not ring or len(ring) < 3:
            return ring
        # For closed rings, remove last point, interpolate, then re-close
        is_closed = ring[0] == ring[-1]
        work_ring = ring[:-1] if is_closed else ring
        smoothed = interpolate_curve_points(work_ring, max_dist)
        if is_closed and smoothed:
            smoothed.append(smoothed[0])
        return smoothed
    
    if geom_type == "LineString":
        return smooth_linestring(coordinates)
    elif geom_type == "Polygon":
        if not coordinates or not coordinates[0]:
            return coordinates
        return [smooth_ring(coordinates[0])]
    elif geom_type == "MultiLineString":
        return [smooth_linestring(line) for line in coordinates]
    elif geom_type == "MultiPolygon":
        result = []
        for poly in coordinates:
            if poly and poly[0]:
                result.append([smooth_ring(poly[0])])
        return result
    
    return coordinates


def geojson_bounds(coordinates, geom_type: str) -> Tuple[float, float, float, float]:
    """Calculate bounding box for geometry coordinates."""
    def flatten_coords(coords):
        if not coords:
            return []
        if isinstance(coords, (int, float, Decimal)):
            return [float(coords)]
        if len(coords) == 2:
            try:
                lon = float(coords[0]) if not isinstance(coords[0], (int, float)) else coords[0]
                lat = float(coords[1]) if not isinstance(coords[1], (int, float)) else coords[1]
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
    """Check if geometry intersects with tile bounding box."""
    minx, miny, maxx, maxy = geojson_bounds(coordinates, geom_type)
    tile_minx, tile_miny, tile_maxx, tile_maxy = tile_bbox
    
    if maxx < tile_minx or minx > tile_maxx or maxy < tile_miny or miny > tile_maxy:
        return False
    return True


def normalize_coordinates_robust(coordinates):
    """Normalize coordinates to ensure all values are floats."""
    if isinstance(coordinates, (int, float, Decimal)):
        return float(coordinates)
    
    if isinstance(coordinates, (list, tuple)):
        if len(coordinates) == 2:
            first, second = coordinates[0], coordinates[1]
            if (isinstance(first, (int, float, Decimal)) and 
                isinstance(second, (int, float, Decimal))):
                return [float(first), float(second)]
        
        return [normalize_coordinates_robust(item) for item in coordinates]
    
    return coordinates


def clip_polygon_sutherland_hodgman(vertices, minx, miny, maxx, maxy):
    """Clip polygon to bounding box using Sutherland-Hodgman algorithm."""
    if not vertices or len(vertices) < 3:
        return []
    
    def inside(p, edge):
        if edge == 'left':
            return p[0] >= minx
        elif edge == 'right':
            return p[0] <= maxx
        elif edge == 'bottom':
            return p[1] >= miny
        elif edge == 'top':
            return p[1] <= maxy
        return False
    
    def compute_intersection(p1, p2, edge):
        x1, y1 = float(p1[0]), float(p1[1]) 
        x2, y2 = float(p2[0]), float(p2[1])
        
        # Evitar división por cero
        if edge == 'left':
            x = minx
            y = y1 + (y2 - y1) * (minx - x1) / (x2 - x1) if abs(x2 - x1) > 1e-9 else y1
        elif edge == 'right':
            x = maxx
            y = y1 + (y2 - y1) * (maxx - x1) / (x2 - x1) if abs(x2 - x1) > 1e-9 else y1
        elif edge == 'bottom':
            y = miny
            x = x1 + (x2 - x1) * (miny - y1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
        elif edge == 'top':
            y = maxy
            x = x1 + (x2 - x1) * (maxy - y1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
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
    """Clip line segment to bounding box using Cohen-Sutherland algorithm."""
    minx, miny, maxx, maxy = bbox
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    
    def compute_outcode(x, y):
        code = 0
        if x < minx:
            code |= 1
        if x > maxx:
            code |= 2
        if y < miny:
            code |= 4
        if y > maxy:
            code |= 8
        return code
    
    outcode1 = compute_outcode(x1, y1)
    outcode2 = compute_outcode(x2, y2)
    
    while True:
        if outcode1 == 0 and outcode2 == 0:
            return ([x1, y1], [x2, y2])
        if (outcode1 & outcode2) != 0:
            return None
        
        outcode = outcode1 if outcode1 != 0 else outcode2
        
        # Evitar división por cero
        if outcode & 8:
            x = x1 + (x2 - x1) * (maxy - y1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
            y = maxy
        elif outcode & 4:
            x = x1 + (x2 - x1) * (miny - y1) / (y2 - y1) if abs(y2 - y1) > 1e-9 else x1
            y = miny
        elif outcode & 2:
            y = y1 + (y2 - y1) * (maxx - x1) / (x2 - x1) if abs(x2 - x1) > 1e-9 else y1
            x = maxx
        elif outcode & 1:
            y = y1 + (y2 - y1) * (minx - x1) / (x2 - x1) if abs(x2 - x1) > 1e-9 else y1
            x = minx
            
        if outcode == outcode1:
            x1, y1 = x, y
            outcode1 = compute_outcode(x1, y1)
        else:
            x2, y2 = x, y
            outcode2 = compute_outcode(x2, y2)


def clip_coordinates_to_tile(coordinates, geom_type: str, tile_bbox: Tuple[float, float, float, float]):
    """Clip geometry coordinates to tile bounding box."""
    minx, miny, maxx, maxy = tile_bbox
    
    def clip_linestring(coords):
        if not coords or len(coords) < 2:
            return []
        clipped = []
        # TOLERANCIA AUMENTADA A 1e-5 para corregir los problemas de continuidad.
        TOLERANCE = 1e-5 
        for i in range(len(coords) - 1):
            result = clip_line_to_bbox(coords[i], coords[i + 1], tile_bbox)
            if result:
                p1, p2 = result
                
                # REVISIÓN DE CLIPPING: Usar distancia para verificar duplicados en el borde del tile.
                is_duplicate = False
                if clipped:
                    # Compara la distancia entre el último punto añadido (p2 del segmento anterior) y p1 del segmento actual
                    dist = math.hypot(clipped[-1][0] - p1[0], clipped[-1][1] - p1[1])
                    if dist < TOLERANCE:
                        is_duplicate = True
                        
                if not is_duplicate:
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
    """Convert lat/lon to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_bbox(tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
    """Calculate tile bounding box in lat/lon coordinates."""
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
    return (lon_min, lat_min, lon_max, lat_max)


def coords_to_pixels(coordinates, zoom: int, tile_x: int, tile_y: int) -> List[Tuple[int, int]]:
    """Convert lat/lon coordinates to pixel coordinates within tile."""
    n = 2.0 ** zoom
    pixels = []
    
    for coord in coordinates:
        lon = float(coord[0]) if not isinstance(coord[0], (int, float)) else coord[0]
        lat = float(coord[1]) if not isinstance(coord[1], (int, float)) else coord[1]
            
        x = ((lon + 180.0) / 360.0 * n - tile_x) * TILE_SIZE
        lat_rad = math.radians(lat)
        y = ((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n - tile_y) * TILE_SIZE
        
        # Scale float pixel (0-256) to uint16 pixel (0-65536)
        x_uint16 = int((x * (UINT16_TILE_SIZE - 1)) / (TILE_SIZE - 1))
        y_uint16 = int((y * (UINT16_TILE_SIZE - 1)) / (TILE_SIZE - 1))
        
        # Clamp to valid uint16 range but preserve precision
        x_uint16 = max(-UINT16_TILE_SIZE, min(UINT16_TILE_SIZE * 2, x_uint16))
        y_uint16 = max(-UINT16_TILE_SIZE, min(UINT16_TILE_SIZE * 2, y_uint16))
        
        pixels.append((x_uint16, y_uint16))
    
    return pixels


def pack_varint(n):
    """Pack integer as variable-length integer."""
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
    """Pack signed integer using zigzag encoding."""
    return pack_varint((n << 1) ^ (n >> 31))


def get_layer_priority(tags: Dict) -> int:
    """Calculate rendering priority based on OSM layer tag and feature type."""
    if 'layer' in tags:
        try:
            layer_val = int(tags['layer'])
            return layer_val * 1000
        except (ValueError, TypeError):
            pass

    # Water features at bottom
    if 'natural' in tags and tags['natural'] in ['water', 'coastline', 'bay']:
        return 100
    if 'waterway' in tags and tags.get('waterway') in ['riverbank', 'dock', 'boatyard']:
        return 100

    # Land use and natural areas
    if 'landuse' in tags:
        return 200
    if 'natural' in tags and tags['natural'] in ['wood', 'forest', 'scrub', 'heath', 'grassland', 'beach', 'sand', 'wetland']:
        return 200
    if 'leisure' in tags and tags['leisure'] in ['park', 'nature_reserve', 'garden']:
        return 200

    # Waterways
    if 'waterway' in tags and tags['waterway'] in ['river', 'stream', 'canal']:
        return 300

    # Natural features
    if 'natural' in tags and tags['natural'] in ['peak', 'ridge', 'volcano', 'cliff']:
        return 400

    # Underground features
    if 'tunnel' in tags and tags['tunnel'] == 'yes':
        return 500

    # Railways
    if 'railway' in tags:
        return 600

    # Minor roads
    if 'highway' in tags and tags['highway'] in ['path', 'footway', 'cycleway', 'steps', 'pedestrian', 'track']:
        return 700

    # Tertiary roads
    if 'highway' in tags and tags['highway'] in ['tertiary', 'tertiary_link']:
        return 800

    # Secondary roads
    if 'highway' in tags and tags['highway'] in ['secondary', 'secondary_link']:
        return 900

    # Primary roads
    if 'highway' in tags and tags['highway'] in ['primary', 'primary_link']:
        return 1000

    # Trunk roads
    if 'highway' in tags and tags['highway'] in ['trunk', 'trunk_link']:
        return 1100

    # Motorways
    if 'highway' in tags and tags['highway'] in ['motorway', 'motorway_link']:
        return 1200

    # Above-ground structures
    if 'bridge' in tags and tags['bridge'] == 'yes':
        return 1300
    if 'aeroway' in tags:
        return 1300

    # Buildings
    if 'building' in tags:
        return 1400

    # Points of interest
    if 'amenity' in tags:
        return 1500

    return 200


def geometry_to_commands(geom_type: str, coordinates, color: int, zoom: int, tile_x: int, tile_y: int, 
                         color_index: Optional[int] = None, tags: Dict = None) -> List[Dict]:
    """Convert geometry to draw commands with appropriate line widths."""
    commands = []
    
    if color_index is not None:
        commands.append({'type': DRAW_COMMANDS['SET_COLOR_INDEX'], 'color_index': color_index})
    else:
        commands.append({'type': DRAW_COMMANDS['SET_COLOR'], 'color': color})
    
    line_width = get_line_width_from_tags(tags, zoom, coordinates, geom_type)
    
    def process_linestring(coords, width):
        if len(coords) < 2:
            return []
        
        pixels = coords_to_pixels(coords, zoom, tile_x, tile_y)
        
        # Don't remove duplicate pixels - they're needed for smooth wide lines
        unique_pixels = pixels
        
        if len(unique_pixels) < 2:
            return []
        
        # Always use POLYLINE for better rendering of wide lines
        return [{'type': DRAW_COMMANDS['POLYLINE'], 'points': unique_pixels, 'width': width}]
    
    def process_polygon(coords, width):
        if not coords or not coords[0] or len(coords[0]) < 3:
            return []
        
        exterior = coords[0]
        pixels = coords_to_pixels(exterior, zoom, tile_x, tile_y)
        
        # Don't remove duplicates for polygons either
        unique_pixels = pixels
        
        # Close the ring if it's not already closed
        if unique_pixels and unique_pixels[0] != unique_pixels[-1]:
            unique_pixels.append(unique_pixels[0])
        
        if len(set(unique_pixels)) < 3:
            return []
        
        return [{'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': unique_pixels, 'width': width}]
    
    if geom_type == "LineString":
        commands.extend(process_linestring(coordinates, line_width))
    elif geom_type == "Polygon":
        commands.extend(process_polygon(coordinates, line_width))
    elif geom_type == "MultiLineString":
        for line in coordinates:
            line_commands = process_linestring(line, line_width)
            if line_commands:
                commands.extend(line_commands)
    elif geom_type == "MultiPolygon":
        for poly in coordinates:
            poly_commands = process_polygon(poly, line_width)
            if poly_commands:
                commands.extend(poly_commands)
    
    return commands


def pack_draw_commands(commands: List[Dict]) -> bytes:
    """Pack draw commands into binary format."""
    out = bytearray()
    out += pack_varint(len(commands))
    
    for cmd in commands:
        cmd_type = cmd['type']
        out += pack_varint(cmd_type)
        
        if cmd_type == DRAW_COMMANDS['SET_COLOR']:
            out += struct.pack("B", cmd.get('color', 0xFF))
        elif cmd_type == DRAW_COMMANDS['SET_COLOR_INDEX']:
            out += pack_varint(cmd.get('color_index', 0))
        elif cmd_type in [DRAW_COMMANDS['POLYLINE'], DRAW_COMMANDS['STROKE_POLYGON']]:
            points = cmd['points']
            width = cmd.get('width', 1)
            out += pack_varint(width)
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
    """Compress configuration into optimized GOQL query."""
    META = {'tile_size', 'viewport_size', 'toolbar_width', 'statusbar_height', 'max_cache_size', 
            'thread_pool_size', 'background_colors', 'log_level', 'config_file', 'fps_limit', 'fill_polygons'}
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
            is_lin = (tag in LINEAR_VALUES and val in LINEAR_VALUES[tag]) or tag in LINEAR_TAGS
            is_area = tag in AREA_VALUES and val in AREA_VALUES[tag]
            
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
            parts.append("w[" + tag + "~\"^(" + pat + ")$\"]")
    
    area_grp, area_single = group(area_q)
    for tag, vals in sorted(area_grp.items()):
        if len(vals) == 1:
            parts.append("nwa[" + tag + "=" + vals[0] + "]")
        else:
            pat = '|'.join(vals)
            parts.append("nwa[" + tag + "~\"^(" + pat + ")$\"]")
    
    for tag in sorted(area_single):
        parts.append("nwa[" + tag + "]")
    
    return ", ".join(parts) if parts else "*"


def process_feature(feature: Dict, config: Dict, zoom: int, tiles_data: Dict):
    """Process a single GeoJSON feature and add it to tiles_data."""
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
    
    coordinates = normalize_coordinates_robust(coordinates)
    
    # 1. Aplicar suavizado (smoothing) para alta fidelidad en curvas (Z >= 16).
    coordinates = smooth_geometry_coords(coordinates, geom_type, zoom, props)
    
    hex_color = style.get('color', '#FFFFFF')
    color = hex_to_rgb332(hex_color)
    color_index = hex_to_color_index(hex_color)
    priority = style.get('priority', 50)
    layer_priority = get_layer_priority(props)
    
    minx, miny, maxx, maxy = geojson_bounds(coordinates, geom_type)
    tile_x_min, tile_y_min = deg2num(miny, minx, zoom)
    tile_x_max, tile_y_max = deg2num(maxy, maxx, zoom)
    
    # 3. Iterar y clip (Cohen-Sutherland/Sutherland-Hodgman)
    for tx in range(min(tile_x_min, tile_x_max), max(tile_x_min, tile_x_max) + 1):
        for ty in range(min(tile_y_min, tile_y_max), max(tile_y_min, tile_y_max) + 1):
            tile_bounds = tile_bbox(tx, ty, zoom)
            
            if not coords_intersect_tile(coordinates, geom_type, tile_bounds):
                continue
            
            clipped = clip_coordinates_to_tile(coordinates, geom_type, tile_bounds)
            if not clipped:
                continue
            
            commands = geometry_to_commands(geom_type, clipped, color, zoom, tx, ty, color_index, props)
            if commands:
                tile_key = (tx, ty)
                if tile_key not in tiles_data:
                    tiles_data[tile_key] = []
                combined_priority = layer_priority + priority
                tiles_data[tile_key].append({'commands': commands, 'priority': combined_priority})


def get_style_for_tags(tags: Dict, config: Dict) -> Optional[Dict]:
    """Find style configuration for given OSM tags."""
    for k, v in tags.items():
        keyval = k + "=" + str(v)
        if keyval in config and isinstance(config[keyval], dict):
            return config[keyval]
    
    for k in tags:
        if k in config and isinstance(config[k], dict):
            return config[k]
    
    return None


def sort_and_flatten_commands(tile_entries: List[Dict]) -> List[Dict]:
    """Sort tile entries by priority and flatten to command list."""
    sorted_entries = sorted(tile_entries, key=lambda x: x['priority'])
    all_commands = []
    for entry in sorted_entries:
        all_commands.extend(entry['commands'])
    return all_commands


def write_single_tile(job):
    """Write a single tile to disk."""
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
    """Format seconds into human-readable time string."""
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
    """Merge source tiles data into target dictionary."""
    for tile_key, entries in source.items():
        if tile_key in target:
            target[tile_key].extend(entries)
        else:
            target[tile_key] = entries


def write_tiles_batch(tiles_data: Dict, zoom: int, output_dir: str, max_file_size: int, 
                      max_workers: int, persistent_tiles: Dict):
    """Merge current batch into persistent tiles storage."""
    merge_tiles_data(persistent_tiles, tiles_data)
    return len(tiles_data)


def write_final_tiles(persistent_tiles: Dict, zoom: int, output_dir: str, max_file_size: int, max_workers: int):
    """Write all accumulated tiles to disk using thread pool."""
    if not persistent_tiles:
        return 0
    
    tile_jobs = [(tx, ty, entries, zoom, output_dir, max_file_size) 
                 for (tx, ty), entries in persistent_tiles.items()]
    tiles_written = len(tile_jobs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(write_single_tile, job) for job in tile_jobs]
        for future in as_completed(futures):
            future.result()
    
    return tiles_written


def generate_tiles(gol_file: str, output_dir: str, config_file: str, zoom_levels: List[int], 
                   max_file_size: int = 65536, base_batch_size: int = 10000):
    """Main tile generation function."""
    with open(config_file) as f:
        config = json.load(f)
    
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
        adaptive_batch = get_adaptive_batch_size(zoom, base_batch_size, available_memory_mb)
        logger.info(f"Processing zoom {zoom} (batch size: {adaptive_batch}, workers: {max_workers})...")
        
        process = subprocess.Popen(
            [gol_cmd, "query", gol_file, query, "-f", "geojson"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=65536
        )
        
        tiles_data = {}
        persistent_tiles = {}
        feature_count = 0
        batch_count = 0
        stderr_content = ""
        
        try:
            for feature in ijson.items(process.stdout, "features.item"):
                feature_count += 1
                process_feature(feature, config, zoom, tiles_data)
                
                if feature_count % adaptive_batch == 0:
                    batch_count += 1
                    write_tiles_batch(tiles_data, zoom, output_dir, max_file_size, max_workers, persistent_tiles)
                    print(f"\rZoom {zoom}: {feature_count} features, {len(persistent_tiles)} unique tiles (batch {batch_count})...", 
                          end='', flush=True)
                    tiles_data.clear()
                    gc.collect()
                elif feature_count % 5000 == 0:
                    print(f"\rZoom {zoom}: {feature_count} features, {len(tiles_data)} tiles in buffer, {len(persistent_tiles)} persistent...", 
                          end='', flush=True)
            
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
    """Command-line entry point for tile generator."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description='Generate map tiles from GoL file with OSM-compliant width handling'
    )
    parser.add_argument("gol_file", help="Input GoL file path")
    parser.add_argument("output_dir", help="Output directory for tiles")
    parser.add_argument("config_file", help="JSON config file with feature definitions")
    parser.add_argument("--zoom", default="6-17", help="Zoom level(s) to generate (e.g., '12' or '10-15')")
    parser.add_argument("--max-file-size", type=int, default=128, help="Maximum tile file size in KB")
    parser.add_argument("--batch-size", type=int, default=10000, 
                       help="Base batch size (auto-adjusted per zoom level and system RAM)")
    
    args = parser.parse_args()
    
    if '-' in args.zoom:
        start, end = map(int, args.zoom.split('-'))
        zoom_levels = list(range(start, end + 1))
    else:
        zoom_levels = [int(args.zoom)]
    
    max_file_size_bytes = args.max_file_size * 1024
    
    logger.info("=" * 50)
    logger.info("OSM-Compliant Tile Generator (CLEANED and CLIPPING FIXED)")
    logger.info("=" * 50)
    logger.info(f"Input: {args.gol_file}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Zoom levels: {zoom_levels}")
    logger.info(f"Max tile size: {args.max_file_size}KB")
    logger.info(f"Base batch size: {args.batch_size} features (adaptive per zoom and system RAM)")
    logger.info(f"Simplification: DISABLED (Geometry preserved)") 
    logger.info(f"Width handling: Hybrid (OSM Tags + CartoCSS Defaults) - SCALED DOWN for Small Screens")
    logger.info("=" * 50)
    
    generate_tiles(
        args.gol_file,
        args.output_dir,
        args.config_file,
        zoom_levels,
        max_file_size_bytes,
        args.batch_size
    )
    
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
    
    if total_count > 0:
        if total_size < 1024:
            size_str = f"{total_size}B"
        elif total_size < 1024 * 1024:
            size_str = f"{round(total_size / 1024, 1)}KB"
        elif total_size < 1024 * 1024 * 1024:
            size_str = f"{round(total_size / (1024 * 1024), 1)}MB"
        else:
            size_str = f"{round(total_size / (1024 * 1024 * 1024), 2)}GB"
        
        avg_size = round(total_size / total_count)
    else:
        size_str = "0B"
        avg_size = 0
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("=" * 50)
    logger.info("Generation Summary")
    logger.info("=" * 50)
    logger.info(f"Total tiles written: {total_count}")
    logger.info(f"Total size: {size_str}")
    logger.info(f"Average tile size: {avg_size} bytes")
    logger.info(f"Palette file: {os.path.join(args.output_dir, 'palette.bin')}")
    logger.info(f"Total processing time: {format_time(elapsed_time)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()