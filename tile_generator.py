#!/usr/bin/env python3
import osmium
import shapely.wkb as wkblib
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, box, shape
)
import math
import struct
import json
import os
import shutil
from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import tempfile
import subprocess
import sys
import fiona
import gc
import threading
import time
import sqlite3
import pickle
import hashlib

import ijson
import decimal

import gc
import psutil
import threading
from collections import OrderedDict
from threading import RLock
import sys
import weakref

from concurrent.futures import ProcessPoolExecutor, as_completed

max_workers = min(os.cpu_count() or 4, 4)

TILE_SIZE = 256
DRAW_COMMANDS = {
    'LINE': 1,
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,
    'HORIZONTAL_LINE': 5,
    'VERTICAL_LINE': 6,
    'SET_COLOR': 0x80,  # State command for direct RGB332 color
    'SET_COLOR_INDEX': 0x81,  # State command for palette index
    'RECTANGLE': 0x82,  # Optimized rectangle for buildings
    'STRAIGHT_LINE': 0x83,  # Optimized straight line for highways
    'HIGHWAY_SEGMENT': 0x84,  # Highway segment with continuity
    'GRID_PATTERN': 0x85,  # Urban grid pattern
    'BLOCK_PATTERN': 0x86,  # City block pattern
    'CIRCLE': 0x87,  # Circle/roundabout
    'RELATIVE_MOVE': 0x88,  # Relative coordinate movement
    'PREDICTED_LINE': 0x89,  # Predictive line based on pattern
    'COMPRESSED_POLYLINE': 0x8A,  # Huffman-compressed polyline
}

UINT16_TILE_SIZE = 65536

# Global variables for dynamic palette
GLOBAL_COLOR_PALETTE = {}  # hex_color -> index
GLOBAL_INDEX_TO_RGB332 = {}  # index -> rgb332_value

# Global variables for feature detection
DETECTED_FEATURE_TYPES = set()  # highway, building, waterway, etc.

# Step 6: Advanced compression globals
COMMAND_FREQUENCY = Counter()  # For Huffman encoding
HUFFMAN_CODES = {}  # command_type -> encoded_bits
COORDINATE_PREDICTORS = {}  # For coordinate prediction
PATTERN_CACHE = {}  # For pattern detection cache

# Database configuration for feature storage
DB_BATCH_SIZE = 10000

class FeatureDatabase:
    """Database for storing and retrieving features by zoom level and tile coordinates"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zoom_level INTEGER NOT NULL,
                tile_x INTEGER NOT NULL,
                tile_y INTEGER NOT NULL,
                feature_data BLOB NOT NULL,
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom_tile 
            ON features(zoom_level, tile_x, tile_y)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom 
            ON features(zoom_level)
        """)
        self.conn.commit()
    
    def insert_feature(self, zoom_level, tile_x, tile_y, feature_data, priority=5):
        """Insert a feature into the database"""
        self.conn.execute("""
            INSERT INTO features (zoom_level, tile_x, tile_y, feature_data, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (zoom_level, tile_x, tile_y, pickle.dumps(feature_data), priority))
    
    def get_features_for_tile(self, zoom_level, tile_x, tile_y):
        """Get all features for a specific tile"""
        cursor = self.conn.execute("""
            SELECT feature_data, priority FROM features 
            WHERE zoom_level = ? AND tile_x = ? AND tile_y = ?
            ORDER BY priority
        """, (zoom_level, tile_x, tile_y))
        
        features = []
        for row in cursor.fetchall():
            feature_data = pickle.loads(row[0])
            features.append(feature_data)
        return features
    
    def get_tiles_for_zoom(self, zoom_level):
        """Get all tile coordinates for a specific zoom level"""
        cursor = self.conn.execute("""
            SELECT DISTINCT tile_x, tile_y FROM features 
            WHERE zoom_level = ?
        """, (zoom_level,))
        return cursor.fetchall()
    
    def count_features_for_zoom(self, zoom_level):
        """Count total features for a zoom level"""
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM features WHERE zoom_level = ?
        """, (zoom_level,))
        return cursor.fetchone()[0]
    
    def clear_zoom(self, zoom_level):
        """Clear all features for a specific zoom level"""
        self.conn.execute("DELETE FROM features WHERE zoom_level = ?", (zoom_level,))
        self.conn.commit()
    
    def close(self):
        """Close the database connection"""
        self.conn.close()
    
    def commit(self):
        """Commit pending transactions"""
        self.conn.commit()

def extract_layer_to_tmpfile(args):
    layer, i, layers, geojson_file, pbf_file, config, LAYER_FIELDS, config_fields = args
    logs = []
    possible = LAYER_FIELDS[layer]
    available = get_layer_fields_from_pbf(pbf_file, layer)
    allowed = possible & available & config_fields
    where_clause = build_ogr2ogr_where_clause_from_config(config, allowed)
    if not where_clause:
        logs.append(f"Skipping layer {layer}: no matching fields in config/PBF.")
        return None, logs
    logs.append(f"[{i+1}/{len(layers)}] Extracted layer: {layer} (fields: {', '.join(sorted(allowed))})")
    tmp_layer_file = f"{geojson_file}_{layer}.tmp"
    if os.path.exists(tmp_layer_file):
        os.remove(tmp_layer_file)
    select_fields = ",".join(sorted(allowed))
    cmd = [
        "ogr2ogr",
        "-f", "GeoJSON",
        "-nlt", "PROMOTE_TO_MULTI",
        "-where", where_clause,
        "-select", select_fields,
        tmp_layer_file,
        pbf_file,
        layer
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logs.append(f"Error running ogr2ogr for layer {layer}: {result.stderr.decode()}")
        return None, logs
    
    gc.collect()
    return tmp_layer_file, logs

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError

def read_tmp_geojson_features_stream(tmp_file, config_fields):
    with open(tmp_file, "r", encoding="utf-8") as f:
        for feat in ijson.items(f, "features.item"):
            feat['properties'] = {k: v for k, v in feat['properties'].items() if k in config_fields}
            yield feat

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def deg2pixel(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = ((lon_deg + 180.0) / 360.0 * n * TILE_SIZE)
    y = ((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE)
    return x, y

def coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y):
    pixel_coords = []
    for lon, lat in coords:
        px_global, py_global = deg2pixel(lat, lon, zoom)
        x = (px_global - tile_x * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1)
        y = ((py_global - tile_y * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1))
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(UINT16_TILE_SIZE - 1, x))
        y = max(0, min(UINT16_TILE_SIZE - 1, y))
        pixel_coords.append((x, y))
    return pixel_coords

def remove_duplicate_points(points):
    if len(points) <= 1:
        return points
    result = [points[0]]
    for pt in points[1:]:
        if pt != result[-1]:
            result.append(pt)
    return result

def hex_to_rgb332_direct(hex_color):
    try:
        if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#"):
            return 0xFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r & 0xE0) | ((g & 0xE0) >> 3) | (b >> 6))
    except Exception:
        return 0xFF

def hex_to_color_index(hex_color):
    global GLOBAL_COLOR_PALETTE
    return GLOBAL_COLOR_PALETTE.get(hex_color, None)

def insert_palette_commands(commands):
    if not commands:
        return commands, 0
    
    result = []
    current_color_index = None
    color_commands_inserted = 0
    
    for cmd in commands:
        cmd_color_hex = cmd.get('color_hex')  # We need the original hex
        cmd_color_rgb332 = cmd.get('color')   # RGB332 for fallback
        
        # Try using palette first
        color_index = hex_to_color_index(cmd_color_hex) if cmd_color_hex else None
        
        if color_index is not None:
            # Use optimized palette
            if color_index != current_color_index:
                result.append({
                    'type': DRAW_COMMANDS['SET_COLOR_INDEX'], 
                    'color_index': color_index
                })
                current_color_index = color_index
                color_commands_inserted += 1
        else:
            # Fallback to direct SET_COLOR
            if cmd_color_rgb332 != current_color_index:  # Reset current_color_index
                result.append({
                    'type': DRAW_COMMANDS['SET_COLOR'], 
                    'color': cmd_color_rgb332
                })
                current_color_index = cmd_color_rgb332
                color_commands_inserted += 1
        
        # Add command without color fields
        cmd_copy = {k: v for k, v in cmd.items() if k not in ['color', 'color_hex']}
        result.append(cmd_copy)
    
    # Calculate real savings
    original_commands = len(commands)
    optimized_commands = len(result)
    colors_removed = original_commands  # Each original command had a color
    colors_added = color_commands_inserted  # SET_COLOR* commands added
    net_savings = colors_removed - colors_added  # Net savings in bytes
    
    return result, net_savings

def get_style_for_tags(tags, config):
    for k, v in tags.items():
        keyval = f"{k}={v}"
        if keyval in config:
            return config[keyval], keyval
    for k in tags:
        if k in config:
            return config[k], k
    return {}, None

def tile_latlon_bounds(tile_x, tile_y, zoom, pixel_margin=0):
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lat_rad1 = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_max = math.degrees(lat_rad1)
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_rad2 = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n)))
    lat_min = math.degrees(lat_rad2)
    return lon_min, lat_min, lon_max, lat_max

def is_area(tags):
    AREA_TAGS = {
        'building','landuse','amenity','leisure','tourism','waterway','natural','man_made',
        'boundary','place','aeroway','area','shop','craft','office','historic','public_transport',
        'emergency','military','ruins','power','sport','route','parking','park','garden','cemetery',
        'playground','school','university','hospital','forest','wood','meadow','farmland','orchard',
        'vineyard','wetland','scrub','heath','grass','beach','lake','reservoir','basin','pond',
        'swimming_pool','pitch','golf_course','stadium','sports_centre','theatre','museum','zoo','theme_park',
    }
    AREA_TAGS_EXCEPTIONS = {
        'waterway': {'riverbank', 'dock', 'reservoir', 'basin', 'canal', 'pond', 'ditch', 'fish_pass', 'moat', 'wetland'},
        'natural': {'water', 'wood', 'scrub', 'wetland', 'heath', 'grassland', 'sand', 'beach', 'glacier', 'fell', 'bare_rock', 'scree', 'shingle', 'bay', 'cape'},
    }
    if 'area' in tags:
        val = tags['area'].lower()
        if val == 'yes':
            return True
        elif val == 'no':
            return False
    for k, v in tags.items():
        if k in AREA_TAGS:
            if k == 'waterway':
                if v in AREA_TAGS_EXCEPTIONS.get('waterway', set()):
                    return True
                else:
                    return False
            if k == 'natural':
                if v in AREA_TAGS_EXCEPTIONS.get('natural', set()):
                    return True
                else:
                    return False
            return True
    return False

def get_simplify_tolerance_for_zoom(zoom):
    if zoom <= 10:
        return 0.05
    else:
        return None

def clamp_uint16(x):
    return max(0, min(UINT16_TILE_SIZE - 1, int(x)))

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

def pack_draw_commands(commands):
    out = bytearray()
    out += pack_varint(len(commands))
    
    for cmd in commands:
        t = cmd['type']
        out += pack_varint(t)
        
        if t == DRAW_COMMANDS['SET_COLOR']:
            # Original SET_COLOR command (direct RGB332)
            color = cmd['color'] & 0xFF
            out += struct.pack("B", color)
        elif t == DRAW_COMMANDS['SET_COLOR_INDEX']:
            # SET_COLOR_INDEX command (palette index)
            color_index = cmd['color_index'] & 0xFF
            out += pack_varint(color_index)
        elif t == DRAW_COMMANDS['RECTANGLE']:
            # Optimized RECTANGLE command
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == DRAW_COMMANDS['STRAIGHT_LINE']:
            # Optimized STRAIGHT_LINE command
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == DRAW_COMMANDS['GRID_PATTERN']:
            # Step 6: Grid pattern command
            x, y, width, spacing, count = map(clamp_uint16, [cmd['x'], cmd['y'], cmd['width'], cmd['spacing'], cmd['count']])
            direction = cmd.get('direction', 'horizontal')
            out += pack_zigzag(x)
            out += pack_zigzag(y)
            out += pack_zigzag(width)
            out += pack_zigzag(spacing)
            out += pack_varint(count)
            out += struct.pack("B", 1 if direction == 'horizontal' else 0)
        elif t == DRAW_COMMANDS['CIRCLE']:
            # Step 6: Circle command
            center_x, center_y, radius = map(clamp_uint16, [cmd['center_x'], cmd['center_y'], cmd['radius']])
            out += pack_zigzag(center_x)
            out += pack_zigzag(center_y)
            out += pack_zigzag(radius)
        elif t == DRAW_COMMANDS['PREDICTED_LINE']:
            # Step 6: Predicted line command (only end point needed)
            end_x, end_y = map(clamp_uint16, [cmd['end_x'], cmd['end_y']])
            out += pack_zigzag(end_x)
            out += pack_zigzag(end_y)
        else:
            # Original geometric commands
            if t == DRAW_COMMANDS['LINE']:
                x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
                out += pack_zigzag(x1)
                out += pack_zigzag(y1)
                out += pack_zigzag(x2 - x1)
                out += pack_zigzag(y2 - y1)
            elif t == DRAW_COMMANDS['POLYLINE'] or t == DRAW_COMMANDS['STROKE_POLYGON']:
                pts = cmd['points']
                out += pack_varint(len(pts))
                prev_x, prev_y = 0, 0
                for i, (x, y) in enumerate(pts):
                    x, y = clamp_uint16(x), clamp_uint16(y)
                    if i == 0:
                        out += pack_zigzag(x)
                        out += pack_zigzag(y)
                    else:
                        out += pack_zigzag(x - prev_x)
                        out += pack_zigzag(y - prev_y)
                    prev_x, prev_y = x, y
            elif t == DRAW_COMMANDS['HORIZONTAL_LINE']:
                x1, x2, y = clamp_uint16(cmd['x1']), clamp_uint16(cmd['x2']), clamp_uint16(cmd['y'])
                out += pack_zigzag(x1)
                out += pack_zigzag(x2 - x1)
                out += pack_zigzag(y)
            elif t == DRAW_COMMANDS['VERTICAL_LINE']:
                x, y1, y2 = clamp_uint16(cmd['x']), clamp_uint16(cmd['y1']), clamp_uint16(cmd['y2'])
                out += pack_zigzag(x)
                out += pack_zigzag(y1)
                out += pack_zigzag(y2 - y1)
    return out

def ensure_closed_ring(ring):
    if len(ring) < 3:
        return ring
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring

def geometry_to_draw_commands(geom, color, tags, zoom, tile_x, tile_y, simplify_tolerance=None, hex_color=None):
    commands = []
    def process_geom(g):
        local_cmds = []
        if g.is_empty:
            return local_cmds
        if g.geom_type == "Polygon":
            exterior = remove_duplicate_points(list(g.exterior.coords))
            exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
            exterior_pixels = ensure_closed_ring(exterior_pixels)
            if len(set(exterior_pixels)) >= 3:
                cmd = {'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color}
                if hex_color:
                    cmd['color_hex'] = hex_color
                local_cmds.append(cmd)
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                exterior = remove_duplicate_points(list(poly.exterior.coords))
                exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                if len(set(exterior_pixels)) >= 3:
                    cmd = {'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color}
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    local_cmds.append(cmd)
        elif g.geom_type == "LineString":
            coords = remove_duplicate_points(list(g.coords))
            if len(coords) < 2:
                return local_cmds
            pixel_coords = remove_duplicate_points(coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y))
            if len(pixel_coords) < 2:
                return local_cmds
            is_closed = coords[0] == coords[-1]
            if is_closed and is_area(tags):
                if len(set(pixel_coords)) >= 3:
                    cmd = {'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': pixel_coords, 'color': color}
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    local_cmds.append(cmd)
            else:
                if len(pixel_coords) == 2:
                    x1, y1 = pixel_coords[0]
                    x2, y2 = pixel_coords[1]
                    cmd = {'color': color}
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    if y1 == y2:
                        cmd.update({'type': DRAW_COMMANDS['HORIZONTAL_LINE'], 'x1': x1, 'x2': x2, 'y': y1})
                    elif x1 == x2:
                        cmd.update({'type': DRAW_COMMANDS['VERTICAL_LINE'], 'x': x1, 'y1': y1, 'y2': y2})
                    else:
                        cmd.update({'type': DRAW_COMMANDS['LINE'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                    local_cmds.append(cmd)
                else:
                    cmd = {'type': DRAW_COMMANDS['POLYLINE'], 'points': pixel_coords, 'color': color}
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    local_cmds.append(cmd)
        elif g.geom_type == "MultiLineString":
            for linestring in g.geoms:
                local_cmds.extend(process_geom(linestring))
        elif g.geom_type == "GeometryCollection":
            for subgeom in g.geoms:
                local_cmds.extend(process_geom(subgeom))
        return local_cmds
    if hasattr(geom, "is_valid") and not geom.is_empty:
        commands.extend(process_geom(geom))
    return commands

def get_layer_fields_from_pbf(pbf_file, layer):
    try:
        result = subprocess.run(
            ["ogrinfo", "-so", "-geom=NO", pbf_file, layer],
            capture_output=True, encoding="utf-8"
        )
        if result.returncode != 0:
            return set()
        lines = result.stdout.splitlines()
        fields = set()
        in_field_section = False
        for line in lines:
            line = line.strip()
            # Check if we're in the fields section
            if line.startswith("Layer name:") or line.startswith("Geometry:") or line.startswith("Feature Count:"):
                continue
            if ":" in line and not line.startswith("  "):
                # This is a field definition like "highway: String (0.0)"
                field_name = line.split(":")[0].strip()
                if field_name and not field_name.startswith("OGR"):
                    fields.add(field_name)
        return fields
    except Exception as e:
        print(f"Error getting fields for layer {layer}: {e}")
        return set()

def build_ogr2ogr_where_clause_from_config(config, allowed_fields):
    conds = []
    for k in config.keys():
        if "=" in k:
            key, val = k.split("=", 1)
            if key in allowed_fields:
                conds.append(f'("{key}" = \'{val}\')')
        else:
            if k in allowed_fields:
                conds.append(f'("{k}" IS NOT NULL)')
    where_clause = " OR ".join(conds)
    return where_clause

def get_config_fields(config):
    fields = set()
    for k in config.keys():
        if "=" in k:
            key, _ = k.split("=", 1)
            fields.add(key)
        else:
            fields.add(k)
    return fields

def count_features(tmp_files, config_fields):
    total = 0
    for tmp_file in tmp_files:
        with open(tmp_file, "r", encoding="utf-8") as f:
            for _ in ijson.items(f, "features.item"):
                total += 1
    return total

def extract_geojson_from_pbf(pbf_file, geojson_file, config):
    print("Extracting PBF with ogr2ogr using SQL filter and minimal fields based on style...")
    if os.path.exists(geojson_file):
        os.remove(geojson_file)
    
    # All OSM fields that might be needed - each layer can have any of these
    ALL_OSM_FIELDS = {
        "highway", "waterway", "railway", "natural", "place", "boundary", "power", 
        "man_made", "barrier", "aeroway", "route", "building", "landuse", "leisure", 
        "amenity", "shop", "tourism", "historic", "office", "craft", "emergency", 
        "military", "sport", "water"
    }
    
    # For simplicity, allow all fields in all layers
    LAYER_FIELDS = {
        "points": ALL_OSM_FIELDS,
        "lines": ALL_OSM_FIELDS,
        "multilinestrings": ALL_OSM_FIELDS,
        "multipolygons": ALL_OSM_FIELDS,
        "other_relations": ALL_OSM_FIELDS
    }
    
    layers = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]
    config_fields = get_config_fields(config)
    
    print(f"Config requires these fields: {', '.join(sorted(config_fields))}")

    print("Extracting layers ...")
    extract_args = [
        (layer, i, layers, geojson_file, pbf_file, config, LAYER_FIELDS, config_fields)
        for i, layer in enumerate(layers)
    ]
    tmp_files = []
    all_logs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_layer_to_tmpfile, arg) for arg in extract_args]
        for future in as_completed(futures):
            tmp_file, logs = future.result()
            all_logs.extend(logs)
            if tmp_file:
                tmp_files.append(tmp_file)
    for log in all_logs:
        print(log)
    if not tmp_files:
        print("Could not extract any layer from OSM PBF.")
        print("This might mean:")
        print("1. The PBF file doesn't contain the fields required by your config")
        print("2. The config.json might be incorrectly formatted")
        print("3. Try running: ogrinfo -so your.pbf lines | head -30")
        sys.exit(1)

    print("Counting total features ...")
    total_features_to_merge = count_features(tmp_files, config_fields)
    print(f"Total features to merge: {total_features_to_merge}")

    print("Merging and writing temporary GeoJSONs ...")
    total_features = 0
    with open(geojson_file, "w", encoding="utf-8") as out, tqdm(total=total_features_to_merge, desc="Merging features") as pbar:
        out.write('{"type": "FeatureCollection", "features": [\n')
        first = True
        counter = 0
        for tmp_file in tmp_files:
            for feat in read_tmp_geojson_features_stream(tmp_file, config_fields):
                if not first:
                    out.write(',\n')
                json.dump(feat, out, default=decimal_default)
                first = False
                total_features += 1
                counter += 1
                pbar.update(1)
                if counter % 5000 == 0:
                    gc.collect()
        out.write('\n]}\n')

    for tmp_file in tmp_files:
        os.remove(tmp_file)
    print(f"Total merged features: {total_features}")
    print(f"GeoJSON file generated successfully at {geojson_file}")
    gc.collect()

def precompute_global_color_palette(config):
    global GLOBAL_COLOR_PALETTE, GLOBAL_INDEX_TO_RGB332
    
    print("Analyzing colors from features.json to build dynamic palette...")
    
    # Extract all unique colors from JSON
    unique_colors = set()
    for feature_key, feature_config in config.items():
        if isinstance(feature_config, dict) and 'color' in feature_config:
            hex_color = feature_config['color']
            if hex_color and isinstance(hex_color, str) and hex_color.startswith("#"):
                unique_colors.add(hex_color)
    
    # Sort colors for consistency
    sorted_colors = sorted(list(unique_colors))
    
    # Create palette maps
    GLOBAL_COLOR_PALETTE = {}
    GLOBAL_INDEX_TO_RGB332 = {}
    
    for index, hex_color in enumerate(sorted_colors):
        rgb332_value = hex_to_rgb332_direct(hex_color)
        GLOBAL_COLOR_PALETTE[hex_color] = index
        GLOBAL_INDEX_TO_RGB332[index] = rgb332_value
    
    print(f"Dynamic color palette created:")
    print(f"  - Total unique colors: {len(unique_colors)}")
    print(f"  - Palette indices: 0-{len(unique_colors)-1}")
    print(f"  - Memory saving potential: {len(unique_colors)} colors -> compact indices")
    
    # Show some examples
    examples = list(sorted_colors)[:5]
    for hex_color in examples:
        index = GLOBAL_COLOR_PALETTE[hex_color]
        rgb332 = GLOBAL_INDEX_TO_RGB332[index]
        print(f"    {hex_color} -> index {index} -> RGB332 {rgb332}")
    
    if len(unique_colors) > 5:
        print(f"    ... and {len(unique_colors) - 5} more colors")
    
    return len(unique_colors)

def detect_feature_types(config):
    global DETECTED_FEATURE_TYPES
    
    print("\nAnalyzing features.json for feature-specific optimizations...")
    
    feature_types = set()
    
    for feature_key, feature_config in config.items():
        if isinstance(feature_config, dict):
            # Detect highways/roads
            if any(keyword in feature_key.lower() for keyword in ['highway', 'road', 'street', 'primary', 'secondary', 'trunk', 'motorway']):
                feature_types.add('highway')
            
            # Detect buildings
            if any(keyword in feature_key.lower() for keyword in ['building', 'residential', 'commercial', 'industrial']):
                feature_types.add('building')
            
            # Detect waterways
            if any(keyword in feature_key.lower() for keyword in ['waterway', 'river', 'stream', 'canal']):
                feature_types.add('waterway')
            
            # Detect natural features
            if any(keyword in feature_key.lower() for keyword in ['natural', 'landuse', 'forest', 'park']):
                feature_types.add('natural')
    
    DETECTED_FEATURE_TYPES = feature_types
    
    print(f"Feature types detected for optimization:")
    for ftype in sorted(feature_types):
        print(f"  ✓ {ftype}")
    
    if not feature_types:
        print("  → No specific feature types detected, using general optimizations")
    
    return feature_types

def write_palette_bin(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    palette_path = os.path.join(output_dir, "palette.bin")
    print(f"Writing palette to {palette_path} ({len(GLOBAL_INDEX_TO_RGB332)} colors)...")
    with open(palette_path, "wb") as fp:
        for idx in range(len(GLOBAL_INDEX_TO_RGB332)):
            rgb332_val = GLOBAL_INDEX_TO_RGB332[idx]
            fp.write(bytes([rgb332_val]))
    print("Palette written OK.")

def process_features_to_database(geojson_file, config, db_path, zoom_levels):
    """Process all features once and store them in database by zoom level and tile"""
    print("Processing features and storing in database...")
    
    db = FeatureDatabase(db_path)
    config_fields = get_config_fields(config)
    
    # Get total feature count for progress tracking
    total_features = 0
    with fiona.open(geojson_file) as src:
        total_features = len(src)
    
    print(f"Processing {total_features} features for {len(zoom_levels)} zoom levels...")
    
    feature_count = 0
    batch_features = []
    
    with fiona.open(geojson_file) as src:
        for feat in tqdm(src, total=total_features, desc="Processing features"):
            tags = {k: v for k, v in feat['properties'].items() if k in config_fields}
            style, stylekey = get_style_for_tags(tags, config)
            if not style:
                continue
            
            geom = shape(feat['geometry'])
            if not geom.is_valid or geom.is_empty:
                continue
            
            zoom_filter = style.get("zoom", 6)
            priority = style.get("priority", 5)
            hex_color = style.get("color", "#FFFFFF")
            color = hex_to_rgb332_direct(hex_color)
            
            # Process this feature for all relevant zoom levels
            for zoom in zoom_levels:
                if zoom < zoom_filter:
                    continue
                
                simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
                feature_geom = geom
                
                if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
                    try:
                        feature_geom = feature_geom.simplify(simplify_tolerance, preserve_topology=True)
                    except Exception:
                        pass
                
                if feature_geom.is_empty or not feature_geom.is_valid:
                    continue
                
                # Calculate tile bounds for this zoom level
                minx, miny, maxx, maxy = feature_geom.bounds
                xtile_min, ytile_min = deg2num(miny, minx, zoom)
                xtile_max, ytile_max = deg2num(maxy, maxx, zoom)
                
                # Store feature for each tile it intersects
                for xt in range(min(xtile_min, xtile_max), max(xtile_min, xtile_max) + 1):
                    for yt in range(min(ytile_min, ytile_max), max(ytile_min, ytile_max) + 1):
                        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tile_latlon_bounds(xt, yt, zoom)
                        tile_bbox = box(t_lon_min, t_lat_min, t_lon_max, t_lat_max)
                        
                        try:
                            clipped_geom = feature_geom.intersection(tile_bbox)
                        except Exception:
                            continue
                        
                        if not clipped_geom.is_empty:
                            feature_data = {
                                "geom": clipped_geom,
                                "color": color,
                                "color_hex": hex_color,
                                "tags": tags,
                                "priority": priority
                            }
                            
                            batch_features.append((zoom, xt, yt, feature_data, priority))
                            
                            # Batch insert to avoid memory buildup
                            if len(batch_features) >= DB_BATCH_SIZE:
                                for zoom_batch, x_batch, y_batch, feat_data, prio in batch_features:
                                    db.insert_feature(zoom_batch, x_batch, y_batch, feat_data, prio)
                                db.commit()
                                batch_features.clear()
                                gc.collect()
            
            feature_count += 1
            if feature_count % 1000 == 0:
                gc.collect()
    
    # Insert remaining features
    if batch_features:
        for zoom_batch, x_batch, y_batch, feat_data, prio in batch_features:
            db.insert_feature(zoom_batch, x_batch, y_batch, feat_data, prio)
        db.commit()
    
    print(f"Processed {feature_count} features and stored in database")
    
    # Print statistics for each zoom level
    for zoom in zoom_levels:
        count = db.count_features_for_zoom(zoom)
        print(f"Zoom {zoom}: {count} features stored")
    
    db.close()
    gc.collect()

def write_tile_batch(batch, output_dir, zoom, max_file_size, simplify_tolerance):
    """Write a batch of tiles and return statistics"""
    tile_sizes = []
    
    # Use fewer workers for tile batches to control memory
    batch_workers = min(max_workers, 2)
    with ProcessPoolExecutor(max_workers=batch_workers) as executor:
        futures = [executor.submit(tile_worker, job) for job in batch]
        for future in as_completed(futures):
            tile_size, _ = future.result()
            tile_sizes.append(tile_size)
    
    return tile_sizes

def generate_tiles_from_database(db_path, output_dir, zoom, max_file_size=65536):
    """Generate tiles for a specific zoom level from the database"""
    print(f"\n=== Processing zoom level {zoom} from database ===")
    
    db = FeatureDatabase(db_path)
    
    # Get all tiles for this zoom level
    tiles = db.get_tiles_for_zoom(zoom)
    if not tiles:
        print(f"No tiles found for zoom {zoom}")
        db.close()
        return
    
    print(f"Found {len(tiles)} tiles for zoom {zoom}")
    
    # Process tiles in batches
    all_tile_sizes = []
    TILE_BATCH_SIZE = 5000
    total_batches = (len(tiles) + TILE_BATCH_SIZE - 1) // TILE_BATCH_SIZE
    
    with tqdm(total=len(tiles), desc=f"Writing tiles (zoom {zoom})") as pbar:
        for batch_idx in range(0, len(tiles), TILE_BATCH_SIZE):
            batch_tiles = tiles[batch_idx:batch_idx + TILE_BATCH_SIZE]
            batch_jobs = []
            
            for (tile_x, tile_y) in batch_tiles:
                features = db.get_features_for_tile(zoom, tile_x, tile_y)
                if features:
                    batch_jobs.append((tile_x, tile_y, features, zoom, output_dir, max_file_size, None))
            
            if batch_jobs:
                # Write this batch
                batch_sizes = write_tile_batch(batch_jobs, output_dir, zoom, max_file_size, None)
                all_tile_sizes.extend(batch_sizes)
            
            pbar.update(len(batch_tiles))
            gc.collect()
    
    avg_tile_size = sum(all_tile_sizes) / len(all_tile_sizes) if all_tile_sizes else 0
    print(f"Zoom {zoom}: {len(all_tile_sizes)} tiles, average size = {avg_tile_size:.2f} bytes")
    
    db.close()
    gc.collect()

def tile_worker(args):
    start_time = time.time()
    x, y, feats, zoom, output_dir, max_file_size, simplify_tolerance = args

    ordered_feats = sorted(feats, key=lambda f: f.get("priority", 5))

    all_commands = []
    for feat in ordered_feats:
        # Get hex color if available
        hex_color = feat.get("color_hex")
        cmds = geometry_to_draw_commands(
            feat["geom"], feat["color"], feat["tags"], zoom, x, y, 
            simplify_tolerance=simplify_tolerance, hex_color=hex_color
        )
        all_commands.extend(cmds)

    # Apply palette optimization to all commands
    optimized_commands, bytes_saved = insert_palette_commands(all_commands)

    tile_dir = os.path.join(output_dir, str(zoom), str(x))
    os.makedirs(tile_dir, exist_ok=True)
    filename = os.path.join(tile_dir, f"{y}.bin")

    # Pack all optimized commands at once
    buffer = pack_draw_commands(optimized_commands)
    
    # Check if buffer exceeds max file size
    if len(buffer) > max_file_size:
        # If too large, fall back to individual command packing with size limit
        buffer = bytearray()
        num_cmds_written = 0
        header = pack_varint(0)
        buffer += header

        for cmd in optimized_commands:
            cmd_bytes = pack_draw_commands([cmd])[len(pack_varint(1)):]
            tmp_num_cmds = num_cmds_written + 1
            tmp_header = pack_varint(tmp_num_cmds)
            tmp_buffer_size = len(tmp_header) + len(buffer[len(header):]) + len(cmd_bytes)
            if tmp_buffer_size > max_file_size:
                break
            buffer = tmp_header + buffer[len(header):] + cmd_bytes
            header = tmp_header
            num_cmds_written += 1

        if num_cmds_written == 0:
            buffer = pack_varint(0)

    with open(filename, "wb") as f:
        f.write(buffer)

    tile_size = len(buffer)
    
    del all_commands, optimized_commands, ordered_feats, buffer, feats
    gc.collect()
    
    elapsed = time.time() - start_time
    return tile_size, elapsed

def main():
    parser = argparse.ArgumentParser(description="OSM vector tile generator (memory-optimized with complete optimization pipeline)")
    parser.add_argument("pbf_file", help="Path to .pbf file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("config_file", help="JSON config with features/colors")
    parser.add_argument("--zoom", help="Zoom level or range (e.g. 12 or 6-17)", default="6-17")
    parser.add_argument("--max-file-size", help="Maximum file size in KB", type=int, default=128)
    parser.add_argument("--db-path", help="Path for temporary database", default="features.db")
    args = parser.parse_args()
    
    if "-" in args.zoom:
        start, end = map(int, args.zoom.split("-"))
        zoom_levels = list(range(start, end + 1))
    else:
        zoom_levels = [int(args.zoom)]
    max_file_size = args.max_file_size * 1024

    with open(args.config_file, "r") as f:
        config = json.load(f)

    # Pre-compute dynamic palette based on JSON
    palette_size = precompute_global_color_palette(config)
    print(f"\nDynamic palette ready with {palette_size} colors from your features.json")

    write_palette_bin(args.output_dir)
    
    # Detect feature types for specific optimizations
    feature_types = detect_feature_types(config)
    print(f"Ready for feature-specific optimizations: {', '.join(sorted(feature_types)) if feature_types else 'general optimization only'}")

    geojson_tmp = os.path.abspath("tmp_extract.geojson")
    extract_geojson_from_pbf(args.pbf_file, geojson_tmp, config)

    # Process features once and store in database
    process_features_to_database(geojson_tmp, config, args.db_path, zoom_levels)

    # Generate tiles for each zoom level from database
    for zoom in zoom_levels:
        generate_tiles_from_database(args.db_path, args.output_dir, zoom, max_file_size)
        gc.collect()
    
    print("\nProcess completed successfully.")

    # Cleanup
    if os.path.exists(geojson_tmp):
        os.remove(geojson_tmp)
    if os.path.exists(args.db_path):
        os.remove(args.db_path)
    
    del config
    gc.collect()

if __name__ == "__main__":
    main()
