#!/usr/bin/env python3
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, box, shape
)
import math
import struct
import json
import os
from collections import Counter
from tqdm import tqdm
import argparse
import subprocess
import sys
import gc
import time
import sqlite3
import pickle

import ijson
import logging
import signal
import atexit

from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration constants for the tile generator"""
    
    # Tile and coordinate system
    TILE_SIZE = 256
    UINT16_TILE_SIZE = 65536
    
    # Processing limits
    MAX_WORKERS = min(os.cpu_count() or 4, 4)
    DB_BATCH_SIZE = 10000
    TILE_BATCH_SIZE = 5000
    
    # File size limits
    MAX_FILE_SIZE_KB_LIMIT = 10240  # 10MB
    DEFAULT_MAX_FILE_SIZE_KB = 128
    
    # Zoom level limits
    MIN_ZOOM_LEVEL = 0
    MAX_ZOOM_LEVEL = 20
    DEFAULT_ZOOM_RANGE = "6-17"
    
    # File extensions
    PBF_EXTENSION = ".pbf"
    JSON_EXTENSION = ".json"
    GEOJSON_EXTENSION = ".geojson"
    
    # Default values
    DEFAULT_DB_PATH = "features.db"
    DEFAULT_COLOR = "#FFFFFF"
    DEFAULT_PRIORITY = 5
    DEFAULT_ZOOM_FILTER = 6
    
    # Simplify tolerances by zoom level
    SIMPLIFY_TOLERANCES = {
        "low_zoom": 0.05,    # For zoom <= 10
        "high_zoom": None    # For zoom > 10
    }
    
    # OSM field categories
    OSM_FIELD_CATEGORIES = {
        "transport": {"highway", "railway", "aeroway", "route"},
        "water": {"waterway", "natural", "water"},
        "landuse": {"landuse", "leisure", "amenity", "tourism"},
        "infrastructure": {"power", "man_made", "barrier"},
        "boundaries": {"boundary", "place"},
        "buildings": {"building"},
        "services": {"shop", "craft", "office", "emergency", "military"},
        "recreation": {"sport", "historic"}
    }
    
    # All OSM fields that might be needed
    ALL_OSM_FIELDS = {
        "highway", "waterway", "railway", "natural", "place", "boundary", "power", 
        "man_made", "barrier", "aeroway", "route", "building", "landuse", "leisure", 
        "amenity", "shop", "tourism", "historic", "office", "craft", "emergency", 
        "military", "sport", "water"
    }
    
    # Layer field mappings
    LAYER_FIELDS = {
        "points": ALL_OSM_FIELDS,
        "lines": ALL_OSM_FIELDS,
        "multilinestrings": ALL_OSM_FIELDS,
        "multipolygons": ALL_OSM_FIELDS,
        "other_relations": ALL_OSM_FIELDS
    }
    
    # Processing layers
    PROCESSING_LAYERS = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]
    
    # Drawing command codes
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

# Global variables to track files for cleanup
_db_file_to_cleanup = None
_temp_files_to_cleanup = set()

def cleanup_database():
    """Clean up database file if it exists"""
    global _db_file_to_cleanup
    if _db_file_to_cleanup and os.path.exists(_db_file_to_cleanup):
        try:
            os.remove(_db_file_to_cleanup)
            logger.debug(f"Cleaned up database file: {_db_file_to_cleanup}")
        except Exception as e:
            logger.debug(f"Could not remove database file {_db_file_to_cleanup}: {e}")

def cleanup_temp_files():
    """Clean up temporary files if they exist"""
    global _temp_files_to_cleanup
    for temp_file in list(_temp_files_to_cleanup):
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.debug(f"Could not remove temporary file {temp_file}: {e}")
    _temp_files_to_cleanup.clear()

def cleanup_all():
    """Clean up all tracked files"""
    cleanup_database()
    cleanup_temp_files()

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure cleanup"""
    logger.info("Script interrupted - cleaning up...")
    cleanup_all()
    sys.exit(1)

# Register cleanup functions
atexit.register(cleanup_all)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler) # Termination signal

# Use Config class constants instead of global variables

# Global variables for dynamic palette
GLOBAL_COLOR_PALETTE = {}  # hex_color -> index
GLOBAL_INDEX_TO_RGB332 = {}  # index -> rgb332_value


# Database configuration moved to Config class

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


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def deg2pixel(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = ((lon_deg + 180.0) / 360.0 * n * Config.TILE_SIZE)
    y = ((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * Config.TILE_SIZE)
    return x, y

def coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y):
    pixel_coords = []
    for lon, lat in coords:
        px_global, py_global = deg2pixel(lat, lon, zoom)
        x = (px_global - tile_x * Config.TILE_SIZE) * (Config.UINT16_TILE_SIZE - 1) / (Config.TILE_SIZE - 1)
        y = ((py_global - tile_y * Config.TILE_SIZE) * (Config.UINT16_TILE_SIZE - 1) / (Config.TILE_SIZE - 1))
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(Config.UINT16_TILE_SIZE - 1, x))
        y = max(0, min(Config.UINT16_TILE_SIZE - 1, y))
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
    except (ValueError, TypeError) as e:
        # Invalid color format - use default white color
        return 0xFF

def hex_to_color_index(hex_color):
    global GLOBAL_COLOR_PALETTE
    return GLOBAL_COLOR_PALETTE.get(hex_color, None)

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
        return Config.SIMPLIFY_TOLERANCES["low_zoom"]
    else:
        return Config.SIMPLIFY_TOLERANCES["high_zoom"]

def clamp_uint16(x):
    return max(0, min(Config.UINT16_TILE_SIZE - 1, int(x)))

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
                    'type': Config.DRAW_COMMANDS['SET_COLOR_INDEX'], 
                    'color_index': color_index
                })
                current_color_index = color_index
                color_commands_inserted += 1
        else:
            # Fallback to direct SET_COLOR
            if cmd_color_rgb332 != current_color_index:  # Reset current_color_index
                result.append({
                    'type': Config.DRAW_COMMANDS['SET_COLOR'], 
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

def pack_draw_commands(commands):
    out = bytearray()
    out += pack_varint(len(commands))
    
    for cmd in commands:
        t = cmd['type']
        out += pack_varint(t)
        
        if t == Config.DRAW_COMMANDS['SET_COLOR']:
            # Original SET_COLOR command (direct RGB332)
            color = cmd['color'] & 0xFF
            out += struct.pack("B", color)
        elif t == Config.DRAW_COMMANDS['SET_COLOR_INDEX']:
            # SET_COLOR_INDEX command (palette index)
            color_index = cmd['color_index'] & 0xFF
            out += pack_varint(color_index)
        elif t == Config.DRAW_COMMANDS['RECTANGLE']:
            # Optimized RECTANGLE command
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == Config.DRAW_COMMANDS['STRAIGHT_LINE']:
            # Optimized STRAIGHT_LINE command
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == Config.DRAW_COMMANDS['GRID_PATTERN']:
            # Step 6: Grid pattern command
            x, y, width, spacing, count = map(clamp_uint16, [cmd['x'], cmd['y'], cmd['width'], cmd['spacing'], cmd['count']])
            direction = cmd.get('direction', 'horizontal')
            out += pack_zigzag(x)
            out += pack_zigzag(y)
            out += pack_zigzag(width)
            out += pack_zigzag(spacing)
            out += pack_varint(count)
            out += struct.pack("B", 1 if direction == 'horizontal' else 0)
        elif t == Config.DRAW_COMMANDS['CIRCLE']:
            # Step 6: Circle command
            center_x, center_y, radius = map(clamp_uint16, [cmd['center_x'], cmd['center_y'], cmd['radius']])
            out += pack_zigzag(center_x)
            out += pack_zigzag(center_y)
            out += pack_zigzag(radius)
        elif t == Config.DRAW_COMMANDS['PREDICTED_LINE']:
            # Step 6: Predicted line command (only end point needed)
            end_x, end_y = map(clamp_uint16, [cmd['end_x'], cmd['end_y']])
            out += pack_zigzag(end_x)
            out += pack_zigzag(end_y)
        else:
            # Original geometric commands
            if t == Config.DRAW_COMMANDS['LINE']:
                x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
                out += pack_zigzag(x1)
                out += pack_zigzag(y1)
                out += pack_zigzag(x2 - x1)
                out += pack_zigzag(y2 - y1)
            elif t == Config.DRAW_COMMANDS['POLYLINE'] or t == Config.DRAW_COMMANDS['STROKE_POLYGON']:
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
            elif t == Config.DRAW_COMMANDS['HORIZONTAL_LINE']:
                x1, x2, y = clamp_uint16(cmd['x1']), clamp_uint16(cmd['x2']), clamp_uint16(cmd['y'])
                out += pack_zigzag(x1)
                out += pack_zigzag(x2 - x1)
                out += pack_zigzag(y)
            elif t == Config.DRAW_COMMANDS['VERTICAL_LINE']:
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
                cmd = {'type': Config.DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color}
                if hex_color:
                    cmd['color_hex'] = hex_color
                local_cmds.append(cmd)
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                exterior = remove_duplicate_points(list(poly.exterior.coords))
                exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                if len(set(exterior_pixels)) >= 3:
                    cmd = {'type': Config.DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color}
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
                    cmd = {'type': Config.DRAW_COMMANDS['STROKE_POLYGON'], 'points': pixel_coords, 'color': color}
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
                        cmd.update({'type': Config.DRAW_COMMANDS['HORIZONTAL_LINE'], 'x1': x1, 'x2': x2, 'y': y1})
                    elif x1 == x2:
                        cmd.update({'type': Config.DRAW_COMMANDS['VERTICAL_LINE'], 'x': x1, 'y1': y1, 'y2': y2})
                    else:
                        cmd.update({'type': Config.DRAW_COMMANDS['LINE'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
                    local_cmds.append(cmd)
                else:
                    cmd = {'type': Config.DRAW_COMMANDS['POLYLINE'], 'points': pixel_coords, 'color': color}
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
        for line in lines:
            line = line.strip()
            if ":" in line:
                field = line.split(":", 1)[0].strip().replace('"', '')
                if field:
                    fields.add(field)
        return fields
    except Exception:
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

def validate_pbf_file(pbf_file):
    """Validate PBF file exists and is readable"""
    if not os.path.exists(pbf_file):
        raise FileNotFoundError(f"PBF file not found: {pbf_file}")
    
    if not os.path.isfile(pbf_file):
        raise ValueError(f"Path is not a file: {pbf_file}")
    
    if not pbf_file.lower().endswith(Config.PBF_EXTENSION):
        raise ValueError(f"File must have {Config.PBF_EXTENSION} extension: {pbf_file}")
    
    if not os.access(pbf_file, os.R_OK):
        raise PermissionError(f"Cannot read PBF file: {pbf_file}")
    
    # Check file size (should not be empty)
    file_size = os.path.getsize(pbf_file)
    if file_size == 0:
        raise ValueError(f"PBF file is empty: {pbf_file}")
    
    logger.debug(f"PBF file validation passed: {pbf_file} ({file_size} bytes)")

def validate_config_file(config_file):
    """Validate configuration file exists and is valid JSON"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    if not os.path.isfile(config_file):
        raise ValueError(f"Path is not a file: {config_file}")
    
    if not config_file.lower().endswith(Config.JSON_EXTENSION):
        raise ValueError(f"Config file must have {Config.JSON_EXTENSION} extension: {config_file}")
    
    if not os.access(config_file, os.R_OK):
        raise PermissionError(f"Cannot read config file: {config_file}")
    
    # Try to parse JSON to validate format
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a JSON object")
        
        if not config:
            raise ValueError("Config file cannot be empty")
        
        logger.debug(f"Config file validation passed: {config_file}")
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file}: {e}")

def validate_output_dir(output_dir):
    """Validate output directory exists or can be created"""
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError(f"Output path exists but is not a directory: {output_dir}")
        
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to output directory: {output_dir}")
    else:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            raise PermissionError(f"Cannot create output directory {output_dir}: {e}")
    
    logger.debug(f"Output directory validation passed: {output_dir}")

def validate_zoom_range(zoom_str):
    """Validate zoom range string and return zoom levels"""
    if not zoom_str:
        raise ValueError("Zoom range cannot be empty")
    
    if "-" in zoom_str:
        try:
            start, end = map(int, zoom_str.split("-"))
            if start < Config.MIN_ZOOM_LEVEL or end < Config.MIN_ZOOM_LEVEL:
                raise ValueError(f"Zoom levels cannot be negative (minimum: {Config.MIN_ZOOM_LEVEL})")
            if start > end:
                raise ValueError("Start zoom level cannot be greater than end zoom level")
            if end > Config.MAX_ZOOM_LEVEL:
                raise ValueError(f"Zoom level cannot exceed {Config.MAX_ZOOM_LEVEL}")
            zoom_levels = list(range(start, end + 1))
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid zoom range format: {zoom_str}. Use format like '6-17' or '12'")
            raise
    else:
        try:
            zoom = int(zoom_str)
            if zoom < Config.MIN_ZOOM_LEVEL:
                raise ValueError(f"Zoom level cannot be negative (minimum: {Config.MIN_ZOOM_LEVEL})")
            if zoom > Config.MAX_ZOOM_LEVEL:
                raise ValueError(f"Zoom level cannot exceed {Config.MAX_ZOOM_LEVEL}")
            zoom_levels = [zoom]
        except ValueError:
            raise ValueError(f"Invalid zoom level: {zoom_str}. Must be a number or range like '6-17'")
    
    logger.debug(f"Zoom range validation passed: {zoom_levels}")
    return zoom_levels

def validate_max_file_size(max_file_size_kb):
    """Validate maximum file size parameter"""
    if not isinstance(max_file_size_kb, int):
        raise ValueError("Max file size must be an integer")
    
    if max_file_size_kb <= 0:
        raise ValueError("Max file size must be positive")
    
    if max_file_size_kb > Config.MAX_FILE_SIZE_KB_LIMIT:
        raise ValueError(f"Max file size cannot exceed {Config.MAX_FILE_SIZE_KB_LIMIT} KB ({Config.MAX_FILE_SIZE_KB_LIMIT // 1024}MB)")
    
    logger.debug(f"Max file size validation passed: {max_file_size_kb} KB")

def validate_db_path(db_path):
    """Validate database path"""
    if not db_path:
        raise ValueError("Database path cannot be empty")
    
    # Check if parent directory exists and is writable
    parent_dir = os.path.dirname(os.path.abspath(db_path))
    if parent_dir and not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Cannot create database directory {parent_dir}: {e}")
    
    if parent_dir and not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Cannot write to database directory: {parent_dir}")
    
    logger.debug(f"Database path validation passed: {db_path}")

def extract_layer_to_temp_file(pbf_file, layer, where_clause, select_fields):
    """Extract layer data from PBF to temporary GeoJSON file"""
    tmp_filename = f"tmp_{layer}_{os.getpid()}{Config.GEOJSON_EXTENSION}"
    
    # Register temporary file for cleanup
    global _temp_files_to_cleanup
    _temp_files_to_cleanup.add(tmp_filename)
    
    cmd = [
        "ogr2ogr",
        "-f", "GeoJSON",
        "-nlt", "PROMOTE_TO_MULTI",
        "-where", where_clause,
        "-select", select_fields,
        tmp_filename,
        pbf_file,
        layer
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # ogr2ogr command failed - layer might be empty or invalid
        return None
    
    # Check if file was created and has content
    if not os.path.exists(tmp_filename) or os.path.getsize(tmp_filename) == 0:
        return None
    
    return tmp_filename

def process_feature_for_zoom_levels(feat, config, config_fields, zoom_levels, db, batch_features):
    """Process a single feature for all relevant zoom levels"""
    feat['properties'] = {k: v for k, v in feat['properties'].items() if k in config_fields}
    
    tags = feat['properties']
    style, stylekey = get_style_for_tags(tags, config)
    if not style:
        return 0
    
    try:
        geom = shape(feat['geometry'])
    except (ValueError, TypeError) as e:
        # Invalid geometry data - skip this feature
        return 0
        
    if not geom.is_valid or geom.is_empty:
        return 0
    
    zoom_filter = style.get("zoom", 6)
    priority = style.get("priority", 5)
    hex_color = style.get("color", "#FFFFFF")
    color = hex_to_rgb332_direct(hex_color)
    
    features_added = 0
    
    # Process this feature for all relevant zoom levels
    for zoom in zoom_levels:
        if zoom < zoom_filter:
            continue
        
        simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
        feature_geom = geom
        
        if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
            try:
                feature_geom = feature_geom.simplify(simplify_tolerance, preserve_topology=True)
            except (ValueError, TypeError) as e:
                # Geometry simplification failed - use original geometry
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
                except (ValueError, TypeError) as e:
                    # Geometry intersection failed - skip this tile
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
                    features_added += 1
                    
                    # Batch insert to avoid memory buildup
                    if len(batch_features) >= Config.DB_BATCH_SIZE:
                        for zoom_batch, x_batch, y_batch, feat_data, prio in batch_features:
                            db.insert_feature(zoom_batch, x_batch, y_batch, feat_data, prio)
                        db.commit()
                        batch_features.clear()
                        gc.collect()
    
    return features_added

def process_layer_directly_to_database(pbf_file, layer, config, db, zoom_levels, config_fields, LAYER_FIELDS):
    """Process a single layer directly from PBF to database"""
    possible = LAYER_FIELDS[layer]
    available = get_layer_fields_from_pbf(pbf_file, layer)
    allowed = possible & available & config_fields
    where_clause = build_ogr2ogr_where_clause_from_config(config, allowed)
    
    if not where_clause:
        return 0
    
    select_fields = ",".join(sorted(allowed))
    
    # Extract layer to temporary file
    tmp_filename = extract_layer_to_temp_file(pbf_file, layer, where_clause, select_fields)
    if not tmp_filename:
        return 0
    
    try:
        # Process features from temporary file
        features_processed = 0
        batch_features = []
        
        with open(tmp_filename, "r", encoding="utf-8") as f:
            for feat in ijson.items(f, "features.item"):
                features_added = process_feature_for_zoom_levels(
                    feat, config, config_fields, zoom_levels, db, batch_features
                )
                features_processed += 1
                
                if features_processed % 1000 == 0:
                    gc.collect()
        
        # Insert remaining features for this layer
        if batch_features:
            for zoom_batch, x_batch, y_batch, feat_data, prio in batch_features:
                db.insert_feature(zoom_batch, x_batch, y_batch, feat_data, prio)
            db.commit()
        
        return features_processed
        
    finally:
        # Always clean up the temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        # Remove from tracking set since we cleaned it up manually
        _temp_files_to_cleanup.discard(tmp_filename)

def process_pbf_directly_to_database(pbf_file, config, db_path, zoom_levels):
    """Process PBF directly to database with minimal temporary files"""
    global _db_file_to_cleanup
    _db_file_to_cleanup = db_path  # Register for cleanup
    
    logger.info("Processing PBF directly to database (minimal temporary files)")
    
    db = FeatureDatabase(db_path)
    config_fields = get_config_fields(config)
    
    # All OSM fields that might be needed
    ALL_OSM_FIELDS = {
        "highway", "waterway", "railway", "natural", "place", "boundary", "power", 
        "man_made", "barrier", "aeroway", "route", "building", "landuse", "leisure", 
        "amenity", "shop", "tourism", "historic", "office", "craft", "emergency", 
        "military", "sport", "water"
    }
    
    LAYER_FIELDS = {
        "points": ALL_OSM_FIELDS,
        "lines": ALL_OSM_FIELDS,
        "multilinestrings": ALL_OSM_FIELDS,
        "multipolygons": ALL_OSM_FIELDS,
        "other_relations": ALL_OSM_FIELDS
    }
    
    layers = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]
    logger.info(f"Config requires these fields: {', '.join(sorted(config_fields))}")
    
    total_features_processed = 0
    
    # Process each layer separately and immediately clean up
    for i, layer in enumerate(layers):
        logger.info(f"[{i+1}/{len(layers)}] Processing layer: {layer} directly to database")
        
        layer_features = process_layer_directly_to_database(
            pbf_file, layer, config, db, zoom_levels, config_fields, LAYER_FIELDS
        )
        
        total_features_processed += layer_features
        logger.info(f"Layer {layer}: {layer_features} features processed")
        
        # Force garbage collection after each layer
        gc.collect()
    
    logger.info(f"Total processed: {total_features_processed} features directly from PBF")
    
    # Print statistics for each zoom level
    for zoom in zoom_levels:
        count = db.count_features_for_zoom(zoom)
        logger.info(f"Zoom {zoom}: {count} features stored")
    
    db.close()
    gc.collect()
    
    return total_features_processed

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

def write_tile_batch(batch, output_dir, zoom, max_file_size, simplify_tolerance):
    """Write a batch of tiles and return statistics"""
    tile_sizes = []
    
    # Use fewer workers for tile batches to control memory
    batch_workers = min(Config.MAX_WORKERS, 2)
    with ProcessPoolExecutor(max_workers=batch_workers) as executor:
        futures = [executor.submit(tile_worker, job) for job in batch]
        for future in as_completed(futures):
            tile_size, _ = future.result()
            tile_sizes.append(tile_size)
    
    return tile_sizes

def generate_tiles_from_database(db_path, output_dir, zoom, max_file_size=65536):
    """Generate tiles for a specific zoom level from the database"""
    logger.info(f"Processing zoom level {zoom} from database")
    
    db = FeatureDatabase(db_path)
    
    # Get all tiles for this zoom level
    tiles = db.get_tiles_for_zoom(zoom)
    if not tiles:
        logger.warning(f"No tiles found for zoom {zoom}")
        db.close()
        return
    
    logger.info(f"Found {len(tiles)} tiles for zoom {zoom}")
    
    # Process tiles in batches
    all_tile_sizes = []
    total_batches = (len(tiles) + Config.TILE_BATCH_SIZE - 1) // Config.TILE_BATCH_SIZE
    
    with tqdm(total=len(tiles), desc=f"Writing tiles (zoom {zoom})") as pbar:
        for batch_idx in range(0, len(tiles), Config.TILE_BATCH_SIZE):
            batch_tiles = tiles[batch_idx:batch_idx + Config.TILE_BATCH_SIZE]
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

def precompute_global_color_palette(config):
    global GLOBAL_COLOR_PALETTE, GLOBAL_INDEX_TO_RGB332
    
    logger.info("Analyzing colors from features.json to build dynamic palette")
    
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
    
    logger.info(f"Dynamic color palette created:")
    logger.info(f"  - Total unique colors: {len(unique_colors)}")
    logger.info(f"  - Palette indices: 0-{len(unique_colors)-1}")
    logger.info(f"  - Memory saving potential: {len(unique_colors)} colors -> compact indices")
    
    # Show some examples
    examples = list(sorted_colors)[:5]
    for hex_color in examples:
        index = GLOBAL_COLOR_PALETTE[hex_color]
        rgb332 = GLOBAL_INDEX_TO_RGB332[index]
        # Color mapping details (debug level)
        pass
    
    if len(unique_colors) > 5:
        # Additional colors info (debug level)
        pass
    
    return len(unique_colors)


def write_palette_bin(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    palette_path = os.path.join(output_dir, "palette.bin")
    logger.info(f"Writing palette to {palette_path} ({len(GLOBAL_INDEX_TO_RGB332)} colors)")
    with open(palette_path, "wb") as fp:
        for idx in range(len(GLOBAL_INDEX_TO_RGB332)):
            rgb332_val = GLOBAL_INDEX_TO_RGB332[idx]
            fp.write(bytes([rgb332_val]))
    logger.info("Palette written successfully")

def main():
    parser = argparse.ArgumentParser(description="OSM vector tile generator (direct processing with database)")
    parser.add_argument("pbf_file", help="Path to .pbf file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("config_file", help="JSON config with features/colors")
    parser.add_argument("--zoom", help="Zoom level or range (e.g. 12 or 6-17)", default=Config.DEFAULT_ZOOM_RANGE)
    parser.add_argument("--max-file-size", help="Maximum file size in KB", type=int, default=Config.DEFAULT_MAX_FILE_SIZE_KB)
    parser.add_argument("--db-path", help="Path for temporary database", default=Config.DEFAULT_DB_PATH)
    args = parser.parse_args()
    
    try:
        # Validate all input parameters
        logger.info("Validating input parameters...")
        validate_pbf_file(args.pbf_file)
        validate_output_dir(args.output_dir)
        validate_db_path(args.db_path)
        validate_max_file_size(args.max_file_size)
        zoom_levels = validate_zoom_range(args.zoom)
        config = validate_config_file(args.config_file)
        
        logger.info("All input parameters validated successfully")
        
    except (FileNotFoundError, ValueError, PermissionError) as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    
    # Clean up any existing database file from previous runs
    if os.path.exists(args.db_path):
        try:
            os.remove(args.db_path)
            logger.debug(f"Removed existing database file: {args.db_path}")
        except Exception as e:
            logger.debug(f"Could not remove existing database file {args.db_path}: {e}")
    
    # Clean up any existing temporary files from previous runs
    import glob
    existing_temp_files = glob.glob(f"tmp_*{Config.GEOJSON_EXTENSION}")
    for temp_file in existing_temp_files:
        try:
            os.remove(temp_file)
            logger.debug(f"Removed existing temporary file: {temp_file}")
        except Exception as e:
            logger.debug(f"Could not remove existing temporary file {temp_file}: {e}")
    
    max_file_size = args.max_file_size * 1024

    # Pre-compute dynamic palette based on JSON
    palette_size = precompute_global_color_palette(config)
    logger.info(f"Dynamic palette ready with {palette_size} colors from your features.json")

    write_palette_bin(args.output_dir)

    # Process features directly from PBF to database (no intermediate GeoJSON)
    process_pbf_directly_to_database(args.pbf_file, config, args.db_path, zoom_levels)

    # Generate tiles for each zoom level from database
    for zoom in zoom_levels:
        generate_tiles_from_database(args.db_path, args.output_dir, zoom, max_file_size)
        gc.collect()
    
    logger.info("Process completed successfully")

    # File cleanup is handled automatically by atexit and signal handlers
    cleanup_all()
    
    del config
    gc.collect()

if __name__ == "__main__":
    main()
