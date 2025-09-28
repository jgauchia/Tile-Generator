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

max_workers = os.cpu_count()

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


class LRUCache:
   
    def __init__(self, capacity=1000, memory_limit_mb=100):
        """
            Args:
                capacity: Maximum number of elements
                memory_limit_mb: Memory limit in MB
        """
        self.capacity = capacity
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.lock = RLock()
        self._memory_usage = 0
        self._total_hits = 0
        self._total_misses = 0
        self._session_hits = 0   
        self._session_misses = 0  
        
    def _estimate_size(self, value):
        """Estimates the memory usage of a cached value"""
        if isinstance(value, list):
            return len(value) * 200  
        elif isinstance(value, dict):
            return sys.getsizeof(value)
        else:
            return 100 
    
    def get(self, key):
        """Retrieves a value from the cache and marks it as recently used"""
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                self._total_hits += 1
                self._session_hits += 1
                return value
            else:
                self._total_misses += 1
                self._session_misses += 1
                return None
    
    def put(self, key, value):
        """Stores a value in the cache"""
        with self.lock:
            value_size = self._estimate_size(value)
            
            if key in self.cache:
                old_value = self.cache.pop(key)
                self._memory_usage -= self._estimate_size(old_value)
            
            while (len(self.cache) >= self.capacity or 
                   self._memory_usage + value_size > self.memory_limit_bytes):
                if not self.cache:
                    break
                oldest_key, oldest_value = self.cache.popitem(last=False)
                self._memory_usage -= self._estimate_size(oldest_value)
            
            self.cache[key] = value
            self._memory_usage += value_size
    
    def clear(self):
        """Completely clears the cache"""
        with self.lock:
            self.cache.clear()
            self._memory_usage = 0
            self._session_hits = 0
            self._session_misses = 0
    
    def stats(self):
        """Returns cache statistics"""
        with self.lock:
            total_requests = self._total_hits + self._total_misses
            if total_requests > 0:
                total_hit_rate = (self._total_hits / total_requests) * 100
            else:
                total_hit_rate = 0
            
            session_requests = self._session_hits + self._session_misses
            if session_requests > 0:
                session_hit_rate = (self._session_hits / session_requests) * 100
            else:
                session_hit_rate = 0
            
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'memory_mb': round(self._memory_usage / (1024 * 1024), 2),
                'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024),
                'total_hit_rate_percent': round(total_hit_rate, 1),
                'session_hit_rate_percent': round(session_hit_rate, 1),
                'total_hits': self._total_hits,
                'total_misses': self._total_misses,
                'session_hits': self._session_hits,
                'session_misses': self._session_misses
            }

    def reset_session_stats(self):
        """Resets only session statistics"""
        with self.lock:
            self._session_hits = 0
            self._session_misses = 0

class ObjectPool:
    """
    Object pool for efficient reuse.
    Replaces frequent creation/destruction of small objects.
    """
   
    def __init__(self, factory, reset_func, initial_size=100, max_size=1000):
        """
            Args:
                factory: Function that creates new objects
                reset_func: Function that resets objects for reuse
                initial_size: Initial size of the pool
                max_size: Maximum size of the pool
        """
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.lock = RLock()

        self.pool = []
        for _ in range(min(initial_size, max_size)):
            try:
                obj = factory()
                self.pool.append(obj)
            except Exception:
                break
        
        self._objects_requested = 0     
        self._objects_from_pool = 0     
        self._objects_created_new = 0    
        self._objects_returned = 0       

    def get(self):
        """Gets an object from the pool or creates a new one"""
        with self.lock:
            self._objects_requested += 1
            
            if self.pool:
                obj = self.pool.pop()
                self._objects_from_pool += 1
                return obj
            else:
                self._objects_created_new += 1
                return self.factory()
    
    def release(self, obj):
        """Returns an object to the pool after resetting it"""
        if obj is None:
            return
        
        try:
            self.reset_func(obj)
            
            with self.lock:
                self._objects_returned += 1
                if len(self.pool) < self.max_size:
                    self.pool.append(obj)
        except Exception:
            pass
    
    def clear(self):
        """Clears the pool"""
        with self.lock:
            self.pool.clear()
    
    def stats(self):
        """Returns pool statistics"""
        with self.lock:
            if self._objects_requested > 0:
                reuse_rate = (self._objects_from_pool / self._objects_requested) * 100
            else:
                reuse_rate = 0
            
            if self._objects_requested > 0:
                return_rate = (self._objects_returned / self._objects_requested) * 100
            else:
                return_rate = 0
            
            return {
                'available': len(self.pool),
                'max_size': self.max_size,
                'total_requested': self._objects_requested,
                'from_pool': self._objects_from_pool,
                'created_new': self._objects_created_new,
                'returned': self._objects_returned,
                'reuse_rate_percent': round(reuse_rate, 1),
                'return_rate_percent': round(return_rate, 1)
            }

    def reset_stats(self):
        """Resets statistics without resetting the pool"""
        with self.lock:
            self._objects_requested = 0
            self._objects_from_pool = 0
            self._objects_created_new = 0
            self._objects_returned = 0

class MemoryManager:
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        available_memory = psutil.virtual_memory().available
        safe_memory_limit = available_memory * 0.6 
        
        geometry_cache_mb = min(200, int(safe_memory_limit / (1024 * 1024 * 20)))
        geometry_cache_size = min(2000, geometry_cache_mb * 5)  # ~5 items por MB
        
        print(f"Initializing MemoryManager:")
        print(f"  - Available memory: {available_memory / (1024*1024*1024):.1f} GB")
        print(f"  - Safe limit: {safe_memory_limit / (1024*1024*1024):.1f} GB")
        print(f"  - Geometry cache: {geometry_cache_mb} MB, {geometry_cache_size} items")
        
        self.geometry_cache = LRUCache(
            capacity=geometry_cache_size,
            memory_limit_mb=geometry_cache_mb
        )
        
        self.color_cache = LRUCache(
            capacity=config.get('color_cache_size', 500),
            memory_limit_mb=config.get('color_cache_mb', 20)
        )
        
        self.command_pool = ObjectPool(
            factory=dict,
            reset_func=lambda d: d.clear(),
            initial_size=config.get('command_pool_initial', 500),
            max_size=config.get('command_pool_max', 2000)
        )
        
        self.point_pool = ObjectPool(
            factory=list,
            reset_func=lambda l: l.clear(),
            initial_size=config.get('point_pool_initial', 300),
            max_size=config.get('point_pool_max', 1000)
        )

        self.coord_pool = ObjectPool(
            factory=lambda: [],
            reset_func=lambda l: l.clear(),
            initial_size=100,
            max_size=500
        )
        
        self._start_memory = psutil.Process().memory_info().rss
    
    def get_geometry_from_cache(self, key):
        """Wrapper for accessing the geometry cache"""
        return self.geometry_cache.get(key)
    
    def put_geometry_in_cache(self, key, value):
        """Wrapper for storing in the geometry cache"""
        self.geometry_cache.put(key, value)
    
    def get_command(self):
        """Gets a reusable dictionary for commands"""
        return self.command_pool.get()
    
    def release_command(self, cmd):
        """Returns a command to the pool"""
        self.command_pool.release(cmd)
    
    def get_point_list(self):
        """Gets a reusable list for points"""
        return self.point_pool.get()
    
    def release_point_list(self, points):
        """Returns a list of points to the pool"""
        self.point_pool.release(points)
    
    def get_coord_list(self):
        """Gets a reusable list for coordinates"""
        return self.coord_pool.get()
    
    def release_coord_list(self, coords):
        """Returns a list of coordinates to the pool"""
        self.coord_pool.release(coords)
    
    def cleanup(self):
        """Clears all caches and pools"""
        self.geometry_cache.clear()
        self.color_cache.clear()
        self.command_pool.clear()
        self.point_pool.clear()
        self.coord_pool.clear()
        gc.collect()
    
    def get_memory_stats(self):
        """Returns full memory statistics"""
        current_memory = psutil.Process().memory_info().rss
        memory_growth = current_memory - self._start_memory
        
        return {
            'current_memory_mb': round(current_memory / (1024*1024), 2),
            'memory_growth_mb': round(memory_growth / (1024*1024), 2),
            'geometry_cache': self.geometry_cache.stats(),
            'color_cache': self.color_cache.stats(),
            'command_pool': self.command_pool.stats(),
            'point_pool': self.point_pool.stats(),
            'coord_pool': self.coord_pool.stats()
        }
    
    def print_stats(self):
        """Prints memory statistics in a readable format"""
        stats = self.get_memory_stats()

        print("\n=== MEMORY STATISTICS ===")
        print(f"Current memory: {stats['current_memory_mb']} MB")
        print(f"Memory growth: {stats['memory_growth_mb']} MB")

        print(f"\nGeometry Cache:")
        gc = stats['geometry_cache']
        print(f"  - Items: {gc['size']}/{gc['capacity']}")
        print(f"  - Memory: {gc['memory_mb']}/{gc['memory_limit_mb']} MB")
        print(f"  - Total hit rate: {gc['total_hit_rate_percent']}% ({gc['total_hits']} hits, {gc['total_misses']} misses)")
        print(f"  - Session hit rate: {gc['session_hit_rate_percent']}% (current zoom level)")

        print(f"\nCommand Pool:")
        cp = stats['command_pool']
        print(f"  - Available: {cp['available']}/{cp['max_size']}")
        print(f"  - Requests: {cp['total_requested']} (from pool: {cp['from_pool']}, new: {cp['created_new']})")
        print(f"  - Reuse rate: {cp['reuse_rate_percent']}%")
        print(f"  - Return rate: {cp['return_rate_percent']}%")

        print(f"\nPoint Pool:")
        pp = stats['point_pool']
        print(f"  - Available: {pp['available']}/{pp['max_size']}")
        print(f"  - Requests: {pp['total_requested']} (from pool: {pp['from_pool']}, new: {pp['created_new']})")
        print(f"  - Reuse rate: {pp['reuse_rate_percent']}%")
        print(f"  - Return rate: {pp['return_rate_percent']}%")

memory_manager = None

def initialize_memory_manager(config=None):
    """Initializes the global memory manager"""
    global memory_manager
    memory_manager = MemoryManager(config)
    return memory_manager

def get_memory_manager():
    """Gets the instance of the memory manager"""
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager()
    return memory_manager

def optimized_geometry_hash(geom):
    """
    Fast hash for geometries based on bounds and type.
    Replaces the expensive hash(geom.wkt).
    """
    try:
        bounds = geom.bounds  
        geom_type = geom.geom_type
        
        if hasattr(geom, 'coords'):
            coord_count = len(list(geom.coords))
        else:
            coord_count = 0
            
        return hash((bounds, geom_type, coord_count))
    except Exception:
        if hasattr(geom, 'wkt'):
            return hash(geom.wkt)
        else:
            return hash(str(geom))

def geometry_to_draw_commands_optimized(geom, color, tags, zoom, tile_x, tile_y, 
                                       simplify_tolerance=None, hex_color=None):

    mm = get_memory_manager()
    
    geom_key = optimized_geometry_hash(geom)
    
    cached_commands = mm.get_geometry_from_cache(geom_key)
    if cached_commands is not None:
        result_commands = []
        for cached_cmd in cached_commands:
            cmd = mm.get_command()
            cmd.update(cached_cmd)  
            cmd['color'] = color
            if hex_color:
                cmd['color_hex'] = hex_color
            result_commands.append(cmd)
        return result_commands
    
    result_commands = []
    
    def process_geom(g):
        if g.is_empty:
            return
            
        if g.geom_type == "Polygon":
            coord_list = mm.get_coord_list()
            try:
                exterior = list(g.exterior.coords)
                exterior = remove_duplicate_points(exterior)
                exterior_pixels = coords_to_pixel_coords_uint16_optimized(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                
                if len(set(exterior_pixels)) >= 3:
                    cmd = mm.get_command()
                    cmd.update({
                        'type': DRAW_COMMANDS['STROKE_POLYGON'], 
                        'points': exterior_pixels, 
                        'color': color
                    })
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    result_commands.append(cmd)
            finally:
                mm.release_coord_list(coord_list)
                
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                process_geom(poly)
                
        elif g.geom_type == "LineString":
            coord_list = mm.get_coord_list()
            try:
                coords = remove_duplicate_points(list(g.coords))
                if len(coords) < 2:
                    return
                    
                pixel_coords = remove_duplicate_points(
                    coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y)
                )
                if len(pixel_coords) < 2:
                    return
                    
                is_closed = coords[0] == coords[-1]
                if is_closed and is_area(tags):
                    if len(set(pixel_coords)) >= 3:
                        cmd = mm.get_command()
                        cmd.update({
                            'type': DRAW_COMMANDS['STROKE_POLYGON'], 
                            'points': pixel_coords, 
                            'color': color
                        })
                        if hex_color:
                            cmd['color_hex'] = hex_color
                        result_commands.append(cmd)
                else:
                    if len(pixel_coords) == 2:
                        x1, y1 = pixel_coords[0]
                        x2, y2 = pixel_coords[1]
                        cmd = mm.get_command()
                        cmd['color'] = color
                        if hex_color:
                            cmd['color_hex'] = hex_color
                        
                        if y1 == y2:
                            cmd.update({
                                'type': DRAW_COMMANDS['HORIZONTAL_LINE'], 
                                'x1': x1, 'x2': x2, 'y': y1
                            })
                        elif x1 == x2:
                            cmd.update({
                                'type': DRAW_COMMANDS['VERTICAL_LINE'], 
                                'x': x1, 'y1': y1, 'y2': y2
                            })
                        else:
                            cmd.update({
                                'type': DRAW_COMMANDS['LINE'], 
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                            })
                        result_commands.append(cmd)
                    else:
                        cmd = mm.get_command()
                        cmd.update({
                            'type': DRAW_COMMANDS['POLYLINE'], 
                            'points': pixel_coords, 
                            'color': color
                        })
                        if hex_color:
                            cmd['color_hex'] = hex_color
                        result_commands.append(cmd)
            finally:
                mm.release_coord_list(coord_list)
                
        elif g.geom_type == "MultiLineString":
            for linestring in g.geoms:
                process_geom(linestring)
        elif g.geom_type == "GeometryCollection":
            for subgeom in g.geoms:
                process_geom(subgeom)
    
    if hasattr(geom, "is_valid") and not geom.is_empty:
        process_geom(geom)
    
    if result_commands:
        cache_commands = []
        for cmd in result_commands:
            cache_cmd = {k: v for k, v in cmd.items() if k not in ['color', 'color_hex']}
            cache_commands.append(cache_cmd)
        mm.put_geometry_in_cache(geom_key, cache_commands)
    
    return result_commands

def coords_to_pixel_coords_uint16_optimized(coords, zoom, tile_x, tile_y):

    mm = get_memory_manager()
    point_list = mm.get_point_list()
    
    try:
        for lon, lat in coords:
            px_global, py_global = deg2pixel(lat, lon, zoom)
            x = (px_global - tile_x * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1)
            y = ((py_global - tile_y * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1))
            x = int(round(x))
            y = int(round(y))
            x = max(0, min(UINT16_TILE_SIZE - 1, x))
            y = max(0, min(UINT16_TILE_SIZE - 1, y))
            point_list.append((x, y))
        
        result = list(point_list)
        return result
    finally:
        mm.release_point_list(point_list)





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
    global POINT_POOL
    POINT_POOL.clear()      
    for lon, lat in coords:
        px_global, py_global = deg2pixel(lat, lon, zoom)
        x = (px_global - tile_x * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1)
        y = ((py_global - tile_y * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1))
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(UINT16_TILE_SIZE - 1, x))
        y = max(0, min(UINT16_TILE_SIZE - 1, y))
        POINT_POOL.append((x, y))
    
    return list(POINT_POOL)  

def remove_duplicate_points(points):
    if len(points) <= 1:
        return points
    result = [points[0]]
    for pt in points[1:]:
        if pt != result[-1]:
            result.append(pt)
    return result

def optimize_coordinate_precision(points, zoom_level):
    if zoom_level <= 10:
        quantization = 4
        return [(x//quantization*quantization, y//quantization*quantization) 
                for x, y in points]
    elif zoom_level <= 12:
        quantization = 2
        return [(x//quantization*quantization, y//quantization*quantization) 
                for x, y in points]
    else:
        return points

def eliminate_micro_movements(points, threshold=2):
    if len(points) < 3:
        return points
    
    result = [points[0]]
    for point in points[1:]:
        distance = math.sqrt((point[0] - result[-1][0])**2 + 
                           (point[1] - result[-1][1])**2)
        if distance >= threshold:
            result.append(point)
    
    if len(result) > 1 and result[-1] != points[-1]:
        result.append(points[-1])
    
    return result

def validate_and_clean_commands(commands):
    clean_commands = []
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['LINE']:
            # Remove zero-length lines
            if cmd['x1'] == cmd['x2'] and cmd['y1'] == cmd['y2']:
                continue
        elif cmd['type'] == DRAW_COMMANDS['POLYLINE']:
            # Remove polylines with insufficient points
            points = cmd.get('points', [])
            if len(points) < 2:
                continue
        elif cmd['type'] == DRAW_COMMANDS['STROKE_POLYGON']:
            # Remove degenerate polygons
            points = cmd.get('points', [])
            unique_points = []
            for p in points:
                if p not in unique_points:
                    unique_points.append(p)
            if len(unique_points) < 3:
                continue
        elif cmd['type'] == DRAW_COMMANDS['HORIZONTAL_LINE']:
            # Remove zero-length horizontal lines
            if cmd['x1'] == cmd['x2']:
                continue
        elif cmd['type'] == DRAW_COMMANDS['VERTICAL_LINE']:
            # Remove zero-length vertical lines
            if cmd['y1'] == cmd['y2']:
                continue
        elif cmd['type'] == DRAW_COMMANDS['RECTANGLE']:
            # Remove zero-area rectangles
            if cmd['x1'] == cmd['x2'] or cmd['y1'] == cmd['y2']:
                continue
        elif cmd['type'] == DRAW_COMMANDS['STRAIGHT_LINE']:
            # Remove zero-length straight lines
            if cmd['x1'] == cmd['x2'] and cmd['y1'] == cmd['y2']:
                continue
        
        clean_commands.append(cmd)
    
    return clean_commands

def geometry_hash(geom_data):
    if isinstance(geom_data, list):
        # For point lists (polylines, polygons)
        return hash(tuple(tuple(p) for p in geom_data))
    elif isinstance(geom_data, tuple):
        # For simple geometries (lines, rectangles)
        return hash(geom_data)
    else:
        return hash(str(geom_data))

def detect_urban_grid_pattern(commands, tolerance=10):
    if not commands:
        return commands, 0
    
    # Analyze line commands for grid patterns
    horizontal_lines = []
    vertical_lines = []
    
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['LINE']:
            x1, y1, x2, y2 = cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']
            
            # Check if line is approximately horizontal
            if abs(y2 - y1) <= tolerance:
                horizontal_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2, cmd))
            
            # Check if line is approximately vertical  
            elif abs(x2 - x1) <= tolerance:
                vertical_lines.append((min(y1, y2), max(y1, y2), (x1 + x2) // 2, cmd))
    
    # Detect regular spacing in horizontal lines
    grid_patterns_detected = 0
    optimized_commands = []
    processed_commands = set()
    
    if len(horizontal_lines) >= 3:
        # Sort by Y coordinate
        h_lines_sorted = sorted(horizontal_lines, key=lambda x: x[2])
        
        # Look for regular spacing
        for i in range(len(h_lines_sorted) - 2):
            spacing1 = h_lines_sorted[i+1][2] - h_lines_sorted[i][2]
            spacing2 = h_lines_sorted[i+2][2] - h_lines_sorted[i+1][2]
            
            # If spacing is regular, create grid pattern
            if abs(spacing1 - spacing2) <= tolerance and spacing1 > tolerance:
                # Create GRID_PATTERN command
                min_x = min(h_lines_sorted[i][0], h_lines_sorted[i+1][0], h_lines_sorted[i+2][0])
                max_x = max(h_lines_sorted[i][1], h_lines_sorted[i+1][1], h_lines_sorted[i+2][1])
                start_y = h_lines_sorted[i][2]
                
                grid_cmd = {
                    'type': DRAW_COMMANDS['GRID_PATTERN'],
                    'x': min_x,
                    'y': start_y,
                    'width': max_x - min_x,
                    'spacing': spacing1,
                    'count': 3,
                    'direction': 'horizontal'
                }
                
                # Preserve color from original commands
                original_cmd = h_lines_sorted[i][3]
                if 'color' in original_cmd:
                    grid_cmd['color'] = original_cmd['color']
                if 'color_hex' in original_cmd:
                    grid_cmd['color_hex'] = original_cmd['color_hex']
                
                optimized_commands.append(grid_cmd)
                
                # Mark these commands as processed
                processed_commands.add(id(h_lines_sorted[i][3]))
                processed_commands.add(id(h_lines_sorted[i+1][3]))
                processed_commands.add(id(h_lines_sorted[i+2][3]))
                
                grid_patterns_detected += 1
                break  # Only process one grid pattern per call
    
    # Add unprocessed commands
    for cmd in commands:
        if id(cmd) not in processed_commands:
            optimized_commands.append(cmd)
    
    return optimized_commands, grid_patterns_detected

def detect_circular_features(commands, tolerance=5):
    if not commands:
        return commands, 0
    
    optimized_commands = []
    circles_detected = 0
    
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['STROKE_POLYGON']:
            points = cmd.get('points', [])
            
            if len(points) >= 8:  # Enough points to potentially be a circle
                is_circle, center, radius = is_approximately_circular(points, tolerance)
                
                if is_circle:
                    # Convert to CIRCLE command
                    circle_cmd = {
                        'type': DRAW_COMMANDS['CIRCLE'],
                        'center_x': center[0],
                        'center_y': center[1],
                        'radius': radius
                    }
                    
                    # Preserve color information
                    if 'color' in cmd:
                        circle_cmd['color'] = cmd['color']
                    if 'color_hex' in cmd:
                        circle_cmd['color_hex'] = cmd['color_hex']
                    if 'priority' in cmd:
                        circle_cmd['priority'] = cmd['priority']
                    
                    optimized_commands.append(circle_cmd)
                    circles_detected += 1
                else:
                    optimized_commands.append(cmd)
            else:
                optimized_commands.append(cmd)
        else:
            optimized_commands.append(cmd)
    
    return optimized_commands, circles_detected

def is_approximately_circular(points, tolerance):
    if len(points) < 6:
        return False, None, None
    
    # Calculate centroid
    sum_x = sum(p[0] for p in points[:-1])  # Exclude closing point
    sum_y = sum(p[1] for p in points[:-1])
    count = len(points) - 1
    center_x = sum_x / count
    center_y = sum_y / count
    center = (center_x, center_y)
    
    # Calculate distances from center to all points
    distances = []
    for point in points[:-1]:  # Exclude closing point
        distance = math.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
        distances.append(distance)
    
    # Check if all distances are approximately equal (circular)
    avg_radius = sum(distances) / len(distances)
    
    # Check variance
    variance = sum((d - avg_radius)**2 for d in distances) / len(distances)
    
    # If variance is low, it's approximately circular
    is_circular = variance <= tolerance * tolerance
    
    return is_circular, center, int(avg_radius) if is_circular else None

def apply_coordinate_prediction(commands):
    if not commands:
        return commands, 0
    
    optimized_commands = []
    predictions_applied = 0
    
    # Track movement patterns
    last_point = None
    movement_vector = None
    
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['LINE']:
            x1, y1, x2, y2 = cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']
            current_start = (x1, y1)
            current_end = (x2, y2)
            
            # If we can predict this line based on previous movement
            if (last_point and movement_vector and 
                abs(current_start[0] - (last_point[0] + movement_vector[0])) <= 3 and
                abs(current_start[1] - (last_point[1] + movement_vector[1])) <= 3):
                
                # Use predicted line command
                predicted_cmd = {
                    'type': DRAW_COMMANDS['PREDICTED_LINE'],
                    'end_x': x2,
                    'end_y': y2
                }
                
                # Preserve color information
                if 'color' in cmd:
                    predicted_cmd['color'] = cmd['color']
                if 'color_hex' in cmd:
                    predicted_cmd['color_hex'] = cmd['color_hex']
                if 'priority' in cmd:
                    predicted_cmd['priority'] = cmd['priority']
                
                optimized_commands.append(predicted_cmd)
                predictions_applied += 1
            else:
                optimized_commands.append(cmd)
            
            # Update tracking
            if last_point:
                movement_vector = (current_start[0] - last_point[0], 
                                 current_start[1] - last_point[1])
            last_point = current_end
            
        else:
            optimized_commands.append(cmd)
    
    return optimized_commands, predictions_applied

def detect_geometric_primitives(commands):
    if not commands:
        return commands, 0
    
    optimized_commands = []
    primitives_detected = 0
    
    for cmd in commands:
        # Rectangle detection is already handled in optimize_buildings()
        # Circle detection is handled in detect_circular_features()
        # This function handles other geometric primitives
        
        if cmd['type'] == DRAW_COMMANDS['POLYLINE']:
            points = cmd.get('points', [])
            
            # Detect if polyline is actually a simple rectangle outline
            if len(points) == 5 and points[0] == points[-1]:  # Closed shape
                is_rect, bbox = is_rectangle(points)
                if is_rect:
                    min_x, min_y, max_x, max_y = bbox
                    rect_cmd = {
                        'type': DRAW_COMMANDS['RECTANGLE'],
                        'x1': min_x,
                        'y1': min_y,
                        'x2': max_x,
                        'y2': max_y
                    }
                    
                    # Preserve color information
                    if 'color' in cmd:
                        rect_cmd['color'] = cmd['color']
                    if 'color_hex' in cmd:
                        rect_cmd['color_hex'] = cmd['color_hex']
                    if 'priority' in cmd:
                        rect_cmd['priority'] = cmd['priority']
                    
                    optimized_commands.append(rect_cmd)
                    primitives_detected += 1
                    continue
        
        optimized_commands.append(cmd)
    
    return optimized_commands, primitives_detected

def apply_tile_boundary_optimization(commands, tile_x, tile_y, zoom):
    # This is a complex optimization that would require coordination
    # between adjacent tiles. For now, we implement a simpler version
    # that optimizes geometries near tile edges.
    
    optimized_commands = []
    boundary_optimizations = 0
    
    # Define boundary regions (10% of tile size on each edge)
    boundary_threshold = UINT16_TILE_SIZE * 0.1
    
    for cmd in commands:
        # Check if geometry is near tile boundary
        is_near_boundary = False
        
        if cmd['type'] in [DRAW_COMMANDS['LINE'], DRAW_COMMANDS['STRAIGHT_LINE']]:
            x1, y1, x2, y2 = cmd.get('x1', 0), cmd.get('y1', 0), cmd.get('x2', 0), cmd.get('y2', 0)
            if (x1 <= boundary_threshold or x1 >= UINT16_TILE_SIZE - boundary_threshold or
                x2 <= boundary_threshold or x2 >= UINT16_TILE_SIZE - boundary_threshold or
                y1 <= boundary_threshold or y1 >= UINT16_TILE_SIZE - boundary_threshold or
                y2 <= boundary_threshold or y2 >= UINT16_TILE_SIZE - boundary_threshold):
                is_near_boundary = True
        
        if is_near_boundary:
            # Apply boundary-specific optimizations
            # For now, we just mark these for potential cross-tile optimization
            cmd['boundary_optimized'] = True
            boundary_optimizations += 1
        
        optimized_commands.append(cmd)
    
    return optimized_commands, boundary_optimizations

def apply_advanced_compression_techniques(commands, tile_x=0, tile_y=0, zoom=12):
    if not commands:
        return commands, {}
    
    optimization_stats = {
        'grid_patterns': 0,
        'circles': 0,
        'predictions': 0,
        'primitives': 0,
        'boundary_opts': 0,
        'total_optimizations': 0
    }
    
    # Apply each technique in sequence
    optimized_commands = commands
    
    # 1. Urban Grid Pattern Detection
    optimized_commands, grid_count = detect_urban_grid_pattern(optimized_commands)
    optimization_stats['grid_patterns'] = grid_count
    
    # 2. Circular Feature Detection
    optimized_commands, circle_count = detect_circular_features(optimized_commands)
    optimization_stats['circles'] = circle_count
    
    # 3. Coordinate Prediction
    optimized_commands, prediction_count = apply_coordinate_prediction(optimized_commands)
    optimization_stats['predictions'] = prediction_count
    
    # 4. Geometric Primitive Detection
    optimized_commands, primitive_count = detect_geometric_primitives(optimized_commands)
    optimization_stats['primitives'] = primitive_count
    
    # 5. Tile Boundary Optimization
    optimized_commands, boundary_count = apply_tile_boundary_optimization(optimized_commands, tile_x, tile_y, zoom)
    optimization_stats['boundary_opts'] = boundary_count
    
    # Calculate total optimizations
    optimization_stats['total_optimizations'] = (
        grid_count + circle_count + prediction_count + 
        primitive_count + boundary_count
    )
    
    return optimized_commands, optimization_stats

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

def is_rectangle(points, tolerance=5):
    if len(points) < 4:
        return False, None
    
    # A rectangle must have exactly 4 corners (plus closing point)
    if len(points) != 5 or points[0] != points[-1]:
        return False, None
    
    # Get the 4 unique corners
    corners = points[:4]
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distances.append(dist)
    
    # In a rectangle, opposite sides must be equal
    if (abs(distances[0] - distances[2]) <= tolerance and 
        abs(distances[1] - distances[3]) <= tolerance):
        
        # Calculate bounding box for the rectangle
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        return True, (min_x, min_y, max_x, max_y)
    
    return False, None

def is_straight_line(points, tolerance=3):
    if len(points) < 3:
        return True  # 2 points always form a straight line
    
    # Calculate the straight line between first and last point
    start = points[0]
    end = points[-1]
    
    # Check that all intermediate points are close to this line
    for point in points[1:-1]:
        # Calculate distance from point to line
        distance = point_to_line_distance(point, start, end)
        if distance > tolerance:
            return False
    
    return True

def point_to_line_distance(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # If the line is a point
    if x1 == x2 and y1 == y2:
        return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    
    # Point-to-line distance formula
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return numerator / denominator if denominator > 0 else 0

def optimize_buildings(commands):
    if 'building' not in DETECTED_FEATURE_TYPES:
        return commands, 0
    
    optimized_commands = []
    rectangles_optimized = 0
    
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['STROKE_POLYGON']:
            points = cmd.get('points', [])
            is_rect, bbox = is_rectangle(points)
            
            if is_rect and bbox:
                # Convert to optimized RECTANGLE command
                min_x, min_y, max_x, max_y = bbox
                optimized_cmd = {
                    'type': DRAW_COMMANDS['RECTANGLE'],
                    'x1': min_x,
                    'y1': min_y,
                    'x2': max_x,
                    'y2': max_y,
                    'priority': cmd.get('priority', 5)
                }
                
                # Preserve color information
                if 'color' in cmd:
                    optimized_cmd['color'] = cmd['color']
                if 'color_hex' in cmd:
                    optimized_cmd['color_hex'] = cmd['color_hex']
                
                optimized_commands.append(optimized_cmd)
                rectangles_optimized += 1
            else:
                # Keep original command
                optimized_commands.append(cmd)
        else:
            optimized_commands.append(cmd)
    
    return optimized_commands, rectangles_optimized

def optimize_highways(commands):
    if 'highway' not in DETECTED_FEATURE_TYPES:
        return commands, 0
    
    optimized_commands = []
    straight_lines_optimized = 0
    
    for cmd in commands:
        if cmd['type'] == DRAW_COMMANDS['POLYLINE']:
            points = cmd.get('points', [])
            
            if len(points) >= 3 and is_straight_line(points):
                # Convert to optimized STRAIGHT_LINE command
                start = points[0]
                end = points[-1]
                optimized_cmd = {
                    'type': DRAW_COMMANDS['STRAIGHT_LINE'],
                    'x1': start[0],
                    'y1': start[1],
                    'x2': end[0],
                    'y2': end[1],
                    'priority': cmd.get('priority', 5)
                }
                
                # Preserve color information
                if 'color' in cmd:
                    optimized_cmd['color'] = cmd['color']
                if 'color_hex' in cmd:
                    optimized_cmd['color_hex'] = cmd['color_hex']
                
                optimized_commands.append(optimized_cmd)
                straight_lines_optimized += 1
            else:
                # Keep original command
                optimized_commands.append(cmd)
        else:
            optimized_commands.append(cmd)
    
    return optimized_commands, straight_lines_optimized

def apply_feature_specific_optimizations(commands):
    if not DETECTED_FEATURE_TYPES:
        return commands, 0
    
    # Apply optimizations in sequence
    optimized_commands = commands
    total_optimizations = 0
    
    if 'building' in DETECTED_FEATURE_TYPES:
        optimized_commands, building_opts = optimize_buildings(optimized_commands)
        total_optimizations += building_opts
    
    if 'highway' in DETECTED_FEATURE_TYPES:
        optimized_commands, highway_opts = optimize_highways(optimized_commands)
        total_optimizations += highway_opts
    
    return optimized_commands, total_optimizations

def apply_performance_optimizations(commands, zoom_level):
    if not commands:
        return commands
    
    # Optimize coordinate precision based on zoom level
    for cmd in commands:
        if 'points' in cmd:
            cmd['points'] = optimize_coordinate_precision(cmd['points'], zoom_level)
            cmd['points'] = eliminate_micro_movements(cmd['points'])
    
    # Clean and validate commands
    commands = validate_and_clean_commands(commands)
    
    return commands

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
    if zoom <= 8:
        return 0.0008 
    elif zoom <= 12:
        return 0.0003  
    else:
        return 0.0     

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

    mm = get_memory_manager()
    
    geom_key = optimized_geometry_hash(geom)
    
    cached_commands = mm.get_geometry_from_cache(geom_key)
    if cached_commands is not None:
        result_commands = []
        for cached_cmd in cached_commands:
            cmd = mm.get_command()
            cmd.update(cached_cmd)  
            cmd['color'] = color
            if hex_color:
                cmd['color_hex'] = hex_color
            result_commands.append(cmd)
        return result_commands
    
    result_commands = [] 
    
    def process_geom(g):
        if g.is_empty:
            return
            
        if g.geom_type == "Polygon":
            exterior = remove_duplicate_points(list(g.exterior.coords))
            exterior_pixels = coords_to_pixel_coords_uint16_optimized(exterior, zoom, tile_x, tile_y)
            exterior_pixels = ensure_closed_ring(exterior_pixels)
            if len(set(exterior_pixels)) >= 3:
                cmd = mm.get_command() 
                cmd.update({
                    'type': DRAW_COMMANDS['STROKE_POLYGON'], 
                    'points': exterior_pixels, 
                    'color': color
                })
                if hex_color:
                    cmd['color_hex'] = hex_color
                result_commands.append(cmd)  
                
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                exterior = remove_duplicate_points(list(poly.exterior.coords))
                exterior_pixels = coords_to_pixel_coords_uint16_optimized(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                if len(set(exterior_pixels)) >= 3:
                    cmd = mm.get_command() 
                    cmd.update({
                        'type': DRAW_COMMANDS['STROKE_POLYGON'], 
                        'points': exterior_pixels, 
                        'color': color
                    })
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    result_commands.append(cmd)
                    
        elif g.geom_type == "LineString":
            coords = remove_duplicate_points(list(g.coords))
            if len(coords) < 2:
                return
                
            pixel_coords = remove_duplicate_points(
                coords_to_pixel_coords_uint16_optimized(coords, zoom, tile_x, tile_y)
            )
            if len(pixel_coords) < 2:
                return
                
            is_closed = coords[0] == coords[-1]
            if is_closed and is_area(tags):
                if len(set(pixel_coords)) >= 3:
                    cmd = mm.get_command() 
                    cmd.update({
                        'type': DRAW_COMMANDS['STROKE_POLYGON'], 
                        'points': pixel_coords, 
                        'color': color
                    })
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    result_commands.append(cmd)
            else:
                if len(pixel_coords) == 2:
                    x1, y1 = pixel_coords[0]
                    x2, y2 = pixel_coords[1]
                    cmd = mm.get_command()  
                    cmd['color'] = color
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    
                    if y1 == y2:
                        cmd.update({
                            'type': DRAW_COMMANDS['HORIZONTAL_LINE'], 
                            'x1': x1, 'x2': x2, 'y': y1
                        })
                    elif x1 == x2:
                        cmd.update({
                            'type': DRAW_COMMANDS['VERTICAL_LINE'], 
                            'x': x1, 'y1': y1, 'y2': y2
                        })
                    else:
                        cmd.update({
                            'type': DRAW_COMMANDS['LINE'], 
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                        })
                    result_commands.append(cmd)
                else:
                    cmd = mm.get_command()  # Usar pool
                    cmd.update({
                        'type': DRAW_COMMANDS['POLYLINE'], 
                        'points': pixel_coords, 
                        'color': color
                    })
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    result_commands.append(cmd)
                    
                if 'natural' in tags and tags['natural'] == 'coastline':
                    print("COASTLINE CMD:", pixel_coords)
                    
        elif g.geom_type == "MultiLineString":
            for linestring in g.geoms:
                process_geom(linestring)
        elif g.geom_type == "GeometryCollection":
            for subgeom in g.geoms:
                process_geom(subgeom)
    
    if hasattr(geom, "is_valid") and not geom.is_empty:
        process_geom(geom)
    
    if result_commands:
        cache_commands = []
        for cmd in result_commands:
            cache_cmd = {k: v for k, v in cmd.items() if k not in ['color', 'color_hex']}
            cache_commands.append(cache_cmd)
        
        mm.put_geometry_in_cache(geom_key, cache_commands)
    
    return result_commands  

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

def count_features(tmp_files, config_fields):
    total = 0
    for tmp_file in tmp_files:
        with open(tmp_file, "r", encoding="utf-8") as f:
            for _ in ijson.items(f, "features.item"):
                total += 1
    return total

def extract_geojson_from_pbf(pbf_file, geojson_file, config, zoom_levels):
    print("Extracting PBF with ogr2ogr using SQL filter and minimal fields based on style...")
    if os.path.exists(geojson_file):
        os.remove(geojson_file)
    LAYER_FIELDS = {
        "points": {"highway", "place", "natural", "amenity", "railway"},
        "lines": {"highway", "waterway", "railway", "natural", "place"},
        "multilinestrings": {"highway", "waterway", "railway", "natural", "place"},
        "multipolygons": {"building", "landuse", "leisure", "natural", "place", "amenity"},
        "other_relations": {"place", "natural"}
    }
    layers = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]

    max_zoom = max(zoom_levels)
    filtered_config = {}
    for k, v in config.items():
        if isinstance(v, dict) and "zoom" in v and v["zoom"] <= max_zoom:
            filtered_config[k] = v
        elif isinstance(v, dict) and "zoom" not in v:
            filtered_config[k] = v

    config_fields = get_config_fields(filtered_config)

    print("Extracting layers ...")
    extract_args = [
        (layer, i, layers, geojson_file, pbf_file, filtered_config, LAYER_FIELDS, config_fields)
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
                if counter % 10000 == 0:
                    gc.collect()
        out.write('\n]}\n')

    for tmp_file in tmp_files:
        os.remove(tmp_file)
    print(f"Total merged features: {total_features}")
    print(f"GeoJSON file generated successfully at {geojson_file}")
    gc.collect()
    return total_features_to_merge

def streaming_assign_features_to_tiles_by_zoom(geojson_file, config, output_dir, zoom_levels, max_file_size=65536, total_features=None, summary_stats=None):
    import psutil
    process = psutil.Process(os.getpid())
    config_fields = get_config_fields(config)
    zoom_to_valid_tags = {}
    for zoom in zoom_levels:
        valid_tags = set()
        for k, v in config.items():
            if v.get("zoom", 0) <= zoom:
                valid_tags.add(k)
        zoom_to_valid_tags[zoom] = valid_tags

    total_features_count = total_features if total_features is not None else None

    for zoom in zoom_levels:
        print(f"\n========== Processing zoom level {zoom} ==========")
        print(f"[Zoom {zoom}] Reading relevant features from GeoJSON...")

        mm = get_memory_manager()
        mm.geometry_cache.reset_session_stats()

        valid_tags = zoom_to_valid_tags[zoom]
        tile_buffers = defaultdict(list)
        assigned_features = 0

        mem_start = process.memory_info().rss / 1024 / 1024
        t0 = time.time()

        with open(geojson_file, "r", encoding="utf-8") as f, tqdm(desc=f"[Zoom {zoom}] Reading & assignment", total=total_features_count) as pbar_read:
            for feat in ijson.items(f, "features.item"):
                tags = {k: v for k, v in feat['properties'].items() if k in config_fields}
                style, stylekey = get_style_for_tags(tags, config)
                if not style:
                    pbar_read.update(1)
                    continue
                if stylekey not in valid_tags:
                    pbar_read.update(1)
                    continue
                if not feat.get('geometry'):
                    pbar_read.update(1)
                    continue
                try:
                    geom = shape(feat['geometry'])
                except Exception:
                    pbar_read.update(1)
                    continue
                if not geom.is_valid or geom.is_empty:
                    pbar_read.update(1)
                    continue

                simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
                feature_geom = geom
                if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
                    try:
                        feature_geom = feature_geom.simplify(simplify_tolerance, preserve_topology=True)
                    except Exception:
                        pass
                if feature_geom.is_empty or not feature_geom.is_valid:
                    pbar_read.update(1)
                    continue

                priority = style.get("priority", 5)
                hex_color = style.get("color", "#FFFFFF")
                color = hex_to_rgb332_direct(hex_color)

                minx, miny, maxx, maxy = feature_geom.bounds
                n = 2 ** zoom
                xtile_min, ytile_min = deg2num(miny, minx, zoom)
                xtile_max, ytile_max = deg2num(maxy, maxx, zoom)

                for xt in range(min(xtile_min, xtile_max), max(xtile_min, xtile_max) + 1):
                    for yt in range(min(ytile_min, ytile_max), max(ytile_min, ytile_max) + 1):
                        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tile_latlon_bounds(xt, yt, zoom)
                        tile_bbox = box(t_lon_min, t_lat_min, t_lon_max, t_lat_max)
                        try:
                            clipped_geom = feature_geom.intersection(tile_bbox)
                        except Exception:
                            continue
                        if not clipped_geom.is_empty:
                            # CORREGIDO: Usar nombre de función original
                            cmds = geometry_to_draw_commands(
                                clipped_geom, color, tags, zoom, xt, yt,
                                simplify_tolerance=simplify_tolerance, hex_color=hex_color
                            )
                            for cmd in cmds:
                                cmd['priority'] = priority
                                tile_buffers[(xt, yt)].append(cmd)
                assigned_features += 1
                pbar_read.update(1)
        print(f"[Zoom {zoom}] Assigned features: {assigned_features}")

        print(f"[Zoom {zoom}] Optimizing with ALL optimization layers...")
        total_bytes_saved = 0
        tiles_optimized = 0
        total_tiles_processed = 0
        total_feature_optimizations = 0
        total_advanced_optimizations = {}

        tile_bin_sizes = []
        with tqdm(total=len(tile_buffers), desc=f"[Zoom {zoom}] Creating optimized tiles") as pbar_tiles:
            for (xt, yt), cmds in tile_buffers.items():
                tile_dir = os.path.join(output_dir, str(zoom), str(xt))
                os.makedirs(tile_dir, exist_ok=True)
                filename = os.path.join(tile_dir, f"{yt}.bin")

                cmds_sorted = sorted(cmds, key=lambda c: (c['priority'], c['color']))
                cmds_performance_optimized = apply_performance_optimizations(cmds_sorted, zoom)
                cmds_feature_optimized, feature_optimizations = apply_feature_specific_optimizations(cmds_performance_optimized)
                total_feature_optimizations += feature_optimizations
                cmds_advanced_optimized, advanced_stats = apply_advanced_compression_techniques(cmds_feature_optimized, xt, yt, zoom)
                for key, value in advanced_stats.items():
                    total_advanced_optimizations[key] = total_advanced_optimizations.get(key, 0) + value
                cmds_optimized, bytes_saved = insert_palette_commands(cmds_advanced_optimized)
                buffer = pack_draw_commands(cmds_optimized)
                with open(filename, "wb") as fbin:
                    fbin.write(buffer)
                size_bin = os.path.getsize(filename)
                tile_bin_sizes.append(size_bin)
                if bytes_saved > 0:
                    tiles_optimized += 1
                    total_bytes_saved += bytes_saved
                total_tiles_processed += 1
                pbar_tiles.update(1)

        dir_zoom = os.path.join(output_dir, str(zoom))
        total_dir_size = 0
        for dirpath, dirnames, filenames in os.walk(dir_zoom):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_dir_size += os.path.getsize(fp)

        if tile_bin_sizes:
            min_bin_size = min(tile_bin_sizes)
            max_bin_size = max(tile_bin_sizes)
            avg_bin_size = sum(tile_bin_sizes) / len(tile_bin_sizes)
        else:
            min_bin_size = 0
            max_bin_size = 0
            avg_bin_size = 0

        avg_savings_per_tile = total_bytes_saved / max(total_tiles_processed, 1)
        optimization_ratio = (tiles_optimized / max(total_tiles_processed, 1)) * 100

        print(f"[Zoom {zoom}] ALL Optimization Layers Applied:")
        print(f"  - Feature types detected: {', '.join(sorted(DETECTED_FEATURE_TYPES)) if DETECTED_FEATURE_TYPES else 'none'}")
        print(f"  - Feature optimizations applied: {total_feature_optimizations}")
        print(f"  - Performance optimizations: coordinate quantization, micro-movement elimination, validation")
        print(f"  - Advanced compression:")
        for key, value in total_advanced_optimizations.items():
            if value > 0:
                print(f"    • {key}: {value}")
        print(f"  - Tiles with palette optimization: {tiles_optimized}/{total_tiles_processed} ({optimization_ratio:.1f}%)")
        print(f"  - Total bytes saved (palette): {total_bytes_saved} bytes")
        print(f"  - Average savings per tile: {avg_savings_per_tile:.1f} bytes")
        
        cache_stats = mm.geometry_cache.stats()
        print(f"  - Geometry cache entries: {cache_stats['size']}/{cache_stats['capacity']}")
        
        print(f"  - Total directory size: {total_dir_size} bytes")
        print(f"  - Min bin size: {min_bin_size} bytes")
        print(f"  - Max bin size: {max_bin_size} bytes")
        print(f"  - Avg bin size: {avg_bin_size:.1f} bytes")

        tile_buffers.clear()
        gc.collect()
        t1 = time.time()
        mem_end = process.memory_info().rss / 1024 / 1024

        if summary_stats is not None:
            notes = f"{', '.join(sorted(DETECTED_FEATURE_TYPES)) if DETECTED_FEATURE_TYPES else 'general'}"
            notes += f" | ADV: {total_advanced_optimizations.get('total_optimizations', 0)}"
            notes += f" | PAL: {tiles_optimized}/{total_tiles_processed}"
            notes += f" | {total_bytes_saved}B | {len(GLOBAL_COLOR_PALETTE)} colors"

            summary_stats.append({
                "Zoom level": zoom,
                "Number of elements": assigned_features,
                "Memory usage (MB)": int(mem_end),
                "Total dir size (B)": total_dir_size,
                "Min bin size (B)": min_bin_size,
                "Max bin size (B)": max_bin_size,
                "Avg bin size (B)": round(avg_bin_size, 1),
                "Processing time (s)": round(t1 - t0, 2),
                "Notes": notes[:68]
            })

        print(f"[Zoom {zoom}] Tiles written with optimization pipeline.")
        
        print(f"[Zoom {zoom}] Memory statistics:")
        mm.print_stats()

        if zoom < max(zoom_levels):
            mm.command_pool.reset_stats()
            mm.point_pool.reset_stats()
            print(f"[Zoom {zoom}] Cache cleared for next zoom level\n")

def print_summary_table(summary_stats):
    print('\n' + '+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*17 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*19 + '+' + '-'*70 + '+')
    print('| {:<10} | {:<19} | {:<18} | {:<15} | {:<14} | {:<14} | {:<14} | {:<17} | {:<68} |'.format(
        "Zoom level", "Number of elements", "Memory usage (MB)", "Total dir size", "Min bin size", "Max bin size", "Avg bin size", "Processing time (s)", "Notes"))
    print('+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*17 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*19 + '+' + '-'*70 + '+')

    for entry in summary_stats:
        print('| {:<10} | {:<19} | {:<18} | {:<15} | {:<14} | {:<14} | {:<14} | {:<17} | {:<68} |'.format(
            entry["Zoom level"],
            entry["Number of elements"],
            entry["Memory usage (MB)"],
            entry["Total dir size (B)"],
            entry["Min bin size (B)"],
            entry["Max bin size (B)"],
            entry["Avg bin size (B)"],
            entry["Processing time (s)"],
            entry["Notes"][:68]
        ))
    print('+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*17 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*16 + '+' + '-'*19 + '+' + '-'*70 + '+')

def write_palette_bin(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    palette_path = os.path.join(output_dir, "palette.bin")
    print(f"Writing palette to {palette_path} ({len(GLOBAL_INDEX_TO_RGB332)} colors)...")
    with open(palette_path, "wb") as fp:
        for idx in range(len(GLOBAL_INDEX_TO_RGB332)):
            rgb332_val = GLOBAL_INDEX_TO_RGB332[idx]
            fp.write(bytes([rgb332_val]))
    print("Palette written OK.")

def main():
    parser = argparse.ArgumentParser(description="OSM vector tile generator with COMPLETE optimization pipeline")
    parser.add_argument("pbf_file", help="Path to .pbf file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("config_file", help="JSON config with features/colors")
    parser.add_argument("--zoom", help="Zoom level or range (e.g. 12 or 6-17)", default="6-17")
    parser.add_argument("--max-file-size", help="Maximum file size in KB", type=int, default=128)
    
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
    
    print("  - Urban grid pattern detection")
    print("  - Circular feature optimization")
    print("  - Coordinate prediction")
    print("  - Geometric primitive detection")
    print("  - Tile boundary optimization")

    # Initialize performance optimization pools
    initialize_memory_manager()

    geojson_tmp = os.path.abspath("tmp_extract.geojson")
    total_features_to_merge = extract_geojson_from_pbf(args.pbf_file, geojson_tmp, config, zoom_levels)

    print("Reading features and assigning to tiles with optimization pipeline ...")
    summary_stats = []
    streaming_assign_features_to_tiles_by_zoom(geojson_tmp, config, args.output_dir, zoom_levels, max_file_size, total_features=total_features_to_merge, summary_stats=summary_stats)
    print("Process completed successfully with optimization pipeline .")

    print("\nProcessing completed. Summary:")
    print_summary_table(summary_stats)

    if os.path.exists(geojson_tmp):
        os.remove(geojson_tmp)
    gc.collect()

    memory_manager.cleanup()

if __name__ == "__main__":
    main()