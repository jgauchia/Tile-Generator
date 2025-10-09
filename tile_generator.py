#!/usr/bin/env python3
"""
Optimized tile generator for OSM PBF files with dynamic resource allocation,
memory pooling, and streaming database operations.
"""

from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, box, shape
)
import math
import struct
import json
import os
import gc
import time
import sqlite3
import pickle
import lz4.frame
from collections import deque, Counter
from tqdm import tqdm
import argparse
import sys
import logging
import atexit
import psutil

# Try to import osmium for PBF processing
try:
    import osmium
    OSM_PYOSMIUM_AVAILABLE = True
except ImportError:
    OSM_PYOSMIUM_AVAILABLE = False

from concurrent.futures import ProcessPoolExecutor, as_completed
from tabulate import tabulate
from typing import List, Dict, Tuple, Optional, Union, Set, Any, Iterator, TYPE_CHECKING
from contextlib import contextmanager, ExitStack

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry as Geometry

# Memory pool for object reuse
class MemoryPool:
    """Memory pool for reusing objects to reduce garbage collection"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.point_pool = deque(maxlen=max_size)
        self.command_pool = deque(maxlen=max_size)
        self.coord_pool = deque(maxlen=max_size)
        self.feature_pool = deque(maxlen=max_size)
        
    def get_point(self) -> Tuple[int, int]:
        """Get a point tuple from pool or create new one"""
        if self.point_pool:
            return self.point_pool.popleft()
        return (0, 0)
    
    def return_point(self, point: Tuple[int, int]):
        """Return a point tuple to the pool"""
        if len(self.point_pool) < self.max_size:
            self.point_pool.append(point)
    
    def get_command(self) -> Dict:
        """Get a command dict from pool or create new one"""
        if self.command_pool:
            return self.command_pool.popleft()
        return {}
    
    def return_command(self, command: Dict):
        """Return a command dict to the pool"""
        if len(self.command_pool) < self.max_size:
            command.clear()  # Clear the dict for reuse
            self.command_pool.append(command)
    
    def get_coords(self) -> List[Tuple[int, int]]:
        """Get a coords list from pool or create new one"""
        if self.coord_pool:
            return self.coord_pool.popleft()
        return []
    
    def return_coords(self, coords: List[Tuple[int, int]]):
        """Return a coords list to the pool"""
        if len(self.coord_pool) < self.max_size:
            coords.clear()  # Clear the list for reuse
            self.coord_pool.append(coords)
    
    def get_feature(self) -> Dict:
        """Get a feature dict from pool or create new one"""
        if self.feature_pool:
            return self.feature_pool.popleft()
        return {}
    
    def return_feature(self, feature: Dict):
        """Return a feature dict to the pool"""
        if len(self.feature_pool) < self.max_size:
            feature.clear()  # Clear the dict for reuse
            self.feature_pool.append(feature)

# Global memory pool instance
memory_pool = MemoryPool(max_size=2000)

# Drawing command codes
# Global variables
DRAW_COMMANDS = {
    'LINE': 1,
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,  # Polygon outline only (no fill)
    'HORIZONTAL_LINE': 5,
    'VERTICAL_LINE': 6,
    'SET_COLOR': 0x80,  # State command for direct RGB332 color
    'SET_COLOR_INDEX': 0x81,  # State command for palette index
    'SET_LAYER': 0x88,  # Layer indicator command (moved up for priority)
    'RECTANGLE': 0x82,  # Optimized rectangle for buildings
    'STRAIGHT_LINE': 0x83,  # Optimized straight line for highways
    'HIGHWAY_SEGMENT': 0x84,  # Highway segment with continuity
    'GRID_PATTERN': 0x85,  # Urban grid pattern
    'BLOCK_PATTERN': 0x86,  # City block pattern
    'CIRCLE': 0x87,  # Circle/roundabout
    'RELATIVE_MOVE': 0x89,  # Relative coordinate movement
    'PREDICTED_LINE': 0x8A,  # Predictive line based on pattern
    'COMPRESSED_POLYLINE': 0x8B,  # Huffman-compressed polyline
    'OPTIMIZED_POLYGON': 0x8C,  # Optimized polygon (contour only, fill decided by viewer)
    'HOLLOW_POLYGON': 0x8D,  # Polygon outline only (optimized for boundaries)
    'OPTIMIZED_TRIANGLE': 0x8E,  # Optimized triangle (contour only, fill decided by viewer)
    'OPTIMIZED_RECTANGLE': 0x8F,  # Optimized rectangle (contour only, fill decided by viewer)
    'OPTIMIZED_CIRCLE': 0x90,  # Optimized circle (contour only, fill decided by viewer)
    'SIMPLE_RECTANGLE': 0x96,  # Simple rectangle (x, y, width, height)
    'SIMPLE_CIRCLE': 0x97,  # Simple circle (center_x, center_y, radius)
    'SIMPLE_TRIANGLE': 0x98,  # Simple triangle (x1, y1, x2, y2, x3, y3)
    'DASHED_LINE': 0x99,  # Dashed line with pattern
    'DOTTED_LINE': 0x9A,  # Dotted line with pattern
}

# Rendering layers (rendering order from bottom to top)
RENDER_LAYERS = {
    'TERRAIN': 0,      # Background terrain, water bodies
    'WATER': 1,        # Rivers, lakes, oceans
    'BUILDINGS': 2,    # Buildings, structures
    'OUTLINES': 3,     # Polygon outlines, borders
    'ROADS': 4,        # Roads, highways, paths
    'LABELS': 5        # Text labels, symbols
}

# Tile configuration
TILE_SIZE = 256
UINT16_TILE_SIZE = 65536

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
    
    # Processing limits - balanced for CPU and memory
    MAX_WORKERS = max(1, (os.cpu_count() or 4) // 2)  # Use half of available CPU cores
    DB_BATCH_SIZE = 50000  # Increased for better performance
    TILE_BATCH_SIZE = 2000  # Balanced batch size
    
    # Parallelization optimization - balanced approach
    TILE_COMPLEXITY_THRESHOLD = 100  # Features per tile threshold
    MEMORY_AWARE_BATCH_SIZE = 1000  # Balanced batches for memory-intensive tiles
    WORKER_MEMORY_LIMIT = 150 * 1024 * 1024  # 150MB per worker
    
    # I/O optimization
    FILE_BUFFER_SIZE = 65536  # 64KB buffer for file operations
    TILE_WRITE_BUFFER_SIZE = 32768  # 32KB buffer for tile writing
    GEOJSON_READ_BUFFER_SIZE = 8192  # 8KB buffer for GeoJSON reading
    BATCH_WRITE_SIZE = 100  # Write tiles in batches
    
    # Algorithm optimization - balanced cache sizes
    COORDINATE_PRECISION = 6  # Decimal places for coordinate precision
    MATH_CACHE_SIZE = 750  # Balanced cache size for mathematical operations
    PIXEL_COORD_CACHE_SIZE = 3000  # Balanced cache size for pixel coordinate calculations
    
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
        "low_zoom": 0.0001,  # For zoom <= 10 - Mucho más detallado
        "high_zoom": 0.00001 # For zoom > 10 - Máximo detalle
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
    
    # Drawing command codes and rendering layers are now imported from map_constants.py

# Global variables to track files for cleanup
_db_file_to_cleanup = None
_temp_files_to_cleanup = set()

def calculate_dynamic_resources() -> Dict[str, Any]:
    """Calculate dynamic resources based on total system memory (70% usage)"""
    memory_info = psutil.virtual_memory()
    cpu_count = os.cpu_count() or 4
    
    # Use 70% of total memory (convert bytes to MB)
    total_memory_mb = memory_info.total / (1024 * 1024)
    available_memory_mb = total_memory_mb * 0.7
    
    # Calculate optimal workers (use all cores for Pyosmium processing)
    max_workers = cpu_count
    
    # Calculate batch sizes based on available memory (more aggressive)
    db_batch_size = min(100000, max(10000, int(available_memory_mb * 3)))  # 3MB worth of features per batch
    tile_batch_size = min(2000, max(1000, int(available_memory_mb * 1)))   # 1MB worth of tiles per batch
    
    # Calculate worker memory limit (70% of available memory per worker)
    worker_memory_limit = int(available_memory_mb / max_workers)
    
    return {
        'max_workers': max_workers,
        'db_batch_size': db_batch_size,
        'tile_batch_size': tile_batch_size,
        'worker_memory_limit': worker_memory_limit,
        'available_memory_mb': available_memory_mb,
        'total_memory_mb': total_memory_mb,
        'memory_usage_percent': memory_info.percent
    }

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics in MB
    """
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024 * 1024),  # MB
        'available': memory.available / (1024 * 1024),  # MB
        'used': memory.used / (1024 * 1024),  # MB
        'percent': memory.percent,
        'free': memory.free / (1024 * 1024)  # MB
    }

def get_memory_pressure() -> str:
    """
    Assess current memory pressure level.
    
    Returns:
        Memory pressure level: 'low', 'medium', 'high', 'critical'
    """
    memory_info = get_memory_usage()
    percent_used = memory_info['percent']
    
    if percent_used < 60:
        return 'low'
    elif percent_used < 75:
        return 'medium'
    elif percent_used < 90:
        return 'high'
    else:
        return 'critical'

# Memory monitoring and management

def calculate_optimal_workers(memory_pressure: str, base_workers: int) -> int:
    """
    Calculate optimal number of workers based on memory pressure.
    
    Args:
        memory_pressure: Current memory pressure level
        base_workers: Base number of workers from CPU cores
        
    Returns:
        Optimal number of workers
    """
    if memory_pressure == 'low':
        return base_workers
    elif memory_pressure == 'medium':
        return max(1, base_workers // 2)
    elif memory_pressure == 'high':
        return max(1, base_workers // 3)
    else:  # critical
        return 1

def calculate_optimal_batch_size(memory_pressure: str, base_batch_size: int) -> int:
    """
    Calculate optimal batch size based on memory pressure.
    
    Args:
        memory_pressure: Current memory pressure level
        base_batch_size: Base batch size
        
    Returns:
        Optimal batch size
    """
    if memory_pressure == 'low':
        return base_batch_size
    elif memory_pressure == 'medium':
        return max(100, base_batch_size // 2)
    elif memory_pressure == 'high':
        return max(50, base_batch_size // 4)
    else:  # critical
        return max(10, base_batch_size // 8)

def force_memory_cleanup():
    """
    Force aggressive memory cleanup when memory pressure is high.
    """
    logger.debug("Performing aggressive memory cleanup...")
    
    # Clear all caches
    clear_all_caches()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    logger.debug("Memory cleanup completed")

def monitor_memory_and_adjust(resources: Dict[str, Any] = None):
    """
    Monitor memory usage and adjust processing parameters dynamically.
    
    Args:
        resources: Dynamic resources calculated from system memory (70% of total)
    
    Returns:
        Tuple of (adjusted_workers, adjusted_batch_size, memory_pressure)
    """
    memory_pressure = get_memory_pressure()
    memory_info = get_memory_usage()
    
    logger.debug(f"Memory: {memory_info['used']:.1f}MB used ({memory_info['percent']:.1f}%), "
                f"{memory_info['available']:.1f}MB available, pressure: {memory_pressure}")
    
    # Use dynamic resources if provided, otherwise fall back to config
    if resources:
        base_workers = resources['max_workers']
        base_batch_size = resources['tile_batch_size']
    else:
        base_workers = Config.MAX_WORKERS
        base_batch_size = Config.TILE_BATCH_SIZE
    
    # Adjust workers based on memory pressure
    optimal_workers = calculate_optimal_workers(memory_pressure, base_workers)
    
    # Adjust batch size based on memory pressure
    optimal_batch_size = calculate_optimal_batch_size(memory_pressure, base_batch_size)
    
    # Force cleanup if memory pressure is critical
    if memory_pressure in ['high', 'critical']:
        force_memory_cleanup()
    
    return optimal_workers, optimal_batch_size, memory_pressure

@contextmanager
def managed_database(db_path: str) -> Iterator['FeatureDatabase']:
    """Context manager for database connections with automatic cleanup"""
    db = None
    try:
        db = FeatureDatabase(db_path)
        
        # Register for cleanup
        global _db_file_to_cleanup
        _db_file_to_cleanup = db_path
        
        yield db
        
    finally:
        if db:
            db.close()
        # Don't reset _db_file_to_cleanup here - let cleanup_all() handle it

@contextmanager
def memory_management():
    """Context manager for memory management with optimized garbage collection"""
    try:
        yield
    finally:
        # Only collect garbage if we have significant memory pressure
        import sys
        if sys.getsizeof(gc.get_objects()) > 50 * 1024 * 1024:  # 50MB threshold
            gc.collect()
        
        # Log memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
        except ImportError:
            pass  # psutil not available


def smart_gc_collect():
    """Perform garbage collection only when memory pressure is high"""
    import sys
    
    # Check if we should collect garbage based on object count and memory usage
    object_count = len(gc.get_objects())
    memory_usage = sys.getsizeof(gc.get_objects())
    
    # Collect if we have many objects or high memory usage
    if object_count > 100000 or memory_usage > 100 * 1024 * 1024:  # 100MB threshold
        gc.collect()
        logger.debug(f"Garbage collection performed: {object_count} objects, {memory_usage / 1024 / 1024:.1f} MB")

@contextmanager
def resource_monitor():
    """Context manager for monitoring resource usage"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"Operation completed in {elapsed:.2f} seconds")

def cleanup_database():
    """Clean up database file and WAL mode files if they exist"""
    global _db_file_to_cleanup
    if _db_file_to_cleanup and os.path.exists(_db_file_to_cleanup):
        try:
            # Remove main database file
            os.remove(_db_file_to_cleanup)
            logger.info(f"Cleaned up database file: {_db_file_to_cleanup}")
            
            # Remove WAL mode files if they exist
            wal_file = _db_file_to_cleanup + "-wal"
            shm_file = _db_file_to_cleanup + "-shm"
            
            if os.path.exists(wal_file):
                os.remove(wal_file)
                logger.info(f"Cleaned up WAL file: {wal_file}")
            
            if os.path.exists(shm_file):
                os.remove(shm_file)
                logger.info(f"Cleaned up SHM file: {shm_file}")
                
        except Exception as e:
            logger.debug(f"Could not remove database files {_db_file_to_cleanup}: {e}")
        finally:
            # Reset the global variable after cleanup
            _db_file_to_cleanup = None

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


# Register cleanup functions
atexit.register(cleanup_all)

# Use Config class constants instead of global variables

# Global variables for dynamic palette
GLOBAL_COLOR_PALETTE = {}  # hex_color -> index
GLOBAL_INDEX_TO_RGB332 = {}  # index -> rgb332_value

# Global caches for geometry optimization
GEOMETRY_CACHE = {}  # Cache for simplified geometries by zoom level
TILE_BBOX_CACHE = {}  # Cache for tile bounding boxes
SIMPLIFY_TOLERANCE_CACHE = {}  # Cache for simplify tolerances


# Database configuration moved to Config class

class FeatureDatabase:
    """Optimized database for storing and retrieving features by zoom level and tile coordinates"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Enable WAL mode for better concurrency and performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        # Optimize SQLite settings for performance
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster than FULL, still safe
        self.conn.execute("PRAGMA cache_size=10000")    # 10MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")     # Store temp tables in memory
        self.conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
        
        # Create optimized table structure
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
        
        # Create optimized composite indexes for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom_tile_priority 
            ON features(zoom_level, tile_x, tile_y, priority)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom_tile 
            ON features(zoom_level, tile_x, tile_y)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom 
            ON features(zoom_level)
        """)
        
        # Additional optimized indexes for specific query patterns
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tile_coords 
            ON features(tile_x, tile_y, zoom_level)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority_zoom 
            ON features(priority, zoom_level)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom_tile_distinct 
            ON features(zoom_level, tile_x, tile_y) 
            WHERE zoom_level IS NOT NULL
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_zoom_count 
            ON features(zoom_level) 
            WHERE zoom_level IS NOT NULL
        """)
        
        # Prepare statements for better performance
        self._prepare_statements()
        
        self.conn.commit()
    
    def _prepare_statements(self):
        """Store SQL statements for better performance (Python sqlite3 doesn't have prepare)"""
        self.insert_sql = """
            INSERT INTO features (zoom_level, tile_x, tile_y, feature_data, priority)
            VALUES (?, ?, ?, ?, ?)
        """
        self.select_sql = """
            SELECT feature_data, priority FROM features 
            WHERE zoom_level = ? AND tile_x = ? AND tile_y = ?
            ORDER BY priority ASC
        """
        self.tiles_sql = """
            SELECT DISTINCT tile_x, tile_y FROM features 
            WHERE zoom_level = ?
            ORDER BY tile_x, tile_y
        """
        self.count_sql = """
            SELECT COUNT(*) FROM features WHERE zoom_level = ?
        """
        self.clear_sql = """
            DELETE FROM features WHERE zoom_level = ?
        """
        self.tile_stats_sql = """
            SELECT tile_x, tile_y, COUNT(*) as feature_count, 
                   AVG(LENGTH(feature_data)) as avg_size
            FROM features 
            WHERE zoom_level = ?
            GROUP BY tile_x, tile_y
            ORDER BY tile_x, tile_y
        """
    
    def insert_feature(self, zoom_level, tile_x, tile_y, feature_data, priority=5):
        """Insert a feature into the database using optimized SQL with compression"""
        # Convert Shapely geometry to WKT for serialization
        serializable_data = feature_data.copy()
        if 'geom' in serializable_data and hasattr(serializable_data['geom'], 'wkt'):
            serializable_data['geom'] = serializable_data['geom'].wkt
        
        # Serialize and compress data
        serialized_data = pickle.dumps(serializable_data)
        compressed_data = compress_feature_data(serialized_data)
        
        self.conn.execute(self.insert_sql, (zoom_level, tile_x, tile_y, compressed_data, priority))
    
    def insert_features_batch(self, features_batch: List[Tuple[int, int, int, Dict[str, Any], int]]):
        """Insert multiple features in a single transaction for better performance"""
        if not features_batch:
            return
        
        try:
            # Use executemany for batch insert with compression
            data = []
            for zoom_level, tile_x, tile_y, feature_data, priority in features_batch:
                # Convert Shapely geometry to WKT for serialization
                serializable_data = feature_data.copy()
                if 'geom' in serializable_data and hasattr(serializable_data['geom'], 'wkt'):
                    serializable_data['geom'] = serializable_data['geom'].wkt
                
                # Serialize and compress data
                serialized_data = pickle.dumps(serializable_data)
                compressed_data = compress_feature_data(serialized_data)
                data.append((zoom_level, tile_x, tile_y, compressed_data, priority))
            
            self.conn.executemany(self.insert_sql, data)
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            # Fallback to individual inserts
            for zoom_level, tile_x, tile_y, feature_data, priority in features_batch:
                self.insert_feature(zoom_level, tile_x, tile_y, feature_data, priority)
    
    def get_features_for_tile(self, zoom_level, tile_x, tile_y):
        """Get all features for a specific tile using optimized SQL"""
        # Use row factory for better performance
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(self.select_sql, (zoom_level, tile_x, tile_y))
        
        features = []
        for row in cursor:
            # Decompress and deserialize data
            compressed_data = row['feature_data']
            decompressed_data = decompress_feature_data(compressed_data)
            feature_data = pickle.loads(decompressed_data)
            
            # Convert WKT back to Shapely geometry if needed
            if 'geom' in feature_data and isinstance(feature_data['geom'], str):
                try:
                    from shapely import wkt
                    feature_data['geom'] = wkt.loads(feature_data['geom'])
                except ImportError:
                    # Fallback if shapely.wkt is not available
                    pass
            features.append(feature_data)
        
        # Reset row factory to default
        self.conn.row_factory = None
        return features
    
    def get_tiles_for_zoom(self, zoom_level):
        """Get all tile coordinates for a specific zoom level using optimized SQL"""
        # Use row factory for better performance
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(self.tiles_sql, (zoom_level,))
        tiles = [(row['tile_x'], row['tile_y']) for row in cursor]
        
        # Reset row factory to default
        self.conn.row_factory = None
        return tiles
    
    def count_features_for_zoom(self, zoom_level):
        """Count total features for a zoom level using optimized SQL"""
        cursor = self.conn.execute(self.count_sql, (zoom_level,))
        return cursor.fetchone()[0]
    
    def get_tile_statistics(self, zoom_level):
        """Get tile statistics for a zoom level using optimized SQL"""
        # Use row factory for better performance
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(self.tile_stats_sql, (zoom_level,))
        stats = [(row['tile_x'], row['tile_y'], row['feature_count'], row['avg_size']) for row in cursor]
        
        # Reset row factory to default
        self.conn.row_factory = None
        return stats
    
    def stream_features_for_tile(self, zoom_level, tile_x, tile_y):
        """Stream features for a specific tile using generator for memory efficiency"""
        # Use row factory for better performance
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(self.select_sql, (zoom_level, tile_x, tile_y))
        
        try:
            for row in cursor:
                # Decompress and deserialize data
                compressed_data = row['feature_data']
                decompressed_data = decompress_feature_data(compressed_data)
                feature_data = pickle.loads(decompressed_data)
                
                # Convert WKT back to Shapely geometry if needed
                if 'geom' in feature_data and isinstance(feature_data['geom'], str):
                    try:
                        from shapely import wkt
                        feature_data['geom'] = wkt.loads(feature_data['geom'])
                    except ImportError:
                        # Fallback if shapely.wkt is not available
                        pass
                yield feature_data
        finally:
            # Reset row factory to default
            self.conn.row_factory = None
    
    def stream_tiles_for_zoom(self, zoom_level):
        """Stream tile coordinates for a specific zoom level using generator for memory efficiency"""
        # Use row factory for better performance
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.execute(self.tiles_sql, (zoom_level,))
        
        try:
            for row in cursor:
                yield (row['tile_x'], row['tile_y'])
        finally:
            # Reset row factory to default
            self.conn.row_factory = None
    
    def close(self) -> None:
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def commit(self) -> None:
        """Commit pending transactions"""
        if self.conn:
            self.conn.commit()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup"""
        self.close()
        if exc_type is not None:
            logger.error(f"Database error: {exc_val}")
        return False  # Don't suppress exceptions


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    """
    Convert latitude/longitude coordinates to tile coordinates.
    
    Args:
        lat_deg: Latitude in degrees
        lon_deg: Longitude in degrees  
        zoom: Zoom level (0-18)
        
    Returns:
        Tuple of (x, y) tile coordinates
        
    Example:
        >>> deg2num(42.5, 1.5, 10)
        (512, 384)
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def coords_to_pixel_coords_uint16(coords: List[Tuple[float, float]], zoom: int, tile_x: int, tile_y: int) -> List[Tuple[int, int]]:
    """
    Convert coordinate list to pixel coordinates relative to a specific tile (simplified version).
    
    Args:
        coords: List of (longitude, latitude) coordinate tuples
        zoom: Zoom level (0-18)
        tile_x: X coordinate of the target tile
        tile_y: Y coordinate of the target tile
        
    Returns:
        List of (x, y) pixel coordinates relative to the tile (0-65535 range)
    """
    n = 2.0 ** zoom
    # Get coords list from pool
    uint16_coords = memory_pool.get_coords()
    
    for lon, lat in coords:
        # Calculate pixel coordinates within the tile
        x = int(((lon + 180.0) / 360.0 * n - tile_x) * Config.TILE_SIZE)
        y = int(((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n - tile_y) * Config.TILE_SIZE)
        
        # Clamp to tile bounds
        x = max(0, min(Config.TILE_SIZE - 1, x))
        y = max(0, min(Config.TILE_SIZE - 1, y))
    
    # Convert to UINT16 range
        x_uint16 = int((x * (Config.UINT16_TILE_SIZE - 1)) / (Config.TILE_SIZE - 1))
        y_uint16 = int((y * (Config.UINT16_TILE_SIZE - 1)) / (Config.TILE_SIZE - 1))
        x_uint16 = max(0, min(Config.UINT16_TILE_SIZE - 1, x_uint16))
        y_uint16 = max(0, min(Config.UINT16_TILE_SIZE - 1, y_uint16))
        
        uint16_coords.append((x_uint16, y_uint16))
    
    return uint16_coords

def remove_duplicate_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 1:
        return points
    result = [points[0]]
    for pt in points[1:]:
        if pt != result[-1]:
            result.append(pt)
    return result

def hex_to_rgb332_direct(hex_color: str) -> int:
    """
    Convert hex color string to RGB332 format (8-bit color).
    
    Args:
        hex_color: Hex color string (e.g., "#FF0000")
        
    Returns:
        8-bit RGB332 color value (0-255)
        
    Example:
        >>> hex_to_rgb332_direct("#FF0000")
        224  # Red in RGB332 format
    """
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

def hex_to_color_index(hex_color: str) -> Optional[int]:
    global GLOBAL_COLOR_PALETTE
    return GLOBAL_COLOR_PALETTE.get(hex_color, None)

def determine_render_layer(tags: Dict[str, str], geometry_type: str) -> str:
    """
    Determine the rendering layer for a feature based on its tags and geometry type.
    
    Args:
        tags: OSM tags dictionary
        geometry_type: Type of geometry ('polygon', 'line', 'point')
    
    Returns:
        Layer name for rendering order
    """
    # Water bodies and terrain
    if tags.get('natural') in ['water', 'coastline', 'bay', 'strait']:
        return 'WATER'
    if tags.get('waterway') in ['river', 'stream', 'canal', 'ditch']:
        return 'WATER'
    if tags.get('landuse') in ['water', 'reservoir']:
        return 'WATER'
    
    # Buildings and structures
    if tags.get('building') or tags.get('amenity') in ['school', 'hospital', 'university']:
        return 'BUILDINGS'
    if tags.get('landuse') in ['residential', 'commercial', 'industrial']:
        return 'BUILDINGS'
    
    # Roads and transportation
    if tags.get('highway') or tags.get('railway'):
        return 'ROADS'
    if tags.get('aeroway') in ['runway', 'taxiway']:
        return 'ROADS'
    
    # Default based on geometry type
    if geometry_type == 'polygon':
        return 'TERRAIN'
    elif geometry_type == 'line':
        return 'ROADS'
    else:
        return 'LABELS'

def get_style_for_tags(tags: Dict[str, str], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Find matching style configuration for feature tags.
    
    Args:
        tags: Dictionary of feature tags (e.g., {"highway": "primary"})
        config: Style configuration dictionary
        
    Returns:
        Tuple of (style_dict, matching_key) or ({}, None) if no match
        
    Example:
        >>> tags = {"highway": "primary", "surface": "asphalt"}
        >>> config = {"highway=primary": {"color": "#FF0000"}}
        >>> get_style_for_tags(tags, config)
        ({"color": "#FF0000"}, "highway=primary")
    """
    for k, v in tags.items():
        keyval = f"{k}={v}"
        if keyval in config:
            return config[keyval], keyval
    for k in tags:
        if k in config:
            return config[k], k
    return {}, None


def is_area(tags: Dict[str, str]) -> bool:
    """
    Determine if a feature should be treated as an area based on its tags.
    
    Args:
        tags: Dictionary of feature tags
        
    Returns:
        True if the feature should be treated as an area, False otherwise
        
    Example:
        >>> is_area({"building": "house"})
        True
        >>> is_area({"highway": "primary"})
        False
    """
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


def get_simplify_tolerance_for_zoom(zoom: int) -> float:
    """
    Get simplify tolerance for zoom level.
    
    Args:
        zoom: Zoom level
        
    Returns:
        Simplify tolerance value
    """
    if zoom <= 10:
        return Config.SIMPLIFY_TOLERANCES["low_zoom"]
    else:
        return Config.SIMPLIFY_TOLERANCES["high_zoom"]


def get_tile_bbox_cached(tile_x: int, tile_y: int, zoom: int) -> 'box':
    """
    Get tile bounding box with caching.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        Shapely box geometry for tile bounds
    """
    global TILE_BBOX_CACHE
    
    cache_key = (tile_x, tile_y, zoom)
    if cache_key not in TILE_BBOX_CACHE:
        lon_min, lat_min, lon_max, lat_max = optimized_tile_latlon_bounds(tile_x, tile_y, zoom)
        TILE_BBOX_CACHE[cache_key] = box(lon_min, lat_min, lon_max, lat_max)
    
    return TILE_BBOX_CACHE[cache_key]


def get_simplified_geometry_cached(geom: 'Geometry', zoom: int) -> 'Geometry':
    """
    Get simplified geometry with caching.
    
    Args:
        geom: Original geometry
        zoom: Zoom level
        
    Returns:
        Simplified geometry
    """
    global GEOMETRY_CACHE
    
    # Create cache key based on geometry bounds and zoom
    bounds = geom.bounds
    cache_key = (bounds, zoom)
    
    if cache_key not in GEOMETRY_CACHE:
        simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
        
        if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
            try:
                simplified = geom.simplify(simplify_tolerance, preserve_topology=True)
                GEOMETRY_CACHE[cache_key] = simplified
            except (ValueError, TypeError):
                # Geometry simplification failed - use original geometry
                GEOMETRY_CACHE[cache_key] = geom
        else:
            GEOMETRY_CACHE[cache_key] = geom
    
    return GEOMETRY_CACHE[cache_key]


def clear_geometry_caches():
    """Clear all geometry caches to free memory"""
    global GEOMETRY_CACHE, TILE_BBOX_CACHE, SIMPLIFY_TOLERANCE_CACHE
    GEOMETRY_CACHE.clear()
    TILE_BBOX_CACHE.clear()
    SIMPLIFY_TOLERANCE_CACHE.clear()
    logger.debug("Geometry caches cleared")


def clear_all_caches():
    """Clear all caches to free memory"""
    clear_geometry_caches()
    clear_algorithm_caches()
    logger.debug("All caches cleared")




def compress_feature_data(data: bytes) -> bytes:
    """Compress feature data using lz4 for better I/O performance"""
    try:
        return lz4.frame.compress(data)
    except Exception as e:
        logger.debug(f"Compression failed, using original data: {e}")
        return data

def decompress_feature_data(data: bytes) -> bytes:
    """Decompress feature data using lz4"""
    try:
        return lz4.frame.decompress(data)
    except Exception as e:
        logger.debug(f"Decompression failed, using original data: {e}")
        return data

def precompute_tile_bounds_for_zoom(zoom_level: int, bounds: Tuple[float, float, float, float]) -> Set[Tuple[int, int]]:
    """
    Pre-compute all valid tile coordinates for a zoom level within given bounds.
    
    Args:
        zoom_level: Zoom level to compute tiles for
        bounds: (min_lon, min_lat, max_lon, max_lat) bounds
    
    Returns:
        Set of (tile_x, tile_y) coordinates
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Calculate tile bounds
    min_tile_x, max_tile_y = deg2num(max_lat, min_lon, zoom_level)
    max_tile_x, min_tile_y = deg2num(min_lat, max_lon, zoom_level)
    
    # Generate all tile coordinates within bounds
    tiles = set()
    for tile_x in range(min_tile_x, max_tile_x + 1):
        for tile_y in range(min_tile_y, max_tile_y + 1):
            tiles.add((tile_x, tile_y))
    
    return tiles


def optimized_file_write(filepath: str, data: bytes, buffer_size: int = None) -> None:
    """
    Optimized file write with buffering and error handling.
    
    Args:
        filepath: Path to file to write
        data: Data to write
        buffer_size: Buffer size for writing (default: TILE_WRITE_BUFFER_SIZE)
    """
    if buffer_size is None:
        buffer_size = Config.TILE_WRITE_BUFFER_SIZE
    
    try:
        with open(filepath, "wb", buffering=buffer_size) as f:
            f.write(data)
            f.flush()  # Ensure data is written to disk
    except IOError as e:
        logger.error(f"Failed to write file {filepath}: {e}")
        raise


def create_directory_structure(base_path: str, tile_coords: List[Tuple[int, int]]) -> None:
    """
    Pre-create directory structure for tiles to avoid repeated os.makedirs calls.
    
    Args:
        base_path: Base output directory
        tile_coords: List of (x, y) tile coordinates
    """
    dirs_to_create = set()
    for x, y in tile_coords:
        tile_dir = os.path.join(base_path, str(x))
        dirs_to_create.add(tile_dir)
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)




    """
    Calculate tile bounds for lat/lon coordinates.
    
    Args:
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        pixel_margin: Pixel margin for bounds
        
    Returns:
        Tuple of (lon_min, lat_min, lon_max, lat_max)
    """
    n = 2.0 ** zoom
    
    # Calculate latitude bounds
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n))))
    
    # Calculate longitude bounds
    lon_min = tile_x / n * 360.0 - 180.0
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    
    # Apply pixel margin
    if pixel_margin > 0:
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        lat_margin = lat_range * pixel_margin / Config.TILE_SIZE
        lon_margin = lon_range * pixel_margin / Config.TILE_SIZE
        
        lat_min -= lat_margin
        lat_max += lat_margin
        lon_min -= lon_margin
        lon_max += lon_margin
    
    return (lon_min, lat_min, lon_max, lat_max)



def clear_algorithm_caches():
    """Clear all algorithm-related caches to free memory"""
    logger.debug("Algorithm caches cleared")





def create_optimized_tile_batches(tiles: List[Tuple[int, int, List[Dict[str, Any]]]]) -> List[List[Tuple[int, int, List[Dict[str, Any]]]]]:
    """
    Create simple tile batches for processing.
    
    Args:
        tiles: List of tile data tuples
        
    Returns:
        List of tile batches
    """
    batches = []
    
    # Create simple batches
    for i in range(0, len(tiles), Config.TILE_BATCH_SIZE):
        batch = tiles[i:i + Config.TILE_BATCH_SIZE]
        if batch:
            batches.append(batch)
    
    logger.debug(f"Created {len(batches)} batches with {len(tiles)} tiles")
    
    return batches


def should_process_geometry_for_tile(geom: 'Geometry', tile_x: int, tile_y: int, zoom: int) -> bool:
    """
    Pre-filter geometry to avoid expensive intersection operations.
    
    Args:
        geom: Geometry to check
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        True if geometry should be processed for this tile
    """
    try:
        # Get tile bounding box
        tile_bbox = get_tile_bbox_cached(tile_x, tile_y, zoom)
        
        # Quick bounds check - much faster than intersection
        geom_bounds = geom.bounds
        tile_bounds = tile_bbox.bounds
        
        # Check if geometry bounds overlap with tile bounds
        return not (geom_bounds[2] < tile_bounds[0] or  # geom right < tile left
                   geom_bounds[0] > tile_bounds[2] or  # geom left > tile right
                   geom_bounds[3] < tile_bounds[1] or  # geom bottom < tile top
                   geom_bounds[1] > tile_bounds[3])    # geom top > tile bottom
                   
    except Exception:
        # If bounds check fails, process the geometry (fallback)
        return True


def optimized_geometry_intersection(geom: 'Geometry', tile_x: int, tile_y: int, zoom: int) -> Optional['Geometry']:
    """
    Optimized geometry intersection with pre-filtering and caching.
    
    Args:
        geom: Geometry to intersect
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        zoom: Zoom level
        
    Returns:
        Intersected geometry or None if no intersection
    """
    try:
        # Pre-filter to avoid expensive operations
        if not should_process_geometry_for_tile(geom, tile_x, tile_y, zoom):
            return None
        
        # Get cached tile bounding box
        tile_bbox = get_tile_bbox_cached(tile_x, tile_y, zoom)
        
        # Perform intersection
        clipped_geom = geom.intersection(tile_bbox)
        
        # Check if intersection is valid and not empty
        if clipped_geom.is_empty or not clipped_geom.is_valid:
            return None
            
        return clipped_geom
        
    except (ValueError, TypeError):
        # Geometry intersection failed - skip this tile
        return None

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
            color = cmd.get('color', 0xFF)  # Default to white if color is None
            if color is not None:
                color = color & 0xFF
            else:
                color = 0xFF  # Default white color
            out += struct.pack("B", color)
        elif t == DRAW_COMMANDS['SET_COLOR_INDEX']:
            # SET_COLOR_INDEX command (palette index)
            color_index = cmd.get('color_index', 0)  # Default to 0 if color_index is None
            if color_index is not None:
                color_index = color_index & 0xFF
            else:
                color_index = 0  # Default color index
            out += pack_varint(color_index)
        elif t == DRAW_COMMANDS['SET_LAYER']:
            # SET_LAYER command (layer indicator)
            layer = cmd.get('layer', 0)  # Default to TERRAIN layer if layer is None
            if layer is not None:
                layer = layer & 0xFF
            else:
                layer = 0  # Default to TERRAIN layer
            out += pack_varint(layer)
        elif t == DRAW_COMMANDS['RECTANGLE']:
            # Optimized RECTANGLE command
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == DRAW_COMMANDS['STRAIGHT_LINE']:
            # Optimized STRAIGHT_LINE command
            points = cmd['points']
            out += pack_varint(len(points))
            prev_x, prev_y = 0, 0
            for i, (x, y) in enumerate(points):
                x, y = clamp_uint16(x), clamp_uint16(y)
                if i == 0:
                    out += pack_zigzag(x)
                    out += pack_zigzag(y)
                else:
                    out += pack_zigzag(x - prev_x)
                    out += pack_zigzag(y - prev_y)
                prev_x, prev_y = x, y
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
        elif t == DRAW_COMMANDS['OPTIMIZED_POLYGON']:
            # Optimized polygon command (contour only, fill decided by viewer)
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
        elif t == DRAW_COMMANDS['HOLLOW_POLYGON']:
            # Optimized hollow polygon command (outline only)
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
        elif t == DRAW_COMMANDS['OPTIMIZED_TRIANGLE']:
            # Optimized triangle command (contour only, fill decided by viewer)
            x1, y1, x2, y2, x3, y3 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2'], cmd['x3'], cmd['y3']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
            out += pack_zigzag(x3 - x2)
            out += pack_zigzag(y3 - y2)
        elif t == DRAW_COMMANDS['OPTIMIZED_RECTANGLE']:
            # Optimized rectangle command (contour only, fill decided by viewer)
            x, y, width, height = map(clamp_uint16, [cmd['x'], cmd['y'], cmd['width'], cmd['height']])
            out += pack_zigzag(x)
            out += pack_zigzag(y)
            out += pack_zigzag(width)
            out += pack_zigzag(height)
        elif t == DRAW_COMMANDS['OPTIMIZED_CIRCLE']:
            # Optimized circle command (contour only, fill decided by viewer)
            center_x, center_y, radius = map(clamp_uint16, [cmd['center_x'], cmd['center_y'], cmd['radius']])
            out += pack_zigzag(center_x)
            out += pack_zigzag(center_y)
            out += pack_zigzag(radius)
        elif t == DRAW_COMMANDS['PREDICTED_LINE']:
            # Step 6: Predicted line command (only end point needed)
            end_x, end_y = map(clamp_uint16, [cmd['end_x'], cmd['end_y']])
            out += pack_zigzag(end_x)
            out += pack_zigzag(end_y)
        elif t == DRAW_COMMANDS['SIMPLE_RECTANGLE']:
            # Simple rectangle command (x, y, width, height)
            x, y, width, height = map(clamp_uint16, [cmd['x'], cmd['y'], cmd['width'], cmd['height']])
            out += pack_zigzag(x)
            out += pack_zigzag(y)
            out += pack_zigzag(width)
            out += pack_zigzag(height)
        elif t == DRAW_COMMANDS['SIMPLE_CIRCLE']:
            # Simple circle command (center_x, center_y, radius)
            center_x, center_y, radius = map(clamp_uint16, [cmd['center_x'], cmd['center_y'], cmd['radius']])
            out += pack_zigzag(center_x)
            out += pack_zigzag(center_y)
            out += pack_zigzag(radius)
        elif t == DRAW_COMMANDS['SIMPLE_TRIANGLE']:
            # Simple triangle command (x1, y1, x2, y2, x3, y3)
            x1, y1, x2, y2, x3, y3 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2'], cmd['x3'], cmd['y3']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2)
            out += pack_zigzag(y2)
            out += pack_zigzag(x3)
            out += pack_zigzag(y3)
        elif t == DRAW_COMMANDS['DASHED_LINE']:
            # Dashed line command (x1, y1, x2, y2, dash_length, gap_length)
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            dash_length = cmd.get('dash_length', 4)
            gap_length = cmd.get('gap_length', 2)
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2)
            out += pack_zigzag(y2)
            out += pack_varint(dash_length)
            out += pack_varint(gap_length)
        elif t == DRAW_COMMANDS['DOTTED_LINE']:
            # Dotted line command (x1, y1, x2, y2, dot_spacing)
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            dot_spacing = cmd.get('dot_spacing', 2)
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2)
            out += pack_zigzag(y2)
            out += pack_varint(dot_spacing)
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
    # Get commands list from pool
    commands = memory_pool.get_coords()  # Reuse coords pool for commands list
    
    def process_geom(g):
        local_cmds = memory_pool.get_coords()  # Reuse coords pool for local commands
        if g.is_empty:
            return local_cmds
        
        # Use original geometry processing without aggressive optimization
        if g.geom_type == "Polygon":
            exterior = remove_duplicate_points(list(g.exterior.coords))
            exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
            exterior_pixels = ensure_closed_ring(exterior_pixels)
            if len(set(exterior_pixels)) >= 3:
                # Try to detect simple shapes first
                simple_shape = detect_simple_shapes(g, exterior_pixels)
                if simple_shape:
                    # Use optimized command for simple shape
                    cmd = memory_pool.get_command()
                    cmd.update(simple_shape)
                    cmd['color'] = color
                    if hex_color:
                        cmd['color_hex'] = hex_color
                    local_cmds.append(cmd)
                else:
                    # Use STROKE_POLYGON for polygons without fill (contour only)
                    cmd = memory_pool.get_command()
                    cmd.update({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color})
                    if hex_color:
                        cmd['color_hex'] = hex_color
                local_cmds.append(cmd)
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                exterior = remove_duplicate_points(list(poly.exterior.coords))
                exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                if len(set(exterior_pixels)) >= 3:
                    # Try to detect simple shapes first
                    simple_shape = detect_simple_shapes(poly, exterior_pixels)
                    if simple_shape:
                        # Use optimized command for simple shape
                        cmd = memory_pool.get_command()
                        cmd.update(simple_shape)
                        cmd['color'] = color
                        if hex_color:
                            cmd['color_hex'] = hex_color
                        local_cmds.append(cmd)
                    else:
                        # Use STROKE_POLYGON for polygons without fill (contour only)
                        cmd = memory_pool.get_command()
                        cmd.update({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color})
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
                    # Use STROKE_POLYGON for closed lines that are areas (without fill)
                    cmd = memory_pool.get_command()
                    cmd.update({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': pixel_coords, 'color': color})
                if hex_color:
                    cmd['color_hex'] = hex_color
                local_cmds.append(cmd)
            else:
                if len(pixel_coords) == 2:
                    x1, y1 = pixel_coords[0]
                    x2, y2 = pixel_coords[1]
                    cmd = memory_pool.get_command()
                    cmd.update({'color': color})
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
                    cmd = memory_pool.get_command()
                    cmd.update({'type': DRAW_COMMANDS['POLYLINE'], 'points': pixel_coords, 'color': color})
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
        # Add layer command at the beginning (using priority from features.json)
        layer_cmd = memory_pool.get_command()
        layer_cmd.update({
            'type': DRAW_COMMANDS['SET_LAYER'],  # SET_LAYER
            'layer': get_layer_from_priority(tags)
        })
        commands.append(layer_cmd)
        
        commands.extend(process_geom(geom))
    return commands

def get_config_fields(config):
    """Extract field names from configuration"""
    fields = set()
    for k in config.keys():
        if "=" in k:
            key, _ = k.split("=", 1)
            fields.add(key)
        else:
            fields.add(k)
    return fields

def validate_pbf_file(pbf_file: str) -> None:
    """
    Validate PBF file exists, is readable, and has correct format.
    
    Args:
        pbf_file: Path to the PBF file to validate
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PBF file or is empty
        PermissionError: If file is not readable
        
    Example:
        >>> validate_pbf_file("data.osm.pbf")
        # No exception if file is valid
    """
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

def validate_config_file(config_file: str) -> Dict[str, Any]:
    """
    Validate configuration file exists and contains valid JSON.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file contains invalid JSON
        
    Example:
        >>> config = validate_config_file("features.json")
        >>> print(config["highway=primary"]["color"])
        "#FF0000"
    """
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

def validate_output_dir(output_dir: str) -> None:
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

def validate_zoom_range(zoom_str: str) -> List[int]:
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

def validate_max_file_size(max_file_size_kb: int) -> None:
    """Validate maximum file size parameter"""
    if not isinstance(max_file_size_kb, int):
        raise ValueError("Max file size must be an integer")
    
    if max_file_size_kb <= 0:
        raise ValueError("Max file size must be positive")
    
    if max_file_size_kb > Config.MAX_FILE_SIZE_KB_LIMIT:
        raise ValueError(f"Max file size cannot exceed {Config.MAX_FILE_SIZE_KB_LIMIT} KB ({Config.MAX_FILE_SIZE_KB_LIMIT // 1024}MB)")
    
    logger.debug(f"Max file size validation passed: {max_file_size_kb} KB")

def validate_db_path(db_path: str) -> None:
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

def get_layer_fields_from_pbf(pbf_file: str, layer: str) -> Set[str]:
    """Get available fields from a PBF layer"""
    cmd = ["ogrinfo", "-so", pbf_file, layer]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Could not get fields for layer {layer}: {result.stderr}")
        return set()
    
    fields = set()
    for line in result.stdout.split('\n'):
        if ':' in line and not line.strip().startswith('Geometry'):
            field_name = line.split(':')[0].strip()
            if field_name and not field_name.startswith('FID'):
                fields.add(field_name)
    
    logger.debug(f"Available fields for layer {layer}: {sorted(fields)}")
    return fields

def check_pbf_layers(pbf_file: str) -> List[str]:
    """Check what layers are available in the PBF file"""
    cmd = ["ogrinfo", pbf_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Could not read PBF file: {result.stderr}")
        return []
    
    layers = []
    for line in result.stdout.split('\n'):
        # Look for lines like "1: points (Point)" or "2: lines (Line String)"
        if ':' in line and not line.startswith('INFO:'):
            parts = line.split(':')
            if len(parts) >= 2:
                layer_name = parts[1].strip().split(' ')[0]  # Get just the layer name before the parentheses
                if layer_name and layer_name not in ['Open', 'using', 'driver']:
                    layers.append(layer_name)
    
    logger.info(f"Available layers in PBF: {layers}")
    return layers

def extract_layer_to_temp_file(pbf_file: str, layer: str, where_clause: str, select_fields: str) -> Optional[str]:
    """Extract layer data from PBF to temporary GeoJSON file"""
    
    # Create temporary file name manually (don't use context manager yet)
    import uuid
    tmp_filename = f"tmp_{layer}_{uuid.uuid4().hex[:8]}.geojson"
    
    # Register for cleanup
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
    
    with resource_monitor():
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # ogr2ogr command failed - layer might be empty or invalid
        logger.warning(f"ogr2ogr failed for layer {layer}")
        return None
    
    # Check if file was created and has content
    if not os.path.exists(tmp_filename):
        logger.warning(f"Output file not created for layer {layer}")
        return None
        
    file_size = os.path.getsize(tmp_filename)
    if file_size == 0:
        logger.warning(f"Empty output file for layer {layer}")
        return None
    
    # Return the filename (it will be cleaned up manually later)
    return tmp_filename

def process_feature_for_zoom_levels(feat: Dict[str, Any], config: Dict[str, Any], config_fields: Set[str], zoom_levels: List[int], db: 'FeatureDatabase', batch_features: List[Tuple[int, int, int, Dict[str, Any], int]]) -> int:
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
        
        # Use cached geometry simplification
        feature_geom = get_simplified_geometry_cached(geom, zoom)
        
        if feature_geom.is_empty or not feature_geom.is_valid:
            continue
        
        # Calculate tile bounds for this zoom level
        minx, miny, maxx, maxy = feature_geom.bounds
        xtile_min, ytile_min = deg2num(miny, minx, zoom)
        xtile_max, ytile_max = deg2num(maxy, maxx, zoom)
        
        # Store feature for each tile it intersects
        for xt in range(min(xtile_min, xtile_max), max(xtile_min, xtile_max) + 1):
            for yt in range(min(ytile_min, ytile_max), max(ytile_min, ytile_max) + 1):
                # Use optimized intersection with pre-filtering
                clipped_geom = optimized_geometry_intersection(feature_geom, xt, yt, zoom)
                
                if clipped_geom is not None:
                    # Create feature data dictionary
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
                        db.insert_features_batch(batch_features)
                        db.commit()
                        batch_features.clear()
                        gc.collect()
    
    return features_added

def process_layer_directly_to_database(pbf_file: str, layer: str, config: Dict[str, Any], db: 'FeatureDatabase', zoom_levels: List[int], config_fields: Set[str], LAYER_FIELDS: Dict[str, Set[str]]) -> int:
    """Process a single layer directly from PBF to database using Pyosmium"""
    logger.info(f"Processing layer {layer} directly with Pyosmium (no temporary files)")
    
    if not OSM_PYOSMIUM_AVAILABLE:
        logger.warning("Pyosmium not available, falling back to ogr2ogr with temporary files")
        return process_layer_streaming_from_pbf(pbf_file, layer, "", "", config, config_fields, zoom_levels, db)
    
    # Create Pyosmium handler for this specific layer
    handler = OSMFeatureHandler(config, db, zoom_levels, config_fields)
    
    # Filter handler to only process this layer type
    original_way = handler.way
    original_relation = handler.relation
    
    def filtered_way(w):
        if not w.tags:
            return
        
        # Convert tags to dict
        tags = {tag.k: tag.v for tag in w.tags}
        
        # Check if this way matches the layer
        layer_matches = False
        for pattern_key, pattern in handler.tag_patterns.items():
            if pattern['type'] == 'tag_value':
                if pattern['key'] in tags and tags[pattern['key']] == pattern['value']:
                    layer_matches = True
                    break
            elif pattern['type'] == 'tag_exists':
                if pattern['key'] in tags:
                    layer_matches = True
                    break
        
        if layer_matches:
            original_way(w)
    
    def filtered_relation(r):
        if not r.tags:
            return
        
        # Convert tags to dict
        tags = {tag.k: tag.v for tag in r.tags}
        
        # Check if this relation matches the layer
        layer_matches = False
        for pattern_key, pattern in handler.tag_patterns.items():
            if pattern['type'] == 'tag_value':
                if pattern['key'] in tags and tags[pattern['key']] == pattern['value']:
                    layer_matches = True
                    break
            elif pattern['type'] == 'tag_exists':
                if pattern['key'] in tags:
                    layer_matches = True
                    break
        
        if layer_matches:
            original_relation(r)
    
    # Apply filtered handlers
    handler.way = filtered_way
    handler.relation = filtered_relation
    
    # Process the file
    handler.apply_file(pbf_file)
    
    return handler.features_processed

def process_layer_streaming_from_pbf(pbf_file: str, layer: str, where_clause: str, select_fields: str, config: Dict[str, Any], config_fields: Set[str], zoom_levels: List[int], db: 'FeatureDatabase') -> int:
    """Process PBF layer using optimized temporary file approach"""
    # Use temporary file but with optimized processing
    tmp_filename = extract_layer_to_temp_file(pbf_file, layer, where_clause, select_fields)
    if not tmp_filename:
        logger.warning(f"No features found in layer {layer}")
        return 0
    
    try:
        features_processed = 0
        batch_features = []
        
        with open(tmp_filename, "r", encoding="utf-8", buffering=Config.GEOJSON_READ_BUFFER_SIZE) as f:
            # Use ijson for streaming JSON parsing
            for feat in ijson.items(f, "features.item"):
                features_added = process_feature_for_zoom_levels(
                    feat, config, config_fields, zoom_levels, db, batch_features
                )
                features_processed += features_added
                
                # Batch insert to avoid memory buildup
                if len(batch_features) >= Config.DB_BATCH_SIZE:
                    db.insert_features_batch(batch_features)
                    db.commit()
                    batch_features.clear()
                    gc.collect()
                
                # Periodic memory management
                if features_processed % 10000 == 0:
                    clear_geometry_caches()
                    clear_algorithm_caches()
        
        # Insert remaining features
        if batch_features:
            db.insert_features_batch(batch_features)
            db.commit()
        
        logger.info(f"Processed {features_processed} features from layer {layer}")
        return features_processed
        
    except Exception as e:
        logger.error(f"Error processing layer {layer}: {e}")
        return 0
    finally:
        # Clean up temporary file
        if tmp_filename and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
                _temp_files_to_cleanup.discard(tmp_filename)
                logger.debug(f"Cleaned up temporary file: {tmp_filename}")
            except Exception as e:
                logger.debug(f"Could not remove temporary file {tmp_filename}: {e}")

def process_pbf_directly_to_database(pbf_file: str, config: Dict[str, Any], db_path: str, zoom_levels: List[int], resources: Dict[str, Any] = None) -> int:
    """Process PBF directly to database using Pyosmium for maximum performance"""
    logger.info("Processing PBF directly to database using Pyosmium (maximum performance)")
    
    try:
        import osmium
    except ImportError:
        logger.error("Pyosmium not installed. Install with: pip install osmium")
        logger.info("Falling back to ogr2ogr method...")
        return process_pbf_directly_to_database_fallback(pbf_file, config, db_path, zoom_levels)
    
    with managed_database(db_path) as db:
        with memory_management():
            config_fields = get_config_fields(config)
            logger.info(f"Config requires these fields: {', '.join(sorted(config_fields))}")
            
            # Create Pyosmium handler
            handler = OSMFeatureHandler(config, db, zoom_levels, config_fields)
            
            # Process PBF file with Pyosmium
            logger.info("Starting Pyosmium processing...")
            start_time = time.time()
            
            try:
                handler.apply_file(pbf_file)
                processing_time = time.time() - start_time
                
                total_features_processed = handler.features_processed
                logger.info(f"Pyosmium processing completed in {processing_time:.2f}s")
                logger.info(f"Total processed: {total_features_processed} features directly from PBF")
            except Exception as e:
                logger.error(f"Error during PBF processing: {e}")
                raise
            
            # Print statistics for each zoom level with table
            table_data = []
            total_features = 0
            for zoom in sorted(zoom_levels):
                count = db.count_features_for_zoom(zoom)
                total_features += count
                table_data.append([zoom, f"{count:,}", f"Level {zoom}"])
            
            # Add total row
            table_data.append(["TOTAL", f"{total_features:,}", "All levels combined"])
            
            print(tabulate(table_data, 
                         headers=["Zoom", "Features", "Description"], 
                         tablefmt="grid", 
                         stralign="left", 
                         numalign="right"))
            
            return total_features_processed

class OSMFeatureHandler:
    """Pyosmium handler for processing OSM features directly from PBF files"""
    
    def __init__(self, config: Dict[str, Any], db: 'FeatureDatabase', zoom_levels: List[int], config_fields: Set[str]):
        self.config = config
        self.db = db
        self.zoom_levels = zoom_levels
        self.config_fields = config_fields
        self.features_processed = 0
        self.batch_features = []
        self.batch_size = 1000  # Process in batches for memory efficiency
        
        # Progress tracking
        self.total_features_scanned = 0
        self.total_features_filtered = 0
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.progress_interval = 1.0  # Show progress every 1 second
        
        # Pre-compile tag patterns for performance
        self.tag_patterns = self._compile_tag_patterns()
        
        # Initialize batch processing
        self._init_batch_processing()
    
    def _compile_tag_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Pre-compile tag patterns from config for fast matching"""
        patterns = {}
        
        for key, value in self.config.items():
            if isinstance(value, dict) and 'zoom' in value:
                # Parse tag pattern (e.g., "highway=motorway" or "building")
                if '=' in key:
                    tag_key, tag_value = key.split('=', 1)
                    patterns[key] = {
                        'type': 'tag_value',
                        'key': tag_key,
                        'value': tag_value,
                        'config': value
                    }
                else:
                    patterns[key] = {
                        'type': 'tag_exists',
                        'key': key,
                        'config': value
                    }
        
        logger.info(f"Compiled {len(patterns)} tag patterns for Pyosmium processing")
        return patterns
    
    def _init_batch_processing(self):
        """Initialize batch processing variables"""
        self.batch_features = []
        self.features_processed = 0
    
    def _update_progress(self):
        """Update and display progress information"""
        current_time = time.time()
        if current_time - self.last_progress_time >= self.progress_interval:
            elapsed_time = current_time - self.start_time
            features_per_second = self.total_features_scanned / elapsed_time if elapsed_time > 0 else 0
            
            print(f"\r📊 Progress: {self.total_features_scanned:,} scanned, {self.total_features_filtered:,} filtered, {features_per_second:.0f} features/s, {elapsed_time:.1f}s elapsed", end='', flush=True)
            self.last_progress_time = current_time
    
    def _should_process_feature(self, tags: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Check if feature should be processed based on tags and config"""
        for pattern_key, pattern in self.tag_patterns.items():
            if pattern['type'] == 'tag_value':
                if pattern['key'] in tags and tags[pattern['key']] == pattern['value']:
                    return pattern['config']
            elif pattern['type'] == 'tag_exists':
                if pattern['key'] in tags:
                    return pattern['config']
        
        return None
    
    def _process_feature_batch(self):
        """Process accumulated features in batch"""
        if self.batch_features:
            self.db.insert_features_batch(self.batch_features)
            self.db.commit()
            self.batch_features = []
    
    def _add_feature_to_batch(self, feature_data: Tuple[int, int, int, Dict[str, Any], int]):
        """Add feature to batch for processing"""
        self.batch_features.append(feature_data)
        
        # Process batch when it reaches batch_size
        if len(self.batch_features) >= self.batch_size:
            self._process_feature_batch()
    
    def way(self, w):
        """Process OSM way (lines, polygons)"""
        # Update progress counter
        self.total_features_scanned += 1
        
        if not w.tags:
            return
        
        # Convert tags to dict
        tags = {tag.k: tag.v for tag in w.tags}
        
        # Check if this way should be processed
        style_config = self._should_process_feature(tags)
        if not style_config:
            return
        
        # Update filtered counter
        self.total_features_filtered += 1
        
        # Update progress display
        self._update_progress()
        
        # Filter tags to only include config fields
        filtered_tags = {k: v for k, v in tags.items() if k in self.config_fields}
        
        # Get geometry from way
        try:
            # Convert way to geometry
            if len(w.nodes) < 2:
                return
            
            # Create geometry from way nodes
            coords = []
            for node in w.nodes:
                try:
                    # Check if node has valid location
                    if hasattr(node, 'location') and node.location.valid():
                        coords.append([node.location.lon, node.location.lat])
                    else:
                        # Skip this way if any node is invalid
                        if self.features_processed < 5:
                            logger.debug(f"❌ Way {w.id} has invalid node, skipping")
                        return
                except Exception as e:
                    # Skip this way if any node is invalid
                    if self.features_processed < 5:
                        logger.debug(f"❌ Way {w.id} node error: {e}, skipping")
                    return
            
            if len(coords) < 2:
                if self.features_processed < 5:
                    logger.debug(f"❌ Way {w.id} has insufficient coordinates: {len(coords)}")
                return
            
            # Determine if it's a polygon or line
            is_polygon = coords[0] == coords[-1] and len(coords) > 3
            
            
            if is_polygon:
                # Create polygon geometry
                from shapely.geometry import Polygon
                geom = Polygon(coords)
            else:
                # Create line geometry
                from shapely.geometry import LineString
                geom = LineString(coords)
            
            if not geom.is_valid or geom.is_empty:
                return
            
            # Process for all relevant zoom levels
            self._process_geometry_for_zoom_levels(geom, tags, style_config, w.id)
            
        except Exception as e:
            logger.debug(f"Error processing way {w.id}: {e}")
            return
    
    def relation(self, r):
        """Process OSM relation (multipolygons, etc.)"""
        # Update progress counter
        self.total_features_scanned += 1
        
        if not r.tags:
            return
        
        # Convert tags to dict
        tags = {tag.k: tag.v for tag in r.tags}
        
        # Check if this relation should be processed
        style_config = self._should_process_feature(tags)
        if not style_config:
            return
        
        # Update filtered counter
        self.total_features_filtered += 1
        
        # Update progress display
        self._update_progress()
        
        # Filter tags to only include config fields
        filtered_tags = {k: v for k, v in tags.items() if k in self.config_fields}
        
        # Process relation geometry (simplified for now)
        # In a full implementation, you'd need to handle multipolygons
        # For now, we'll skip complex relations
        logger.debug(f"Skipping complex relation {r.id}")
    
    def _process_geometry_for_zoom_levels(self, geom, tags: Dict[str, str], style_config: Dict[str, Any], osm_id: int):
        """Process geometry for all relevant zoom levels"""
        zoom_filter = style_config.get("zoom", 6)
        priority = style_config.get("priority", 5)
        hex_color = style_config.get("color", "#FFFFFF")
        color = hex_to_rgb332_direct(hex_color)
        
        for zoom in self.zoom_levels:
            if zoom < zoom_filter:
                continue
            
            # Simplify geometry for zoom level
            tolerance = get_simplify_tolerance_for_zoom(zoom)
            simplified_geom = geom.simplify(tolerance, preserve_topology=True)
            
            if simplified_geom.is_empty or not simplified_geom.is_valid:
                continue
            
            # Calculate tile bounds for this zoom level
            minx, miny, maxx, maxy = simplified_geom.bounds
            xtile_min, ytile_min = deg2num(float(miny), float(minx), zoom)
            xtile_max, ytile_max = deg2num(float(maxy), float(maxx), zoom)
            
            # Store feature for each tile it intersects
            for xt in range(min(xtile_min, xtile_max), max(xtile_min, xtile_max) + 1):
                for yt in range(min(ytile_min, ytile_max), max(ytile_min, ytile_max) + 1):
                    # Use optimized intersection with pre-filtering
                    # Convert tile coordinates to lat/lon bounds
                    n = 2.0 ** zoom
                    lon_min = xt / n * 360.0 - 180.0
                    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yt / n))))
                    lon_max = (xt + 1) / n * 360.0 - 180.0
                    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (yt + 1) / n))))
                    
                    clipped_geom = simplified_geom.intersection(
                        box(lon_min, lat_min, lon_max, lat_max)
                    )
                    
                    if clipped_geom.is_empty or not clipped_geom.is_valid:
                        continue
                    
                    # Create feature data
                    feature_data = {
                        'geom': clipped_geom,
                        'color': color,
                        'color_hex': hex_color,
                        'tags': tags,
                        'priority': priority
                    }
                    
                    # Add to batch
                    self._add_feature_to_batch((zoom, xt, yt, feature_data, priority))
                    self.features_processed += 1
    
    def apply_file(self, pbf_file: str):
        """Apply the handler to a PBF file"""
        if not OSM_PYOSMIUM_AVAILABLE:
            raise ImportError("Pyosmium not available")
        
        # Create a SimpleHandler and use it to process the file
        class PyosmiumHandler(osmium.SimpleHandler):
            def __init__(self, parent_handler):
                super().__init__()
                self.parent = parent_handler
            
            def way(self, w):
                self.parent.way(w)
            
            def relation(self, r):
                self.parent.relation(r)
        
        # Create and use the handler
        handler = PyosmiumHandler(self)
        
        # Try to process with node locations first
        try:
            # Use apply_file with locations=True to load node coordinates
            handler.apply_file(pbf_file, locations=True)
        except Exception as e:
            logger.warning(f"Failed to process with locations=True: {e}")
            # Fallback to regular processing
            handler.apply_file(pbf_file)
        
        # Process any remaining features in batch
        self._process_feature_batch()
        
        # Final progress message with 100% completion
        elapsed_time = time.time() - self.start_time
        features_per_second = self.total_features_scanned / elapsed_time if elapsed_time > 0 else 0
        print(f"\r📊 Progress: {self.total_features_scanned:,} scanned, {self.total_features_filtered:,} filtered (100.0%), {features_per_second:.0f} features/s, {elapsed_time:.1f}s elapsed", end='', flush=True)
        print()  # New line after final progress

def process_pbf_directly_to_database_fallback(pbf_file: str, config: Dict[str, Any], db_path: str, zoom_levels: List[int]) -> int:
    """Fallback method using ogr2ogr (original implementation)"""
    logger.info("Processing PBF directly to database (fallback with minimal temporary files)")
    
    # First, check what layers are available in the PBF file
    available_layers = check_pbf_layers(pbf_file)
    if not available_layers:
        logger.error("No layers found in PBF file")
        return 0
    
    with managed_database(db_path) as db:
        with memory_management():
            config_fields = get_config_fields(config)
            
            logger.info(f"Config requires these fields: {', '.join(sorted(config_fields))}")
            
            total_features_processed = 0
            
            # Process each layer separately and immediately clean up
            for i, layer in enumerate(Config.PROCESSING_LAYERS):
                if layer not in available_layers:
                    logger.warning(f"Layer {layer} not found in PBF file, skipping")
                    continue
                    
                logger.info(f"[{i+1}/{len(Config.PROCESSING_LAYERS)}] Processing layer: {layer} directly to database")
                
                with resource_monitor():
                    layer_features = process_layer_directly_to_database(
                        pbf_file, layer, config, db, zoom_levels, config_fields, Config.LAYER_FIELDS
                    )
                
                total_features_processed += layer_features
                logger.info(f"Layer {layer}: {layer_features} features processed")
                
                # Memory management between layers
                with memory_management():
                    pass
            
            logger.info(f"Total processed: {total_features_processed} features directly from PBF")
            
            # Print statistics for each zoom level with table
            table_data = []
            total_features = 0
            for zoom in sorted(zoom_levels):
                count = db.count_features_for_zoom(zoom)
                total_features += count
                table_data.append([zoom, f"{count:,}", f"Level {zoom}"])
            
            # Add total row
            table_data.append(["TOTAL", f"{total_features:,}", "All levels combined"])
            
            print(tabulate(table_data, 
                         headers=["Zoom", "Features", "Description"], 
                         tablefmt="grid", 
                         stralign="left", 
                         numalign="right"))
            
            return total_features_processed


def detect_simple_shapes(geometry, pixel_coords: List[Tuple[int, int]]) -> Optional[Dict]:
    """Detect simple shapes that can use optimized commands"""
    if len(pixel_coords) < 3:
        return None
    
    # Detect rectangle
    if len(pixel_coords) == 4:
        px_coords = [p[0] for p in pixel_coords]
        py_coords = [p[1] for p in pixel_coords]
        min_x, max_x = min(px_coords), max(px_coords)
        min_y, max_y = min(py_coords), max(py_coords)
        
        # Check if it's a perfect rectangle
        if (min_x, min_y) in pixel_coords and (max_x, min_y) in pixel_coords and \
           (max_x, max_y) in pixel_coords and (min_x, max_y) in pixel_coords:
            return {
                'type': DRAW_COMMANDS['SIMPLE_RECTANGLE'],
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            }
    
    # Detect triangle
    if len(pixel_coords) == 3:
        return {
            'type': DRAW_COMMANDS['SIMPLE_TRIANGLE'],
            'x1': pixel_coords[0][0],
            'y1': pixel_coords[0][1],
            'x2': pixel_coords[1][0],
            'y2': pixel_coords[1][1],
            'x3': pixel_coords[2][0],
            'y3': pixel_coords[2][1]
        }
    
    # Detect approximate circle
    if len(pixel_coords) >= 8:  # Circle needs at least 8 points
        px_coords = [p[0] for p in pixel_coords]
        py_coords = [p[1] for p in pixel_coords]
        center_x = sum(px_coords) // len(px_coords)
        center_y = sum(py_coords) // len(py_coords)
        
        # Calculate average radius
        distances = [((x - center_x)**2 + (y - center_y)**2)**0.5 for x, y in pixel_coords]
        avg_radius = sum(distances) / len(distances)
        
        # Check if it's approximately circular (radius variation < 20%)
        radius_variation = max(distances) - min(distances)
        if radius_variation < avg_radius * 0.2:
            return {
                'type': DRAW_COMMANDS['SIMPLE_CIRCLE'],
                'center_x': center_x,
                'center_y': center_y,
                'radius': int(avg_radius)
            }
    
    return None


def get_layer_from_priority(tags: Dict[str, str]) -> int:
    """Get layer based on priority from features.json"""
    # If no configuration is loaded, use simple logic
    if not hasattr(Config, 'FEATURES_CONFIG') or not Config.FEATURES_CONFIG:
        # Simple logic based on feature type
        if 'building' in tags:
            return RENDER_LAYERS['BUILDINGS']
        elif 'highway' in tags:
            return RENDER_LAYERS['ROADS']
        elif 'natural' in tags or 'water' in tags:
            return RENDER_LAYERS['WATER']
        else:
            return RENDER_LAYERS['TERRAIN']
    
    # Buscar coincidencia en features.json
    for key, value in tags.items():
        keyval = f"{key}={value}"
        if keyval in Config.FEATURES_CONFIG:
            priority = Config.FEATURES_CONFIG[keyval].get('priority', 5)
            # Mapear prioridad a capa basado en el análisis del features.json
            if priority <= 2:  # Terreno y agua de fondo
                return RENDER_LAYERS['TERRAIN']
            elif priority <= 6:  # Agua y elementos naturales
                return RENDER_LAYERS['WATER']
            elif priority <= 9:  # Edificios y estructuras
                return RENDER_LAYERS['BUILDINGS']
            elif priority <= 25:  # Carreteras y transporte
                return RENDER_LAYERS['ROADS']
            else:  # Etiquetas y elementos de texto
                return RENDER_LAYERS['LABELS']
        
        # Buscar solo por key
        if key in Config.FEATURES_CONFIG:
            priority = Config.FEATURES_CONFIG[key].get('priority', 5)
            if priority <= 2:
                return RENDER_LAYERS['TERRAIN']
            elif priority <= 6:
                return RENDER_LAYERS['WATER']
            elif priority <= 9:
                return RENDER_LAYERS['BUILDINGS']
            elif priority <= 25:
                return RENDER_LAYERS['ROADS']
            else:
                return RENDER_LAYERS['LABELS']
    
    # Fallback
    return RENDER_LAYERS['TERRAIN']

# Initialize configuration
try:
    Config.FEATURES_CONFIG = json.load(open('features.json', 'r'))
except FileNotFoundError:
    logger.warning("features.json not found, using default configuration")
    Config.FEATURES_CONFIG = {}

def process_geometry_batch(geometries: List[Dict], zoom: int, tile_x: int, tile_y: int, simplify_tolerance: float) -> List[Dict]:
    """
    Process a batch of geometries efficiently, maintaining layer order and priorities.
    
    Args:
        geometries: List of geometry dictionaries with 'geom', 'color', 'tags', 'priority', 'color_hex'
        zoom: Zoom level
        tile_x: Tile X coordinate
        tile_y: Tile Y coordinate
        simplify_tolerance: Simplify tolerance for zoom level
        
    Returns:
        List of draw commands
    """
    batch_commands = []
    
    # Group geometries by type for efficient processing
    polygons = []
    linestrings = []
    other_geoms = []
    
    for geom_data in geometries:
        geom = geom_data['geom']
        if geom.geom_type == "Polygon":
            polygons.append(geom_data)
        elif geom.geom_type == "LineString":
            linestrings.append(geom_data)
        else:
            other_geoms.append(geom_data)
    
    # Process polygons in batch (most common and expensive)
    if polygons:
        polygon_commands = process_polygon_batch(polygons, zoom, tile_x, tile_y, simplify_tolerance)
        batch_commands.extend(polygon_commands)
    
    # Process linestrings in batch
    if linestrings:
        linestring_commands = process_linestring_batch(linestrings, zoom, tile_x, tile_y, simplify_tolerance)
        batch_commands.extend(linestring_commands)
    
    # Process other geometries individually (less common)
    for geom_data in other_geoms:
        cmds = geometry_to_draw_commands(
            geom_data["geom"], geom_data["color"], geom_data["tags"], zoom, tile_x, tile_y,
            simplify_tolerance=simplify_tolerance, hex_color=geom_data.get("color_hex")
        )
        batch_commands.extend(cmds)
    
    return batch_commands

def process_polygon_batch(polygons: List[Dict], zoom: int, tile_x: int, tile_y: int, simplify_tolerance: float) -> List[Dict]:
    """Process a batch of polygons efficiently"""
    commands = []
    
    for geom_data in polygons:
        geom = geom_data['geom']
        color = geom_data['color']
        hex_color = geom_data.get('color_hex')
        
        if geom.is_empty:
            continue
            
        # Process exterior ring
        exterior = remove_duplicate_points(list(geom.exterior.coords))
        exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
        exterior_pixels = ensure_closed_ring(exterior_pixels)
        
        if len(set(exterior_pixels)) >= 3:
            # Try to detect simple shapes first
            simple_shape = detect_simple_shapes(geom, exterior_pixels)
            if simple_shape:
                # Use optimized command for simple shape
                cmd = simple_shape.copy()
                cmd['color'] = color
                if hex_color:
                    cmd['color_hex'] = hex_color
                commands.append(cmd)
            else:
                # Use STROKE_POLYGON for polygons without fill (contour only)
                cmd = {'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color}
                if hex_color:
                    cmd['color_hex'] = hex_color
                commands.append(cmd)
    
    return commands

def process_linestring_batch(linestrings: List[Dict], zoom: int, tile_x: int, tile_y: int, simplify_tolerance: float) -> List[Dict]:
    """Process a batch of linestrings efficiently"""
    commands = []
    
    for geom_data in linestrings:
        geom = geom_data['geom']
        color = geom_data['color']
        hex_color = geom_data.get('color_hex')
        
        if geom.is_empty:
            continue
            
        coords = remove_duplicate_points(list(geom.coords))
        pixel_coords = coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y)
        
        if len(pixel_coords) >= 2:
            # Try to detect simple lines (horizontal/vertical)
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
                commands.append(cmd)
            else:
                cmd = {'type': DRAW_COMMANDS['POLYLINE'], 'points': pixel_coords, 'color': color}
                if hex_color:
                    cmd['color_hex'] = hex_color
                commands.append(cmd)
    
    return commands

def tile_worker(args):
    start_time = time.time()
    x, y, feats, zoom, output_dir, max_file_size, simplify_tolerance = args

    # Group features by layer for correct rendering order
    features_by_layer = {}
    for feat in feats:
        layer = get_layer_from_priority(feat.get("tags", {}))
        if layer not in features_by_layer:
            features_by_layer[layer] = []
        features_by_layer[layer].append(feat)
    
    # Sort features within each layer by priority
    for layer in features_by_layer:
        features_by_layer[layer].sort(key=lambda f: f.get("priority", 5))

    all_commands = []
    
    # Process layers in rendering order (0=TERRAIN, 1=WATER, 2=BUILDINGS, etc.)
    for layer in sorted(features_by_layer.keys()):
        layer_features = features_by_layer[layer]
        
        # Process layer as a batch for better performance
        layer_commands = process_geometry_batch(layer_features, zoom, x, y, simplify_tolerance)
        all_commands.extend(layer_commands)

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

    # Use optimized file write
    optimized_file_write(filename, buffer)

    tile_size = len(buffer)
    
    # Clean up memory pool periodically
    if tile_size % 100 == 0:  # Every 100 tiles
        memory_pool.point_pool.clear()
        memory_pool.command_pool.clear()
        memory_pool.coord_pool.clear()
        memory_pool.feature_pool.clear()
    
    del all_commands, optimized_commands, buffer, feats
    gc.collect()
    
    elapsed = time.time() - start_time
    return tile_size, elapsed

def write_tile_batch(batch, output_dir, zoom, max_file_size, simplify_tolerance, resources=None):
    """Write a batch of tiles with simple parallel processing"""
    tile_sizes = []
    
    # Monitor memory and get optimal settings
    optimal_workers, optimal_batch_size, memory_pressure = monitor_memory_and_adjust(resources)
    
    # Use simple worker count based on memory pressure
    batch_workers = min(optimal_workers, len(batch), Config.MAX_WORKERS)
    batch_workers = max(1, batch_workers)
    
    # Use debug level to avoid cluttering the progress bar
    logger.debug(f"Processing batch of {len(batch)} tiles with {batch_workers} workers "
                f"(memory pressure: {memory_pressure})")
    
    # Use ProcessPoolExecutor for parallelization
    with ProcessPoolExecutor(max_workers=batch_workers) as executor:
        futures = [executor.submit(tile_worker, job) for job in batch]
        for i, future in enumerate(as_completed(futures)):
            tile_size, _ = future.result()
            tile_sizes.append(tile_size)
            
            # Simple garbage collection
            if i % 10 == 0:
                gc.collect()
    
    return tile_sizes

def generate_tiles_from_database(db_path: str, output_dir: str, zoom: int, max_file_size: int = 65536, resources: Dict[str, Any] = None) -> dict:
    """Generate tiles for a specific zoom level from the database"""
    logger.info(f"Processing zoom level {zoom} from database")
    
    with managed_database(db_path) as db:
        with memory_management():
            # Get tile count for progress tracking
            tile_count = len(db.get_tiles_for_zoom(zoom))
            if tile_count == 0:
                logger.warning(f"No tiles found for zoom {zoom}")
                return {'tiles': 0, 'avg_size': 0}
            
            logger.info(f"Found {tile_count} tiles for zoom {zoom}")
            
            # Process tiles in streaming batches to reduce memory usage
            all_tile_sizes = []
            processed_tiles = 0
            
            # Create batches of tiles for processing
            tile_batch_size = resources.get('tile_batch_size', 1000) if resources else 1000
            tile_batches = []
            current_batch = []
            
            # Stream tiles and create batches
            for (tile_x, tile_y) in db.stream_tiles_for_zoom(zoom):
                current_batch.append((tile_x, tile_y))
                if len(current_batch) >= tile_batch_size:
                    tile_batches.append(current_batch)
                    current_batch = []
            
            # Add remaining tiles
            if current_batch:
                tile_batches.append(current_batch)
            
            # Process tiles in streaming batches
            with tqdm(total=tile_count, desc=f"Writing tiles (zoom {zoom})") as pbar:
                for batch_idx, tile_batch in enumerate(tile_batches):
                    # Process each batch of tiles
                    batch_jobs = []
                    for (tile_x, tile_y) in tile_batch:
                        # Stream features for this tile
                        features = list(db.stream_features_for_tile(zoom, tile_x, tile_y))
                        if features:
                            batch_jobs.append((tile_x, tile_y, features, zoom, output_dir, max_file_size, None))
                    
                    if batch_jobs:
                        # Write this batch
                        with resource_monitor():
                            batch_sizes = write_tile_batch(batch_jobs, output_dir, zoom, max_file_size, None, resources)
                        all_tile_sizes.extend(batch_sizes)
                        processed_tiles += len(batch_jobs)
                    
                    pbar.update(len(tile_batch))
                    
                    # Monitor memory after each batch
                    if batch_idx % 5 == 0:  # Check every 5 batches
                        memory_pressure = get_memory_pressure()
                        if memory_pressure in ['high', 'critical']:
                            logger.warning(f"High memory pressure detected: {memory_pressure}, "
                                         f"forcing cleanup after batch {batch_idx}")
                            force_memory_cleanup()
                    
                    # Memory management between batches
                    with memory_management():
                        pass
            
            avg_tile_size = sum(all_tile_sizes) / len(all_tile_sizes) if all_tile_sizes else 0
            # Return statistics
            return {'tiles': len(all_tile_sizes), 'avg_size': avg_tile_size}

def precompute_global_color_palette(config: Dict[str, Any]) -> int:
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
    
    # Write palette in the format expected by tile_viewer.py:
    # 4 bytes: number of colors (little-endian)
    # 3 bytes per color: RGB888 values
    num_colors = len(GLOBAL_INDEX_TO_RGB332)
    
    # Create palette data
    palette_data = bytearray()
    palette_data.extend(struct.pack('<I', num_colors))  # Header: number of colors
    
    # Write RGB888 colors in index order
    for index in range(num_colors):
        if index in GLOBAL_INDEX_TO_RGB332:
            rgb332 = GLOBAL_INDEX_TO_RGB332[index]
            # Convert RGB332 to RGB888 (proper bit expansion)
            r = (rgb332 & 0xE0) | ((rgb332 & 0xE0) >> 3) | ((rgb332 & 0xE0) >> 6)  # Expand 3 bits to 8
            g = ((rgb332 & 0x1C) << 3) | ((rgb332 & 0x1C) >> 2) | ((rgb332 & 0x1C) << 1)  # Expand 3 bits to 8
            b = ((rgb332 & 0x03) << 6) | ((rgb332 & 0x03) << 4) | ((rgb332 & 0x03) << 2) | (rgb332 & 0x03)  # Expand 2 bits to 8
            palette_data.extend([r, g, b])
        else:
            # Default color if index not found
            palette_data.extend([255, 255, 255])
    
    optimized_file_write(palette_path, bytes(palette_data))
    
    logger.info("Palette written successfully")

def main() -> None:
    """
    Main entry point for the OSM vector tile generator.
    
    Processes OpenStreetMap PBF files directly to vector tiles using a database
    for efficient feature storage and retrieval. Supports multiple zoom levels
    and configurable styling through JSON configuration files.
    
    Command line arguments:
        pbf_file: Path to input OSM PBF file
        output_dir: Directory for generated vector tiles
        config_file: JSON file with feature styling configuration
        --zoom: Zoom level or range (e.g., "12" or "6-17")
        --max-file-size: Maximum tile file size in KB
        --db-path: Path for temporary SQLite database
        
    Example:
        python tile_generator_direct.py data.osm.pbf tiles/ features.json --zoom 6-17
    """
    script_start_time = time.time()
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
        
        # Display initial system information with dynamic resource allocation
        memory_info = get_memory_usage()
        
        # Calculate dynamic resources based on 70% of total system memory
        resources = calculate_dynamic_resources()
        max_workers = resources['max_workers']
        db_batch_size = resources['db_batch_size']
        tile_batch_size = resources['tile_batch_size']
        worker_memory_limit = resources['worker_memory_limit']
        
        logger.info(f"System information:")
        logger.info(f"  CPU cores: {max_workers}")
        logger.info(f"  Max workers: {max_workers}")
        logger.info(f"  DB batch size: {db_batch_size:,}")
        logger.info(f"  Tile batch size: {tile_batch_size:,}")
        logger.info(f"  Total memory: {resources['total_memory_mb']:.1f}MB")
        logger.info(f"  Allocated memory: {resources['available_memory_mb']:.1f}MB (70% of total)")
        logger.info(f"  Worker memory limit: {worker_memory_limit:.1f}MB per worker")
        logger.info(f"  Memory pressure: {get_memory_pressure()}")
        
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
    process_pbf_directly_to_database(args.pbf_file, config, args.db_path, zoom_levels, resources)

    # Generate tiles for each zoom level from database and collect statistics
    all_zoom_stats = {}
    for zoom in zoom_levels:
        stats = generate_tiles_from_database(args.db_path, args.output_dir, zoom, max_file_size, resources)
        all_zoom_stats[zoom] = stats
        gc.collect()
    
    # Print final statistics table
    table_data = []
    total_tiles = 0
    for zoom in sorted(zoom_levels):
        stats = all_zoom_stats.get(zoom, {'tiles': 0, 'avg_size': 0})
        tiles = stats['tiles']
        avg_size = stats['avg_size']
        total_tiles += tiles
        
        if tiles > 0:
            table_data.append([zoom, f"{tiles:,}", f"{avg_size:.2f}"])
        else:
            table_data.append([zoom, "0", "N/A"])
    
    # Add total row
    table_data.append(["TOTAL", f"{total_tiles:,}", "-"])
    
    print(tabulate(table_data, 
                 headers=["Zoom", "Tiles", "Avg Size (bytes)"], 
                 tablefmt="grid", 
                 stralign="left", 
                 numalign="right"))
    
    logger.info("Process completed successfully")
    
    # Clean up temporary files and database
    logger.info("Cleaning up temporary files and database...")
    cleanup_all()
    
    # Clear all caches to free memory
    clear_all_caches()
    
    del config
    smart_gc_collect()
    
    # Calculate and display total execution time
    total_elapsed = time.time() - script_start_time
    days = int(total_elapsed // 86400)
    hours = int((total_elapsed % 86400) // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    
    print(f"\n⏱️  Total execution time: {days:02d}d {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Script completed in {total_elapsed:.2f} seconds ({days:02d}d {hours:02d}:{minutes:02d}:{seconds:02d})")

if __name__ == "__main__":
    main()
