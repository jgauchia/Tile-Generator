import struct
import sys
import os
import math
import pygame
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from functools import lru_cache

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'tile_size': 256,
    'viewport_size': 768,
    'toolbar_width': 160,
    'statusbar_height': 40,
    'max_cache_size': 1000,
    'thread_pool_size': 4,
    'background_colors': [(0, 0, 0), (255, 255, 255)],
    'log_level': 'INFO',
    'config_file': 'features.json',
    'fps_limit': 30
}

class Config:
    """Configuration management class"""
    def __init__(self, config_file=None):
        self.config = DEFAULT_CONFIG.copy()
        self.config_file = config_file or DEFAULT_CONFIG['config_file']
        self.load_config()
    
    def load_config(self):
        """Load configuration from file if it exists"""
        try:
            if os.path.exists(self.config_file):
                import json
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge file config with defaults
                self.config.update(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.info(f"Config file {self.config_file} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading config file {self.config_file}: {e}")
            logger.info("Using default configuration")
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            import json
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return tile_cache.get_stats()

# Global configuration instance
config = Config()

# Initialize constants from configuration
TILE_SIZE = config.get('tile_size', 256)
VIEWPORT_SIZE = config.get('viewport_size', 768)
TOOLBAR_WIDTH = config.get('toolbar_width', 160)
STATUSBAR_HEIGHT = config.get('statusbar_height', 40)
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

DRAW_COMMANDS = {
    1: "LINE",
    2: "POLYLINE",
    3: "STROKE_POLYGON",
    5: "HORIZONTAL_LINE",
    6: "VERTICAL_LINE",
    0x80: "SET_COLOR",        # Original command (direct RGB332)
    0x81: "SET_COLOR_INDEX",  # Palette command (index)
    # Feature-specific optimized commands
    0x82: "RECTANGLE",        # Optimized rectangle for buildings
    0x83: "STRAIGHT_LINE",    # Optimized straight line for highways
    0x84: "HIGHWAY_SEGMENT",  # Highway segment with continuity
    # Advanced compression commands
    0x85: "GRID_PATTERN",     # Urban grid pattern
    0x86: "BLOCK_PATTERN",    # City block pattern
    0x87: "CIRCLE",           # Circle/roundabout
    0x88: "RELATIVE_MOVE",    # Relative coordinate movement
    0x89: "PREDICTED_LINE",   # Predictive line based on pattern
    0x8A: "COMPRESSED_POLYLINE",  # Huffman-compressed polyline
}

UINT16_TILE_SIZE = 65536

# Global variables for palette (can be loaded dynamically)
GLOBAL_PALETTE = {}  # Loaded from configuration file or deduced

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TileCache:
    """LRU Cache for tile surfaces to limit memory usage"""
    def __init__(self, max_size=None):
        self.cache = OrderedDict()
        self.max_size = max_size or config.get('max_cache_size', 1000)
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get tile surface from cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        """Put tile surface in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                removed_key, removed_value = self.cache.popitem(last=False)
                logger.debug(f"Evicted tile from cache: {removed_key}")
        self.cache[key] = value
    
    def clear(self):
        """Clear all cached surfaces"""
        self.cache.clear()
        logger.info("Tile cache cleared")
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# Global tile cache instance
tile_cache = TileCache()

class TileLoader:
    """Persistent thread pool for tile loading operations"""
    def __init__(self, max_workers=None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers or config.get('thread_pool_size', 4))
        self.active_futures = set()
    
    def submit_tile_load(self, tile_info, callback=None):
        """Submit a tile loading task"""
        future = self.executor.submit(self._load_single_tile, tile_info)
        if callback:
            future.add_done_callback(callback)
        self.active_futures.add(future)
        return future
    
    def _load_single_tile(self, tile_info):
        """Load a single tile (internal method) with error recovery"""
        try:
            x, y, zoom_level, directory, bg_color, fill_mode = tile_info
            key = (zoom_level, x, y, bg_color, fill_mode)
            
            # Check cache first
            cached_surface = tile_cache.get(key)
            if cached_surface is not None:
                return key, cached_surface
            
            tile_file = get_tile_file(directory, x, y)
            if not tile_file:
                logger.warning(f"No tile file found for ({x}, {y}) in {directory}")
                return None, None
            
            # Validate tile file exists
            if not os.path.exists(tile_file):
                logger.warning(f"Tile file does not exist: {tile_file}")
                return None, None
            
            surface = render_tile_surface({'x': x, 'y': y, 'file': tile_file}, bg_color, fill_mode)
            if surface is None:
                logger.error(f"Failed to render tile surface for {tile_file}")
                return None, None
                
            tile_cache.put(key, surface)
            return key, surface
            
        except Exception as e:
            logger.error(f"Error loading tile {tile_info}: {e}")
            return None, None
    
    def cleanup_completed_futures(self):
        """Remove completed futures from active set"""
        completed = [f for f in self.active_futures if f.done()]
        for future in completed:
            self.active_futures.discard(future)
        return len(completed)
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)
        logger.info("Tile loader thread pool shutdown")

# Global tile loader instance
tile_loader = TileLoader()

def load_global_palette_from_config(config_file):
    """
    Loads the global palette from the configuration file.
    This allows the viewer to know the palette used by the generator.
    """
    global GLOBAL_PALETTE
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract unique colors from JSON (same algorithm as generator)
        unique_colors = set()
        for feature_key, feature_config in config.items():
            if isinstance(feature_config, dict) and 'color' in feature_config:
                hex_color = feature_config['color']
                if hex_color and isinstance(hex_color, str) and hex_color.startswith("#"):
                    unique_colors.add(hex_color)
        
        sorted_colors = sorted(list(unique_colors))
        
        # Create palette: index -> RGB888
        GLOBAL_PALETTE = {}
        for index, hex_color in enumerate(sorted_colors):
            rgb888 = hex_to_rgb888(hex_color)
            GLOBAL_PALETTE[index] = rgb888
        
        logger.info(f"Loaded dynamic palette: {len(GLOBAL_PALETTE)} colors")
        return True
    except Exception as e:
        logger.warning(f"Could not load palette from config: {e}")
        logger.info("Will use fallback palette if needed")
        return False

def hex_to_rgb888(hex_color):
    """Converts hex color to RGB888"""
    try:
        if not hex_color or not hex_color.startswith("#"):
            return (255, 255, 255)
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return (r, g, b)
    except:
        return (255, 255, 255)

# Functions removed - not used in current implementation

def darken_color(rgb, amount=0.3):
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)

@lru_cache(maxsize=1000)
def uint16_to_tile_pixel(val):
    """Convert uint16 coordinate to tile pixel with caching"""
    return int(round(val * (TILE_SIZE - 1) / (UINT16_TILE_SIZE - 1)))

def get_button_icons():
    """Create beautiful, modern button icons"""
    
    # Background toggle icon (sun/moon)
    icon_surface_bg = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_bg.fill((0, 0, 0, 0))
    # Sun rays
    for i in range(8):
        angle = i * 45
        x1 = 12 + int(8 * math.cos(math.radians(angle)))
        y1 = 12 + int(8 * math.sin(math.radians(angle)))
        x2 = 12 + int(10 * math.cos(math.radians(angle)))
        y2 = 12 + int(10 * math.sin(math.radians(angle)))
        pygame.draw.line(icon_surface_bg, (255, 255, 0), (x1, y1), (x2, y2), 2)
    # Sun center
    pygame.draw.circle(icon_surface_bg, (255, 255, 0), (12, 12), 6, 0)
    pygame.draw.circle(icon_surface_bg, (255, 200, 0), (12, 12), 4, 0)
    
    # Tile labels icon (document with lines)
    icon_surface_label = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_label.fill((0, 0, 0, 0))
    # Document background
    pygame.draw.rect(icon_surface_label, (255, 255, 255), (6, 4, 12, 16), 0)
    pygame.draw.rect(icon_surface_label, (200, 200, 200), (6, 4, 12, 16), 2)
    # Document fold
    pygame.draw.polygon(icon_surface_label, (240, 240, 240), [(18, 4), (18, 8), (14, 8)], 0)
    pygame.draw.polygon(icon_surface_label, (180, 180, 180), [(18, 4), (18, 8), (14, 8)], 1)
    # Text lines
    pygame.draw.line(icon_surface_label, (100, 100, 100), (8, 8), (16, 8), 1)
    pygame.draw.line(icon_surface_label, (100, 100, 100), (8, 10), (14, 10), 1)
    pygame.draw.line(icon_surface_label, (100, 100, 100), (8, 12), (16, 12), 1)
    pygame.draw.line(icon_surface_label, (100, 100, 100), (8, 14), (12, 14), 1)
    
    # GPS cursor icon (crosshair with target)
    icon_surface_gps = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_gps.fill((0, 0, 0, 0))
    # Outer circle
    pygame.draw.circle(icon_surface_gps, (0, 255, 0), (12, 12), 10, 2)
    # Inner circle
    pygame.draw.circle(icon_surface_gps, (0, 255, 0), (12, 12), 6, 2)
    # Crosshair lines
    pygame.draw.line(icon_surface_gps, (0, 255, 0), (12, 2), (12, 6), 2)
    pygame.draw.line(icon_surface_gps, (0, 255, 0), (12, 18), (12, 22), 2)
    pygame.draw.line(icon_surface_gps, (0, 255, 0), (2, 12), (6, 12), 2)
    pygame.draw.line(icon_surface_gps, (0, 255, 0), (18, 12), (22, 12), 2)
    # Center dot
    pygame.draw.circle(icon_surface_gps, (0, 255, 0), (12, 12), 2, 0)
    
    # Fill polygons icon (paint bucket)
    icon_surface_fill = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_fill.fill((0, 0, 0, 0))
    # Paint bucket body
    pygame.draw.rect(icon_surface_fill, (255, 100, 100), (8, 6, 8, 10), 0)
    pygame.draw.rect(icon_surface_fill, (200, 80, 80), (8, 6, 8, 10), 1)
    # Paint bucket spout
    pygame.draw.polygon(icon_surface_fill, (255, 100, 100), [(10, 16), (14, 16), (12, 20)], 0)
    pygame.draw.polygon(icon_surface_fill, (200, 80, 80), [(10, 16), (14, 16), (12, 20)], 1)
    # Paint drops
    pygame.draw.circle(icon_surface_fill, (255, 100, 100), (6, 18), 1, 0)
    pygame.draw.circle(icon_surface_fill, (255, 100, 100), (18, 20), 1, 0)
    pygame.draw.circle(icon_surface_fill, (255, 100, 100), (4, 22), 1, 0)
    
    return icon_surface_bg, icon_surface_label, icon_surface_gps, icon_surface_fill

def index_available_tiles(directory, progress_callback=None):
    available_tiles = set()
    if not os.path.isdir(directory):
        logger.error(f"Directory does not exist: {directory}")
        return available_tiles
    x_dirs = [x_str for x_str in os.listdir(directory) if os.path.isdir(os.path.join(directory, x_str))]
    total_x = len(x_dirs)
    def index_xdir(x_str):
        x_path = os.path.join(directory, x_str)
        try:
            x = int(x_str)
        except:
            return []
        files = os.listdir(x_path)
        y_dict = {}
        for fname in files:
            if fname.endswith('.bin') or fname.endswith('.png'):
                y_str = fname.split('.')[0]
                if y_str.isdigit():
                    y = int(y_str)
                    if y not in y_dict or fname.endswith('.bin'):
                        y_dict[y] = fname
        return [(x, y) for y in y_dict]
    results = []
    with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as pool:
        futures = {pool.submit(index_xdir, x_str): i_x for i_x, x_str in enumerate(x_dirs)}
        for i, future in enumerate(as_completed(futures)):
            tiles = future.result()
            results.extend(tiles)
            if progress_callback is not None:
                percent = (i + 1) / max(total_x, 1)
                progress_callback(percent, "Indexing tiles...")
    available_tiles.update(results)
    return available_tiles

def get_tile_file(directory, x, y):
    """Get tile file path with validation"""
    bin_path = f"{directory}/{x}/{y}.bin"
    png_path = f"{directory}/{x}/{y}.png"
    if os.path.isfile(bin_path):
        return bin_path
    elif os.path.isfile(png_path):
        return png_path
    return None

def is_tile_visible(x, y, viewport_x, viewport_y):
    """Check if tile is visible in current viewport"""
    tile_px = x * TILE_SIZE - viewport_x
    tile_py = y * TILE_SIZE - viewport_y
    
    # Check if tile intersects with viewport
    return not (tile_px + TILE_SIZE < 0 or tile_px > VIEWPORT_SIZE or 
               tile_py + TILE_SIZE < 0 or tile_py > VIEWPORT_SIZE)

def get_visible_tiles(available_tiles, viewport_x, viewport_y):
    """Get list of tiles that are currently visible"""
    min_tile_x = int(viewport_x // TILE_SIZE)
    max_tile_x = int((viewport_x + VIEWPORT_SIZE) // TILE_SIZE)
    min_tile_y = int(viewport_y // TILE_SIZE)
    max_tile_y = int((viewport_y + VIEWPORT_SIZE) // TILE_SIZE)

    visible_tiles = []
    for x in range(min_tile_x, max_tile_x + 1):
        for y in range(min_tile_y, max_tile_y + 1):
            if (x, y) in available_tiles:
                visible_tiles.append((x, y))
    
    return visible_tiles

def read_varint(data, offset):
    result = 0
    shift = 0
    while True:
        if offset >= len(data):
            return result, offset
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, offset

def read_zigzag(data, offset):
    v, offset = read_varint(data, offset)
    return (v >> 1) ^ -(v & 1), offset

def rgb332_to_rgb888(c):
    r = (c & 0xE0)
    g = (c & 0x1C) << 3
    b = (c & 0x03) << 6
    return (r, g, b)

def is_tile_border_point(pt):
    x, y = pt
    return (x == 0 or x == TILE_SIZE-1) or (y == 0 or y == TILE_SIZE-1)

def create_error_tile(error_message="Error"):
    """Create a tile surface indicating an error"""
    surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
    surface.fill((255, 0, 0))  # Red background for errors
    try:
        font = pygame.font.SysFont(None, 24)
        text = font.render(error_message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(TILE_SIZE//2, TILE_SIZE//2))
        surface.blit(text, text_rect)
    except Exception:
        pass  # If even error rendering fails, just return red surface
    return surface

def load_png_tile(filepath):
    """Load and render PNG tile with error recovery"""
    try:
        if not os.path.exists(filepath):
            logger.warning(f"PNG file does not exist: {filepath}")
            return create_error_tile("Missing")
        
        img = pygame.image.load(filepath)
        if img is None:
            logger.error(f"Failed to load PNG image: {filepath}")
            return create_error_tile("Load Failed")
            
        img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        surface.blit(img, (0, 0))
        return surface
    except pygame.error as e:
        logger.error(f"Pygame error loading PNG {filepath}: {e}")
        return create_error_tile("Pygame Error")
    except Exception as e:
        logger.error(f"Unexpected error loading PNG {filepath}: {e}")
        return create_error_tile("Unknown Error")

def parse_command_header(data, offset):
    """Parse the command header and return number of commands"""
    try:
        num_cmds, offset = read_varint(data, offset)
        return num_cmds, offset
    except Exception as e:
        logger.error(f"Error parsing command header: {e}")
        return 0, offset

def handle_color_command(cmd_type, data, offset, current_color):
    """Handle color setting commands"""
    if cmd_type == 0x80:  # SET_COLOR (direct RGB332)
        if offset >= len(data):
            return current_color, offset, True
        current_color = data[offset]
        offset += 1
        return current_color, offset, True
    elif cmd_type == 0x81:  # SET_COLOR_INDEX (new palette command)
        if offset >= len(data):
            return current_color, offset, True
        color_index, offset = read_varint(data, offset)
        
        # Convert index to RGB using global palette
        if color_index in GLOBAL_PALETTE:
            rgb = GLOBAL_PALETTE[color_index]
            # Simulate RGB332 to maintain compatibility
            current_color = ((rgb[0] & 0xE0) | ((rgb[1] & 0xE0) >> 3) | (rgb[2] >> 6))
        else:
            # Fallback if we don't have the palette
            current_color = 255  # Default color
            logger.warning(f"Unknown palette index {color_index}")
        return current_color, offset, True
    
    return current_color, offset, False

def render_line_command(data, offset, surface, rgb, current_position, movement_vector):
    """Render LINE command"""
    x1, offset = read_zigzag(data, offset)
    y1, offset = read_zigzag(data, offset)
    dx, offset = read_zigzag(data, offset)
    dy, offset = read_zigzag(data, offset)
    x2 = x1 + dx
    y2 = y1 + dy
    pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y1)),
                     (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y2)), 1)
    # Update position and movement for predictions
    current_position = (x2, y2)
    movement_vector = (dx, dy)
    return offset, current_position, movement_vector

def render_polyline_command(data, offset, surface, rgb):
    """Render POLYLINE command"""
    n_pts, offset = read_varint(data, offset)
    pts = []
    x, y = 0, 0
    for i in range(n_pts):
        if i == 0:
            x, offset = read_zigzag(data, offset)
            y, offset = read_zigzag(data, offset)
        else:
            dx, offset = read_zigzag(data, offset)
            dy, offset = read_zigzag(data, offset)
            x += dx
            y += dy
        pts.append((uint16_to_tile_pixel(x), uint16_to_tile_pixel(y)))
    if len(pts) >= 2:
        pygame.draw.lines(surface, rgb, False, pts, 1)
        current_position = (x, y)
    return offset, current_position

def render_polygon_command(data, offset, surface, rgb, fill_mode):
    """Render STROKE_POLYGON command"""
    n_pts, offset = read_varint(data, offset)
    pts = []
    x, y = 0, 0
    for i in range(n_pts):
        if i == 0:
            x, offset = read_zigzag(data, offset)
            y, offset = read_zigzag(data, offset)
        else:
            dx, offset = read_zigzag(data, offset)
            dy, offset = read_zigzag(data, offset)
            x += dx
            y += dy
        pts.append((uint16_to_tile_pixel(x), uint16_to_tile_pixel(y)))
    
    if fill_mode and len(pts) >= 3:
        pygame.draw.polygon(surface, rgb, pts, 0)
        closed = pts[0] == pts[-1] if len(pts) > 0 else False
        for i in range(len(pts)-1):
            p1 = pts[i]
            p2 = pts[i+1]
            if is_tile_border_point(p1) and is_tile_border_point(p2):
                continue
            else:
                border_rgb = darken_color(rgb)
                pygame.draw.line(surface, border_rgb, p1, p2, 2)
        if closed and len(pts) > 1:
            p1 = pts[-1]
            p2 = pts[0]
            if not (is_tile_border_point(p1) and is_tile_border_point(p2)):
                border_rgb = darken_color(rgb)
                pygame.draw.line(surface, border_rgb, p1, p2, 2)
        
    if len(pts) >= 2:
        closed = pts[0] == pts[-1] if len(pts) > 0 else False
        for i in range(len(pts)-1):
            p1 = pts[i]
            p2 = pts[i+1]
            if not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2):
                continue
            else:
                pygame.draw.line(surface, rgb, p1, p2, 1)
        if closed and len(pts) > 1:
            p1 = pts[-1]
            p2 = pts[0]
            if not (not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2)):
                pygame.draw.line(surface, rgb, p1, p2, 1)
    return offset

def render_geometry_command(cmd_type, data, offset, surface, current_color, fill_mode, current_position, movement_vector):
    """Render geometry commands"""
    rgb = rgb332_to_rgb888(current_color) if current_color is not None else (255, 255, 255)
    
    if cmd_type == 1:  # LINE
        offset, current_position, movement_vector = render_line_command(data, offset, surface, rgb, current_position, movement_vector)
    elif cmd_type == 2:  # POLYLINE
        offset, current_position = render_polyline_command(data, offset, surface, rgb)
    elif cmd_type == 3:  # STROKE_POLYGON
        offset = render_polygon_command(data, offset, surface, rgb, fill_mode)
    elif cmd_type == 5:  # HORIZONTAL_LINE
        x1, offset = read_zigzag(data, offset)
        dx, offset = read_zigzag(data, offset)
        y, offset = read_zigzag(data, offset)
        x2 = x1 + dx
        pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y)),
                         (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y)), 1)
        current_position = (x2, y)
    elif cmd_type == 6:  # VERTICAL_LINE
        x, offset = read_zigzag(data, offset)
        y1, offset = read_zigzag(data, offset)
        dy, offset = read_zigzag(data, offset)
        y2 = y1 + dy
        pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x), uint16_to_tile_pixel(y1)),
                         (uint16_to_tile_pixel(x), uint16_to_tile_pixel(y2)), 1)
        current_position = (x, y2)
    elif cmd_type == 0x82:  # RECTANGLE
        x1, offset = read_zigzag(data, offset)
        y1, offset = read_zigzag(data, offset)
        dx, offset = read_zigzag(data, offset)
        dy, offset = read_zigzag(data, offset)
        x2 = x1 + dx
        y2 = y1 + dy
        
        px1 = uint16_to_tile_pixel(x1)
        py1 = uint16_to_tile_pixel(y1)
        px2 = uint16_to_tile_pixel(x2)
        py2 = uint16_to_tile_pixel(y2)
        
        rect = pygame.Rect(min(px1, px2), min(py1, py2), 
                         abs(px2 - px1), abs(py2 - py1))
        
        if fill_mode:
            pygame.draw.rect(surface, rgb, rect, 0)
            pygame.draw.rect(surface, darken_color(rgb), rect, 1)
        else:
            pygame.draw.rect(surface, rgb, rect, 1)
    elif cmd_type == 0x83:  # STRAIGHT_LINE
        x1, offset = read_zigzag(data, offset)
        y1, offset = read_zigzag(data, offset)
        dx, offset = read_zigzag(data, offset)
        dy, offset = read_zigzag(data, offset)
        x2 = x1 + dx
        y2 = y1 + dy
        
        pygame.draw.line(surface, rgb, 
                       (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y1)),
                       (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y2)), 2)
        current_position = (x2, y2)
        movement_vector = (dx, dy)
    elif cmd_type == 0x84:  # HIGHWAY_SEGMENT
        x1, offset = read_zigzag(data, offset)
        y1, offset = read_zigzag(data, offset)
        dx, offset = read_zigzag(data, offset)
        dy, offset = read_zigzag(data, offset)
        x2 = x1 + dx
        y2 = y1 + dy
        
        pygame.draw.line(surface, rgb, 
                       (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y1)),
                       (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y2)), 2)
        current_position = (x2, y2)
        movement_vector = (dx, dy)
    elif cmd_type == 0x85:  # GRID_PATTERN
        x, offset = read_zigzag(data, offset)
        y, offset = read_zigzag(data, offset)
        width, offset = read_zigzag(data, offset)
        spacing, offset = read_zigzag(data, offset)
        count, offset = read_varint(data, offset)
        direction, offset = data[offset], offset + 1
        
        px = uint16_to_tile_pixel(x)
        py = uint16_to_tile_pixel(y)
        pwidth = uint16_to_tile_pixel(width)
        pspacing = uint16_to_tile_pixel(spacing)
        
        if direction == 1:  # Horizontal
            for i in range(count):
                line_y = py + i * pspacing
                if 0 <= line_y < TILE_SIZE:
                    pygame.draw.line(surface, rgb, (px, line_y), 
                                   (px + pwidth, line_y), 1)
        else:  # Vertical
            for i in range(count):
                line_x = px + i * pspacing
                if 0 <= line_x < TILE_SIZE:
                    pygame.draw.line(surface, rgb, (line_x, py), 
                                   (line_x, py + pwidth), 1)
    elif cmd_type == 0x87:  # CIRCLE
        center_x, offset = read_zigzag(data, offset)
        center_y, offset = read_zigzag(data, offset)
        radius, offset = read_zigzag(data, offset)
        
        pcenter_x = uint16_to_tile_pixel(center_x)
        pcenter_y = uint16_to_tile_pixel(center_y)
        pradius = uint16_to_tile_pixel(radius)
        
        if pradius > 0:
            if fill_mode:
                pygame.draw.circle(surface, rgb, (pcenter_x, pcenter_y), pradius, 0)
                pygame.draw.circle(surface, darken_color(rgb), (pcenter_x, pcenter_y), pradius, 1)
            else:
                pygame.draw.circle(surface, rgb, (pcenter_x, pcenter_y), pradius, 1)
    elif cmd_type == 0x89:  # PREDICTED_LINE
        end_x, offset = read_zigzag(data, offset)
        end_y, offset = read_zigzag(data, offset)
        
        start_x = current_position[0] + movement_vector[0]
        start_y = current_position[1] + movement_vector[1]
        
        pygame.draw.line(surface, rgb,
                       (uint16_to_tile_pixel(start_x), uint16_to_tile_pixel(start_y)),
                       (uint16_to_tile_pixel(end_x), uint16_to_tile_pixel(end_y)), 1)
        
        current_position = (end_x, end_y)
        movement_vector = (end_x - start_x, end_y - start_y)
    elif cmd_type == 0x86:  # BLOCK_PATTERN
        logger.warning("BLOCK_PATTERN command not implemented yet")
    elif cmd_type == 0x88:  # RELATIVE_MOVE
        logger.warning("RELATIVE_MOVE command not implemented yet")
    elif cmd_type == 0x8A:  # COMPRESSED_POLYLINE
        logger.warning("COMPRESSED_POLYLINE command not implemented yet")
    else:
        logger.warning(f"Unknown command type: {cmd_type} (0x{cmd_type:02x})")
    
    return offset, current_position, movement_vector

def render_tile_surface(tile, bg_color, fill_mode):
    """Main function to render tile surface from binary data with error recovery"""
    try:
        surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
        surface.fill(bg_color)
        filepath = tile['file']
        
        if not filepath:
            logger.error("No file path provided for tile")
            return create_error_tile("No File")
        
        if filepath.endswith('.png'):
            png_surface = load_png_tile(filepath)
            if png_surface:
                return png_surface
            return surface

        # Validate file exists and is readable
        if not os.path.exists(filepath):
            logger.warning(f"Tile file does not exist: {filepath}")
            return create_error_tile("Missing")
        
        if not os.access(filepath, os.R_OK):
            logger.error(f"Tile file is not readable: {filepath}")
            return create_error_tile("No Access")

        try:
            with open(filepath, "rb") as f:
                data = f.read()
        except PermissionError as e:
            logger.error(f"Permission denied reading {filepath}: {e}")
            return create_error_tile("Permission")
        except OSError as e:
            logger.error(f"OS error reading {filepath}: {e}")
            return create_error_tile("OS Error")
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return create_error_tile("Read Error")

        if len(data) < 1:
            logger.warning(f"Empty tile file: {filepath}")
            return surface

        offset = 0
        current_color = None
        
        # Initialize rendering state
        current_position = (0, 0)
        movement_vector = (0, 0)
        
        try:
            num_cmds, offset = parse_command_header(data, offset)
            
            for cmd_idx in range(num_cmds):
                if offset >= len(data):
                    logger.warning(f"Command {cmd_idx} extends beyond data length in {filepath}")
                    break
                    
                cmd_type, offset = read_varint(data, offset)
                
                # Handle color commands
                current_color, offset, is_color_cmd = handle_color_command(cmd_type, data, offset, current_color)
                if is_color_cmd:
                    continue
                
                # Handle geometry commands
                offset, current_position, movement_vector = render_geometry_command(cmd_type, data, offset, surface, current_color, fill_mode, current_position, movement_vector)
                                     
        except Exception as e:
            logger.error(f"Error parsing commands in {filepath}: {e}")
            logger.error(f"Error at offset: {offset}, data length: {len(data)}")
            # Return partial surface instead of error tile for parsing errors
            return surface

        return surface
        
    except Exception as e:
        logger.error(f"Critical error in render_tile_surface: {e}")
        return create_error_tile("Critical")

def center_viewport_on_central_tile(available_tiles):
    if not available_tiles:
        return 0, 0
    xs = [x for x, y in available_tiles]
    ys = [y for x, y in available_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    viewport_x = center_x * TILE_SIZE - VIEWPORT_SIZE // 2
    viewport_y = center_y * TILE_SIZE - VIEWPORT_SIZE // 2
    return viewport_x, viewport_y

def clamp_viewport(viewport_x, viewport_y, available_tiles):
    if not available_tiles:
        return viewport_x, viewport_y
    xs = [x for x, y in available_tiles]
    ys = [y for x, y in available_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    viewport_x = max(min_x * TILE_SIZE, min(viewport_x, (max_x * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE))
    viewport_y = max(min_y * TILE_SIZE, min(viewport_y, (max_y * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE))
    return viewport_x, viewport_y

def draw_button(surface, text, rect, bg_color, fg_color, border_color, font, icon=None, pressed=False):
    """Draw a button with improved text handling and multi-line support"""
    radius = 16
    pygame.draw.rect(surface, bg_color, rect, border_radius=radius)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=radius)
    if pressed:
        pygame.draw.rect(surface, border_color, rect, 4, border_radius=radius)
    
    # Calculate content area
    content_x = rect.left + 12
    content_y = rect.centery
    icon_width = 0
    
    if icon is not None:
        icon_rect = icon.get_rect()
        icon_rect.centery = rect.centery
        icon_rect.left = rect.left + 12
        surface.blit(icon, icon_rect)
        content_x = icon_rect.right + 8
        icon_width = icon_rect.width + 8
    
    # Calculate available text width
    max_text_width = rect.width - icon_width - 24  # 12px margin on each side
    
    # Split text into words for multi-line support
    words = text.split()
    if not words:
        return
    
    # Use consistent font size (14px for better readability)
    button_font = pygame.font.SysFont(None, 14)
    line_height = button_font.get_height()
    
    # Calculate how many lines we need
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        word_width = button_font.size(word + " ")[0]
        if current_width + word_width <= max_text_width:
            current_line.append(word)
            current_width += word_width
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                # Single word is too long, force it
                lines.append(word)
                current_line = []
                current_width = 0
    
    if current_line:
        lines.append(" ".join(current_line))
    
    # Limit to 2 lines maximum
    if len(lines) > 2:
        lines = lines[:2]
        # If we have more than 2 lines, truncate the last line with "..."
        if len(lines) == 2:
            last_line = lines[1]
            while button_font.size(last_line + "...")[0] > max_text_width and len(last_line) > 3:
                last_line = last_line[:-1]
            lines[1] = last_line + "..."
    
    # Calculate total text height
    total_text_height = len(lines) * line_height
    
    # Center the text vertically
    start_y = content_y - total_text_height // 2
    
    # Render each line
    for i, line in enumerate(lines):
        label = button_font.render(line, True, fg_color)
        text_rect = label.get_rect(midleft=(content_x, start_y + i * line_height + line_height // 2))
        surface.blit(label, text_rect)

def show_status_progress_bar(surface, percent, text, font):
    bar_max_width = WINDOW_WIDTH // 3
    bar_height = 18
    bar_margin_right = 24
    bar_x = WINDOW_WIDTH - bar_max_width - bar_margin_right
    bar_y = VIEWPORT_SIZE + STATUSBAR_HEIGHT // 2 - bar_height // 2
    pygame.draw.rect(surface, (80, 80, 80), (bar_x, bar_y, bar_max_width, bar_height))
    pygame.draw.rect(surface, (30, 160, 220), (bar_x, bar_y, int(bar_max_width * percent), bar_height))
    pygame.draw.rect(surface, (120, 120, 120), (bar_x, bar_y, bar_max_width, bar_height), 2)
    label = font.render(text, True, (255,255,255))
    label_rect = label.get_rect(midleft=(bar_x + 8, bar_y + bar_height//2 - label.get_height()//2))
    surface.blit(label, label_rect)

def draw_tile_labels(
    screen, font, available_tiles, viewport_x, viewport_y, zoom_level, background_color, show_tile_labels, directory
):
    if not show_tile_labels:
        return
    fg = (0, 0, 0) if background_color == (255,255,255) else (255,255,255)
    label_bg = (240,240,240) if background_color == (255,255,255) else (32,32,32)
    border = (180,180,180) if background_color == (255,255,255) else (64,64,64)
    outline = (120,120,120) if background_color == (255,255,255) else (220,220,220)
    for x, y in available_tiles:
        px = x * TILE_SIZE - viewport_x
        py = y * TILE_SIZE - viewport_y
        if px + TILE_SIZE < 0 or px > VIEWPORT_SIZE or py + TILE_SIZE < 0 or py > VIEWPORT_SIZE:
            continue
        filename = None
        if os.path.isfile(f"{directory}/{x}/{y}.bin"):
            filename = f"{y}.bin"
        elif os.path.isfile(f"{directory}/{x}/{y}.png"):
            filename = f"{y}.png"
        else:
            filename = f"{y}"
        txt = f"x={x} y={y} z={zoom_level} {filename}"
        label_surfs = [font.render(txt, True, fg)]
        lw = max(s.get_width() for s in label_surfs)
        lh = sum(s.get_height() for s in label_surfs)
        margin = 2
        label_rect = pygame.Rect(
            px + margin, py + margin,
            lw + margin * 2, lh + margin * 2
        )
        pygame.draw.rect(screen, label_bg, label_rect)
        pygame.draw.rect(screen, border, label_rect, 1)
        offset_y = label_rect.top + margin
        for surf in label_surfs:
            screen.blit(surf, (label_rect.left + margin, offset_y))
            offset_y += surf.get_height()
        draw_dashed_rect(screen, pygame.Rect(px, py, TILE_SIZE, TILE_SIZE), outline, width=1)

def draw_dashed_rect(surface, rect, color, dash_length=6, gap_length=4, width=1):
    x = rect.left
    while x < rect.right:
        end_x = min(x + dash_length, rect.right)
        pygame.draw.line(surface, color, (x, rect.top), (end_x, rect.top), width)
        x += dash_length + gap_length
    x = rect.left
    while x < rect.right:
        end_x = min(x + dash_length, rect.right)
        pygame.draw.line(surface, color, (x, rect.bottom-1), (end_x, rect.bottom-1), width)
        x += dash_length + gap_length
    y = rect.top
    while y < rect.bottom:
        end_y = min(y + dash_length, rect.bottom)
        pygame.draw.line(surface, color, (rect.left, y), (rect.left, end_y), width)
        y += dash_length + gap_length
    y = rect.top
    while y < rect.bottom:
        end_y = min(y + dash_length, rect.bottom)
        pygame.draw.line(surface, color, (rect.right-1, y), (rect.right-1, end_y), width)
        y += dash_length + gap_length

# Function removed - functionality replaced by pixel_to_latlon_cached

@lru_cache(maxsize=500)
def pixel_to_latlon_cached(tile_x, tile_y, zoom):
    """Cached version of pixel to lat/lon conversion"""
    n = 2.0 ** zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def pixel_to_latlon(px, py, viewport_x, viewport_y, zoom):
    """Convert pixel coordinates to lat/lon"""
    map_px = viewport_x + px
    map_py = viewport_y + py
    tile_x = map_px / TILE_SIZE
    tile_y = map_py / TILE_SIZE
    return pixel_to_latlon_cached(tile_x, tile_y, zoom)

@lru_cache(maxsize=500)
def latlon_to_pixel_cached(lat, lon, zoom):
    """Cached version of lat/lon to pixel conversion"""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    map_px = x * TILE_SIZE
    map_py = y * TILE_SIZE
    return map_px, map_py

def latlon_to_pixel(lat, lon, zoom):
    """Convert lat/lon to pixel coordinates"""
    return latlon_to_pixel_cached(lat, lon, zoom)

# Function removed - not used in current implementation

def decimal_to_gms(decimal, is_latitude=True):
    sign = ""
    if is_latitude:
        sign = "N" if decimal >= 0 else "S"
    else:
        sign = "E" if decimal >= 0 else "W"
    decimal = abs(decimal)
    degrees = int(decimal)
    minutes_full = (decimal - degrees) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return f"{degrees}°{minutes}'{seconds:.2f}\" {sign}"

def main(base_dir):
    # Try to load palette from features.json if it exists
    config_file = "features.json"  # You can change this to the correct path
    if os.path.exists(config_file):
        load_global_palette_from_config(config_file)
    else:
        logger.info(f"Config file {config_file} not found, using fallback colors")

    zoom_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    zoom_levels_list = sorted([int(d) for d in zoom_dirs])
    if not zoom_levels_list:
        logger.error(f"No zoom level directories found in {base_dir}")
        sys.exit(1)
    min_zoom = zoom_levels_list[0]
    max_zoom = zoom_levels_list[-1]
    zoom_levels = list(range(min_zoom, max_zoom+1))
    zoom_idx = 0

    background_color = (0, 0, 0)
    button_color = (0, 0, 0)
    button_fg = (255, 255, 255)
    button_border = (100,100,100)

    toolbar_x = VIEWPORT_SIZE
    toolbar_y = 0
    button_height = 40
    button_margin = 16
    button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin, 100, button_height)
    tile_label_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 2 + button_height, 100, button_height)
    gps_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 3 + button_height * 2, 100, button_height)
    fill_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 4 + button_height * 3, 100, button_height)

    button_text_black = "Black"
    button_text_white = "White"
    button_pressed = False

    show_tile_labels = False
    show_gps_tooltip = False
    fill_polygons_mode = False

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Tile viewer - MAP {os.path.basename(base_dir)}")
    font = pygame.font.SysFont(None, 16)
    font_main = pygame.font.SysFont(None, 18)
    font_b = pygame.font.SysFont(None, 16)
    font_status = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()

    available_tiles = set()

    icon_bg, icon_label, icon_gps, icon_fill = get_button_icons()

    mouse_gps_coords = None
    mouse_gps_rect = None

    show_index_progress = False
    index_progress_percent = 0.0
    index_progress_text = ""
    index_progress_done_drawn = False
    show_render_progress = False
    render_progress_percent = 0.0
    render_progress_text = ""
    render_progress_done_drawn = False

    tiles_loading = False
    tiles_loading_lock = threading.Lock()
    need_redraw = True
    zoom_change_pending = False
    zoom_change_params = None

    def status_index_progress_callback(percent, text):
        nonlocal show_index_progress, index_progress_percent, index_progress_text, need_redraw
        show_index_progress = True
        index_progress_percent = percent
        index_progress_text = text
        need_redraw = True

    def hide_index_progress():
        nonlocal show_index_progress, index_progress_percent, index_progress_text, need_redraw, index_progress_done_drawn
        if index_progress_done_drawn:
            show_index_progress = False
            index_progress_percent = 0.0
            index_progress_text = ""
            index_progress_done_drawn = False
            need_redraw = True

    def status_render_progress_callback(percent, text):
        nonlocal show_render_progress, render_progress_percent, render_progress_text, need_redraw
        show_render_progress = True
        render_progress_percent = percent
        render_progress_text = text
        need_redraw = True

    def hide_render_progress():
        nonlocal show_render_progress, render_progress_percent, render_progress_text, need_redraw, render_progress_done_drawn
        if render_progress_done_drawn:
            show_render_progress = False
            render_progress_percent = 0.0
            render_progress_text = ""
            render_progress_done_drawn = False
            need_redraw = True

    def load_available_tiles(level, progress_callback=None):
        directory = os.path.join(base_dir, str(level))
        available = index_available_tiles(directory, progress_callback)
        return available, directory

    def get_tile_surface(x, y, zoom_level, directory, bg_color, fill_mode):
        """Get tile surface with lazy loading - only load if visible"""
        key = (zoom_level, x, y, bg_color, fill_mode)
        if (x, y) not in available_tiles:
            return None
        
        # Try to get from cache first
        cached_surface = tile_cache.get(key)
        if cached_surface is not None:
            return cached_surface
        
        # Only load if tile is visible (lazy loading)
        if not is_tile_visible(x, y, viewport_x, viewport_y):
            logger.debug(f"Skipping lazy load for non-visible tile ({x}, {y})")
            return None
        
        # Load and cache the surface
        tile_file = get_tile_file(directory, x, y)
        if not tile_file:
            return None
        
        surface = render_tile_surface({'x': x, 'y': y, 'file': tile_file}, bg_color, fill_mode)
        tile_cache.put(key, surface)
        return surface

    def preload_tile_surfaces_threaded(tile_list, zoom_level, directory, bg_color, fill_mode, progress_callback=None, done_callback=None):
        """Preload tiles using the persistent thread pool"""
        total = len(tile_list)
        completed_count = 0
        
        def progress_callback_wrapper(future):
            nonlocal completed_count
            completed_count += 1
            if progress_callback is not None:
                percent = completed_count / max(total, 1)
                progress_callback(percent, "Loading visible tiles...")
            
            if completed_count >= total and done_callback:
                done_callback()
        
        # Submit all tiles to the persistent thread pool
        for x, y in tile_list:
            tile_info = (x, y, zoom_level, directory, bg_color, fill_mode)
            tile_loader.submit_tile_load(tile_info, progress_callback_wrapper)
        
        return None  # No thread to return since we use persistent pool

    def set_tiles_loading(flag):
        nonlocal tiles_loading, need_redraw
        with tiles_loading_lock:
            tiles_loading = flag
            need_redraw = True

    def start_zoom_change(idx, last_mouse_pos, viewport_x, viewport_y, old_zoom_idx):
        nonlocal zoom_change_pending, zoom_change_params
        zoom_change_pending = True
        zoom_change_params = (idx, last_mouse_pos, viewport_x, viewport_y, old_zoom_idx)

    show_index_progress = True
    index_progress_percent = 0.0
    index_progress_text = "Indexing initial tiles..."
    available_tiles, directory = load_available_tiles(zoom_levels[zoom_idx], status_index_progress_callback)
    hide_index_progress()
    viewport_x, viewport_y = center_viewport_on_central_tile(available_tiles)

    dragging = False
    drag_start = None
    running = True
    last_mouse_pos = (VIEWPORT_SIZE // 2, VIEWPORT_SIZE // 2)

    while running:
        if zoom_change_pending and not tiles_loading:
            idx, last_mouse_pos_z, old_vx, old_vy, old_zoom_idx = zoom_change_params
            show_index_progress = True
            index_progress_percent = 0.0
            index_progress_text = "Indexing tiles..."
            index_progress_done_drawn = False
            available_tiles, directory = load_available_tiles(zoom_levels[idx], status_index_progress_callback)
            if idx > old_zoom_idx:
                lat, lon = pixel_to_latlon(last_mouse_pos_z[0], last_mouse_pos_z[1], old_vx, old_vy, zoom_levels[idx-1])
            else:
                lat, lon = pixel_to_latlon(last_mouse_pos_z[0], last_mouse_pos_z[1], old_vx, old_vy, zoom_levels[idx+1])
            map_px, map_py = latlon_to_pixel(lat, lon, zoom_levels[idx])
            viewport_x, viewport_y = int(map_px - last_mouse_pos_z[0]), int(map_py - last_mouse_pos_z[1])
            viewport_x, viewport_y = clamp_viewport(viewport_x, viewport_y, available_tiles)
            zoom_idx = idx
            zoom_change_pending = False
            need_redraw = True
            logger.info(f"Changed to zoom level {zoom_levels[zoom_idx]}")

        mx, my = pygame.mouse.get_pos()
        if show_gps_tooltip and 0 <= mx < VIEWPORT_SIZE and 0 <= my < VIEWPORT_SIZE:
            lat, lon = pixel_to_latlon(mx, my, viewport_x, viewport_y, zoom_levels[zoom_idx])
            mouse_gps_coords = (lat, lon)
            mouse_gps_rect = (mx, my)
        else:
            mouse_gps_coords = None
            mouse_gps_rect = None

        screen.fill((70,70,70))
        pygame.draw.rect(screen, background_color, (0,0,VIEWPORT_SIZE,VIEWPORT_SIZE))

        if available_tiles:
            xs = [x for x, y in available_tiles]
            ys = [y for x, y in available_tiles]
            min_x = max_x = min_y = max_y = 0
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
        else:
            min_x = max_x = min_y = max_y = 0

        # Get visible tiles using the new function
        visible_tiles = get_visible_tiles(available_tiles, viewport_x, viewport_y)

        # Only check for uncached tiles among visible ones (lazy loading)
        uncached_tiles = []
        for x, y in visible_tiles:
            key = (zoom_levels[zoom_idx], x, y, background_color, fill_polygons_mode)
            if tile_cache.get(key) is None:
                uncached_tiles.append((x, y))

        if uncached_tiles and not tiles_loading:
            set_tiles_loading(True)
            show_render_progress = True
            render_progress_percent = 0.0
            render_progress_text = "Loading visible tiles..."
            def done_callback():
                set_tiles_loading(False)
            preload_tile_surfaces_threaded(
                uncached_tiles, zoom_levels[zoom_idx], directory, background_color, fill_polygons_mode,
                status_render_progress_callback, done_callback
            )

        for x, y in visible_tiles:
            surf = get_tile_surface(x, y, zoom_levels[zoom_idx], directory, background_color, fill_polygons_mode)
            if surf:
                px = x * TILE_SIZE - viewport_x
                py = y * TILE_SIZE - viewport_y
                screen.blit(surf, (px, py))

        draw_tile_labels(
            screen, font, available_tiles, viewport_x, viewport_y, zoom_levels[zoom_idx], background_color, show_tile_labels, directory
        )
        pygame.draw.rect(screen, (0,0,0), (toolbar_x, toolbar_y, TOOLBAR_WIDTH, VIEWPORT_SIZE))
        pygame.draw.line(screen, (160,160,160), (toolbar_x,0), (toolbar_x, VIEWPORT_SIZE))
        draw_button(
            screen,
            button_text_black if background_color == (255, 255, 255) else button_text_white,
            button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_bg, pressed=button_pressed
        )
        label_btn_text = "Tile labels ON" if show_tile_labels else "Tile labels OFF"
        draw_button(
            screen, label_btn_text, tile_label_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_label, pressed=False
        )
        gps_btn_text = "GPS Cursor ON" if show_gps_tooltip else "GPS Cursor OFF"
        draw_button(
            screen, gps_btn_text, gps_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_gps, pressed=False
        )
        fill_btn_text = "Fill polygons ON" if fill_polygons_mode else "Fill polygons OFF"
        draw_button(
            screen, fill_btn_text, fill_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_fill, pressed=fill_polygons_mode
        )
        pygame.draw.rect(screen, (0,0,0), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
        pygame.draw.line(screen, (160,160,160), (0, VIEWPORT_SIZE), (WINDOW_WIDTH, VIEWPORT_SIZE))
        zoom_text = f"Zoom level: {zoom_levels[zoom_idx]}"
        zoom_img = font_status.render(zoom_text, True, (255,255,255))
        screen.blit(zoom_img, (16, VIEWPORT_SIZE + STATUSBAR_HEIGHT//2 - zoom_img.get_height()//2))

        if show_gps_tooltip and mouse_gps_coords is not None:
            lat, lon = mouse_gps_coords[0], mouse_gps_coords[1]
            lat_gms = decimal_to_gms(lat, is_latitude=True)
            lon_gms = decimal_to_gms(lon, is_latitude=False)
            txt = f"lat: {lat:.6f} ({lat_gms})\nlon: {lon:.6f} ({lon_gms})"
            tooltip_lines = txt.split('\n')
            tooltip_surfs = [font.render(line, True, (255,255,255)) for line in tooltip_lines]
            tw = max(s.get_width() for s in tooltip_surfs)
            th = sum(s.get_height() for s in tooltip_surfs)
            tm = 4
            mx, my = mouse_gps_rect[:2]
            tooltip_rect = pygame.Rect(mx+10, my+10, tw+tm*2, th+tm*2)
            pygame.draw.rect(screen, (0,0,0), tooltip_rect)
            pygame.draw.rect(screen, (200,200,200), tooltip_rect, 1)
            yoff = tooltip_rect.top + tm
            for surf in tooltip_surfs:
                screen.blit(surf, (tooltip_rect.left + tm, yoff))
                yoff += surf.get_height()

        if show_index_progress:
            show_status_progress_bar(screen, index_progress_percent, index_progress_text, font_main)
            if index_progress_percent >= 1.0:
                index_progress_done_drawn = True
            else:
                index_progress_done_drawn = False
        if show_render_progress:
            show_status_progress_bar(screen, render_progress_percent, render_progress_text, font_main)
            if render_progress_percent >= 1.0 and not uncached_tiles and not tiles_loading:
                render_progress_done_drawn = True
            else:
                render_progress_done_drawn = False

        pygame.display.flip()
        need_redraw = False

        # Cleanup completed futures periodically
        completed_futures = tile_loader.cleanup_completed_futures()
        if completed_futures > 0:
            logger.debug(f"Cleaned up {completed_futures} completed futures")

        if index_progress_done_drawn:
            hide_index_progress()
        if render_progress_done_drawn:
            hide_render_progress()

        can_interact = not show_index_progress and not show_render_progress and not tiles_loading

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                base_step = VIEWPORT_SIZE // 4
                zoom_factor = 1 + ((zoom_levels[zoom_idx] - min_zoom) * 0.23) if zoom_levels[zoom_idx] > min_zoom else 1
                step = int(base_step * zoom_factor)
                if event.key == pygame.K_LEFT and can_interact:
                    viewport_x = max(min_x * TILE_SIZE, viewport_x - step)
                    need_redraw = True
                elif event.key == pygame.K_RIGHT and can_interact:
                    viewport_x = min(viewport_x + step, (max_x * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE)
                    need_redraw = True
                elif event.key == pygame.K_UP and can_interact:
                    viewport_y = max(min_y * TILE_SIZE, viewport_y - step)
                    need_redraw = True
                elif event.key == pygame.K_DOWN and can_interact:
                    viewport_y = min(viewport_y + step, (max_y * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE)
                    need_redraw = True
                elif event.key == pygame.K_LEFTBRACKET and can_interact and not zoom_change_pending:
                    if zoom_idx > 0:
                        start_zoom_change(zoom_idx-1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.key == pygame.K_RIGHTBRACKET and can_interact and not zoom_change_pending:
                    if zoom_idx < len(zoom_levels) - 1:
                        start_zoom_change(zoom_idx+1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.key == pygame.K_l and can_interact:
                    show_tile_labels = not show_tile_labels
                    need_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and can_interact:
                    if tile_label_button_rect.collidepoint(event.pos):
                        show_tile_labels = not show_tile_labels
                        need_redraw = True
                    elif button_rect.collidepoint(event.pos):
                        background_color = (255, 255, 255) if background_color == (0, 0, 0) else (0, 0, 0)
                        need_redraw = True
                        logger.info(f"Background color changed to {'white' if background_color == (255, 255, 255) else 'black'}")
                    elif gps_button_rect.collidepoint(event.pos):
                        show_gps_tooltip = not show_gps_tooltip
                        need_redraw = True
                    elif fill_button_rect.collidepoint(event.pos):
                        fill_polygons_mode = not fill_polygons_mode
                        tile_cache.clear()
                        need_redraw = True
                    else:
                        dragging = True
                        drag_start = event.pos
                        drag_viewport_start = (viewport_x, viewport_y)
                elif event.button == 4 and can_interact and not zoom_change_pending:
                    if zoom_idx < len(zoom_levels) - 1:
                        start_zoom_change(zoom_idx+1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.button == 5 and can_interact and not zoom_change_pending:
                    if zoom_idx > 0:
                        start_zoom_change(zoom_idx-1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    drag_start = None
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                last_mouse_pos = (mx, my)
                if dragging:
                    dx = drag_start[0] - event.pos[0]
                    dy = drag_start[1] - event.pos[1]
                    viewport_x = drag_viewport_start[0] + dx
                    viewport_y = drag_viewport_start[1] + dy
                    viewport_x, viewport_y = clamp_viewport(viewport_x, viewport_y, available_tiles)
                    need_redraw = True

        clock.tick(config.get('fps_limit', 30))
    
    # Shutdown thread pool before quitting
    tile_loader.shutdown()
    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: python tile_viewer.py VECTORMAP")
        logger.info("Keys: [arrows] move, [ ] [ ] zoom level, mouse scroll: zoom level")
        logger.info("Mouse: drag to pan, buttons for background, tile labels, GPS cursor, fill polygons. [l] toggle labels")
        logger.info("Example: python tile_viewer.py VECTORMAP")
        logger.info("Note: Place features.json in current directory for dynamic palette support")
        logger.info("Advanced Support: GRID_PATTERN, CIRCLE, PREDICTED_LINE commands")
        sys.exit(1)
    main(sys.argv[1])