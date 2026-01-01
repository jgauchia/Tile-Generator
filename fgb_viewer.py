#!/usr/bin/env python3
"""
FlatGeobuf Viewer - ESP32 Map Simulator

Simulates the ESP32 map rendering behavior using FlatGeobuf files.
Displays a 768x768 viewport (3x3 tiles of 256px) centered on given coordinates.

Usage:
    python fgb_viewer.py fgb_dir --lat 42.5063 --lon 1.5218 [--zoom 14]
"""

import os
import sys
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import json

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Install with: pip install pygame")

try:
    import geopandas as gpd
    from shapely.geometry import box, Point, LineString, Polygon
    from shapely.ops import transform
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not found. Install with: pip install geopandas pyogrio")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants matching ESP32 implementation
TILE_SIZE = 256
VIEWPORT_SIZE = 768  # 3x3 tiles
TOOLBAR_WIDTH = 200
STATUSBAR_HEIGHT = 60
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

# Layer rendering order (lower = rendered first = behind)
LAYER_ORDER = [
    'water',
    'landuse',
    'terrain',
    'railways',
    'roads',
    'infrastructure',
    'buildings',
    'amenities',
    'places'
]


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    """Convert lat/lon to tile numbers (floating point for sub-tile precision)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def num2deg(xtile: float, ytile: float, zoom: int) -> Tuple[float, float]:
    """Convert tile numbers to lat/lon."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_bbox_for_viewport(center_lat: float, center_lon: float, zoom: int) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box for 768x768 viewport centered on coordinates.
    Returns (min_lon, min_lat, max_lon, max_lat)
    """
    # Get tile coordinates for center
    center_x, center_y = deg2num(center_lat, center_lon, zoom)

    # Calculate extent in tiles (768px = 3 tiles of 256px)
    tiles_extent = VIEWPORT_SIZE / TILE_SIZE / 2.0  # 1.5 tiles from center

    # Get corner tile coordinates
    min_tile_x = center_x - tiles_extent
    max_tile_x = center_x + tiles_extent
    min_tile_y = center_y - tiles_extent
    max_tile_y = center_y + tiles_extent

    # Convert back to lat/lon
    max_lat, min_lon = num2deg(min_tile_x, min_tile_y, zoom)
    min_lat, max_lon = num2deg(max_tile_x, max_tile_y, zoom)

    return (min_lon, min_lat, max_lon, max_lat)


def latlon_to_pixel(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Convert lat/lon to pixel coordinates within viewport."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Normalize to 0-1
    x_norm = (lon - min_lon) / (max_lon - min_lon)
    y_norm = (max_lat - lat) / (max_lat - min_lat)  # Y is inverted

    # Scale to viewport
    px = int(x_norm * VIEWPORT_SIZE)
    py = int(y_norm * VIEWPORT_SIZE)

    return px, py


def pixel_to_latlon(px: int, py: int, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Convert pixel coordinates to lat/lon."""
    min_lon, min_lat, max_lon, max_lat = bbox

    x_norm = px / VIEWPORT_SIZE
    y_norm = py / VIEWPORT_SIZE

    lon = min_lon + x_norm * (max_lon - min_lon)
    lat = max_lat - y_norm * (max_lat - min_lat)

    return lat, lon


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)


def rgb332_to_rgb888(c: int) -> Tuple[int, int, int]:
    """Convert RGB332 to RGB888."""
    r = (c & 0xE0)
    g = (c & 0x1C) << 3
    b = (c & 0x03) << 6
    return (r, g, b)


def darken_color(rgb: Tuple[int, int, int], amount: float = 0.3) -> Tuple[int, int, int]:
    """Darken RGB color by specified amount."""
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)


class FGBViewer:
    """FlatGeobuf map viewer simulating ESP32 behavior."""

    def __init__(self, fgb_dir: str, config_file: str = None):
        self.fgb_dir = fgb_dir
        self.config = {}
        self.layer_files: Dict[str, str] = {}  # layer_name -> filepath (flat structure fallback)
        self.zoom_dirs: Dict[int, str] = {}  # zoom_level -> directory path (new structure)

        # Load config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {config_file}")

        # Index FGB files (but don't load them)
        self._index_layers()

        # Viewport state
        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 14
        self.bbox = None

        # Display options
        self.background_color = (255, 255, 255)  # White default
        self.fill_polygons = True
        self.show_tile_grid = False

        # Stats
        self.last_query_stats = {}

    def _index_layers(self):
        """Index available zoom levels with FlatGeobuf files."""
        if not os.path.isdir(self.fgb_dir):
            logger.error(f"Directory not found: {self.fgb_dir}")
            return

        # Check for zoom-level directory structure (new format)
        self.zoom_dirs = {}
        for item in os.listdir(self.fgb_dir):
            item_path = os.path.join(self.fgb_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                zoom_level = int(item)
                self.zoom_dirs[zoom_level] = item_path
                # Count FGB files in this zoom directory
                fgb_count = len([f for f in os.listdir(item_path) if f.endswith('.fgb')])
                logger.info(f"Indexed zoom {zoom_level}: {fgb_count} layers")

        if self.zoom_dirs:
            logger.info(f"Zoom levels available: {sorted(self.zoom_dirs.keys())}")
        else:
            # Fallback to flat structure (old format)
            for filename in os.listdir(self.fgb_dir):
                if filename.endswith('.fgb'):
                    layer_name = filename[:-4]
                    filepath = os.path.join(self.fgb_dir, filename)
                    self.layer_files[layer_name] = filepath
                    file_size = os.path.getsize(filepath) / (1024 * 1024)
                    logger.info(f"Indexed layer: {layer_name} ({file_size:.1f} MB)")
            logger.info(f"Total layers indexed: {len(self.layer_files)} (flat structure)")

    def set_center(self, lat: float, lon: float, zoom: int = None):
        """Set viewport center and optionally zoom."""
        self.center_lat = lat
        self.center_lon = lon
        if zoom is not None:
            self.zoom = zoom
        self.bbox = get_bbox_for_viewport(self.center_lat, self.center_lon, self.zoom)
        logger.info(f"Viewport centered at ({lat:.6f}, {lon:.6f}) zoom {self.zoom}")
        logger.info(f"Bounding box: {self.bbox}")

    def _get_zoom_dir(self) -> Optional[str]:
        """Get the directory for the current zoom level."""
        if not self.zoom_dirs:
            return None

        # Find the best matching zoom directory
        available_zooms = sorted(self.zoom_dirs.keys())
        best_zoom = available_zooms[0]  # Default to lowest

        for z in available_zooms:
            if z <= self.zoom:
                best_zoom = z
            else:
                break

        return self.zoom_dirs.get(best_zoom)

    def query_features(self, layer_name: str) -> Optional[gpd.GeoDataFrame]:
        """Query features from layer within current bbox using R-Tree spatial index.

        This simulates ESP32 behavior - reads only features within bbox directly
        from the FGB file using its built-in R-Tree index.
        """
        if self.bbox is None:
            return None

        # Determine file path based on structure
        if self.zoom_dirs:
            # New zoom-based structure
            zoom_dir = self._get_zoom_dir()
            if zoom_dir is None:
                return None
            filepath = os.path.join(zoom_dir, f"{layer_name}.fgb")
            if not os.path.exists(filepath):
                return None
        else:
            # Flat structure fallback
            if layer_name not in self.layer_files:
                return None
            filepath = self.layer_files[layer_name]

        # Query directly from file using bbox (uses FGB R-Tree index)
        import time
        start = time.time()

        try:
            # gpd.read_file with bbox parameter uses R-Tree index for FlatGeobuf
            result = gpd.read_file(filepath, bbox=self.bbox)

            elapsed = (time.time() - start) * 1000  # ms

            # Calculate bytes read (memory usage of loaded data)
            bytes_read = result.memory_usage(deep=True).sum() if len(result) > 0 else 0

            self.last_query_stats[layer_name] = {
                'features': len(result),
                'time_ms': elapsed,
                'read_kb': bytes_read / 1024
            }

            return result
        except Exception as e:
            logger.debug(f"Spatial query error for {layer_name}: {e}")
            return None

    def render_to_surface(self, surface: pygame.Surface):
        """Render all visible features to pygame surface."""
        if self.bbox is None:
            return

        # Clear with background color
        surface.fill(self.background_color)

        # Render layers in order
        for layer_name in LAYER_ORDER:
            # Check if layer exists (either in zoom dirs or flat structure)
            if self.zoom_dirs:
                zoom_dir = self._get_zoom_dir()
                if zoom_dir and os.path.exists(os.path.join(zoom_dir, f"{layer_name}.fgb")):
                    self._render_layer(surface, layer_name)
            elif layer_name in self.layer_files:
                self._render_layer(surface, layer_name)

        # Draw tile grid if enabled
        if self.show_tile_grid:
            self._draw_tile_grid(surface)

    def _render_layer(self, surface: pygame.Surface, layer_name: str):
        """Render a single layer."""
        features = self.query_features(layer_name)
        if features is None or len(features) == 0:
            return

        # Sort by priority if available
        if 'priority' in features.columns:
            features = features.sort_values('priority')

        for idx, row in features.iterrows():
            self._render_feature(surface, row, layer_name)

    def _render_feature(self, surface: pygame.Surface, feature, layer_name: str):
        """Render a single feature."""
        geom = feature.geometry
        if geom is None or geom.is_empty:
            return

        # Get color from properties or default
        if 'color_rgb332' in feature.index and feature['color_rgb332']:
            color = rgb332_to_rgb888(int(feature['color_rgb332']))
        else:
            color = self._get_layer_default_color(layer_name)

        geom_type = geom.geom_type

        if geom_type == 'Point':
            self._render_point(surface, geom, color)
        elif geom_type == 'LineString':
            self._render_linestring(surface, geom, color)
        elif geom_type == 'Polygon':
            self._render_polygon(surface, geom, color)
        elif geom_type == 'MultiLineString':
            for line in geom.geoms:
                self._render_linestring(surface, line, color)
        elif geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                self._render_polygon(surface, poly, color)

    def _render_point(self, surface: pygame.Surface, point, color: Tuple[int, int, int]):
        """Render a point feature."""
        px, py = latlon_to_pixel(point.y, point.x, self.bbox)
        if 0 <= px < VIEWPORT_SIZE and 0 <= py < VIEWPORT_SIZE:
            pygame.draw.circle(surface, color, (px, py), 3)

    def _render_linestring(self, surface: pygame.Surface, line, color: Tuple[int, int, int], width: int = 1):
        """Render a linestring feature."""
        points = []
        for coord in line.coords:
            px, py = latlon_to_pixel(coord[1], coord[0], self.bbox)
            points.append((px, py))

        if len(points) >= 2:
            # Clip to viewport (simple)
            pygame.draw.lines(surface, color, False, points, width)

    def _render_polygon(self, surface: pygame.Surface, polygon, color: Tuple[int, int, int]):
        """Render a polygon feature."""
        if polygon.is_empty:
            return

        # Get exterior ring
        exterior = polygon.exterior
        points = []
        for coord in exterior.coords:
            px, py = latlon_to_pixel(coord[1], coord[0], self.bbox)
            points.append((px, py))

        if len(points) >= 3:
            if self.fill_polygons:
                pygame.draw.polygon(surface, color, points)
                # Draw border
                border_color = darken_color(color, 0.4)
                pygame.draw.polygon(surface, border_color, points, 1)
            else:
                pygame.draw.polygon(surface, color, points, 1)

    def _get_layer_default_color(self, layer_name: str) -> Tuple[int, int, int]:
        """Get default color for layer."""
        defaults = {
            'water': (136, 201, 250),      # Light blue
            'landuse': (200, 230, 200),    # Light green
            'roads': (255, 255, 255),      # White
            'railways': (128, 128, 128),   # Gray
            'buildings': (221, 221, 221),  # Light gray
            'amenities': (255, 200, 200),  # Light red
            'infrastructure': (200, 200, 200),
            'terrain': (139, 115, 85),     # Brown
            'places': (100, 100, 100)
        }
        return defaults.get(layer_name, (200, 200, 200))

    def _draw_tile_grid(self, surface: pygame.Surface):
        """Draw tile grid overlay."""
        grid_color = (100, 100, 100)

        # Draw vertical lines
        for i in range(4):
            x = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (x, 0), (x, VIEWPORT_SIZE), 1)

        # Draw horizontal lines
        for i in range(4):
            y = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (0, y), (VIEWPORT_SIZE, y), 1)


def draw_button(surface, text, rect, bg_color, fg_color, border_color, font, pressed=False):
    """Draw a button."""
    radius = 8
    pygame.draw.rect(surface, bg_color, rect, border_radius=radius)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=radius)
    if pressed:
        pygame.draw.rect(surface, border_color, rect, 4, border_radius=radius)

    label = font.render(text, True, fg_color)
    text_rect = label.get_rect(center=rect.center)
    surface.blit(label, text_rect)


def format_coord(decimal: float, is_latitude: bool = True) -> str:
    """Format decimal coordinate as degrees/minutes/seconds."""
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

    return f"{degrees}°{minutes}'{seconds:.1f}\"{sign}"


def main():
    parser = argparse.ArgumentParser(
        description='FlatGeobuf Map Viewer - ESP32 Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fgb_viewer.py ./output --lat 42.5063 --lon 1.5218
    python fgb_viewer.py ./output --lat 42.5063 --lon 1.5218 --zoom 16
    python fgb_viewer.py ./output --lat 41.3851 --lon 2.1734 --zoom 14 --config features.json

Controls:
    Arrow keys / Mouse drag: Pan map
    Mouse wheel / [ ] keys: Zoom in/out
    B: Toggle background color
    F: Toggle polygon fill
    G: Toggle tile grid
    Q / ESC: Quit
        """
    )

    parser.add_argument('fgb_dir', help='Directory containing FGB files')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--zoom', type=int, default=14, help='Zoom level (default: 14)')
    parser.add_argument('--config', help='Features configuration JSON file')

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        logger.error("pygame is required. Install with: pip install pygame")
        sys.exit(1)

    if not GEOPANDAS_AVAILABLE:
        logger.error("geopandas is required. Install with: pip install geopandas pyogrio")
        sys.exit(1)

    if not os.path.isdir(args.fgb_dir):
        logger.error(f"Directory not found: {args.fgb_dir}")
        sys.exit(1)

    # Initialize viewer
    viewer = FGBViewer(args.fgb_dir, args.config)
    viewer.set_center(args.lat, args.lon, args.zoom)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"FGB Viewer - {os.path.basename(args.fgb_dir)}")

    font = pygame.font.SysFont(None, 18)
    font_small = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()

    # Create viewport surface
    viewport_surface = pygame.Surface((VIEWPORT_SIZE, VIEWPORT_SIZE))

    # Button definitions
    button_margin = 10
    button_height = 35
    button_width = TOOLBAR_WIDTH - 20

    bg_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin, button_width, button_height)
    fill_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 2 + button_height, button_width, button_height)
    grid_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 3 + button_height * 2, button_width, button_height)

    # State
    dragging = False
    drag_start = None
    drag_center_start = None
    need_redraw = True
    running = True

    # Pan speed (degrees per key press)
    pan_speed_base = 0.01

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Calculate pan speed based on zoom
                pan_speed = pan_speed_base / (2 ** (viewer.zoom - 10))

                if event.key == pygame.K_LEFT:
                    viewer.set_center(viewer.center_lat, viewer.center_lon - pan_speed)
                    need_redraw = True
                elif event.key == pygame.K_RIGHT:
                    viewer.set_center(viewer.center_lat, viewer.center_lon + pan_speed)
                    need_redraw = True
                elif event.key == pygame.K_UP:
                    viewer.set_center(viewer.center_lat + pan_speed, viewer.center_lon)
                    need_redraw = True
                elif event.key == pygame.K_DOWN:
                    viewer.set_center(viewer.center_lat - pan_speed, viewer.center_lon)
                    need_redraw = True
                elif event.key == pygame.K_LEFTBRACKET:
                    if viewer.zoom > 1:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom - 1)
                        need_redraw = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    if viewer.zoom < 19:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom + 1)
                        need_redraw = True
                elif event.key == pygame.K_b:
                    if viewer.background_color == (255, 255, 255):
                        viewer.background_color = (0, 0, 0)
                    else:
                        viewer.background_color = (255, 255, 255)
                    need_redraw = True
                elif event.key == pygame.K_f:
                    viewer.fill_polygons = not viewer.fill_polygons
                    need_redraw = True
                elif event.key == pygame.K_g:
                    viewer.show_tile_grid = not viewer.show_tile_grid
                    need_redraw = True
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if event.button == 1:  # Left click
                    if bg_button_rect.collidepoint(mx, my):
                        if viewer.background_color == (255, 255, 255):
                            viewer.background_color = (0, 0, 0)
                        else:
                            viewer.background_color = (255, 255, 255)
                        need_redraw = True
                    elif fill_button_rect.collidepoint(mx, my):
                        viewer.fill_polygons = not viewer.fill_polygons
                        need_redraw = True
                    elif grid_button_rect.collidepoint(mx, my):
                        viewer.show_tile_grid = not viewer.show_tile_grid
                        need_redraw = True
                    elif mx < VIEWPORT_SIZE and my < VIEWPORT_SIZE:
                        dragging = True
                        drag_start = (mx, my)
                        drag_center_start = (viewer.center_lat, viewer.center_lon)

                elif event.button == 4:  # Scroll up - zoom in
                    if viewer.zoom < 19:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom + 1)
                        need_redraw = True
                elif event.button == 5:  # Scroll down - zoom out
                    if viewer.zoom > 1:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom - 1)
                        need_redraw = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging and drag_start and drag_center_start:
                    dx = event.pos[0] - drag_start[0]
                    dy = event.pos[1] - drag_start[1]

                    # Convert pixel delta to lat/lon delta
                    if viewer.bbox:
                        min_lon, min_lat, max_lon, max_lat = viewer.bbox
                        lon_per_pixel = (max_lon - min_lon) / VIEWPORT_SIZE
                        lat_per_pixel = (max_lat - min_lat) / VIEWPORT_SIZE

                        new_lon = drag_center_start[1] - dx * lon_per_pixel
                        new_lat = drag_center_start[0] + dy * lat_per_pixel

                        viewer.set_center(new_lat, new_lon)
                        need_redraw = True

        # Render
        if need_redraw:
            # Render map to viewport
            viewer.render_to_surface(viewport_surface)

            # Clear screen
            screen.fill((50, 50, 50))

            # Draw viewport
            screen.blit(viewport_surface, (0, 0))

            # Draw toolbar background
            pygame.draw.rect(screen, (30, 30, 30), (VIEWPORT_SIZE, 0, TOOLBAR_WIDTH, VIEWPORT_SIZE))
            pygame.draw.line(screen, (100, 100, 100), (VIEWPORT_SIZE, 0), (VIEWPORT_SIZE, VIEWPORT_SIZE))

            # Draw buttons
            button_bg = (50, 50, 50)
            button_fg = (255, 255, 255)
            button_border = (100, 100, 100)

            bg_text = "Background: White" if viewer.background_color == (255, 255, 255) else "Background: Black"
            draw_button(screen, bg_text, bg_button_rect, button_bg, button_fg, button_border, font_small)

            fill_text = "Fill: ON" if viewer.fill_polygons else "Fill: OFF"
            draw_button(screen, fill_text, fill_button_rect, button_bg, button_fg, button_border, font_small)

            grid_text = "Grid: ON" if viewer.show_tile_grid else "Grid: OFF"
            draw_button(screen, grid_text, grid_button_rect, button_bg, button_fg, button_border, font_small)

            # Draw info in toolbar
            info_y = button_margin * 4 + button_height * 3 + 20
            info_color = (200, 200, 200)

            # Coordinates
            lat_text = f"Lat: {viewer.center_lat:.6f}"
            lon_text = f"Lon: {viewer.center_lon:.6f}"
            zoom_text = f"Zoom: {viewer.zoom}"

            screen.blit(font_small.render(lat_text, True, info_color), (VIEWPORT_SIZE + 10, info_y))
            screen.blit(font_small.render(lon_text, True, info_color), (VIEWPORT_SIZE + 10, info_y + 18))
            screen.blit(font_small.render(zoom_text, True, info_color), (VIEWPORT_SIZE + 10, info_y + 36))

            # DMS format
            lat_dms = format_coord(viewer.center_lat, True)
            lon_dms = format_coord(viewer.center_lon, False)
            screen.blit(font_small.render(lat_dms, True, info_color), (VIEWPORT_SIZE + 10, info_y + 60))
            screen.blit(font_small.render(lon_dms, True, info_color), (VIEWPORT_SIZE + 10, info_y + 78))

            # Layer query stats (R-Tree queries)
            layer_y = info_y + 110
            screen.blit(font_small.render("R-Tree Queries:", True, info_color), (VIEWPORT_SIZE + 10, layer_y))
            total_features = 0
            total_time = 0
            total_size_kb = 0
            for i, layer_name in enumerate(LAYER_ORDER):
                if layer_name in viewer.last_query_stats:
                    stats = viewer.last_query_stats[layer_name]
                    read_kb = stats.get('read_kb', 0)
                    text = f"  {layer_name}: {stats['features']} ({stats['time_ms']:.0f}ms) [{read_kb:.0f}KB]"
                    total_features += stats['features']
                    total_time += stats['time_ms']
                    total_size_kb += read_kb
                    screen.blit(font_small.render(text, True, (150, 150, 150)), (VIEWPORT_SIZE + 10, layer_y + 18 + i * 14))

            # Total stats
            summary_y = layer_y + 18 + len(LAYER_ORDER) * 14 + 10
            screen.blit(font_small.render(f"Total: {total_features} features", True, (100, 200, 100)), (VIEWPORT_SIZE + 10, summary_y))
            screen.blit(font_small.render(f"Query time: {total_time:.0f}ms", True, (100, 200, 100)), (VIEWPORT_SIZE + 10, summary_y + 16))
            screen.blit(font_small.render(f"Data read: {total_size_kb:.0f}KB", True, (100, 200, 100)), (VIEWPORT_SIZE + 10, summary_y + 32))

            # Draw status bar
            pygame.draw.rect(screen, (30, 30, 30), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
            pygame.draw.line(screen, (100, 100, 100), (0, VIEWPORT_SIZE), (WINDOW_WIDTH, VIEWPORT_SIZE))

            # Status text
            if viewer.bbox:
                min_lon, min_lat, max_lon, max_lat = viewer.bbox
                bbox_text = f"BBox: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})"
                screen.blit(font_small.render(bbox_text, True, (200, 200, 200)), (10, VIEWPORT_SIZE + 10))

                # Resolution info
                meters_per_pixel = 156543.03392 * math.cos(math.radians(viewer.center_lat)) / (2 ** viewer.zoom)
                res_text = f"Resolution: {meters_per_pixel:.2f} m/px | Viewport: {VIEWPORT_SIZE}x{VIEWPORT_SIZE}px"
                screen.blit(font_small.render(res_text, True, (200, 200, 200)), (10, VIEWPORT_SIZE + 30))

            pygame.display.flip()
            need_redraw = False

        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
