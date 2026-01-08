#!/usr/bin/env python3
"""
FlatGeobuf Viewer - ESP32 Map Simulator

Simulates the ESP32 map rendering behavior using tile-based FlatGeobuf files.
Displays a 768x768 viewport (3x3 tiles of 256px) centered on given coordinates.

Usage:
    python fgb_viewer.py fgb_dir --lat 42.5063 --lon 1.5218 [--zoom 14]
"""

import os
import sys
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Set
import json

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Install with: pip install pygame")

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box, Point, LineString, Polygon
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
    """Calculate bounding box for 768x768 viewport based on 3x3 tile grid."""
    # Get center tile
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    center_tile_x = int(center_x)
    center_tile_y = int(center_y)

    # Bbox covers exactly 3x3 tiles around center tile
    min_tile_x = center_tile_x - 1
    max_tile_x = center_tile_x + 2  # +2 because we need the right edge of tile +1
    min_tile_y = center_tile_y - 1
    max_tile_y = center_tile_y + 2  # +2 because we need the bottom edge of tile +1

    max_lat, min_lon = num2deg(min_tile_x, min_tile_y, zoom)
    min_lat, max_lon = num2deg(max_tile_x, max_tile_y, zoom)
    return (min_lon, min_lat, max_lon, max_lat)


def latlon_to_pixel(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Convert lat/lon to pixel coordinates within viewport."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x_norm = (lon - min_lon) / (max_lon - min_lon)
    y_norm = (max_lat - lat) / (max_lat - min_lat)
    px = int(x_norm * VIEWPORT_SIZE)
    py = int(y_norm * VIEWPORT_SIZE)
    return px, py


def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    """Convert RGB565 to RGB888."""
    r = ((c >> 11) & 0x1F) << 3
    g = ((c >> 5) & 0x3F) << 2
    b = (c & 0x1F) << 3
    return (r, g, b)


def darken_color(rgb: Tuple[int, int, int], amount: float = 0.3) -> Tuple[int, int, int]:
    """Darken RGB color by specified amount."""
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)


class FGBViewer:
    """FlatGeobuf map viewer simulating ESP32 behavior with tile-based files."""

    def __init__(self, fgb_dir: str, config_file: str = None):
        self.fgb_dir = fgb_dir
        self.config = {}
        self.available_zooms: Set[int] = set()

        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded config from {config_file}")

        self._index_tiles()

        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 14
        self.bbox = None
        self.background_color = (255, 255, 255)
        self.fill_polygons = True
        self.show_tile_grid = False
        self.last_query_stats = {}
        self.cached_features = None  # Store features for click identification
        self.selected_feature = None  # Currently selected feature info

    def _index_tiles(self):
        """Index available zoom levels from tile structure."""
        if not os.path.isdir(self.fgb_dir):
            logger.error(f"Directory not found: {self.fgb_dir}")
            return

        # Scan for zoom level directories (numbers)
        for name in os.listdir(self.fgb_dir):
            zoom_path = os.path.join(self.fgb_dir, name)
            if os.path.isdir(zoom_path) and name.isdigit():
                zoom = int(name)
                # Verify it has tile subdirectories
                for x_name in os.listdir(zoom_path):
                    x_path = os.path.join(zoom_path, x_name)
                    if os.path.isdir(x_path) and x_name.isdigit():
                        self.available_zooms.add(zoom)
                        break

        if self.available_zooms:
            zooms_str = ", ".join(map(str, sorted(self.available_zooms)))
            logger.info(f"Available zoom levels: {zooms_str}")
        else:
            logger.warning("No tile directories found (expected z/x/y.fgb structure)")

    def _get_tiles_for_viewport(self) -> List[Tuple[int, int]]:
        """Get list of tiles (x, y) that cover the current viewport."""
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        center_tile_x = int(center_x)
        center_tile_y = int(center_y)

        tiles = []
        # 3x3 grid around center
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy
                # Validate tile coordinates
                max_tile = 2 ** self.zoom - 1
                if 0 <= tile_x <= max_tile and 0 <= tile_y <= max_tile:
                    tiles.append((tile_x, tile_y))

        return tiles

    def _get_tile_path(self, x: int, y: int) -> str:
        """Get file path for a tile."""
        return os.path.join(self.fgb_dir, str(self.zoom), str(x), f"{y}.fgb")

    def set_center(self, lat: float, lon: float, zoom: int = None):
        """Set viewport center and optionally zoom."""
        self.center_lat = lat
        self.center_lon = lon
        if zoom is not None:
            self.zoom = zoom
        self.bbox = get_bbox_for_viewport(self.center_lat, self.center_lon, self.zoom)

    def query_features(self) -> Optional[gpd.GeoDataFrame]:
        """Query features from tiles within current viewport."""
        if self.bbox is None:
            return None

        import time
        start = time.time()

        tiles = self._get_tiles_for_viewport()
        all_gdfs = []
        tiles_loaded = 0
        tiles_missing = 0

        for tile_x, tile_y in tiles:
            tile_path = self._get_tile_path(tile_x, tile_y)
            if os.path.exists(tile_path):
                try:
                    gdf = gpd.read_file(tile_path)
                    if len(gdf) > 0:
                        all_gdfs.append(gdf)
                    tiles_loaded += 1
                except Exception as e:
                    logger.warning(f"Error reading tile {tile_path}: {e}")
            else:
                tiles_missing += 1

        elapsed = (time.time() - start) * 1000

        if not all_gdfs:
            self.last_query_stats = {
                'tiles_loaded': 0,
                'tiles_missing': tiles_missing,
                'features': 0,
                'time_ms': elapsed
            }
            return None

        # Combine all tile features
        result = pd.concat(all_gdfs, ignore_index=True)
        result = gpd.GeoDataFrame(result, crs="EPSG:4326")

        # Filter by min_zoom (unpacked from zoom_priority high nibble)
        if 'zoom_priority' in result.columns:
            result = result[(result['zoom_priority'].astype(int) // 16) <= self.zoom]

        self.last_query_stats = {
            'tiles_loaded': tiles_loaded,
            'tiles_missing': tiles_missing,
            'features': len(result),
            'time_ms': elapsed
        }

        return result

    def render_to_surface(self, surface: pygame.Surface):
        """Render all visible features to pygame surface."""
        if self.bbox is None:
            return

        surface.fill(self.background_color)

        features = self.query_features()
        if features is None or len(features) == 0:
            self.cached_features = None
            return

        # Sort by priority (unpacked from zoom_priority low nibble)
        if 'zoom_priority' in features.columns:
            features = features.iloc[(features['zoom_priority'].astype(int) % 16).argsort()]

        # Cache for click identification
        self.cached_features = features

        for idx, row in features.iterrows():
            self._render_feature(surface, row)

        if self.show_tile_grid:
            self._draw_tile_grid(surface)

    def _render_feature(self, surface: pygame.Surface, feature):
        """Render a single feature."""
        geom = feature.geometry
        if geom is None or geom.is_empty:
            return

        if 'color_rgb565' in feature.index and feature['color_rgb565']:
            color = rgb565_to_rgb888(int(feature['color_rgb565']))
        else:
            color = (200, 200, 200)

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
        elif geom_type == 'GeometryCollection':
            for g in geom.geoms:
                self._render_geometry(surface, g, color)

    def _render_geometry(self, surface: pygame.Surface, geom, color: Tuple[int, int, int]):
        """Render any geometry type."""
        if geom is None or geom.is_empty:
            return
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
            pygame.draw.lines(surface, color, False, points, width)

    def _render_polygon(self, surface: pygame.Surface, polygon, color: Tuple[int, int, int]):
        """Render a polygon feature."""
        if polygon.is_empty:
            return

        exterior = polygon.exterior
        points = []
        for coord in exterior.coords:
            px, py = latlon_to_pixel(coord[1], coord[0], self.bbox)
            points.append((px, py))

        if len(points) >= 3:
            if self.fill_polygons:
                pygame.draw.polygon(surface, color, points)
                border_color = darken_color(color, 0.4)
                pygame.draw.polygon(surface, border_color, points, 1)
            else:
                pygame.draw.polygon(surface, color, points, 1)

    def _draw_tile_grid(self, surface: pygame.Surface):
        """Draw tile grid overlay with tile coordinates."""
        grid_color = (100, 100, 100)
        text_color = (80, 80, 80)
        font = pygame.font.SysFont(None, 14)

        # Draw grid lines
        for i in range(4):
            x = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (x, 0), (x, VIEWPORT_SIZE), 1)
        for i in range(4):
            y = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (0, y), (VIEWPORT_SIZE, y), 1)

        # Draw tile coordinates
        tiles = self._get_tiles_for_viewport()
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        center_tile_x = int(center_x)
        center_tile_y = int(center_y)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy
                # Screen position for this tile
                screen_x = (dx + 1) * TILE_SIZE + 5
                screen_y = (dy + 1) * TILE_SIZE + 5
                tile_path = self._get_tile_path(tile_x, tile_y)
                exists = os.path.exists(tile_path)
                color = (0, 100, 0) if exists else (150, 50, 50)
                label = font.render(f"{tile_x}/{tile_y}", True, color)
                surface.blit(label, (screen_x, screen_y))

    def identify_feature_at(self, pixel_x: int, pixel_y: int) -> Optional[dict]:
        """Identify feature at given pixel coordinates.

        Returns dict with feature info or None if no feature found.
        """
        if self.cached_features is None or self.bbox is None:
            return None

        # Convert pixel to lat/lon
        min_lon, min_lat, max_lon, max_lat = self.bbox
        lon = min_lon + (pixel_x / VIEWPORT_SIZE) * (max_lon - min_lon)
        lat = max_lat - (pixel_y / VIEWPORT_SIZE) * (max_lat - min_lat)

        click_point = Point(lon, lat)

        # Search features in reverse priority order (top features first)
        features_sorted = self.cached_features.sort_values('priority', ascending=False) if 'priority' in self.cached_features.columns else self.cached_features

        for idx, row in features_sorted.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Check if point is within/near the geometry
            if geom.geom_type == 'Polygon' or geom.geom_type == 'MultiPolygon':
                if geom.contains(click_point):
                    return self._feature_to_dict(row)
            elif geom.geom_type == 'LineString' or geom.geom_type == 'MultiLineString':
                # For lines, use a small buffer (tolerance in degrees ~5 pixels)
                tolerance = (max_lon - min_lon) / VIEWPORT_SIZE * 5
                if geom.buffer(tolerance).contains(click_point):
                    return self._feature_to_dict(row)
            elif geom.geom_type == 'Point':
                tolerance = (max_lon - min_lon) / VIEWPORT_SIZE * 10
                if click_point.distance(geom) < tolerance:
                    return self._feature_to_dict(row)

        return None

    def _feature_to_dict(self, row) -> dict:
        """Convert feature row to dict with relevant info."""
        info = {}
        if 'color_rgb565' in row.index and row['color_rgb565']:
            r, g, b = rgb565_to_rgb888(int(row['color_rgb565']))
            info['color'] = f"#{r:02x}{g:02x}{b:02x}"
        else:
            info['color'] = 'N/A'
        if 'zoom_priority' in row.index:
            zp = int(row['zoom_priority'])
            info['min_zoom'] = zp >> 4
            info['priority'] = (zp & 0x0F) * 7
        info['geom_type'] = row.geometry.geom_type if row.geometry else 'N/A'
        return info


def draw_button(surface, text, rect, bg_color, fg_color, border_color, font, pressed=False):
    """Draw a button."""
    radius = 8
    pygame.draw.rect(surface, bg_color, rect, border_radius=radius)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=radius)
    label = font.render(text, True, fg_color)
    text_rect = label.get_rect(center=rect.center)
    surface.blit(label, text_rect)


def format_coord(decimal: float, is_latitude: bool = True) -> str:
    """Format decimal coordinate as degrees/minutes/seconds."""
    sign = "N" if is_latitude else "E"
    if decimal < 0:
        sign = "S" if is_latitude else "W"
    decimal = abs(decimal)
    degrees = int(decimal)
    minutes_full = (decimal - degrees) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return f"{degrees}°{minutes}'{seconds:.1f}\"{sign}"


def main():
    parser = argparse.ArgumentParser(
        description='FlatGeobuf Map Viewer - ESP32 Simulator (Tile Structure)',
        epilog="""
Controls:
    Arrow keys / Mouse drag: Pan map
    Mouse wheel / [ ] keys: Zoom in/out
    B: Toggle background color
    F: Toggle polygon fill
    G: Toggle tile grid (shows tile coordinates)
    Q / ESC: Quit
        """
    )

    parser.add_argument('fgb_dir', help='Directory containing tile-based FGB files (z/x/y.fgb)')
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

    viewer = FGBViewer(args.fgb_dir, args.config)
    viewer.set_center(args.lat, args.lon, args.zoom)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"FGB Viewer (Tiles) - {os.path.basename(args.fgb_dir)}")

    font = pygame.font.SysFont(None, 18)
    font_small = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()

    viewport_surface = pygame.Surface((VIEWPORT_SIZE, VIEWPORT_SIZE))

    button_margin = 10
    button_height = 35
    button_width = TOOLBAR_WIDTH - 20

    bg_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin, button_width, button_height)
    fill_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 2 + button_height, button_width, button_height)
    grid_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 3 + button_height * 2, button_width, button_height)

    dragging = False
    drag_start = None
    drag_center_start = None
    need_redraw = True
    running = True
    pan_speed_base = 0.01

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
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
                    viewer.background_color = (0, 0, 0) if viewer.background_color == (255, 255, 255) else (255, 255, 255)
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

                if event.button == 1:
                    if bg_button_rect.collidepoint(mx, my):
                        viewer.background_color = (0, 0, 0) if viewer.background_color == (255, 255, 255) else (255, 255, 255)
                        need_redraw = True
                    elif fill_button_rect.collidepoint(mx, my):
                        viewer.fill_polygons = not viewer.fill_polygons
                        need_redraw = True
                    elif grid_button_rect.collidepoint(mx, my):
                        viewer.show_tile_grid = not viewer.show_tile_grid
                        need_redraw = True
                    elif mx < VIEWPORT_SIZE and my < VIEWPORT_SIZE:
                        # Left click = drag
                        dragging = True
                        drag_start = (mx, my)
                        drag_center_start = (viewer.center_lat, viewer.center_lon)

                elif event.button == 3:  # Right click = identify feature
                    if mx < VIEWPORT_SIZE and my < VIEWPORT_SIZE:
                        feature_info = viewer.identify_feature_at(mx, my)
                        viewer.selected_feature = feature_info
                        need_redraw = True

                elif event.button == 4:
                    if viewer.zoom < 19:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom + 1)
                        need_redraw = True
                elif event.button == 5:
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

                    if viewer.bbox:
                        min_lon, min_lat, max_lon, max_lat = viewer.bbox
                        lon_per_pixel = (max_lon - min_lon) / VIEWPORT_SIZE
                        lat_per_pixel = (max_lat - min_lat) / VIEWPORT_SIZE

                        new_lon = drag_center_start[1] - dx * lon_per_pixel
                        new_lat = drag_center_start[0] + dy * lat_per_pixel

                        viewer.set_center(new_lat, new_lon)
                        need_redraw = True

        if need_redraw:
            viewer.render_to_surface(viewport_surface)
            screen.fill((50, 50, 50))
            screen.blit(viewport_surface, (0, 0))

            # Toolbar
            pygame.draw.rect(screen, (30, 30, 30), (VIEWPORT_SIZE, 0, TOOLBAR_WIDTH, VIEWPORT_SIZE))

            button_bg = (50, 50, 50)
            button_fg = (255, 255, 255)
            button_border = (100, 100, 100)

            bg_text = "Background: White" if viewer.background_color == (255, 255, 255) else "Background: Black"
            draw_button(screen, bg_text, bg_button_rect, button_bg, button_fg, button_border, font_small)

            fill_text = "Fill: ON" if viewer.fill_polygons else "Fill: OFF"
            draw_button(screen, fill_text, fill_button_rect, button_bg, button_fg, button_border, font_small)

            grid_text = "Grid: ON" if viewer.show_tile_grid else "Grid: OFF"
            draw_button(screen, grid_text, grid_button_rect, button_bg, button_fg, button_border, font_small)

            info_y = button_margin * 4 + button_height * 3 + 20
            info_color = (200, 200, 200)

            screen.blit(font_small.render(f"Lat: {viewer.center_lat:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y))
            screen.blit(font_small.render(f"Lon: {viewer.center_lon:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 18))
            screen.blit(font_small.render(f"Zoom: {viewer.zoom}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 36))

            screen.blit(font_small.render(format_coord(viewer.center_lat, True), True, info_color), (VIEWPORT_SIZE + 10, info_y + 60))
            screen.blit(font_small.render(format_coord(viewer.center_lon, False), True, info_color), (VIEWPORT_SIZE + 10, info_y + 78))

            # Query stats
            stats_y = info_y + 110
            screen.blit(font_small.render("Query Stats:", True, info_color), (VIEWPORT_SIZE + 10, stats_y))
            if viewer.last_query_stats:
                s = viewer.last_query_stats
                screen.blit(font_small.render(f"  Tiles: {s.get('tiles_loaded', 0)}/{s.get('tiles_loaded', 0) + s.get('tiles_missing', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 18))
                screen.blit(font_small.render(f"  Features: {s.get('features', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 32))
                screen.blit(font_small.render(f"  Time: {s.get('time_ms', 0):.0f}ms", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 46))

            # Selected feature info
            feature_y = stats_y + 80
            screen.blit(font_small.render("Click Feature:", True, info_color), (VIEWPORT_SIZE + 10, feature_y))
            if viewer.selected_feature:
                f = viewer.selected_feature
                line_y = feature_y + 18
                highlight_color = (100, 200, 100)
                for key, value in f.items():
                    text = f"  {key}: {value}"
                    screen.blit(font_small.render(text, True, highlight_color), (VIEWPORT_SIZE + 10, line_y))
                    line_y += 14
            else:
                screen.blit(font_small.render("  (click on map)", True, (100, 100, 100)), (VIEWPORT_SIZE + 10, feature_y + 18))

            # Status bar
            pygame.draw.rect(screen, (30, 30, 30), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
            if viewer.bbox:
                min_lon, min_lat, max_lon, max_lat = viewer.bbox
                bbox_text = f"BBox: ({min_lat:.4f}, {min_lon:.4f}) to ({max_lat:.4f}, {max_lon:.4f})"
                screen.blit(font_small.render(bbox_text, True, (200, 200, 200)), (10, VIEWPORT_SIZE + 10))

                meters_per_pixel = 156543.03392 * math.cos(math.radians(viewer.center_lat)) / (2 ** viewer.zoom)
                res_text = f"Resolution: {meters_per_pixel:.2f} m/px"
                screen.blit(font_small.render(res_text, True, (200, 200, 200)), (10, VIEWPORT_SIZE + 30))

                # Show available zooms
                if viewer.available_zooms:
                    zooms_str = f"Available: {min(viewer.available_zooms)}-{max(viewer.available_zooms)}"
                    screen.blit(font_small.render(zooms_str, True, (150, 150, 150)), (400, VIEWPORT_SIZE + 10))

            pygame.display.flip()
            need_redraw = False

        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
