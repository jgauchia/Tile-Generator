#!/usr/bin/env python3
"""
NAV Tile Viewer - ESP32 Map Simulator

Displays NAV binary tiles in a 768x768 viewport (3x3 tiles of 256px).

Usage:
    python nav_viewer.py nav_dir --lat 42.5063 --lon 1.5218 [--zoom 14]
"""

import os
import sys
import math
import struct
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Set

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Install with: pip install pygame")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TILE_SIZE = 256
VIEWPORT_SIZE = 768
TOOLBAR_WIDTH = 200
STATUSBAR_HEIGHT = 60
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

# NAV format constants
NAV_MAGIC = b'NAV1'
COORD_SCALE = 10000000  # 1e7

# Geometry types
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    """Convert lat/lon to tile numbers."""
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
    """Calculate bounding box for 3x3 tile viewport."""
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    center_tile_x = int(center_x)
    center_tile_y = int(center_y)

    min_tile_x = center_tile_x - 1
    max_tile_x = center_tile_x + 2
    min_tile_y = center_tile_y - 1
    max_tile_y = center_tile_y + 2

    max_lat, min_lon = num2deg(min_tile_x, min_tile_y, zoom)
    min_lat, max_lon = num2deg(max_tile_x, max_tile_y, zoom)
    return (min_lon, min_lat, max_lon, max_lat)


def latlon_to_pixel(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """Convert lat/lon to pixel coordinates."""
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
    """Darken RGB color."""
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)


class NavFeature:
    """Parsed NAV feature."""
    def __init__(self):
        self.geom_type = 0
        self.color_rgb565 = 0xFFFF
        self.zoom_priority = 0
        self.coords = []  # List of (lon, lat) floats

    @property
    def min_zoom(self):
        return self.zoom_priority >> 4

    @property
    def priority(self):
        return (self.zoom_priority & 0x0F) * 7


def read_nav_tile(path: str) -> List[NavFeature]:
    """Read NAV tile file and return list of features."""
    features = []

    try:
        with open(path, 'rb') as f:
            # Read header (20 bytes)
            magic = f.read(4)
            if magic != NAV_MAGIC:
                logger.warning(f"Invalid magic in {path}")
                return features

            version = struct.unpack('<B', f.read(1))[0]
            feature_count = struct.unpack('<H', f.read(2))[0]
            reserved = struct.unpack('<B', f.read(1))[0]
            bbox = struct.unpack('<4i', f.read(16))

            # Read features
            for _ in range(feature_count):
                feature = NavFeature()

                # Feature header (6 bytes)
                feature.geom_type = struct.unpack('<B', f.read(1))[0]
                feature.color_rgb565 = struct.unpack('<H', f.read(2))[0]
                feature.zoom_priority = struct.unpack('<B', f.read(1))[0]
                coord_count = struct.unpack('<H', f.read(2))[0]

                # Read coordinates
                for _ in range(coord_count):
                    lon_int = struct.unpack('<i', f.read(4))[0]
                    lat_int = struct.unpack('<i', f.read(4))[0]
                    lon = lon_int / COORD_SCALE
                    lat = lat_int / COORD_SCALE
                    feature.coords.append((lon, lat))

                # For polygons, read ring info
                if feature.geom_type == GEOM_POLYGON:
                    ring_count = struct.unpack('<B', f.read(1))[0]
                    for _ in range(ring_count):
                        ring_end = struct.unpack('<H', f.read(2))[0]

                features.append(feature)

    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")

    return features


class NAVViewer:
    """NAV tile viewer."""

    def __init__(self, nav_dir: str):
        self.nav_dir = nav_dir
        self.available_zooms: Set[int] = set()
        self._index_tiles()

        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 14
        self.bbox = None
        self.background_color = (255, 255, 255)
        self.fill_polygons = True
        self.show_tile_grid = False
        self.last_query_stats = {}
        self.cached_features = None
        self.selected_feature = None

    def _index_tiles(self):
        """Index available zoom levels."""
        if not os.path.isdir(self.nav_dir):
            logger.error(f"Directory not found: {self.nav_dir}")
            return

        for name in os.listdir(self.nav_dir):
            zoom_path = os.path.join(self.nav_dir, name)
            if os.path.isdir(zoom_path) and name.isdigit():
                zoom = int(name)
                for x_name in os.listdir(zoom_path):
                    x_path = os.path.join(zoom_path, x_name)
                    if os.path.isdir(x_path) and x_name.isdigit():
                        self.available_zooms.add(zoom)
                        break

        if self.available_zooms:
            zooms_str = ", ".join(map(str, sorted(self.available_zooms)))
            logger.info(f"Available zoom levels: {zooms_str}")

    def _get_tiles_for_viewport(self) -> List[Tuple[int, int]]:
        """Get list of tiles for current viewport."""
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        center_tile_x = int(center_x)
        center_tile_y = int(center_y)

        tiles = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy
                max_tile = 2 ** self.zoom - 1
                if 0 <= tile_x <= max_tile and 0 <= tile_y <= max_tile:
                    tiles.append((tile_x, tile_y))

        return tiles

    def _get_tile_path(self, x: int, y: int) -> str:
        """Get file path for a tile."""
        return os.path.join(self.nav_dir, str(self.zoom), str(x), f"{y}.nav")

    def set_center(self, lat: float, lon: float, zoom: int = None):
        """Set viewport center."""
        self.center_lat = lat
        self.center_lon = lon
        if zoom is not None:
            self.zoom = zoom
        self.bbox = get_bbox_for_viewport(self.center_lat, self.center_lon, self.zoom)

    def query_features(self) -> List[NavFeature]:
        """Query features from tiles."""
        if self.bbox is None:
            return []

        import time
        start = time.time()

        tiles = self._get_tiles_for_viewport()
        all_features = []
        tiles_loaded = 0
        tiles_missing = 0

        for tile_x, tile_y in tiles:
            tile_path = self._get_tile_path(tile_x, tile_y)
            if os.path.exists(tile_path):
                features = read_nav_tile(tile_path)
                # Filter by zoom
                features = [f for f in features if f.min_zoom <= self.zoom]
                all_features.extend(features)
                tiles_loaded += 1
            else:
                tiles_missing += 1

        elapsed = (time.time() - start) * 1000

        self.last_query_stats = {
            'tiles_loaded': tiles_loaded,
            'tiles_missing': tiles_missing,
            'features': len(all_features),
            'time_ms': elapsed
        }

        return all_features

    def render_to_surface(self, surface: pygame.Surface):
        """Render features to pygame surface."""
        if self.bbox is None:
            return

        surface.fill(self.background_color)

        features = self.query_features()
        if not features:
            self.cached_features = None
            return

        # Sort by priority
        features.sort(key=lambda f: f.priority)
        self.cached_features = features

        for feature in features:
            self._render_feature(surface, feature)

        if self.show_tile_grid:
            self._draw_tile_grid(surface)

    def _render_feature(self, surface: pygame.Surface, feature: NavFeature):
        """Render a single feature."""
        if not feature.coords:
            return

        color = rgb565_to_rgb888(feature.color_rgb565)

        if feature.geom_type == GEOM_POINT:
            self._render_point(surface, feature, color)
        elif feature.geom_type == GEOM_LINESTRING:
            self._render_linestring(surface, feature, color)
        elif feature.geom_type == GEOM_POLYGON:
            self._render_polygon(surface, feature, color)

    def _render_point(self, surface: pygame.Surface, feature: NavFeature, color: Tuple[int, int, int]):
        """Render point."""
        if feature.coords:
            lon, lat = feature.coords[0]
            px, py = latlon_to_pixel(lat, lon, self.bbox)
            if 0 <= px < VIEWPORT_SIZE and 0 <= py < VIEWPORT_SIZE:
                pygame.draw.circle(surface, color, (px, py), 3)

    def _render_linestring(self, surface: pygame.Surface, feature: NavFeature, color: Tuple[int, int, int]):
        """Render linestring."""
        points = []
        for lon, lat in feature.coords:
            px, py = latlon_to_pixel(lat, lon, self.bbox)
            points.append((px, py))

        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 1)

    def _render_polygon(self, surface: pygame.Surface, feature: NavFeature, color: Tuple[int, int, int]):
        """Render polygon."""
        points = []
        for lon, lat in feature.coords:
            px, py = latlon_to_pixel(lat, lon, self.bbox)
            points.append((px, py))

        if len(points) >= 3:
            if self.fill_polygons:
                pygame.draw.polygon(surface, color, points)
                border_color = darken_color(color, 0.4)
                pygame.draw.polygon(surface, border_color, points, 1)
            else:
                pygame.draw.polygon(surface, color, points, 1)

    def _draw_tile_grid(self, surface: pygame.Surface):
        """Draw tile grid overlay."""
        grid_color = (100, 100, 100)
        font = pygame.font.SysFont(None, 14)

        for i in range(4):
            x = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (x, 0), (x, VIEWPORT_SIZE), 1)
        for i in range(4):
            y = i * TILE_SIZE
            pygame.draw.line(surface, grid_color, (0, y), (VIEWPORT_SIZE, y), 1)

        tiles = self._get_tiles_for_viewport()
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        center_tile_x = int(center_x)
        center_tile_y = int(center_y)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                tile_x = center_tile_x + dx
                tile_y = center_tile_y + dy
                screen_x = (dx + 1) * TILE_SIZE + 5
                screen_y = (dy + 1) * TILE_SIZE + 5
                tile_path = self._get_tile_path(tile_x, tile_y)
                exists = os.path.exists(tile_path)
                color = (0, 100, 0) if exists else (150, 50, 50)
                label = font.render(f"{tile_x}/{tile_y}", True, color)
                surface.blit(label, (screen_x, screen_y))

    def identify_feature_at(self, pixel_x: int, pixel_y: int) -> Optional[dict]:
        """Identify feature at pixel coordinates."""
        if self.cached_features is None or self.bbox is None:
            return None

        min_lon, min_lat, max_lon, max_lat = self.bbox
        lon = min_lon + (pixel_x / VIEWPORT_SIZE) * (max_lon - min_lon)
        lat = max_lat - (pixel_y / VIEWPORT_SIZE) * (max_lat - min_lat)

        # Search in reverse priority order
        for feature in reversed(self.cached_features):
            if self._point_in_feature(lon, lat, feature):
                return {
                    'color': f"#{rgb565_to_rgb888(feature.color_rgb565)[0]:02x}{rgb565_to_rgb888(feature.color_rgb565)[1]:02x}{rgb565_to_rgb888(feature.color_rgb565)[2]:02x}",
                    'min_zoom': feature.min_zoom,
                    'priority': feature.priority,
                    'geom_type': ['?', 'Point', 'Line', 'Polygon'][feature.geom_type],
                    'coords': len(feature.coords)
                }

        return None

    def _point_in_feature(self, lon: float, lat: float, feature: NavFeature) -> bool:
        """Check if point is in/near feature."""
        if feature.geom_type == GEOM_POLYGON:
            # Simple point-in-polygon test
            n = len(feature.coords)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = feature.coords[i]
                xj, yj = feature.coords[j]
                if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside
        elif feature.geom_type == GEOM_LINESTRING:
            # Check distance to line segments
            tolerance = (self.bbox[2] - self.bbox[0]) / VIEWPORT_SIZE * 5
            for i in range(len(feature.coords) - 1):
                x1, y1 = feature.coords[i]
                x2, y2 = feature.coords[i + 1]
                dist = self._point_to_segment_dist(lon, lat, x1, y1, x2, y2)
                if dist < tolerance:
                    return True
        elif feature.geom_type == GEOM_POINT and feature.coords:
            x, y = feature.coords[0]
            tolerance = (self.bbox[2] - self.bbox[0]) / VIEWPORT_SIZE * 10
            if abs(lon - x) < tolerance and abs(lat - y) < tolerance:
                return True
        return False

    def _point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def draw_button(surface, text, rect, bg_color, fg_color, border_color, font):
    """Draw a button."""
    pygame.draw.rect(surface, bg_color, rect, border_radius=8)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=8)
    label = font.render(text, True, fg_color)
    text_rect = label.get_rect(center=rect.center)
    surface.blit(label, text_rect)


def main():
    parser = argparse.ArgumentParser(description='NAV Tile Viewer')
    parser.add_argument('nav_dir', help='Directory with NAV tiles')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--zoom', type=int, default=14, help='Zoom level')

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        logger.error("pygame required")
        sys.exit(1)

    viewer = NAVViewer(args.nav_dir)
    viewer.set_center(args.lat, args.lon, args.zoom)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"NAV Viewer - {os.path.basename(args.nav_dir)}")

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
                        dragging = True
                        drag_start = (mx, my)
                        drag_center_start = (viewer.center_lat, viewer.center_lon)

                elif event.button == 3:
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

            # Query stats
            stats_y = info_y + 70
            screen.blit(font_small.render("Query Stats:", True, info_color), (VIEWPORT_SIZE + 10, stats_y))
            if viewer.last_query_stats:
                s = viewer.last_query_stats
                screen.blit(font_small.render(f"  Tiles: {s.get('tiles_loaded', 0)}/{s.get('tiles_loaded', 0) + s.get('tiles_missing', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 18))
                screen.blit(font_small.render(f"  Features: {s.get('features', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 32))
                screen.blit(font_small.render(f"  Time: {s.get('time_ms', 0):.0f}ms", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 46))

            # Selected feature
            feature_y = stats_y + 80
            screen.blit(font_small.render("Click Feature:", True, info_color), (VIEWPORT_SIZE + 10, feature_y))
            if viewer.selected_feature:
                line_y = feature_y + 18
                for key, value in viewer.selected_feature.items():
                    text = f"  {key}: {value}"
                    screen.blit(font_small.render(text, True, (100, 200, 100)), (VIEWPORT_SIZE + 10, line_y))
                    line_y += 14
            else:
                screen.blit(font_small.render("  (right-click)", True, (100, 100, 100)), (VIEWPORT_SIZE + 10, feature_y + 18))

            # Status bar
            pygame.draw.rect(screen, (30, 30, 30), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
            screen.blit(font_small.render("NAV Format - IceNav Navigation Tiles", True, (200, 200, 200)), (10, VIEWPORT_SIZE + 10))

            if viewer.available_zooms:
                zooms_str = f"Available: {min(viewer.available_zooms)}-{max(viewer.available_zooms)}"
                screen.blit(font_small.render(zooms_str, True, (150, 150, 150)), (10, VIEWPORT_SIZE + 30))

            pygame.display.flip()
            need_redraw = False

        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
