#!/usr/bin/env python3
"""
NAV Tile Viewer - ESP32 Map Simulator

Displays NAV binary tiles using the optimized Delta VarInt relative coordinate format.
Simulates the ESP32 rendering pipeline in a 768x768 viewport.

Usage:
    python tile_viewer.py nav_dir --lat 42.5063 --lon 1.5218 [--zoom 14]
"""

import os
import sys
import math
import struct
import argparse
import logging
import time
from typing import Dict, List, Tuple, Optional, Set

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Install with: pip install pygame")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI Constants
TILE_SIZE = 256
VIEWPORT_SIZE = 768
TOOLBAR_WIDTH = 200
STATUSBAR_HEIGHT = 60
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

# NAV format constants
NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    """Convert lat/lon to fractional tile numbers (Web Mercator)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile


def num2deg(xtile: float, ytile: float, zoom: int) -> Tuple[float, float]:
    """Convert fractional tile numbers to lat/lon."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    """Convert 16-bit RGB565 to 24-bit RGB888."""
    r = ((c >> 11) & 0x1F) << 3
    g = ((c >> 5) & 0x3F) << 2
    b = (c & 0x1F) << 3
    return (r, g, b)


def darken_color(rgb: Tuple[int, int, int], amount: float = 0.15) -> Tuple[int, int, int]:
    """Apply subtle darkening to an RGB color (closer to OSM style)."""
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)


class NavFeature:
    """Parsed NAV feature with relative coordinates and multiple rings."""
    def __init__(self):
        self.geom_type = 0
        self.color_rgb565 = 0xFFFF
        self.zoom_priority = 0
        self.width = 1
        self.bbox = (0, 0, 0, 0)
        self.coords: List[Tuple[int, int]] = []  # All points for all rings
        self.ring_ends: List[int] = []  # Indices where each ring ends
        self.tile_x = 0
        self.tile_y = 0

    @property
    def min_zoom(self) -> int:
        return self.zoom_priority >> 4

    @property
    def priority(self) -> int:
        return self.zoom_priority & 0x0F

    def get_rings(self) -> List[List[Tuple[int, int]]]:
        """Split the single coords list into multiple rings."""
        if not self.ring_ends:
            return [self.coords]
        rings = []
        start = 0
        for end in self.ring_ends:
            rings.append(self.coords[start:end])
            start = end
        return rings



def zigzag_decode(n: int) -> int:
    """ZigZag decode an integer."""
    return (n >> 1) ^ -(n & 1)


def read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    """Read a VarInt from buffer at offset. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if offset >= len(buffer):
            raise IndexError("VarInt read out of bounds")
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7


def read_nav_tile(path: str, tile_x: int, tile_y: int) -> List[NavFeature]:
    """Read optimized NAV tile file (VarInt Delta Encoded) and return list of features."""
    features = []
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != NAV_MAGIC:
                return features

            feature_count = struct.unpack('<H', f.read(2))[0]
            f.read(16)  # Skip global tile BBox

            for _ in range(feature_count):
                feature = NavFeature()
                feature.tile_x = tile_x
                feature.tile_y = tile_y

                # Read Header (13 bytes)
                header_data = f.read(13)
                if len(header_data) < 13:
                    break
                    
                feature.geom_type = header_data[0]
                feature.color_rgb565 = struct.unpack('<H', header_data[1:3])[0]
                feature.zoom_priority = header_data[3]
                feature.width = header_data[4]
                feature.bbox = struct.unpack('<BBBB', header_data[5:9])
                coord_count = struct.unpack('<H', header_data[9:11])[0]
                payload_size = struct.unpack('<H', header_data[11:13])[0]

                # Read Payload
                payload = f.read(payload_size)
                if len(payload) < payload_size:
                    break
                
                # Decode Coordinates (VarInt + ZigZag + Delta)
                offset = 0
                last_x, last_y = 0, 0
                
                for _ in range(coord_count):
                    dx, offset = read_varint(payload, offset)
                    dy, offset = read_varint(payload, offset)
                    
                    dx = zigzag_decode(dx)
                    dy = zigzag_decode(dy)
                    
                    px = last_x + dx
                    py = last_y + dy
                    
                    feature.coords.append((px, py))
                    last_x, last_y = px, py

                # Decode Ring Info (if Polygon)
                if feature.geom_type == GEOM_POLYGON and offset < len(payload):
                    # Remaining payload is ring info
                    # We can use struct to unpack from the remaining bytes
                    # First 2 bytes are ring_count
                    try:
                        ring_count = struct.unpack_from('<H', payload, offset)[0]
                        offset += 2
                        for _ in range(ring_count):
                            ring_end = struct.unpack_from('<H', payload, offset)[0]
                            feature.ring_ends.append(ring_end)
                            offset += 2
                    except struct.error:
                        pass # Should not happen if file is valid

                features.append(feature)
    except Exception as e:
        logger.debug(f"Error reading tile {path}: {e}")
    return features


class NAVViewer:
    """Main viewer application logic."""
    def __init__(self, nav_dir: str):
        self.nav_dir = nav_dir
        self.available_zooms: Set[int] = set()
        self._index_tiles()

        self.center_lat = 0.0
        self.center_lon = 0.0
        self.zoom = 14
        self.background_color = (255, 255, 255)
        self.fill_polygons = True
        self.show_tile_grid = False
        
        self.last_query_stats = {}
        self.cached_features: Optional[List[NavFeature]] = None
        self.selected_feature = None
        self.last_viewport_key = None
        self.cached_query_features: List[NavFeature] = []

    def _index_tiles(self):
        """Build an index of available zoom levels on disk."""
        if not os.path.isdir(self.nav_dir):
            return
        for name in os.listdir(self.nav_dir):
            if name.isdigit() and os.path.isdir(os.path.join(self.nav_dir, name)):
                self.available_zooms.add(int(name))

    def _get_tiles_for_viewport(self) -> List[Tuple[int, int]]:
        """Identify which tiles are needed to cover the 768x768 viewport."""
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        # Viewport is 3x3 tiles, but needs 4x4 for coverage when not aligned
        min_tx = int(math.floor(center_x - 1.5))
        max_tx = int(math.floor(center_x + 1.5))
        min_ty = int(math.floor(center_y - 1.5))
        max_ty = int(math.floor(center_y + 1.5))

        tiles = []
        max_tile = (2 ** self.zoom) - 1
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if 0 <= tx <= max_tile and 0 <= ty <= max_tile:
                    tiles.append((tx, ty))
        return tiles

    def _get_tile_path(self, x: int, y: int) -> str:
        return os.path.join(self.nav_dir, str(self.zoom), str(x), f"{y}.nav")

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return the (min_lon, min_lat, max_lon, max_lat) of the current viewport."""
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = center_x - 1.5, center_y - 1.5
        br_x, br_y = center_x + 1.5, center_y + 1.5
        lat1, lon1 = num2deg(tl_x, tl_y, self.zoom)
        lat2, lon2 = num2deg(br_x, br_y, self.zoom)
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def set_center(self, lat: float, lon: float, zoom: int = None):
        self.center_lat = lat
        self.center_lon = lon
        if zoom is not None:
            self.zoom = zoom
        self.last_viewport_key = None # Invalidate cache

    def query_features(self) -> List[NavFeature]:
        """Load features from required tiles for current viewport."""
        current_key = (self.zoom, round(self.center_lat, 6), round(self.center_lon, 6))
        if self.last_viewport_key == current_key:
            return self.cached_query_features

        start = time.time()
        tiles = self._get_tiles_for_viewport()
        all_features = []
        loaded = 0

        for tx, ty in tiles:
            path = self._get_tile_path(tx, ty)
            if os.path.exists(path):
                features = read_nav_tile(path, tx, ty)
                all_features.extend([f for f in features if f.min_zoom <= self.zoom])
                loaded += 1

        self.last_query_stats = {
            'tiles_loaded': loaded,
            'tiles_total': len(tiles),
            'features': len(all_features),
            'time_ms': (time.time() - start) * 1000
        }
        self.last_viewport_key = current_key
        self.cached_query_features = all_features
        return all_features

    def render_to_surface(self, surface: pygame.Surface):
        surface.fill(self.background_color)
        features = self.query_features()
        if not features:
            self.cached_features = []
            return

        self.cached_features = features # Update for identify_feature_at

        # Group features by tile for clipped rendering
        features_by_tile = {}
        for f in features:
            tile_key = (f.tile_x, f.tile_y)
            if tile_key not in features_by_tile:
                features_by_tile[tile_key] = []
            features_by_tile[tile_key].append(f)

        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = center_x - 1.5, center_y - 1.5

        # Render each tile with its own clipping rect
        for (tx, ty), tile_features in features_by_tile.items():
            # Calculate tile viewport position
            sx = int((tx - tl_x) * TILE_SIZE)
            sy = int((ty - tl_y) * TILE_SIZE)
            
            # Set clip to this tile's 256x256 area
            surface.set_clip(pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE))
            
            tile_features.sort(key=lambda f: f.priority)
            for feature in tile_features:
                self._render_feature(surface, feature)
            
            surface.set_clip(None)

        if self.show_tile_grid:
            self._draw_tile_grid(surface)

    def _tile_coord_to_screen(self, tx: int, ty: int, px: int, py: int) -> Tuple[int, int]:
        """Convert relative tile coordinate (0-4096) to viewport pixels."""
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = center_x - 1.5, center_y - 1.5
        
        # Unit position (in tiles) relative to viewport top-left
        fx = (tx - tl_x) + (px / 4096.0)
        fy = (ty - tl_y) + (py / 4096.0)
        
        return int(fx * TILE_SIZE), int(fy * TILE_SIZE)

    def _render_feature(self, surface: pygame.Surface, feature: NavFeature):
        if not feature.coords: return
        color = rgb565_to_rgb888(feature.color_rgb565)

        if feature.geom_type == GEOM_POINT:
            px, py = feature.coords[0]
            sx, sy = self._tile_coord_to_screen(feature.tile_x, feature.tile_y, px, py)
            if 0 <= sx < VIEWPORT_SIZE and 0 <= sy < VIEWPORT_SIZE:
                pygame.draw.circle(surface, color, (sx, sy), 3)

        elif feature.geom_type == GEOM_LINESTRING:
            pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in feature.coords]
            if len(pts) >= 2:
                pygame.draw.lines(surface, color, False, pts, max(1, feature.width))

        elif feature.geom_type == GEOM_POLYGON:
            rings = feature.get_rings()
            if not rings: return
            
            # First ring is exterior, others are holes
            for i, ring in enumerate(rings):
                pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in ring]
                if len(pts) >= 3:
                    if self.fill_polygons:
                        # Draw exterior with feature color, holes with background color
                        ring_color = color if i == 0 else self.background_color
                        pygame.draw.polygon(surface, ring_color, pts)
                        # Only outline the exterior ring, and only if it's not a hole
                        if i == 0:
                            pygame.draw.polygon(surface, darken_color(color), pts, 1)
                    else:
                        pygame.draw.polygon(surface, color, pts, 1)

    def _draw_tile_grid(self, surface: pygame.Surface):
        grid_color = (100, 100, 100)
        font = pygame.font.SysFont(None, 14)
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = center_x - 1.5, center_y - 1.5

        for ty in range(int(math.floor(tl_y)), int(math.floor(tl_y + 4))):
            for tx in range(int(math.floor(tl_x)), int(math.floor(tl_x + 4))):
                sx, sy = int((tx - tl_x) * TILE_SIZE), int((ty - tl_y) * TILE_SIZE)
                pygame.draw.rect(surface, grid_color, (sx, sy, TILE_SIZE, TILE_SIZE), 1)
                
                path = self._get_tile_path(tx, ty)
                exists = os.path.exists(path)
                color = (0, 0, 0) if exists else (150, 50, 50)
                label = font.render(f"{tx}/{ty}", True, color, (255, 255, 255))
                surface.blit(label, (sx + 5, sy + 5))

    def identify_feature_at(self, pixel_x: int, pixel_y: int) -> Optional[dict]:
        if not self.cached_features: return None
        center_x, center_y = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = center_x - 1.5, center_y - 1.5
        fx, fy = tl_x + (pixel_x / TILE_SIZE), tl_y + (pixel_y / TILE_SIZE)

        for feature in reversed(self.cached_features):
            if self._point_in_feature(fx, fy, feature):
                bx1, by1, bx2, by2 = feature.bbox
                return {
                    'type': ['?', 'Point', 'Line', 'Polygon'][feature.geom_type],
                    'color': f"#{rgb565_to_rgb888(feature.color_rgb565)[0]:02x}{rgb565_to_rgb888(feature.color_rgb565)[1]:02x}{rgb565_to_rgb888(feature.color_rgb565)[2]:02x}",
                    'zoom': feature.min_zoom,
                    'pts': len(feature.coords),
                    'bbox': f"({bx1},{by1})-({bx2},{by2})"
                }
        return None

    def _point_in_feature(self, fx: float, fy: float, feature: NavFeature) -> bool:
        # Convert feature to global tile units for intersection test
        f_pts = [(feature.tile_x + px/4096.0, feature.tile_y + py/4096.0) for px, py in feature.coords]
        if feature.geom_type == GEOM_POLYGON:
            inside = False
            for i in range(len(f_pts)):
                p1, p2 = f_pts[i], f_pts[i - 1]
                if ((p1[1] > fy) != (p2[1] > fy)) and (fx < (p2[0] - p1[0]) * (fy - p1[1]) / (p2[1] - p1[1]) + p1[0]):
                    inside = not inside
            return inside
        elif feature.geom_type == GEOM_LINESTRING:
            tol = 5.0 / TILE_SIZE
            for i in range(len(f_pts) - 1):
                p1, p2 = f_pts[i], f_pts[i + 1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                if dx == 0 and dy == 0: d = math.sqrt((fx-p1[0])**2 + (fy-p1[1])**2)
                else:
                    t = max(0, min(1, ((fx - p1[0]) * dx + (fy - p1[1]) * dy) / (dx*dx + dy*dy)))
                    d = math.sqrt((fx - (p1[0] + t*dx))**2 + (fy - (p1[1] + t*dy))**2)
                if d < tol: return True
        return False


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
                    new_zoom = viewer.zoom - 1
                    if new_zoom in viewer.available_zooms:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, new_zoom)
                        need_redraw = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    new_zoom = viewer.zoom + 1
                    if new_zoom in viewer.available_zooms:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, new_zoom)
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
                    new_zoom = viewer.zoom + 1
                    if new_zoom in viewer.available_zooms:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, new_zoom)
                        need_redraw = True
                elif event.button == 5:
                    new_zoom = viewer.zoom - 1
                    if new_zoom in viewer.available_zooms:
                        viewer.set_center(viewer.center_lat, viewer.center_lon, new_zoom)
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

            # Toolbar and Info Panel
            pygame.draw.rect(screen, (30, 30, 30), (VIEWPORT_SIZE, 0, TOOLBAR_WIDTH, VIEWPORT_SIZE))

            button_bg = (50, 50, 50)
            button_fg = (255, 255, 255)
            button_border = (100, 100, 100)

            bg_text = "BG: White" if viewer.background_color == (255, 255, 255) else "BG: Black"
            draw_button(screen, bg_text, bg_button_rect, button_bg, button_fg, button_border, font_small)

            fill_text = f"Fill: {'ON' if viewer.fill_polygons else 'OFF'}"
            draw_button(screen, fill_text, fill_button_rect, button_bg, button_fg, button_border, font_small)

            grid_text = f"Grid: {'ON' if viewer.show_tile_grid else 'OFF'}"
            draw_button(screen, grid_text, grid_button_rect, button_bg, button_fg, button_border, font_small)

            info_y = button_margin * 4 + button_height * 3 + 20
            info_color = (200, 200, 200)

            screen.blit(font_small.render(f"Lat: {viewer.center_lat:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y))
            screen.blit(font_small.render(f"Lon: {viewer.center_lon:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 18))
            screen.blit(font_small.render(f"Zoom: {viewer.zoom}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 36))

            # Query statistics
            stats_y = info_y + 70
            screen.blit(font_small.render("Query Stats:", True, info_color), (VIEWPORT_SIZE + 10, stats_y))
            if viewer.last_query_stats:
                s = viewer.last_query_stats
                screen.blit(font_small.render(f"  Tiles: {s.get('tiles_loaded', 0)}/{s.get('tiles_total', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 18))
                screen.blit(font_small.render(f"  Features: {s.get('features', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 32))
                screen.blit(font_small.render(f"  Time: {s.get('time_ms', 0):.0f}ms", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 46))

            # Feature Identification (Right-click)
            feature_y = stats_y + 80
            screen.blit(font_small.render("Selected Feature:", True, info_color), (VIEWPORT_SIZE + 10, feature_y))
            if viewer.selected_feature:
                line_y = feature_y + 18
                for key, value in viewer.selected_feature.items():
                    text = f"  {key}: {value}"
                    screen.blit(font_small.render(text, True, (100, 200, 100)), (VIEWPORT_SIZE + 10, line_y))
                    line_y += 14
            else:
                screen.blit(font_small.render("  (Right-click to select)", True, (100, 100, 100)), (VIEWPORT_SIZE + 10, feature_y + 18))

            # Status Bar
            pygame.draw.rect(screen, (30, 30, 30), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
            screen.blit(font_small.render("NAV Format - Optimized ESP32 Maps", True, (200, 200, 200)), (10, VIEWPORT_SIZE + 10))

            if viewer.available_zooms:
                zooms_str = f"Available Zooms: {min(viewer.available_zooms)}-{max(viewer.available_zooms)}"
                screen.blit(font_small.render(zooms_str, True, (150, 150, 150)), (10, VIEWPORT_SIZE + 30))

            pygame.display.flip()
            need_redraw = False

        clock.tick(30)

    pygame.quit()


if __name__ == '__main__':
    main()
