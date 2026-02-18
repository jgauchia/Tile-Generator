#!/usr/bin/env python3
"""
NAV Tile Viewer - ESP32 Map Simulator

Displays NAV binary tiles using the optimized int16 relative coordinate format.
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
    from pygame import gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not found. Install with: pip install pygame")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# UI Constants
TILE_SIZE = 256
VIEWPORT_SIZE = 768
TOOLBAR_WIDTH = 350  # Increased for better stats/legend visibility
STATUSBAR_HEIGHT = 60
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

# NAV format constants
NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3
GEOM_TEXT = 4


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
        self.needs_casing = False  # Bit 7 of width byte - for two-pass rendering
        self.bbox = (0, 0, 0, 0)
        self.coords: List[Tuple[int, int]] = []  # All points for all rings
        self.ring_ends: List[int] = []  # Indices where each ring ends
        self.tile_x = 0
        self.tile_y = 0
        self.text: str = ''  # For GEOM_TEXT features
        self.font_size: int = 0  # 0=small, 1=medium, 2=large

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


def read_nav_tile(path: str, tile_x: int, tile_y: int) -> List[NavFeature]:
    """Read optimized NAV tile file and return list of features with rings."""
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

                feature.geom_type = struct.unpack('<B', f.read(1))[0]
                feature.color_rgb565 = struct.unpack('<H', f.read(2))[0]
                feature.zoom_priority = struct.unpack('<B', f.read(1))[0]

                # Width byte encoding: bit 7 = casing flag, bits 0-6 = actual width
                width_byte = struct.unpack('<B', f.read(1))[0]
                feature.needs_casing = (width_byte & 0x80) != 0
                feature.width = width_byte & 0x7F

                feature.bbox = struct.unpack('<BBBB', f.read(4))
                coord_count = struct.unpack('<H', f.read(2))[0]
                f.read(1)

                if feature.geom_type == GEOM_TEXT:
                    # Read raw data block (coord_count * 4 bytes)
                    data = f.read(coord_count * 4)
                    if len(data) >= 5:
                        px, py = struct.unpack('<hh', data[0:4])
                        feature.coords.append((px, py))
                        text_len = data[4]
                        feature.text = data[5:5 + text_len].decode('utf-8', errors='replace')
                        feature.font_size = feature.width
                else:
                    for _ in range(coord_count):
                        px, py = struct.unpack('<hh', f.read(4))
                        feature.coords.append((px, py))

                    if feature.geom_type == GEOM_POLYGON:
                        ring_count = struct.unpack('<H', f.read(2))[0]
                        for _ in range(ring_count):
                            feature.ring_ends.append(struct.unpack('<H', f.read(2))[0])

                features.append(feature)
    except Exception as e:
        logger.debug(f"Error reading tile {path}: {e}")
    return features


class NAVViewer:
    """Main viewer application logic."""
    def __init__(self, nav_dir: str, config_file: Optional[str] = None):
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

        # New: Priority filter
        self.priority_filter_min = 0
        self.priority_filter_max = 15

        # New: Color to OSM tag mapping
        self.color_to_tags: Dict[str, List[str]] = {}
        if config_file:
            self._load_config(config_file)

    def _index_tiles(self):
        """Build an index of available zoom levels on disk."""
        if not os.path.isdir(self.nav_dir):
            return
        for name in os.listdir(self.nav_dir):
            if name.isdigit() and os.path.isdir(os.path.join(self.nav_dir, name)):
                self.available_zooms.add(int(name))

    def _load_config(self, config_file: str):
        """Load features.json and build color to tag mapping."""
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Build reverse mapping: RGB565 hex → OSM tags
            for tag, props in config.items():
                if isinstance(props, dict) and 'color' in props:
                    hex_color = props['color']
                    # Convert to RGB565 hex
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                    rgb888 = rgb565_to_rgb888(rgb565)
                    color_key = f"#{rgb888[0]:02x}{rgb888[1]:02x}{rgb888[2]:02x}"

                    if color_key not in self.color_to_tags:
                        self.color_to_tags[color_key] = []
                    self.color_to_tags[color_key].append(tag)

            logger.info(f"Loaded {len(self.color_to_tags)} unique colors from config")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

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

        # New: Apply priority filter
        features = [f for f in features if self.priority_filter_min <= f.priority <= self.priority_filter_max]

        # CRITICAL FIX: Sort ALL features globally by priority before rendering
        # This ensures proper Z-order (landuse/forests below, roads above)
        features.sort(key=lambda f: f.priority)

        # FOUR-PASS RENDERING for bridges over roads
        # Bridge features (nibble 15 + casing) must be drawn AFTER all at-grade road cores,
        # otherwise the at-grade road cores cover the bridge casings.
        BRIDGE_NIBBLE = 15

        # Pass 1: At-grade road casings (priority < BRIDGE_NIBBLE)
        for feature in features:
            if feature.geom_type == GEOM_LINESTRING and feature.needs_casing and feature.priority < BRIDGE_NIBBLE:
                self._render_road_casing(surface, feature)

        # Pass 2: All non-text, non-bridge features (polygons + at-grade road cores + railways without casing)
        for feature in features:
            if feature.geom_type == GEOM_TEXT:
                continue
            if feature.geom_type == GEOM_LINESTRING and feature.needs_casing and feature.priority == BRIDGE_NIBBLE:
                continue  # Skip bridges, rendered in pass 3-4
            self._render_feature(surface, feature)

        # Pass 3: Bridge casings (priority == BRIDGE_NIBBLE with casing flag)
        for feature in features:
            if feature.geom_type == GEOM_LINESTRING and feature.needs_casing and feature.priority == BRIDGE_NIBBLE:
                self._render_road_casing(surface, feature)

        # Pass 4: Bridge cores (priority == BRIDGE_NIBBLE with casing flag)
        for feature in features:
            if feature.geom_type == GEOM_LINESTRING and feature.needs_casing and feature.priority == BRIDGE_NIBBLE:
                self._render_feature(surface, feature)

        # Pass 5: Text labels on top of everything
        for feature in features:
            if feature.geom_type == GEOM_TEXT:
                self._render_feature(surface, feature)

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

    def _draw_smooth_line(self, surface: pygame.Surface, color: Tuple[int, int, int],
                          points: List[Tuple[int, int]], width: int):
        """Draw line using oriented rectangles for square ends, no AA, no joints."""
        if len(points) < 2:
            return

        # Very thin lines: use simple line
        if width <= 1:
            pygame.draw.lines(surface, color, False, points, 1)
            return

        half_w = width / 2.0

        # Draw each segment as an oriented rectangle
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = math.hypot(dx, dy)

            if dist == 0:
                continue

            # Perpendicular unit vector
            nx = -dy / dist
            ny = dx / dist

            # Rectangle corners
            poly_pts = [
                (p1[0] + nx * half_w, p1[1] + ny * half_w),
                (p1[0] - nx * half_w, p1[1] - ny * half_w),
                (p2[0] - nx * half_w, p2[1] - ny * half_w),
                (p2[0] + nx * half_w, p2[1] + ny * half_w)
            ]

            # Draw filled polygon only (no AA, no joints)
            pygame.gfxdraw.filled_polygon(surface, poly_pts, color)

    def _render_road_casing(self, surface: pygame.Surface, feature: NavFeature):
        """Render road casing (border) for two-pass rendering - Pass 1 only."""
        if not feature.coords or len(feature.coords) < 2:
            return

        # Convert coordinates to screen space
        pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y)
               for x, y in feature.coords]

        # Get road color and darken it for the casing
        road_color = rgb565_to_rgb888(feature.color_rgb565)
        casing_color = darken_color(road_color, amount=0.3)  # 30% darker than road color

        # Draw casing: wider than the road core with smooth joins
        casing_width = feature.width + 2
        self._draw_smooth_line(surface, casing_color, pts, casing_width)

    def _render_feature(self, surface: pygame.Surface, feature: NavFeature):
        if not feature.coords: return
        color = rgb565_to_rgb888(feature.color_rgb565)

        if feature.geom_type == GEOM_TEXT:
            px, py = feature.coords[0]
            sx, sy = self._tile_coord_to_screen(feature.tile_x, feature.tile_y, px, py)
            if 0 <= sx < VIEWPORT_SIZE and 0 <= sy < VIEWPORT_SIZE and feature.text:
                font_sizes = {0: 15, 1: 17, 2: 20}
                size = font_sizes.get(feature.font_size, 15)
                text_font = pygame.font.SysFont(None, size)
                lines = feature.text.split('\n')
                line_height = text_font.get_linesize()
                total_height = line_height * len(lines)
                y_start = sy - total_height // 2
                for i, line in enumerate(lines):
                    rendered = text_font.render(line, True, color)
                    surface.blit(rendered, (sx - rendered.get_width() // 2, y_start + i * line_height))
            return

        elif feature.geom_type == GEOM_POINT:
            px, py = feature.coords[0]
            sx, sy = self._tile_coord_to_screen(feature.tile_x, feature.tile_y, px, py)
            if 0 <= sx < VIEWPORT_SIZE and 0 <= sy < VIEWPORT_SIZE:
                pygame.draw.circle(surface, color, (sx, sy), 3)

        elif feature.geom_type == GEOM_LINESTRING:
            pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in feature.coords]
            if len(pts) >= 2:
                # DEBUG: Log roads with priority >= 11 being drawn
                if feature.priority >= 11:
                    # Check if any point is in viewport
                    in_viewport = any(0 <= x < VIEWPORT_SIZE and 0 <= y < VIEWPORT_SIZE for x, y in pts)
                    print(f"[DRAW LINE] tile={feature.tile_x}/{feature.tile_y}, priority={feature.priority}, "
                          f"screen_pts={len(pts)}, in_viewport={in_viewport}, first_pt={pts[0]}, width={feature.width}")
                # Draw road core with smooth joins (casing already drawn in pass 1 for roads with needs_casing flag)
                self._draw_smooth_line(surface, color, pts, max(1, feature.width))

        elif feature.geom_type == GEOM_POLYGON:
            rings = feature.get_rings()
            if not rings:
                print(f"[POLYGON EMPTY] No rings for priority={feature.priority}, color=#{color[0]:02x}{color[1]:02x}{color[2]:02x}")
                return

            # DEBUG: Print info for green-ish polygons (grassland #cdebb0 = RGB 205,235,176)
            if 195 <= color[0] <= 210 and 225 <= color[1] <= 240 and 170 <= color[2] <= 185:
                first_ring = rings[0] if rings else []
                print(f"[DEBUG GRASSLAND] tile={feature.tile_x}/{feature.tile_y}, color={color}, "
                      f"rgb565=0x{feature.color_rgb565:04x}, rings={len(rings)}, "
                      f"pts_first_ring={len(first_ring)}, "
                      f"min_zoom={feature.min_zoom}, priority={feature.priority}")
                if first_ring and len(first_ring) >= 3:
                    # Convert to screen coords to check visibility
                    screen_pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in first_ring]
                    xs = [p[0] for p in screen_pts]
                    ys = [p[1] for p in screen_pts]
                    print(f"  SCREEN: X=[{min(xs)}, {max(xs)}], Y=[{min(ys)}, {max(ys)}], "
                          f"viewport=[0, {VIEWPORT_SIZE}]")

            # DEBUG: Print info for blue-ish polygons (water)
            if 150 <= color[2] <= 235 and 160 <= color[1] <= 220 and 150 <= color[0] <= 180:
                first_ring = rings[0] if rings else []
                print(f"DEBUG POLYGON water: tile={feature.tile_x}/{feature.tile_y}, color={color}, "
                      f"rgb565=0x{feature.color_rgb565:04x}, rings={len(rings)}, "
                      f"pts_first_ring={len(first_ring)}, "
                      f"min_zoom={feature.min_zoom}, priority={feature.priority}")
                if first_ring and len(first_ring) >= 3:
                    print(f"  RAW coords: first={first_ring[0]}, second={first_ring[1]}, last={first_ring[-1]}")

            # Only draw exterior ring (first ring)
            # Inner rings (holes) are NOT drawn - they stay transparent
            # This allows features below to show through (e.g., islands in rivers)
            for i, ring in enumerate(rings):
                if i > 0 and self.fill_polygons:
                    continue  # Skip holes when filling (keep transparent)

                pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in ring]

                # DEBUG grassland rendering
                if 195 <= color[0] <= 210 and 225 <= color[1] <= 240 and 170 <= color[2] <= 185:
                    xs = [p[0] for p in pts] if pts else []
                    ys = [p[1] for p in pts] if pts else []
                    print(f"  [RENDER ATTEMPT] ring {i}, raw_pts={len(ring)}, screen_pts={len(pts)}, tile={feature.tile_x}/{feature.tile_y}")
                    if len(pts) >= 3:
                        print(f"    COORDS: X=[{min(xs)}, {max(xs)}], Y=[{min(ys)}, {max(ys)}], viewport=[0, {VIEWPORT_SIZE}]")
                    else:
                        print(f"    SKIPPED: not enough screen points!")

                if len(pts) >= 3:
                    # DEBUG grassland drawing
                    if 195 <= color[0] <= 210 and 225 <= color[1] <= 240 and 170 <= color[2] <= 185:
                        mode = "FILLED" if self.fill_polygons else "OUTLINE"
                        print(f"    DRAWING {mode}: color={color}")

                    # DEBUG: Print when drawing water polygon
                    if 150 <= color[2] <= 235 and 160 <= color[1] <= 220 and 150 <= color[0] <= 180:
                        if len(pts) >= 80:  # Large polygons - show full extent
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            print(f"  -> SCREEN coords ring {i}: first={pts[0]}, second={pts[1]}, last={pts[-1]}")
                            print(f"     ALL {len(pts)} points: X=[{min(xs)}, {max(xs)}] ({max(xs)-min(xs)} px), Y=[{min(ys)}, {max(ys)}] ({max(ys)-min(ys)} px)")
                        else:
                            print(f"  -> SCREEN coords ring {i}: first={pts[0]}, second={pts[1]}, last={pts[-1]}")

                    if self.fill_polygons:
                        pygame.draw.polygon(surface, color, pts, 0)  # 0 = filled
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

        # Test features in priority order (highest first) for correct hit-testing
        sorted_features = sorted(self.cached_features, key=lambda f: f.priority, reverse=True)
        for feature in sorted_features:
            if self._point_in_feature(fx, fy, feature):
                bx1, by1, bx2, by2 = feature.bbox
                color_hex = f"#{rgb565_to_rgb888(feature.color_rgb565)[0]:02x}{rgb565_to_rgb888(feature.color_rgb565)[1]:02x}{rgb565_to_rgb888(feature.color_rgb565)[2]:02x}"

                # Find OSM tags from color
                osm_tags = self.color_to_tags.get(color_hex, ['unknown'])

                return {
                    'type': ['?', 'Point', 'Line', 'Polygon', 'Text'][feature.geom_type if feature.geom_type < 5 else 0],
                    'color': color_hex,
                    'tags': ', '.join(osm_tags[:3]),  # Show up to 3 tags
                    'zoom': feature.min_zoom,
                    'priority': feature.priority,
                    'pts': len(feature.coords),
                    'bbox': f"({bx1},{by1})-({bx2},{by2})"
                }
        return None

    def get_feature_stats(self) -> Dict:
        """Calculate detailed statistics about loaded features."""
        if not self.cached_features:
            return {}

        stats = {
            'by_type': {},
            'by_priority': {},
            'by_color': {},
        }

        for f in self.cached_features:
            # By type
            type_name = ['?', 'Point', 'Line', 'Polygon', 'Text'][f.geom_type if f.geom_type < 5 else 0]
            stats['by_type'][type_name] = stats['by_type'].get(type_name, 0) + 1

            # By priority
            stats['by_priority'][f.priority] = stats['by_priority'].get(f.priority, 0) + 1

            # By color
            color_hex = f"#{rgb565_to_rgb888(f.color_rgb565)[0]:02x}{rgb565_to_rgb888(f.color_rgb565)[1]:02x}{rgb565_to_rgb888(f.color_rgb565)[2]:02x}"
            stats['by_color'][color_hex] = stats['by_color'].get(color_hex, 0) + 1

        return stats

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
    parser.add_argument('--config', type=str, help='Features JSON config file (optional, for OSM tag mapping)')

    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        logger.error("pygame required")
        sys.exit(1)

    viewer = NAVViewer(args.nav_dir, args.config)
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
    stats_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 4 + button_height * 3, button_width, button_height)
    legend_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 5 + button_height * 4, button_width, button_height)

    # Priority filter sliders
    slider_y_base = button_margin * 6 + button_height * 5 + 40
    slider_height = 20
    min_priority_slider_rect = pygame.Rect(VIEWPORT_SIZE + 10, slider_y_base, button_width, slider_height)
    max_priority_slider_rect = pygame.Rect(VIEWPORT_SIZE + 10, slider_y_base + 40, button_width, slider_height)

    show_stats_window = False
    show_legend_window = False

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
                elif event.key == pygame.K_r:
                    # Force reload tiles (clear cache)
                    viewer.last_viewport_key = None
                    need_redraw = True
                elif event.key == pygame.K_s:
                    show_stats_window = not show_stats_window
                    show_legend_window = False  # Close legend if stats opened
                    need_redraw = True
                elif event.key == pygame.K_l:
                    show_legend_window = not show_legend_window
                    show_stats_window = False  # Close stats if legend opened
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
                    elif stats_button_rect.collidepoint(mx, my):
                        show_stats_window = not show_stats_window
                        need_redraw = True
                    elif legend_button_rect.collidepoint(mx, my):
                        show_legend_window = not show_legend_window
                        need_redraw = True
                    elif min_priority_slider_rect.collidepoint(mx, my):
                        # Set min priority from slider position
                        ratio = (mx - min_priority_slider_rect.x) / min_priority_slider_rect.width
                        viewer.priority_filter_min = int(ratio * 15)
                        need_redraw = True
                    elif max_priority_slider_rect.collidepoint(mx, my):
                        # Set max priority from slider position
                        ratio = (mx - max_priority_slider_rect.x) / max_priority_slider_rect.width
                        viewer.priority_filter_max = int(ratio * 15)
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

            stats_text = "Stats" + (" ✓" if show_stats_window else "")
            draw_button(screen, stats_text, stats_button_rect, button_bg, button_fg, button_border, font_small)

            legend_text = "Legend" + (" ✓" if show_legend_window else "")
            draw_button(screen, legend_text, legend_button_rect, button_bg, button_fg, button_border, font_small)

            info_color = (200, 200, 200)

            # Priority filter sliders
            slider_label_y = slider_y_base - 25
            screen.blit(font_small.render("Priority Filter:", True, info_color), (VIEWPORT_SIZE + 10, slider_label_y))

            # Min slider
            pygame.draw.rect(screen, (60, 60, 60), min_priority_slider_rect)
            min_handle_x = VIEWPORT_SIZE + 10 + int((viewer.priority_filter_min / 15) * button_width)
            pygame.draw.circle(screen, (150, 150, 255), (min_handle_x, slider_y_base + slider_height // 2), 8)
            screen.blit(font_small.render(f"Min: {viewer.priority_filter_min}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, slider_y_base + 22))

            # Max slider
            pygame.draw.rect(screen, (60, 60, 60), max_priority_slider_rect)
            max_handle_x = VIEWPORT_SIZE + 10 + int((viewer.priority_filter_max / 15) * button_width)
            pygame.draw.circle(screen, (255, 150, 150), (max_handle_x, slider_y_base + 40 + slider_height // 2), 8)
            screen.blit(font_small.render(f"Max: {viewer.priority_filter_max}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, slider_y_base + 62))

            info_y = slider_y_base + 100

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
            if not show_stats_window and not show_legend_window:
                screen.blit(font_small.render("Selected Feature:", True, info_color), (VIEWPORT_SIZE + 10, feature_y))
                if viewer.selected_feature:
                    line_y = feature_y + 18
                    for key, value in viewer.selected_feature.items():
                        text = f"  {key}: {value}"
                        screen.blit(font_small.render(text, True, (100, 200, 100)), (VIEWPORT_SIZE + 10, line_y))
                        line_y += 14
                else:
                    screen.blit(font_small.render("  (Right-click to select)", True, (100, 100, 100)), (VIEWPORT_SIZE + 10, feature_y + 18))

            # Stats window
            elif show_stats_window:
                feature_stats = viewer.get_feature_stats()
                screen.blit(font_small.render("Feature Statistics:", True, (255, 255, 100)), (VIEWPORT_SIZE + 10, feature_y))
                line_y = feature_y + 20

                screen.blit(font_small.render("By Type:", True, info_color), (VIEWPORT_SIZE + 10, line_y))
                line_y += 15
                for type_name, count in sorted(feature_stats.get('by_type', {}).items()):
                    screen.blit(font_small.render(f"  {type_name}: {count}", True, (150, 150, 150)), (VIEWPORT_SIZE + 15, line_y))
                    line_y += 13

                line_y += 5
                screen.blit(font_small.render("By Priority:", True, info_color), (VIEWPORT_SIZE + 10, line_y))
                line_y += 15
                for priority, count in sorted(feature_stats.get('by_priority', {}).items()):
                    screen.blit(font_small.render(f"  P{priority}: {count}", True, (150, 150, 150)), (VIEWPORT_SIZE + 15, line_y))
                    line_y += 13
                    if line_y > VIEWPORT_SIZE - 20: break

            # Legend window
            elif show_legend_window:
                screen.blit(font_small.render("Color Legend:", True, (255, 255, 100)), (VIEWPORT_SIZE + 10, feature_y))
                line_y = feature_y + 20

                feature_stats = viewer.get_feature_stats()
                sorted_colors = sorted(feature_stats.get('by_color', {}).items(), key=lambda x: -x[1])[:15]  # Top 15

                for color_hex, count in sorted_colors:
                    # Draw color swatch
                    r = int(color_hex[1:3], 16)
                    g = int(color_hex[3:5], 16)
                    b = int(color_hex[5:7], 16)
                    pygame.draw.rect(screen, (r, g, b), (VIEWPORT_SIZE + 10, line_y, 15, 12))

                    # Draw tags
                    tags = viewer.color_to_tags.get(color_hex, ['?'])
                    tag_text = tags[0] if len(tags) == 1 else f"{tags[0]}..."
                    screen.blit(font_small.render(f"{tag_text} ({count})", True, (150, 150, 150)), (VIEWPORT_SIZE + 30, line_y))
                    line_y += 14
                    if line_y > VIEWPORT_SIZE - 20: break

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
