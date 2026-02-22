#!/usr/bin/env python3
"""
NAV Tile Viewer - ESP32 Map Simulator (Packed Containers Version)

Displays map tiles from consolidated Zxx.nav packed files.
Simulates the ESP32 rendering pipeline with offset-based lookup.

Usage:
    python tile_viewer.py maps_dir --lat 42.5063 --lon 1.5218 [--zoom 14]
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

# Format constants
PACK_MAGIC = b'NPK1'
NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3

def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile

def num2deg(xtile: float, ytile: float, zoom: int) -> Tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    r = ((c >> 11) & 0x1F) << 3
    g = ((c >> 5) & 0x3F) << 2
    b = (c & 0x1F) << 3
    return (r, g, b)

def darken_color(rgb: Tuple[int, int, int], amount: float = 0.15) -> Tuple[int, int, int]:
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)

def zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)

def read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): return result, offset
        shift += 7

class NavFeature:
    def __init__(self):
        self.geom_type = 0
        self.color_rgb565 = 0xFFFF
        self.zoom_priority = 0
        self.width = 1
        self.bbox = (0, 0, 0, 0)
        self.coords: List[Tuple[int, int]] = []
        self.ring_ends: List[int] = []
        self.tile_x = 0
        self.tile_y = 0

    @property
    def min_zoom(self) -> int: return self.zoom_priority >> 4
    @property
    def priority(self) -> int: return self.zoom_priority & 0x0F

    def get_rings(self) -> List[List[Tuple[int, int]]]:
        if not self.ring_ends: return [self.coords]
        rings = []
        start = 0
        for end in self.ring_ends:
            rings.append(self.coords[start:end])
            start = end
        return rings

class NavPack:
    """Represents a single Zxx.nav pack file with its index."""
    def __init__(self, path: str):
        self.path = path
        self.zoom = 0
        self.index: Dict[Tuple[int, int], Tuple[int, int]] = {} # (x,y) -> (offset, size)
        self.file_handle = None
        self._load_index()

    def _load_index(self):
        try:
            self.file_handle = open(self.path, 'rb')
            magic = self.file_handle.read(4)
            if magic != PACK_MAGIC:
                logger.error(f"Invalid Pack magic in {self.path}")
                return
            self.zoom = struct.unpack('<B', self.file_handle.read(1))[0]
            count = struct.unpack('<I', self.file_handle.read(4))[0]
            # Read entire index table
            index_data = self.file_handle.read(count * 16)
            for i in range(count):
                entry = struct.unpack_from('<IIII', index_data, i * 16)
                self.index[(entry[0], entry[1])] = (entry[2], entry[3])
            logger.info(f"Loaded Pack {os.path.basename(self.path)}: Zoom {self.zoom}, {count} tiles.")
        except Exception as e:
            logger.error(f"Failed to load pack {self.path}: {e}")

    def get_tile_features(self, x: int, y: int) -> List[NavFeature]:
        if (x, y) not in self.index: return []
        offset, size = self.index[(x, y)]
        self.file_handle.seek(offset)
        data = self.file_handle.read(size)
        return self._parse_tile_data(data, x, y)

    def _parse_tile_data(self, data: bytes, tile_x: int, tile_y: int) -> List[NavFeature]:
        features = []
        if len(data) < 22 or data[:4] != NAV_MAGIC: return features
        feature_count = struct.unpack('<H', data[4:6])[0]
        pos = 22 # Skip tile header
        for _ in range(feature_count):
            if pos + 13 > len(data): break
            feat = NavFeature()
            feat.tile_x, feat.tile_y = tile_x, tile_y
            feat.geom_type = data[pos]
            feat.color_rgb565 = struct.unpack_from('<H', data, pos + 1)[0]
            feat.zoom_priority = data[pos + 3]
            feat.width = data[pos + 4]
            feat.bbox = struct.unpack_from('<BBBB', data, pos + 5)
            coord_count = struct.unpack_from('<H', data, pos + 9)[0]
            payload_size = struct.unpack_from('<H', data, pos + 11)[0]
            pos += 13
            payload = data[pos:pos + payload_size]
            pos += payload_size
            # Decode coords
            offset = 0
            lx, ly = 0, 0
            for _ in range(coord_count):
                dx, offset = read_varint(payload, offset)
                dy, offset = read_varint(payload, offset)
                dx, dy = zigzag_decode(dx), zigzag_decode(dy)
                px, py = lx + dx, ly + dy
                feat.coords.append((px, py))
                lx, ly = px, py
            # Decode rings
            if feat.geom_type == GEOM_POLYGON and offset < len(payload):
                ring_count = struct.unpack_from('<H', payload, offset)[0]
                offset += 2
                for _ in range(ring_count):
                    feat.ring_ends.append(struct.unpack_from('<H', payload, offset)[0])
                    offset += 2
            features.append(feat)
        return features

    def __del__(self):
        if self.file_handle: self.file_handle.close()

class NAVViewer:
    def __init__(self, nav_dir: str):
        self.nav_dir = nav_dir
        self.packs: Dict[int, NavPack] = {} # zoom -> NavPack
        self._load_packs()
        self.center_lat, self.center_lon = 0.0, 0.0
        self.zoom = 14
        self.background_color = (255, 255, 255)
        self.fill_polygons = True
        self.show_tile_grid = False
        self.last_query_stats = {}
        self.cached_features = []
        self.selected_feature = None
        self.last_viewport_key = None
        self.cached_query_features = []

    def _load_packs(self):
        if not os.path.isdir(self.nav_dir): return
        for f in os.listdir(self.nav_dir):
            if f.startswith('Z') and f.endswith('.nav'):
                path = os.path.join(self.nav_dir, f)
                pack = NavPack(path)
                if pack.zoom > 0: self.packs[pack.zoom] = pack

    def _get_tiles_for_viewport(self) -> List[Tuple[int, int]]:
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        min_tx, max_tx = int(math.floor(cx - 1.5)), int(math.floor(cx + 1.5))
        min_ty, max_ty = int(math.floor(cy - 1.5)), int(math.floor(cy + 1.5))
        tiles = []
        max_tile = (2 ** self.zoom) - 1
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                if 0 <= tx <= max_tile and 0 <= ty <= max_tile: tiles.append((tx, ty))
        return tiles

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        lat1, lon1 = num2deg(cx - 1.5, cy - 1.5, self.zoom)
        lat2, lon2 = num2deg(cx + 1.5, cy + 1.5, self.zoom)
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def set_center(self, lat: float, lon: float, zoom: int = None):
        self.center_lat, self.center_lon = lat, lon
        if zoom is not None: self.zoom = zoom
        self.last_viewport_key = None

    def query_features(self) -> List[NavFeature]:
        current_key = (self.zoom, round(self.center_lat, 6), round(self.center_lon, 6))
        if self.last_viewport_key == current_key: return self.cached_query_features
        start = time.time()
        tiles = self._get_tiles_for_viewport()
        all_features = []
        loaded = 0
        if self.zoom in self.packs:
            pack = self.packs[self.zoom]
            for tx, ty in tiles:
                feats = pack.get_tile_features(tx, ty)
                if feats:
                    all_features.extend(feats)
                    loaded += 1
        self.last_query_stats = {'tiles_loaded': loaded, 'tiles_total': len(tiles), 'features': len(all_features), 'time_ms': (time.time() - start) * 1000}
        self.last_viewport_key = current_key
        self.cached_query_features = all_features
        return all_features

    def render_to_surface(self, surface: pygame.Surface):
        surface.fill(self.background_color)
        features = self.query_features()
        if not features:
            self.cached_features = []
            return
        self.cached_features = features
        features_by_tile = {}
        for f in features:
            tile_key = (f.tile_x, f.tile_y)
            if tile_key not in features_by_tile: features_by_tile[tile_key] = []
            features_by_tile[tile_key].append(f)
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = cx - 1.5, cy - 1.5
        for (tx, ty), tile_features in features_by_tile.items():
            sx, sy = int((tx - tl_x) * TILE_SIZE), int((ty - tl_y) * TILE_SIZE)
            surface.set_clip(pygame.Rect(sx, sy, TILE_SIZE, TILE_SIZE))
            tile_features.sort(key=lambda f: f.priority)
            for feature in tile_features: self._render_feature(surface, feature)
            surface.set_clip(None)
        if self.show_tile_grid: self._draw_tile_grid(surface)

    def _tile_coord_to_screen(self, tx: int, ty: int, px: int, py: int) -> Tuple[int, int]:
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = cx - 1.5, cy - 1.5
        fx, fy = (tx - tl_x) + (px / 4096.0), (ty - tl_y) + (py / 4096.0)
        return int(fx * TILE_SIZE), int(fy * TILE_SIZE)

    def _render_feature(self, surface: pygame.Surface, feature: NavFeature):
        if not feature.coords: return
        color = rgb565_to_rgb888(feature.color_rgb565)
        if feature.geom_type == GEOM_POINT:
            sx, sy = self._tile_coord_to_screen(feature.tile_x, feature.tile_y, feature.coords[0][0], feature.coords[0][1])
            if 0 <= sx < VIEWPORT_SIZE and 0 <= sy < VIEWPORT_SIZE: pygame.draw.circle(surface, color, (sx, sy), 3)
        elif feature.geom_type == GEOM_LINESTRING:
            pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in feature.coords]
            if len(pts) >= 2: pygame.draw.lines(surface, color, False, pts, max(1, feature.width))
        elif feature.geom_type == GEOM_POLYGON:
            rings = feature.get_rings()
            for i, ring in enumerate(rings):
                pts = [self._tile_coord_to_screen(feature.tile_x, feature.tile_y, x, y) for x, y in ring]
                if len(pts) >= 3:
                    if self.fill_polygons:
                        pygame.draw.polygon(surface, color if i == 0 else self.background_color, pts)
                        if i == 0: pygame.draw.polygon(surface, darken_color(color), pts, 1)
                    else: pygame.draw.polygon(surface, color, pts, 1)

    def _draw_tile_grid(self, surface: pygame.Surface):
        grid_color = (100, 100, 100)
        font = pygame.font.SysFont(None, 14)
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        tl_x, tl_y = cx - 1.5, cy - 1.5
        for ty in range(int(math.floor(tl_y)), int(math.floor(tl_y + 4))):
            for tx in range(int(math.floor(tl_x)), int(math.floor(tl_x + 4))):
                sx, sy = int((tx - tl_x) * TILE_SIZE), int((ty - tl_y) * TILE_SIZE)
                pygame.draw.rect(surface, grid_color, (sx, sy, TILE_SIZE, TILE_SIZE), 1)
                label = font.render(f"{tx}/{ty}", True, (0, 0, 0), (255, 255, 255))
                surface.blit(label, (sx + 5, sy + 5))

    def identify_feature_at(self, pixel_x: int, pixel_y: int) -> Optional[dict]:
        if not self.cached_features: return None
        cx, cy = deg2num(self.center_lat, self.center_lon, self.zoom)
        fx, fy = (cx - 1.5) + (pixel_x / TILE_SIZE), (cy - 1.5) + (pixel_y / TILE_SIZE)
        for feature in reversed(self.cached_features):
            if self._point_in_feature(fx, fy, feature):
                bx1, by1, bx2, by2 = feature.bbox
                return {'type': ['?', 'Point', 'Line', 'Polygon'][feature.geom_type], 'color': f"#{rgb565_to_rgb888(feature.color_rgb565)[0]:02x}{rgb565_to_rgb888(feature.color_rgb565)[1]:02x}{rgb565_to_rgb888(feature.color_rgb565)[2]:02x}", 'zoom': feature.min_zoom, 'pts': len(feature.coords), 'bbox': f"({bx1},{by1})-({bx2},{by2})"}
        return None

    def _point_in_feature(self, fx: float, fy: float, feature: NavFeature) -> bool:
        f_pts = [(feature.tile_x + px/4096.0, feature.tile_y + py/4096.0) for px, py in feature.coords]
        if feature.geom_type == GEOM_POLYGON:
            inside = False
            for i in range(len(f_pts)):
                p1, p2 = f_pts[i], f_pts[i - 1]
                if ((p1[1] > fy) != (p2[1] > fy)) and (fx < (p2[0] - p1[0]) * (fy - p1[1]) / (p2[1] - p1[1]) + p1[0]): inside = not inside
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
    pygame.draw.rect(surface, bg_color, rect, border_radius=8)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=8)
    label = font.render(text, True, fg_color)
    surface.blit(label, label.get_rect(center=rect.center))

def main():
    parser = argparse.ArgumentParser(description='NAV-PACK Tile Viewer')
    parser.add_argument('nav_dir', help='Directory with Zxx.nav pack files')
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--zoom', type=int, default=14, help='Zoom level')
    args = parser.parse_args()
    if not PYGAME_AVAILABLE: sys.exit(1)
    viewer = NAVViewer(args.nav_dir)
    viewer.set_center(args.lat, args.lon, args.zoom)
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Packed Tiles Viewer - {os.path.basename(args.nav_dir)}")
    font_small = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()
    viewport_surface = pygame.Surface((VIEWPORT_SIZE, VIEWPORT_SIZE))
    button_margin, button_height, button_width = 10, 35, TOOLBAR_WIDTH - 20
    bg_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin, button_width, button_height)
    fill_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 2 + button_height, button_width, button_height)
    grid_button_rect = pygame.Rect(VIEWPORT_SIZE + 10, button_margin * 3 + button_height * 2, button_width, button_height)
    dragging, drag_start, drag_center_start, need_redraw, running = False, None, None, True, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                pan_speed = 0.01 / (2 ** (viewer.zoom - 10))
                if event.key == pygame.K_LEFT: viewer.set_center(viewer.center_lat, viewer.center_lon - pan_speed); need_redraw = True
                elif event.key == pygame.K_RIGHT: viewer.set_center(viewer.center_lat, viewer.center_lon + pan_speed); need_redraw = True
                elif event.key == pygame.K_UP: viewer.set_center(viewer.center_lat + pan_speed, viewer.center_lon); need_redraw = True
                elif event.key == pygame.K_DOWN: viewer.set_center(viewer.center_lat - pan_speed, viewer.center_lon); need_redraw = True
                elif event.key == pygame.K_LEFTBRACKET:
                    if (viewer.zoom - 1) in viewer.packs: viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom - 1); need_redraw = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    if (viewer.zoom + 1) in viewer.packs: viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom + 1); need_redraw = True
                elif event.key == pygame.K_b: viewer.background_color = (0, 0, 0) if viewer.background_color == (255, 255, 255) else (255, 255, 255); need_redraw = True
                elif event.key == pygame.K_f: viewer.fill_polygons = not viewer.fill_polygons; need_redraw = True
                elif event.key == pygame.K_g: viewer.show_tile_grid = not viewer.show_tile_grid; need_redraw = True
                elif event.key in (pygame.K_q, pygame.K_ESCAPE): running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    if bg_button_rect.collidepoint(mx, my): viewer.background_color = (0, 0, 0) if viewer.background_color == (255, 255, 255) else (255, 255, 255); need_redraw = True
                    elif fill_button_rect.collidepoint(mx, my): viewer.fill_polygons = not viewer.fill_polygons; need_redraw = True
                    elif grid_button_rect.collidepoint(mx, my): viewer.show_tile_grid = not viewer.show_tile_grid; need_redraw = True
                    elif mx < VIEWPORT_SIZE and my < VIEWPORT_SIZE: dragging, drag_start, drag_center_start = True, (mx, my), (viewer.center_lat, viewer.center_lon)
                elif event.button == 3 and mx < VIEWPORT_SIZE and my < VIEWPORT_SIZE: viewer.selected_feature = viewer.identify_feature_at(mx, my); need_redraw = True
                elif event.button == 4 and (viewer.zoom + 1) in viewer.packs: viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom + 1); need_redraw = True
                elif event.button == 5 and (viewer.zoom - 1) in viewer.packs: viewer.set_center(viewer.center_lat, viewer.center_lon, viewer.zoom - 1); need_redraw = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                min_lon, min_lat, max_lon, max_lat = viewer.bbox
                viewer.set_center(drag_center_start[0] + (event.pos[1] - drag_start[1]) * ((max_lat - min_lat) / VIEWPORT_SIZE), drag_center_start[1] - (event.pos[0] - drag_start[0]) * ((max_lon - min_lon) / VIEWPORT_SIZE)); need_redraw = True
        if need_redraw:
            viewer.render_to_surface(viewport_surface)
            screen.fill((50, 50, 50))
            screen.blit(viewport_surface, (0, 0))
            pygame.draw.rect(screen, (30, 30, 30), (VIEWPORT_SIZE, 0, TOOLBAR_WIDTH, VIEWPORT_SIZE))
            draw_button(screen, "BG: White" if viewer.background_color == (255, 255, 255) else "BG: Black", bg_button_rect, (50, 50, 50), (255, 255, 255), (100, 100, 100), font_small)
            draw_button(screen, f"Fill: {'ON' if viewer.fill_polygons else 'OFF'}", fill_button_rect, (50, 50, 50), (255, 255, 255), (100, 100, 100), font_small)
            draw_button(screen, f"Grid: {'ON' if viewer.show_tile_grid else 'OFF'}", grid_button_rect, (50, 50, 50), (255, 255, 255), (100, 100, 100), font_small)
            info_y, info_color = button_margin * 4 + button_height * 3 + 20, (200, 200, 200)
            screen.blit(font_small.render(f"Lat: {viewer.center_lat:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y))
            screen.blit(font_small.render(f"Lon: {viewer.center_lon:.6f}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 18))
            screen.blit(font_small.render(f"Zoom: {viewer.zoom}", True, info_color), (VIEWPORT_SIZE + 10, info_y + 36))
            stats_y = info_y + 70
            screen.blit(font_small.render("Query Stats:", True, info_color), (VIEWPORT_SIZE + 10, stats_y))
            if viewer.last_query_stats:
                s = viewer.last_query_stats
                screen.blit(font_small.render(f"  Tiles: {s.get('tiles_loaded', 0)}/{s.get('tiles_total', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 18))
                screen.blit(font_small.render(f"  Features: {s.get('features', 0)}", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 32))
                screen.blit(font_small.render(f"  Time: {s.get('time_ms', 0):.0f}ms", True, (150, 150, 150)), (VIEWPORT_SIZE + 10, stats_y + 46))
            feature_y = stats_y + 80
            screen.blit(font_small.render("Selected Feature:", True, info_color), (VIEWPORT_SIZE + 10, feature_y))
            if viewer.selected_feature:
                line_y = feature_y + 18
                for k, v in viewer.selected_feature.items():
                    screen.blit(font_small.render(f"  {k}: {v}", True, (100, 200, 100)), (VIEWPORT_SIZE + 10, line_y))
                    line_y += 14
            else: screen.blit(font_small.render("  (Right-click to select)", True, (100, 100, 100)), (VIEWPORT_SIZE + 10, feature_y + 18))
            pygame.draw.rect(screen, (30, 30, 30), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
            screen.blit(font_small.render("Packed Format - Optimized ESP32 Maps", True, (200, 200, 200)), (10, VIEWPORT_SIZE + 10))
            if viewer.packs: screen.blit(font_small.render(f"Available Zooms: {min(viewer.packs.keys())}-{max(viewer.packs.keys())}", True, (150, 150, 150)), (10, VIEWPORT_SIZE + 30))
            pygame.display.flip()
            need_redraw = False
        clock.tick(30)
    pygame.quit()

if __name__ == '__main__': main()
