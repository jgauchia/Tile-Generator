"""Headless render of a job's NAVMAP, served as tiles for the page.

The viewer (``tile_viewer.py``) already knows how to draw a pack; this module
only drives it without a display, keeps the tiles around every render and
prepares the area of a job in the background.  The packs are read, never
written.
"""

import math
import os
import sys
import threading
from pathlib import Path

# No display in the service: SDL must be told before pygame is imported.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pygame  # noqa: E402

import tile_viewer  # noqa: E402

# Fonts are all the render asks for: no window, no sound.
pygame.font.init()

TILE_SIZE = 256
VIEWPORT_TILES = 3
VIEWPORT_SIZE = VIEWPORT_TILES * TILE_SIZE


class Preview:
    """The tiles of one job, rendered on demand and kept under its folder."""

    def __init__(self, nav_dir, cache_dir, bbox, zoom_min, zoom_max):
        self.nav_dir = Path(nav_dir)
        self.cache_dir = Path(cache_dir)
        self.bbox = [float(value) for value in bbox]
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._viewer = None

    def cancel(self):
        """The job is gone: no more renders, the cache goes with its packs."""
        self._cancelled.set()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def covers(self, zoom, x, y):
        """Whether a tile touches the area of the job."""
        if not self.zoom_min <= zoom <= self.zoom_max:
            return False
        min_x, min_y, max_x, max_y = self._tile_range(zoom)
        return min_x <= x <= max_x and min_y <= y <= max_y

    def tile(self, zoom, x, y):
        """PNG bytes of one tile, drawing the 3x3 viewport around it if needed."""
        path = self._cache_path(zoom, x, y)
        if path.is_file():
            return path.read_bytes()
        with self._lock:
            if not path.is_file() and not self.cancelled:
                self._render_viewport(zoom, x, y)
        return path.read_bytes()

    def warm(self):
        """Prepare the area of the job at the zoom that fits it and its neighbours."""
        for zoom in sorted(self._warm_zooms()):
            for x, y in self._viewport_centers(zoom):
                if self.cancelled:
                    return
                path = self._cache_path(zoom, x, y)
                if path.is_file():
                    continue
                with self._lock:
                    if self.cancelled:
                        return
                    if not path.is_file():
                        self._render_viewport(zoom, x, y)

    def _render_viewport(self, zoom, x, y):
        """Draw the viewport centred on a tile and keep the tiles of the job.

        The viewer paints its landcover over whole tiles, so every tile is cut
        to the rectangle that was asked for: the preview covers the area of the
        job and nothing of it spills over the limits drawn on the map.
        """
        lat, lon = tile_viewer.num2deg(x + 0.5, y + 0.5, zoom)
        viewer = self._viewer_for()
        viewer.set_center(lat, lon, zoom)
        surface = pygame.Surface((VIEWPORT_SIZE, VIEWPORT_SIZE))
        viewer.render_to_surface(surface)
        keep = self._job_rect(zoom, x, y)
        for i in range(-1, 2):
            for j in range(-1, 2):
                tile_x = x + i
                tile_y = y + j
                if not self.covers(zoom, tile_x, tile_y):
                    continue
                tile_rect = pygame.Rect((i + 1) * TILE_SIZE, (j + 1) * TILE_SIZE,
                                        TILE_SIZE, TILE_SIZE)
                part = tile_rect.clip(keep)
                if part.width <= 0 or part.height <= 0:
                    continue
                tile = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                tile.blit(surface, (part.left - tile_rect.left, part.top - tile_rect.top),
                          part)
                path = self._cache_path(zoom, tile_x, tile_y)
                path.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(tile, str(path))

    def _job_rect(self, zoom, x, y):
        """The rectangle of the job in the pixels of the viewport around a tile.

        Rounded inwards (floor on the left and top edges, ceiling on the right
        and bottom ones) so the tiles never paint outside the rectangle.
        """
        west, south, east, north = self.bbox
        left, top = tile_viewer.deg2num(north, west, zoom)
        right, bottom = tile_viewer.deg2num(south, east, zoom)
        # The viewport starts at the tile (x - 1, y - 1) of the tile grid.
        px = (left - (x - 1)) * TILE_SIZE
        py = (top - (y - 1)) * TILE_SIZE
        return pygame.Rect(math.floor(px), math.floor(py),
                           math.ceil((right - (x - 1)) * TILE_SIZE) - math.floor(px),
                           math.ceil((bottom - (y - 1)) * TILE_SIZE) - math.floor(py))

    def _viewer_for(self):
        if self._viewer is None:
            self._viewer = tile_viewer.NAVViewer(str(self.nav_dir))
        return self._viewer

    def _cache_path(self, zoom, x, y):
        return self.cache_dir / f"Z{zoom}" / str(x) / f"{y}.png"

    def _tile_range(self, zoom):
        west, south, east, north = self.bbox
        min_x, min_y = tile_viewer.deg2num(north, west, zoom)
        max_x, max_y = tile_viewer.deg2num(south, east, zoom)
        return (int(math.floor(min_x)), int(math.floor(min_y)),
                int(math.floor(max_x)), int(math.floor(max_y)))

    def _fit_zoom(self):
        """The highest zoom whose 3x3 viewport still covers the whole area."""
        for zoom in range(self.zoom_max, self.zoom_min - 1, -1):
            min_x, min_y, max_x, max_y = self._tile_range(zoom)
            if max_x - min_x < VIEWPORT_TILES and max_y - min_y < VIEWPORT_TILES:
                return zoom
        return self.zoom_min

    def _warm_zooms(self):
        fit = self._fit_zoom()
        return {zoom for zoom in (fit - 1, fit, fit + 1)
                if self.zoom_min <= zoom <= self.zoom_max}

    def _viewport_centers(self, zoom):
        min_x, min_y, max_x, max_y = self._tile_range(zoom)
        centers = []
        for y in range(min_y, max_y + 1, VIEWPORT_TILES):
            for x in range(min_x, max_x + 1, VIEWPORT_TILES):
                centers.append((min(x + 1, max_x), min(y + 1, max_y)))
        return centers
