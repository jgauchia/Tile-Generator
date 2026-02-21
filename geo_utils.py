"""
Geographic utilities for NAV tile generation.

Coordinates, colors, simplification, feature-to-config lookups.
"""

import math
import logging
from typing import Dict, List, Tuple, Set, Any, Optional

try:
    import shapely
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from constants import LAYER_MAPPING

logger = logging.getLogger(__name__)

# Cache for zoom level parameters
_ZOOM_PARAMS_CACHE = {}


def _get_zoom_params(zoom: int) -> Dict:
    """Get cached zoom parameters or compute them."""
    if zoom not in _ZOOM_PARAMS_CACHE:
        n = 2.0 ** zoom
        _ZOOM_PARAMS_CACHE[zoom] = {
            'n': n,
            'lon_scale': n / 360.0,
            'lon_offset': 180.0
        }
    return _ZOOM_PARAMS_CACHE[zoom]


def meters_to_pixels(width_meters: float, zoom: int, lat: float = 45.0) -> int:
    """Convert width in meters to pixels at given zoom level.

    Uses approximation for given latitude (default 45° for Europe).
    Formula: meters_per_pixel ≈ 156543 * cos(lat) / 2^zoom
    """
    meters_per_pixel = 156543.0 * math.cos(math.radians(lat)) / (2 ** zoom)
    pixels = int(width_meters / meters_per_pixel + 0.5)
    return max(1, min(15, pixels))  # Clamp to 1-15


def lon_to_tile_x(lon: float, zoom: int) -> int:
    """Convert longitude to tile X coordinate."""
    params = _get_zoom_params(zoom)
    return int((lon + params['lon_offset']) * params['lon_scale'])


def lat_to_tile_y(lat: float, zoom: int) -> int:
    """Convert latitude to tile Y coordinate."""
    params = _get_zoom_params(zoom)
    lat_rad = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * params['n'])


def tile_y_to_lat(ty: int, zoom: int) -> float:
    """Convert tile Y coordinate to latitude (north edge of tile)."""
    n = 2.0 ** zoom
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n)))
    return math.degrees(lat_rad)


def lat_to_mercator_y(lat: float) -> float:
    """Convert latitude (degrees) to Web Mercator Y (radians)."""
    lat_rad = math.radians(lat)
    lat_rad = max(-0.999 * math.pi / 2, min(0.999 * math.pi / 2, lat_rad))
    return math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))


def get_feature_tiles(coords: List[Tuple[float, float]], zoom: int, is_polygon: bool = False) -> Set[Tuple[int, int]]:
    """Get all tiles that a feature intersects at given zoom level."""
    tiles = set()

    if is_polygon and len(coords) >= 3:
        # Calculate min/max without creating intermediate lists
        min_lon = max_lon = coords[0][0]
        min_lat = max_lat = coords[0][1]

        for lon, lat in coords[1:]:
            if lon < min_lon: min_lon = lon
            elif lon > max_lon: max_lon = lon
            if lat < min_lat: min_lat = lat
            elif lat > max_lat: max_lat = lat

        min_x = lon_to_tile_x(min_lon, zoom)
        max_x = lon_to_tile_x(max_lon, zoom)
        min_y = lat_to_tile_y(max_lat, zoom)
        max_y = lat_to_tile_y(min_lat, zoom)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tiles.add((x, y))
    else:
        for lon, lat in coords:
            x = lon_to_tile_x(lon, zoom)
            y = lat_to_tile_y(lat, zoom)
            tiles.add((x, y))

    return tiles


def get_layer_for_tags(tags: Dict[str, str]) -> Optional[str]:
    """Determine which layer a feature belongs to based on its tags."""

    # Explicit rule for all water-related features
    if (tags.get('natural') == 'water' or
        tags.get('natural') == 'bay' or
        'waterway' in tags or
        'water' in tags or
        tags.get('landuse') == 'reservoir'):
        return 'water'

    # Explicit rule for all highways and railways (BEFORE boundary filter,
    # because some roads run along administrative boundaries and carry both tags)
    if 'highway' in tags:
        return 'roads'
    if 'railway' in tags:
        return 'roads'

    # Do not create polygons for abstract features like boundaries or place names
    # Exception: place=island/islet are real geographic features
    if 'boundary' in tags or 'admin_level' in tags:
        return None
    if 'place' in tags and tags['place'] not in ('island', 'islet'):
        return None

    # Explicit rule for buildings to ensure they are always on top of scenery
    if 'building' in tags or tags.get('aeroway') == 'hangar':
        return 'buildings'

    for layer_name, feature_keys in LAYER_MAPPING.items():
        for feature_key in feature_keys:
            if '=' in feature_key:
                key, value = feature_key.split('=', 1)
                if key in tags and tags[key] == value:
                    return layer_name
            else:
                if feature_key in tags:
                    return layer_name
    return None


def get_config_value_for_tags(
    tags: Dict[str, str],
    config: Dict,
    attribute: str,
    default: Any
) -> Any:
    """
    Generic helper to get configuration values for feature tags, with key priority.
    """
    preferred_keys = ['building', 'natural', 'waterway', 'landuse', 'highway', 'railway', 'water']

    # 1. Prioritize preferred keys
    for key in preferred_keys:
        if key in tags:
            value = tags[key]
            # Try exact match first (key=value)
            feature_key = f"{key}={value}"
            if feature_key in config and isinstance(config[feature_key], dict):
                return config[feature_key].get(attribute, default)

            # Then try key-only match
            if key in config and isinstance(config[key], dict):
                return config[key].get(attribute, default)

    # 1.5. Surface overrides leisure for sport pitches only
    if 'leisure' in tags and 'surface' in tags:
        surface_key = f"surface={tags['surface']}"
        if surface_key in config and isinstance(config[surface_key], dict):
            return config[surface_key].get(attribute, default)

    # 2. Check remaining tags
    for key, value in tags.items():
        if key in preferred_keys:
            continue

        feature_key = f"{key}={value}"
        if feature_key in config and isinstance(config[feature_key], dict):
            return config[feature_key].get(attribute, default)

        if key in config and isinstance(config[key], dict):
            return config[key].get(attribute, default)

    return default


def get_zoom_for_tags(tags: Dict[str, str], config: Dict) -> int:
    """Get minimum zoom level for feature based on config."""
    return get_config_value_for_tags(tags, config, 'zoom', 6)


def get_color_for_tags(tags: Dict[str, str], config: Dict) -> str:
    """Get color for feature based on config."""
    return get_config_value_for_tags(tags, config, 'color', '#FFFFFF')


def get_nibble_for_tags(tags: Dict[str, str], config: Dict) -> int:
    """Get z-order nibble (0-15) for feature based on config."""
    return get_config_value_for_tags(tags, config, 'priority', 2)



def hex_to_rgb565(hex_color: str) -> int:
    """Convert hex color to RGB565 format."""
    try:
        if not hex_color or not hex_color.startswith("#"):
            return 0xFFFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    except (ValueError, IndexError):
        logger.warning(f"Invalid hex color format: {hex_color}, using default")
        return 0xFFFF


def lighten_rgb565(color: int, factor: float = 0.4) -> int:
    """Lighten RGB565 color."""
    r = ((color >> 11) & 0x1F) * 255 // 31
    g = ((color >> 5) & 0x3F) * 255 // 63
    b = (color & 0x1F) * 255 // 31
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def darken_rgb565(color: int, factor: float = 0.4) -> int:
    """Darken RGB565 color."""
    r = ((color >> 11) & 0x1F) * 255 // 31
    g = ((color >> 5) & 0x3F) * 255 // 63
    b = (color & 0x1F) * 255 // 31
    r = max(0, int(r * (1 - factor)))
    g = max(0, int(g * (1 - factor)))
    b = max(0, int(b * (1 - factor)))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def pack_zoom_priority(min_zoom: int, priority_nibble: int) -> int:
    """Pack min_zoom and priority into a single byte."""
    return (min(min_zoom, 15) << 4) | (min(priority_nibble, 15) & 0x0F)


def get_simplify_tolerance(zoom: int) -> float:
    """Calculate simplification tolerance based on zoom level."""
    tile_width_degrees = 360.0 / (2.0 ** zoom)
    pixel_size_degrees = tile_width_degrees / 256.0
    return pixel_size_degrees * 0.25  # Reduced from 0.5 to preserve roundabouts and curves


def densify_linestring(coords: List[Tuple[float, float]], max_segment_degrees: float) -> List[Tuple[float, float]]:
    """Add intermediate points to linestring for smoother curves."""
    if len(coords) <= 1 or not SHAPELY_AVAILABLE:
        return coords

    try:
        from shapely.geometry import LineString
        from shapely.ops import segmentize
        line = LineString(coords)
        # Segmentize adds points so no segment is longer than max_segment_degrees
        densified = segmentize(line, max_segment_degrees)
        return list(densified.coords)
    except Exception:
        return coords


