#!/usr/bin/env python3
"""
PBF to NAV Tile Converter

Converts OpenStreetMap .pbf files to NAV binary format (.nav) with tile structure.
NAV format uses int32 coordinates (scaled by 1e7) for ~50% size reduction vs FlatGeobuf.

Usage:
    python pbf_to_nav.py input.pbf output_dir features.json [--zoom 6-17]

Output structure:
    output_dir/
    ├── 13/
    │   ├── 4123/
    │   │   ├── 2456.nav
    │   │   └── ...
    │   └── ...
    └── 14/
        └── ...
"""

import os
import sys
import json
import argparse
import logging
import math
import struct
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import time

try:
    import osmium
    from osmium import osm
    import osmium.geom
except ImportError:
    print("Error: osmium not found. Install with: pip install osmium")
    sys.exit(1)

try:
    from shapely.geometry import Polygon
    import shapely.wkb
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not found. Multipolygon support disabled.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NAV format constants
NAV_MAGIC = b'NAV1'
NAV_VERSION = 1
COORD_SCALE = 10000000  # 1e7 for ~1cm precision

# Geometry types
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3

# Layer rendering priority (lower = rendered first = behind)
LAYER_PRIORITY = {
    'water': 10,
    'landuse': 20,
    'terrain': 30,
    'railways': 40,
    'roads': 50,
    'infrastructure': 60,
    'buildings': 70,
    'amenities': 80,
    'places': 90
}

# Layer definitions based on feature types
LAYER_MAPPING = {
    'water': [
        'natural=water', 'natural=coastline', 'natural=bay',
        'waterway=riverbank', 'waterway=dock', 'waterway=boatyard',
        'waterway=river', 'waterway=stream', 'waterway=canal',
        'natural=spring', 'natural=wetland'
    ],
    'landuse': [
        'natural=beach', 'natural=sand', 'natural=wood',
        'landuse=forest', 'natural=forest', 'natural=scrub',
        'natural=heath', 'natural=grassland', 'landuse=meadow',
        'landuse=grass', 'landuse=orchard', 'landuse=vineyard',
        'landuse=farmland', 'landuse=park', 'leisure=park',
        'leisure=nature_reserve', 'leisure=garden', 'leisure=pitch',
        'leisure=golf_course', 'leisure=recreation_ground', 'landuse=recreation_ground',
        'landuse=residential', 'place=suburb',
        'landuse=commercial', 'landuse=retail', 'landuse=industrial',
        'landuse=construction', 'landuse=cemetery', 'landuse=allotments',
        'leisure=stadium', 'leisure=sports_centre', 'leisure=playground',
        'amenity=parking'
    ],
    'roads': [
        'highway=motorway', 'highway=motorway_link',
        'highway=trunk', 'highway=trunk_link',
        'highway=primary', 'highway=primary_link',
        'highway=secondary', 'highway=secondary_link',
        'highway=tertiary', 'highway=tertiary_link',
        'highway=residential', 'highway=living_street',
        'highway=unclassified', 'highway=service',
        'highway=pedestrian', 'highway=track',
        'highway=path', 'highway=footway',
        'highway=cycleway', 'highway=steps',
        'highway=crossing', 'highway=bus_stop'
    ],
    'railways': [
        'railway=rail', 'railway=subway', 'railway=tram'
    ],
    'buildings': [
        'building', 'man_made=tower'
    ],
    'amenities': [
        'amenity=hospital',
        'amenity=school', 'amenity=university',
        'amenity=place_of_worship'
    ],
    'infrastructure': [
        'bridge=yes', 'man_made=bridge',
        'aeroway=runway', 'aeroway=taxiway', 'aeroway=apron',
        'tunnel=yes'
    ],
    'terrain': [
        'natural=peak', 'natural=ridge',
        'natural=volcano', 'natural=cliff',
        'natural=tree_row', 'natural=tree'
    ],
    'places': [
        'place=state', 'place=town',
        'place=village', 'place=hamlet'
    ]
}


def lon_to_tile_x(lon: float, zoom: int) -> int:
    """Convert longitude to tile X coordinate."""
    n = 2.0 ** zoom
    return int((lon + 180.0) / 360.0 * n)


def lat_to_tile_y(lat: float, zoom: int) -> int:
    """Convert latitude to tile Y coordinate."""
    n = 2.0 ** zoom
    lat_rad = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)


def tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    """Get bounding box for a tile (min_lon, min_lat, max_lon, max_lat)."""
    n = 2.0 ** zoom
    min_lon = x / n * 360.0 - 180.0
    max_lon = (x + 1) / n * 360.0 - 180.0
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (min_lon, min_lat, max_lon, max_lat)


def get_feature_tiles(coords: List[Tuple[float, float]], zoom: int, is_polygon: bool = False) -> Set[Tuple[int, int]]:
    """Get all tiles that a feature intersects at given zoom level."""
    tiles = set()

    if is_polygon and len(coords) >= 3:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

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


def get_layer_for_tags(tags: Dict[str, str]) -> str:
    """Determine which layer a feature belongs to based on its tags."""
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


def get_zoom_for_tags(tags: Dict[str, str], config: Dict) -> int:
    """Get minimum zoom level for feature based on config."""
    for key, value in tags.items():
        feature_key = f"{key}={value}"
        if feature_key in config and isinstance(config[feature_key], dict):
            return config[feature_key].get('zoom', 6)
        if key in config and isinstance(config[key], dict):
            return config[key].get('zoom', 6)
    return 6


def get_color_for_tags(tags: Dict[str, str], config: Dict) -> str:
    """Get color for feature based on config."""
    for key, value in tags.items():
        feature_key = f"{key}={value}"
        if feature_key in config and isinstance(config[feature_key], dict):
            return config[feature_key].get('color', '#FFFFFF')
        if key in config and isinstance(config[key], dict):
            return config[key].get('color', '#FFFFFF')
    return '#FFFFFF'


def get_priority_for_tags(tags: Dict[str, str], config: Dict) -> int:
    """Get rendering priority for feature based on config."""
    for key, value in tags.items():
        feature_key = f"{key}={value}"
        if feature_key in config and isinstance(config[feature_key], dict):
            return config[feature_key].get('priority', 50)
        if key in config and isinstance(config[key], dict):
            return config[key].get('priority', 50)
    return 50


def hex_to_rgb565(hex_color: str) -> int:
    """Convert hex color to RGB565 format."""
    try:
        if not hex_color or not hex_color.startswith("#"):
            return 0xFFFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
    except:
        return 0xFFFF


def pack_zoom_priority(min_zoom: int, priority: int) -> int:
    """Pack min_zoom and priority into a single byte."""
    zoom_nibble = min(min_zoom, 15) & 0x0F
    priority_nibble = min(priority // 7, 15) & 0x0F
    return (zoom_nibble << 4) | priority_nibble


def coord_to_int32(lon: float, lat: float) -> Tuple[int, int]:
    """Convert float coordinates to int32 (scaled by 1e7)."""
    return (int(lon * COORD_SCALE), int(lat * COORD_SCALE))


def get_simplify_tolerance(zoom: int) -> float:
    """Calculate simplification tolerance based on zoom level."""
    tile_width_degrees = 360.0 / (2.0 ** zoom)
    pixel_size_degrees = tile_width_degrees / 256.0
    return pixel_size_degrees


class OSMHandler(osmium.SimpleHandler):
    """Handler for processing OSM data from PBF files."""

    def __init__(self, config: Dict, zoom_range: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.min_zoom, self.max_zoom = zoom_range
        self.features: List[Dict] = []
        self.stats = {
            'ways_processed': 0,
            'relations_processed': 0,
            'areas_processed': 0,
            'features_extracted': 0,
            'features_filtered': 0
        }
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.progress_interval = 5
        self.interesting_tags = self._build_interesting_tags()
        self.processed_way_ids: Set[int] = set()
        self.wkbfab = osmium.geom.WKBFactory()

    def _build_interesting_tags(self) -> Set[str]:
        """Build set of tag keys we're interested in."""
        tags = set()
        for key in self.config:
            if isinstance(self.config[key], dict):
                if '=' in key:
                    tag_key = key.split('=')[0]
                    tags.add(tag_key)
                else:
                    tags.add(key)
        return tags

    def _log_progress(self):
        """Log progress periodically."""
        current_time = time.time()
        if current_time - self.last_progress_time >= self.progress_interval:
            self.last_progress_time = current_time
            ways = self.stats['ways_processed']
            extracted = self.stats['features_extracted']
            elapsed = current_time - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            print(f"\r  Progress: {ways:,} ways, {extracted:,} features [{mins}m {secs:02d}s]", end='', flush=True)

    def _has_interesting_tags(self, tags) -> bool:
        """Check if tags contain any interesting keys."""
        for tag in tags:
            if tag.k in self.interesting_tags:
                return True
        return False

    def _tags_to_dict(self, tags) -> Dict[str, str]:
        """Convert osmium tags to dictionary."""
        return {tag.k: tag.v for tag in tags}

    def _is_feature_in_config(self, tags: Dict[str, str]) -> bool:
        """Check if feature matches any entry in config."""
        for key, value in tags.items():
            feature_key = f"{key}={value}"
            if feature_key in self.config and isinstance(self.config[feature_key], dict):
                return True
            if key in self.config and isinstance(self.config[key], dict):
                return True
        return False

    def way(self, w):
        """Process way - extract roads and linear features."""
        self.stats['ways_processed'] += 1
        self._log_progress()

        if not self._has_interesting_tags(w.tags):
            self.stats['features_filtered'] += 1
            return

        tags = self._tags_to_dict(w.tags)

        if not self._is_feature_in_config(tags):
            self.stats['features_filtered'] += 1
            return

        layer = get_layer_for_tags(tags)
        if layer is None:
            self.stats['features_filtered'] += 1
            return

        min_zoom = get_zoom_for_tags(tags, self.config)
        if min_zoom > self.max_zoom:
            self.stats['features_filtered'] += 1
            return

        coords = []
        for node in w.nodes:
            if node.location.valid():
                coords.append((node.location.lon, node.location.lat))

        if len(coords) < 2:
            self.stats['features_filtered'] += 1
            return

        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        is_area_tags = (
            'building' in tags or
            'landuse' in tags or
            ('natural' in tags and tags.get('natural') in ['water', 'wood', 'forest', 'beach', 'sand', 'wetland', 'grassland', 'scrub', 'heath']) or
            ('leisure' in tags and tags.get('leisure') in ['park', 'garden', 'pitch', 'golf_course', 'nature_reserve', 'playground', 'sports_centre', 'stadium', 'common']) or
            ('amenity' in tags and tags.get('amenity') in ['parking', 'school', 'university', 'hospital', 'marketplace']) or
            ('waterway' in tags and tags.get('waterway') in ['riverbank', 'dock', 'boatyard']) or
            tags.get('area') == 'yes'
        )

        color = get_color_for_tags(tags, self.config)
        priority = get_priority_for_tags(tags, self.config)
        color_rgb565 = hex_to_rgb565(color)
        layer_base_priority = LAYER_PRIORITY.get(layer, 50)
        combined_priority = layer_base_priority + (priority % 10)

        if is_closed and is_area_tags and 'highway' not in tags:
            feature = {
                'geom_type': GEOM_POLYGON,
                'coords': coords,
                'color_rgb565': color_rgb565,
                'zoom_priority': pack_zoom_priority(min_zoom, combined_priority)
            }
            self.features.append(feature)
            self.stats['features_extracted'] += 1
            self.processed_way_ids.add(w.id)
            return

        feature = {
            'geom_type': GEOM_LINESTRING,
            'coords': coords,
            'color_rgb565': color_rgb565,
            'zoom_priority': pack_zoom_priority(min_zoom, combined_priority)
        }
        self.features.append(feature)
        self.stats['features_extracted'] += 1

    def relation(self, r):
        """Process relation."""
        self.stats['relations_processed'] += 1

    def area(self, a):
        """Process area - handles multipolygon relations."""
        self.stats['areas_processed'] += 1
        self._log_progress()

        if not self._has_interesting_tags(a.tags):
            self.stats['features_filtered'] += 1
            return

        tags = self._tags_to_dict(a.tags)

        if not self._is_feature_in_config(tags):
            self.stats['features_filtered'] += 1
            return

        layer = get_layer_for_tags(tags)
        if layer is None:
            self.stats['features_filtered'] += 1
            return

        if 'highway' in tags:
            self.stats['features_filtered'] += 1
            return

        min_zoom = get_zoom_for_tags(tags, self.config)
        if min_zoom > self.max_zoom:
            self.stats['features_filtered'] += 1
            return

        if a.from_way() and a.orig_id() in self.processed_way_ids:
            return

        try:
            wkb = self.wkbfab.create_multipolygon(a)
            geom = shapely.wkb.loads(wkb, hex=True)

            color = get_color_for_tags(tags, self.config)
            priority = get_priority_for_tags(tags, self.config)
            color_rgb565 = hex_to_rgb565(color)
            layer_base_priority = LAYER_PRIORITY.get(layer, 50)
            combined_priority = layer_base_priority + (priority % 10)

            polygons = []
            if geom.geom_type == 'Polygon':
                polygons = [geom]
            elif geom.geom_type == 'MultiPolygon':
                polygons = list(geom.geoms)

            for poly in polygons:
                if poly.is_empty or not poly.exterior:
                    continue

                coords = list(poly.exterior.coords)
                if len(coords) < 4:
                    continue

                feature = {
                    'geom_type': GEOM_POLYGON,
                    'coords': coords,
                    'color_rgb565': color_rgb565,
                    'zoom_priority': pack_zoom_priority(min_zoom, combined_priority)
                }
                self.features.append(feature)
                self.stats['features_extracted'] += 1

        except Exception as e:
            self.stats['features_filtered'] += 1


def simplify_coords(coords: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """Simple Douglas-Peucker-like simplification."""
    if len(coords) <= 2:
        return coords

    # Use shapely for simplification if available
    if SHAPELY_AVAILABLE:
        from shapely.geometry import LineString
        line = LineString(coords)
        simplified = line.simplify(tolerance, preserve_topology=True)
        return list(simplified.coords)

    return coords


def write_nav_tile(features: List[Dict], output_path: str, zoom: int) -> bool:
    """Write features to NAV binary tile format."""
    if not features:
        return False

    tolerance = get_simplify_tolerance(zoom)

    # Calculate bounding box
    all_lons = []
    all_lats = []
    for f in features:
        for lon, lat in f['coords']:
            all_lons.append(lon)
            all_lats.append(lat)

    if not all_lons:
        return False

    min_lon = min(all_lons)
    min_lat = min(all_lats)
    max_lon = max(all_lons)
    max_lat = max(all_lats)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        # Write header (20 bytes)
        f.write(NAV_MAGIC)  # 4 bytes
        f.write(struct.pack('<B', NAV_VERSION))  # 1 byte
        f.write(struct.pack('<H', len(features)))  # 2 bytes
        f.write(struct.pack('<B', 0))  # 1 byte reserved
        f.write(struct.pack('<i', int(min_lon * COORD_SCALE)))  # 4 bytes
        f.write(struct.pack('<i', int(min_lat * COORD_SCALE)))  # 4 bytes
        f.write(struct.pack('<i', int(max_lon * COORD_SCALE)))  # 4 bytes
        f.write(struct.pack('<i', int(max_lat * COORD_SCALE)))  # 4 bytes

        # Write features
        for feature in features:
            coords = feature['coords']

            # Simplify coordinates
            if len(coords) > 2:
                coords = simplify_coords(coords, tolerance)

            # Skip if too few coordinates after simplification
            if feature['geom_type'] == GEOM_LINESTRING and len(coords) < 2:
                continue
            if feature['geom_type'] == GEOM_POLYGON and len(coords) < 4:
                continue

            # Write feature header
            f.write(struct.pack('<B', feature['geom_type']))  # 1 byte
            f.write(struct.pack('<H', feature['color_rgb565']))  # 2 bytes
            f.write(struct.pack('<B', feature['zoom_priority']))  # 1 byte
            f.write(struct.pack('<H', len(coords)))  # 2 bytes

            # Write coordinates as int32
            for lon, lat in coords:
                f.write(struct.pack('<i', int(lon * COORD_SCALE)))
                f.write(struct.pack('<i', int(lat * COORD_SCALE)))

            # For polygons, write ring info (single ring for now)
            if feature['geom_type'] == GEOM_POLYGON:
                f.write(struct.pack('<B', 1))  # 1 ring
                f.write(struct.pack('<H', len(coords)))  # ring end

    return True


def convert_pbf_to_nav(input_pbf: str, output_dir: str, config_file: str,
                        zoom_range: Tuple[int, int] = (6, 17)):
    """Main conversion function - generates NAV tile files."""

    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    file_size_mb = os.path.getsize(input_pbf) / (1024 * 1024)
    logger.info(f"Processing PBF file: {input_pbf} ({file_size_mb:.1f} MB)")
    logger.info(f"Zoom range: {zoom_range[0]}-{zoom_range[1]}")
    logger.info(f"Output format: NAV binary tiles (.nav)")

    start_time = time.time()

    handler = OSMHandler(config, zoom_range)
    logger.info("Processing OSM data...")
    handler.apply_file(input_pbf, locations=True, idx='flex_mem')
    print()

    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")
    logger.info(f"Statistics:")
    logger.info(f"  Ways processed: {handler.stats['ways_processed']:,}")
    logger.info(f"  Areas processed: {handler.stats['areas_processed']:,}")
    logger.info(f"  Features extracted: {handler.stats['features_extracted']:,}")
    logger.info(f"  Features filtered: {handler.stats['features_filtered']:,}")

    logger.info("Generating NAV tile files...")

    total_tiles = 0
    total_size = 0

    for zoom in range(zoom_range[0], zoom_range[1] + 1):
        tile_features: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

        for feature in handler.features:
            min_zoom = feature['zoom_priority'] >> 4
            if min_zoom > zoom:
                continue

            is_polygon = feature['geom_type'] == GEOM_POLYGON
            tiles = get_feature_tiles(feature['coords'], zoom, is_polygon)

            for tile in tiles:
                tile_features[tile].append(feature)

        if not tile_features:
            continue

        num_tiles = len(tile_features)
        tiles_written = 0
        tile_items = list(tile_features.items())

        for i, ((x, y), features) in enumerate(tile_items):
            progress = (i + 1) / num_tiles
            bar_width = 30
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\r  Zoom {zoom:2d}: [{bar}] {i+1}/{num_tiles} tiles", end='', flush=True)

            tile_dir = os.path.join(output_dir, str(zoom), str(x))
            tile_path = os.path.join(tile_dir, f"{y}.nav")

            if write_nav_tile(features, tile_path, zoom):
                tiles_written += 1
                total_size += os.path.getsize(tile_path)

        print(f"\r  Zoom {zoom:2d}: {tiles_written} tiles written" + " " * 30)
        total_tiles += tiles_written

    total_time = time.time() - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:02d}s"
    else:
        time_str = f"{total_time:.2f}s"

    logger.info("=" * 50)
    logger.info("Conversion Summary")
    logger.info("=" * 50)
    logger.info(f"Input: {input_pbf}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Format: NAV binary tiles (.nav)")
    logger.info(f"Total tiles: {total_tiles}")
    logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    logger.info(f"Total time: {time_str}")
    logger.info("=" * 50)

    return total_tiles


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenStreetMap PBF to NAV binary tile format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NAV Format - IceNav Navigation Tiles:
  - int32 coordinates (scaled by 1e7) for ~50% size reduction
  - Simple binary format optimized for ESP32
  - Same z/x/y tile structure as standard map tiles

Examples:
    python pbf_to_nav.py andorra.pbf ./output features.json
    python pbf_to_nav.py spain.pbf ./output features.json --zoom 10-17
        """
    )

    parser.add_argument('input_pbf', help='Input PBF file path')
    parser.add_argument('output_dir', help='Output directory for NAV tiles')
    parser.add_argument('config_file', help='Features configuration JSON file')
    parser.add_argument('--zoom', default='6-17',
                        help='Zoom level range (e.g., "6-17" or "12")')

    args = parser.parse_args()

    if not os.path.exists(args.input_pbf):
        logger.error(f"Input file not found: {args.input_pbf}")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        sys.exit(1)

    if '-' in args.zoom:
        min_zoom, max_zoom = map(int, args.zoom.split('-'))
    else:
        min_zoom = max_zoom = int(args.zoom)

    convert_pbf_to_nav(args.input_pbf, args.output_dir, args.config_file, (min_zoom, max_zoom))


if __name__ == '__main__':
    main()
