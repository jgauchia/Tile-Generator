#!/usr/bin/env python3
"""
PBF to FlatGeobuf Converter - Tile Structure

Converts OpenStreetMap .pbf files to FlatGeobuf (.fgb) format with tile structure.
Generates individual FGB files per tile (z/x/y structure like PNG tiles).

Usage:
    python pbf_to_fgb.py input.pbf output_dir features.json [--zoom 6-17]

Output structure:
    output_dir/
    ├── 13/
    │   ├── 4123/
    │   │   ├── 2456.fgb
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
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import time

try:
    import osmium
    from osmium import osm
except ImportError:
    print("Error: osmium not found. Install with: pip install osmium")
    sys.exit(1)

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, MultiPolygon, Point, box
    from shapely.ops import transform
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        'leisure=golf_course', 'landuse=residential', 'place=suburb',
        'landuse=commercial', 'landuse=retail', 'landuse=industrial',
        'landuse=construction', 'landuse=cemetery', 'landuse=allotments',
        'leisure=stadium', 'leisure=sports_centre', 'leisure=playground'
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
        'amenity=parking', 'amenity=hospital',
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


def get_feature_tiles(coords: List[Tuple[float, float]], zoom: int) -> Set[Tuple[int, int]]:
    """Get all tiles that a feature intersects at given zoom level."""
    tiles = set()
    for lon, lat in coords:
        x = lon_to_tile_x(lon, zoom)
        y = lat_to_tile_y(lat, zoom)
        tiles.add((x, y))
    return tiles


def get_layer_for_tags(tags: Dict[str, str]) -> Optional[str]:
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


def hex_to_rgb332(hex_color: str) -> int:
    """Convert hex color to RGB332 format."""
    try:
        if not hex_color or not hex_color.startswith("#"):
            return 0xFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r & 0xE0) | ((g & 0xE0) >> 3) | (b >> 6))
    except:
        return 0xFF


def get_simplify_tolerance(zoom: int) -> float:
    """
    Calculate simplification tolerance based on zoom level.

    At each zoom, a tile is 256 pixels wide and covers:
        tile_width_degrees = 360 / (2^zoom)
        pixel_size_degrees = tile_width_degrees / 256

    We use 1 pixel as tolerance - points closer than this are redundant.
    """
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
            'features_extracted': 0,
            'features_filtered': 0
        }
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.progress_interval = 5
        self.interesting_tags = self._build_interesting_tags()

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
        """Process way - extract roads, buildings, etc."""
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
        is_area = is_closed and (
            'building' in tags or
            'landuse' in tags or
            'natural' in tags and tags.get('natural') in ['water', 'wood', 'forest', 'beach', 'sand', 'wetland', 'grassland', 'scrub', 'heath'] or
            'leisure' in tags and tags.get('leisure') in ['park', 'garden', 'pitch', 'golf_course', 'nature_reserve', 'playground', 'sports_centre', 'stadium'] or
            'amenity' in tags and tags.get('amenity') in ['parking'] or
            'waterway' in tags and tags.get('waterway') in ['riverbank', 'dock', 'boatyard'] or
            tags.get('area') == 'yes'
        )

        color = get_color_for_tags(tags, self.config)
        priority = get_priority_for_tags(tags, self.config)
        color_rgb332 = hex_to_rgb332(color)

        # Combine layer priority with feature priority
        layer_base_priority = LAYER_PRIORITY.get(layer, 50)
        combined_priority = layer_base_priority + (priority % 10)

        feature = {
            'geometry_type': 'Polygon' if is_area else 'LineString',
            'coordinates': coords,
            'properties': {
                'osm_id': w.id,
                'min_zoom': min_zoom,
                'color_rgb332': color_rgb332,
                'priority': combined_priority,
                'layer': layer,
                'feature_type': self._get_primary_tag(tags)
            }
        }

        self.features.append(feature)
        self.stats['features_extracted'] += 1

    def _get_primary_tag(self, tags: Dict[str, str]) -> str:
        """Get the primary identifying tag for a feature."""
        priority_keys = ['highway', 'railway', 'waterway', 'building', 'natural', 'landuse', 'leisure', 'amenity', 'aeroway']
        for key in priority_keys:
            if key in tags:
                return f"{key}={tags[key]}"
        return 'unknown'

    def relation(self, r):
        """Process relation - for multipolygons, etc."""
        self.stats['relations_processed'] += 1
        pass


def count_coords(geom) -> int:
    """Count total coordinates in a geometry."""
    if hasattr(geom, 'exterior'):  # Polygon
        return len(geom.exterior.coords) + sum(len(ring.coords) for ring in geom.interiors)
    elif hasattr(geom, 'coords'):  # LineString/Point
        return len(geom.coords)
    return 0


# Global simplification statistics
simplify_stats = {'nodes_before': 0, 'nodes_after': 0}


def write_tile_fgb(features: List[Dict], output_path: str, zoom: int):
    """Write features to a single tile FlatGeobuf file.

    Features are NOT clipped to tile boundaries to avoid visible seams.
    Each feature is stored complete in every tile it touches.
    The renderer clips to viewport naturally.

    Geometries are simplified based on zoom level - points representing
    less than 1 pixel are removed to reduce file size.
    """
    global simplify_stats

    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas not available")
        return False

    if not features:
        return False

    # Calculate simplification tolerance (1 pixel in degrees)
    tolerance = get_simplify_tolerance(zoom)

    geometries = []
    properties_list = []

    for feature in features:
        coords = feature['coordinates']
        geom_type = feature['geometry_type']
        props = feature['properties']

        try:
            if geom_type == 'LineString':
                if len(coords) >= 2:
                    geom = LineString(coords)
                    nodes_before = len(geom.coords)
                    # Simplify: remove points closer than 1 pixel
                    geom = geom.simplify(tolerance, preserve_topology=True)
                    # After simplification, ensure still valid LineString (min 2 points)
                    if not geom.is_empty and len(geom.coords) >= 2:
                        simplify_stats['nodes_before'] += nodes_before
                        simplify_stats['nodes_after'] += len(geom.coords)
                        geometries.append(geom)
                        properties_list.append(props)
            elif geom_type == 'Polygon':
                if len(coords) >= 4:
                    geom = Polygon(coords)
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    nodes_before = count_coords(geom)
                    # Simplify: remove points closer than 1 pixel
                    geom = geom.simplify(tolerance, preserve_topology=True)
                    if geom.is_valid and not geom.is_empty:
                        simplify_stats['nodes_before'] += nodes_before
                        simplify_stats['nodes_after'] += count_coords(geom)
                        geometries.append(geom)
                        properties_list.append(props)
        except Exception as e:
            continue

    if not geometries:
        return False

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")

    try:
        import warnings
        # Silence pyogrio "Created X records" messages
        pyogrio_logger = logging.getLogger('pyogrio')
        old_level = pyogrio_logger.level
        pyogrio_logger.setLevel(logging.WARNING)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf.to_file(output_path, driver="FlatGeobuf", spatial_index=True)
        pyogrio_logger.setLevel(old_level)
        return True
    except Exception as e:
        logger.error(f"Error writing {output_path}: {e}")
        return False


def convert_pbf_to_fgb(input_pbf: str, output_dir: str, config_file: str,
                        zoom_range: Tuple[int, int] = (6, 17)):
    """Main conversion function - generates tile-based FGB files."""
    global simplify_stats
    simplify_stats = {'nodes_before': 0, 'nodes_after': 0}

    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    file_size_mb = os.path.getsize(input_pbf) / (1024 * 1024)
    logger.info(f"Processing PBF file: {input_pbf} ({file_size_mb:.1f} MB)")
    logger.info(f"Zoom range: {zoom_range[0]}-{zoom_range[1]}")

    start_time = time.time()

    handler = OSMHandler(config, zoom_range)
    logger.info("Processing OSM data (this may take a while for large files)...")
    handler.apply_file(input_pbf, locations=True, idx='flex_mem')
    print()

    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")
    logger.info(f"Statistics:")
    logger.info(f"  Ways processed: {handler.stats['ways_processed']:,}")
    logger.info(f"  Features extracted: {handler.stats['features_extracted']:,}")
    logger.info(f"  Features filtered: {handler.stats['features_filtered']:,}")

    # Group features by tile for each zoom level
    logger.info("Generating tile-based FlatGeobuf files...")

    total_tiles = 0
    total_size = 0

    for zoom in range(zoom_range[0], zoom_range[1] + 1):
        # Group features by tile at this zoom level
        tile_features: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)

        for feature in handler.features:
            # Only include features visible at this zoom
            if feature['properties']['min_zoom'] > zoom:
                continue

            # Get all tiles this feature touches
            tiles = get_feature_tiles(feature['coordinates'], zoom)
            for tile in tiles:
                tile_features[tile].append(feature)

        if not tile_features:
            continue

        num_tiles = len(tile_features)
        tiles_written = 0
        tile_items = list(tile_features.items())

        for i, ((x, y), features) in enumerate(tile_items):
            # Progress bar
            progress = (i + 1) / num_tiles
            bar_width = 30
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\r  Zoom {zoom:2d}: [{bar}] {i+1}/{num_tiles} tiles", end='', flush=True)

            # Create directory structure: output_dir/zoom/x/y.fgb
            tile_dir = os.path.join(output_dir, str(zoom), str(x))
            tile_path = os.path.join(tile_dir, f"{y}.fgb")

            if write_tile_fgb(features, tile_path, zoom):
                tiles_written += 1
                total_size += os.path.getsize(tile_path)

        # Clear line and show final count
        print(f"\r  Zoom {zoom:2d}: {tiles_written} tiles written" + " " * 30)
        total_tiles += tiles_written

    # Summary
    total_time = time.time() - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:02d}s"
    else:
        time_str = f"{total_time:.2f}s"

    # Simplification statistics
    nodes_before = simplify_stats['nodes_before']
    nodes_after = simplify_stats['nodes_after']
    if nodes_before > 0:
        reduction = (1 - nodes_after / nodes_before) * 100
    else:
        reduction = 0

    logger.info("=" * 50)
    logger.info("Conversion Summary")
    logger.info("=" * 50)
    logger.info(f"Input: {input_pbf}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total tiles: {total_tiles}")
    logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    logger.info(f"Nodes before simplification: {nodes_before:,}")
    logger.info(f"Nodes after simplification: {nodes_after:,}")
    logger.info(f"Node reduction: {reduction:.1f}%")
    logger.info(f"Total time: {time_str}")
    logger.info("=" * 50)

    return total_tiles


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenStreetMap PBF to tile-based FlatGeobuf format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pbf_to_fgb.py andorra.pbf ./output features.json
    python pbf_to_fgb.py spain.pbf ./output features.json --zoom 10-17

Output structure (tile-based):
    output_dir/
    ├── 13/
    │   ├── 4123/
    │   │   ├── 2456.fgb
    │   │   └── ...
    │   └── ...
    └── 14/
        └── ...

Each tile FGB file contains features clipped to that tile with properties:
    - layer: Layer name for render ordering
    - priority: Render priority (lower = behind)
    - color_rgb332: 8-bit color
    - min_zoom: Minimum zoom for visibility
        """
    )

    parser.add_argument('input_pbf', help='Input PBF file path')
    parser.add_argument('output_dir', help='Output directory for FGB tiles')
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

    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas is required. Install with: pip install geopandas pyogrio")
        sys.exit(1)

    if '-' in args.zoom:
        min_zoom, max_zoom = map(int, args.zoom.split('-'))
    else:
        min_zoom = max_zoom = int(args.zoom)

    convert_pbf_to_fgb(args.input_pbf, args.output_dir, args.config_file, (min_zoom, max_zoom))


if __name__ == '__main__':
    main()
