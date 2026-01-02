#!/usr/bin/env python3
"""
PBF to FlatGeobuf Converter

Converts OpenStreetMap .pbf files to FlatGeobuf (.fgb) format with spatial indexing.
Generates ONE unified file per zoom range with all layers combined.

Usage:
    python pbf_to_fgb.py input.pbf output_dir features.json [--zoom 6-17]

Output structure:
    output_dir/
    ├── z6-9.fgb     # All layers combined for zooms 6-9
    ├── z10-12.fgb   # All layers combined for zooms 10-12
    └── z13-17.fgb   # All layers combined for zooms 13-17
"""

import os
import sys
import json
import argparse
import logging
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
    import fiona
    from fiona.crs import from_epsg
    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Polygon, MultiPolygon, Point, mapping
    from shapely.ops import transform
    from shapely import simplify
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Zoom ranges for unified files
ZOOM_RANGES = [
    (6, 9),    # Low zoom: major features only
    (10, 12),  # Medium zoom: more detail
    (13, 17),  # High zoom: full detail
]

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


def write_unified_fgb(features: List[Dict], output_path: str, max_zoom: int):
    """Write all features to a single unified FlatGeobuf file."""
    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas not available. Install with: pip install geopandas")
        return False

    if not features:
        logger.warning(f"No features to write")
        return False

    # Filter features for this zoom range
    filtered_features = [f for f in features if f['properties']['min_zoom'] <= max_zoom]

    if not filtered_features:
        logger.warning(f"No features for max zoom {max_zoom}")
        return False

    logger.info(f"Writing {len(filtered_features)} features to {output_path}")

    geometries = []
    properties_list = []

    for feature in filtered_features:
        coords = feature['coordinates']
        geom_type = feature['geometry_type']
        props = feature['properties']

        try:
            if geom_type == 'LineString':
                if len(coords) >= 2:
                    geom = LineString(coords)
                    geometries.append(geom)
                    properties_list.append(props)
            elif geom_type == 'Polygon':
                if len(coords) >= 4:
                    geom = Polygon(coords)
                    if geom.is_valid:
                        geometries.append(geom)
                        properties_list.append(props)
                    else:
                        geom = geom.buffer(0)
                        if geom.is_valid and not geom.is_empty:
                            geometries.append(geom)
                            properties_list.append(props)
        except Exception as e:
            logger.debug(f"Error creating geometry: {e}")
            continue

    if not geometries:
        logger.warning(f"No valid geometries")
        return False

    gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")

    try:
        import warnings
        pyogrio_logger = logging.getLogger('pyogrio')
        fiona_logger = logging.getLogger('fiona')
        old_pyogrio_level = pyogrio_logger.level
        old_fiona_level = fiona_logger.level
        pyogrio_logger.setLevel(logging.ERROR)
        fiona_logger.setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gdf.to_file(output_path, driver="FlatGeobuf", spatial_index=True)

        pyogrio_logger.setLevel(old_pyogrio_level)
        fiona_logger.setLevel(old_fiona_level)

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Successfully wrote {len(gdf)} features ({file_size_mb:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Error writing FlatGeobuf: {e}")
        return False


def convert_pbf_to_fgb(input_pbf: str, output_dir: str, config_file: str,
                        zoom_range: Tuple[int, int] = (6, 17)):
    """Main conversion function - generates unified files per zoom range."""

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

    # Write unified FGB files per zoom range
    logger.info("Writing unified FlatGeobuf files...")

    files_written = []
    total_size = 0

    for min_z, max_z in ZOOM_RANGES:
        # Skip ranges outside requested zoom
        if max_z < zoom_range[0] or min_z > zoom_range[1]:
            continue

        # Adjust range to requested bounds
        actual_min = max(min_z, zoom_range[0])
        actual_max = min(max_z, zoom_range[1])

        output_filename = f"z{actual_min}-{actual_max}.fgb"
        output_path = os.path.join(output_dir, output_filename)

        if write_unified_fgb(handler.features, output_path, actual_max):
            files_written.append(output_path)
            total_size += os.path.getsize(output_path)

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

    logger.info("=" * 50)
    logger.info("Conversion Summary")
    logger.info("=" * 50)
    logger.info(f"Input: {input_pbf}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files written:")
    for filepath in files_written:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"  {os.path.basename(filepath)}: {size_mb:.2f} MB")
    logger.info(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    logger.info(f"Total time: {time_str}")
    logger.info("=" * 50)

    return files_written


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenStreetMap PBF to unified FlatGeobuf format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pbf_to_fgb.py andorra.pbf ./output features.json
    python pbf_to_fgb.py spain.pbf ./output features.json --zoom 10-17

Output structure (unified files):
    output_dir/
    ├── z6-9.fgb     # All layers for zooms 6-9
    ├── z10-12.fgb   # All layers for zooms 10-12
    └── z13-17.fgb   # All layers for zooms 13-17

Each file contains ALL layers combined with properties:
    - layer: Layer name for render ordering
    - priority: Render priority (lower = behind)
    - color_rgb332: 8-bit color
    - min_zoom: Minimum zoom for visibility
        """
    )

    parser.add_argument('input_pbf', help='Input PBF file path')
    parser.add_argument('output_dir', help='Output directory for FGB files')
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
