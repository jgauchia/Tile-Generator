#!/usr/bin/env python3
"""
PBF to FlatGeobuf Converter

Converts OpenStreetMap .pbf files to FlatGeobuf (.fgb) format with spatial indexing.
Filters features based on features.json configuration and organizes output by layer.

Usage:
    python pbf_to_fgb.py input.pbf output_dir features.json [--zoom 6-17]
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

# Layer definitions based on feature types
LAYER_MAPPING = {
    # Water features
    'water': [
        'natural=water', 'natural=coastline', 'natural=bay',
        'waterway=riverbank', 'waterway=dock', 'waterway=boatyard',
        'waterway=river', 'waterway=stream', 'waterway=canal',
        'natural=spring', 'natural=wetland'
    ],
    # Land use and natural areas
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
    # Roads and paths
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
    # Railways
    'railways': [
        'railway=rail', 'railway=subway', 'railway=tram'
    ],
    # Buildings
    'buildings': [
        'building', 'man_made=tower'
    ],
    # Amenities
    'amenities': [
        'amenity=parking', 'amenity=hospital',
        'amenity=school', 'amenity=university',
        'amenity=place_of_worship'
    ],
    # Bridges and aeroways
    'infrastructure': [
        'bridge=yes', 'man_made=bridge',
        'aeroway=runway', 'aeroway=taxiway', 'aeroway=apron',
        'tunnel=yes'
    ],
    # Natural terrain features
    'terrain': [
        'natural=peak', 'natural=ridge',
        'natural=volcano', 'natural=cliff',
        'natural=tree_row', 'natural=tree'
    ],
    # Places
    'places': [
        'place=state', 'place=town',
        'place=village', 'place=hamlet'
    ]
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
                # Key-only match (e.g., 'building')
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


def get_simplification_tolerance(zoom: int) -> float:
    """Get Douglas-Peucker simplification tolerance based on zoom level."""
    # Higher zoom = less simplification (more detail)
    tolerances = {
        6: 0.001,
        7: 0.0008,
        8: 0.0005,
        9: 0.0003,
        10: 0.0002,
        11: 0.00015,
        12: 0.0001,
        13: 0.00008,
        14: 0.00005,
        15: 0.00003,
        16: 0.000015,
        17: 0.000012,
        18: 0.00001,
        19: 0.000008,
    }
    return tolerances.get(zoom, 0.000005)


class OSMHandler(osmium.SimpleHandler):
    """Handler for processing OSM data from PBF files."""

    def __init__(self, config: Dict, zoom_range: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.min_zoom, self.max_zoom = zoom_range

        # Storage for features by layer
        self.features_by_layer: Dict[str, List[Dict]] = defaultdict(list)

        # Statistics
        self.stats = {
            'ways_processed': 0,
            'relations_processed': 0,
            'features_extracted': 0,
            'features_filtered': 0
        }

        # Progress tracking
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.progress_interval = 5  # Log every 5 seconds

        # Build set of interesting tags from config
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
            # Print in-place progress
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

        # Get layer and zoom
        layer = get_layer_for_tags(tags)
        if layer is None:
            self.stats['features_filtered'] += 1
            return

        min_zoom = get_zoom_for_tags(tags, self.config)
        if min_zoom > self.max_zoom:
            self.stats['features_filtered'] += 1
            return

        # Build geometry from node locations (provided by osmium with locations=True)
        coords = []
        for node in w.nodes:
            if node.location.valid():
                coords.append((node.location.lon, node.location.lat))

        if len(coords) < 2:
            self.stats['features_filtered'] += 1
            return

        # Determine geometry type
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

        # Get attributes
        color = get_color_for_tags(tags, self.config)
        priority = get_priority_for_tags(tags, self.config)
        color_rgb332 = hex_to_rgb332(color)

        # Create feature
        feature = {
            'geometry_type': 'Polygon' if is_area else 'LineString',
            'coordinates': coords,
            'properties': {
                'osm_id': w.id,
                'min_zoom': min_zoom,
                'color_rgb332': color_rgb332,
                'priority': priority,
                'layer': layer,
                # Store primary tag for reference
                'feature_type': self._get_primary_tag(tags)
            }
        }

        self.features_by_layer[layer].append(feature)
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
        # TODO: Handle multipolygon relations for complex areas
        pass


def write_fgb_layer(features: List[Dict], output_path: str, layer_name: str):
    """Write features to FlatGeobuf file using GeoPandas."""
    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas not available. Install with: pip install geopandas")
        return False

    if not features:
        logger.warning(f"No features to write for layer {layer_name}")
        return False

    logger.debug(f"Writing {len(features)} features to {output_path}")

    # Convert features to GeoDataFrame
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
                    geometries.append(geom)
                    properties_list.append(props)
            elif geom_type == 'Polygon':
                if len(coords) >= 4:
                    geom = Polygon(coords)
                    if geom.is_valid:
                        geometries.append(geom)
                        properties_list.append(props)
                    else:
                        # Try to fix invalid polygon
                        geom = geom.buffer(0)
                        if geom.is_valid and not geom.is_empty:
                            geometries.append(geom)
                            properties_list.append(props)
        except Exception as e:
            logger.debug(f"Error creating geometry: {e}")
            continue

    if not geometries:
        logger.warning(f"No valid geometries for layer {layer_name}")
        return False

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")

    # Write to FlatGeobuf with spatial index
    try:
        # Suppress pyogrio/fiona logging during write
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

        # Restore logging levels
        pyogrio_logger.setLevel(old_pyogrio_level)
        fiona_logger.setLevel(old_fiona_level)

        logger.debug(f"Successfully wrote {len(gdf)} features to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error writing FlatGeobuf: {e}")
        return False


def convert_pbf_to_fgb(input_pbf: str, output_dir: str, config_file: str,
                        zoom_range: Tuple[int, int] = (6, 17)):
    """Main conversion function."""

    # Load configuration
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process PBF file
    file_size_mb = os.path.getsize(input_pbf) / (1024 * 1024)
    logger.info(f"Processing PBF file: {input_pbf} ({file_size_mb:.1f} MB)")
    logger.info(f"Zoom range: {zoom_range[0]}-{zoom_range[1]}")

    start_time = time.time()

    handler = OSMHandler(config, zoom_range)

    # First pass: cache nodes and extract features
    logger.info("Processing OSM data (this may take a while for large files)...")
    handler.apply_file(input_pbf, locations=True, idx='flex_mem')
    print()  # New line after progress

    # Log statistics
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")
    logger.info(f"Statistics:")
    logger.info(f"  Ways processed: {handler.stats['ways_processed']:,}")
    logger.info(f"  Features extracted: {handler.stats['features_extracted']:,}")
    logger.info(f"  Features filtered: {handler.stats['features_filtered']:,}")

    # Write FlatGeobuf files by zoom level and layer
    logger.info("Writing FlatGeobuf files by zoom level...")

    files_written = []
    total_files = 0
    zoom_sizes = {}  # zoom -> size in bytes

    # Calculate total expected files for progress bar
    num_zooms = zoom_range[1] - zoom_range[0] + 1
    num_layers = len(handler.features_by_layer)
    total_expected = num_zooms * num_layers
    current_file = 0
    bar_width = 40

    for zoom in range(zoom_range[0], zoom_range[1] + 1):
        # Create zoom directory
        zoom_dir = os.path.join(output_dir, str(zoom))
        os.makedirs(zoom_dir, exist_ok=True)

        zoom_files = 0
        zoom_features = 0
        zoom_size = 0

        for layer_name, features in handler.features_by_layer.items():
            current_file += 1

            # Update progress bar
            progress = current_file / total_expected
            filled = int(bar_width * progress)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"\r  [{bar}] {current_file}/{total_expected} (zoom {zoom}: {layer_name})", end='', flush=True)

            # Filter features for this zoom level (min_zoom <= current zoom)
            zoom_features_list = [f for f in features if f['properties']['min_zoom'] <= zoom]

            if zoom_features_list:
                output_path = os.path.join(zoom_dir, f"{layer_name}.fgb")
                if write_fgb_layer(zoom_features_list, output_path, layer_name):
                    files_written.append(output_path)
                    file_size = os.path.getsize(output_path)
                    zoom_files += 1
                    zoom_features += len(zoom_features_list)
                    zoom_size += file_size
                    total_files += 1

        zoom_sizes[zoom] = zoom_size

    print()  # New line after progress bar

    # Calculate total time in hh:mm:ss
    total_time = time.time() - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:02d}s"
    else:
        time_str = f"{total_time:.2f}s"

    # Calculate total size
    total_size = sum(zoom_sizes.values())

    # Summary
    logger.info("=" * 50)
    logger.info("Conversion Summary")
    logger.info("=" * 50)
    logger.info(f"Input: {input_pbf}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Zoom levels: {zoom_range[0]}-{zoom_range[1]}")
    logger.info(f"Total files written: {total_files}")
    logger.info(f"Total time: {time_str}")
    logger.info("")
    logger.info("Size by zoom level:")
    for zoom in sorted(zoom_sizes.keys()):
        size_mb = zoom_sizes[zoom] / (1024 * 1024)
        logger.info(f"  Zoom {zoom:2d}: {size_mb:8.2f} MB")
    logger.info(f"  {'─' * 20}")
    total_mb = total_size / (1024 * 1024)
    logger.info(f"  Total:   {total_mb:8.2f} MB")
    logger.info("=" * 50)

    return files_written


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenStreetMap PBF to FlatGeobuf format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pbf_to_fgb.py andorra.pbf ./output features.json
    python pbf_to_fgb.py spain.pbf ./output features.json --zoom 10-17

Output structure:
    output_dir/
    ├── 6/
    │   ├── water.fgb
    │   ├── roads.fgb (motorway, trunk, primary only)
    │   └── ...
    ├── 10/
    │   ├── water.fgb
    │   ├── roads.fgb (+ secondary, tertiary)
    │   └── railways.fgb
    ├── 14/
    │   ├── roads.fgb (+ residential, service)
    │   ├── buildings.fgb
    │   └── ...
    └── 17/
        └── ... (all features)
        """
    )

    parser.add_argument('input_pbf', help='Input PBF file path')
    parser.add_argument('output_dir', help='Output directory for FGB files')
    parser.add_argument('config_file', help='Features configuration JSON file')
    parser.add_argument('--zoom', default='6-17',
                        help='Zoom level range (e.g., "6-17" or "12")')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_pbf):
        logger.error(f"Input file not found: {args.input_pbf}")
        sys.exit(1)

    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        sys.exit(1)

    # Check dependencies
    if not GEOPANDAS_AVAILABLE:
        logger.error("GeoPandas is required. Install with: pip install geopandas pyogrio")
        sys.exit(1)

    # Parse zoom range
    if '-' in args.zoom:
        min_zoom, max_zoom = map(int, args.zoom.split('-'))
    else:
        min_zoom = max_zoom = int(args.zoom)

    # Run conversion
    convert_pbf_to_fgb(args.input_pbf, args.output_dir, args.config_file, (min_zoom, max_zoom))


if __name__ == '__main__':
    main()
