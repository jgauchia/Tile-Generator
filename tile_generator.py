#!/usr/bin/env python3
"""
PBF to NAV Tile Converter

Converts OpenStreetMap .pbf files to NAV binary format (.nav) with tile structure.
NAV format optimized for ESP32:
- 22-byte tile header (Magic, Count, Bbox)
- 11-byte feature header (Type, Color, Zoom/Priority, Width, BBox, Count)
- int16 relative coordinates (0-4096 range with safety margin)
- ~50% size reduction vs previous version

Width field encoding (1 byte):
- Bit 7 (0x80): Casing flag for two-pass rendering (motorway/trunk/primary)
- Bits 0-6 (0x7F): Actual width in pixels (0-127)

Usage:
    python tile_generator.py input.pbf output_dir features.json [--zoom 6-17]
"""

import os
import sys
import json
import argparse
import logging
import math
import time
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import osmium
    import osmium.area
    import osmium.index
except ImportError:
    print("Error: osmium not found. Install with: pip install osmium")
    sys.exit(1)

from constants import GEOM_POINT, GEOM_POLYGON, GEOM_TEXT, LINE_COLOR_PER_ZOOM
from geo_utils import (
    get_simplify_tolerance, get_feature_tiles,
    lon_to_tile_x, lat_to_tile_y, hex_to_rgb565,
)
from osm_handlers import BoundaryScanner, OSMHandler
from tile_writer import write_nav_tile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

    # First pass: scan for boundary relations (fast, no locations needed)
    logger.info("Pass 1: Scanning boundary relations...")
    scanner = BoundaryScanner(config, zoom_range[1])
    scanner.apply_file(input_pbf)
    logger.info(f"  Boundary ways found: {len(scanner.boundary_ways):,}")

    # Second pass: extract all features (including multipolygon relations)
    handler = OSMHandler(config, zoom_range)
    handler.boundary_ways = scanner.boundary_ways

    area_manager = osmium.area.AreaManager()

    # AreaManager requires TWO passes:
    logger.info("Pass 2a: Scanning multipolygon relations...")
    osmium.apply(input_pbf, area_manager.first_pass_handler())

    logger.info("Pass 2b: Building areas and extracting features...")
    idx = osmium.index.create_map('flex_mem')
    nlw = osmium.NodeLocationsForWays(idx)
    nlw.apply_nodes_to_ways = True
    # Chain handlers: nlw -> handler (nodes/ways) -> area_manager.second_pass (areas)
    osmium.apply(input_pbf, nlw, handler, area_manager.second_pass_handler(handler))
    print()

    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")
    logger.info(f"Statistics:")
    logger.info(f"  Nodes (peaks): {handler.stats['nodes_processed'] - handler.stats['text_labels']:,}")
    logger.info(f"  Text labels (places): {handler.stats['text_labels']:,}")
    logger.info(f"  Ways processed: {handler.stats['ways_processed']:,}")
    logger.info(f"  Areas processed: {handler.stats['areas_processed']:,}")
    logger.info(f"  Boundary ways extracted: {handler.stats['boundary_ways_extracted']:,}")
    logger.info(f"  Features extracted: {handler.stats['features_extracted']:,}")
    logger.info(f"  Features filtered: {handler.stats['features_filtered']:,}")
    logger.info(f"  Area filter breakdown:")
    logger.info(f"    - Boundary admin: {handler.stats['area_boundary']:,}")
    logger.info(f"    - Not in config: {handler.stats['area_no_config']:,}")
    logger.info(f"    - No layer mapping: {handler.stats['area_no_layer']:,}")
    logger.info(f"    - Zoom filtered: {handler.stats['area_zoom_filtered']:,}")
    logger.info(f"    - Exceptions: {handler.stats['area_exception']:,}")

    logger.info("Calculating bounding box from ALL features...")
    min_lon, max_lon = 180.0, -180.0
    min_lat, max_lat = 90.0, -90.0
    feature_count = 0
    for feature in handler.features:
        feature_count += 1
        for lon, lat in feature['coords']:
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
    logger.info(f"  BBox from {feature_count:,} features")
    logger.info(f"  BBox: lon=[{min_lon:.4f}, {max_lon:.4f}], lat=[{min_lat:.4f}, {max_lat:.4f}]")

    logger.info("Generating NAV tile files...")

    total_tiles = 0
    total_size = 0

    for zoom in range(zoom_range[0], zoom_range[1] + 1):
        tile_features = defaultdict(list)
        tolerance = get_simplify_tolerance(zoom)

        # Phase 1: Prepare and filter features for this zoom level
        zoom_start = time.time()
        prepared_count = 0
        total_to_process = len(handler.features)

        # Collect text labels for collision detection
        text_candidates = []

        for feature in handler.features:
            min_zoom = feature['zoom_priority'] >> 4
            if min_zoom > zoom:
                continue

            # Filter secondary roads without ref at z9 (keep only numbered departmental roads)
            if zoom == 9 and feature.get('highway_type') == 'secondary' and not feature.get('has_ref'):
                continue

            # NOTE: Simplification moved AFTER clipping to avoid inter-tile gaps
            coords = feature['coords']

            if not coords:
                continue

            # Convert point features to symbol polygons
            if feature['geom_type'] == GEOM_POINT:
                lon, lat = coords[0]
                tile_width_deg = 360.0 / (2.0 ** zoom)
                pixel_deg = tile_width_deg / 256.0
                size = pixel_deg * 3  # 3 pixel radius

                shape = feature.get('shape', 'circle')
                # Correct for Mercator distortion
                lat_size = size / math.cos(math.radians(lat))

                if shape == 'triangle':
                    # Equilateral triangle: height = size * sqrt(3)/2 ≈ size * 0.866
                    h = lat_size * 0.866
                    sym_coords = [
                        (lon, lat + h * 0.667),             # top (1/3 above center)
                        (lon - size, lat - h * 0.333),      # bottom-left
                        (lon + size, lat - h * 0.333),      # bottom-right
                        (lon, lat + h * 0.667),
                    ]
                else:  # square dot (2x2 pixels)
                    s = pixel_deg  # 1 pixel
                    ls = s / math.cos(math.radians(lat))
                    sym_coords = [
                        (lon - s, lat + ls),
                        (lon + s, lat + ls),
                        (lon + s, lat - ls),
                        (lon - s, lat - ls),
                        (lon - s, lat + ls),
                    ]

                zoom_feature = {
                    'geom_type': GEOM_POLYGON,
                    'coords': sym_coords,
                    'color_rgb565': feature['color_rgb565'],
                    'zoom_priority': feature['zoom_priority'],
                    'width_meters': 0.0
                }
            elif feature['geom_type'] == GEOM_TEXT:
                zoom_feature = {
                    'geom_type': GEOM_TEXT,
                    'coords': coords,
                    'color_rgb565': feature['color_rgb565'],
                    'zoom_priority': feature['zoom_priority'],
                    'font_size': feature.get('font_size', 0),
                    'text': feature['text'],
                }
            else:
                # Create a lightweight record for this zoom
                color_rgb565 = feature['color_rgb565']
                # Override color per zoom if defined
                hw_type = feature.get('highway_type', '')
                if hw_type and hw_type in LINE_COLOR_PER_ZOOM:
                    color_override = LINE_COLOR_PER_ZOOM[hw_type].get(zoom)
                    if color_override:
                        color_rgb565 = hex_to_rgb565(color_override)

                zoom_feature = {
                    'geom_type': feature['geom_type'],
                    'coords': coords,
                    'color_rgb565': color_rgb565,
                    'zoom_priority': feature['zoom_priority'],
                    'width_meters': feature.get('width_meters', 0.0),
                    'width_pixels': feature.get('width_pixels', 0),
                    'highway_type': hw_type,
                    'inner_rings': feature.get('inner_rings', []),
                    'layer': feature.get('layer', ''),  # Preserve layer for water detection
                    'is_bridge': feature.get('is_bridge', False),  # Preserve for casing
                    'is_building': feature.get('is_building', False),  # Preserve for outline
                    'name': feature.get('name', ''),  # Preserve name for debugging
                    'id': feature.get('id', 0),  # Preserve OSM ID for debugging
                }

            # Text labels: collect for collision detection
            if zoom_feature['geom_type'] == GEOM_TEXT:
                text_candidates.append(zoom_feature)
            else:
                is_polygon = zoom_feature['geom_type'] == GEOM_POLYGON
                tiles = get_feature_tiles(zoom_feature['coords'], zoom, is_polygon)

                for tile in tiles:
                    tile_features[tile].append(zoom_feature)

            prepared_count += 1
            if prepared_count % 25000 == 0:
                print(f"\r  Zoom {zoom:2d}: Preparing features... {prepared_count:,} / {total_to_process:,}", end='', flush=True)

        # Phase 1b: Text label collision detection
        # PRIORITY: Place names (fixed) > Road labels (moveable)
        if text_candidates:
            tile_width_deg = 360.0 / (2.0 ** zoom)
            pixel_deg = tile_width_deg / 256.0
            char_w = pixel_deg * 7  # half-width per char in degrees (1.75x for actual render size)
            label_h = pixel_deg * 11  # half-height in degrees (1.375x for actual render size)

            # Separate place names from road labels
            place_names = [f for f in text_candidates if 'coords_candidates' not in f]
            road_labels = [f for f in text_candidates if 'coords_candidates' in f]

            # STEP 1: Place names - sorted by population, place if no visual overlap
            # Population hierarchy: highest population wins in case of label collision
            place_names.sort(key=lambda f: -f.get('population', 0))

            placed_boxes = []
            places_placed = 0
            places_dropped_overlap = 0

            for pf in place_names:
                text_len = len(pf['text'])
                half_w = char_w * text_len / 2
                half_h = label_h
                lon, lat = pf['coords'][0]
                box = (lon - half_w, lat - half_h, lon + half_w, lat + half_h)

                # Check for visual overlap with already placed labels
                overlap = False
                for pb in placed_boxes:
                    if (box[0] < pb[2] and box[2] > pb[0] and
                        box[1] < pb[3] and box[3] > pb[1]):
                        overlap = True
                        break

                if not overlap:
                    # No visual collision - place this label
                    placed_boxes.append(box)
                    places_placed += 1

                    # Distribute to tiles
                    tiles = get_feature_tiles(pf['coords'], zoom, False)
                    expanded = set()
                    for (tx, ty) in tiles:
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                expanded.add((tx + dx, ty + dy))
                    for tile in expanded:
                        tile_features[tile].append(pf)
                else:
                    # Label overlaps with higher-population city - drop
                    places_dropped_overlap += 1

            # STEP 2: Road labels - try candidate positions, avoid place names
            roads_placed = 0
            roads_dropped = 0

            for rf in road_labels:
                text_len = len(rf['text'])
                half_w = char_w * text_len / 2
                half_h = label_h

                # Try 3 candidate positions along the road
                candidates_to_try = rf.get('coords_candidates', [rf['coords'][0]])
                placed = False

                for candidate_pos in candidates_to_try:
                    lon, lat = candidate_pos
                    box = (lon - half_w, lat - half_h, lon + half_w, lat + half_h)

                    # Check overlap with ALL placed labels (places + roads)
                    overlap = False
                    for pb in placed_boxes:
                        if (box[0] < pb[2] and box[2] > pb[0] and
                            box[1] < pb[3] and box[3] > pb[1]):
                            overlap = True
                            break

                    if not overlap:
                        # Found position without collision
                        rf['coords'] = [(lon, lat)]
                        placed_boxes.append(box)
                        roads_placed += 1
                        placed = True

                        # Distribute to tiles
                        tiles = get_feature_tiles(rf['coords'], zoom, False)
                        expanded = set()
                        for (tx, ty) in tiles:
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    expanded.add((tx + dx, ty + dy))
                        for tile in expanded:
                            tile_features[tile].append(rf)
                        break

                if not placed:
                    roads_dropped += 1

            if places_dropped_overlap > 0 or roads_dropped > 0:
                print(f"\r  Zoom {zoom:2d}: Labels: {places_placed} places, {roads_placed} roads, {places_dropped_overlap} places dropped (overlap), {roads_dropped} roads dropped")

        # Phase 2: Calculate tile grid from global bbox and write all tiles
        min_tx = lon_to_tile_x(min_lon, zoom)
        max_tx = lon_to_tile_x(max_lon, zoom)
        min_ty = lat_to_tile_y(max_lat, zoom)  # lat is inverted for Y
        max_ty = lat_to_tile_y(min_lat, zoom)

        num_tiles = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
        if num_tiles <= 0:
            print(f"\r  Zoom {zoom:2d}: No tiles to generate for this area.")
            continue

        tiles_written = 0
        print() # New line after preparation phase

        num_workers = min(os.cpu_count() or 1, 4, num_tiles)
        BATCH_SIZE = 2000  # Process tiles in batches to limit memory usage

        # Build jobs as a generator to avoid holding all features in memory
        def tile_job_iter():
            for y in range(min_ty, max_ty + 1):
                for x in range(min_tx, max_tx + 1):
                    features = tile_features.get((x, y))
                    if not features:
                        continue
                    tile_dir = os.path.join(output_dir, str(zoom), str(x))
                    tile_path = os.path.join(tile_dir, f"{y}.nav")
                    features.sort(key=lambda f: f['zoom_priority'] & 0x0F)
                    yield (features, tile_path, zoom, x, y, tolerance)

        completed = 0
        job_iter = tile_job_iter()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while True:
                batch = []
                for job in job_iter:
                    batch.append(job)
                    if len(batch) >= BATCH_SIZE:
                        break
                if not batch:
                    break

                futures = {executor.submit(write_nav_tile, *job): job for job in batch}
                batch.clear()  # Free batch memory, futures hold refs now

                for future in as_completed(futures):
                    completed += 1
                    progress = completed / num_tiles
                    bar_width = 25
                    filled = int(bar_width * progress)
                    bar = '\u2588' * filled + '\u2591' * (bar_width - filled)
                    print(f"\r  Zoom {zoom:2d}: Tiles [{bar}] {completed}/{num_tiles}", end='', flush=True)

                    result = future.result()
                    if result:
                        tiles_written += 1
                        tile_path = futures[future][1]
                        try:
                            total_size += os.path.getsize(tile_path)
                        except OSError:
                            pass

                futures.clear()  # Free completed futures before next batch

        # Clear memory before next zoom level
        tile_features.clear()

        zoom_elapsed = time.time() - zoom_start
        print(f"\r  Zoom {zoom:2d}: {tiles_written} tiles written. ({zoom_elapsed:.1f}s)" + " " * 20)
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
  - int16 relative coordinates (0-4096) for ~50% size reduction
  - Pre-calculated projection for ultra-fast rendering on ESP32
  - BBox-based culling for improved performance
  - Simple binary format optimized for streaming
        """
    )

    parser.add_argument('input_pbf', help='Input PBF file path')
    parser.add_argument('output_dir', help='Output directory for NAV tiles')
    parser.add_argument('config_file', help='Features configuration JSON file')
    parser.add_argument('--zoom', default='6-17',
                        help='Zoom level range (e.g., "6-17" or "12")')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging (show per-tile filtering stats)')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

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
