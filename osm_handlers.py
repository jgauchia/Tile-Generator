"""
OSM data extraction handlers.

Classes BoundaryScanner and OSMHandler for processing PBF files.
"""

import logging
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import osmium
import osmium.geom

try:
    import shapely.wkb
except ImportError:
    pass

from constants import (
    GEOM_POINT, GEOM_LINESTRING, GEOM_POLYGON, GEOM_TEXT,
    POINT_FEATURES, TEXT_FEATURES, WIDTH_TAGS,
)
from geo_utils import (
    SHAPELY_AVAILABLE,
    hex_to_rgb565, lighten_rgb565, darken_rgb565,
    pack_zoom_priority, get_layer_for_tags,
    get_zoom_for_tags, get_color_for_tags,
    densify_linestring,
)

logger = logging.getLogger(__name__)


class BoundaryScanner(osmium.SimpleHandler):
    """First pass: collect way IDs that are members of boundary relations."""

    def __init__(self, config: Dict, max_zoom: int):
        super().__init__()
        self.config = config
        self.max_zoom = max_zoom
        self.boundary_ways: Dict[int, List[Dict]] = {}  # way_id -> list of boundary configs

    def relation(self, r):
        tags = {tag.k: tag.v for tag in r.tags}
        if tags.get('type') != 'boundary' or tags.get('boundary') != 'administrative':
            return

        admin_level = tags.get('admin_level', '')
        feature_key = f"boundary=administrative;admin_level={admin_level}"
        if feature_key not in self.config or not isinstance(self.config[feature_key], dict):
            return

        cfg = self.config[feature_key]
        min_zoom = cfg.get('zoom', 6)
        if min_zoom > self.max_zoom:
            return

        color_rgb565 = hex_to_rgb565(cfg.get('color', '#000000'))

        width_pixels = cfg.get('width', 1)

        way_info = {
            'color_rgb565': color_rgb565,
            'zoom_priority': pack_zoom_priority(min_zoom, 15),
            'width_pixels': width_pixels,
        }

        for member in r.members:
            if member.type == 'w':
                if member.ref not in self.boundary_ways:
                    self.boundary_ways[member.ref] = []
                self.boundary_ways[member.ref].append(way_info)


class OSMHandler(osmium.SimpleHandler):
    """Handler for processing OSM data from PBF files."""

    def __init__(self, config: Dict, zoom_range: Tuple[int, int]):
        super().__init__()
        self.config = config
        self.min_zoom, self.max_zoom = zoom_range
        self.features: List[Dict] = []
        self.boundary_ways: Dict[int, Dict] = {}  # Set by caller after BoundaryScanner
        self.stats = {
            'nodes_processed': 0,
            'text_labels': 0,
            'ways_processed': 0,
            'areas_processed': 0,
            'boundary_ways_extracted': 0,
            'features_extracted': 0,
            'features_filtered': 0,
            'area_no_config': 0,
            'area_no_layer': 0,
            'area_zoom_filtered': 0,
            'area_exception': 0,
            'area_boundary': 0
        }
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.progress_interval = 5
        self.interesting_tags = self._build_interesting_tags()
        self.processed_way_ids: Set[int] = set()
        self.wkbfab = osmium.geom.WKBFactory()
        self.road_label_counters: Dict[str, int] = defaultdict(int)  # ref -> segment count

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

    def node(self, n):
        """Process node - extract point features (peaks) and text labels (places)."""
        if not n.location.valid():
            return

        tags = self._tags_to_dict(n.tags)

        for key, value in tags.items():
            feature_key = f"{key}={value}"

            # Point symbols (peaks, volcanoes)
            if feature_key in POINT_FEATURES and feature_key in self.config:
                cfg = self.config[feature_key]
                min_zoom = cfg.get('zoom', 12)
                if min_zoom > self.max_zoom:
                    return

                color_rgb565 = hex_to_rgb565(cfg.get('color', '#000000'))
                layer = get_layer_for_tags({key: value})

                self.features.append({
                    'geom_type': GEOM_POINT,
                    'coords': [(n.location.lon, n.location.lat)],
                    'color_rgb565': color_rgb565,
                    'zoom_priority': pack_zoom_priority(min_zoom, 15),
                    'width_meters': 0.0,
                    'shape': POINT_FEATURES[feature_key],
                    'layer': layer or 'terrain',
                })
                self.stats['nodes_processed'] += 1
                return

            # Text labels (places)
            if feature_key in TEXT_FEATURES and feature_key in self.config:
                name = tags.get('name', '')
                if not name:
                    self.stats['features_filtered'] += 1
                    return

                text_cfg = TEXT_FEATURES[feature_key]
                cfg = self.config[feature_key]

                # Determine zoom from population
                population = 0
                pop_str = tags.get('population', '0')
                try:
                    population = int(pop_str.replace(',', '').replace(' ', ''))
                except (ValueError, TypeError):
                    pass

                min_zoom = text_cfg['zoom_rules'][-1][1]  # default: last rule
                for min_pop, z in text_cfg['zoom_rules']:
                    if population >= min_pop:
                        min_zoom = z
                        break

                if min_zoom > self.max_zoom:
                    return

                color_rgb565 = hex_to_rgb565(cfg.get('color', '#000000'))

                # Priority based on population - higher population = on top
                if population >= 500000:
                    nibble = 15  # Major cities (Paris, Lyon, Marseille, Toulouse...)
                elif population >= 100000:
                    nibble = 14  # Large cities
                elif population >= 15000:
                    nibble = 13  # Towns (Cugnaux, Muret...)
                else:
                    nibble = 12  # Small towns and villages

                # Split long names on 2 lines at hyphen or space near middle
                if len(name) > 12:
                    mid = len(name) // 2
                    best = -1
                    best_dist = len(name)
                    for i, c in enumerate(name):
                        if c in ('-', ' '):
                            dist = abs(i - mid)
                            if dist < best_dist:
                                best_dist = dist
                                best = i
                    if best > 0:
                        if name[best] == '-':
                            name = name[:best+1] + '\n' + name[best+1:]
                        else:
                            name = name[:best] + '\n' + name[best+1:]

                name_bytes = name.encode('utf-8')[:255]

                self.features.append({
                    'geom_type': GEOM_TEXT,
                    'coords': [(n.location.lon, n.location.lat)],
                    'color_rgb565': color_rgb565,
                    'zoom_priority': pack_zoom_priority(min_zoom, nibble),
                    'font_size': text_cfg['font_size'],
                    'text': name_bytes,
                    'population': population,
                    'layer': 'places',
                })
                self.stats['text_labels'] += 1
                self.stats['nodes_processed'] += 1
                return

    def way(self, w):
        """Process way - extract roads and linear features."""
        self.stats['ways_processed'] += 1
        self._log_progress()

        # Check if this way is part of boundary relations
        if w.id in self.boundary_ways:
            coords = []
            for node in w.nodes:
                if node.location.valid():
                    coords.append((node.location.lon, node.location.lat))
            if len(coords) >= 2:
                for bnd in self.boundary_ways[w.id]:
                    self.features.append({
                        'geom_type': GEOM_LINESTRING,
                        'coords': coords,
                        'color_rgb565': bnd['color_rgb565'],
                        'zoom_priority': bnd['zoom_priority'],
                        'width_meters': 0.0,
                        'width_pixels': bnd.get('width_pixels', 1),
                        'name': w.tags.get('name', ''),
                        'layer': 'boundaries',
                        'id': w.id,
                        'highway_type': '',
                    })
                    self.stats['boundary_ways_extracted'] += 1

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

        # Railway service tracks (yard, siding, spur): push to z13 per OSM Carto
        if 'railway' in tags and tags.get('service') in ('yard', 'siding', 'spur', 'crossover'):
            min_zoom = max(min_zoom, 13)
            if min_zoom > self.max_zoom:
                self.stats['features_filtered'] += 1
                return

        is_closed = len(coords) >= 4 and coords[0] == coords[-1]

        # Tags that automatically qualify a closed way as an area/polygon
        area_qualifiers = {
            'building', 'landuse', 'water', 'amenity', 'leisure', 'natural',
            'waterway', 'man_made', 'aeroway', 'historic', 'military'
        }

        has_area_tag = any(k in tags for k in area_qualifiers)
        # Explicitly force area for specific water body tags
        if tags.get('natural') == 'bay' or \
           tags.get('landuse') == 'reservoir' or \
           tags.get('waterway') == 'riverbank':
            has_area_tag = True
        is_area_tags = is_closed and (has_area_tag or tags.get('area') == 'yes')

        color = get_color_for_tags(tags, self.config)
        color_rgb565 = hex_to_rgb565(color)

        if is_closed and is_area_tags and 'highway' not in tags:
            # This logic will be handled by area(), but we might catch some here.
            # Assign a polygon nibble just in case.
            nibble = 3 if layer == 'water' else 2

            subclass = tags.get('natural', '') or tags.get('landuse', '') or tags.get('leisure', '')

            feature = {
                'id': w.id,
                'geom_type': GEOM_POLYGON,
                'coords': coords,
                'color_rgb565': color_rgb565,
                'zoom_priority': pack_zoom_priority(min_zoom, nibble),
                'width_meters': 0.0,  # Polygons don't use width
                'subclass': subclass,  # Store for merge logic
                'layer': layer,
                'is_building': layer == 'buildings',
                'name': tags.get('name', ''),
            }
            self.features.append(feature)
            self.stats['features_extracted'] += 1
            self.processed_way_ids.add(w.id)
            return

        # Extract width in meters for roads/railways/waterways
        width_meters = 0.0
        if any(tag in tags for tag in WIDTH_TAGS):
            width_meters = self._get_width_meters(tags)

        # Do not draw centerlines for wide water bodies (polygons should be used instead)
        if layer == 'water' and width_meters >= 2.0:
            self.stats['features_filtered'] += 1
            return

        # Store line type for zoom-based width lookup
        highway_type = tags.get('highway', '') or tags.get('railway', '') or tags.get('aeroway', '')
        ref = tags.get('ref', '')
        old_ref = tags.get('old_ref', '')
        name = tags.get('name', '')

        # Fixed Z-order (nibble) for rendering priority (9-15: Roads/Railways, above water at 8)
        priority_map = {
            # Z=15: Railways (above all roads for level crossings priority)
            'rail': 15, 'subway': 15, 'tram': 15, 'light_rail': 15,
            'narrow_gauge': 15, 'funicular': 15, 'monorail': 15,
            # Z=14: Major roads & motorways
            'motorway': 14, 'trunk': 14, 'primary': 14,
            # Z=13: Secondary roads
            'secondary': 13, 'tertiary': 13,
            # Z=12: Residential and minor roads
            'residential': 12, 'unclassified': 12, 'living_street': 12, 'pedestrian': 12,
            # Z=10-11: Links/ramps differentiated by hierarchy
            'motorway_link': 12, 'trunk_link': 11, 'primary_link': 11, 'secondary_link': 10, 'tertiary_link': 10,
            # Z=9: Service, tracks and paths
            'service': 9, 'track': 9, 'path': 9, 'footway': 9, 'cycleway': 9
        }

        if layer == 'water':
            nibble = 8
        else:
            nibble = priority_map.get(highway_type, 9)  # Default for other minor ways

        # Bridges: all bridges render at nibble 15 (above all at-grade roads)
        is_bridge = tags.get('bridge') in ('yes', 'viaduct')
        if is_bridge:
            nibble = 15

        # Tunnels: shift down to ensure below ground level while preserving hierarchy
        # motorway tunnel: max(13-11,1)=2, secondary tunnel: max(12-11,1)=1, etc.
        if tags.get('tunnel') in ('yes', 'culvert'):
            nibble = max(nibble - 11, 1)  # Shift down by 11, minimum 1

        # Densify curves for smooth rendering (add intermediate points)
        # Only for roads and railways - NOT aeroways (runways/taxiways should stay straight)
        is_aeroway = tags.get('aeroway', '') != ''
        if highway_type and not is_aeroway and len(coords) >= 2:
            coords = densify_linestring(coords, max_segment_degrees=0.0001)

        feature = {
            'id': w.id,
            'geom_type': GEOM_LINESTRING,
            'coords': coords,
            'color_rgb565': color_rgb565,
            'zoom_priority': pack_zoom_priority(min_zoom, nibble),
            'width_meters': width_meters,
            'highway_type': highway_type,
            'layer': layer,
            'is_bridge': is_bridge,
            'has_ref': bool(ref),
            'ref': ref,
            'old_ref': old_ref,
            'name': name,
        }
        self.features.append(feature)
        self.stats['features_extracted'] += 1

        # Create road number label for major roads with ref
        # Display at z10: A* (autoroutes), N* (nationales), D1xxx with old_ref=N* (major former nationales)
        if ref and highway_type in ('motorway', 'trunk', 'primary', 'secondary'):
            # Filter by road number (ref), not highway_type:
            # - A* : motorways (all)
            # - N* : national roads (all)
            # - D1000-D1999 : only major former national roads with old_ref=N* (e.g., D1124 was N124)
            should_create_label = False
            if ref.startswith('A') or ref.startswith('N'):
                should_create_label = True
            elif ref.startswith('D'):
                # Extract number from D-road (e.g., "D1124" -> 1124)
                try:
                    d_number = int(ref[1:])
                    # Only D1000-D1999 (major former nationales) with old_ref=N*
                    if 1000 <= d_number <= 1999 and old_ref and old_ref.startswith('N'):
                        should_create_label = True
                except (ValueError, IndexError):
                    pass  # Invalid D-road format, skip

            if should_create_label:
                # Space out labels: only create one every 25 segments
                self.road_label_counters[ref] += 1
                if self.road_label_counters[ref] % 25 == 1:
                    # Generate 3 candidate positions (25%, 50%, 75%) for collision avoidance
                    candidates = []
                    for ratio in [0.25, 0.5, 0.75]:
                        idx = int(len(coords) * ratio)
                        candidates.append(coords[idx])

                    ref_label = {
                        'geom_type': GEOM_TEXT,
                        'coords': [candidates[1]],
                        'coords_candidates': candidates,
                        'color_rgb565': darken_rgb565(color_rgb565),  # Text: dark
                        'bg_color_rgb565': lighten_rgb565(color_rgb565),  # Background: light
                        'border_color_rgb565': color_rgb565,  # Border: original
                        'zoom_priority': pack_zoom_priority(10, 98),
                        'font_size': 2,
                        'text': ref.encode('utf-8')[:32],
                        'population': 0,
                    }
                    self.features.append(ref_label)
                    self.stats['features_extracted'] += 1

    def _get_width_meters(self, tags: Dict[str, str]) -> float:
        """Extract width in meters from OSM tags.

        Priority:
        1. width=* tag (meters)
        2. lanes=* tag (lanes * 3.5m)
        3. Return 0 (will become 1 pixel default)
        """
        # Check for explicit width tag
        if 'width' in tags:
            try:
                width_str = tags['width'].replace('m', '').replace(' ', '').strip()
                return float(width_str)
            except (ValueError, TypeError):
                pass

        # Check for lanes tag
        if 'lanes' in tags:
            try:
                lanes = int(tags['lanes'])
                return lanes * 3.5  # Standard lane width
            except (ValueError, TypeError):
                pass

        return 0.0

    def area(self, a):
        """Process area - handles multipolygon relations."""
        self.stats['areas_processed'] += 1
        self._log_progress()

        tags = self._tags_to_dict(a.tags)

        # Skip boundary relations
        if tags.get('boundary') == 'administrative':
            self.stats['area_boundary'] += 1
            return

        # Check if feature is in config and has a layer mapping
        if not self._is_feature_in_config(tags):
            self.stats['area_no_config'] += 1
            return

        layer = get_layer_for_tags(tags)
        if layer is None:
            self.stats['area_no_layer'] += 1
            return

        # Force water layer identity for correct hole processing (islands)
        if (tags.get('natural') == 'water' or
            tags.get('natural') == 'bay' or
            tags.get('waterway') == 'riverbank' or
            tags.get('landuse') == 'reservoir'):
            layer = 'water'

        # Force buildings layer for any polygon with building tag
        # This ensures hangars (aeroway=hangar + building=yes/hangar) render
        # as buildings (#d9d0c9 beige) not as aeroway infrastructure (#dadae0 grey)
        # Per OSM wiki: hangars have both aeroway=hangar and building=* tags
        if 'building' in tags:
            layer = 'buildings'

        # Removed the 'highway in tags' filter that was causing issues

        min_zoom = get_zoom_for_tags(tags, self.config)
        if min_zoom > self.max_zoom:
            self.stats['area_zoom_filtered'] += 1
            return

        # Construction de la geometrie
        try:
            wkb = self.wkbfab.create_multipolygon(a)
            geom = shapely.wkb.loads(wkb, hex=True)
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    self.stats['area_exception'] += 1
                    return

            # Fixed Z-order (nibble) for polygon layers
            layer_to_nibble = {
                'aeroways': 1,                   # Z=1: Airport base
                'landuse': 2, 'terrain': 2,      # Z=2: Landcover (residential, forest, farmland)
                'leisure': 4, 'amenities': 4,    # Z=4: Parks, recreation grounds, amenities
                'pitch': 5,                      # Z=5: Pitches above recreation grounds
                'surface': 5,                    # Z=5: Ground cover (grass, meadow) inside leisure zones
                'parking': 5,                    # Z=5: Parking lots inside leisure zones
                'infrastructure': 6,
                'buildings': 7,                  # Z=7: Buildings (above infrastructure)
                'water': 8,                      # Z=8: Water above all polygons, below roads (9+)
            }
            nibble = layer_to_nibble.get(layer, 2)

            # leisure=track renders above other leisure polygons (sports_centre background)
            if tags.get('leisure') == 'track':
                nibble = 6

            # Bridge polygons render above water (nibble 8)
            if tags.get('bridge') in ('yes', 'viaduct') or tags.get('man_made') == 'bridge':
                nibble = 9

            color = get_color_for_tags(tags, self.config)
            color_rgb565 = hex_to_rgb565(color)

            # Force water color to ensure consistency, overriding JSON
            if layer == 'water':
                color_rgb565 = hex_to_rgb565("#aad3df")

            # Extract subclass for landcover discrimination (wood/forest vs farmland)
            subclass = tags.get('natural', '') or tags.get('landuse', '') or tags.get('leisure', '')

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

                inner_rings = []
                if poly.interiors:
                    for interior in poly.interiors:
                        if len(interior.coords) >= 4:
                            inner_rings.append(list(interior.coords))

                feature_data = {
                    'geom_type': GEOM_POLYGON,
                    'coords': coords,
                    'color_rgb565': color_rgb565,
                    'zoom_priority': pack_zoom_priority(min_zoom, nibble),
                    'width_meters': 0.0,
                    'inner_rings': inner_rings,
                    'subclass': subclass,  # Store for merge logic
                    'layer': layer,  # Store layer name for inner_rings handling
                    'is_building': layer == 'buildings',
                    'name': tags.get('name', ''),
                }

                self.features.append(feature_data)
                self.stats['features_extracted'] += 1
        except Exception as e:
            self.stats['area_exception'] += 1
            # Debug: log first 10 errors
            if self.stats['area_exception'] <= 10:
                logger.warning(f"Area extraction failed: {e} | tags: {tags}")
