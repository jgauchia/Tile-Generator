import osmium
import shapely.wkb as wkblib
from shapely.geometry import (
    LineString, MultiLineString, Polygon, MultiPolygon, GeometryCollection, box, shape
)
import math
import struct
import json
import os
import shutil
from collections import defaultdict
from tqdm import tqdm
import argparse
import tempfile
import subprocess
import sys
import fiona
import gc

TILE_SIZE = 256
DRAW_COMMANDS = {
    'LINE': 1,
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,
    'HORIZONTAL_LINE': 5,
    'VERTICAL_LINE': 6,
}

UINT16_TILE_SIZE = 65536

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def deg2pixel(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = ((lon_deg + 180.0) / 360.0 * n * TILE_SIZE)
    y = ((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE)
    return x, y

def coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y):
    pixel_coords = []
    for lon, lat in coords:
        px_global, py_global = deg2pixel(lat, lon, zoom)
        x = (px_global - tile_x * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1)
        y = (py_global - tile_y * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1)
        x = int(round(x))
        y = int(round(y))
        x = max(0, min(UINT16_TILE_SIZE - 1, x))
        y = max(0, min(UINT16_TILE_SIZE - 1, y))
        pixel_coords.append((x, y))
    return pixel_coords

def remove_duplicate_points(points):
    if len(points) <= 1:
        return points
    result = [points[0]]
    for pt in points[1:]:
        if pt != result[-1]:
            result.append(pt)
    return result

def hex_to_rgb565(hex_color):
    try:
        if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#"):
            return 0xFFFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        return min(65535, max(0, rgb565))
    except Exception:
        return 0xFFFF

def hex_to_rgb332(hex_color):
    try:
        if not hex_color or not isinstance(hex_color, str) or not hex_color.startswith("#"):
            return 0xFF
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return ((r & 0xE0) | ((g & 0xE0) >> 3) | (b >> 6))
    except Exception:
        return 0xFF

def get_style_for_tags(tags, config):
    for k, v in tags.items():
        keyval = f"{k}={v}"
        if keyval in config:
            return config[keyval], keyval
    for k in tags:
        if k in config:
            return config[k], k
    return {}, None

def try_make_valid_polygon(coords):
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    try:
        poly = Polygon(coords)
        if poly.is_valid:
            return poly
        fixed = poly.buffer(0)
        if fixed.is_empty:
            return None
        if isinstance(fixed, (Polygon, MultiPolygon)):
            if isinstance(fixed, MultiPolygon):
                largest = max(fixed.geoms, key=lambda g: g.area)
                return largest
            return fixed
        elif isinstance(fixed, GeometryCollection):
            polys = [g for g in fixed.geoms if isinstance(g, Polygon)]
            if polys:
                largest = max(polys, key=lambda g: g.area)
                return largest
            return None
        else:
            return None
    except Exception:
        return None

def tile_latlon_bounds(tile_x, tile_y, zoom, pixel_margin=0):
    n = 2.0 ** zoom
    lon_min = tile_x / n * 360.0 - 180.0
    lat_rad1 = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_max = math.degrees(lat_rad1)
    lon_max = (tile_x + 1) / n * 360.0 - 180.0
    lat_rad2 = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + 1) / n)))
    lat_min = math.degrees(lat_rad2)
    return lon_min, lat_min, lon_max, lat_max

def is_area(tags):
    AREA_TAGS = {
        'building','landuse','amenity','leisure','tourism','waterway','natural','man_made',
        'boundary','place','aeroway','area','shop','craft','office','historic','public_transport',
        'emergency','military','ruins','power','sport','route','parking','park','garden','cemetery',
        'playground','school','university','hospital','forest','wood','meadow','farmland','orchard',
        'vineyard','wetland','scrub','heath','grass','beach','lake','reservoir','basin','pond',
        'swimming_pool','pitch','golf_course','stadium','sports_centre','theatre','museum','zoo','theme_park',
    }
    AREA_TAGS_EXCEPTIONS = {
        'waterway': {'riverbank', 'dock', 'reservoir', 'basin', 'canal', 'pond', 'ditch', 'fish_pass', 'moat', 'wetland'},
        'natural': {'water', 'wood', 'scrub', 'wetland', 'heath', 'grassland', 'sand', 'beach', 'glacier', 'fell', 'bare_rock', 'scree', 'shingle', 'bay', 'cape'},
    }
    if 'area' in tags:
        val = tags['area'].lower()
        if val == 'yes':
            return True
        elif val == 'no':
            return False
    for k, v in tags.items():
        if k in AREA_TAGS:
            if k == 'waterway':
                if v in AREA_TAGS_EXCEPTIONS.get('waterway', set()):
                    return True
                else:
                    return False
            if k == 'natural':
                if v in AREA_TAGS_EXCEPTIONS.get('natural', set()):
                    return True
                else:
                    return False
            return True
    return False

def get_simplify_tolerance_for_zoom(zoom):
    if zoom <= 10:
        return 0.05
    else:
        return None

def clamp_uint16(x):
    return max(0, min(UINT16_TILE_SIZE - 1, int(x)))

def pack_varint(n):
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return out

def pack_zigzag(n):
    return pack_varint((n << 1) ^ (n >> 31))

def pack_draw_commands(commands):
    out = bytearray()
    out += pack_varint(len(commands))
    for cmd in commands:
        t = cmd['type']
        color = cmd['color'] & 0xFF
        out += pack_varint(t)
        out += struct.pack("B", color)
        if t == DRAW_COMMANDS['LINE']:
            x1, y1, x2, y2 = map(clamp_uint16, [cmd['x1'], cmd['y1'], cmd['x2'], cmd['y2']])
            out += pack_zigzag(x1)
            out += pack_zigzag(y1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y2 - y1)
        elif t == DRAW_COMMANDS['POLYLINE'] or t == DRAW_COMMANDS['STROKE_POLYGON']:
            pts = cmd['points']
            out += pack_varint(len(pts))
            prev_x, prev_y = 0, 0
            for i, (x, y) in enumerate(pts):
                x, y = clamp_uint16(x), clamp_uint16(y)
                if i == 0:
                    out += pack_zigzag(x)
                    out += pack_zigzag(y)
                else:
                    out += pack_zigzag(x - prev_x)
                    out += pack_zigzag(y - prev_y)
                prev_x, prev_y = x, y
        elif t == DRAW_COMMANDS['HORIZONTAL_LINE']:
            x1, x2, y = clamp_uint16(cmd['x1']), clamp_uint16(cmd['x2']), clamp_uint16(cmd['y'])
            out += pack_zigzag(x1)
            out += pack_zigzag(x2 - x1)
            out += pack_zigzag(y)
        elif t == DRAW_COMMANDS['VERTICAL_LINE']:
            x, y1, y2 = clamp_uint16(cmd['x']), clamp_uint16(cmd['y1']), clamp_uint16(cmd['y2'])
            out += pack_zigzag(x)
            out += pack_zigzag(y1)
            out += pack_zigzag(y2 - y1)
    return out

def ensure_closed_ring(ring):
    if len(ring) < 3:
        return ring
    if ring[0] != ring[-1]:
        return ring + [ring[0]]
    return ring

def geometry_to_draw_commands(geom, color, tags, zoom, tile_x, tile_y, simplify_tolerance=None):
    commands = []
    def process_geom(g):
        local_cmds = []
        if g.is_empty:
            return local_cmds
        if g.geom_type == "Polygon":
            exterior = remove_duplicate_points(list(g.exterior.coords))
            exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
            exterior_pixels = ensure_closed_ring(exterior_pixels)
            if len(set(exterior_pixels)) >= 3:
                local_cmds.append({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color})
        elif g.geom_type == "MultiPolygon":
            for poly in g.geoms:
                exterior = remove_duplicate_points(list(poly.exterior.coords))
                exterior_pixels = coords_to_pixel_coords_uint16(exterior, zoom, tile_x, tile_y)
                exterior_pixels = ensure_closed_ring(exterior_pixels)
                if len(set(exterior_pixels)) >= 3:
                    local_cmds.append({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': exterior_pixels, 'color': color})
        elif g.geom_type == "LineString":
            coords = remove_duplicate_points(list(g.coords))
            if len(coords) < 2:
                return local_cmds
            pixel_coords = remove_duplicate_points(coords_to_pixel_coords_uint16(coords, zoom, tile_x, tile_y))
            if len(pixel_coords) < 2:
                return local_cmds
            is_closed = coords[0] == coords[-1]
            if is_closed and is_area(tags):
                if len(set(pixel_coords)) >= 3:
                    local_cmds.append({'type': DRAW_COMMANDS['STROKE_POLYGON'], 'points': pixel_coords, 'color': color})
            else:
                if len(pixel_coords) == 2:
                    x1, y1 = pixel_coords[0]
                    x2, y2 = pixel_coords[1]
                    if y1 == y2:
                        local_cmds.append({'type': DRAW_COMMANDS['HORIZONTAL_LINE'], 'x1': x1, 'x2': x2, 'y': y1, 'color': color})
                    elif x1 == x2:
                        local_cmds.append({'type': DRAW_COMMANDS['VERTICAL_LINE'], 'x': x1, 'y1': y1, 'y2': y2, 'color': color})
                    else:
                        local_cmds.append({'type': DRAW_COMMANDS['LINE'], 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'color': color})
                else:
                    local_cmds.append({'type': DRAW_COMMANDS['POLYLINE'], 'points': pixel_coords, 'color': color})
                if 'natural' in tags and tags['natural'] == 'coastline':
                    print("COASTLINE CMD:", pixel_coords)
        elif g.geom_type == "MultiLineString":
            for linestring in g.geoms:
                local_cmds.extend(process_geom(linestring))
        elif g.geom_type == "GeometryCollection":
            for subgeom in g.geoms:
                local_cmds.extend(process_geom(subgeom))
        return local_cmds
    if hasattr(geom, "is_valid") and not geom.is_empty:
        commands.extend(process_geom(geom))
    return commands

def get_layer_fields_from_pbf(pbf_file, layer):
    try:
        result = subprocess.run(
            ["ogrinfo", "-so", "-geom=NO", pbf_file, layer],
            capture_output=True, encoding="utf-8"
        )
        if result.returncode != 0:
            return set()
        lines = result.stdout.splitlines()
        fields = set()
        for line in lines:
            line = line.strip()
            if ":" in line:
                field = line.split(":", 1)[0].strip().replace('"', '')
                if field:
                    fields.add(field)
        return fields
    except Exception:
        return set()

def build_ogr2ogr_where_clause_from_config(config, allowed_fields):
    conds = []
    for k in config.keys():
        if "=" in k:
            key, val = k.split("=", 1)
            if key in allowed_fields:
                conds.append(f'("{key}" = \'{val}\')')
        else:
            if k in allowed_fields:
                conds.append(f'("{k}" IS NOT NULL)')
    where_clause = " OR ".join(conds)
    return where_clause

def get_config_fields(config):
    # Recoge los campos que aparecen en las claves del config
    fields = set()
    for k in config.keys():
        if "=" in k:
            key, _ = k.split("=", 1)
            fields.add(key)
        else:
            fields.add(k)
    return fields

def extract_geojson_from_pbf(pbf_file, geojson_file, config):
    print("Extracting PBF with ogr2ogr using SQL filter and minimal fields based on style...")
    if os.path.exists(geojson_file):
        os.remove(geojson_file)
    LAYER_FIELDS = {
        "points": {"highway", "place", "natural", "amenity", "railway"},
        "lines": {"highway", "waterway", "railway", "natural", "place"},
        "multilinestrings": {"highway", "waterway", "railway", "natural", "place"},
        "multipolygons": {"building", "landuse", "leisure", "natural", "place", "amenity"},
        "other_relations": {"place", "natural"}
    }
    layers = ["points", "lines", "multilinestrings", "multipolygons", "other_relations"]
    tmp_files = []
    config_fields = get_config_fields(config)
    for i, layer in enumerate(layers):
        possible = LAYER_FIELDS[layer]
        available = get_layer_fields_from_pbf(pbf_file, layer)
        allowed = possible & available & config_fields
        where_clause = build_ogr2ogr_where_clause_from_config(config, allowed)
        if not where_clause:
            print(f"Skipping layer {layer}: no matching fields in config/PBF.")
            continue
        print(f"[{i+1}/{len(layers)}] Extracting layer: {layer} (fields: {', '.join(sorted(allowed))})")
        tmp_layer_file = f"{geojson_file}_{layer}.tmp"
        if os.path.exists(tmp_layer_file):
            os.remove(tmp_layer_file)
        select_fields = ",".join(sorted(allowed))
        cmd = [
            "ogr2ogr",
            "-f", "GeoJSON",
            "-nlt", "PROMOTE_TO_MULTI",
            "-where", where_clause,
            "-select", select_fields,
            tmp_layer_file,
            pbf_file,
            layer
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"Error running ogr2ogr for layer {layer}:", result.stderr.decode())
            continue
        tmp_files.append(tmp_layer_file)
    if not tmp_files:
        print("Could not extract any layer from OSM PBF.")
        sys.exit(1)
    print("Merging temporary GeoJSONs...")
    features = []
    file_feature_counts = []
    print("Counting features in temporary files...")
    for tmp_file in tmp_files:
        with open(tmp_file, "r", encoding="utf-8") as f:
            gj = json.load(f)
            n = len(gj.get("features", []))
            file_feature_counts.append((tmp_file, n))
    total_features = sum(n for _, n in file_feature_counts)
    with tqdm(total=total_features, desc="Merging features") as pbar:
        for tmp_file, n in file_feature_counts:
            with open(tmp_file, "r", encoding="utf-8") as f:
                gj = json.load(f)
                feats = gj.get("features", [])
                # Limpiar las propiedades, solo dejar las presentes en config_fields
                for feat in feats:
                    feat['properties'] = {k: v for k, v in feat['properties'].items() if k in config_fields}
                features.extend(feats)
                pbar.update(len(feats))
            os.remove(tmp_file)
    print(f"Total merged features: {len(features)}")
    print(f"Writing merged features to {geojson_file}...")
    geojson_final = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(geojson_file, "w", encoding="utf-8") as f:
        json.dump(geojson_final, f)
    print(f"GeoJSON file generated successfully at {geojson_file}")
    del features
    gc.collect()

def read_features_from_geojson(geojson_file, config):
    features = []
    config_fields = get_config_fields(config)
    with fiona.open(geojson_file) as src:
        for feat in src:
            tags = {k: v for k, v in feat['properties'].items() if k in config_fields}
            geom = shape(feat['geometry'])
            style, stylekey = get_style_for_tags(tags, config)
            if not style:
                continue
            priority = style.get("priority", 5)
            color = hex_to_rgb332(style.get("color", "#FFFFFF"))
            zoom_filter = style.get("zoom", 6)
            features.append({
                "geom": geom,
                "color": color,
                "zoom_filter": zoom_filter,
                "tags": tags,
                "priority": priority
            })
    return features

def generate_tiles_all_zooms(features, output_dir, zoom_levels, max_file_size=65536):
    tile_features_by_zoom = {z: defaultdict(list) for z in zoom_levels}
    print("Assigning features to tiles for all zoom levels...")

    for feat in tqdm(features, desc="Assigning features"):
        for zoom in zoom_levels:
            if zoom < feat["zoom_filter"]:
                continue
            geom = feat["geom"]
            if not geom.is_valid or geom.is_empty:
                continue
            simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
            feature_geom = geom
            if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
                try:
                    feature_geom = feature_geom.simplify(simplify_tolerance, preserve_topology=True)
                except Exception:
                    pass
            if feature_geom.is_empty or not feature_geom.is_valid:
                continue

            minx, miny, maxx, maxy = feature_geom.bounds
            n = 2 ** zoom
            xtile_min, ytile_min = deg2num(miny, minx, zoom)
            xtile_max, ytile_max = deg2num(maxy, maxx, zoom)

            for xt in range(min(xtile_min, xtile_max), max(xtile_min, xtile_max) + 1):
                for yt in range(min(ytile_min, ytile_max), max(ytile_min, ytile_max) + 1):
                    t_lon_min, t_lat_min, t_lon_max, t_lat_max = tile_latlon_bounds(xt, yt, zoom)
                    tile_bbox = box(t_lon_min, t_lat_min, t_lon_max, t_lat_max)
                    try:
                        clipped_geom = feature_geom.intersection(tile_bbox)
                    except Exception:
                        continue
                    if not clipped_geom.is_empty:
                        new_feat = feat.copy()
                        new_feat["geom"] = clipped_geom
                        tile_features_by_zoom[zoom][(xt, yt)].append(new_feat)

    print("Writing tiles for all zoom levels...")

    for zoom in zoom_levels:
        tiles_features = tile_features_by_zoom[zoom]
        jobs = []
        tile_sizes = []
        for (x, y), feats in tiles_features.items():
            jobs.append((x, y, feats, zoom, output_dir, max_file_size, get_simplify_tolerance_for_zoom(zoom)))
        if not jobs:
            jobs.append((0, 0, [], zoom, output_dir, max_file_size, get_simplify_tolerance_for_zoom(zoom)))

        for job in tqdm(jobs, desc=f"Writing tiles (zoom {zoom})"):
            tile_size = tile_worker(job)
            tile_sizes.append(tile_size)
        avg_tile_size = sum(tile_sizes) / len(tile_sizes) if tile_sizes else 0
        print(f"Zoom {zoom}: average tile size = {avg_tile_size:.2f} bytes")

def tile_worker(args):
    x, y, feats, zoom, output_dir, max_file_size, simplify_tolerance = args

    ordered_feats = sorted(feats, key=lambda f: f.get("priority", 5))

    all_commands = []
    for feat in ordered_feats:
        cmds = geometry_to_draw_commands(
            feat["geom"], feat["color"], feat["tags"], zoom, x, y, simplify_tolerance=simplify_tolerance
        )
        all_commands.extend(cmds)

    tile_dir = os.path.join(output_dir, str(zoom), str(x))
    os.makedirs(tile_dir, exist_ok=True)
    filename = os.path.join(tile_dir, f"{y}.bin")

    buffer = bytearray()
    num_cmds_written = 0
    header = pack_varint(0)
    buffer += header

    for cmd in all_commands:
        cmd_bytes = pack_draw_commands([cmd])[len(pack_varint(1)):]
        tmp_num_cmds = num_cmds_written + 1
        tmp_header = pack_varint(tmp_num_cmds)
        tmp_buffer_size = len(tmp_header) + len(buffer[len(header):]) + len(cmd_bytes)
        if tmp_buffer_size > max_file_size:
            break
        buffer = tmp_header + buffer[len(header):] + cmd_bytes
        header = tmp_header
        num_cmds_written += 1

    if num_cmds_written == 0:
        buffer = pack_varint(0)
    with open(filename, "wb") as f:
        f.write(buffer)

    tile_size = len(buffer)
    del all_commands, ordered_feats, buffer
    gc.collect()
    return tile_size

def main():
    parser = argparse.ArgumentParser(description="OSM vector tile generator (bin format, uint16 tile coordinates, visually lossless, optimized fields)")
    parser.add_argument("pbf_file", help="Path to .pbf file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("config_file", help="JSON config with features/colors")
    parser.add_argument("--zoom", help="Zoom level or range (e.g. 12 or 6-17)", default="6-17")
    parser.add_argument("--max-file-size", help="Maximum file size in KB", type=int, default=128)
    args = parser.parse_args()
    if "-" in args.zoom:
        start, end = map(int, args.zoom.split("-"))
        zoom_levels = list(range(start, end + 1))
    else:
        zoom_levels = [int(args.zoom)]
    max_file_size = args.max_file_size * 1024

    with open(args.config_file, "r") as f:
        config = json.load(f)

    geojson_tmp = os.path.abspath("tmp_extract.geojson")
    extract_geojson_from_pbf(args.pbf_file, geojson_tmp, config)

    print("Reading features from merged GeoJSON...")
    features = read_features_from_geojson(geojson_tmp, config)
    print(f"Total features collected: {len(features)}")

    generate_tiles_all_zooms(features, args.output_dir, zoom_levels, max_file_size)
    print("Process completed successfully.")

    if os.path.exists(geojson_tmp):
        os.remove(geojson_tmp)
    del features, config
    gc.collect()

if __name__ == "__main__":
    main()