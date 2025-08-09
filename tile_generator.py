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
import threading
import time

import ijson
import decimal

from concurrent.futures import ProcessPoolExecutor, as_completed

max_workers = os.cpu_count()

TILE_SIZE = 256
DRAW_COMMANDS = {
    'LINE': 1,
    'POLYLINE': 2,
    'STROKE_POLYGON': 3,
    'HORIZONTAL_LINE': 5,
    'VERTICAL_LINE': 6,
    'SET_COLOR': 0x80,  # Comando de estado para optimización de colores
}

UINT16_TILE_SIZE = 65536

def extract_layer_to_tmpfile(args):
    layer, i, layers, geojson_file, pbf_file, config, LAYER_FIELDS, config_fields = args
    logs = []
    possible = LAYER_FIELDS[layer]
    available = get_layer_fields_from_pbf(pbf_file, layer)
    allowed = possible & available & config_fields
    where_clause = build_ogr2ogr_where_clause_from_config(config, allowed)
    if not where_clause:
        logs.append(f"Skipping layer {layer}: no matching fields in config/PBF.")
        return None, logs
    logs.append(f"[{i+1}/{len(layers)}] Extracted layer: {layer} (fields: {', '.join(sorted(allowed))})")
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
        logs.append(f"Error running ogr2ogr for layer {layer}: {result.stderr.decode()}")
        return None, logs
    return tmp_layer_file, logs

def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError

def read_tmp_geojson_features_stream(tmp_file, config_fields):
    with open(tmp_file, "r", encoding="utf-8") as f:
        for feat in ijson.items(f, "features.item"):
            feat['properties'] = {k: v for k, v in feat['properties'].items() if k in config_fields}
            yield feat

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
        y = ((py_global - tile_y * TILE_SIZE) * (UINT16_TILE_SIZE - 1) / (TILE_SIZE - 1))
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

def insert_color_commands(commands):
    """
    Inserta comandos SET_COLOR y elimina campo color de comandos de geometría.
    Esto reduce la redundancia cuando múltiples comandos consecutivos usan el mismo color.
    """
    if not commands:
        return commands
    
    result = []
    current_color = None
    color_commands_inserted = 0
    
    for cmd in commands:
        cmd_color = cmd.get('color')
        
        # Si cambia el color, insertar SET_COLOR
        if cmd_color != current_color:
            result.append({
                'type': DRAW_COMMANDS['SET_COLOR'], 
                'color': cmd_color
            })
            current_color = cmd_color
            color_commands_inserted += 1
        
        # Agregar comando sin campo color
        cmd_copy = {k: v for k, v in cmd.items() if k != 'color'}
        result.append(cmd_copy)
    
    # Calcular ahorro real: eliminamos N colores incrustados, agregamos M comandos SET_COLOR
    original_commands = len(commands)
    optimized_commands = len(result)
    colors_removed = original_commands  # Cada comando original tenía un color
    colors_added = color_commands_inserted  # Comandos SET_COLOR agregados
    net_savings = colors_removed - colors_added  # Ahorro neto en bytes
    
    return result, net_savings

def pack_draw_commands(commands):
    """
    Empaqueta comandos de dibujo en formato binario optimizado.
    Soporta comandos SET_COLOR para reducir redundancia de colores.
    """
    out = bytearray()
    out += pack_varint(len(commands))
    
    for cmd in commands:
        t = cmd['type']
        out += pack_varint(t)
        
        if t == DRAW_COMMANDS['SET_COLOR']:
            # Solo empaqueta el color
            color = cmd['color'] & 0xFF
            out += struct.pack("B", color)
        else:
            # Para comandos de geometría, NO incluir color (se usa current_color)
            # Empaquetar datos geométricos
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
    fields = set()
    for k in config.keys():
        if "=" in k:
            key, _ = k.split("=", 1)
            fields.add(key)
        else:
            fields.add(k)
    return fields

def count_features(tmp_files, config_fields):
    total = 0
    for tmp_file in tmp_files:
        with open(tmp_file, "r", encoding="utf-8") as f:
            for _ in ijson.items(f, "features.item"):
                total += 1
    return total

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
    config_fields = get_config_fields(config)

    print("Extracting layers ...")
    extract_args = [
        (layer, i, layers, geojson_file, pbf_file, config, LAYER_FIELDS, config_fields)
        for i, layer in enumerate(layers)
    ]
    tmp_files = []
    all_logs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_layer_to_tmpfile, arg) for arg in extract_args]
        for future in as_completed(futures):
            tmp_file, logs = future.result()
            all_logs.extend(logs)
            if tmp_file:
                tmp_files.append(tmp_file)
    for log in all_logs:
        print(log)
    if not tmp_files:
        print("Could not extract any layer from OSM PBF.")
        sys.exit(1)

    print("Counting total features ...")
    total_features_to_merge = count_features(tmp_files, config_fields)
    print(f"Total features to merge: {total_features_to_merge}")

    print("Merging and writing temporary GeoJSONs ...")
    total_features = 0
    with open(geojson_file, "w", encoding="utf-8") as out, tqdm(total=total_features_to_merge, desc="Merging features") as pbar:
        out.write('{"type": "FeatureCollection", "features": [\n')
        first = True
        counter = 0
        for tmp_file in tmp_files:
            for feat in read_tmp_geojson_features_stream(tmp_file, config_fields):
                if not first:
                    out.write(',\n')
                json.dump(feat, out, default=decimal_default)
                first = False
                total_features += 1
                counter += 1
                pbar.update(1)
                if counter % 10000 == 0:
                    gc.collect()
        out.write('\n]}\n')

    for tmp_file in tmp_files:
        os.remove(tmp_file)
    print(f"Total merged features: {total_features}")
    print(f"GeoJSON file generated successfully at {geojson_file}")
    gc.collect()
    return total_features_to_merge

def streaming_assign_features_to_tiles_by_zoom(geojson_file, config, output_dir, zoom_levels, max_file_size=65536, total_features=None, summary_stats=None):
    config_fields = get_config_fields(config)

    # Prepare a mapping: zoom -> set(tags) that should be visible in that zoom
    zoom_to_valid_tags = {}
    for zoom in zoom_levels:
        valid_tags = set()
        for k, v in config.items():
            if v.get("zoom", 0) <= zoom:
                valid_tags.add(k)
        zoom_to_valid_tags[zoom] = valid_tags

    # For total features progress bar
    total_features_count = total_features if total_features is not None else None

    import psutil
    process = psutil.Process(os.getpid())

    for zoom in zoom_levels:
        print(f"\n========== Processing zoom level {zoom} ==========")
        print(f"[Zoom {zoom}] Step 1: Reading relevant features from GeoJSON...")

        # Assign valid tags for this zoom
        valid_tags = zoom_to_valid_tags[zoom]

        # tile_buffers: {(xt, yt): [list of commands]}
        tile_buffers = defaultdict(list)
        assigned_features = 0

        mem_start = process.memory_info().rss / 1024 / 1024
        t0 = time.time()

        # Step 1: reading and assignment (streaming, only relevant features)
        with open(geojson_file, "r", encoding="utf-8") as f, tqdm(desc=f"[Zoom {zoom}] Reading & assignment", total=total_features_count) as pbar_read:
            for feat in ijson.items(f, "features.item"):
                tags = {k: v for k, v in feat['properties'].items() if k in config_fields}
                style, stylekey = get_style_for_tags(tags, config)
                if not style:
                    pbar_read.update(1)
                    continue
                # Does this feature apply to the current zoom?
                if stylekey not in valid_tags:
                    pbar_read.update(1)
                    continue
                if not feat.get('geometry'):
                    pbar_read.update(1)
                    continue
                try:
                    geom = shape(feat['geometry'])
                except Exception:
                    pbar_read.update(1)
                    continue
                if not geom.is_valid or geom.is_empty:
                    pbar_read.update(1)
                    continue

                simplify_tolerance = get_simplify_tolerance_for_zoom(zoom)
                feature_geom = geom
                if simplify_tolerance is not None and geom.geom_type in ("LineString", "MultiLineString"):
                    try:
                        feature_geom = feature_geom.simplify(simplify_tolerance, preserve_topology=True)
                    except Exception:
                        pass
                if feature_geom.is_empty or not feature_geom.is_valid:
                    pbar_read.update(1)
                    continue

                priority = style.get("priority", 5)
                color = hex_to_rgb332(style.get("color", "#FFFFFF"))

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
                            cmds = geometry_to_draw_commands(
                                clipped_geom, color, tags, zoom, xt, yt, simplify_tolerance=simplify_tolerance
                            )
                            for cmd in cmds:
                                cmd['priority'] = priority
                                tile_buffers[(xt, yt)].append(cmd)
                assigned_features += 1
                pbar_read.update(1)
        print(f"[Zoom {zoom}] Assigned features: {assigned_features}")

        print(f"[Zoom {zoom}] Step 2: Optimizing and creating tiles...")
        total_bytes_saved = 0
        tiles_optimized = 0
        total_tiles_processed = 0
        
        with tqdm(total=len(tile_buffers), desc=f"[Zoom {zoom}] Creating optimized tiles") as pbar_tiles:
            for (xt, yt), cmds in tile_buffers.items():
                tile_dir = os.path.join(output_dir, str(zoom), str(xt))
                os.makedirs(tile_dir, exist_ok=True)
                filename = os.path.join(tile_dir, f"{yt}.bin")
                
                # Paso 1: Ordenar por prioridad y color
                cmds_sorted = sorted(cmds, key=lambda c: (c['priority'], c['color']))
                
                # Paso 2: Insertar comandos SET_COLOR y calcular ahorro
                cmds_optimized, bytes_saved = insert_color_commands(cmds_sorted)
                
                # Empaquetar comandos optimizados
                buffer = pack_draw_commands(cmds_optimized)
                
                # Escribir archivo
                with open(filename, "wb") as fbin:
                    fbin.write(buffer)
                
                # Estadísticas de optimización
                if bytes_saved > 0:
                    tiles_optimized += 1
                    total_bytes_saved += bytes_saved
                
                total_tiles_processed += 1
                pbar_tiles.update(1)
        
        # Calcular estadísticas de optimización mejoradas
        avg_savings_per_tile = total_bytes_saved / max(total_tiles_processed, 1)
        optimization_ratio = (tiles_optimized / max(total_tiles_processed, 1)) * 100
        
        print(f"[Zoom {zoom}] SET_COLOR Optimization results:")
        print(f"  - Tiles optimized: {tiles_optimized}/{total_tiles_processed} ({optimization_ratio:.1f}%)")
        print(f"  - Total bytes saved: {total_bytes_saved} bytes")
        print(f"  - Average savings per optimized tile: {avg_savings_per_tile:.1f} bytes")
        
        tile_buffers.clear()
        gc.collect()
        t1 = time.time()
        mem_end = process.memory_info().rss / 1024 / 1024

        if summary_stats is not None:
            summary_stats.append({
                "Zoom level": zoom,
                "Number of elements": assigned_features,
                "Memory usage (MB)": int(mem_end),
                "Processing time (s)": round(t1 - t0, 2),
                "Notes": f"SET_COLOR: {tiles_optimized}/{total_tiles_processed} tiles, {total_bytes_saved}B saved"
            })

        print(f"[Zoom {zoom}] Tiles written with SET_COLOR optimization.")

def print_summary_table(summary_stats):
    print('\n' + '+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*21 + '+' + '-'*50 + '+')
    print('| {:<10} | {:<19} | {:<18} | {:<19} | {:<48} |'.format(
        "Zoom level", "Number of elements", "Memory usage (MB)", "Processing time (s)", "Notes"))
    print('+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*21 + '+' + '-'*50 + '+')

    for entry in summary_stats:
        print('| {:<10} | {:<19} | {:<18} | {:<19} | {:<48} |'.format(
            entry["Zoom level"],
            entry["Number of elements"],
            entry["Memory usage (MB)"],
            entry["Processing time (s)"],
            entry["Notes"][:48]  # Truncate long notes
        ))
    print('+' + '-'*12 + '+' + '-'*21 + '+' + '-'*20 + '+' + '-'*21 + '+' + '-'*50 + '+')

def main():
    parser = argparse.ArgumentParser(description="OSM vector tile generator with SET_COLOR optimization (Paso 2 COMPLETADO)")
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
    total_features_to_merge = extract_geojson_from_pbf(args.pbf_file, geojson_tmp, config)

    print("Reading features and assigning to tiles with SET_COLOR optimization...")
    summary_stats = []
    streaming_assign_features_to_tiles_by_zoom(geojson_tmp, config, args.output_dir, zoom_levels, max_file_size, total_features=total_features_to_merge, summary_stats=summary_stats)
    print("Process completed successfully with SET_COLOR optimization.")

    print("\nProcessing completed. Summary:")
    print_summary_table(summary_stats)

    if os.path.exists(geojson_tmp):
        os.remove(geojson_tmp)
    gc.collect()

if __name__ == "__main__":
    main()