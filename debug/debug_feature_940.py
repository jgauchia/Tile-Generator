#!/usr/bin/env python3
"""Analyze feature #940 rendering with different viewport positions."""

import struct
import math

NAV_MAGIC = b'NAV1'
GEOM_POLYGON = 3
TILE_SIZE = 256

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

# Read feature #940
nav_path = 'output_vect/test/15/16511/11962.nav'
tile_x, tile_y, zoom = 16511, 11962, 15

with open(nav_path, 'rb') as f:
    magic = f.read(4)
    feature_count = struct.unpack('<H', f.read(2))[0]
    f.read(16)

    for feat_idx in range(feature_count):
        geom_type = struct.unpack('<B', f.read(1))[0]
        color_rgb565 = struct.unpack('<H', f.read(2))[0]
        zoom_priority = struct.unpack('<B', f.read(1))[0]
        width = struct.unpack('<B', f.read(1))[0]
        bbox = struct.unpack('<BBBB', f.read(4))
        coord_count = struct.unpack('<H', f.read(2))[0]
        f.read(1)

        coords = []
        for _ in range(coord_count):
            px, py = struct.unpack('<hh', f.read(4))
            coords.append((px, py))

        ring_ends = []
        if geom_type == GEOM_POLYGON:
            ring_count = struct.unpack('<H', f.read(2))[0]
            for _ in range(ring_count):
                ring_ends.append(struct.unpack('<H', f.read(2))[0])

        if feat_idx == 940:
            print(f"=== FEATURE #940 - THE GARONNE ===\n")
            print(f"Geometry: POLYGON")
            print(f"Points: {coord_count}")
            print(f"Color: 0x{color_rgb565:04x} (#aad3df water)")
            print(f"Rings: {len(ring_ends) if ring_ends else 1}")

            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            print(f"\nTile coordinates:")
            print(f"  X: [{min(x_coords)}, {max(x_coords)}] (span={max(x_coords)-min(x_coords)})")
            print(f"  Y: [{min(y_coords)}, {max(y_coords)}] (span={max(y_coords)-min(y_coords)})")
            print(f"\nFirst 10 points: {coords[:10]}")
            print(f"Last 3 points: {coords[-3:]}")

            # Calculate center of tile for proper viewing
            tile_center_lat, tile_center_lon = num2deg(tile_x + 0.5, tile_y + 0.5, zoom)
            print(f"\n=== VIEWPORT ANALYSIS ===")
            print(f"\nTile {tile_x}/{tile_y} center: lat={tile_center_lat:.6f}, lon={tile_center_lon:.6f}")

            # Test with tile centered in viewport
            print(f"\n--- Test 1: Tile CENTERED in viewport ---")
            center_lat, center_lon = tile_center_lat, tile_center_lon
            center_x, center_y = deg2num(center_lat, center_lon, zoom)
            tl_x, tl_y = center_x - 1.5, center_y - 1.5
            tx_rel = tile_x - tl_x
            ty_rel = tile_y - tl_y

            print(f"Center: lat={center_lat:.6f}, lon={center_lon:.6f}")
            print(f"Tile position in viewport: ({tx_rel:.3f}, {ty_rel:.3f})")

            # Calculate screen coords
            screen_coords = []
            for px, py in coords:
                fx = tx_rel + (px / 4096.0)
                fy = ty_rel + (py / 4096.0)
                sx = int(fx * TILE_SIZE)
                sy = int(fy * TILE_SIZE)
                screen_coords.append((sx, sy))

            sx_min = min(c[0] for c in screen_coords)
            sx_max = max(c[0] for c in screen_coords)
            sy_min = min(c[1] for c in screen_coords)
            sy_max = max(c[1] for c in screen_coords)

            print(f"Screen coords: X=[{sx_min}, {sx_max}] ({sx_max-sx_min} px), Y=[{sy_min}, {sy_max}] ({sy_max-sy_min} px)")

            # Clip rect
            clip_x = int(tx_rel * TILE_SIZE)
            clip_y = int(ty_rel * TILE_SIZE)
            print(f"Tile clip rect: [{clip_x}, {clip_x+TILE_SIZE}] × [{clip_y}, {clip_y+TILE_SIZE}]")

            # Visible area
            vis_x_min = max(sx_min, clip_x, 0)
            vis_x_max = min(sx_max, clip_x + TILE_SIZE, 768)
            vis_y_min = max(sy_min, clip_y, 0)
            vis_y_max = min(sy_max, clip_y + TILE_SIZE, 768)

            if vis_x_max > vis_x_min and vis_y_max > vis_y_min:
                print(f"VISIBLE in viewport [0-768]: X=[{vis_x_min}, {vis_x_max}] ({vis_x_max-vis_x_min} px), Y=[{vis_y_min}, {vis_y_max}] ({vis_y_max-vis_y_min} px)")
            else:
                print(f"NOT VISIBLE in viewport [0-768]")

            print(f"\n✓ To view this polygon, run:")
            print(f"  python tile_viewer_devel.py output_vect/test --lat {tile_center_lat:.6f} --lon {tile_center_lon:.6f} --zoom {zoom}")

            break
