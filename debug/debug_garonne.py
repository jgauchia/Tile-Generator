#!/usr/bin/env python3
"""Debug script to analyze the Garonne polygon rendering issue."""

import struct
import math

NAV_MAGIC = b'NAV1'
GEOM_POLYGON = 3
TILE_SIZE = 256

def deg2num(lat_deg: float, lon_deg: float, zoom: int):
    """Convert lat/lon to fractional tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile

def analyze_tile(nav_path, tile_x, tile_y, center_lat, center_lon, zoom):
    """Analyze water polygons in a tile."""
    print(f"\n{'='*80}")
    print(f"Analyzing tile {tile_x}/{tile_y} at zoom {zoom}")
    print(f"Center: lat={center_lat:.6f}, lon={center_lon:.6f}")
    print(f"{'='*80}\n")

    # Calculate viewport
    center_x, center_y = deg2num(center_lat, center_lon, zoom)
    tl_x, tl_y = center_x - 1.5, center_y - 1.5
    tx_rel = tile_x - tl_x
    ty_rel = tile_y - tl_y

    print(f"Viewport: center_tile=({center_x:.3f}, {center_y:.3f}), top_left=({tl_x:.3f}, {tl_y:.3f})")
    print(f"Tile position in viewport: ({tx_rel:.3f}, {ty_rel:.3f})\n")

    with open(nav_path, 'rb') as f:
        magic = f.read(4)
        if magic != NAV_MAGIC:
            print("ERROR: Invalid NAV magic")
            return

        feature_count = struct.unpack('<H', f.read(2))[0]
        f.read(16)  # Skip global BBox
        print(f"Total features in tile: {feature_count}\n")

        water_count = 0
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

            # Check if this is a water polygon (color close to 0xae9b)
            if geom_type == GEOM_POLYGON and 0xae00 <= color_rgb565 <= 0xaeff:
                water_count += 1

                # Calculate extent in tile coords
                x_coords = [c[0] for c in coords]
                y_coords = [c[1] for c in coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_span = x_max - x_min
                y_span = y_max - y_min

                # Skip tiny polygons
                if x_span < 100 and y_span < 100:
                    continue

                print(f"--- Water Polygon #{feat_idx} (water #{water_count}) ---")
                print(f"  Color: 0x{color_rgb565:04x}")
                print(f"  Points: {coord_count} (rings: {len(ring_ends) if ring_ends else 1})")
                print(f"  BBox (8-bit): {bbox}")
                print(f"  Tile coords: X=[{x_min}, {x_max}] (span={x_span}), Y=[{y_min}, {y_max}] (span={y_span})")

                # Convert to screen coordinates
                screen_coords = []
                for px, py in coords:
                    fx = tx_rel + (px / 4096.0)
                    fy = ty_rel + (py / 4096.0)
                    sx = int(fx * TILE_SIZE)
                    sy = int(fy * TILE_SIZE)
                    screen_coords.append((sx, sy))

                sx_coords = [c[0] for c in screen_coords]
                sy_coords = [c[1] for c in screen_coords]
                sx_min, sx_max = min(sx_coords), max(sx_coords)
                sy_min, sy_max = min(sy_coords), max(sy_coords)

                print(f"  Screen coords: X=[{sx_min}, {sx_max}] (span={sx_max-sx_min}), Y=[{sy_min}, {sy_max}] (span={sy_max-sy_min})")

                # Check clip rect
                clip_x = int(tx_rel * TILE_SIZE)
                clip_y = int(ty_rel * TILE_SIZE)
                print(f"  Clip rect: [{clip_x}, {clip_x + TILE_SIZE}] × [{clip_y}, {clip_y + TILE_SIZE}]")

                # Calculate visible area
                vis_x_min = max(sx_min, clip_x)
                vis_x_max = min(sx_max, clip_x + TILE_SIZE)
                vis_y_min = max(sy_min, clip_y)
                vis_y_max = min(sy_max, clip_y + TILE_SIZE)

                if vis_x_max > vis_x_min and vis_y_max > vis_y_min:
                    print(f"  VISIBLE area: X=[{vis_x_min}, {vis_x_max}] ({vis_x_max-vis_x_min} px), Y=[{vis_y_min}, {vis_y_max}] ({vis_y_max-vis_y_min} px)")
                else:
                    print(f"  VISIBLE area: NONE (completely clipped)")

                print(f"  First 5 points (tile): {coords[:5]}")
                print(f"  First 5 points (screen): {screen_coords[:5]}")
                print()

if __name__ == '__main__':
    # Toulouse coordinates from the issue
    nav_path = 'output_vect/ariege/15/16511/11962.nav'
    tile_x, tile_y = 16511, 11962
    zoom = 15

    # Center of Toulouse
    center_lat = 43.6047
    center_lon = 1.4442

    analyze_tile(nav_path, tile_x, tile_y, center_lat, center_lon, zoom)
