#!/usr/bin/env python3
import struct
import sys

NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3
GEOM_TEXT = 4

def read_nav_file(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != NAV_MAGIC:
            print(f"Invalid magic: {magic}")
            return

        feature_count = struct.unpack('<H', f.read(2))[0]
        f.read(16)  # Skip bbox

        print(f"Total features: {feature_count}\n")

        water_polygons = []

        for i in range(feature_count):
            geom_type = struct.unpack('<B', f.read(1))[0]
            color_rgb565 = struct.unpack('<H', f.read(2))[0]
            zoom_priority = struct.unpack('<B', f.read(1))[0]
            width_byte = struct.unpack('<B', f.read(1))[0]
            bbox = struct.unpack('<BBBB', f.read(4))
            coord_count = struct.unpack('<H', f.read(2))[0]
            f.read(1)  # padding

            # Read coords
            coords = []
            if geom_type == GEOM_TEXT:
                # TEXT has different format: read entire data block
                data = f.read(coord_count * 4)
            else:
                for _ in range(coord_count):
                    px, py = struct.unpack('<hh', f.read(4))
                    coords.append((px, py))

            # Read rings if polygon
            ring_ends = []
            if geom_type == GEOM_POLYGON:
                ring_count = struct.unpack('<H', f.read(2))[0]
                for _ in range(ring_count):
                    ring_ends.append(struct.unpack('<H', f.read(2))[0])

            # Check if water (rgb565=0xae9b) and polygon
            if color_rgb565 == 0xae9b and geom_type == GEOM_POLYGON:
                water_polygons.append({
                    'index': i,
                    'points': coord_count,
                    'rings': len(ring_ends),
                    'coords': coords
                })

        print(f"Water polygons: {len(water_polygons)}")
        for wp in water_polygons:
            coords = wp['coords']
            if coords:
                xs = [x for x, y in coords]
                ys = [y for x, y in coords]
                y_min, y_max = min(ys), max(ys)
                print(f"  [{wp['index']}] points={wp['points']}, Y: {y_min} to {y_max}")
                if wp['points'] > 20:
                    print(f"      X range: {min(xs)} to {max(xs)} (span={max(xs)-min(xs)})")
                    print(f"      Y range: {y_min} to {y_max} (span={y_max-y_min})")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_nav_read.py <path/to/tile.nav>")
        sys.exit(1)
    read_nav_file(sys.argv[1])
