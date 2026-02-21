#!/usr/bin/env python3
"""Read a NAV tile (Delta+ZigZag+VarInt format) and print water polygon stats."""

import struct
import sys
from typing import Tuple

NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3
GEOM_TEXT = 4


def zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7


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
            payload_size = struct.unpack('<H', f.read(2))[0]
            payload = f.read(payload_size)

            coords = []
            if geom_type == GEOM_TEXT:
                if len(payload) >= 5:
                    px, py = struct.unpack('<hh', payload[0:4])
                    coords.append((px, py))
            else:
                offset = 0
                last_x, last_y = 0, 0
                for _ in range(coord_count):
                    zx, offset = read_varint(payload, offset)
                    zy, offset = read_varint(payload, offset)
                    last_x += zigzag_decode(zx)
                    last_y += zigzag_decode(zy)
                    coords.append((last_x, last_y))

            ring_ends = []
            if geom_type == GEOM_POLYGON:
                ring_count = struct.unpack('<H', payload[offset:offset + 2])[0]
                offset += 2
                for _ in range(ring_count):
                    ring_ends.append(struct.unpack('<H', payload[offset:offset + 2])[0])
                    offset += 2

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
