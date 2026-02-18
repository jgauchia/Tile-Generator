#!/usr/bin/env python3
"""Debug NAV files to check if roads are present."""

import struct
from typing import List, Tuple

NAV_MAGIC = b'NAV1'
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3

def zigzag_decode(n: int) -> int:
    """ZigZag decode an integer."""
    return (n >> 1) ^ -(n & 1)

def read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    """Read a VarInt from buffer at offset. Returns (value, new_offset)."""
    result = 0
    shift = 0
    while True:
        if offset >= len(buffer):
            raise IndexError("VarInt read out of bounds")
        b = buffer[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, offset
        shift += 7

def rgb565_to_rgb888(c: int) -> Tuple[int, int, int]:
    """Convert 16-bit RGB565 to 24-bit RGB888."""
    r = ((c >> 11) & 0x1F) << 3
    g = ((c >> 5) & 0x3F) << 2
    b = (c & 0x1F) << 3
    return (r, g, b)

def analyze_nav_tile(path: str):
    """Analyze NAV tile and print statistics."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {path}")
    print(f"{'='*80}")

    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != NAV_MAGIC:
                print(f"ERROR: Invalid magic: {magic}")
                return

            feature_count = struct.unpack('<H', f.read(2))[0]
            bbox = struct.unpack('<HHHH', f.read(8))
            f.read(8)  # Skip remaining bbox

            print(f"Feature count: {feature_count}")
            print(f"Global BBox: {bbox}")

            linestrings = []
            polygons = []
            points = []

            for idx in range(feature_count):
                # Read Header (13 bytes)
                header_data = f.read(13)
                if len(header_data) < 13:
                    print(f"ERROR: Truncated header at feature {idx}")
                    break

                geom_type = header_data[0]
                color_rgb565 = struct.unpack('<H', header_data[1:3])[0]
                zoom_priority = header_data[3]
                width = header_data[4] & 0x7F
                needs_casing = bool(header_data[4] & 0x80)
                bbox = struct.unpack('<BBBB', header_data[5:9])
                coord_count = struct.unpack('<H', header_data[9:11])[0]
                payload_size = struct.unpack('<H', header_data[11:13])[0]

                # Read Payload
                payload = f.read(payload_size)
                if len(payload) < payload_size:
                    print(f"ERROR: Truncated payload at feature {idx}")
                    break

                # Decode Coordinates
                offset = 0
                last_x, last_y = 0, 0
                coords = []

                for _ in range(coord_count):
                    dx, offset = read_varint(payload, offset)
                    dy, offset = read_varint(payload, offset)

                    dx = zigzag_decode(dx)
                    dy = zigzag_decode(dy)

                    px = last_x + dx
                    py = last_y + dy

                    coords.append((px, py))
                    last_x, last_y = px, py

                feature_info = {
                    'idx': idx,
                    'geom_type': geom_type,
                    'color': color_rgb565,
                    'color_rgb': rgb565_to_rgb888(color_rgb565),
                    'zoom_priority': zoom_priority,
                    'min_zoom': zoom_priority >> 4,
                    'priority': zoom_priority & 0x0F,
                    'width': width,
                    'casing': needs_casing,
                    'bbox': bbox,
                    'coord_count': coord_count,
                    'coords': coords
                }

                if geom_type == GEOM_LINESTRING:
                    linestrings.append(feature_info)
                elif geom_type == GEOM_POLYGON:
                    polygons.append(feature_info)
                elif geom_type == GEOM_POINT:
                    points.append(feature_info)

            print(f"\nFeature types:")
            print(f"  LineStrings: {len(linestrings)}")
            print(f"  Polygons: {len(polygons)}")
            print(f"  Points: {len(points)}")

            # Show some LineString details (likely roads)
            print(f"\n{'='*80}")
            print(f"LineStrings (roads) - showing first 20:")
            print(f"{'='*80}")
            for i, ls in enumerate(linestrings[:20]):
                rgb = ls['color_rgb']
                print(f"  [{ls['idx']:3d}] pts={ls['coord_count']:3d}, "
                      f"width={ls['width']:2d}, casing={ls['casing']}, "
                      f"color=RGB({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d}), "
                      f"zoom={ls['min_zoom']:2d}, priority={ls['priority']:2d}, "
                      f"bbox={ls['bbox']}")
                # Show first few coords
                if ls['coords']:
                    print(f"       First coords: {ls['coords'][:3]}")
                    if any(px < -8192 or px > 12288 or py < -8192 or py > 12288
                           for px, py in ls['coords']):
                        print(f"       ⚠️  WARNING: Coords out of visible range!")

    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Analyze the problematic tiles
    tiles = [
        "output_vect/test/15/16514/11962.nav",  # Bd Pierre et Marie Curie + Bd Silvio Trentin
        "output_vect/test/15/16515/11962.nav",  # Bd Pierre et Marie Curie
        "output_vect/test/15/16511/11966.nav",  # Avenue de Lardenne
    ]

    for tile in tiles:
        analyze_nav_tile(tile)
