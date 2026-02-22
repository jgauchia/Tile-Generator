#!/usr/bin/env python3
"""Debug script - show ALL water polygons."""

import struct

NAV_MAGIC = b'NAV1'
GEOM_POLYGON = 3

nav_path = 'output_vect/ariege/15/16511/11962.nav'

with open(nav_path, 'rb') as f:
    magic = f.read(4)
    if magic != NAV_MAGIC:
        print("ERROR: Invalid NAV magic")
        exit(1)

    feature_count = struct.unpack('<H', f.read(2))[0]
    f.read(16)  # Skip global BBox

    print(f"Total features: {feature_count}\n")
    print("Water polygons (0xae00-0xaeff):")
    print("-" * 100)

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

        # Show ALL water polygons
        if geom_type == GEOM_POLYGON and 0xae00 <= color_rgb565 <= 0xaeff:
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            print(f"#{feat_idx:4d}: pts={coord_count:3d}, color=0x{color_rgb565:04x}, "
                  f"X=[{x_min:5d},{x_max:5d}] span={x_max-x_min:4d}, "
                  f"Y=[{y_min:5d},{y_max:5d}] span={y_max-y_min:4d}, "
                  f"rings={len(ring_ends) if ring_ends else 1}")

print("-" * 100)
print(f"\nLooking for polygon with 87 points and X=-409 to 1902, Y=2572 to 4017")
