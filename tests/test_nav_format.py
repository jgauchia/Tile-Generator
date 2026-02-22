#!/usr/bin/env python3
"""Integration tests: write NAV tiles and read them back, verify roundtrip."""

import sys
import os
import struct
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_writer import _zigzag_encode, _to_varint
from tile_viewer import read_nav_tile, _zigzag_decode, _read_varint, GEOM_POLYGON, GEOM_LINESTRING, GEOM_TEXT

NAV_MAGIC = b'NAV1'
COORD_SCALE = 10000000


def _write_test_tile(path, features_data):
    """Write a minimal NAV tile with given features.

    features_data: list of dicts with keys depending on geom_type:
      Polygon/Line: geom_type, color, zoom_priority, width_byte, rings (list of list of (x,y))
      Text: geom_type, color, zoom_priority, font_size, px, py, text_bytes, shield (optional dict)
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sHiiii', NAV_MAGIC, len(features_data),
                            0, 0, 0, 0))

        for feat in features_data:
            gt = feat['geom_type']

            if gt == GEOM_TEXT:
                px, py = feat['px'], feat['py']
                text_bytes = feat['text_bytes']
                text_len = len(text_bytes)
                has_shield = 'shield' in feat
                data_size = 4 + 1 + text_len + (4 if has_shield else 0)
                coord_count = (data_size + 3) // 4
                padded_size = coord_count * 4

                text_payload = bytearray()
                text_payload.extend(struct.pack('<hh', px, py))
                text_payload.extend(struct.pack('<B', text_len))
                text_payload.extend(text_bytes)
                if has_shield:
                    text_payload.extend(struct.pack('<H', feat['shield']['bg']))
                    text_payload.extend(struct.pack('<H', feat['shield']['border']))
                padding = padded_size - data_size
                if padding > 0:
                    text_payload.extend(b'\x00' * padding)

                bx = max(0, min(255, px >> 4))
                by = max(0, min(255, py >> 4))

                f.write(struct.pack('<B', GEOM_TEXT))
                f.write(struct.pack('<H', feat['color']))
                f.write(struct.pack('<B', feat['zoom_priority']))
                f.write(struct.pack('<B', feat.get('font_size', 0)))
                f.write(struct.pack('<BBBB', bx, by, bx, by))
                f.write(struct.pack('<H', coord_count))
                f.write(struct.pack('<H', len(text_payload)))
                f.write(text_payload)
            else:
                rings = feat['rings']
                total_points = sum(len(r) for r in rings)

                coord_buffer = bytearray()
                last_x, last_y = 0, 0
                f_min_x, f_min_y = 4096, 4096
                f_max_x, f_max_y = 0, 0
                for ring in rings:
                    for px, py in ring:
                        dx = px - last_x
                        dy = py - last_y
                        coord_buffer.extend(_to_varint(_zigzag_encode(dx)))
                        coord_buffer.extend(_to_varint(_zigzag_encode(dy)))
                        last_x, last_y = px, py
                        c_px = max(0, min(4096, px))
                        c_py = max(0, min(4096, py))
                        f_min_x, f_min_y = min(f_min_x, c_px), min(f_min_y, c_py)
                        f_max_x, f_max_y = max(f_max_x, c_px), max(f_max_y, c_py)

                extra_payload = bytearray()
                if gt == GEOM_POLYGON:
                    extra_payload.extend(struct.pack('<H', len(rings)))
                    current_end = 0
                    for ring in rings:
                        current_end += len(ring)
                        extra_payload.extend(struct.pack('<H', current_end))

                payload_size = len(coord_buffer) + len(extra_payload)
                bx1 = max(0, min(255, f_min_x >> 4))
                by1 = max(0, min(255, f_min_y >> 4))
                bx2 = max(0, min(255, f_max_x >> 4))
                by2 = max(0, min(255, f_max_y >> 4))

                f.write(struct.pack('<B', gt))
                f.write(struct.pack('<H', feat['color']))
                f.write(struct.pack('<B', feat['zoom_priority']))
                f.write(struct.pack('<B', feat.get('width_byte', 0)))
                f.write(struct.pack('<BBBB', bx1, by1, bx2, by2))
                f.write(struct.pack('<H', total_points))
                f.write(struct.pack('<H', payload_size))
                f.write(coord_buffer)
                f.write(extra_payload)


def test_simple_polygon():
    coords = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]
    feat = {
        'geom_type': GEOM_POLYGON,
        'color': 0x1234,
        'zoom_priority': 0xA3,
        'rings': [coords],
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)

        assert len(features) == 1, f"expected 1 feature, got {len(features)}"
        f = features[0]
        assert f.geom_type == GEOM_POLYGON
        assert f.color_rgb565 == 0x1234
        assert f.zoom_priority == 0xA3
        assert f.coords == coords, f"coords mismatch:\n  expected: {coords}\n  got:      {f.coords}"
        assert f.ring_ends == [5], f"ring_ends mismatch: expected [5], got {f.ring_ends}"
    finally:
        os.unlink(path)

    print("  simple polygon: OK")


def test_polygon_with_hole():
    exterior = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]
    hole = [(1000, 1000), (2000, 1000), (2000, 2000), (1000, 2000), (1000, 1000)]
    feat = {
        'geom_type': GEOM_POLYGON,
        'color': 0x5678,
        'zoom_priority': 0x52,
        'rings': [exterior, hole],
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)

        assert len(features) == 1
        f = features[0]
        assert f.geom_type == GEOM_POLYGON
        assert f.ring_ends == [5, 10], f"ring_ends mismatch: expected [5, 10], got {f.ring_ends}"
        assert f.coords == exterior + hole
        rings = f.get_rings()
        assert len(rings) == 2
        assert rings[0] == exterior
        assert rings[1] == hole
    finally:
        os.unlink(path)

    print("  polygon with hole: OK")


def test_linestring():
    coords = [(100, 200), (500, 600), (1000, 300), (1500, 800),
              (2000, 100), (2500, 500), (3000, 700), (3500, 400),
              (4000, 900), (4096, 50)]
    feat = {
        'geom_type': GEOM_LINESTRING,
        'color': 0xABCD,
        'zoom_priority': 0xEF,
        'width_byte': 5,
        'rings': [coords],
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)

        assert len(features) == 1
        f = features[0]
        assert f.geom_type == GEOM_LINESTRING
        assert f.color_rgb565 == 0xABCD
        assert f.coords == coords, f"coords mismatch:\n  expected: {coords}\n  got:      {f.coords}"
    finally:
        os.unlink(path)

    print("  linestring (10 pts): OK")


def test_text_feature():
    text = "Rue Test"
    text_bytes = text.encode('utf-8')
    feat = {
        'geom_type': GEOM_TEXT,
        'color': 0x0000,
        'zoom_priority': 0xE8,
        'font_size': 1,
        'px': 2048,
        'py': 2048,
        'text_bytes': text_bytes,
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)

        assert len(features) == 1
        f = features[0]
        assert f.geom_type == GEOM_TEXT
        assert f.text == text, f"text mismatch: expected '{text}', got '{f.text}'"
        assert f.coords == [(2048, 2048)]
    finally:
        os.unlink(path)

    print("  text feature: OK")


def test_text_with_shield():
    text = "D42"
    text_bytes = text.encode('utf-8')
    feat = {
        'geom_type': GEOM_TEXT,
        'color': 0x0000,
        'zoom_priority': 0xE8,
        'font_size': 0,
        'px': 1000,
        'py': 500,
        'text_bytes': text_bytes,
        'shield': {'bg': 0xFFFF, 'border': 0x0000},
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)

        assert len(features) == 1
        f = features[0]
        assert f.geom_type == GEOM_TEXT
        assert f.text == text
        assert f.coords == [(1000, 500)]
    finally:
        os.unlink(path)

    print("  text with shield: OK")


def test_mixed_features():
    """Multiple feature types in a single tile."""
    polygon = {
        'geom_type': GEOM_POLYGON,
        'color': 0x1111,
        'zoom_priority': 0x01,
        'rings': [[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]],
    }
    line = {
        'geom_type': GEOM_LINESTRING,
        'color': 0x2222,
        'zoom_priority': 0xEF,
        'width_byte': 3,
        'rings': [[(50, 50), (200, 200), (400, 100)]],
    }
    text = {
        'geom_type': GEOM_TEXT,
        'color': 0x0000,
        'zoom_priority': 0xE8,
        'font_size': 2,
        'px': 300,
        'py': 300,
        'text_bytes': b'Place',
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [polygon, line, text])
        features = read_nav_tile(path, 5, 10)

        assert len(features) == 3, f"expected 3 features, got {len(features)}"
        assert features[0].geom_type == GEOM_POLYGON
        assert features[0].coords == [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        assert features[1].geom_type == GEOM_LINESTRING
        assert features[1].coords == [(50, 50), (200, 200), (400, 100)]
        assert features[2].geom_type == GEOM_TEXT
        assert features[2].text == 'Place'
        assert features[2].tile_x == 5
        assert features[2].tile_y == 10
    finally:
        os.unlink(path)

    print("  mixed features (polygon + line + text): OK")


def test_casing_flag():
    """Width byte bit 7 encodes casing/building flag."""
    feat = {
        'geom_type': GEOM_LINESTRING,
        'color': 0x3333,
        'zoom_priority': 0xEF,
        'width_byte': 0x85,  # casing=1, width=5
        'rings': [[(0, 0), (100, 100)]],
    }

    with tempfile.NamedTemporaryFile(suffix='.nav', delete=False) as tmp:
        path = tmp.name

    try:
        _write_test_tile(path, [feat])
        features = read_nav_tile(path, 0, 0)
        f = features[0]
        assert f.needs_casing is True
        assert f.width == 5 / 2.0
    finally:
        os.unlink(path)

    print("  casing flag: OK")


if __name__ == '__main__':
    print("=== NAV Format Integration Tests ===")
    test_simple_polygon()
    test_polygon_with_hole()
    test_linestring()
    test_text_feature()
    test_text_with_shield()
    test_mixed_features()
    test_casing_flag()
    print("\nAll tests passed")
