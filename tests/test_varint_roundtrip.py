#!/usr/bin/env python3
"""Roundtrip tests for ZigZag and VarInt encoding used in NAV format."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_writer import _zigzag_encode, _to_varint
from tile_viewer import _zigzag_decode, _read_varint


def test_zigzag_roundtrip():
    values = [0, 1, -1, 127, -128, 4096, -4096, 32767, -32768]
    for v in values:
        encoded = _zigzag_encode(v)
        decoded = _zigzag_decode(encoded)
        assert decoded == v, f"zigzag roundtrip failed: {v} -> {encoded} -> {decoded}"
    print("  zigzag roundtrip: OK")


def test_zigzag_known_values():
    assert _zigzag_encode(0) == 0
    assert _zigzag_encode(-1) == 1
    assert _zigzag_encode(1) == 2
    assert _zigzag_encode(-2) == 3
    assert _zigzag_encode(2) == 4
    print("  zigzag known values: OK")


def test_varint_roundtrip():
    values = [0, 1, 127, 128, 255, 16383, 16384, 65535]
    for v in values:
        encoded = _to_varint(v)
        decoded, end_offset = _read_varint(bytes(encoded), 0)
        assert decoded == v, f"varint roundtrip failed: {v} -> {list(encoded)} -> {decoded}"
        assert end_offset == len(encoded), f"varint offset wrong: expected {len(encoded)}, got {end_offset}"
    print("  varint roundtrip: OK")


def test_varint_encoding_size():
    assert len(_to_varint(0)) == 1
    assert len(_to_varint(127)) == 1
    assert len(_to_varint(128)) == 2
    assert len(_to_varint(16383)) == 2
    assert len(_to_varint(16384)) == 3
    print("  varint encoding size: OK")


def test_delta_zigzag_varint_roundtrip():
    coords = [(0, 0), (4096, 0), (4096, 4096), (0, 4096), (0, 0)]

    buf = bytearray()
    last_x, last_y = 0, 0
    for px, py in coords:
        dx = px - last_x
        dy = py - last_y
        buf.extend(_to_varint(_zigzag_encode(dx)))
        buf.extend(_to_varint(_zigzag_encode(dy)))
        last_x, last_y = px, py

    decoded = []
    offset = 0
    last_x, last_y = 0, 0
    payload = bytes(buf)
    for _ in range(len(coords)):
        zx, offset = _read_varint(payload, offset)
        zy, offset = _read_varint(payload, offset)
        last_x += _zigzag_decode(zx)
        last_y += _zigzag_decode(zy)
        decoded.append((last_x, last_y))

    assert decoded == coords, f"delta roundtrip failed:\n  input:   {coords}\n  decoded: {decoded}"
    print("  delta+zigzag+varint roundtrip (background polygon): OK")


def test_delta_negative_coords():
    coords = [(100, 200), (-50, 300), (4200, -100), (0, 0)]

    buf = bytearray()
    last_x, last_y = 0, 0
    for px, py in coords:
        dx = px - last_x
        dy = py - last_y
        buf.extend(_to_varint(_zigzag_encode(dx)))
        buf.extend(_to_varint(_zigzag_encode(dy)))
        last_x, last_y = px, py

    decoded = []
    offset = 0
    last_x, last_y = 0, 0
    payload = bytes(buf)
    for _ in range(len(coords)):
        zx, offset = _read_varint(payload, offset)
        zy, offset = _read_varint(payload, offset)
        last_x += _zigzag_decode(zx)
        last_y += _zigzag_decode(zy)
        decoded.append((last_x, last_y))

    assert decoded == coords, f"delta roundtrip with negatives failed:\n  input:   {coords}\n  decoded: {decoded}"
    print("  delta+zigzag+varint roundtrip (negative coords): OK")


def test_varint_stream_multiple():
    """Multiple varints concatenated in a buffer are read correctly."""
    values = [0, 1, 127, 128, 16384, 65535, 42]
    buf = bytearray()
    for v in values:
        buf.extend(_to_varint(v))

    payload = bytes(buf)
    offset = 0
    decoded = []
    for _ in values:
        v, offset = _read_varint(payload, offset)
        decoded.append(v)

    assert decoded == values, f"stream roundtrip failed:\n  input:   {values}\n  decoded: {decoded}"
    assert offset == len(payload), f"stream did not consume all bytes: {offset} != {len(payload)}"
    print("  varint stream (multiple concat): OK")


if __name__ == '__main__':
    print("=== VarInt Roundtrip Tests ===")
    test_zigzag_roundtrip()
    test_zigzag_known_values()
    test_varint_roundtrip()
    test_varint_encoding_size()
    test_delta_zigzag_varint_roundtrip()
    test_delta_negative_coords()
    test_varint_stream_multiple()
    print("\nAll tests passed")
