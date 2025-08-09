# Tile Binary Format Specification

This document describes the binary format produced by the tile generation scripts (`tile_generator.py`) for vector map tiles. The format is intended for efficient rendering and compact storage of map data at various zoom levels. Applications and libraries can use this specification to parse and render the resulting `.bin` files.

---

## File Overview

Each file represents a single tile for a given zoom level (`z`), x coordinate (`x`), and y coordinate (`y`).  
The filename and directory structure is:  
```
{output_dir}/{z}/{x}/{y}.bin
```

The file contains a sequence of drawing commands encoded in a compact binary format.  
All coordinates are encoded as `uint16` values (range 0–65535) relative to the tile.

---

## File Structure

The file is a single binary blob with the structure:

| Field          | Type      | Description                               |
|----------------|-----------|-------------------------------------------|
| num_commands   | varint    | Number of drawing commands in the tile    |
| commands[]     | variable  | Sequence of drawing commands              |

- All variable-length integers use [protobuf varint encoding](https://developers.google.com/protocol-buffers/docs/encoding#varints).
- All signed integers are encoded with [zigzag encoding](https://developers.google.com/protocol-buffers/docs/encoding#signed-integers).

---

## Drawing Command Structure

Each command is encoded as follows:

| Field        | Type      | Description                                 |
|--------------|-----------|---------------------------------------------|
| type         | varint    | Drawing command type (see below)            |
| color        | uint8     | Color index (RGB332, 8 bits, 0xFF=none)     |
| parameters   | variable  | Command-specific data (see below)           |

### Command Types

| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| LINE                | 1     | Single line (from x1,y1 to x2,y2)                    |
| POLYLINE            | 2     | Polyline (sequence of points)                        |
| STROKE_POLYGON      | 3     | Closed polygon outline (sequence of points)          |
| HORIZONTAL_LINE     | 5     | Horizontal line (from x1 to x2 at y)                 |
| VERTICAL_LINE       | 6     | Vertical line (from y1 to y2 at x)                   |

---

## Command Parameters

### LINE (type 1)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x1, y1      | int32   | Starting coordinates    |
| x2, y2      | int32   | Ending coordinates      |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1)

### POLYLINE (type 2) and STROKE_POLYGON (type 3)
| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| num_points  | varint  | Number of points                       |
| points[]    | int32   | Sequence of (x, y)                     |

Encoded as:
- varint(num_points)
- zigzag(x0), zigzag(y0) – first point absolute
- zigzag(x1 - x0), zigzag(y1 - y0) – delta encoding for subsequent points

### HORIZONTAL_LINE (type 5)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x1, x2      | int32   | X range (start to end)  |
| y           | int32   | Y coordinate            |

Encoded as:
- zigzag(x1), zigzag(x2 - x1), zigzag(y)

### VERTICAL_LINE (type 6)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x           | int32   | X coordinate            |
| y1, y2      | int32   | Y range (start to end)  |

Encoded as:
- zigzag(x), zigzag(y1), zigzag(y2 - y1)

---

## Color Encoding

- `color` is stored as a single byte (`uint8`) in [RGB332 format](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#RGB332).
- If `color == 0xFF`, no color is specified.

---

## Coordinate System

- All coordinates are relative to the tile, with the top-left of the tile as (0,0) and bottom-right as (65535,65535).
- This allows for sub-pixel precision and scalable rendering at different resolutions.

---

## Example Command Encoding

A polyline with 3 points, color 0xC3:
```
type: 2 (POLYLINE)
color: 0xC3
num_points: 3
points: [(100, 200), (150, 250), (180, 300)]
Encoded:
- varint(2)         # type
- uint8(0xC3)       # color
- varint(3)         # num_points
- zigzag(100)       # x0
- zigzag(200)       # y0
- zigzag(150-100)   # x1-x0
- zigzag(250-200)   # y1-y0
- zigzag(180-150)   # x2-x1
- zigzag(300-250)   # y2-y1
```

---

## Notes

- The format is optimized for compactness and fast decoding.
- All commands and parameters are written in the order described above.
- No metadata or geometry types are stored beyond the command types and coordinates.
- Tiles can be concatenated, split, or loaded individually.

---

## Usage

To use this format in your application:
1. Read the file and parse the initial varint (`num_commands`).
2. For each command:
    - Parse `type` and `color`
    - Decode parameters using varint and zigzag as specified above
3. Render the primitives according to your graphics pipeline.

---

## References

- [Protocol Buffers Varint Encoding](https://developers.google.com/protocol-buffers/docs/encoding#varints)
- [Zigzag Encoding](https://developers.google.com/protocol-buffers/docs/encoding#signed-integers)
- [RGB332 Color Format](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#RGB332)

---