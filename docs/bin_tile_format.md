# Tile Binary Format Specification

This document describes the binary format produced by the tile generation script (`tile_generator.py`) for vector map tiles. The format is intended for efficient rendering and compact storage of map data at various zoom levels. Applications and libraries can use this specification to parse and render the resulting `.bin` files.

---

## File Overview

Each file represents a single tile for a given zoom level (`z`), x coordinate (`x`), and y coordinate (`y`).  
The filename and directory structure is:  
```
{output_dir}/{z}/{x}/{y}.bin
```

The file contains a sequence of drawing commands encoded in a compact binary format.  
All coordinates are encoded as `uint16` values (range 0–65535) relative to the tile.

The format uses **state commands** (SET_COLOR, SET_COLOR_INDEX) to eliminate color redundancy and achieve maximum compression through a dynamic palette system, plus **advanced optimization commands** for geometric patterns and feature-specific compression.

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

Commands are separated into **state commands** and **geometry commands**:

### State Commands (Set Current Color)
| Field        | Type      | Description                                 |
|--------------|-----------|---------------------------------------------|
| type         | varint    | State command type (0x80 or 0x81)          |
| color_data   | variable  | Color information (see below)              |

### Geometry Commands (Use Current Color)
| Field        | Type      | Description                                 |
|--------------|-----------|---------------------------------------------|
| type         | varint    | Drawing command type (see command types)   |
| parameters   | variable  | Command-specific data (no color field)     |

---

## Complete Command Types

### Basic Geometry Commands (0x01-0x06)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| LINE                | 0x01  | Single line (from x1,y1 to x2,y2)                    |
| POLYLINE            | 0x02  | Polyline (sequence of points)                        |
| STROKE_POLYGON      | 0x03  | Closed polygon outline (sequence of points)          |
| HORIZONTAL_LINE     | 0x05  | Horizontal line (from x1 to x2 at y)                 |
| VERTICAL_LINE       | 0x06  | Vertical line (from y1 to y2 at x)                   |

### State Commands (0x80-0x81, 0x88)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| SET_COLOR           | 0x80  | Set current color using RGB332 direct value (fallback) |
| SET_COLOR_INDEX     | 0x81  | Set current color using dynamic palette index        |
| SET_LAYER           | 0x88  | Set rendering layer for proper draw order           |

### Feature-Optimized Commands (0x82-0x84)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| RECTANGLE           | 0x82  | Optimized rectangle for buildings                     |
| STRAIGHT_LINE       | 0x83  | Optimized straight line for highways                  |
| HIGHWAY_SEGMENT     | 0x84  | Highway segment with continuity                       |

### Advanced Pattern Commands (0x85-0x8B)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| GRID_PATTERN        | 0x85  | Urban grid pattern                                    |
| BLOCK_PATTERN       | 0x86  | City block pattern                                    |
| CIRCLE              | 0x87  | Circle/roundabout                                     |
| RELATIVE_MOVE       | 0x89  | Relative coordinate movement                          |
| PREDICTED_LINE      | 0x8A  | Predictive line based on pattern                     |
| COMPRESSED_POLYLINE | 0x8B  | Huffman-compressed polyline                          |

### Optimized Geometry Commands (0x8C-0x90)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| OPTIMIZED_POLYGON  | 0x8C  | Optimized polygon (contour only, fill decided by viewer) |
| HOLLOW_POLYGON     | 0x8D  | Polygon outline only (optimized for boundaries)      |
| OPTIMIZED_TRIANGLE | 0x8E  | Optimized triangle (contour only, fill decided by viewer) |
| OPTIMIZED_RECTANGLE| 0x8F  | Optimized rectangle (contour only, fill decided by viewer) |
| OPTIMIZED_CIRCLE   | 0x90  | Optimized circle (contour only, fill decided by viewer) |

### Simple Shape Commands (0x96-0x9A)
| Name                | Value | Description                                           |
|---------------------|-------|------------------------------------------------------|
| SIMPLE_RECTANGLE    | 0x96  | Simple rectangle (x, y, width, height)               |
| SIMPLE_CIRCLE       | 0x97  | Simple circle (center_x, center_y, radius)          |
| SIMPLE_TRIANGLE     | 0x98  | Simple triangle (x1, y1, x2, y2, x3, y3)            |
| DASHED_LINE         | 0x99  | Dashed line with pattern                             |
| DOTTED_LINE         | 0x9A  | Dotted line with pattern                             |

---

## State Command Parameters

### SET_COLOR (type 0x80)
Sets the current color for subsequent geometry commands using direct RGB332 value. Used as fallback when color is not in the dynamic palette.

| Field       | Type    | Description                    |
|-------------|---------|--------------------------------|
| color       | uint8   | RGB332 color value (0-255)    |

Encoded as:
- uint8(color)

### SET_COLOR_INDEX (type 0x81)
Sets the current color using a dynamic palette index. This is the primary method for color assignment.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| color_index | varint  | Index into dynamic color palette (0-N)  |

Encoded as:
- varint(color_index)

### SET_LAYER (type 0x88)
Sets the rendering layer for proper draw order. Commands are grouped by layer and rendered in ascending order (lower layers first).

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| layer       | varint  | Layer number (0-255, lower layers drawn first) |

Encoded as:
- varint(layer)

**Usage**: Establishes rendering order for complex tiles with multiple feature types. Essential for proper visual layering on ESP32.

---

## Basic Geometry Command Parameters

**Important**: Geometry commands do **NOT** include color fields. The current color is set by the most recent SET_COLOR or SET_COLOR_INDEX command.

### LINE (type 0x01)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x1, y1      | int32   | Starting coordinates    |
| x2, y2      | int32   | Ending coordinates      |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1)

### POLYLINE (type 0x02) and STROKE_POLYGON (type 0x03)
| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| num_points  | varint  | Number of points                       |
| points[]    | int32   | Sequence of (x, y)                     |

Encoded as:
- varint(num_points)
- zigzag(x0), zigzag(y0) – first point absolute
- zigzag(x1 - x0), zigzag(y1 - y0) – delta encoding for subsequent points

### HORIZONTAL_LINE (type 0x05)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x1, x2      | int32   | X range (start to end)  |
| y           | int32   | Y coordinate            |

Encoded as:
- zigzag(x1), zigzag(x2 - x1), zigzag(y)

### VERTICAL_LINE (type 0x06)
| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x           | int32   | X coordinate            |
| y1, y2      | int32   | Y range (start to end)  |

Encoded as:
- zigzag(x), zigzag(y1), zigzag(y2 - y1)

---

## Feature-Optimized Command Parameters

### RECTANGLE (type 0x82)
Optimized command for rectangular buildings. Provides significant compression for typical building footprints.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x1, y1      | int32   | Top-left corner                          |
| x2, y2      | int32   | Bottom-right corner                      |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1)

**Optimization**: 60-80% size reduction compared to STROKE_POLYGON for rectangular buildings.

### STRAIGHT_LINE (type 0x83)
Optimized command for straight highway segments. More efficient than POLYLINE for roads without curves.

| Field       | Type    | Description             |
|-------------|---------|-------------------------|
| x1, y1      | int32   | Starting coordinates    |
| x2, y2      | int32   | Ending coordinates      |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1)

**Optimization**: 40-60% reduction for straight roads compared to multi-point POLYLINE.

### HIGHWAY_SEGMENT (type 0x84)
Highway segment with continuity information. Used for connected road networks.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| end_x       | int32   | Ending x coordinate                      |
| end_y       | int32   | Ending y coordinate                      |

Encoded as:
- zigzag(end_x), zigzag(end_y)

**Note**: Start coordinates are assumed from previous segment or current position.
**Optimization**: 30-50% reduction for connected highway networks.

---

## Optimized Geometry Command Parameters

### OPTIMIZED_POLYGON (type 0x8C)
Optimized polygon command for complex shapes. Uses contour-only rendering with fill decision deferred to the viewer.

| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| num_points  | varint  | Number of points                       |
| points[]    | int32   | Sequence of (x, y)                     |

Encoded as:
- varint(num_points)
- zigzag(x0), zigzag(y0) – first point absolute
- zigzag(x1 - x0), zigzag(y1 - y0) – delta encoding for subsequent points

**Optimization**: 30-50% reduction compared to STROKE_POLYGON for complex shapes.

### HOLLOW_POLYGON (type 0x8D)
Specialized polygon outline command optimized for boundary rendering.

| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| num_points  | varint  | Number of points                       |
| points[]    | int32   | Sequence of (x, y)                     |

Encoded as:
- varint(num_points)
- zigzag(x0), zigzag(y0) – first point absolute
- zigzag(x1 - x0), zigzag(y1 - y0) – delta encoding for subsequent points

**Optimization**: Optimized for boundary-only rendering with minimal overhead.

### OPTIMIZED_TRIANGLE (type 0x8E)
Optimized triangle command for triangular features.

| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| x1, y1      | int32   | First vertex                           |
| x2, y2      | int32   | Second vertex                          |
| x3, y3      | int32   | Third vertex                           |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1), zigzag(x3 - x2), zigzag(y3 - y2)

**Optimization**: 40-60% reduction compared to OPTIMIZED_POLYGON for triangular shapes.

### OPTIMIZED_RECTANGLE (type 0x8F)
Optimized rectangle command with enhanced compression.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x1, y1      | int32   | Top-left corner                          |
| x2, y2      | int32   | Bottom-right corner                      |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1)

**Optimization**: Enhanced version of RECTANGLE with improved compression.

### OPTIMIZED_CIRCLE (type 0x90)
Optimized circle command with enhanced precision.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| center_x    | int32   | Circle center x coordinate               |
| center_y    | int32   | Circle center y coordinate               |
| radius      | int32   | Circle radius                            |

Encoded as:
- zigzag(center_x), zigzag(center_y), zigzag(radius)

**Optimization**: Enhanced version of CIRCLE with improved precision and compression.

---

## Simple Shape Command Parameters

### SIMPLE_RECTANGLE (type 0x96)
Simple rectangle command using width/height format.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x, y        | int32   | Top-left corner                          |
| width       | int32   | Rectangle width                          |
| height      | int32   | Rectangle height                         |

Encoded as:
- zigzag(x), zigzag(y), zigzag(width), zigzag(height)

**Usage**: Alternative rectangle format for width/height-based rendering.

### SIMPLE_CIRCLE (type 0x97)
Simple circle command with standard parameters.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| center_x    | int32   | Circle center x coordinate               |
| center_y    | int32   | Circle center y coordinate               |
| radius      | int32   | Circle radius                            |

Encoded as:
- zigzag(center_x), zigzag(center_y), zigzag(radius)

**Usage**: Standard circle format for consistent rendering.

### SIMPLE_TRIANGLE (type 0x98)
Simple triangle command with three vertices.

| Field       | Type    | Description                            |
|-------------|---------|----------------------------------------|
| x1, y1      | int32   | First vertex                           |
| x2, y2      | int32   | Second vertex                          |
| x3, y3      | int32   | Third vertex                           |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2), zigzag(y2), zigzag(x3), zigzag(y3)

**Usage**: Standard triangle format for consistent rendering.

### DASHED_LINE (type 0x99)
Dashed line command with configurable pattern.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x1, y1      | int32   | Starting coordinates                    |
| x2, y2      | int32   | Ending coordinates                       |
| dash_length | int32   | Length of each dash segment              |
| gap_length  | int32   | Length of each gap segment               |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1), zigzag(dash_length), zigzag(gap_length)

**Usage**: Renders dashed lines for roads, boundaries, and other features requiring dashed patterns.

### DOTTED_LINE (type 0x9A)
Dotted line command with configurable pattern.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x1, y1      | int32   | Starting coordinates                    |
| x2, y2      | int32   | Ending coordinates                       |
| dot_size    | int32   | Size of each dot                         |
| gap_length  | int32   | Length of each gap segment               |

Encoded as:
- zigzag(x1), zigzag(y1), zigzag(x2 - x1), zigzag(y2 - y1), zigzag(dot_size), zigzag(gap_length)

**Usage**: Renders dotted lines for trails, boundaries, and other features requiring dotted patterns.

---

### GRID_PATTERN (type 0x85)
Optimized command for urban grid patterns (perpendicular streets). Detects and compresses regular street layouts.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| x, y        | int32   | Grid origin point                        |
| width       | int32   | Grid width                               |
| spacing     | int32   | Line spacing                             |
| count       | varint  | Number of lines                          |
| direction   | uint8   | Direction (1=horizontal, 0=vertical)     |

Encoded as:
- zigzag(x), zigzag(y), zigzag(width), zigzag(spacing), varint(count), uint8(direction)

**Optimization**: 70-85% reduction for regular street grids compared to individual LINE commands.

### BLOCK_PATTERN (type 0x86)
City block pattern command for rectangular urban layouts.

| Field         | Type    | Description                              |
|---------------|---------|------------------------------------------|
| x, y          | int32   | Pattern origin                           |
| block_width   | int32   | Width of each block                      |
| block_height  | int32   | Height of each block                     |
| rows          | varint  | Number of block rows                     |
| cols          | varint  | Number of block columns                  |

Encoded as:
- zigzag(x), zigzag(y), zigzag(block_width), zigzag(block_height), varint(rows), varint(cols)

**Optimization**: 60-80% reduction for regular city block layouts.

### CIRCLE (type 0x87)
Optimized command for circular features like roundabouts, plazas, and circular buildings.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| center_x    | int32   | Circle center x coordinate               |
| center_y    | int32   | Circle center y coordinate               |
| radius      | int32   | Circle radius                            |

Encoded as:
- zigzag(center_x), zigzag(center_y), zigzag(radius)

**Optimization**: 50-70% reduction for circular polygons compared to STROKE_POLYGON.

### RELATIVE_MOVE (type 0x88)
Sets relative position for subsequent coordinate commands. Improves coordinate compression.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| dx          | int32   | X offset from current position           |
| dy          | int32   | Y offset from current position           |

Encoded as:
- zigzag(dx), zigzag(dy)

**Usage**: Establishes new coordinate reference point for delta encoding optimization.

### PREDICTED_LINE (type 0x89)
Line command using coordinate prediction based on movement patterns. Start point is predicted from previous commands.

| Field       | Type    | Description                              |
|-------------|---------|------------------------------------------|
| end_x       | int32   | Ending x coordinate                      |
| end_y       | int32   | Ending y coordinate                      |

Encoded as:
- zigzag(end_x), zigzag(end_y)

**Optimization**: 30-50% reduction for predictable paths like continuous roads.

### COMPRESSED_POLYLINE (type 0x8B)
Polyline with Huffman-compressed coordinates. Used for complex paths with repetitive patterns.

| Field           | Type    | Description                              |
|-----------------|---------|------------------------------------------|
| num_points      | varint  | Number of points                         |
| compressed_data | variable| Huffman-encoded coordinate deltas        |

Encoded as:
- varint(num_points), variable_length_huffman_data

**Optimization**: 20-40% reduction for polylines with repetitive coordinate patterns.

---

## Color Encoding

### RGB332 Format
- `color` is stored as a single byte (`uint8`) in [RGB332 format](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#RGB332).
- **Bit layout**: `RRRGGGBB` (3 bits red, 3 bits green, 2 bits blue)

### Dynamic Palette
- **Automatic generation**: Palette is built from unique colors in `features.json`
- **Index mapping**: Each hex color (e.g., `#ff0000`) receives a unique index (0-N)
- **Alphabetical ordering**: Colors are sorted alphabetically for consistency
- **Efficient encoding**: Indices use varint encoding (smaller for low indices)
- **Primary method**: Most colors use SET_COLOR_INDEX for optimal compression

---

## Coordinate System

- All coordinates are relative to the tile, with the top-left of the tile as (0,0) and bottom-right as (65535,65535).
- This allows for sub-pixel precision and scalable rendering at different resolutions.
- **Delta encoding**: Most commands use coordinate differences for better compression.
- **Zigzag encoding**: Signed coordinate differences are encoded efficiently.

---

## Example Command Encodings

### Basic Geometry with Palette
```
# Set color using palette index
SET_COLOR_INDEX 3
- varint(0x81), varint(3)

# Draw optimized rectangle (building)
RECTANGLE (100,100) to (200,150)
- varint(0x82), zigzag(100), zigzag(100), zigzag(100), zigzag(50)

# Draw straight highway
STRAIGHT_LINE (300,200) to (400,200)
- varint(0x83), zigzag(300), zigzag(200), zigzag(100), zigzag(0)
```

### Advanced Pattern Commands
```
# Urban grid pattern
GRID_PATTERN at (0,0), width=1000, spacing=50, 10 lines, horizontal
- varint(0x85), zigzag(0), zigzag(0), zigzag(1000), zigzag(50), varint(10), uint8(1)

# Circular roundabout
CIRCLE center=(500,500), radius=30
- varint(0x87), zigzag(500), zigzag(500), zigzag(30)

# Predicted line continuation
PREDICTED_LINE end=(600,250)
- varint(0x89), zigzag(600), zigzag(250)
```

### Complete Tile Example with Layer Ordering
```
# Tile header
varint(12)  # 12 commands total

# Layer 0: Background features (water, natural areas)
SET_LAYER 0
- varint(0x88), varint(0)

SET_COLOR_INDEX 2
- varint(0x81), varint(2)

# Water polygon (optimized)
OPTIMIZED_POLYGON 4 points: (0,0), (100,0), (100,50), (0,50)
- varint(0x8C), varint(4), zigzag(0), zigzag(0), zigzag(100), zigzag(0), zigzag(0), zigzag(50), zigzag(-100), zigzag(0)

# Layer 1: Roads and infrastructure
SET_LAYER 1
- varint(0x88), varint(1)

SET_COLOR_INDEX 5
- varint(0x81), varint(5)

# Main highway (straight line)
STRAIGHT_LINE (200,100) to (400,100)
- varint(0x83), zigzag(200), zigzag(100), zigzag(200), zigzag(0)

# Dashed boundary line
DASHED_LINE (300,150) to (500,150), dash=10, gap=5
- varint(0x99), zigzag(300), zigzag(150), zigzag(200), zigzag(0), zigzag(10), zigzag(5)

# Layer 2: Buildings and structures
SET_LAYER 2
- varint(0x88), varint(2)

SET_COLOR_INDEX 1
- varint(0x81), varint(1)

# Multiple buildings as optimized rectangles
OPTIMIZED_RECTANGLE (150,200) to (200,250)
- varint(0x8F), zigzag(150), zigzag(200), zigzag(50), zigzag(50)

# Circular building
OPTIMIZED_CIRCLE center=(300,225), radius=25
- varint(0x90), zigzag(300), zigzag(225), zigzag(25)

# Simple triangle building
SIMPLE_TRIANGLE (400,200), (450,200), (425,250)
- varint(0x98), zigzag(400), zigzag(200), zigzag(450), zigzag(200), zigzag(425), zigzag(250)
```

---

## Optimization Benefits

### Feature-Specific Optimizations
- **Buildings**: RECTANGLE and OPTIMIZED_RECTANGLE commands reduce polygon data by 60-80%
- **Highways**: STRAIGHT_LINE eliminates unnecessary polyline points
- **Connected roads**: HIGHWAY_SEGMENT uses coordinate continuity
- **Complex polygons**: OPTIMIZED_POLYGON provides 30-50% reduction for complex shapes
- **Triangular features**: OPTIMIZED_TRIANGLE offers 40-60% reduction for triangular shapes
- **Circular features**: OPTIMIZED_CIRCLE provides enhanced precision and compression

### Advanced Pattern Recognition
- **Urban grids**: GRID_PATTERN compresses regular street layouts by 70-85%
- **Circles**: CIRCLE and OPTIMIZED_CIRCLE commands compress roundabouts by 50-70%
- **Predictions**: PREDICTED_LINE reduces coordinate data by 30-50%
- **Pattern lines**: DASHED_LINE and DOTTED_LINE provide efficient pattern rendering

### Layer-Based Rendering
- **Proper ordering**: SET_LAYER ensures correct visual layering
- **Grouped commands**: Commands are grouped by layer for efficient rendering
- **ESP32 optimization**: Essential for proper draw order on embedded systems
- **Memory efficiency**: Layer grouping reduces state changes during rendering

### Dynamic Palette System
- **Primary optimization**: Replaces RGB332 values with compact indices
- **Automatic generation**: Palette built from `features.json` configuration
- **Maximum compression**: Frequently used colors get low indices (smaller varint)
- **Benefit**: 25-40% file size reduction compared to embedded colors

### State Command Architecture
- Eliminates redundant color fields in geometry commands
- One color command can apply to multiple geometry commands
- **Benefit**: Additional 15-25% file size reduction in dense tiles

---

## Reader Implementation

### Basic Parser Structure
```c
uint32_t current_color = 0xFF; // Default color
coordinate_t current_position = {0, 0}; // For relative commands

while (commands_remaining > 0) {
    uint32_t command_type = read_varint();
    
    switch (command_type) {
        // State commands
        case 0x80: // SET_COLOR
            current_color = read_uint8();
            break;
            
        case 0x81: // SET_COLOR_INDEX
            uint32_t index = read_varint();
            current_color = palette[index];
            break;
            
        case 0x88: // SET_LAYER
            current_layer = read_varint();
            break;
        
        // Basic geometry
        case 0x01: case 0x02: case 0x03: case 0x05: case 0x06:
            render_basic_geometry(command_type, current_color);
            break;
            
        // Feature-optimized commands
        case 0x82: // RECTANGLE
            render_rectangle(current_color);
            break;
            
        case 0x83: // STRAIGHT_LINE
            render_straight_line(current_color);
            break;
            
        case 0x84: // HIGHWAY_SEGMENT
            render_highway_segment(current_color, &current_position);
            break;
            
        // Advanced pattern commands
        case 0x85: // GRID_PATTERN
            render_grid_pattern(current_color);
            break;
            
        case 0x86: // BLOCK_PATTERN
            render_block_pattern(current_color);
            break;
            
        case 0x87: // CIRCLE
            render_circle(current_color);
            break;
            
        case 0x89: // RELATIVE_MOVE
            update_relative_position(&current_position);
            break;
            
        case 0x8A: // PREDICTED_LINE
            render_predicted_line(current_color, &current_position);
            break;
            
        case 0x8B: // COMPRESSED_POLYLINE
            render_compressed_polyline(current_color);
            break;
            
        // Optimized geometry commands
        case 0x8C: // OPTIMIZED_POLYGON
            render_optimized_polygon(current_color);
            break;
            
        case 0x8D: // HOLLOW_POLYGON
            render_hollow_polygon(current_color);
            break;
            
        case 0x8E: // OPTIMIZED_TRIANGLE
            render_optimized_triangle(current_color);
            break;
            
        case 0x8F: // OPTIMIZED_RECTANGLE
            render_optimized_rectangle(current_color);
            break;
            
        case 0x90: // OPTIMIZED_CIRCLE
            render_optimized_circle(current_color);
            break;
            
        // Simple shape commands
        case 0x96: // SIMPLE_RECTANGLE
            render_simple_rectangle(current_color);
            break;
            
        case 0x97: // SIMPLE_CIRCLE
            render_simple_circle(current_color);
            break;
            
        case 0x98: // SIMPLE_TRIANGLE
            render_simple_triangle(current_color);
            break;
            
        case 0x99: // DASHED_LINE
            render_dashed_line(current_color);
            break;
            
        case 0x9A: // DOTTED_LINE
            render_dotted_line(current_color);
            break;
            
        default:
            // Unknown command, skip safely
            skip_unknown_command(command_type);
            break;
    }
    commands_remaining--;
}
```

### Advanced Pattern Rendering
```c
void render_grid_pattern(uint32_t color) {
    int32_t x = read_zigzag();
    int32_t y = read_zigzag();
    int32_t width = read_zigzag();
    int32_t spacing = read_zigzag();
    uint32_t count = read_varint();
    uint8_t direction = read_uint8();
    
    // Render regular grid lines
    for (uint32_t i = 0; i < count; i++) {
        if (direction == 1) { // Horizontal
            draw_line(x, y + i * spacing, x + width, y + i * spacing, color);
        } else { // Vertical
            draw_line(x + i * spacing, y, x + i * spacing, y + width, color);
        }
    }
}

void render_circle(uint32_t color) {
    int32_t center_x = read_zigzag();
    int32_t center_y = read_zigzag();
    int32_t radius = read_zigzag();
    
    // Render circle outline
    draw_circle_outline(center_x, center_y, radius, color);
}
```

---

## Performance Considerations

### File Size Reduction
- **Feature optimizations**: 40-80% reduction for specific geometry types
- **Pattern recognition**: 70-85% reduction for regular urban layouts
- **Dynamic palette**: 25-40% reduction compared to embedded colors
- **State commands**: Additional 15-25% reduction in dense tiles
- **Total improvement**: Up to 85% smaller than unoptimized format

### Rendering Performance
- **Fewer state changes**: Grouped commands reduce GPU state changes
- **Optimized primitives**: Native circle and rectangle rendering
- **Pattern efficiency**: Single command renders multiple elements
- **Coordinate prediction**: Reduces coordinate parsing overhead
- **Cache efficiency**: Sequential commands improve cache hits

### Memory Usage
- **Palette storage**: Minimal overhead (typically <200 bytes)
- **Parser state**: Few variables (color, position)
- **Pattern expansion**: Commands expand to multiple primitives efficiently
- **Prediction buffers**: Small coordinate history for pattern detection

---

## Compatibility and Forward Compatibility

### Version Compatibility
- **Basic parsers**: Can safely ignore commands ≥ 0x82
- **Advanced parsers**: Support all optimization commands (0x82-0x9A)
- **Fallback rendering**: Unknown commands can be skipped
- **State preservation**: Color and layer state persists across unknown commands

### Implementation Guidelines
- Always implement basic commands (0x01-0x06, 0x80-0x81, 0x88)
- Feature-optimized commands (0x82-0x84) are recommended for buildings and roads
- Advanced pattern commands (0x85-0x8B) provide significant compression benefits
- Optimized geometry commands (0x8C-0x90) offer enhanced rendering efficiency
- Simple shape commands (0x96-0x9A) provide alternative rendering formats
- Unknown command types should be skipped gracefully
- Maintain color, position, and layer state across all commands

---

## Usage

### Basic Implementation Requirements
1. Read the file and parse the initial varint (`num_commands`)
2. Initialize `current_color = 0xFF`, `current_position = {0, 0}`, `current_layer = 0`
3. Load dynamic palette from `features.json`
4. Implement state command handlers (SET_COLOR, SET_COLOR_INDEX, SET_LAYER)
5. Implement basic geometry commands (LINE, POLYLINE, STROKE_POLYGON, etc.)

### Advanced Implementation (Recommended)
1. Add feature-optimized command handlers (RECTANGLE, STRAIGHT_LINE, HIGHWAY_SEGMENT)
2. Implement pattern commands (GRID_PATTERN, CIRCLE, BLOCK_PATTERN)
3. Add coordinate prediction support (PREDICTED_LINE, RELATIVE_MOVE)
4. Implement optimized geometry commands (OPTIMIZED_POLYGON, OPTIMIZED_TRIANGLE, etc.)
5. Add simple shape commands (SIMPLE_RECTANGLE, SIMPLE_CIRCLE, SIMPLE_TRIANGLE)
6. Implement pattern line commands (DASHED_LINE, DOTTED_LINE)
7. Optimize rendering pipeline for grouped commands and layer ordering
8. Cache pattern expansions for repeated tiles

### Error Handling
- Invalid command types should be skipped gracefully
- Out-of-bounds palette indices should use default color (0xFF)
- Missing palette should fall back to direct RGB332 interpretation
- Corrupted files should fail safely without crashing
- Coordinate overflow should clamp to tile boundaries (0-65535)

---

## References

- [Protocol Buffers Varint Encoding](https://developers.google.com/protocol-buffers/docs/encoding#varints)
- [Zigzag Encoding](https://developers.google.com/protocol-buffers/docs/encoding#signed-integers)
- [RGB332 Color Format](https://en.wikipedia.org/wiki/List_of_monochrome_and_RGB_palettes#RGB332)

---