import struct
import sys
import os
import math
import pygame
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

TILE_SIZE = 256
VIEWPORT_SIZE = 768

TOOLBAR_WIDTH = 160
STATUSBAR_HEIGHT = 40
WINDOW_WIDTH = VIEWPORT_SIZE + TOOLBAR_WIDTH
WINDOW_HEIGHT = VIEWPORT_SIZE + STATUSBAR_HEIGHT

DRAW_COMMANDS = {
    1: "LINE",
    2: "POLYLINE",
    3: "STROKE_POLYGON",
    5: "HORIZONTAL_LINE",
    6: "VERTICAL_LINE",
}

UINT16_TILE_SIZE = 65536

def rgb565_to_rgb888(c):
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    return (
        int((r * 255) / 31),
        int((g * 255) / 63),
        int((b * 255) / 31)
    )

def lighten_color(rgb, amount=0.12):
    return tuple(
        min(255, int(v + (255 - v) * amount))
        for v in rgb
    )

def darken_color(rgb, amount=0.3):
    return tuple(max(0, int(v * (1 - amount))) for v in rgb)

def uint16_to_tile_pixel(val):
    return int(round(val * (TILE_SIZE - 1) / (UINT16_TILE_SIZE - 1)))

def get_button_icons():
    icon_surface_bg = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_bg.fill((0, 0, 0, 0))
    pygame.draw.circle(icon_surface_bg, (255, 255, 255), (12, 12), 10, 2)
    pygame.draw.circle(icon_surface_bg, (0, 0, 0), (12, 12), 7, 0)
    icon_surface_label = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_label.fill((0, 0, 0, 0))
    pygame.draw.rect(icon_surface_label, (255,255,255), (4, 6, 16, 12), 2)
    pygame.draw.line(icon_surface_label, (255,255,255), (6, 10), (18, 10), 2)
    pygame.draw.line(icon_surface_label, (255,255,255), (6, 14), (14, 14), 2)
    icon_surface_gps = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_gps.fill((0, 0, 0, 0))
    pygame.draw.circle(icon_surface_gps, (255,255,255), (12, 12), 10, 2)
    pygame.draw.line(icon_surface_gps, (255,255,255), (12, 5), (12, 19), 2)
    pygame.draw.line(icon_surface_gps, (255,255,255), (5, 12), (19, 12), 2)
    pygame.draw.circle(icon_surface_gps, (255,255,255), (12,12), 3, 0)
    icon_surface_fill = pygame.Surface((24, 24), pygame.SRCALPHA)
    icon_surface_fill.fill((0, 0, 0, 0))
    pygame.draw.polygon(icon_surface_fill, (255,255,255), [(4,18),(12,4),(20,18)], 0)
    pygame.draw.polygon(icon_surface_fill, (0,0,0), [(4,18),(12,4),(20,18)], 2)
    return icon_surface_bg, icon_surface_label, icon_surface_gps, icon_surface_fill

def index_available_tiles(directory, progress_callback=None):
    available_tiles = set()
    if not os.path.isdir(directory):
        print(f"Directory does not exist: {directory}")
        return available_tiles
    x_dirs = [x_str for x_str in os.listdir(directory) if os.path.isdir(os.path.join(directory, x_str))]
    total_x = len(x_dirs)
    def index_xdir(x_str):
        x_path = os.path.join(directory, x_str)
        try:
            x = int(x_str)
        except:
            return []
        files = os.listdir(x_path)
        y_dict = {}
        for fname in files:
            if fname.endswith('.bin') or fname.endswith('.png'):
                y_str = fname.split('.')[0]
                if y_str.isdigit():
                    y = int(y_str)
                    if y not in y_dict or fname.endswith('.bin'):
                        y_dict[y] = fname
        return [(x, y) for y in y_dict]
    results = []
    with ThreadPoolExecutor(min(8, os.cpu_count() or 4)) as pool:
        futures = {pool.submit(index_xdir, x_str): i_x for i_x, x_str in enumerate(x_dirs)}
        for i, future in enumerate(as_completed(futures)):
            tiles = future.result()
            results.extend(tiles)
            if progress_callback is not None:
                percent = (i + 1) / max(total_x, 1)
                progress_callback(percent, "Indexing tiles...")
    available_tiles.update(results)
    return available_tiles

def get_tile_file(directory, x, y):
    bin_path = f"{directory}/{x}/{y}.bin"
    png_path = f"{directory}/{x}/{y}.png"
    if os.path.isfile(bin_path):
        return bin_path
    elif os.path.isfile(png_path):
        return png_path
    return None

def read_varint(data, offset):
    result = 0
    shift = 0
    while True:
        b = data[offset]
        offset += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, offset

def read_zigzag(data, offset):
    v, offset = read_varint(data, offset)
    return (v >> 1) ^ -(v & 1), offset

def rgb332_to_rgb888(c):
    r = (c & 0xE0)
    g = (c & 0x1C) << 3
    b = (c & 0x03) << 6
    return (r, g, b)

def uint16_to_tile_pixel(x):
    return int((x / 65535) * (TILE_SIZE - 1))

def is_tile_border_point(pt):
    x, y = pt
    return (x == 0 or x == TILE_SIZE-1) or (y == 0 or y == TILE_SIZE-1)

def render_tile_surface(tile, bg_color, fill_mode):
    surface = pygame.Surface((TILE_SIZE, TILE_SIZE))
    surface.fill(bg_color)
    filepath = tile['file']
    if filepath.endswith('.png'):
        try:
            img = pygame.image.load(filepath)
            img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
            surface.blit(img, (0, 0))
        except Exception as e:
            print(f"Error loading PNG {filepath}: {e}")
        return surface

    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return surface

    if len(data) < 1:
        return surface

    offset = 0
    num_cmds, offset = read_varint(data, offset)
    for _ in range(num_cmds):
        if offset >= len(data):
            break
        try:
            cmd_type, offset = read_varint(data, offset)
            color = data[offset]
            offset += 1
            rgb = rgb332_to_rgb888(color)
            if cmd_type == 1:  # LINE
                x1, offset = read_zigzag(data, offset)
                y1, offset = read_zigzag(data, offset)
                dx, offset = read_zigzag(data, offset)
                dy, offset = read_zigzag(data, offset)
                x2 = x1 + dx
                y2 = y1 + dy
                pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y1)),
                                 (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y2)), 1)
            elif cmd_type == 2:  # POLYLINE
                n_pts, offset = read_varint(data, offset)
                pts = []
                for i in range(n_pts):
                    if i == 0:
                        x, offset = read_zigzag(data, offset)
                        y, offset = read_zigzag(data, offset)
                    else:
                        dx, offset = read_zigzag(data, offset)
                        dy, offset = read_zigzag(data, offset)
                        x += dx
                        y += dy
                    pts.append((uint16_to_tile_pixel(x), uint16_to_tile_pixel(y)))
                if len(pts) >= 2:
                    pygame.draw.lines(surface, rgb, False, pts, 1)
            elif cmd_type == 3:  # STROKE_POLYGON
                n_pts, offset = read_varint(data, offset)
                pts = []
                for i in range(n_pts):
                    if i == 0:
                        x, offset = read_zigzag(data, offset)
                        y, offset = read_zigzag(data, offset)
                    else:
                        dx, offset = read_zigzag(data, offset)
                        dy, offset = read_zigzag(data, offset)
                        x += dx
                        y += dy
                    pts.append((uint16_to_tile_pixel(x), uint16_to_tile_pixel(y)))
                if fill_mode and len(pts) >= 3:
    # Dibuja primero el relleno con el MISMO color que el borde
                    pygame.draw.polygon(surface, rgb, pts, 0)
                    closed = pts[0] == pts[-1]
                    for i in range(len(pts)-1):
                        p1 = pts[i]
                        p2 = pts[i+1]
                        if not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2):
                            # pygame.draw.line(surface, bg_color, p1, p2, 1)
                            continue
                        else:
                            border_rgb = darken_color(rgb)
                            pygame.draw.line(surface, border_rgb, p1, p2, 2)
                    if closed:
                        p1 = pts[-1]
                        p2 = pts[0]
                        if not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2):
                            # pygame.draw.line(surface, bg_color, p1, p2, 1)
                            continue
                        else:
                            border_rgb = darken_color(rgb)
                            pygame.draw.line(surface, border_rgb, p1, p2, 2)
                if len(pts) >= 2:
                    closed = pts[0] == pts[-1]
                    for i in range(len(pts)-1):
                        p1 = pts[i]
                        p2 = pts[i+1]
                        if not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2):
                            # pygame.draw.line(surface, bg_color, p1, p2, 1)
                            continue
                        else:
                            pygame.draw.line(surface, rgb, p1, p2, 1)
                    if closed:
                        p1 = pts[-1]
                        p2 = pts[0]
                        if not fill_mode and is_tile_border_point(p1) and is_tile_border_point(p2):
                            # pygame.draw.line(surface, bg_color, p1, p2, 1)
                            continue
                        else:
                            pygame.draw.line(surface, rgb, p1, p2, 1)
            elif cmd_type == 5:  # HORIZONTAL_LINE
                x1, offset = read_zigzag(data, offset)
                dx, offset = read_zigzag(data, offset)
                y, offset = read_zigzag(data, offset)
                x2 = x1 + dx
                pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x1), uint16_to_tile_pixel(y)),
                                 (uint16_to_tile_pixel(x2), uint16_to_tile_pixel(y)), 1)
            elif cmd_type == 6:  # VERTICAL_LINE
                x, offset = read_zigzag(data, offset)
                y1, offset = read_zigzag(data, offset)
                dy, offset = read_zigzag(data, offset)
                y2 = y1 + dy
                pygame.draw.line(surface, rgb, (uint16_to_tile_pixel(x), uint16_to_tile_pixel(y1)),
                                 (uint16_to_tile_pixel(x), uint16_to_tile_pixel(y2)), 1)
        except Exception as e:
            print(f"Error reading command in {filepath}: {e}")
            break

    return surface

def center_viewport_on_central_tile(available_tiles):
    if not available_tiles:
        return 0, 0
    xs = [x for x, y in available_tiles]
    ys = [y for x, y in available_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    viewport_x = center_x * TILE_SIZE - VIEWPORT_SIZE // 2
    viewport_y = center_y * TILE_SIZE - VIEWPORT_SIZE // 2
    return viewport_x, viewport_y

def clamp_viewport(viewport_x, viewport_y, available_tiles):
    if not available_tiles:
        return viewport_x, viewport_y
    xs = [x for x, y in available_tiles]
    ys = [y for x, y in available_tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    viewport_x = max(min_x * TILE_SIZE, min(viewport_x, (max_x * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE))
    viewport_y = max(min_y * TILE_SIZE, min(viewport_y, (max_y * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE))
    return viewport_x, viewport_y

def draw_button(surface, text, rect, bg_color, fg_color, border_color, font, icon=None, pressed=False):
    radius = 16
    pygame.draw.rect(surface, bg_color, rect, border_radius=radius)
    pygame.draw.rect(surface, border_color, rect, 2, border_radius=radius)
    if pressed:
        pygame.draw.rect(surface, border_color, rect, 4, border_radius=radius)
    content_x = rect.left + 12
    content_y = rect.centery
    if icon is not None:
        icon_rect = icon.get_rect()
        icon_rect.centery = rect.centery
        icon_rect.left = rect.left + 12
        surface.blit(icon, icon_rect)
        content_x = icon_rect.right + 8
    max_text_width = rect.width - (content_x - rect.left) - 8
    font_size = font.get_height()
    label = font.render(text, True, fg_color)
    while label.get_width() > max_text_width and font_size > 10:
        font_size -= 1
        font = pygame.font.SysFont(None, font_size)
        label = font.render(text, True, fg_color)
    text_rect = label.get_rect(midleft=(content_x, rect.centery))
    surface.blit(label, text_rect)

def show_status_progress_bar(surface, percent, text, font):
    bar_max_width = WINDOW_WIDTH // 3
    bar_height = 18
    bar_margin_right = 24
    bar_x = WINDOW_WIDTH - bar_max_width - bar_margin_right
    bar_y = VIEWPORT_SIZE + STATUSBAR_HEIGHT // 2 - bar_height // 2
    pygame.draw.rect(surface, (80, 80, 80), (bar_x, bar_y, bar_max_width, bar_height))
    pygame.draw.rect(surface, (30, 160, 220), (bar_x, bar_y, int(bar_max_width * percent), bar_height))
    pygame.draw.rect(surface, (120, 120, 120), (bar_x, bar_y, bar_max_width, bar_height), 2)
    label = font.render(text, True, (255,255,255))
    label_rect = label.get_rect(midleft=(bar_x + 8, bar_y + bar_height//2 - label.get_height()//2))
    surface.blit(label, label_rect)

def draw_tile_labels(
    screen, font, available_tiles, viewport_x, viewport_y, zoom_level, background_color, show_tile_labels, directory
):
    if not show_tile_labels:
        return
    fg = (0, 0, 0) if background_color == (255,255,255) else (255,255,255)
    label_bg = (240,240,240) if background_color == (255,255,255) else (32,32,32)
    border = (180,180,180) if background_color == (255,255,255) else (64,64,64)
    outline = (120,120,120) if background_color == (255,255,255) else (220,220,220)
    for x, y in available_tiles:
        px = x * TILE_SIZE - viewport_x
        py = y * TILE_SIZE - viewport_y
        if px + TILE_SIZE < 0 or px > VIEWPORT_SIZE or py + TILE_SIZE < 0 or py > VIEWPORT_SIZE:
            continue
        filename = None
        if os.path.isfile(f"{directory}/{x}/{y}.bin"):
            filename = f"{y}.bin"
        elif os.path.isfile(f"{directory}/{x}/{y}.png"):
            filename = f"{y}.png"
        else:
            filename = f"{y}"
        txt = f"x={x} y={y} z={zoom_level} {filename}"
        label_surfs = [font.render(txt, True, fg)]
        lw = max(s.get_width() for s in label_surfs)
        lh = sum(s.get_height() for s in label_surfs)
        margin = 2
        label_rect = pygame.Rect(
            px + margin, py + margin,
            lw + margin * 2, lh + margin * 2
        )
        pygame.draw.rect(screen, label_bg, label_rect)
        pygame.draw.rect(screen, border, label_rect, 1)
        offset_y = label_rect.top + margin
        for surf in label_surfs:
            screen.blit(surf, (label_rect.left + margin, offset_y))
            offset_y += surf.get_height()
        draw_dashed_rect(screen, pygame.Rect(px, py, TILE_SIZE, TILE_SIZE), outline, width=1)

def draw_dashed_rect(surface, rect, color, dash_length=6, gap_length=4, width=1):
    x = rect.left
    while x < rect.right:
        end_x = min(x + dash_length, rect.right)
        pygame.draw.line(surface, color, (x, rect.top), (end_x, rect.top), width)
        x += dash_length + gap_length
    x = rect.left
    while x < rect.right:
        end_x = min(x + dash_length, rect.right)
        pygame.draw.line(surface, color, (x, rect.bottom-1), (end_x, rect.bottom-1), width)
        x += dash_length + gap_length
    y = rect.top
    while y < rect.bottom:
        end_y = min(y + dash_length, rect.bottom)
        pygame.draw.line(surface, color, (rect.left, y), (rect.left, end_y), width)
        y += dash_length + gap_length
    y = rect.top
    while y < rect.bottom:
        end_y = min(y + dash_length, rect.bottom)
        pygame.draw.line(surface, color, (rect.right-1, y), (rect.right-1, end_y), width)
        y += dash_length + gap_length

def tile_xy_to_latlon(x, y, z):
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def pixel_to_latlon(px, py, viewport_x, viewport_y, zoom):
    n = 2.0 ** zoom
    map_px = viewport_x + px
    map_py = viewport_y + py
    tile_x = map_px / TILE_SIZE
    tile_y = map_py / TILE_SIZE
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def latlon_to_pixel(lat, lon, zoom):
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    map_px = x * TILE_SIZE
    map_py = y * TILE_SIZE
    return map_px, map_py

def decimal_to_gms(decimal, is_latitude=True):
    sign = ""
    if is_latitude:
        sign = "N" if decimal >= 0 else "S"
    else:
        sign = "E" if decimal >= 0 else "W"
    decimal = abs(decimal)
    degrees = int(decimal)
    minutes_full = (decimal - degrees) * 60
    minutes = int(minutes_full)
    seconds = (minutes_full - minutes) * 60
    return f"{degrees}°{minutes}'{seconds:.2f}\" {sign}"

def main(base_dir):
    zoom_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    zoom_levels_list = sorted([int(d) for d in zoom_dirs])
    if not zoom_levels_list:
        print(f"No zoom level directories found in {base_dir}")
        sys.exit(1)
    min_zoom = zoom_levels_list[0]
    max_zoom = zoom_levels_list[-1]
    zoom_levels = list(range(min_zoom, max_zoom+1))
    zoom_idx = 0

    background_color = (0, 0, 0)
    button_color = (0, 0, 0)
    button_fg = (255, 255, 255)
    button_border = (100,100,100)

    toolbar_x = VIEWPORT_SIZE
    toolbar_y = 0
    button_height = 40
    button_margin = 16
    button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin, 100, button_height)
    tile_label_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 2 + button_height, 100, button_height)
    gps_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 3 + button_height * 2, 100, button_height)
    fill_button_rect = pygame.Rect(toolbar_x + 30, toolbar_y + button_margin * 4 + button_height * 3, 100, button_height)

    button_text_black = "Black"
    button_text_white = "White"
    button_pressed = False

    show_tile_labels = False
    show_gps_tooltip = False
    fill_polygons_mode = False

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Map: {base_dir}")
    font = pygame.font.SysFont(None, 16)
    font_main = pygame.font.SysFont(None, 18)
    font_b = pygame.font.SysFont(None, 16)
    font_status = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()

    available_tiles = set()
    tile_surfaces = {}

    icon_bg, icon_label, icon_gps, icon_fill = get_button_icons()

    mouse_gps_coords = None
    mouse_gps_rect = None

    show_index_progress = False
    index_progress_percent = 0.0
    index_progress_text = ""
    index_progress_done_drawn = False
    show_render_progress = False
    render_progress_percent = 0.0
    render_progress_text = ""
    render_progress_done_drawn = False

    tiles_loading = False
    tiles_loading_lock = threading.Lock()
    need_redraw = True
    zoom_change_pending = False
    zoom_change_params = None

    def status_index_progress_callback(percent, text):
        nonlocal show_index_progress, index_progress_percent, index_progress_text, need_redraw
        show_index_progress = True
        index_progress_percent = percent
        index_progress_text = text
        need_redraw = True

    def hide_index_progress():
        nonlocal show_index_progress, index_progress_percent, index_progress_text, need_redraw, index_progress_done_drawn
        if index_progress_done_drawn:
            show_index_progress = False
            index_progress_percent = 0.0
            index_progress_text = ""
            index_progress_done_drawn = False
            need_redraw = True

    def status_render_progress_callback(percent, text):
        nonlocal show_render_progress, render_progress_percent, render_progress_text, need_redraw
        show_render_progress = True
        render_progress_percent = percent
        render_progress_text = text
        need_redraw = True

    def hide_render_progress():
        nonlocal show_render_progress, render_progress_percent, render_progress_text, need_redraw, render_progress_done_drawn
        if render_progress_done_drawn:
            show_render_progress = False
            render_progress_percent = 0.0
            render_progress_text = ""
            render_progress_done_drawn = False
            need_redraw = True

    def load_available_tiles(level, progress_callback=None):
        directory = os.path.join(base_dir, str(level))
        available = index_available_tiles(directory, progress_callback)
        return available, directory

    def get_tile_surface(x, y, zoom_level, directory, bg_color, fill_mode):
        key = (zoom_level, x, y, bg_color, fill_mode)
        if (x, y) not in available_tiles:
            return None
        if key not in tile_surfaces:
            tile_file = get_tile_file(directory, x, y)
            if not tile_file:
                return None
            tile_surfaces[key] = render_tile_surface({'x': x, 'y': y, 'file': tile_file}, bg_color, fill_mode)
        return tile_surfaces[key]

    def preload_tile_surfaces_threaded(tile_list, zoom_level, directory, bg_color, fill_mode, progress_callback=None, done_callback=None):
        def loader():
            total = len(tile_list)
            def load_single(tile):
                x, y = tile
                tile_file = get_tile_file(directory, x, y)
                if not tile_file:
                    return None, None
                surface = render_tile_surface({'x': x, 'y': y, 'file': tile_file}, bg_color, fill_mode)
                return (zoom_level, x, y, bg_color, fill_mode), surface
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(load_single, tile): tile for tile in tile_list}
                for i, future in enumerate(as_completed(futures)):
                    key, surface = future.result()
                    if key and surface:
                        tile_surfaces[key] = surface
                    if progress_callback is not None:
                        percent = (i + 1) / max(total, 1)
                        progress_callback(percent, "Loading visible tiles...")
            if done_callback:
                done_callback()
        t = threading.Thread(target=loader)
        t.start()
        return t

    def set_tiles_loading(flag):
        nonlocal tiles_loading, need_redraw
        with tiles_loading_lock:
            tiles_loading = flag
            need_redraw = True

    def start_zoom_change(idx, last_mouse_pos, viewport_x, viewport_y, old_zoom_idx):
        nonlocal zoom_change_pending, zoom_change_params
        zoom_change_pending = True
        zoom_change_params = (idx, last_mouse_pos, viewport_x, viewport_y, old_zoom_idx)

    show_index_progress = True
    index_progress_percent = 0.0
    index_progress_text = "Indexing initial tiles..."
    available_tiles, directory = load_available_tiles(zoom_levels[zoom_idx], status_index_progress_callback)
    hide_index_progress()
    viewport_x, viewport_y = center_viewport_on_central_tile(available_tiles)

    dragging = False
    drag_start = None
    running = True
    last_mouse_pos = (VIEWPORT_SIZE // 2, VIEWPORT_SIZE // 2)

    while running:
        if zoom_change_pending and not tiles_loading:
            idx, last_mouse_pos_z, old_vx, old_vy, old_zoom_idx = zoom_change_params
            show_index_progress = True
            index_progress_percent = 0.0
            index_progress_text = "Indexing tiles..."
            index_progress_done_drawn = False
            available_tiles, directory = load_available_tiles(zoom_levels[idx], status_index_progress_callback)
            if idx > old_zoom_idx:
                lat, lon = pixel_to_latlon(last_mouse_pos_z[0], last_mouse_pos_z[1], old_vx, old_vy, zoom_levels[idx-1])
            else:
                lat, lon = pixel_to_latlon(last_mouse_pos_z[0], last_mouse_pos_z[1], old_vx, old_vy, zoom_levels[idx+1])
            map_px, map_py = latlon_to_pixel(lat, lon, zoom_levels[idx])
            viewport_x, viewport_y = int(map_px - last_mouse_pos_z[0]), int(map_py - last_mouse_pos_z[1])
            viewport_x, viewport_y = clamp_viewport(viewport_x, viewport_y, available_tiles)
            zoom_idx = idx
            zoom_change_pending = False
            need_redraw = True
            print(f"Changed to zoom level {zoom_levels[zoom_idx]}")

        mx, my = pygame.mouse.get_pos()
        if show_gps_tooltip and 0 <= mx < VIEWPORT_SIZE and 0 <= my < VIEWPORT_SIZE:
            lat, lon = pixel_to_latlon(mx, my, viewport_x, viewport_y, zoom_levels[zoom_idx])
            mouse_gps_coords = (lat, lon)
            mouse_gps_rect = (mx, my)
        else:
            mouse_gps_coords = None
            mouse_gps_rect = None

        screen.fill((70,70,70))
        pygame.draw.rect(screen, background_color, (0,0,VIEWPORT_SIZE,VIEWPORT_SIZE))

        if available_tiles:
            xs = [x for x, y in available_tiles]
            ys = [y for x, y in available_tiles]
            min_x = max_x = min_y = max_y = 0
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
        else:
            min_x = max_x = min_y = max_y = 0

        min_tile_x = int(viewport_x // TILE_SIZE)
        max_tile_x = int((viewport_x + VIEWPORT_SIZE) // TILE_SIZE)
        min_tile_y = int(viewport_y // TILE_SIZE)
        max_tile_y = int((viewport_y + VIEWPORT_SIZE) // TILE_SIZE)

        visible_tiles = []
        for x in range(min_tile_x, max_tile_x + 1):
            for y in range(min_tile_y, max_tile_y + 1):
                if (x, y) in available_tiles:
                    visible_tiles.append((x, y))

        uncached_tiles = []
        for x, y in visible_tiles:
            key = (zoom_levels[zoom_idx], x, y, background_color, fill_polygons_mode)
            if key not in tile_surfaces:
                uncached_tiles.append((x, y))

        if uncached_tiles and not tiles_loading:
            set_tiles_loading(True)
            show_render_progress = True
            render_progress_percent = 0.0
            render_progress_text = "Loading visible tiles..."
            def done_callback():
                set_tiles_loading(False)
            preload_tile_surfaces_threaded(
                uncached_tiles, zoom_levels[zoom_idx], directory, background_color, fill_polygons_mode,
                status_render_progress_callback, done_callback
            )

        for x, y in visible_tiles:
            surf = get_tile_surface(x, y, zoom_levels[zoom_idx], directory, background_color, fill_polygons_mode)
            if surf:
                px = x * TILE_SIZE - viewport_x
                py = y * TILE_SIZE - viewport_y
                screen.blit(surf, (px, py))

        draw_tile_labels(
            screen, font, available_tiles, viewport_x, viewport_y, zoom_levels[zoom_idx], background_color, show_tile_labels, directory
        )
        pygame.draw.rect(screen, (0,0,0), (toolbar_x, toolbar_y, TOOLBAR_WIDTH, VIEWPORT_SIZE))
        pygame.draw.line(screen, (160,160,160), (toolbar_x,0), (toolbar_x, VIEWPORT_SIZE))
        draw_button(
            screen,
            button_text_black if background_color == (255, 255, 255) else button_text_white,
            button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_bg, pressed=button_pressed
        )
        label_btn_text = "Tile labels ON" if show_tile_labels else "Tile labels OFF"
        draw_button(
            screen, label_btn_text, tile_label_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_label, pressed=False
        )
        gps_btn_text = "GPS Cursor ON" if show_gps_tooltip else "GPS Cursor OFF"
        draw_button(
            screen, gps_btn_text, gps_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_gps, pressed=False
        )
        fill_btn_text = "Fill polygons ON" if fill_polygons_mode else "Fill polygons OFF"
        draw_button(
            screen, fill_btn_text, fill_button_rect, button_color, button_fg, button_border, font_b,
            icon=icon_fill, pressed=fill_polygons_mode
        )
        pygame.draw.rect(screen, (0,0,0), (0, VIEWPORT_SIZE, WINDOW_WIDTH, STATUSBAR_HEIGHT))
        pygame.draw.line(screen, (160,160,160), (0, VIEWPORT_SIZE), (WINDOW_WIDTH, VIEWPORT_SIZE))
        zoom_text = f"Zoom level: {zoom_levels[zoom_idx]}"
        zoom_img = font_status.render(zoom_text, True, (255,255,255))
        screen.blit(zoom_img, (16, VIEWPORT_SIZE + STATUSBAR_HEIGHT//2 - zoom_img.get_height()//2))

        if show_gps_tooltip and mouse_gps_coords is not None:
            lat, lon = mouse_gps_coords[0], mouse_gps_coords[1]
            lat_gms = decimal_to_gms(lat, is_latitude=True)
            lon_gms = decimal_to_gms(lon, is_latitude=False)
            txt = f"lat: {lat:.6f} ({lat_gms})\nlon: {lon:.6f} ({lon_gms})"
            tooltip_lines = txt.split('\n')
            tooltip_surfs = [font.render(line, True, (255,255,255)) for line in tooltip_lines]
            tw = max(s.get_width() for s in tooltip_surfs)
            th = sum(s.get_height() for s in tooltip_surfs)
            tm = 4
            mx, my = mouse_gps_rect[:2]
            tooltip_rect = pygame.Rect(mx+10, my+10, tw+tm*2, th+tm*2)
            pygame.draw.rect(screen, (0,0,0), tooltip_rect)
            pygame.draw.rect(screen, (200,200,200), tooltip_rect, 1)
            yoff = tooltip_rect.top + tm
            for surf in tooltip_surfs:
                screen.blit(surf, (tooltip_rect.left + tm, yoff))
                yoff += surf.get_height()

        if show_index_progress:
            show_status_progress_bar(screen, index_progress_percent, index_progress_text, font_main)
            if index_progress_percent >= 1.0:
                index_progress_done_drawn = True
            else:
                index_progress_done_drawn = False
        if show_render_progress:
            show_status_progress_bar(screen, render_progress_percent, render_progress_text, font_main)
            if render_progress_percent >= 1.0 and not uncached_tiles and not tiles_loading:
                render_progress_done_drawn = True
            else:
                render_progress_done_drawn = False

        pygame.display.flip()
        need_redraw = False

        if index_progress_done_drawn:
            hide_index_progress()
        if render_progress_done_drawn:
            hide_render_progress()

        can_interact = not show_index_progress and not show_render_progress and not tiles_loading

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                base_step = VIEWPORT_SIZE // 4
                zoom_factor = 1 + ((zoom_levels[zoom_idx] - min_zoom) * 0.23) if zoom_levels[zoom_idx] > min_zoom else 1
                step = int(base_step * zoom_factor)
                if event.key == pygame.K_LEFT and can_interact:
                    viewport_x = max(min_x * TILE_SIZE, viewport_x - step)
                    need_redraw = True
                elif event.key == pygame.K_RIGHT and can_interact:
                    viewport_x = min(viewport_x + step, (max_x * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE)
                    need_redraw = True
                elif event.key == pygame.K_UP and can_interact:
                    viewport_y = max(min_y * TILE_SIZE, viewport_y - step)
                    need_redraw = True
                elif event.key == pygame.K_DOWN and can_interact:
                    viewport_y = min(viewport_y + step, (max_y * TILE_SIZE + TILE_SIZE) - VIEWPORT_SIZE)
                    need_redraw = True
                elif event.key == pygame.K_LEFTBRACKET and can_interact and not zoom_change_pending:
                    if zoom_idx > 0:
                        start_zoom_change(zoom_idx-1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.key == pygame.K_RIGHTBRACKET and can_interact and not zoom_change_pending:
                    if zoom_idx < len(zoom_levels) - 1:
                        start_zoom_change(zoom_idx+1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.key == pygame.K_l and can_interact:
                    show_tile_labels = not show_tile_labels
                    need_redraw = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and can_interact:
                    if tile_label_button_rect.collidepoint(event.pos):
                        show_tile_labels = not show_tile_labels
                        need_redraw = True
                    elif button_rect.collidepoint(event.pos):
                        background_color = (255, 255, 255) if background_color == (0, 0, 0) else (0, 0, 0)
                        need_redraw = True
                        print(f"Background color changed to {'white' if background_color == (255, 255, 255) else 'black'}")
                    elif gps_button_rect.collidepoint(event.pos):
                        show_gps_tooltip = not show_gps_tooltip
                        need_redraw = True
                    elif fill_button_rect.collidepoint(event.pos):
                        fill_polygons_mode = not fill_polygons_mode
                        tile_surfaces.clear()
                        need_redraw = True
                    else:
                        dragging = True
                        drag_start = event.pos
                        drag_viewport_start = (viewport_x, viewport_y)
                elif event.button == 4 and can_interact and not zoom_change_pending:
                    if zoom_idx < len(zoom_levels) - 1:
                        start_zoom_change(zoom_idx+1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
                elif event.button == 5 and can_interact and not zoom_change_pending:
                    if zoom_idx > 0:
                        start_zoom_change(zoom_idx-1, last_mouse_pos, viewport_x, viewport_y, zoom_idx)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    drag_start = None
            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                last_mouse_pos = (mx, my)
                if dragging:
                    dx = drag_start[0] - event.pos[0]
                    dy = drag_start[1] - event.pos[1]
                    viewport_x = drag_viewport_start[0] + dx
                    viewport_y = drag_viewport_start[1] + dy
                    viewport_x, viewport_y = clamp_viewport(viewport_x, viewport_y, available_tiles)
                    need_redraw = True

        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tile_viewer.py VECTORMAP")
        print("Keys: [arrows] move, [ ] [ ] zoom level, mouse scroll: zoom level")
        print("Mouse: drag to pan, buttons for background, tile labels, GPS cursor, fill polygons. [l] toggle labels")
        print("Example: python tile_viewer.py VECTORMAP")
        sys.exit(1)
    main(sys.argv[1])