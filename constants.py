"""
Constants for NAV tile generation.

Pure data definitions — no imports, no logic.
"""

# NAV format constants
NAV_MAGIC = b'NAV1'
COORD_SCALE = 10000000  # 1e7 for ~1cm precision
LAND_BG_COLOR = '#f2efe9'  # OSM Carto default land background

# Geometry types
GEOM_POINT = 1
GEOM_LINESTRING = 2
GEOM_POLYGON = 3
GEOM_TEXT = 4

# RENDERING Z-ORDER (nibble 0-15, lower = behind)
# Polygons:               Lines:
#  1: aeroways              1: tunnels (motorway)
#  2: landuse, terrain      2: tunnels (secondary)
#  4: leisure, amenities    3: tunnels (minor)
#  5: pitch, surface, parking
#  6: infrastructure, leisure=track
#  7: buildings
#  8: water                 8: water lines
#  9: bridge polygons       9: service, path, footway
#                          10-11: links
#                          12: residential
#                          13: secondary, tertiary
#                          14: motorway, trunk, primary
#                          15: railways, bridges (road lines)

# Perceptual filtering: minimum visible area in pixels squared
K_VISIBILITY = 2.0
# Anti-pitting: holes must be N times more visible than objects to be kept
K_HOLE_FACTOR = 10.0

# Point features to extract from nodes (rendered as symbols)
# shape: 'triangle' for peaks, 'circle' for places
POINT_FEATURES = {
    'natural=peak': 'triangle',
    'natural=volcano': 'triangle',
}

# Place features to extract as text labels
# Maps to (base_font_size, base_zoom, population_zoom_rules)
# population_zoom_rules: list of (min_pop, zoom) sorted descending
TEXT_FEATURES = {
    'place=city': {
        'font_size': 2,
        'zoom_rules': [(1000000, 4), (500000, 5), (100000, 6), (0, 8)],
    },
    'place=town': {
        'font_size': 1,
        'zoom_rules': [(50000, 8), (15000, 9), (5000, 10), (0, 11)],
    },
    'place=village': {
        'font_size': 0,
        'zoom_rules': [(2000, 11), (500, 12), (0, 13)],
    },
    'place=suburb': {
        'font_size': 0,
        'zoom_rules': [(0, 12)],
    },
    'place=hamlet': {
        'font_size': 0,
        'zoom_rules': [(0, 14)],
    },
}

# Tags that support width (LineStrings only)
WIDTH_TAGS = {'highway', 'railway', 'waterway'}

# Fixed width in pixels per feature type and zoom level (OSM Carto style)
# Format: type_value -> {zoom: pixels}
LINE_WIDTH_PER_ZOOM = {
    # z6-z11: original OSM Carto widths (sausage artifact not visible at low zoom)
    # z12+: halved (min 1) to avoid fillCircle sausage artifact on T-Deck
    'motorway':      {6: 1,  7: 1,  8: 2,  9: 2,  10: 2,  11: 3,  12: 3,  13: 3,  14: 3,  15: 3,  16: 5,  17: 9,  18: 11, 19: 14},
    'motorway_link': {                            10: 1,  11: 2,  12: 1,  13: 2,  14: 2,  15: 2,  16: 4,  17: 6,  18: 7,  19: 8},
    'trunk':         {6: 1,  7: 1,  8: 2,  9: 2,  10: 2,  11: 2,  12: 2,  13: 3,  14: 3,  15: 3,  16: 5,  17: 9,  18: 11, 19: 14},
    'trunk_link':    {                            10: 1,  11: 2,  12: 1,  13: 2,  14: 2,  15: 2,  16: 4,  17: 6,  18: 7,  19: 8},
    'primary':       {              8: 1,  9: 1,  10: 2,  11: 2,  12: 2,  13: 3,  14: 3,  15: 3,  16: 5,  17: 9,  18: 11, 19: 14},
    'primary_link':  {                            10: 1,  11: 1,  12: 1,  13: 2,  14: 2,  15: 2,  16: 4,  17: 6,  18: 7,  19: 8},
    'secondary':     {                            10: 1,  11: 1,  12: 2,  13: 3,  14: 3,  15: 2,  16: 5,  17: 9,  18: 11, 19: 14},
    'secondary_link':{                            10: 1,  11: 1,  12: 1,  13: 2,  14: 2,  15: 2,  16: 4,  17: 6,  18: 7,  19: 8},
    'tertiary':      {                            10: 1,  11: 1,  12: 1,  13: 2,  14: 3,  15: 2,  16: 5,  17: 9,  18: 11, 19: 14},
    'tertiary_link': {                                            12: 1,  13: 1,  14: 2,  15: 2,  16: 4,  17: 6,  18: 7,  19: 8},
    'residential':   {                                                    13: 1,  14: 2,  15: 2,  16: 3,  17: 6,  18: 7,  19: 9},
    'pedestrian':    {                                                    13: 1,  14: 2,  15: 2,  16: 3,  17: 6,  18: 7,  19: 9},
    'living_street': {                                                    13: 1,  14: 2,  15: 2,  16: 3,  17: 6,  18: 7,  19: 9},
    'unclassified':  {                                            12: 1,  13: 2,  14: 2,  15: 2,  16: 3,  17: 6,  18: 7,  19: 9},
    'service':       {                                                    13: 1,  14: 1,  15: 1,  16: 2,  17: 3,  18: 4,  19: 5},
    'track':         {                                                    15: 1,  16: 1,  17: 2,  18: 2,  19: 3},
    'footway':       {                                                    13: 1,  14: 1,  15: 1,  16: 1,  17: 1,  18: 1,  19: 1},
    'cycleway':      {                                                    13: 1,  14: 1,  15: 1,  16: 1,  17: 1,  18: 1,  19: 1},
    'path':          {                                                    13: 1,  14: 1,  15: 1,  16: 1,  17: 1,  18: 1,  19: 1},
    'bridleway':     {                                                    13: 1,  14: 1,  15: 1,  16: 1},
    # Railway - based on OpenStreetMap Carto standard (halved from original, min 1)
    'rail':          {                     9: 1,  10: 1,  11: 1,  12: 1,  13: 2,  14: 2,  15: 2,  16: 2,  17: 2,  18: 3,  19: 4},
    'subway':        {                                            12: 1,  13: 1,  14: 1,  15: 2,  16: 2},
    'tram':          {                                                                    15: 1,  16: 2},
    'narrow_gauge':  {                                                    13: 1,  14: 1,  15: 2,  16: 2},
    'funicular':     {                                                    13: 1,  14: 1,  15: 2,  16: 2},
    # Aeroway - typical runway ~45m, taxiway ~23m, helipad ~15m (halved from original, min 1)
    'runway':        {                                              12: 2,  13: 3,  14: 4,  15: 6,  16: 8,  17: 11, 18: 14},
    'taxiway':       {                                              12: 1,  13: 1,  14: 2,  15: 2,  16: 3,  17: 5,  18: 7},
    'helipad':       {                                              12: 1,  13: 2,  14: 2,  15: 2,  16: 3,  17: 4,  18: 5},
}

# Override color per zoom level (RGB565) - only for features that change color by zoom
# Format: type_value -> {zoom: '#hexcolor'}
# If a zoom is not listed, the default JSON color is used
LINE_COLOR_PER_ZOOM = {
    'residential':   {13: '#cccccc'},
    'unclassified':  {12: '#cccccc'},
    'living_street': {12: '#cccccc'},
    'track':         {15: '#ffffff', 16: '#ffffff'},
    'service':       {16: '#cccccc'},
    'secondary':     {10: '#bababa', 11: '#bababa'},
}

# Layer definitions based on feature types
LAYER_MAPPING = {
    'water': [
        'natural=water', 'natural=coastline', 'natural=bay',
        'waterway=riverbank', 'waterway=dock', 'waterway=boatyard',
        'waterway=river', 'waterway=stream', 'waterway=canal',
        'waterway=ditch', 'waterway=drain',
        'natural=spring', 'natural=wetland',
        'water=river', 'water=canal', 'water=reservoir', 'water=pond', 'water=lake', 'water=basin',
        'landuse=reservoir'
    ],
    'islands': [
        'place=island', 'place=islet'
    ],
    'aeroways': [
        'aeroway=aerodrome'
    ],
    'landuse': [
        'natural=beach', 'natural=sand', 'natural=wood',
        'landuse=forest', 'natural=forest', 'natural=scrub',
        'natural=heath',
        'natural=bare_rock', 'natural=rock', 'natural=scree', 'natural=stone',
        'natural=fell', 'natural=moor', 'natural=shrubbery', 'landuse=quarry',
        'landuse=orchard', 'landuse=vineyard',
        'landuse=farmland', 'landuse=farmyard',
        'landuse=residential',
        'landuse=commercial', 'landuse=retail', 'landuse=industrial',
        'landuse=construction', 'landuse=cemetery', 'landuse=allotments',
        'landuse=village_green',
        'landuse=quarry', 'landuse=military', 'landuse=landfill', 'landuse=brownfield',
        'landuse=basin', 'landuse=railway', 'landuse=education',
        'landuse=garages', 'landuse=flowerbed'
    ],
    'surface': [
        'natural=grassland', 'landuse=grass', 'landuse=meadow',
    ],
    'roads': [
        'highway=motorway', 'highway=motorway_link',
        'highway=trunk', 'highway=trunk_link',
        'highway=primary', 'highway=primary_link',
        'highway=secondary', 'highway=secondary_link',
        'highway=tertiary', 'highway=tertiary_link',
        'highway=residential', 'highway=living_street',
        'highway=unclassified', 'highway=service',
        'highway=pedestrian', 'highway=track',
        'highway=path', 'highway=footway',
        'highway=cycleway', 'highway=steps',
        'highway=crossing', 'highway=bus_stop',
        'highway=construction', 'highway=platform'
    ],
    'railways': [
        'railway=rail', 'railway=subway', 'railway=tram',
        'railway=abandoned', 'railway=disused', 'railway=funicular',
        'railway=narrow_gauge', 'railway=platform'
    ],
    'buildings': [
        'building', 'man_made=tower'
    ],
    'amenities': [
        'amenity=hospital',
        'amenity=school', 'amenity=university',
        'amenity=place_of_worship',
        'amenity=grave_yard', 'amenity=marketplace',
    ],
    'parking': [
        'amenity=parking',
        'amenity=parking_space',
    ],
    'infrastructure': [
        'bridge=yes', 'man_made=bridge',
        'aeroway=runway', 'aeroway=taxiway', 'aeroway=apron',
        'aeroway=hangar', 'aeroway=helipad',
        'aeroway=parking_position',
        'tunnel=yes', 'tunnel=culvert',
        'man_made=embankment', 'man_made=pier',
        'waterway=dam', 'waterway=weir'
    ],
    'terrain': [
        'natural=peak', 'natural=ridge',
        'natural=volcano', 'natural=cliff',
        'natural=tree_row', 'natural=tree',
        'natural=arete', 'natural=earth_bank',
        'natural=shingle', 'natural=glacier'
    ],
    'boundaries': [
        'boundary=administrative'
    ],
    'leisure': [
        'leisure=stadium', 'leisure=sports_centre',
        'leisure=sports_hall', 'leisure=track', 'leisure=swimming_pool',
        'leisure=golf_course', 'leisure=common', 'leisure=playground',
        'landuse=park', 'leisure=park', 'leisure=nature_reserve', 'leisure=garden',
        'leisure=recreation_ground', 'landuse=recreation_ground'
    ],
    'pitch': [
        'leisure=pitch',
    ],
    'places': [
        'place=city', 'place=state', 'place=town',
        'place=village', 'place=hamlet', 'place=square'
    ]
}
