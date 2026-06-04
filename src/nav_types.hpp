/**
 * @file nav_types.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Data structures shared between generator and firmware: MapHeader (NPK2 flat index) and IndexEntry.
 * @version 0.8.0
 * @date 2026-05
 */

#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <map>

namespace nav {

const uint8_t GEOM_POINT = 1;
const uint8_t GEOM_LINESTRING = 2;
const uint8_t GEOM_POLYGON = 3;
const uint8_t GEOM_TEXT = 4;

#pragma pack(push, 1)
struct MapHeader
{
    char magic[4];         // "NPK2"
    uint8_t zoom;
    uint32_t tiles_wide;
    uint32_t tiles_high;
    uint32_t bottom_left[2]; // [0]=min_x, [1]=min_y
    uint16_t color_count;    // number of RGB565 entries in the global palette
};

struct IndexEntry
{
    uint32_t offset;
    uint32_t size;
};
#pragma pack(pop)

struct Point
{
    double lon;
    double lat;
};

struct Feature
{
    int64_t id;
    uint8_t geom_type;
    uint16_t color_rgb565;
    uint8_t zoom_priority;
    float width_meters;
    std::vector<Point> points;
    std::vector<uint32_t> ring_ends;
    std::map<int, uint8_t> zoom_widths;

    std::string highway_type;
    uint8_t oneway = 0;
    uint8_t maxspeed = 0;
    std::string ref;
    std::string old_ref;
    std::string name;
    std::string layer;
    std::string shape;

    uint8_t font_size = 0;
    std::vector<uint8_t> text;
    int population = 0;

    bool is_bridge = false;
    bool is_building = false;

    uint16_t bg_color_rgb565 = 0;
    uint16_t border_color_rgb565 = 0;
    std::vector<Point> coords_candidates;
    std::vector<int64_t> osm_node_ids;
};

} // namespace nav
