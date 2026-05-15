/**
 * @file constants.hpp
 * @brief Rendering constants and per-zoom lookup tables (ported from Python constants.py).
 * @version 0.7.0
 * @date 2026-05
 */

#pragma once
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>

namespace nav {
namespace constants {

// Half-pixel widths per zoom. Firmware divides by 2.0f.
// Values <= 8 use drawLine (no AA), > 8 use drawWideLine (AA).
inline const std::unordered_map<std::string, std::map<int,int>>& line_width_per_zoom()
{
    static const std::unordered_map<std::string, std::map<int,int>> T = {
        {"motorway",      {{6,2},{7,2},{8,2},{9,2},{10,3},{11,3},{12,4},{13,5},{14,6},{15,7},{16,10},{17,18},{18,22},{19,28}}},
        {"motorway_link", {{10,2},{11,2},{12,2},{13,3},{14,3},{15,5},{16,8},{17,14},{18,14},{19,16}}},
        {"trunk",         {{6,2},{7,2},{8,2},{9,2},{10,3},{11,3},{12,4},{13,5},{14,6},{15,7},{16,10},{17,18},{18,22},{19,28}}},
        {"trunk_link",    {{10,2},{11,2},{12,2},{13,3},{14,3},{15,5},{16,8},{17,14},{18,14},{19,16}}},
        {"primary",       {{8,2},{9,2},{10,3},{11,3},{12,3},{13,4},{14,5},{15,6},{16,10},{17,16},{18,22},{19,28}}},
        {"primary_link",  {{10,2},{11,2},{12,2},{13,3},{14,3},{15,4},{16,8},{17,12},{18,14},{19,16}}},
        {"secondary",     {{10,2},{11,2},{12,3},{13,3},{14,4},{15,5},{16,10},{17,14},{18,22},{19,28}}},
        {"secondary_link",{{10,2},{11,2},{12,2},{13,3},{14,2},{15,3},{16,8},{17,10},{18,14},{19,16}}},
        {"tertiary",      {{11,2},{12,2},{13,3},{14,3},{15,4},{16,10},{17,12},{18,19},{19,28}}},
        {"tertiary_link", {{12,2},{13,2},{14,2},{15,3},{16,8},{17,10},{18,12},{19,16}}},
        {"pedestrian",    {{13,2},{14,2},{15,3},{16,8},{17,12},{18,15},{19,18}}},
        {"residential",   {{13,2},{14,2},{15,3},{16,5},{17,8},{18,11},{19,18}}},
        {"living_street", {{13,2},{14,2},{15,3},{16,5},{17,8},{18,11},{19,18}}},
        {"unclassified",  {{12,2},{13,2},{14,3},{15,3},{16,5},{17,12},{18,11},{19,18}}},
        {"service",       {{13,2},{14,2},{15,2},{16,3},{17,6},{18,8},{19,10}}},
        {"track",         {{15,2},{16,2},{17,4},{18,4},{19,6}}},
        {"footway",       {{13,2},{14,2},{15,2},{16,2},{17,2},{18,2},{19,2}}},
        {"cycleway",      {{13,2},{14,2},{15,2},{16,2},{17,2},{18,2},{19,2}}},
        {"path",          {{13,2},{14,2},{15,2},{16,2},{17,2},{18,2},{19,2}}},
        {"bridleway",     {{13,2},{14,2},{15,2},{16,2},{17,2},{18,2},{19,2}}},
        {"rail",          {{9,2},{10,2},{11,2},{12,2},{13,2},{14,2},{15,2},{16,3},{17,3},{18,4},{19,6}}},
        {"tram",          {{15,2},{16,2},{17,2},{18,3},{19,4}}},
        {"narrow_gauge",  {{13,2},{14,2},{15,2},{16,2},{17,2},{18,3},{19,4}}},
        {"funicular",     {{13,2},{14,2},{15,2},{16,2},{17,2},{18,3},{19,4}}},
        {"cable_car",     {{12,2},{13,2},{14,2},{15,2},{16,2},{17,2},{18,3},{19,4}}},
        {"runway",        {{12,4},{13,6},{14,8},{15,12},{16,16},{17,22},{18,28}}},
        {"taxiway",       {{12,2},{13,3},{14,4},{15,5},{16,6},{17,10},{18,14}}},
        {"helipad",       {{12,2},{13,4},{14,4},{15,5},{16,6},{17,8},{18,10}}},
    };
    return T;
}

// Color overrides per zoom (hex strings, converted at usage).
inline const std::unordered_map<std::string, std::map<int,std::string>>& line_color_per_zoom()
{
    static const std::unordered_map<std::string, std::map<int,std::string>> T = {
        {"residential",   {{13,"#cccccc"}}},
        {"unclassified",  {{12,"#cccccc"}}},
        {"living_street", {{12,"#cccccc"}}},
        {"track",         {{15,"#ffffff"},{16,"#ffffff"}}},
        {"service",       {{16,"#cccccc"}}},
        {"secondary",     {{10,"#bababa"},{11,"#bababa"}}},
    };
    return T;
}

struct PointFeatureDef { std::string shape; };
inline const std::unordered_map<std::string, PointFeatureDef>& point_features()
{
    static const std::unordered_map<std::string, PointFeatureDef> T = {
        {"natural=peak",    {"triangle"}},
        {"natural=volcano", {"triangle"}},
    };
    return T;
}

struct TextFeatureDef
{
    uint8_t font_size;
    std::vector<std::pair<int,int>> zoom_rules; // (min_pop, zoom) sorted desc
};

inline const std::unordered_map<std::string, TextFeatureDef>& text_features()
{
    static const std::unordered_map<std::string, TextFeatureDef> T = {
        {"place=city",    {2, {{1000000,4},{500000,5},{100000,6},{0,8}}}},
        {"place=town",    {1, {{50000,8},{15000,9},{5000,10},{0,11}}}},
        {"place=village", {0, {{2000,11},{500,12},{0,13}}}},
        {"place=suburb",  {0, {{0,12}}}},
        {"place=hamlet",  {0, {{0,14}}}},
    };
    return T;
}

constexpr int ROAD_LABEL_SPACING = 75;
constexpr int PLACE_NAME_BREAK_THRESHOLD = 12;
constexpr int POINT_SYMBOL_SIZE_PX = 3;
constexpr int LABEL_CHAR_WIDTH_PX = 7;
constexpr int LABEL_HEIGHT_PX = 11;

constexpr int POPULATION_MAJOR_CITY = 500000;
constexpr int POPULATION_LARGE_CITY = 100000;
constexpr int POPULATION_TOWN = 15000;

constexpr double CLIP_MARGIN_POLYGON = 0.10;
constexpr double CLIP_MARGIN_LINE = 1.0;
constexpr double K_VISIBILITY = 2.0;
constexpr double K_HOLE_FACTOR = 10.0;
constexpr const char* LAND_BG_COLOR = "#f2efe9";

inline double min_area_deg2_for_zoom(int z)
{
    if (z >= 14) return 0.0;
    double zres_prev = 360.0 / (std::pow(2, z - 1) * 256.0);
    double multiplier;
    if (z <= 7) multiplier = 2.5;
    else if (z == 8) multiplier = 1.8;
    else if (z == 9) multiplier = 2.5;
    else multiplier = 3.0;
    return (zres_prev * zres_prev) * multiplier;
}

inline double post_projection_min_area(int z)
{
    double factor;
    if (z <= 7) factor = 8.0;
    else if (z == 8) factor = 6.0;
    else if (z == 9) factor = 8.0;
    else if (z <= 11) factor = 8.0;
    else if (z == 12) factor = 5.0;
    else if (z == 13) factor = 2.0;
    else if (z == 14) factor = 0.5;
    else factor = 0.1;
    return K_VISIBILITY * factor;
}

} // namespace constants
} // namespace nav
