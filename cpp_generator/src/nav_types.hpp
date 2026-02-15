/**
 * @file nav_types.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Basic data structures and constants for the NAV tile generator.
 * @version 0.4.0
 * @date 2026-02
 */

#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace nav {

/** @brief Scale factor for coordinate conversion (OSM precision) */
const uint32_t COORD_SCALE = 10000000;

/** @brief NAV geometry types */
const uint8_t GEOM_LINESTRING = 2;
const uint8_t GEOM_POLYGON = 3;

/**
 * @struct Point
 * @brief Represents a single geographic point in longitude and latitude.
 */
struct Point
{
    double lon;
    double lat;
};

/**
 * @struct Ring
 * @brief A sequence of points forming a geometric ring.
 */
struct Ring
{
    std::vector<Point> points;
};

/**
 * @struct Feature
 * @brief Represents a map feature with its metadata and geometry rings.
 */
struct Feature
{
    int64_t id;             ///< OSM ID
    uint8_t geom_type;      ///< Geometry type (Linestring or Polygon)
    uint16_t color_rgb565;  ///< Display color in RGB565 format
    uint8_t zoom_priority;  ///< Packed byte: high nibble=min_zoom, low nibble=priority
    float width_meters;     ///< Line width in meters (for Linestrings)
    std::vector<Point> points;       ///< Flattened coordinate points for all rings
    std::vector<uint32_t> ring_ends; ///< End indices for each ring in the points vector
    std::map<int, uint8_t> zoom_widths; ///< Optional aesthetic widths per zoom level
};

} // namespace nav