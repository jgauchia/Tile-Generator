/**
 * @file utils.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Geographic projection and bitwise utility functions.
 * @version 0.4.0
 * @date 2026-02
 */

#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>

namespace nav {
/** @brief Utility functions for coordinate transformation and encoding. */
namespace utils {

/**
 * @brief ZigZag encoding for signed integers (maps negative to positive).
 * @param n Signed 64-bit integer.
 * @return Unsigned 64-bit encoded value.
 */
inline uint64_t zigzag_encode(int64_t n)
{
    return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

/**
 * @brief Encodes a 64-bit value to VarInt format.
 * @param value Value to encode.
 * @return Vector of bytes.
 */
inline std::vector<uint8_t> to_varint(uint64_t value)
{
    std::vector<uint8_t> out;
    while (value >= 0x80)
    {
        out.push_back(static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<uint8_t>(value));
    return out;
}

/**
 * @brief Converts longitude to tile X coordinate.
 * @param lon Longitude.
 * @param z Zoom level.
 * @return Floating point tile X.
 */
inline double lon_to_x(double lon, int z)
{
    double n = std::pow(2.0, z);
    return (lon + 180.0) / 360.0 * n;
}

/**
 * @brief Converts latitude to tile Y coordinate (Mercator).
 * @param lat Latitude.
 * @param z Zoom level.
 * @return Floating point tile Y.
 */
inline double lat_to_y(double lat, int z)
{
    double n = std::pow(2.0, z);
    double lat_rad = lat * M_PI / 180.0;
    return (1.0 - std::asinh(std::tan(lat_rad)) / M_PI) / 2.0 * n;
}

/**
 * @brief Inverse projection: Tile X to Longitude.
 * @param x Tile X coordinate.
 * @param z Zoom level.
 * @return Longitude in degrees.
 */
inline double tile_x_to_lon(double x, int z)
{
    return x / std::pow(2.0, z) * 360.0 - 180.0;
}

/**
 * @brief Inverse projection: Tile Y to Latitude.
 * @param y Tile Y coordinate.
 * @param z Zoom level.
 * @return Latitude in degrees.
 */
inline double tile_y_to_lat(double y, int z)
{
    double n = std::pow(2.0, z);
    double lat_rad = std::atan(std::sinh(M_PI * (1.0 - 2.0 * y / n)));
    return lat_rad * 180.0 / M_PI;
}

/**
 * @brief Converts Hex color string (#RRGGBB) to RGB565 (uint16).
 * @param hex Hex string.
 * @return 16-bit color.
 */
inline uint16_t hex_to_rgb565(const std::string& hex)
{
    if (hex.length() < 7 || hex[0] != '#')
        return 0xFFFF;
    try
    {
        uint8_t r = std::stoi(hex.substr(1, 2), nullptr, 16);
        uint8_t g = std::stoi(hex.substr(3, 2), nullptr, 16);
        uint8_t b = std::stoi(hex.substr(5, 2), nullptr, 16);
        return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
    }
    catch (...)
    {
        return 0xFFFF;
    }
}

/**
 * @brief Packs min_zoom and rendering priority into a single byte.
 * @param min_zoom Minimum visibility zoom.
 * @param priority Feature priority.
 * @return Packed byte.
 */
inline uint8_t pack_zoom_priority(int min_zoom, int priority)
{
    uint8_t zoom_nibble = std::min(min_zoom, 15) & 0x0F;
    uint8_t priority_nibble = std::min(priority / 7, 15) & 0x0F;
    return (zoom_nibble << 4) | priority_nibble;
}

} // namespace utils
} // namespace nav
