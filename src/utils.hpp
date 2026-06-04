/**
 * @file utils.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Geographic projection, color conversion, and coordinate encoding utilities.
 * @version 0.8.0
 * @date 2026-05
 */

#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include "nav_types.hpp"

namespace nav {
namespace utils {

/**
 * @brief ZigZag-encode a signed integer for efficient VarInt storage.
 * @param n Signed value to encode.
 * @return Unsigned ZigZag representation.
 */
inline uint64_t zigzag_encode(int64_t n)
{
    return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

/**
 * @brief Encode an unsigned integer as a VarInt byte sequence.
 * @param value Value to encode.
 * @return Byte vector with 1–10 bytes depending on magnitude.
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
 * @brief Convert longitude to fractional tile X at a given zoom level (Web Mercator).
 * @param lon Longitude in degrees.
 * @param z   Zoom level.
 * @return Fractional tile X coordinate.
 */
inline double lon_to_x(double lon, int z)
{
    double n = std::pow(2.0, z);
    return (lon + 180.0) / 360.0 * n;
}

/**
 * @brief Convert latitude to fractional tile Y at a given zoom level (Web Mercator).
 * @param lat Latitude in degrees.
 * @param z   Zoom level.
 * @return Fractional tile Y coordinate.
 */
inline double lat_to_y(double lat, int z)
{
    double n = std::pow(2.0, z);
    double lat_rad = lat * M_PI / 180.0;
    return (1.0 - std::asinh(std::tan(lat_rad)) / M_PI) / 2.0 * n;
}

/**
 * @brief Convert fractional tile X to longitude (Web Mercator).
 * @param x Fractional tile X coordinate.
 * @param z Zoom level.
 * @return Longitude in degrees.
 */
inline double tile_x_to_lon(double x, int z)
{
    return x / std::pow(2.0, z) * 360.0 - 180.0;
}

/**
 * @brief Convert fractional tile Y to latitude (Web Mercator).
 * @param y Fractional tile Y coordinate.
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
 * @brief Convert latitude to Web Mercator Y projection (clamped to avoid poles).
 * @param lat Latitude in degrees.
 * @return Mercator Y value in radians.
 */
inline double lat_to_mercator_y(double lat)
{
    double lat_rad = lat * M_PI / 180.0;
    lat_rad = std::max(-0.999 * M_PI / 2.0, std::min(0.999 * M_PI / 2.0, lat_rad));
    return std::log(std::tan(lat_rad) + 1.0 / std::cos(lat_rad));
}

/**
 * @brief Convert an HTML hex color string to RGB565.
 * @param hex Color string in `#RRGGBB` format.
 * @return RGB565 value, or 0xFFFF on parse error.
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
 * @brief Lighten an RGB565 color by blending toward white.
 * @param color  Source RGB565 color.
 * @param factor Blend factor toward white (0.0=unchanged, 1.0=white).
 * @return Lightened RGB565 color.
 */
inline uint16_t lighten_rgb565(uint16_t color, float factor = 0.4f)
{
    int r = ((color >> 11) & 0x1F) * 255 / 31;
    int g = ((color >> 5) & 0x3F) * 255 / 63;
    int b = (color & 0x1F) * 255 / 31;
    r = std::min(255, (int)(r + (255 - r) * factor));
    g = std::min(255, (int)(g + (255 - g) * factor));
    b = std::min(255, (int)(b + (255 - b) * factor));
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/**
 * @brief Darken an RGB565 color by scaling each channel toward black.
 * @param color  Source RGB565 color.
 * @param factor Darkening factor (0.0=unchanged, 1.0=black).
 * @return Darkened RGB565 color.
 */
inline uint16_t darken_rgb565(uint16_t color, float factor = 0.4f)
{
    int r = ((color >> 11) & 0x1F) * 255 / 31;
    int g = ((color >> 5) & 0x3F) * 255 / 63;
    int b = (color & 0x1F) * 255 / 31;
    r = std::max(0, (int)(r * (1.0f - factor)));
    g = std::max(0, (int)(g * (1.0f - factor)));
    b = std::max(0, (int)(b * (1.0f - factor)));
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

/**
 * @brief Pack min_zoom and render priority into a single byte.
 * @param min_zoom Minimum zoom at which the feature is visible (clamped to 0–15).
 * @param priority Render priority for Z-ordering (clamped to 0–15).
 * @return Byte with zoom in the high nibble and priority in the low nibble.
 */
inline uint8_t pack_zoom_priority(int min_zoom, int priority)
{
    uint8_t zoom_nibble = std::min(min_zoom, 15) & 0x0F;
    uint8_t priority_nibble = std::min(priority, 15) & 0x0F;
    return (zoom_nibble << 4) | priority_nibble;
}

/**
 * @brief Convert a real-world road width in meters to a pixel width for a given zoom and latitude.
 * @param width_meters Physical width in meters.
 * @param zoom         Tile zoom level.
 * @param lat          Reference latitude for Mercator scale correction (default 45°).
 * @return Pixel width clamped to [1, 15].
 */
inline uint8_t meters_to_pixels(double width_meters, int zoom, double lat = 45.0)
{
    double meters_per_pixel = 156543.0 * std::cos(lat * M_PI / 180.0) / std::pow(2.0, zoom);
    double raw_pixels = width_meters / meters_per_pixel;
    if (zoom >= 13)
        raw_pixels *= 0.7;
    int pixels = static_cast<int>(raw_pixels + 0.5);
    return static_cast<uint8_t>(std::max(1, std::min(15, pixels)));
}

/**
 * @brief Split a place name with a newline near its midpoint for two-line label rendering.
 * @param name      Input label string.
 * @param threshold Character count below which no split is applied.
 * @return Original string if short enough, otherwise split at the nearest space or hyphen to the midpoint.
 */
inline std::string split_place_name(const std::string& name, int threshold = 12)
{
    if ((int)name.size() <= threshold) return name;
    int mid = (int)name.size() / 2;
    int best = -1;
    int best_dist = (int)name.size();
    for (int i = 0; i < (int)name.size(); ++i)
    {
        if (name[i] == '-' || name[i] == ' ')
        {
            int dist = std::abs(i - mid);
            if (dist < best_dist)
            {
                best_dist = dist;
                best = i;
            }
        }
    }
    if (best > 0)
    {
        std::string result = name;
        if (result[best] == '-')
            result.insert(best + 1, "\n");
        else
            result = result.substr(0, best) + "\n" + result.substr(best + 1);
        return result;
    }
    return name;
}

} // namespace utils
} // namespace nav
