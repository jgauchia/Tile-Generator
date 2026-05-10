/**
 * @file utils.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Geographic projection and bitwise utility functions.
 * @version 0.6.0
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

inline uint64_t zigzag_encode(int64_t n)
{
    return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

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

inline double lon_to_x(double lon, int z)
{
    double n = std::pow(2.0, z);
    return (lon + 180.0) / 360.0 * n;
}

inline double lat_to_y(double lat, int z)
{
    double n = std::pow(2.0, z);
    double lat_rad = lat * M_PI / 180.0;
    return (1.0 - std::asinh(std::tan(lat_rad)) / M_PI) / 2.0 * n;
}

inline double tile_x_to_lon(double x, int z)
{
    return x / std::pow(2.0, z) * 360.0 - 180.0;
}

inline double tile_y_to_lat(double y, int z)
{
    double n = std::pow(2.0, z);
    double lat_rad = std::atan(std::sinh(M_PI * (1.0 - 2.0 * y / n)));
    return lat_rad * 180.0 / M_PI;
}

inline double lat_to_mercator_y(double lat)
{
    double lat_rad = lat * M_PI / 180.0;
    lat_rad = std::max(-0.999 * M_PI / 2.0, std::min(0.999 * M_PI / 2.0, lat_rad));
    return std::log(std::tan(lat_rad) + 1.0 / std::cos(lat_rad));
}

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

inline uint8_t pack_zoom_priority(int min_zoom, int priority)
{
    uint8_t zoom_nibble = std::min(min_zoom, 15) & 0x0F;
    uint8_t priority_nibble = std::min(priority, 15) & 0x0F;
    return (zoom_nibble << 4) | priority_nibble;
}

/**
 * @brief Hilbert curve helper: rotate/flip a quadrant appropriately.
 */
inline void hilbert_rot(uint32_t n, uint32_t *x, uint32_t *y, uint32_t rx, uint32_t ry) 
{
    if (ry == 0) 
    {
        if (rx == 1) 
        {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        uint32_t t = *x;
        *x = *y;
        *y = t;
    }
}

/**
 * @brief Convert (x,y) tile coordinates to Hilbert index.
 * @param x Tile X coordinate
 * @param y Tile Y coordinate
 * @param z Zoom level (defines grid size 2^z)
 * @return 64-bit Hilbert distance
 */
inline uint64_t xy_to_hilbert(uint32_t x, uint32_t y, int z) 
{
    uint32_t rx, ry;
    uint64_t d = 0;
    uint32_t n = 1 << z;
    for (uint32_t s = n / 2; s > 0; s /= 2) 
    {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += static_cast<uint64_t>(s) * s * ((3 * rx) ^ ry);
        hilbert_rot(s, &x, &y, rx, ry);
    }
    return d;
}

/**
 * @brief Convert Hilbert index to (x,y) tile coordinates.
 */
inline void hilbert_to_xy(uint64_t d, int z, uint32_t &x, uint32_t &y) 
{
    uint32_t rx, ry, s;
    uint64_t t = d;
    x = y = 0;
    uint32_t n = 1 << z;
    for (s = 1; s < n; s *= 2) 
    {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        hilbert_rot(s, &x, &y, rx, ry);
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
}

inline uint8_t meters_to_pixels(double width_meters, int zoom, double lat = 45.0)
{
    double meters_per_pixel = 156543.0 * std::cos(lat * M_PI / 180.0) / std::pow(2.0, zoom);
    double raw_pixels = width_meters / meters_per_pixel;
    if (zoom >= 13)
        raw_pixels *= 0.7;
    int pixels = static_cast<int>(raw_pixels + 0.5);
    return static_cast<uint8_t>(std::max(1, std::min(15, pixels)));
}

inline std::vector<Point> densify_linestring(const std::vector<Point>& pts, double max_seg = 0.0001)
{
    if (pts.size() < 2) return pts;
    std::vector<Point> out;
    out.push_back(pts[0]);
    for (size_t i = 1; i < pts.size(); ++i)
    {
        double dx = pts[i].lon - pts[i-1].lon;
        double dy = pts[i].lat - pts[i-1].lat;
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist > max_seg)
        {
            int n = static_cast<int>(std::ceil(dist / max_seg));
            for (int j = 1; j < n; ++j)
            {
                double t = (double)j / n;
                out.push_back({pts[i-1].lon + dx * t, pts[i-1].lat + dy * t});
            }
        }
        out.push_back(pts[i]);
    }
    return out;
}

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
