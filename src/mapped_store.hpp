/**
 * @file mapped_store.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief High-performance POSIX memory-mapped storage for map features.
 * @version 1.0.3
 * @date 2026-10
 */

#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <vector>
#include <string>
#include <string_view>
#include <cstring>
#include <stdexcept>
#include "nav_types.hpp"

namespace nav {

struct FeatureView
{
    uint8_t geom_type = 0;
    uint16_t color_rgb565 = 0;
    uint8_t zoom_priority = 0;
    float width_meters = 0.0f;
    const Point* points = nullptr;
    uint32_t n_points = 0;
    const uint32_t* ring_ends = nullptr;
    uint32_t n_rings = 0;
    const uint8_t* widths = nullptr; // 5 bytes per entry: int zoom + uint8_t pixels
    uint32_t n_widths = 0;
    std::string_view highway_type;
    std::string_view layer;
    std::string_view ref;
    bool is_bridge = false;
    bool is_building = false;

    int width_zoom(uint32_t i) const
    {
        int z;
        std::memcpy(&z, widths + i * 5, sizeof(int));
        return z;
    }
    uint8_t width_px(uint32_t i) const { return widths[i * 5 + 4]; }
};

class MappedStore
{
public:
    MappedStore(const std::string& filename, size_t initial_capacity = 256 * 1024 * 1024)
        : file_path(filename), capacity(initial_capacity), current_offset(0)
    {
        fd = open(file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd == -1) throw std::runtime_error("Could not create mapped file: " + file_path);

        if (ftruncate(fd, capacity) == -1)
        {
            close(fd);
            throw std::runtime_error("Could not set initial capacity for: " + file_path);
        }

        mapped_ptr = (uint8_t*)mmap(nullptr, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_ptr == MAP_FAILED)
        {
            close(fd);
            throw std::runtime_error("mmap failed for: " + file_path);
        }
    }

    ~MappedStore()
    {
        if (mapped_ptr && mapped_ptr != MAP_FAILED) munmap(mapped_ptr, capacity);
        if (fd != -1) close(fd);
        if (std::filesystem::exists(file_path)) std::filesystem::remove(file_path);
    }

    size_t append(const Feature& f)
    {
        size_t pts_size = f.points.size() * sizeof(Point);
        size_t ends_size = f.ring_ends.size() * sizeof(uint32_t);
        size_t widths_size = f.zoom_widths.size() * (sizeof(int) + sizeof(uint8_t));

        size_t total_size = sizeof(int64_t) + 4 + 4 + 4 + 4 + pts_size + ends_size + sizeof(size_t) + widths_size;
        total_size += sizeof(uint16_t) + f.highway_type.size();
        total_size += sizeof(uint16_t) + f.layer.size();
        total_size += sizeof(uint16_t) + f.ref.size();
        total_size += sizeof(uint16_t) + f.name.size();
        total_size += 4; // is_bridge + is_building + oneway + maxspeed

        if (current_offset + total_size > capacity) grow();

        uint8_t* p = mapped_ptr + current_offset;
        size_t start_offset = current_offset;

        memcpy(p, &f.id, sizeof(int64_t)); p += sizeof(int64_t);
        *p++ = f.geom_type;
        memcpy(p, &f.color_rgb565, 2); p += 2;
        *p++ = f.zoom_priority;
        memcpy(p, &f.width_meters, 4); p += 4;

        uint32_t n_pts = (uint32_t)f.points.size();
        uint32_t n_rings = (uint32_t)f.ring_ends.size();
        memcpy(p, &n_pts, 4); p += 4;
        memcpy(p, &n_rings, 4); p += 4;

        if (pts_size > 0) { memcpy(p, f.points.data(), pts_size); p += pts_size; }
        if (ends_size > 0) { memcpy(p, f.ring_ends.data(), ends_size); p += ends_size; }

        size_t n_widths = f.zoom_widths.size();
        memcpy(p, &n_widths, sizeof(size_t)); p += sizeof(size_t);
        for (auto const& [z, w] : f.zoom_widths)
        {
            memcpy(p, &z, sizeof(int)); p += sizeof(int);
            *p++ = w;
        }

        // Extended fields
        p = write_string(p, f.highway_type);
        p = write_string(p, f.layer);
        p = write_string(p, f.ref);
        p = write_string(p, f.name);
        *p++ = f.is_bridge ? 1 : 0;
        *p++ = f.is_building ? 1 : 0;
        *p++ = f.oneway;
        *p++ = f.maxspeed;

        current_offset = p - mapped_ptr;
        return start_offset;
    }

    FeatureView get_view(size_t offset) const
    {
        FeatureView f;
        const uint8_t* p = mapped_ptr + offset;

        p += sizeof(int64_t);
        f.geom_type = *p++;
        memcpy(&f.color_rgb565, p, 2); p += 2;
        f.zoom_priority = *p++;
        memcpy(&f.width_meters, p, 4); p += 4;

        uint32_t n_pts, n_rings;
        memcpy(&n_pts, p, 4); p += 4;
        memcpy(&n_rings, p, 4); p += 4;

        f.n_points = n_pts;
        f.points = reinterpret_cast<const Point*>(p);
        p += n_pts * sizeof(Point);

        f.n_rings = n_rings;
        f.ring_ends = reinterpret_cast<const uint32_t*>(p);
        p += n_rings * sizeof(uint32_t);

        size_t n_widths;
        memcpy(&n_widths, p, sizeof(size_t)); p += sizeof(size_t);
        f.n_widths = static_cast<uint32_t>(n_widths);
        f.widths = p;
        p += n_widths * (sizeof(int) + sizeof(uint8_t));

        f.highway_type = read_string_view(p);
        f.layer = read_string_view(p);
        f.ref = read_string_view(p);
        read_string_view(p); // Skip the name; the tile path does not use it.
        f.is_bridge = (*p++ != 0);
        f.is_building = (*p++ != 0);

        return f;
    }

private:
    uint8_t* write_string(uint8_t* p, const std::string& s)
    {
        uint16_t len = (uint16_t)s.size();
        memcpy(p, &len, 2); p += 2;
        if (len > 0) { memcpy(p, s.data(), len); p += len; }
        return p;
    }

    std::string_view read_string_view(const uint8_t*& p) const
    {
        uint16_t len;
        memcpy(&len, p, 2); p += 2;
        std::string_view s(reinterpret_cast<const char*>(p), len);
        p += len;
        return s;
    }

    void grow()
    {
        size_t new_capacity = capacity * 2;
        if (munmap(mapped_ptr, capacity) == -1) throw std::runtime_error("munmap failed during grow");
        if (ftruncate(fd, new_capacity) == -1) throw std::runtime_error("ftruncate failed during grow");
        mapped_ptr = (uint8_t*)mmap(nullptr, new_capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped_ptr == MAP_FAILED) throw std::runtime_error("mmap failed during grow");
        capacity = new_capacity;
    }

    std::string file_path;
    int fd = -1;
    uint8_t* mapped_ptr = nullptr;
    size_t capacity;
    size_t current_offset;
};

} // namespace nav
