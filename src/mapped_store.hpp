/**
 * @file mapped_store.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief High-performance POSIX memory-mapped storage for map features.
 * @version 0.4.0
 * @date 2026-02
 */

#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include "nav_types.hpp"

namespace nav {

/**
 * @class MappedStore
 * @brief Manages a file-backed memory region for storing large amounts of geometric data.
 * 
 * This class uses Linux mmap() to map a file directly into the process address space.
 * It allows storing millions of features without exhausting the physical RAM.
 */
class MappedStore
{
public:
    /**
     * @brief Constructor.
     * @param filename Path to the temporary backing file.
     * @param initial_capacity Initial size of the mapped region in bytes (default 256MB).
     */
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

    /** @brief Destructor. Cleans up mapping and removes the temporary file. */
    ~MappedStore()
    {
        if (mapped_ptr && mapped_ptr != MAP_FAILED) munmap(mapped_ptr, capacity);
        if (fd != -1) close(fd);
        if (std::filesystem::exists(file_path)) std::filesystem::remove(file_path);
    }

    /**
     * @brief Appends a feature to the mapped storage.
     * @param f The feature to serialize and store.
     * @return Offset to the stored feature.
     */
    size_t append(const Feature& f)
    {
        // Calculate required size for serialization
        size_t pts_size = f.points.size() * sizeof(Point);
        size_t ends_size = f.ring_ends.size() * sizeof(uint32_t);
        size_t widths_size = f.zoom_widths.size() * (sizeof(int) + sizeof(uint8_t));
        
        // Header + Points + RingEnds + NumWidths + WidthsData
        size_t total_size = sizeof(int64_t) + 4 + 4 + 4 + pts_size + ends_size + sizeof(size_t) + widths_size;

        if (current_offset + total_size > capacity) grow();

        uint8_t* p = mapped_ptr + current_offset;
        size_t start_offset = current_offset;

        // Serialize fixed-size metadata
        memcpy(p, &f.id, sizeof(int64_t)); p += sizeof(int64_t);
        *p++ = f.geom_type;
        memcpy(p, &f.color_rgb565, 2); p += 2;
        *p++ = f.zoom_priority;
        memcpy(p, &f.width_meters, 4); p += 4;

        // Serialize counts
        uint32_t n_pts = (uint32_t)f.points.size();
        uint32_t n_rings = (uint32_t)f.ring_ends.size();
        memcpy(p, &n_pts, 4); p += 4;
        memcpy(p, &n_rings, 4); p += 4;

        // Serialize variable-size data
        if (pts_size > 0) { memcpy(p, f.points.data(), pts_size); p += pts_size; }
        if (ends_size > 0) { memcpy(p, f.ring_ends.data(), ends_size); p += ends_size; }

        // Serialize dynamic widths map
        size_t n_widths = f.zoom_widths.size();
        memcpy(p, &n_widths, sizeof(size_t)); p += sizeof(size_t);
        for (auto const& [z, w] : f.zoom_widths)
        {
            memcpy(p, &z, sizeof(int)); p += sizeof(int);
            *p++ = w;
        }

        current_offset = p - mapped_ptr;
        return start_offset;
    }

    /**
     * @brief Deserializes a feature from a given offset.
     * @param offset The offset where the feature starts.
     * @return A reconstructed Feature object.
     */
    Feature get(size_t offset) const
    {
        Feature f;
        const uint8_t* p = mapped_ptr + offset;

        memcpy(&f.id, p, sizeof(int64_t)); p += sizeof(int64_t);
        f.geom_type = *p++;
        memcpy(&f.color_rgb565, p, 2); p += 2;
        f.zoom_priority = *p++;
        memcpy(&f.width_meters, p, 4); p += 4;

        uint32_t n_pts, n_rings;
        memcpy(&n_pts, p, 4); p += 4;
        memcpy(&n_rings, p, 4); p += 4;

        if (n_pts > 0)
        {
            f.points.resize(n_pts);
            memcpy(f.points.data(), p, n_pts * sizeof(Point));
            p += n_pts * sizeof(Point);
        }

        if (n_rings > 0)
        {
            f.ring_ends.resize(n_rings);
            memcpy(f.ring_ends.data(), p, n_rings * sizeof(uint32_t));
            p += n_rings * sizeof(uint32_t);
        }

        size_t n_widths;
        memcpy(&n_widths, p, sizeof(size_t)); p += sizeof(size_t);
        for (size_t i = 0; i < n_widths; ++i)
        {
            int z; uint8_t w;
            memcpy(&z, p, sizeof(int)); p += sizeof(int);
            w = *p++;
            f.zoom_widths[z] = w;
        }

        return f;
    }

private:
    /** @brief Doubles the size of the mapped file when capacity is reached. */
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
