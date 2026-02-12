/**
 * @file tile_processor.hpp
 * @brief Tile generation engine including clipping, merging, and binary writing.
 */

#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <map>
#include <unordered_map>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <geos_c.h>
#include "nav_types.hpp"
#include "utils.hpp"

namespace nav {

/**
 * @class TileProcessor
 * @brief Handles the spatial grouping of features and their conversion to binary NAV tiles.
 * 
 * Uses GEOS library for geometry operations (intersection, union, simplification).
 */
class TileProcessor
{
public:
    /** @brief Constructor with output path setup */
    TileProcessor(const std::string& out_dir) : output_dir(out_dir) {}

    /**
     * @brief Processes all features for a given zoom range.
     * @param features_by_zoom Array of vectors containing map features grouped by min_zoom.
     * @param min_z Minimum zoom level.
     * @param max_z Maximum zoom level.
     */
    void process_all(std::vector<Feature> (&features_by_zoom)[18], int min_z, int max_z)
    {
        total_generated_bytes = 0;
        total_generated_files = 0;
        for (int z = min_z; z <= max_z; ++z)
            process_zoom_level(features_by_zoom, z);
    }

    uint64_t get_total_bytes() const { return total_generated_bytes; }
    size_t get_total_files() const { return total_generated_files; }

private:
    std::string output_dir;
    std::atomic<uint64_t> total_generated_bytes{0};
    std::atomic<size_t> total_generated_files{0};

    struct TileCoord
    {
        int x, y;
        bool operator==(const TileCoord& o) const { return x == o.x && y == o.y; }
    };

    struct TileCoordHash
    {
        std::size_t operator()(const TileCoord& k) const
        {
            // Simple hash combination for 2D coordinates
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };

    // Internal structure for processed objects after geometry operations
    struct ProcessedFeature
    {
        uint8_t type;
        uint16_t color;
        uint8_t prio;
        float width;
        std::vector<std::vector<std::pair<int16_t, int16_t>>> rings;
    };

    /**
     * @brief Converts a Feature structure to a GEOS geometry object.
     * @param f Map feature.
     * @param handle Thread-local GEOS context.
     * @return Pointer to GEOSGeometry, or nullptr if invalid.
     */
    GEOSGeometry* feature_to_geos(const Feature& f, GEOSContextHandle_t handle)
    {
        if (f.ring_ends.empty())
            return nullptr;
        
        if (f.geom_type == GEOM_POLYGON)
        {
            // Outer ring (always first)
            uint32_t start = 0;
            uint32_t end = f.ring_ends[0];
            uint32_t size = end - start;
            if (size < 3) return nullptr;

            GEOSCoordSequence* seq = GEOSCoordSeq_create_r(handle, size, 2);
            for (uint32_t i = 0; i < size; ++i)
            {
                GEOSCoordSeq_setX_r(handle, seq, i, f.points[start + i].lon);
                GEOSCoordSeq_setY_r(handle, seq, i, f.points[start + i].lat);
            }
            GEOSGeometry* shell = GEOSGeom_createLinearRing_r(handle, seq);
            if (!shell)
            {
                GEOSCoordSeq_destroy_r(handle, seq);
                return nullptr;
            }
            
            // Inner rings
            std::vector<GEOSGeometry*> holes;
            for (size_t r = 1; r < f.ring_ends.size(); ++r)
            {
                start = f.ring_ends[r-1];
                end = f.ring_ends[r];
                size = end - start;
                if (size < 3) continue;

                GEOSCoordSequence* hseq = GEOSCoordSeq_create_r(handle, size, 2);
                for (uint32_t j = 0; j < size; ++j)
                {
                    GEOSCoordSeq_setX_r(handle, hseq, j, f.points[start + j].lon);
                    GEOSCoordSeq_setY_r(handle, hseq, j, f.points[start + j].lat);
                }
                GEOSGeometry* hole = GEOSGeom_createLinearRing_r(handle, hseq);
                if (hole)
                    holes.push_back(hole);
                else
                    GEOSCoordSeq_destroy_r(handle, hseq);
            }
            
            GEOSGeometry* poly = GEOSGeom_createPolygon_r(handle, shell, holes.data(), holes.size());
            if (!poly)
            {
                GEOSGeom_destroy_r(handle, shell);
                for (auto h : holes) GEOSGeom_destroy_r(handle, h);
                return nullptr;
            }
            if (!GEOSisValid_r(handle, poly))
            {
                GEOSGeometry* v = GEOSMakeValid_r(handle, poly);
                GEOSGeom_destroy_r(handle, poly);
                return v;
            }
            return poly;
        }
        else
        {
            // LineString
            uint32_t size = f.ring_ends[0];
            if (size < 2) return nullptr;
            GEOSCoordSequence* seq = GEOSCoordSeq_create_r(handle, size, 2);
            for (uint32_t i = 0; i < size; ++i)
            {
                GEOSCoordSeq_setX_r(handle, seq, i, f.points[i].lon);
                GEOSCoordSeq_setY_r(handle, seq, i, f.points[i].lat);
            }
            return GEOSGeom_createLineString_r(handle, seq);
        }
    }

    /**
     * @brief Processes all tiles for a specific zoom level in parallel.
     * @param features_by_zoom Source features grouped by zoom.
     * @param z Current zoom level.
     */
    void process_zoom_level(std::vector<Feature> (&features_by_zoom)[18], int z)
    {
        auto start = std::chrono::steady_clock::now();
        
        // tile_map: maps tile coordinates to a list of {bucket_idx, feature_idx}
        // Using unordered_map for O(1) insertion/lookup
        std::unordered_map<TileCoord, std::vector<std::pair<int, size_t>>, TileCoordHash> tile_map;
        size_t zoom_features = 0;

        for (int b = 0; b <= z; ++b)
        {
            const auto& bucket = features_by_zoom[b];
            for (size_t i = 0; i < bucket.size(); ++i)
            {
                const auto& f = bucket[i];
                zoom_features++;

                int min_tx = 1e9, max_tx = -1, min_ty = 1e9, max_ty = -1;
                for (const auto& p : f.points)
                {
                    int tx = static_cast<int>(utils::lon_to_x(p.lon, z));
                    int ty = static_cast<int>(utils::lat_to_y(p.lat, z));
                    min_tx = std::min(min_tx, tx); max_tx = std::max(max_tx, tx);
                    min_ty = std::min(min_ty, ty); max_ty = std::max(max_ty, ty);
                }
                for (int x = min_tx; x <= max_tx; ++x)
                    for (int y = min_ty; y <= max_ty; ++y)
                        tile_map[{x, y}].push_back({b, i});
            }
        }

        if (tile_map.empty())
            return;

        std::vector<std::pair<TileCoord, std::vector<std::pair<int, size_t>>>> tiles(tile_map.begin(), tile_map.end());
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;
        
        std::atomic<size_t> next_tile(0);
        std::atomic<size_t> merged_count{0};
        std::atomic<uint64_t> zoom_bytes{0};
        
        std::vector<std::thread> workers;
        for (unsigned int t = 0; t < num_threads; ++t)
        {
            workers.emplace_back([&]() {
                GEOSContextHandle_t handle = GEOS_init_r();
                while (true)
                {
                    size_t idx = next_tile.fetch_add(1);
                    if (idx >= tiles.size())
                        break;
                    
                    size_t m = 0;
                    uint64_t b = 0;
                    const auto& [coord, indices] = tiles[idx];
                    write_tile(z, coord.x, coord.y, features_by_zoom, indices, handle, m, b);
                    merged_count += m;
                    zoom_bytes += b;
                }
                GEOS_finish_r(handle);
            });
        }

        // Progress display thread
        while (next_tile < tiles.size())
        {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> zoom_elapsed = now - start;
            float progress = (float)next_tile / tiles.size();
            double tps = (zoom_elapsed.count() > 0.1) ? (double)next_tile / zoom_elapsed.count() : 0;
            int barWidth = 20;
            
            std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": [" << std::string(progress * barWidth, '#') 
                      << std::string(barWidth - progress * barWidth, '-') << "] " 
                      << std::setw(3) << int(progress * 100.0) << "% | "
                      << std::setw(8) << (size_t)next_tile << "/" << tiles.size() << " tiles | "
                      << std::fixed << std::setprecision(1) << tps << " t/s | ";
            
            if (zoom_bytes < 1024 * 1024)
                std::cout << std::fixed << std::setprecision(1) << (zoom_bytes / 1024.0) << " KB" << std::flush;
            else
                std::cout << std::fixed << std::setprecision(1) << (zoom_bytes / 1024.0 / 1024.0) << " MB" << std::flush;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        for (auto& w : workers)
            w.join();
        
        total_generated_bytes += zoom_bytes;

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_tps = (elapsed.count() > 0) ? (double)tiles.size() / elapsed.count() : 0;
        
        std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": " << std::setw(8) << tiles.size() << " tiles ("
                  << std::fixed << std::setprecision(1) << avg_tps << " t/s), "
                  << std::setw(8) << (size_t)merged_count << " polygons merged | ";
        
        if (zoom_bytes < 1024 * 1024)
            std::cout << std::fixed << std::setprecision(2) << (zoom_bytes / 1024.0) << " KB";
        else
            std::cout << std::fixed << std::setprecision(2) << (zoom_bytes / 1024.0 / 1024.0) << " MB";
        
        std::cout << " done in " << elapsed.count() << "s" << std::endl;
    }

    /**
     * @brief Performs merging, clipping and simplification for a single tile.
     * @param z Zoom level.
     * @param x Tile X coordinate.
     * @param y Tile Y coordinate.
     * @param features_by_zoom Source features grouped by zoom level buckets.
     * @param indices List of {bucket_idx, feature_idx} relevant to this tile.
     * @param local_handle Thread-local GEOS context.
     * @param merged_out [out] Atomic counter for merged polygons.
     * @param bytes_out [out] Atomic counter for bytes written.
     */
    void write_tile(int z, int x, int y, std::vector<Feature> (&features_by_zoom)[18], 
                   const std::vector<std::pair<int, size_t>>& indices, 
                   GEOSContextHandle_t local_handle, size_t& merged_out, uint64_t& bytes_out)
    {
        struct Style
        {
            uint16_t color; uint8_t prio;
            bool operator<(const Style& o) const { return color < o.color || (color == o.color && prio < o.prio); }
        };
        std::map<Style, std::vector<GEOSGeometry*>> poly_groups;
        std::vector<const Feature*> lines;

        for (const auto& idx_pair : indices)
        {
            const auto& f = features_by_zoom[idx_pair.first][idx_pair.second];
            if (f.geom_type == GEOM_POLYGON)
            {
                GEOSGeometry* g = feature_to_geos(f, local_handle);
                if (g)
                    poly_groups[{f.color_rgb565, f.zoom_priority}].push_back(g);
            }
            else
                lines.push_back(&f);
        }

        double lmin = utils::tile_x_to_lon(x, z); double lmax = utils::tile_x_to_lon(x + 1, z);
        double tmax = utils::tile_y_to_lat(y, z); double tmin = utils::tile_y_to_lat(y + 1, z);
        double lm = (lmax - lmin) * 0.1; double tm = (tmax - tmin) * 0.1;
        
        GEOSCoordSequence* cbox = GEOSCoordSeq_create_r(local_handle, 5, 2);
        GEOSCoordSeq_setX_r(local_handle, cbox, 0, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 0, tmin - tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 1, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cbox, 1, tmin - tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 2, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cbox, 2, tmax + tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 3, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 3, tmax + tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 4, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 4, tmin - tm);
        GEOSGeometry* clip_geom = GEOSGeom_createPolygon_r(local_handle, GEOSGeom_createLinearRing_r(local_handle, cbox), NULL, 0);

        double tolerance = (lmax - lmin) / 512.0;
        std::vector<ProcessedFeature> final_features;

        // Process Polygons (Merge + Clip + Simplify)
        for (auto& [style, geos_polys] : poly_groups)
        {
            size_t before = geos_polys.size();
            GEOSGeometry* coll = GEOSGeom_createCollection_r(local_handle, GEOS_GEOMETRYCOLLECTION, geos_polys.data(), geos_polys.size());
            GEOSGeometry* merged = GEOSUnaryUnion_r(local_handle, coll);
            GEOSGeometry* simplified = GEOSSimplify_r(local_handle, merged, tolerance);
            GEOSGeometry* clipped = GEOSIntersection_r(local_handle, simplified, clip_geom);
            
            int n = GEOSGetNumGeometries_r(local_handle, clipped);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* g = GEOSGetGeometryN_r(local_handle, clipped, i);
                if (GEOSGeomTypeId_r(local_handle, g) != GEOS_POLYGON)
                    continue;
                
                ProcessedFeature pf{GEOM_POLYGON, style.color, style.prio, 0, {}};
                auto process_ring = [&](const GEOSGeometry* ring) {
                    const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, ring);
                    uint32_t sz; GEOSCoordSeq_getSize_r(local_handle, s, &sz);
                    if (sz < 3)
                        return;
                    std::vector<std::pair<int16_t, int16_t>> rpts;
                    for (uint32_t j = 0; j < sz; ++j)
                    {
                        double lon, lat; GEOSCoordSeq_getX_r(local_handle, s, j, &lon); GEOSCoordSeq_getY_r(local_handle, s, j, &lat);
                        rpts.push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096),
                                       static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
                    }
                    pf.rings.push_back(std::move(rpts));
                };
                process_ring(GEOSGetExteriorRing_r(local_handle, g));
                int nh = GEOSGetNumInteriorRings_r(local_handle, g);
                for (int h = 0; h < nh; ++h)
                    process_ring(GEOSGetInteriorRingN_r(local_handle, g, h));
                
                if (!pf.rings.empty())
                    final_features.push_back(std::move(pf));
            }
            merged_out += (before - std::max((int)0, n));
            GEOSGeom_destroy_r(local_handle, coll); GEOSGeom_destroy_r(local_handle, merged);
            GEOSGeom_destroy_r(local_handle, simplified); GEOSGeom_destroy_r(local_handle, clipped);
        }

        // Process Lines (Clip + Simplify)
        for (const auto* f : lines)
        {
            GEOSGeometry* g = feature_to_geos(*f, local_handle);
            if (!g)
                continue;
            GEOSGeometry* simplified = GEOSSimplify_r(local_handle, g, tolerance);
            GEOSGeometry* clipped = GEOSIntersection_r(local_handle, simplified, clip_geom);
            
            int n = GEOSGetNumGeometries_r(local_handle, clipped);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* part = GEOSGetGeometryN_r(local_handle, clipped, i);
                int type = GEOSGeomTypeId_r(local_handle, part);
                if (type != GEOS_LINESTRING && type != GEOS_LINEARRING)
                    continue;

                const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, part);
                uint32_t sz; GEOSCoordSeq_getSize_r(local_handle, s, &sz);
                if (sz < 2)
                    continue;

                ProcessedFeature pf{GEOM_LINESTRING, f->color_rgb565, f->zoom_priority, f->width_meters, {{}}};
                for (uint32_t j = 0; j < sz; ++j)
                {
                    double lon, lat; GEOSCoordSeq_getX_r(local_handle, s, j, &lon); GEOSCoordSeq_getY_r(local_handle, s, j, &lat);
                    pf.rings[0].push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096),
                                          static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
                }
                final_features.push_back(std::move(pf));
            }
            GEOSGeom_destroy_r(local_handle, g); GEOSGeom_destroy_r(local_handle, simplified); GEOSGeom_destroy_r(local_handle, clipped);
        }

        // File writing
        std::string dir = output_dir + "/" + std::to_string(z) + "/" + std::to_string(x);
        std::filesystem::create_directories(dir);
        std::ofstream out(dir + "/" + std::to_string(y) + ".nav", std::ios::binary);
        if (out)
        {
            uint8_t th[22] = {'N', 'A', 'V', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t b_lmin = (int32_t)(lmin*1e7); int32_t b_tmin = (int32_t)(tmin*1e7);
            int32_t b_lmax = (int32_t)(lmax*1e7); int32_t b_tmax = (int32_t)(tmax*1e7);
            memcpy(th+6, &b_lmin, 4); memcpy(th+10, &b_tmin, 4); memcpy(th+14, &b_lmax, 4); memcpy(th+18, &b_tmax, 4);
            out.write((char*)th, 22);
            uint16_t count = 0;
            
            // Write feature data
            for (const auto& pf : final_features)
            {
                std::vector<uint8_t> payload; 
                int16_t lx = 0, ly = 0;
                int16_t minx = 4096, maxx = 0, miny = 4096, maxy = 0; 
                size_t pts = 0;

                for (const auto& r : pf.rings)
                {
                    for (const auto& p : r)
                    {
                        auto vx = utils::to_varint(utils::zigzag_encode(p.first - lx));
                        payload.insert(payload.end(), vx.begin(), vx.end());
                        auto vy = utils::to_varint(utils::zigzag_encode(p.second - ly));
                        payload.insert(payload.end(), vy.begin(), vy.end());
                        lx = p.first; ly = p.second;
                        int16_t cx = std::max((int16_t)0, std::min((int16_t)4096, p.first));
                        int16_t cy = std::max((int16_t)0, std::min((int16_t)4096, p.second));
                        minx = std::min(minx, cx); maxx = std::max(maxx, cx);
                        miny = std::min(miny, cy); maxy = std::max(maxy, cy);
                    }
                    pts += r.size();
                }

                std::vector<uint8_t> ri; 
                if (pf.type == GEOM_POLYGON)
                {
                    uint16_t nr = (uint16_t)pf.rings.size(); 
                    ri.push_back(nr & 0xFF); ri.push_back(nr >> 8);
                    uint16_t cur = 0; 
                    for (const auto& r : pf.rings)
                    {
                        cur += (uint16_t)r.size(); 
                        ri.push_back(cur & 0xFF); ri.push_back(cur >> 8); 
                    }
                }

                uint8_t fh[13] = {
                    pf.type, (uint8_t)(pf.color & 0xFF), (uint8_t)(pf.color >> 8), pf.prio, 
                    (uint8_t)(pf.type == GEOM_POLYGON ? 0 : utils::meters_to_pixels(pf.width, z)),
                    (uint8_t)(minx / 16), (uint8_t)(miny / 16), (uint8_t)(maxx / 16), (uint8_t)(maxy / 16), 
                    (uint8_t)(pts & 0xFF), (uint8_t)(pts >> 8),
                    (uint8_t)((payload.size() + ri.size()) & 0xFF), (uint8_t)((payload.size() + ri.size()) >> 8)
                };

                out.write((char*)fh, 13); 
                out.write((char*)payload.data(), payload.size()); 
                if (!ri.empty()) out.write((char*)ri.data(), ri.size());
                
                count++; 
                if (count == 0xFFFF) break;
            }
            // Capture real size before seeking back to header
            bytes_out = (uint64_t)out.tellp();
            out.seekp(4); 
            out.write((char*)&count, 2);
            total_generated_files++;
        }
        GEOSGeom_destroy_r(local_handle, clip_geom);
    }
};

} // namespace nav