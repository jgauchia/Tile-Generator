/**
 * @file tile_processor.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Optimized tile generation engine with NAV-PACK container support.
 * @version 0.4.0
 * @date 2026-02
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
#include "utils.hpp"
#include "mapped_store.hpp"

namespace nav {

/**
 * @class TileProcessor
 * @brief Handles spatial grouping and conversion to packed binary NAV containers.
 */
class TileProcessor
{
public:
    TileProcessor(const std::string& out_dir) : output_dir(out_dir) {}

    /**
     * @brief Processes all zoom levels and generates Zxx.nav pack files.
     */
    void process_all(std::vector<size_t> (&features_by_zoom)[18], MappedStore& store, int min_z, int max_z)
    {
        total_generated_bytes = 0;
        total_generated_files = 0;
        for (int z = min_z; z <= max_z; ++z)
            process_zoom_level(features_by_zoom, store, z);
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
            return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
        }
    };

    struct PackedTile
    {
        uint32_t x, y;
        std::vector<uint8_t> data;
    };

    struct ProcessedFeature
    {
        uint8_t type;
        uint16_t color;
        uint8_t prio;
        float width;
        std::vector<std::vector<std::pair<int16_t, int16_t>>> rings;
        std::map<int, uint8_t> zoom_widths;
    };

    uint32_t count_geos_points(GEOSContextHandle_t handle, const GEOSGeometry* g)
    {
        if (!g || GEOSisEmpty_r(handle, g)) return 0;
        int type = GEOSGeomTypeId_r(handle, g);
        if (type == GEOS_POINT) return 1;
        if (type == GEOS_LINESTRING || type == GEOS_LINEARRING)
        {
            uint32_t count = 0;
            const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(handle, g);
            if (s) GEOSCoordSeq_getSize_r(handle, s, &count);
            return count;
        }
        if (type == GEOS_POLYGON)
        {
            const GEOSGeometry* ext = GEOSGetExteriorRing_r(handle, g);
            uint32_t count = ext ? count_geos_points(handle, ext) : 0;
            int nh = GEOSGetNumInteriorRings_r(handle, g);
            for (int h = 0; h < nh; ++h)
                count += count_geos_points(handle, GEOSGetInteriorRingN_r(handle, g, h));
            return count;
        }
        if (type == GEOS_MULTIPOLYGON || type == GEOS_MULTILINESTRING || type == GEOS_GEOMETRYCOLLECTION)
        {
            uint32_t count = 0;
            int n = GEOSGetNumGeometries_r(handle, g);
            for (int i = 0; i < n; ++i)
                count += count_geos_points(handle, GEOSGetGeometryN_r(handle, g, i));
            return count;
        }
        return 0;
    }

    GEOSGeometry* feature_to_geos(const Feature& f, GEOSContextHandle_t handle)
    {
        if (f.ring_ends.empty()) return nullptr;
        if (f.geom_type == GEOM_POLYGON)
        {
            uint32_t end = f.ring_ends[0];
            if (end < 3) return nullptr;
            GEOSCoordSequence* seq = GEOSCoordSeq_create_r(handle, end, 2);
            for (uint32_t i = 0; i < end; ++i)
            {
                GEOSCoordSeq_setX_r(handle, seq, i, f.points[i].lon);
                GEOSCoordSeq_setY_r(handle, seq, i, f.points[i].lat);
            }
            GEOSGeometry* shell = GEOSGeom_createLinearRing_r(handle, seq);
            if (!shell) { GEOSCoordSeq_destroy_r(handle, seq); return nullptr; }
            std::vector<GEOSGeometry*> holes;
            for (size_t r = 1; r < f.ring_ends.size(); ++r)
            {
                uint32_t start = f.ring_ends[r - 1], r_end = f.ring_ends[r], r_size = r_end - start;
                if (r_size < 3) continue;
                GEOSCoordSequence* hseq = GEOSCoordSeq_create_r(handle, r_size, 2);
                for (uint32_t j = 0; j < r_size; ++j)
                {
                    GEOSCoordSeq_setX_r(handle, hseq, j, f.points[start + j].lon);
                    GEOSCoordSeq_setY_r(handle, hseq, j, f.points[start + j].lat);
                }
                GEOSGeometry* hole = GEOSGeom_createLinearRing_r(handle, hseq);
                if (hole) holes.push_back(hole); else GEOSCoordSeq_destroy_r(handle, hseq);
            }
            GEOSGeometry* poly = GEOSGeom_createPolygon_r(handle, shell, holes.data(), (uint32_t)holes.size());
            if (!poly) { GEOSGeom_destroy_r(handle, shell); for (auto h : holes) GEOSGeom_destroy_r(handle, h); return nullptr; }
            return poly;
        }
        else
        {
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

    void process_zoom_level(std::vector<size_t> (&features_by_zoom)[18], MappedStore& store, int z)
    {
        auto start = std::chrono::steady_clock::now();
        std::unordered_map<TileCoord, std::vector<size_t>, TileCoordHash> tile_map;
        for (int b = 0; b <= z; ++b)
        {
            const auto& bucket = features_by_zoom[b];
            for (size_t offset : bucket)
            {
                Feature f = store.get(offset);
                int min_tx = 1e9, max_tx = -1, min_ty = 1e9, max_ty = -1;
                for (const auto& p : f.points)
                {
                    int tx = static_cast<int>(utils::lon_to_x(p.lon, z)), ty = static_cast<int>(utils::lat_to_y(p.lat, z));
                    if (tx < min_tx)
                        min_tx = tx;
                    if (tx > max_tx)
                        max_tx = tx;
                    if (ty < min_ty)
                        min_ty = ty;
                    if (ty > max_ty)
                        max_ty = ty;
                }
                for (int x = min_tx; x <= max_tx; ++x)
                    for (int y = min_ty; y <= max_ty; ++y)
                        tile_map[{x, y}].push_back(offset);
            }
        }
        if (tile_map.empty())
            return;
        std::vector<std::pair<TileCoord, std::vector<size_t>>> tiles(tile_map.begin(), tile_map.end());
        std::vector<PackedTile> packed_results(tiles.size());
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;
        std::atomic<size_t> next_tile(0), merged_count{0};
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
                    const auto& [coord, offsets] = tiles[idx];
                    packed_results[idx].x = (uint32_t)coord.x;
                    packed_results[idx].y = (uint32_t)coord.y;
                    packed_results[idx].data = serialize_tile(z, coord.x, coord.y, store, offsets, handle, m);
                    merged_count += m;
                }
                GEOS_finish_r(handle);
            });
        }
        while (next_tile < tiles.size())
        {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> zoom_elapsed = now - start;
            float progress = (float)next_tile / tiles.size();
            double tps = (zoom_elapsed.count() > 0.1) ? (double)next_tile / zoom_elapsed.count() : 0;
            std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": [" << std::string(progress * 20, '#') << std::string(20 - progress * 20, '-') << "] " << std::setw(3) << int(progress * 100.0) << "% | " << std::setw(8) << (size_t)next_tile << "/" << tiles.size() << " tiles | " << std::fixed << std::setprecision(1) << tps << " t/s" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        for (auto& w : workers)
            w.join();
        std::sort(packed_results.begin(), packed_results.end(), [](const PackedTile& a, const PackedTile& b) {
            if (a.y != b.y)
                return a.y < b.y;
            return a.x < b.x;
        });
        std::string pack_path = output_dir + "/Z" + std::to_string(z) + ".nav";
        std::ofstream out(pack_path, std::ios::binary);
        if (out)
        {
            PackHeader ph;
            memcpy(ph.magic, "NPK1", 4);
            ph.zoom = (uint8_t)z;
            ph.tile_count = (uint32_t)packed_results.size();
            out.write((char*)&ph, sizeof(PackHeader));
            std::vector<IndexEntry> index(packed_results.size());
            uint32_t current_offset = sizeof(PackHeader) + (uint32_t)(index.size() * sizeof(IndexEntry));
            for (size_t i = 0; i < packed_results.size(); ++i)
            {
                index[i].x = packed_results[i].x;
                index[i].y = packed_results[i].y;
                index[i].offset = current_offset;
                index[i].size = (uint32_t)packed_results[i].data.size();
                current_offset += index[i].size;
            }
            out.write((char*)index.data(), index.size() * sizeof(IndexEntry));
            for (const auto& pt : packed_results)
                out.write((char*)pt.data.data(), pt.data.size());
            total_generated_bytes += current_offset;
            total_generated_files++;
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_tps = (elapsed.count() > 0) ? (double)tiles.size() / elapsed.count() : 0;
        std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": " << std::setw(8) << tiles.size() << " tiles (" << std::fixed << std::setprecision(1) << avg_tps << " t/s), " << std::setw(8) << (size_t)merged_count << " polygons merged | ";
        if (out.tellp() < 1024 * 1024)
            std::cout << std::fixed << std::setprecision(2) << ((double)out.tellp() / 1024.0) << " KB";
        else
            std::cout << std::fixed << std::setprecision(2) << ((double)out.tellp() / 1024.0 / 1024.0) << " MB";
        std::cout << " done in " << elapsed.count() << "s" << std::endl;
    }

    /**
     * @brief Serializes a single tile into a binary buffer.
     */
    std::vector<uint8_t> serialize_tile(int z, int x, int y, const MappedStore& store, const std::vector<size_t>& offsets, GEOSContextHandle_t local_handle, size_t& merged_out)
    {
        struct Style { uint16_t color; uint8_t prio; bool operator<(const Style& o) const { return color < o.color || (color == o.color && prio < o.prio); } };
        std::map<Style, std::vector<GEOSGeometry*>> poly_groups;
        std::vector<Feature> lines;
        for (size_t offset : offsets)
        {
            Feature f = store.get(offset);
            if (f.geom_type == GEOM_POLYGON)
            {
                GEOSGeometry* g = feature_to_geos(f, local_handle);
                if (g) poly_groups[{f.color_rgb565, f.zoom_priority}].push_back(g);
            }
            else lines.push_back(std::move(f));
        }
        double lmin = utils::tile_x_to_lon(x, z), lmax = utils::tile_x_to_lon(x + 1, z), tmax = utils::tile_y_to_lat(y, z), tmin = utils::tile_y_to_lat(y + 1, z);
        double lm = (lmax - lmin) * 0.2, tm = (tmax - tmin) * 0.2;
        GEOSCoordSequence* cbox = GEOSCoordSeq_create_r(local_handle, 5, 2);
        GEOSCoordSeq_setX_r(local_handle, cbox, 0, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 0, tmin - tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 1, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cbox, 1, tmin - tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 2, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cbox, 2, tmax + tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 3, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 3, tmax + tm);
        GEOSCoordSeq_setX_r(local_handle, cbox, 4, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cbox, 4, tmin - tm);
        GEOSGeometry* clip_geom = GEOSGeom_createPolygon_r(local_handle, GEOSGeom_createLinearRing_r(local_handle, cbox), NULL, 0);
        double px_size = (lmax - lmin) / 256.0;
        double z_factor = 0.25;
        if (z <= 10) z_factor = 1.0;
        else if (z <= 13) z_factor = 0.5;
        else if (z >= 16) z_factor = 0.1;
        double tolerance = px_size * z_factor;
        double min_poly_area = (lmax - lmin) * (tmax - tmin) / 4096.0;
        double min_line_len = (lmax - lmin) / 128.0;
        const size_t MAX_SAFE_POINTS = 2000;
        std::vector<ProcessedFeature> final_features;
        auto protect_esp32 = [&](GEOSGeometry* g) {
            if (!g || GEOSisEmpty_r(local_handle, g)) return (GEOSGeometry*)nullptr;
            GEOSGeometry* current = g;
            uint32_t pts = count_geos_points(local_handle, current);
            double current_tol = tolerance;
            while (pts > MAX_SAFE_POINTS && current_tol < (tolerance * 20))
            {
                current_tol *= 1.5;
                GEOSGeometry* temp = GEOSTopologyPreserveSimplify_r(local_handle, current, current_tol);
                if (temp) { if (current != g) GEOSGeom_destroy_r(local_handle, current); current = temp; pts = count_geos_points(local_handle, current); }
                else break;
            }
            return current;
        };
        for (auto& [style, geos_polys] : poly_groups)
        {
            size_t before = geos_polys.size();
            GEOSGeometry* coll = GEOSGeom_createCollection_r(local_handle, GEOS_GEOMETRYCOLLECTION, geos_polys.data(), (uint32_t)geos_polys.size());
            GEOSGeometry* merged = GEOSUnaryUnion_r(local_handle, coll);
            GEOSGeometry* final_geom = nullptr;
            if (merged)
            {
                GEOSGeometry* clipped = GEOSIntersection_r(local_handle, merged, clip_geom);
                if (clipped)
                {
                    if (!GEOSisValid_r(local_handle, clipped)) { final_geom = GEOSMakeValid_r(local_handle, clipped); GEOSGeom_destroy_r(local_handle, clipped); }
                    else final_geom = clipped;
                }
                GEOSGeom_destroy_r(local_handle, merged);
            }
            if (!final_geom) { GEOSGeom_destroy_r(local_handle, coll); continue; }
            int n = GEOSGetNumGeometries_r(local_handle, final_geom);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* g = GEOSGetGeometryN_r(local_handle, final_geom, i);
                if (!g || GEOSGeomTypeId_r(local_handle, g) != GEOS_POLYGON || GEOSisEmpty_r(local_handle, g)) continue;
                double area; GEOSArea_r(local_handle, g, &area);
                if (area < min_poly_area) continue;
                GEOSGeometry* simplified = GEOSTopologyPreserveSimplify_r(local_handle, g, tolerance);
                if (!simplified) continue;
                GEOSGeometry* safe = protect_esp32(simplified);
                if (!safe || GEOSGeomTypeId_r(local_handle, safe) != GEOS_POLYGON || GEOSisEmpty_r(local_handle, safe))
                {
                    if (safe && safe != simplified) GEOSGeom_destroy_r(local_handle, safe);
                    GEOSGeom_destroy_r(local_handle, simplified);
                    continue;
                }
                ProcessedFeature pf{GEOM_POLYGON, style.color, style.prio, 0, {}, {}};
                auto process_ring = [&](const GEOSGeometry* ring) {
                    if (!ring) return;
                    const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, ring);
                    if (!s) return;
                    uint32_t sz; GEOSCoordSeq_getSize_r(local_handle, s, &sz);
                    if (sz < 3) return;
                    std::vector<std::pair<int16_t, int16_t>> rpts;
                    for (uint32_t j = 0; j < sz; ++j)
                    {
                        double lon, lat; if (GEOSCoordSeq_getX_r(local_handle, s, j, &lon) && GEOSCoordSeq_getY_r(local_handle, s, j, &lat))
                            rpts.push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096), static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
                    }
                    if (rpts.size() >= 3) pf.rings.push_back(std::move(rpts));
                };
                process_ring(GEOSGetExteriorRing_r(local_handle, safe));
                int nh = GEOSGetNumInteriorRings_r(local_handle, safe);
                for (int h = 0; h < nh; ++h) process_ring(GEOSGetInteriorRingN_r(local_handle, safe, h));
                if (!pf.rings.empty()) final_features.push_back(std::move(pf));
                if (safe != simplified) GEOSGeom_destroy_r(local_handle, safe);
                GEOSGeom_destroy_r(local_handle, simplified);
            }
            merged_out += (before - std::max((int)0, n));
            GEOSGeom_destroy_r(local_handle, coll); GEOSGeom_destroy_r(local_handle, final_geom);
        }
        for (const auto& f : lines)
        {
            GEOSGeometry* g = feature_to_geos(f, local_handle);
            if (!g) continue;
            GEOSGeometry* clipped = GEOSIntersection_r(local_handle, g, clip_geom);
            if (!clipped || GEOSisEmpty_r(local_handle, clipped))
            {
                GEOSGeom_destroy_r(local_handle, g);
                if (clipped) GEOSGeom_destroy_r(local_handle, clipped);
                continue;
            }
            int n = GEOSGetNumGeometries_r(local_handle, clipped);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* part = GEOSGetGeometryN_r(local_handle, clipped, i);
                if (!part || (GEOSGeomTypeId_r(local_handle, part) != GEOS_LINESTRING && GEOSGeomTypeId_r(local_handle, part) != GEOS_LINEARRING) || GEOSisEmpty_r(local_handle, part)) continue;
                double len; GEOSLength_r(local_handle, part, &len);
                if (len < (min_line_len / 2.0) && f.zoom_priority < 7) continue;
                GEOSGeometry* simplified = GEOSTopologyPreserveSimplify_r(local_handle, part, tolerance);
                if (!simplified) continue;
                GEOSGeometry* safe = protect_esp32(simplified);
                if (!safe || (GEOSGeomTypeId_r(local_handle, safe) != GEOS_LINESTRING && GEOSGeomTypeId_r(local_handle, safe) != GEOS_LINEARRING) || GEOSisEmpty_r(local_handle, safe))
                {
                    if (safe && safe != simplified) GEOSGeom_destroy_r(local_handle, safe);
                    GEOSGeom_destroy_r(local_handle, simplified);
                    continue;
                }
                const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, safe);
                uint32_t sz; if (s) GEOSCoordSeq_getSize_r(local_handle, s, &sz); else sz = 0;
                if (sz >= 2)
                {
                    ProcessedFeature pf{GEOM_LINESTRING, f.color_rgb565, f.zoom_priority, f.width_meters, {{}}, f.zoom_widths};
                    for (uint32_t j = 0; j < sz; ++j)
                    {
                        double lon, lat; if (GEOSCoordSeq_getX_r(local_handle, s, j, &lon) && GEOSCoordSeq_getY_r(local_handle, s, j, &lat))
                            pf.rings[0].push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096), static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
                    }
                    if (pf.rings[0].size() >= 2) final_features.push_back(std::move(pf));
                }
                if (safe != simplified) GEOSGeom_destroy_r(local_handle, safe);
                GEOSGeom_destroy_r(local_handle, simplified);
            }
            GEOSGeom_destroy_r(local_handle, g); GEOSGeom_destroy_r(local_handle, clipped);
        }
        std::vector<uint8_t> buffer;
        uint16_t count = (uint16_t)std::min((size_t)0xFFFF, final_features.size());
        uint8_t th[22] = {'N', 'A', 'V', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int32_t b_lmin = (int32_t)(lmin * 1e7), b_tmin = (int32_t)(tmin * 1e7), b_lmax = (int32_t)(lmax * 1e7), b_tmax = (int32_t)(tmax * 1e7);
        memcpy(th + 4, &count, 2); memcpy(th + 6, &b_lmin, 4); memcpy(th + 10, &b_tmin, 4); memcpy(th + 14, &b_lmax, 4); memcpy(th + 18, &b_tmax, 4);
        buffer.insert(buffer.end(), th, th + 22);
        for (uint16_t i = 0; i < count; i++)
        {
            const auto& pf = final_features[i];
            uint8_t final_width = 1;
            if (!pf.zoom_widths.empty())
            {
                for (auto const& [w_zoom, w_pixels] : pf.zoom_widths)
                    if (z >= w_zoom) final_width = w_pixels;
            }
            if (final_width == 0) final_width = 1;
            if (final_width > 15) final_width = 15;
            std::vector<uint8_t> payload; int16_t minx = 4096, maxx = 0, miny = 4096, maxy = 0; size_t pts = 0;
            for (const auto& r : pf.rings)
            {
                int16_t lx = 0, ly = 0;
                for (const auto& p : r)
                {
                    auto vx = utils::to_varint(utils::zigzag_encode(p.first - lx));
                    payload.insert(payload.end(), vx.begin(), vx.end());
                    auto vy = utils::to_varint(utils::zigzag_encode(p.second - ly));
                    payload.insert(payload.end(), vy.begin(), vy.end());
                    lx = p.first; ly = p.second;
                    int16_t cx = std::max((int16_t)0, std::min((int16_t)4096, p.first)), cy = std::max((int16_t)0, std::min((int16_t)4096, p.second));
                    if (cx < minx) minx = cx;
                    if (cx > maxx) maxx = cx;
                    if (cy < miny) miny = cy;
                    if (cy > maxy) maxy = cy;
                }
                pts += r.size();
            }
            std::vector<uint8_t> ri; if (pf.type == GEOM_POLYGON)
            {
                uint16_t nr = (uint16_t)pf.rings.size(); ri.push_back(nr & 0xFF); ri.push_back(nr >> 8);
                uint16_t cur = 0; for (const auto& r : pf.rings) { cur += (uint16_t)r.size(); ri.push_back(cur & 0xFF); ri.push_back(cur >> 8); }
            }
            uint8_t fh[13] = {pf.type, (uint8_t)(pf.color & 0xFF), (uint8_t)(pf.color >> 8), pf.prio, final_width, (uint8_t)std::min(255, minx / 16), (uint8_t)std::min(255, miny / 16), (uint8_t)std::min(255, maxx / 16), (uint8_t)std::min(255, maxy / 16), (uint8_t)(pts & 0xFF), (uint8_t)(pts >> 8), (uint8_t)((payload.size() + ri.size()) & 0xFF), (uint8_t)((payload.size() + ri.size()) >> 8)};
            buffer.insert(buffer.end(), fh, fh + 13); buffer.insert(buffer.end(), payload.begin(), payload.end()); if (!ri.empty()) buffer.insert(buffer.end(), ri.begin(), ri.end());
        }
        GEOSGeom_destroy_r(local_handle, clip_geom);
        return buffer;
    }
};

} // namespace nav
