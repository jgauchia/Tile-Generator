/**
 * @file tile_processor.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Optimized tile generation engine with Pure Hilbert Indexing and NAV-PACK container support.
 * @version 0.6.0
 * @date 2026-03
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
#include <unordered_set>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <geos_c.h>
#include "utils.hpp"
#include "mapped_store.hpp"
#include "constants.hpp"
#include "nav_types.hpp"

namespace nav {

class TileProcessor
{
public:
    TileProcessor(const std::string& out_dir) : output_dir(out_dir) {}

    void process_all(std::vector<size_t> (&features_by_zoom)[18], MappedStore& store,
                     int min_z, int max_z,
                     const std::vector<Feature>& text_features,
                     const std::vector<Feature>& point_features)
    {
        total_generated_bytes = 0;
        total_generated_files = 0;
        for (int z = min_z; z <= max_z; ++z)
        {
            auto placed = resolve_text_labels(text_features, z);
            process_zoom_level(features_by_zoom, store, z, placed, point_features);
        }
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
        { return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1); }
    };
    struct PackedTile { uint32_t x, y; uint64_t h; std::vector<uint8_t> data; };

    struct ProcessedFeature
    {
        uint8_t type;
        uint16_t color;
        uint8_t prio;
        float width;
        std::vector<std::vector<std::pair<int16_t, int16_t>>> rings;
        std::map<int, uint8_t> zoom_widths;
        std::string highway_type;
        std::string layer;
        bool is_bridge = false;
        bool is_building = false;
    };

    struct PlacedLabel
    {
        Feature feature;
        std::vector<TileCoord> tiles;
    };

    // --- Collision detection for text labels ---
    std::vector<PlacedLabel> resolve_text_labels(const std::vector<Feature>& all_text, int zoom)
    {
        std::vector<PlacedLabel> result;
        std::vector<const Feature*> place_names, road_labels;

        for (const auto& f : all_text)
        {
            int min_z = f.zoom_priority >> 4;
            if (min_z > zoom) continue;
            if (f.geom_type != GEOM_TEXT) continue;
            if (f.points.empty()) continue;
            if (f.coords_candidates.empty())
                place_names.push_back(&f);
            else
                road_labels.push_back(&f);
        }

        std::sort(place_names.begin(), place_names.end(),
                  [](const Feature* a, const Feature* b) { return a->population > b->population; });

        double tile_w_deg = 360.0 / std::pow(2.0, zoom);
        double pixel_deg = tile_w_deg / 256.0;
        double char_w = pixel_deg * constants::LABEL_CHAR_WIDTH_PX;
        double label_h = pixel_deg * constants::LABEL_HEIGHT_PX;

        struct Box { double x0, y0, x1, y1; };
        std::vector<Box> placed_boxes;

        auto check_overlap = [&](const Box& b) -> bool {
            for (const auto& pb : placed_boxes)
                if (b.x0 < pb.x1 && b.x1 > pb.x0 && b.y0 < pb.y1 && b.y1 > pb.y0)
                    return true;
            return false;
        };

        auto expand_tiles = [&](double lon, double lat) -> std::vector<TileCoord> {
            int tx = static_cast<int>(utils::lon_to_x(lon, zoom));
            int ty = static_cast<int>(utils::lat_to_y(lat, zoom));
            std::vector<TileCoord> tiles;
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    tiles.push_back({tx + dx, ty + dy});
            return tiles;
        };

        // Place names first (higher pop = placed first)
        for (const auto* pf : place_names)
        {
            double half_w = char_w * pf->text.size() / 2.0;
            double half_h = label_h;
            double lon = pf->points[0].lon, lat = pf->points[0].lat;
            Box box{lon - half_w, lat - half_h, lon + half_w, lat + half_h};
            if (!check_overlap(box))
            {
                placed_boxes.push_back(box);
                result.push_back({*pf, expand_tiles(lon, lat)});
            }
        }

        // Road labels with candidate positions
        for (const auto* rf : road_labels)
        {
            double half_w = char_w * rf->text.size() / 2.0;
            double half_h = label_h;
            for (const auto& cand : rf->coords_candidates)
            {
                Box box{cand.lon - half_w, cand.lat - half_h, cand.lon + half_w, cand.lat + half_h};
                if (!check_overlap(box))
                {
                    placed_boxes.push_back(box);
                    Feature f_copy = *rf;
                    f_copy.points = {cand};
                    result.push_back({std::move(f_copy), expand_tiles(cand.lon, cand.lat)});
                    break;
                }
            }
        }

        return result;
    }

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

    // Convert GEOM_POINT features to polygon symbols for a given zoom/tile
    std::vector<ProcessedFeature> make_point_symbols(const std::vector<Feature>& point_features,
                                                      int z, int tile_x, int tile_y)
    {
        std::vector<ProcessedFeature> result;
        double tile_w_deg = 360.0 / std::pow(2.0, z);
        double pixel_deg = tile_w_deg / 256.0;
        double size = pixel_deg * constants::POINT_SYMBOL_SIZE_PX;

        for (const auto& f : point_features)
        {
            int min_z = f.zoom_priority >> 4;
            if (min_z > z) continue;
            if (f.points.empty()) continue;
            double lon = f.points[0].lon, lat = f.points[0].lat;

            int tx = static_cast<int>(utils::lon_to_x(lon, z));
            int ty = static_cast<int>(utils::lat_to_y(lat, z));
            if (tx != tile_x || ty != tile_y) continue;

            double lat_size = size / std::cos(lat * M_PI / 180.0);

            std::vector<std::pair<int16_t, int16_t>> pts;
            if (f.shape == "triangle")
            {
                double h = lat_size * 0.866;
                double top_lat = lat + h * 0.667;
                double bot_lat = lat - h * 0.333;
                double left_lon = lon - size;
                double right_lon = lon + size;
                auto proj = [&](double lo, double la) -> std::pair<int16_t,int16_t> {
                    return {(int16_t)((utils::lon_to_x(lo, z) - tile_x) * 4096),
                            (int16_t)((utils::lat_to_y(la, z) - tile_y) * 4096)};
                };
                pts.push_back(proj(lon, top_lat));
                pts.push_back(proj(left_lon, bot_lat));
                pts.push_back(proj(right_lon, bot_lat));
                pts.push_back(proj(lon, top_lat));
            }
            else
            {
                double s = pixel_deg;
                double ls = s / std::cos(lat * M_PI / 180.0);
                auto proj = [&](double lo, double la) -> std::pair<int16_t,int16_t> {
                    return {(int16_t)((utils::lon_to_x(lo, z) - tile_x) * 4096),
                            (int16_t)((utils::lat_to_y(la, z) - tile_y) * 4096)};
                };
                pts.push_back(proj(lon - s, lat + ls));
                pts.push_back(proj(lon + s, lat + ls));
                pts.push_back(proj(lon + s, lat - ls));
                pts.push_back(proj(lon - s, lat - ls));
                pts.push_back(proj(lon - s, lat + ls));
            }

            ProcessedFeature pf;
            pf.type = GEOM_POLYGON;
            pf.color = f.color_rgb565;
            pf.prio = f.zoom_priority;
            pf.width = 0;
            pf.rings.push_back(std::move(pts));
            result.push_back(std::move(pf));
        }
        return result;
    }

    // Generate bridge deck underlay polygons (z >= 16)
    std::vector<ProcessedFeature> make_bridge_decks(const std::vector<ProcessedFeature>& line_features,
                                                     int z, int tile_x, int tile_y,
                                                     GEOSContextHandle_t handle)
    {
        if (z < 16) return {};

        static const std::unordered_set<std::string> ROAD_TYPES = {
            "motorway","trunk","primary","secondary","tertiary",
            "motorway_link","trunk_link","primary_link","secondary_link","tertiary_link",
            "residential","unclassified","living_street","pedestrian"
        };
        static const std::unordered_set<std::string> RAIL_TYPES = {
            "rail","narrow_gauge","funicular","tram","light_rail"
        };

        const auto& width_table = constants::line_width_per_zoom();

        std::vector<GEOSGeometry*> tight_buffers;
        std::vector<GEOSGeometry*> generous_buffers;

        for (const auto& pf : line_features)
        {
            if (!pf.is_bridge || pf.type != GEOM_LINESTRING) continue;
            bool is_road = ROAD_TYPES.count(pf.highway_type) > 0;
            bool is_rail = RAIL_TYPES.count(pf.highway_type) > 0;
            if (!is_road && !is_rail) continue;

            if (pf.rings.empty() || pf.rings[0].size() < 2) continue;

            GEOSCoordSequence* seq = GEOSCoordSeq_create_r(handle, (uint32_t)pf.rings[0].size(), 2);
            for (size_t j = 0; j < pf.rings[0].size(); ++j)
            {
                GEOSCoordSeq_setX_r(handle, seq, (uint32_t)j, pf.rings[0][j].first);
                GEOSCoordSeq_setY_r(handle, seq, (uint32_t)j, pf.rings[0][j].second);
            }
            GEOSGeometry* line = GEOSGeom_createLineString_r(handle, seq);
            if (!line) continue;

            int road_width = 1;
            auto wit = width_table.find(pf.highway_type);
            if (wit != width_table.end())
            {
                auto zit = wit->second.lower_bound(z);
                if (zit != wit->second.begin()) { --zit; }
                for (auto it = wit->second.begin(); it != wit->second.end(); ++it)
                    if (z >= it->first) road_width = it->second;
            }

            double rail_extra = is_rail ? 2.0 : 1.0;
            double px_scale = 4096.0 / 256.0; 
            double tight_buf = (road_width * 0.4 + rail_extra) * px_scale;
            double extra = 5.0 * (1 << (z - 16));
            double generous_buf = (road_width / 4.0 + extra) * px_scale;

            GEOSGeometry* tb = GEOSBufferWithStyle_r(handle, line, tight_buf, 4, GEOSBUF_CAP_FLAT, GEOSBUF_JOIN_ROUND, 0.0);
            GEOSGeometry* gb = GEOSBufferWithStyle_r(handle, line, generous_buf, 4, GEOSBUF_CAP_FLAT, GEOSBUF_JOIN_ROUND, 0.0);
            GEOSGeom_destroy_r(handle, line);

            if (tb) tight_buffers.push_back(tb);
            if (gb) generous_buffers.push_back(gb);
        }

        std::vector<ProcessedFeature> result;
        if (tight_buffers.empty()) return result;

        GEOSGeometry* tight_coll = GEOSGeom_createCollection_r(handle, GEOS_GEOMETRYCOLLECTION,
                                                                 tight_buffers.data(), (uint32_t)tight_buffers.size());
        GEOSGeometry* tight_union = GEOSUnaryUnion_r(handle, tight_coll);

        if (tight_union && !GEOSisEmpty_r(handle, tight_union))
        {
            GEOSGeometry* deck = tight_union;
            int type = GEOSGeomTypeId_r(handle, tight_union);
            if (type == GEOS_MULTIPOLYGON || type == GEOS_GEOMETRYCOLLECTION)
            {
                int n = GEOSGetNumGeometries_r(handle, tight_union);
                for (int i = 0; i < n; ++i)
                {
                    const GEOSGeometry* part = GEOSGetGeometryN_r(handle, tight_union, i);
                    if (!part || GEOSisEmpty_r(handle, part) || GEOSGeomTypeId_r(handle, part) != GEOS_POLYGON) continue;
                    extract_deck_polygon(handle, part, result, z, tile_x, tile_y);
                }
            }
            else if (type == GEOS_POLYGON)
            {
                extract_deck_polygon(handle, deck, result, z, tile_x, tile_y);
            }
        }

        GEOSGeom_destroy_r(handle, tight_coll);
        if (tight_union) GEOSGeom_destroy_r(handle, tight_union);
        for (auto* g : generous_buffers) GEOSGeom_destroy_r(handle, g);

        return result;
    }

    void extract_deck_polygon(GEOSContextHandle_t handle, const GEOSGeometry* poly,
                               std::vector<ProcessedFeature>& result, int z, int /*tile_x*/, int /*tile_y*/)
    {
        static const uint16_t BRIDGE_COLOR = utils::hex_to_rgb565("#b8b8b8");
        const GEOSGeometry* ext = GEOSGetExteriorRing_r(handle, poly);
        if (!ext) return;
        const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(handle, ext);
        if (!s) return;
        uint32_t sz; GEOSCoordSeq_getSize_r(handle, s, &sz);
        if (sz < 3) return;

        ProcessedFeature pf;
        pf.type = GEOM_POLYGON;
        pf.color = BRIDGE_COLOR;
        pf.prio = utils::pack_zoom_priority(z, 14);
        pf.width = 0;
        std::vector<std::pair<int16_t, int16_t>> rpts;
        for (uint32_t j = 0; j < sz; ++j)
        {
            double x, y;
            GEOSCoordSeq_getX_r(handle, s, j, &x);
            GEOSCoordSeq_getY_r(handle, s, j, &y);
            rpts.push_back({(int16_t)x, (int16_t)y});
        }
        pf.rings.push_back(std::move(rpts));
        result.push_back(std::move(pf));
    }

    void process_zoom_level(std::vector<size_t> (&features_by_zoom)[18], MappedStore& store, int z,
                            const std::vector<PlacedLabel>& placed_labels,
                            const std::vector<Feature>& point_features)
    {
        auto start = std::chrono::steady_clock::now();
        std::unordered_map<TileCoord, std::vector<size_t>, TileCoordHash> tile_map;
        for (int b = 0; b <= z; ++b)
        {
            const auto& bucket = features_by_zoom[b];
            for (size_t offset : bucket)
            {
                Feature f = store.get(offset);
                if (z == 9 && f.highway_type == "secondary" && f.ref.empty())
                    continue;

                int min_tx = 1e9, max_tx = -1, min_ty = 1e9, max_ty = -1;
                for (const auto& p : f.points)
                {
                    int tx = static_cast<int>(utils::lon_to_x(p.lon, z));
                    int ty = static_cast<int>(utils::lat_to_y(p.lat, z));
                    if (tx < min_tx) min_tx = tx;
                    if (tx > max_tx) max_tx = tx;
                    if (ty < min_ty) min_ty = ty;
                    if (ty > max_ty) max_ty = ty;
                }
                for (int x = min_tx; x <= max_tx; ++x)
                    for (int y = min_ty; y <= max_ty; ++y)
                        tile_map[{x, y}].push_back(offset);
            }
        }

        std::unordered_map<TileCoord, std::vector<size_t>, TileCoordHash> tile_label_indices;
        for (size_t li = 0; li < placed_labels.size(); ++li)
        {
            for (const auto& tc : placed_labels[li].tiles)
                tile_label_indices[tc].push_back(li);
        }

        std::unordered_set<TileCoord, TileCoordHash> all_tile_coords;
        for (const auto& [tc, _] : tile_map) all_tile_coords.insert(tc);
        for (const auto& [tc, _] : tile_label_indices) all_tile_coords.insert(tc);

        if (all_tile_coords.empty()) return;

        struct TileWork
        {
            TileCoord coord;
            std::vector<size_t> offsets;
            std::vector<size_t> label_indices;
        };
        std::vector<TileWork> tiles;
        for (const auto& tc : all_tile_coords)
        {
            TileWork tw;
            tw.coord = tc;
            if (tile_map.count(tc)) tw.offsets = tile_map.at(tc);
            if (tile_label_indices.count(tc)) tw.label_indices = tile_label_indices.at(tc);
            tiles.push_back(std::move(tw));
        }

        std::vector<PackedTile> packed_results(tiles.size());
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        std::atomic<size_t> next_tile(0), merged_count{0};
        std::vector<std::thread> workers;

        for (unsigned int t = 0; t < num_threads; ++t)
        {
            workers.emplace_back([&]() {
                GEOSContextHandle_t handle = GEOS_init_r();
                while (true)
                {
                    size_t idx = next_tile.fetch_add(1);
                    if (idx >= tiles.size()) break;
                    const auto& tw = tiles[idx];
                    size_t m = 0;

                    std::vector<const Feature*> tile_labels;
                    for (size_t li : tw.label_indices)
                        tile_labels.push_back(&placed_labels[li].feature);

                    auto pt_syms = make_point_symbols(point_features, z, tw.coord.x, tw.coord.y);

                    packed_results[idx].x = (uint32_t)tw.coord.x;
                    packed_results[idx].y = (uint32_t)tw.coord.y;
                    packed_results[idx].h = utils::xy_to_hilbert(packed_results[idx].x, packed_results[idx].y, z);
                    packed_results[idx].data = serialize_tile(z, tw.coord.x, tw.coord.y,
                                                              store, tw.offsets, handle, m,
                                                              tile_labels, pt_syms);
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
            std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": ["
                      << std::string(progress * 20, '#') << std::string(20 - progress * 20, '-')
                      << "] " << std::setw(3) << int(progress * 100.0) << "% | "
                      << std::setw(8) << (size_t)next_tile << "/" << tiles.size()
                      << " tiles | " << std::fixed << std::setprecision(1) << tps << " t/s" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        for (auto& w : workers) w.join();

        // 1. Sort by Hilbert index to define physical AND index order
        std::sort(packed_results.begin(), packed_results.end(), [](const PackedTile& a, const PackedTile& b) {
            return a.h < b.h;
        });

        std::string pack_path = output_dir + "/Z" + std::to_string(z) + ".nav";
        std::ofstream out(pack_path, std::ios::binary);
        if (out)
        {
            uint32_t index_offset = sizeof(PackHeader);
            uint32_t data_offset = index_offset + (uint32_t)(packed_results.size() * sizeof(IndexEntry));

            std::map<std::vector<uint8_t>, uint32_t> data_to_offset;
            std::vector<size_t> unique_data_indices;
            uint32_t current_data_offset = data_offset;

            std::vector<IndexEntry> index;
            for (size_t i = 0; i < packed_results.size(); ++i)
            {
                auto& pt = packed_results[i];
                auto it = data_to_offset.find(pt.data);
                uint32_t final_offset;
                if (it != data_to_offset.end())
                {
                    final_offset = it->second;
                }
                else
                {
                    final_offset = current_data_offset;
                    data_to_offset[pt.data] = current_data_offset;
                    unique_data_indices.push_back(i);
                    current_data_offset += (uint32_t)pt.data.size();
                }
                index.push_back({pt.h, final_offset, (uint32_t)pt.data.size()});
            }

            PackHeader ph;
            memcpy(ph.magic, "NPK2", 4);
            ph.zoom = (uint8_t)z;
            ph.tile_count = (uint32_t)index.size();
            ph.index_offset = index_offset;
            memset(ph.reserved, 0, sizeof(ph.reserved));

            out.write((char*)&ph, sizeof(PackHeader));
            out.write((char*)index.data(), index.size() * sizeof(IndexEntry));
            
            for (size_t idx : unique_data_indices)
                out.write((char*)packed_results[idx].data.data(), packed_results[idx].data.size());

            total_generated_bytes += current_data_offset;
            total_generated_files++;
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        double avg_tps = (elapsed.count() > 0) ? (double)tiles.size() / elapsed.count() : 0;
        std::cout << "\r\33[2K  Zoom " << std::setw(2) << z << ": "
                  << std::setw(8) << tiles.size() << " tiles ("
                  << std::fixed << std::setprecision(1) << avg_tps << " t/s), "
                  << std::setw(8) << (size_t)merged_count << " polygons merged | "
                  << std::setw(6) << (total_generated_bytes / 1024 / 1024) << " MB done in " << elapsed.count() << "s" << std::endl;
    }

    std::vector<uint8_t> serialize_tile(int z, int x, int y, const MappedStore& store,
                                         const std::vector<size_t>& offsets,
                                         GEOSContextHandle_t local_handle, size_t& merged_out,
                                         const std::vector<const Feature*>& tile_labels,
                                         const std::vector<ProcessedFeature>& point_symbols)
    {
        struct Style {
            uint16_t color; uint8_t prio; std::string subclass; bool is_building;
            bool operator<(const Style& o) const {
                if (color != o.color) return color < o.color;
                if (prio != o.prio) return prio < o.prio;
                if (subclass != o.subclass) return subclass < o.subclass;
                return is_building < o.is_building;
            }
        };
        std::map<Style, std::vector<GEOSGeometry*>> poly_groups;
        std::map<Style, std::vector<Feature>> poly_individual;
        std::vector<Feature> lines;
        std::vector<GEOSGeometry*> island_geoms;
        const auto& width_table = constants::line_width_per_zoom();
        const auto& color_table = constants::line_color_per_zoom();

        double pre_min_area = constants::min_area_deg2_for_zoom(z);

        for (size_t offset : offsets)
        {
            Feature f = store.get(offset);
            if (f.geom_type == GEOM_POLYGON)
            {
                if (pre_min_area > 0 && !f.points.empty())
                {
                    double area2 = 0;
                    size_t npts = f.ring_ends.empty() ? f.points.size() : f.ring_ends[0];
                    for (size_t i = 0; i < npts; ++i)
                    {
                        size_t j = (i + 1) % npts;
                        area2 += f.points[i].lon * f.points[j].lat - f.points[j].lon * f.points[i].lat;
                    }
                    if (std::abs(area2) * 0.5 < pre_min_area) continue;
                }

                if (f.layer == "islands")
                {
                    GEOSGeometry* ig = feature_to_geos(f, local_handle);
                    if (ig && !GEOSisEmpty_r(local_handle, ig))
                        island_geoms.push_back(ig);
                }

                uint8_t nibble = f.zoom_priority & 0x0F;
                bool is_landcover = nibble <= 3 && (f.layer == "wood" || f.layer == "forest");
                bool is_building_merge = f.is_building && z <= 15;
                Style key{f.color_rgb565, f.zoom_priority, f.layer, f.is_building};
                if (is_landcover || is_building_merge)
                {
                    GEOSGeometry* g = feature_to_geos(f, local_handle);
                    if (g) poly_groups[key].push_back(g);
                }
                else
                {
                    poly_individual[key].push_back(std::move(f));
                }
            }
            else
                lines.push_back(std::move(f));
        }

        GEOSGeometry* island_union = nullptr;
        bool island_geoms_owned = false;
        if (!island_geoms.empty())
        {
            if (island_geoms.size() == 1)
            {
                island_union = GEOSGeom_clone_r(local_handle, island_geoms[0]);
            }
            else
            {
                GEOSGeometry* coll = GEOSGeom_createCollection_r(local_handle, GEOS_GEOMETRYCOLLECTION,
                                                                   island_geoms.data(), (uint32_t)island_geoms.size());
                island_geoms_owned = true;
                island_union = GEOSUnaryUnion_r(local_handle, coll);
                GEOSGeom_destroy_r(local_handle, coll);
            }
            if (island_union && !GEOSisValid_r(local_handle, island_union))
            {
                GEOSGeometry* v = GEOSMakeValid_r(local_handle, island_union);
                GEOSGeom_destroy_r(local_handle, island_union);
                island_union = v;
            }
        }

        double lmin = utils::tile_x_to_lon(x, z), lmax = utils::tile_x_to_lon(x + 1, z);
        double tmax = utils::tile_y_to_lat(y, z), tmin = utils::tile_y_to_lat(y + 1, z);

        auto make_clip_box = [&](double margin) -> GEOSGeometry* {
            double lm = (lmax - lmin) * margin, tm = (tmax - tmin) * margin;
            GEOSCoordSequence* cb = GEOSCoordSeq_create_r(local_handle, 5, 2);
            GEOSCoordSeq_setX_r(local_handle, cb, 0, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cb, 0, tmin - tm);
            GEOSCoordSeq_setX_r(local_handle, cb, 1, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cb, 1, tmin - tm);
            GEOSCoordSeq_setX_r(local_handle, cb, 2, lmax + lm); GEOSCoordSeq_setY_r(local_handle, cb, 2, tmax + tm);
            GEOSCoordSeq_setX_r(local_handle, cb, 3, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cb, 3, tmax + tm);
            GEOSCoordSeq_setX_r(local_handle, cb, 4, lmin - lm); GEOSCoordSeq_setY_r(local_handle, cb, 4, tmin - tm);
            return GEOSGeom_createPolygon_r(local_handle, GEOSGeom_createLinearRing_r(local_handle, cb), NULL, 0);
        };
        GEOSGeometry* clip_poly = make_clip_box(constants::CLIP_MARGIN_POLYGON);
        GEOSGeometry* clip_line = make_clip_box(constants::CLIP_MARGIN_LINE);
        double tolerance = (lmax - lmin) / 1024.0;
        double pixel_deg = (lmax - lmin) / 256.0;
        double min_pixel_area = constants::post_projection_min_area(z);
        double min_hole_area_deg2 = 0.0;
        {
            double hole_factor = (z >= 13) ? 1.5 : 10.0;
            min_hole_area_deg2 = (pixel_deg * pixel_deg) * constants::K_VISIBILITY * hole_factor;
        }
        std::vector<ProcessedFeature> final_features;

        auto project_ring = [&](const GEOSGeometry* ring) -> std::vector<std::pair<int16_t, int16_t>> {
            std::vector<std::pair<int16_t, int16_t>> rpts;
            if (!ring) return rpts;
            const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, ring);
            if (!s) return rpts;
            uint32_t sz = 0; GEOSCoordSeq_getSize_r(local_handle, s, &sz);
            if (sz < 3) return rpts;
            for (uint32_t j = 0; j < sz; ++j)
            {
                double lon, lat;
                if (GEOSCoordSeq_getX_r(local_handle, s, j, &lon) && GEOSCoordSeq_getY_r(local_handle, s, j, &lat))
                    rpts.push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096),
                                   static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
            }
            return rpts;
        };

        auto emit_polygon = [&](const GEOSGeometry* safe, uint16_t color, uint8_t prio,
                                const std::string& feat_layer = "", bool is_building = false) {
            if (!safe || GEOSGeomTypeId_r(local_handle, safe) != GEOS_POLYGON || GEOSisEmpty_r(local_handle, safe)) return;
            bool is_water = ((prio & 0x0F) == 8);

            auto ext_pts = project_ring(GEOSGetExteriorRing_r(local_handle, safe));
            if (ext_pts.size() < 3) return;

            if (!is_water && min_pixel_area > 0)
            {
                int16_t minx = 4096, maxx = 0, miny = 4096, maxy = 0;
                for (const auto& p : ext_pts)
                {
                    int16_t cx = std::max((int16_t)0, std::min((int16_t)4096, p.first));
                    int16_t cy = std::max((int16_t)0, std::min((int16_t)4096, p.second));
                    if (cx < minx) minx = cx;
                    if (cx > maxx) maxx = cx;
                    if (cy < miny) miny = cy;
                    if (cy > maxy) maxy = cy;
                }
                double pixel_area = (double)(maxx - minx) * (maxy - miny) / (16.0 * 16.0);
                if (pixel_area < min_pixel_area) return;
            }

            ProcessedFeature pf{GEOM_POLYGON, color, prio, 0, {}, {}, "", feat_layer, false, is_building};
            pf.rings.push_back(std::move(ext_pts));

            int nh = GEOSGetNumInteriorRings_r(local_handle, safe);
            for (int h = 0; h < nh; ++h)
            {
                const GEOSGeometry* hole = GEOSGetInteriorRingN_r(local_handle, safe, h);
                if (!hole) continue;

                if (!is_water && min_hole_area_deg2 > 0)
                {
                    const GEOSCoordSequence* hs = GEOSGeom_getCoordSeq_r(local_handle, hole);
                    if (hs)
                    {
                        uint32_t hsz = 0; GEOSCoordSeq_getSize_r(local_handle, hs, &hsz);
                        double area2 = 0;
                        for (uint32_t j = 0; j < hsz; ++j)
                        {
                            double x0, y0, x1, y1;
                            GEOSCoordSeq_getX_r(local_handle, hs, j, &x0);
                            GEOSCoordSeq_getY_r(local_handle, hs, j, &y0);
                            GEOSCoordSeq_getX_r(local_handle, hs, (j + 1) % hsz, &x1);
                            GEOSCoordSeq_getY_r(local_handle, hs, (j + 1) % hsz, &y1);
                            area2 += x0 * y1 - x1 * y0;
                        }
                        if (std::abs(area2) * 0.5 < min_hole_area_deg2) continue;
                    }
                }

                auto hole_pts = project_ring(hole);
                if (hole_pts.size() >= 3) pf.rings.push_back(std::move(hole_pts));
            }
            if (!pf.rings.empty()) final_features.push_back(std::move(pf));
        };

        for (auto& [style, geos_polys] : poly_groups)
        {
            GEOSGeometry* coll = GEOSGeom_createCollection_r(local_handle, GEOS_GEOMETRYCOLLECTION, geos_polys.data(), (uint32_t)geos_polys.size());
            GEOSGeometry* merged = GEOSUnaryUnion_r(local_handle, coll);
            GEOSGeometry* final_geom = nullptr;
            if (merged)
            {
                GEOSGeometry* simplified = GEOSTopologyPreserveSimplify_r(local_handle, merged, tolerance * 0.5);
                GEOSGeometry* to_clip = (simplified && !GEOSisEmpty_r(local_handle, simplified)) ? simplified : merged;
                GEOSGeometry* clipped = GEOSIntersection_r(local_handle, to_clip, clip_poly);
                if (to_clip != merged) GEOSGeom_destroy_r(local_handle, to_clip);
                if (clipped)
                {
                    if (!GEOSisValid_r(local_handle, clipped))
                    {
                        final_geom = GEOSMakeValid_r(local_handle, clipped);
                        GEOSGeom_destroy_r(local_handle, clipped);
                    }
                    else final_geom = clipped;
                }
                GEOSGeom_destroy_r(local_handle, merged);
            }
            if (!final_geom) { GEOSGeom_destroy_r(local_handle, coll); continue; }

            uint32_t merged_pts = count_geos_points(local_handle, final_geom);
            if (merged_pts > 65535)
            {
                for (auto* gp : geos_polys) emit_polygon(gp, style.color, style.prio);
                GEOSGeom_destroy_r(local_handle, coll); GEOSGeom_destroy_r(local_handle, final_geom);
                continue;
            }

            int n = GEOSGetNumGeometries_r(local_handle, final_geom);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* g = GEOSGetGeometryN_r(local_handle, final_geom, i);
                if (!g || GEOSGeomTypeId_r(local_handle, g) != GEOS_POLYGON || GEOSisEmpty_r(local_handle, g)) continue;
                emit_polygon(g, style.color, style.prio);
            }
            merged_out += ((int)geos_polys.size() - std::max(0, n));
            GEOSGeom_destroy_r(local_handle, coll); GEOSGeom_destroy_r(local_handle, final_geom);
        }

        for (auto& [style, feats] : poly_individual)
        {
            uint8_t nibble = style.prio & 0x0F;
            bool is_water = (nibble == 8);
            bool is_landuse_low = (nibble <= 3) && z < 14;
            for (auto& f : feats)
            {
                GEOSGeometry* g = feature_to_geos(f, local_handle);
                if (!g) continue;
                GEOSGeometry* clipped = GEOSIntersection_r(local_handle, g, clip_poly);
                GEOSGeom_destroy_r(local_handle, g);
                if (!clipped || GEOSisEmpty_r(local_handle, clipped)) { if (clipped) GEOSGeom_destroy_r(local_handle, clipped); continue; }
                if (!GEOSisValid_r(local_handle, clipped)) { GEOSGeometry* v = GEOSMakeValid_r(local_handle, clipped); GEOSGeom_destroy_r(local_handle, clipped); clipped = v; }
                if (!clipped) continue;

                if (is_water && island_union)
                {
                    GEOSGeometry* diff = GEOSDifference_r(local_handle, clipped, island_union);
                    if (diff && !GEOSisEmpty_r(local_handle, diff))
                    {
                        GEOSGeom_destroy_r(local_handle, clipped);
                        clipped = diff;
                        if (!GEOSisValid_r(local_handle, clipped))
                        {
                            GEOSGeometry* v = GEOSMakeValid_r(local_handle, clipped);
                            GEOSGeom_destroy_r(local_handle, clipped);
                            clipped = v;
                        }
                        if (!clipped) continue;
                    }
                    else
                    {
                        if (diff) GEOSGeom_destroy_r(local_handle, diff);
                    }
                }

                GEOSGeometry* result = clipped;
                if (is_landuse_low)
                {
                    GEOSGeometry* s = GEOSTopologyPreserveSimplify_r(local_handle, clipped, tolerance);
                    if (s && !GEOSisEmpty_r(local_handle, s)) result = s;
                }
                else if (!is_water)
                {
                    GEOSGeometry* s = GEOSTopologyPreserveSimplify_r(local_handle, clipped, tolerance * 0.5);
                    if (s && !GEOSisEmpty_r(local_handle, s)) result = s;
                }
                auto emit_geom = [&](const GEOSGeometry* geom) {
                    if (!geom) return;
                    int type = GEOSGeomTypeId_r(local_handle, geom);
                    if (type == GEOS_POLYGON && !GEOSisEmpty_r(local_handle, geom))
                        emit_polygon(geom, style.color, style.prio, f.layer, f.is_building);
                    else if (type == GEOS_MULTIPOLYGON || type == GEOS_GEOMETRYCOLLECTION)
                    {
                        int np = GEOSGetNumGeometries_r(local_handle, geom);
                        for (int i = 0; i < np; ++i)
                        {
                            const GEOSGeometry* part = GEOSGetGeometryN_r(local_handle, geom, i);
                            if (part && GEOSGeomTypeId_r(local_handle, part) == GEOS_POLYGON && !GEOSisEmpty_r(local_handle, part))
                                emit_polygon(part, style.color, style.prio, f.layer, f.is_building);
                        }
                    }
                };
                emit_geom(result);
                if (result != clipped) GEOSGeom_destroy_r(local_handle, result);
                GEOSGeom_destroy_r(local_handle, clipped);
            }
        }

        for (const auto& f : lines)
        {
            GEOSGeometry* g = feature_to_geos(f, local_handle);
            if (!g) continue;
            GEOSGeometry* clipped = GEOSIntersection_r(local_handle, g, clip_line);
            if (!clipped || GEOSisEmpty_r(local_handle, clipped))
            {
                GEOSGeom_destroy_r(local_handle, g);
                if (clipped) GEOSGeom_destroy_r(local_handle, clipped);
                continue;
            }

            uint16_t line_color = f.color_rgb565;
            if (!f.highway_type.empty())
            {
                auto cit = color_table.find(f.highway_type);
                if (cit != color_table.end())
                {
                    auto zit = cit->second.find(z);
                    if (zit != cit->second.end())
                        line_color = utils::hex_to_rgb565(zit->second);
                }
            }

            bool skip_simplify = (f.layer == "water" || f.layer == "roads");
            double line_tol = tolerance;
            if (f.layer == "infrastructure") line_tol = pixel_deg * 0.1;

            int n = GEOSGetNumGeometries_r(local_handle, clipped);
            for (int i = 0; i < n; ++i)
            {
                const GEOSGeometry* part = GEOSGetGeometryN_r(local_handle, clipped, i);
                if (!part || (GEOSGeomTypeId_r(local_handle, part) != GEOS_LINESTRING && GEOSGeomTypeId_r(local_handle, part) != GEOS_LINEARRING) || GEOSisEmpty_r(local_handle, part)) continue;

                const GEOSGeometry* use = part;
                GEOSGeometry* simplified = nullptr;
                if (!skip_simplify)
                {
                    simplified = GEOSTopologyPreserveSimplify_r(local_handle, part, line_tol);
                    if (simplified && !GEOSisEmpty_r(local_handle, simplified)) use = simplified;
                    else { if (simplified) GEOSGeom_destroy_r(local_handle, simplified); simplified = nullptr; }
                }

                if (GEOSGeomTypeId_r(local_handle, use) != GEOS_LINESTRING && GEOSGeomTypeId_r(local_handle, use) != GEOS_LINEARRING)
                { if (simplified) GEOSGeom_destroy_r(local_handle, simplified); continue; }

                const GEOSCoordSequence* s = GEOSGeom_getCoordSeq_r(local_handle, use);
                uint32_t sz = 0; if (s) GEOSCoordSeq_getSize_r(local_handle, s, &sz);
                if (sz >= 2)
                {
                    ProcessedFeature pf{GEOM_LINESTRING, line_color, f.zoom_priority, f.width_meters,
                                       {{}}, f.zoom_widths, f.highway_type, f.layer, f.is_bridge, f.is_building};
                    for (uint32_t j = 0; j < sz; ++j)
                    {
                        double lon, lat;
                        if (GEOSCoordSeq_getX_r(local_handle, s, j, &lon) && GEOSCoordSeq_getY_r(local_handle, s, j, &lat))
                            pf.rings[0].push_back({static_cast<int16_t>((utils::lon_to_x(lon, z) - x) * 4096),
                                                   static_cast<int16_t>((utils::lat_to_y(lat, z) - y) * 4096)});
                    }
                    if (pf.rings[0].size() >= 2) final_features.push_back(std::move(pf));
                }
                if (simplified) GEOSGeom_destroy_r(local_handle, simplified);
            }
            GEOSGeom_destroy_r(local_handle, g); GEOSGeom_destroy_r(local_handle, clipped);
        }

        auto bridge_decks = make_bridge_decks(final_features, z, x, y, local_handle);
        final_features.insert(final_features.begin(), bridge_decks.begin(), bridge_decks.end());

        for (auto& ps : point_symbols)
            final_features.push_back(ps);

        std::sort(final_features.begin(), final_features.end(), [](const ProcessedFeature& a, const ProcessedFeature& b) {
            return (a.prio & 0x0F) < (b.prio & 0x0F);
        });

        {
            static const uint16_t LAND_BG = utils::hex_to_rgb565(constants::LAND_BG_COLOR);
            ProcessedFeature bg{GEOM_POLYGON, LAND_BG, utils::pack_zoom_priority(0, 0), 0, {}, {}, "", "", false, false};
            bg.rings.push_back({{0, 0}, {4096, 0}, {4096, 4096}, {0, 4096}, {0, 0}});
            final_features.insert(final_features.begin(), std::move(bg));
        }

        std::vector<uint8_t> buffer;
        double t_max_merc = utils::lat_to_mercator_y(tmax);
        double t_min_merc = utils::lat_to_mercator_y(tmin);
        double merc_range = t_max_merc - t_min_merc;

        uint16_t count = (uint16_t)std::min((size_t)0xFFFF, final_features.size() + tile_labels.size());
        uint8_t th[22] = {'N', 'A', 'V', '1', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        memcpy(th + 4, &count, 2);
        buffer.insert(buffer.end(), th, th + 22);

        uint16_t written = 0;

        for (const auto& pf : final_features)
        {
            if (written >= 0xFFFE) break;

            uint8_t final_width = 1;
            if (pf.type == GEOM_LINESTRING)
            {
                if (!pf.highway_type.empty())
                {
                    auto wit = width_table.find(pf.highway_type);
                    if (wit != width_table.end())
                    {
                        for (auto const& [w_zoom, w_px] : wit->second)
                            if (z >= w_zoom) final_width = w_px;
                    }
                    else if (pf.width > 0)
                        final_width = utils::meters_to_pixels(pf.width, z);
                }
                else if (!pf.zoom_widths.empty())
                {
                    for (auto const& [w_zoom, w_pixels] : pf.zoom_widths)
                        if (z >= w_zoom) final_width = w_pixels;
                }
                else if (pf.width > 0)
                    final_width = utils::meters_to_pixels(pf.width, z);
            }
            if (final_width == 0) final_width = 1;

            uint8_t width_byte = std::min((int)final_width, 127);
            if (pf.type == GEOM_POLYGON)
            {
                width_byte = 0;
                if (pf.is_building && z >= 16)
                    width_byte |= 0x80;
            }
            else if (pf.is_bridge && z >= 14)
            {
                width_byte |= 0x80;
            }

            std::vector<uint8_t> payload;
            int16_t minx = 4096, maxx = 0, miny = 4096, maxy = 0;
            size_t pts = 0;
            int16_t lx = 0, ly = 0;
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
                    if (cx < minx) minx = cx;
                    if (cx > maxx) maxx = cx;
                    if (cy < miny) miny = cy;
                    if (cy > maxy) maxy = cy;
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
            uint8_t fh[13] = {pf.type, (uint8_t)(pf.color & 0xFF), (uint8_t)(pf.color >> 8), pf.prio, width_byte,
                              (uint8_t)std::min(255, minx / 16), (uint8_t)std::min(255, miny / 16),
                              (uint8_t)std::min(255, maxx / 16), (uint8_t)std::min(255, maxy / 16),
                              (uint8_t)(pts & 0xFF), (uint8_t)(pts >> 8),
                              (uint8_t)((payload.size() + ri.size()) & 0xFF), (uint8_t)((payload.size() + ri.size()) >> 8)};
            buffer.insert(buffer.end(), fh, fh + 13);
            buffer.insert(buffer.end(), payload.begin(), payload.end());
            if (!ri.empty()) buffer.insert(buffer.end(), ri.begin(), ri.end());
            written++;
        }

        for (const auto* label : tile_labels)
        {
            if (written >= 0xFFFE) break;

            double lon = label->points[0].lon, lat = label->points[0].lat;
            int16_t px = (int16_t)((lon - lmin) / (lmax - lmin) * 4096);
            double m_y = utils::lat_to_mercator_y(lat);
            int16_t py = (int16_t)((t_max_merc - m_y) / merc_range * 4096);

            if (px < -8192 || px > 12288 || py < -8192 || py > 12288) continue;

            const auto& text_bytes = label->text;
            uint8_t text_len = (uint8_t)text_bytes.size();
            bool has_shield = (label->bg_color_rgb565 != 0);
            int data_size = 4 + 1 + text_len + (has_shield ? 4 : 0);
            uint16_t coord_count = (data_size + 3) / 4;
            int padded_size = coord_count * 4;

            std::vector<uint8_t> text_payload;
            text_payload.push_back(px & 0xFF); text_payload.push_back(px >> 8);
            text_payload.push_back(py & 0xFF); text_payload.push_back(py >> 8);
            text_payload.push_back(text_len);
            text_payload.insert(text_payload.end(), text_bytes.begin(), text_bytes.end());
            if (has_shield)
            {
                text_payload.push_back(label->bg_color_rgb565 & 0xFF);
                text_payload.push_back(label->bg_color_rgb565 >> 8);
                text_payload.push_back(label->border_color_rgb565 & 0xFF);
                text_payload.push_back(label->border_color_rgb565 >> 8);
            }
            int padding = padded_size - data_size;
            for (int p = 0; p < padding; ++p) text_payload.push_back(0);

            uint8_t bx = (uint8_t)std::max(0, std::min(255, (int)px >> 4));
            uint8_t by = (uint8_t)std::max(0, std::min(255, (int)py >> 4));

            uint8_t th2[13] = {GEOM_TEXT, (uint8_t)(label->color_rgb565 & 0xFF), (uint8_t)(label->color_rgb565 >> 8),
                              label->zoom_priority, label->font_size,
                              bx, by, bx, by,
                              (uint8_t)(coord_count & 0xFF), (uint8_t)(coord_count >> 8),
                              (uint8_t)(text_payload.size() & 0xFF), (uint8_t)(text_payload.size() >> 8)};
            buffer.insert(buffer.end(), th2, th2 + 13);
            buffer.insert(buffer.end(), text_payload.begin(), text_payload.end());
            written++;
        }

        memcpy(buffer.data() + 4, &written, 2);

        GEOSGeom_destroy_r(local_handle, clip_poly); GEOSGeom_destroy_r(local_handle, clip_line);
        if (!island_geoms_owned)
            for (auto* ig : island_geoms) GEOSGeom_destroy_r(local_handle, ig);
        if (island_union) GEOSGeom_destroy_r(local_handle, island_union);
        return buffer;
    }
};

} // namespace nav
