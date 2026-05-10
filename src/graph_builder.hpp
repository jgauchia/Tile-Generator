/**
 * @file graph_builder.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Builds ROUTE.bin routing graph files from road features extracted by OSMHandler.
 * @version 0.6.0
 * @date 2026-05
 */

#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <filesystem>
#include "nav_types.hpp"

namespace nav {

#pragma pack(push, 1)
// File header — 32 bytes, version=2
struct RouteFileHeader
{
    char     magic[4];       // "ROUT"
    uint8_t  version;        // 2
    uint8_t  reserved[3];
    uint32_t cell_count;
    uint32_t reserved2[4];   // padding to 32B
};

// Per-cell index entry — 24 bytes
struct CellIndexEntry
{
    int16_t  lat_floor;
    int16_t  lon_floor;
    uint32_t node_offset;    // index of first node in flat node array (= global node base)
    uint32_t node_count;
    uint32_t edge_offset;    // index of first edge in flat edge array
    uint32_t edge_count;
    uint32_t reserved;       // padding to 24B
};

struct RouteNode
{
    float    lat;
    float    lon;
    uint32_t edge_offset;    // index into this cell's edge block (relative to cell edge_offset)
};

struct RouteEdge
{
    uint32_t dst_node;       // index into this cell's node block (relative to cell node_offset)
    uint32_t cost;
    uint16_t dist_m;
    uint8_t  flags;
    uint16_t name_idx;
    uint8_t  reserved;
};
#pragma pack(pop)

class GraphBuilder
{
public:
    explicit GraphBuilder(const std::string& output_dir) : output_dir_(output_dir) {}

    void add_way(const Feature& f)
    {
        if (f.osm_node_ids.size() < 2) return;
        WayData wd;
        wd.node_ids  = f.osm_node_ids;
        wd.coords    = f.points;
        wd.hw_class  = classify(f.highway_type);
        wd.oneway    = f.oneway;
        wd.maxspeed  = f.maxspeed;
        wd.name      = f.name;
        ways_.push_back(std::move(wd));

        for (int64_t id : ways_.back().node_ids)
            node_ref_count_[id]++;
    }

    void build_and_write()
    {
        if (ways_.empty()) return;

        // Pass 1: collect segments per cell (each segment assigned to cell of src node)
        std::unordered_map<uint64_t, std::vector<SegmentData>> cell_segments;

        for (const WayData& wd : ways_)
        {
            size_t n = wd.node_ids.size();
            if (n < 2 || wd.coords.size() < 2) continue;
            size_t coord_count = std::min(n, wd.coords.size());

            size_t seg_start = 0;
            float  seg_dist  = 0.0f;

            for (size_t i = 1; i < coord_count; ++i)
            {
                seg_dist += haversine_m(wd.coords[i-1].lat, wd.coords[i-1].lon,
                                        wd.coords[i].lat,   wd.coords[i].lon);

                bool is_vertex = (i == coord_count - 1)
                              || (node_ref_count_.count(wd.node_ids[i])
                                  && node_ref_count_.at(wd.node_ids[i]) >= 2);

                if (is_vertex)
                {
                    SegmentData seg;
                    seg.src_osm_id = wd.node_ids[seg_start];
                    seg.dst_osm_id = wd.node_ids[i];
                    seg.src_pt     = wd.coords[seg_start];
                    seg.dst_pt     = wd.coords[i];
                    seg.dist_m     = seg_dist;
                    seg.hw_class   = wd.hw_class;
                    seg.oneway     = wd.oneway;
                    seg.maxspeed   = wd.maxspeed;
                    seg.name       = wd.name;

                    int src_lf = (int)std::floor(seg.src_pt.lat);
                    int src_lo = (int)std::floor(seg.src_pt.lon);
                    cell_segments[cell_key_of(src_lf, src_lo)].push_back(seg);

                    if (wd.oneway != 1)
                    {
                        SegmentData rev = seg;
                        std::swap(rev.src_osm_id, rev.dst_osm_id);
                        std::swap(rev.src_pt,     rev.dst_pt);
                        rev.oneway = (wd.oneway == 2) ? 1 : 0;
                        int dst_lf = (int)std::floor(rev.src_pt.lat);
                        int dst_lo = (int)std::floor(rev.src_pt.lon);
                        cell_segments[cell_key_of(dst_lf, dst_lo)].push_back(rev);
                    }

                    seg_start = i;
                    seg_dist  = 0.0f;
                }
            }
        }

        // Pass 2: assign a global node index to every unique OSM node id across all cells
        // Node ordering: iterate cells in deterministic order, assign indices as nodes appear
        std::vector<uint64_t> cell_keys;
        cell_keys.reserve(cell_segments.size());
        for (const auto& [key, _] : cell_segments)
            cell_keys.push_back(key);
        std::sort(cell_keys.begin(), cell_keys.end());

        std::unordered_map<int64_t, uint32_t> global_node_index;
        // Pre-pass: allocate global indices in cell order so each cell's nodes are contiguous
        struct CellNodeRange { uint32_t base; uint32_t count; };
        std::unordered_map<uint64_t, CellNodeRange> cell_node_ranges;
        std::vector<std::pair<float,float>> global_coords; // lat,lon for each global node

        for (uint64_t key : cell_keys)
        {
            const auto& segs = cell_segments.at(key);
            uint32_t base = (uint32_t)global_coords.size();
            for (const auto& seg : segs)
            {
                if (!global_node_index.count(seg.src_osm_id))
                {
                    global_node_index[seg.src_osm_id] = (uint32_t)global_coords.size();
                    global_coords.push_back({(float)seg.src_pt.lat, (float)seg.src_pt.lon});
                }
                if (!global_node_index.count(seg.dst_osm_id))
                {
                    global_node_index[seg.dst_osm_id] = (uint32_t)global_coords.size();
                    global_coords.push_back({(float)seg.dst_pt.lat, (float)seg.dst_pt.lon});
                }
            }
            uint32_t cnt = (uint32_t)global_coords.size() - base;
            cell_node_ranges[key] = {base, cnt};
        }

        // Pass 3: build per-cell edge lists using global dst_node indices
        std::vector<CellData> cells;
        cells.reserve(cell_keys.size());

        for (uint64_t key : cell_keys)
        {
            int lat_floor = (int)(key >> 32);
            int lon_floor = (int)(key & 0xFFFFFFFF);
            if ((uint32_t)lon_floor >= 0x80000000u)
                lon_floor = (int)(lon_floor - 0x100000000ull);

            const auto& segs = cell_segments.at(key);
            const CellNodeRange& range = cell_node_ranges.at(key);

            // Local node index within this cell (global_idx - range.base)
            // edge_offset is relative to this cell's edge block
            std::unordered_map<uint32_t, std::vector<RouteEdge>> node_edges;

            for (const SegmentData& seg : segs)
            {
                uint32_t src_global = global_node_index.at(seg.src_osm_id);
                uint32_t dst_global = global_node_index.at(seg.dst_osm_id);
                if (src_global == dst_global) continue;

                // src must be local to this cell
                if (src_global < range.base || src_global >= range.base + range.count)
                    continue;

                float    spd   = speed_ms(seg.hw_class, seg.maxspeed);
                uint32_t cost  = (uint32_t)((seg.dist_m / spd) * 10.0f);
                uint16_t dm    = (uint16_t)std::min((float)UINT16_MAX, seg.dist_m);
                uint8_t  flags = (uint8_t)((seg.oneway == 1 ? 1 : 0)
                                          | ((seg.hw_class & 0x07) << 1));

                RouteEdge e{};
                e.dst_node = dst_global;   // global index
                e.cost     = cost;
                e.dist_m   = dm;
                e.flags    = flags;
                e.name_idx = 0;
                node_edges[src_global].push_back(e);
            }

            // Build RouteNode array for this cell (local index = global - range.base)
            CellData cd;
            cd.lat_floor = lat_floor;
            cd.lon_floor = lon_floor;
            cd.nodes.resize(range.count);
            for (uint32_t li = 0; li < range.count; ++li)
            {
                uint32_t gi = range.base + li;
                cd.nodes[li].lat         = global_coords[gi].first;
                cd.nodes[li].lon         = global_coords[gi].second;
                cd.nodes[li].edge_offset = 0;
            }

            // Flatten edges, assign edge_offset (relative to this cell's edge block)
            uint32_t offset = 0;
            for (uint32_t li = 0; li < range.count; ++li)
            {
                uint32_t gi = range.base + li;
                cd.nodes[li].edge_offset = offset;
                auto it = node_edges.find(gi);
                if (it != node_edges.end())
                {
                    for (const auto& e : it->second)
                        cd.edges.push_back(e);
                    offset += (uint32_t)it->second.size();
                }
            }

            if (cd.nodes.empty()) continue;
            print_stats(lat_floor, lon_floor, cd.nodes, cd.edges, range.base);
            cells.push_back(std::move(cd));
        }

        if (cells.empty()) return;
        write_route_bin(cells);
    }

private:
    struct CellData
    {
        int                    lat_floor;
        int                    lon_floor;
        std::vector<RouteNode> nodes;
        std::vector<RouteEdge> edges;
    };

    struct WayData
    {
        std::vector<int64_t> node_ids;
        std::vector<Point>   coords;
        uint8_t              hw_class;
        uint8_t              oneway;
        uint8_t              maxspeed;
        std::string          name;
    };

    struct SegmentData
    {
        int64_t     src_osm_id;
        int64_t     dst_osm_id;
        Point       src_pt;
        Point       dst_pt;
        float       dist_m;
        uint8_t     hw_class;
        uint8_t     oneway;
        uint8_t     maxspeed;
        std::string name;
    };

    std::unordered_map<int64_t, int> node_ref_count_;
    std::vector<WayData>             ways_;
    std::string                      output_dir_;

    static uint64_t cell_key_of(int lat_floor, int lon_floor)
    {
        return ((uint64_t)(uint32_t)lat_floor << 32) | (uint32_t)lon_floor;
    }

    uint8_t classify(const std::string& hw)
    {
        if (hw == "motorway" || hw == "motorway_link")       return 6;
        if (hw == "trunk"    || hw == "trunk_link")          return 5;
        if (hw == "primary"  || hw == "primary_link")        return 4;
        if (hw == "secondary"|| hw == "secondary_link")      return 3;
        if (hw == "tertiary" || hw == "tertiary_link"
         || hw == "unclassified")                            return 2;
        if (hw == "residential" || hw == "living_street")    return 1;
        return 0;
    }

    float speed_ms(uint8_t hw_class, uint8_t maxspeed)
    {
        static const float base_kmh[7] = {20, 30, 50, 70, 90, 110, 130};
        float kmh = (maxspeed > 0 && maxspeed < 255)
                    ? (float)maxspeed
                    : base_kmh[hw_class];
        return kmh / 3.6f;
    }

    static float haversine_m(double lat1, double lon1, double lat2, double lon2)
    {
        double dlat = (lat2 - lat1) * M_PI / 180.0;
        double dlon = (lon2 - lon1) * M_PI / 180.0;
        double a    = sin(dlat * 0.5) * sin(dlat * 0.5)
                    + cos(lat1 * M_PI / 180.0) * cos(lat2 * M_PI / 180.0)
                    * sin(dlon * 0.5) * sin(dlon * 0.5);
        return (float)(6371000.0 * 2.0 * atan2(sqrt(a), sqrt(1.0 - a)));
    }

    void print_stats(int lat_floor, int lon_floor,
                     const std::vector<RouteNode>& nodes,
                     const std::vector<RouteEdge>& edges,
                     uint32_t global_base)
    {
        uint32_t n = (uint32_t)nodes.size();
        uint32_t e = (uint32_t)edges.size();
        if (n == 0) return;

        // BFS from node 0 (local indices; dst_node is global so remap to local)
        std::vector<bool> visited(n, false);
        std::queue<uint32_t> q;
        q.push(0);
        visited[0] = true;
        uint32_t reachable = 0;
        while (!q.empty())
        {
            uint32_t u = q.front(); q.pop();
            reachable++;
            uint32_t e_end = (u + 1 < n) ? nodes[u + 1].edge_offset : e;
            for (uint32_t ei = nodes[u].edge_offset; ei < e_end; ++ei)
            {
                uint32_t v_global = edges[ei].dst_node;
                if (v_global < global_base || v_global >= global_base + n) continue;
                uint32_t v = v_global - global_base;
                if (!visited[v]) { visited[v] = true; q.push(v); }
            }
        }

        uint32_t oneway_count = 0;
        uint32_t class_count[7] = {};
        for (const auto& edge : edges)
        {
            if (edge.flags & 1) oneway_count++;
            class_count[(edge.flags >> 1) & 7]++;
        }

        printf("[GRAPH] Cell R%d_%d: %u nodes, %u edges\n",
               lat_floor, lon_floor, n, e);
        printf("[GRAPH]   Giant component : %u / %u (%.1f%%)\n",
               reachable, n, 100.0f * reachable / n);
        printf("[GRAPH]   Oneway edges    : %u (%.1f%%)\n",
               oneway_count, e > 0 ? 100.0f * oneway_count / e : 0.0f);
        printf("[GRAPH]   Classes (0-6)   :");
        for (int i = 0; i < 7; i++) printf(" %u", class_count[i]);
        printf("\n");

        if (reachable < n * 0.95f)
            fprintf(stderr, "[GRAPH] WARNING: giant component < 95%% in R%d_%d"
                            " — check intersection detection\n",
                    lat_floor, lon_floor);
    }

    void write_route_bin(const std::vector<struct CellData>& cells)
    {
        char dir_buf[256];
        snprintf(dir_buf, sizeof(dir_buf), "%s/ROUTE", output_dir_.c_str());
        std::filesystem::create_directories(dir_buf);

        char path_buf[256];
        snprintf(path_buf, sizeof(path_buf), "%s/ROUTE/ROUTE.bin", output_dir_.c_str());

        FILE* f = fopen(path_buf, "wb");
        if (!f)
        {
            fprintf(stderr, "[GRAPH] Cannot open %s for writing\n", path_buf);
            return;
        }

        // Build cell index
        std::vector<CellIndexEntry> index;
        index.reserve(cells.size());
        uint32_t node_off = 0;
        uint32_t edge_off = 0;
        for (const auto& cd : cells)
        {
            CellIndexEntry ie{};
            ie.lat_floor   = (int16_t)cd.lat_floor;
            ie.lon_floor   = (int16_t)cd.lon_floor;
            ie.node_offset = node_off;
            ie.node_count  = (uint32_t)cd.nodes.size();
            ie.edge_offset = edge_off;
            ie.edge_count  = (uint32_t)cd.edges.size();
            index.push_back(ie);
            node_off += ie.node_count;
            edge_off += ie.edge_count;
        }

        // File header
        RouteFileHeader hdr{};
        memcpy(hdr.magic, "ROUT", 4);
        hdr.version    = 2;
        hdr.cell_count = (uint32_t)cells.size();
        fwrite(&hdr, sizeof(hdr), 1, f);

        // Cell index
        fwrite(index.data(), sizeof(CellIndexEntry), index.size(), f);

        // All nodes (concatenated)
        for (const auto& cd : cells)
            if (!cd.nodes.empty())
                fwrite(cd.nodes.data(), sizeof(RouteNode), cd.nodes.size(), f);

        // All edges (concatenated)
        for (const auto& cd : cells)
            if (!cd.edges.empty())
                fwrite(cd.edges.data(), sizeof(RouteEdge), cd.edges.size(), f);

        fclose(f);
        printf("[GRAPH] Written: %s  (%zu cells, %u nodes, %u edges)\n",
               path_buf, cells.size(), node_off, edge_off);
    }
};

} // namespace nav
