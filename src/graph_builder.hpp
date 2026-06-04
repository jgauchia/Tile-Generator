/**
 * @file graph_builder.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Builds ROUTE.bin routing graph files from road features extracted by OSMHandler.
 * @version 0.8.0
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

enum class RoutingProfile { Car, Pedestrian, Bike };

// Speed table [profile][hw_class 0-6] in km/h. 0 = inaccessible (edge skipped).
// hw_class: 0=service/track, 1=living_street/residential, 2=tertiary/unclassified,
//           3=secondary, 4=primary, 5=trunk, 6=motorway
static const float PROFILE_SPEED_KMH[3][7] = {
    // Car: penalises residential/service vs arterials; prohibits nothing below class 5
    { 20,  25,  50,  70,  90, 110, 130 },
    // Pedestrian: flat 5 km/h everywhere; no motorway/trunk
    {  5,   5,   5,   5,   5,   0,   0 },
    // Bike: slow on service, faster on cycleway-class roads, no motorway/trunk
    { 10,  15,  18,  20,  22,   0,   0 },
};

#pragma pack(push, 1)
// File header — 32 bytes
struct RouteFileHeader
{
    char     magic[4];       // "ROUT"
    uint32_t sub_step_e4;    // 500 for 0.05 deg
    uint32_t cell_count;
    uint32_t reserved[5];    // padding to 32B
};

// Per-cell index entry — 20 bytes
struct CellIndexEntry
{
    int32_t  lat_e4;
    int32_t  lon_e4;
    uint32_t node_offset;    // global index of first node (for cellForNode / nearestNode)
    uint16_t node_count;
    uint32_t data_offset;    // byte offset from start of data block: [nodes][edges] interleaved per cell
    uint16_t edge_count;
};
static_assert(sizeof(CellIndexEntry) == 20, "CellIndexEntry size mismatch");

struct RouteNode
{
    float    lat;
    float    lon;
    uint16_t edge_offset;    // index into this cell's edge block (< edge_count, which is uint16)
};
static_assert(sizeof(RouteNode) == 10, "RouteNode size mismatch");

struct RouteEdge
{
    uint32_t dst_node;       // global index
    uint32_t cost;           // travel time in tenths of second
};
static_assert(sizeof(RouteEdge) == 8, "RouteEdge size mismatch");
#pragma pack(pop)

class GraphBuilder
{
public:
    explicit GraphBuilder(const std::string& output_dir,
                          RoutingProfile profile = RoutingProfile::Car)
        : output_dir_(output_dir), profile_(profile) {}

    void add_way(const Feature& f)
    {
        if (f.osm_node_ids.size() < 2) return;
        WayData wd;
        wd.node_ids  = f.osm_node_ids;
        wd.coords    = f.points;
        wd.hw_class  = classify(f.highway_type);
        wd.oneway    = f.oneway;
        wd.maxspeed  = f.maxspeed;
        ways_.push_back(std::move(wd));

        for (int64_t id : ways_.back().node_ids)
            node_ref_count_[id]++;
    }

    void build_and_write()
    {
        if (ways_.empty()) return;

        // Pass 1: collect segments per cell, splitting cross-cell segments at boundaries.
        // A segment whose src and dst fall in different 0.05° cells is split at each
        // cell-boundary crossing so every resulting sub-segment is wholly contained
        // within one cell.  Synthetic split nodes get negative OSM IDs (unique, never
        // collide with real OSM IDs which are always positive).
        std::unordered_map<uint64_t, std::vector<SegmentData>> cell_segments;
        int64_t synthetic_id = -1;  // decrements for each new synthetic node

        auto emit_segment = [&](SegmentData seg)
        {
            // Split the segment at every cell-boundary it crosses.  We iterate
            // until src and dst are in the same cell.
            while (true)
            {
                uint64_t src_cell = cell_key_of((float)seg.src_pt.lat, (float)seg.src_pt.lon);
                uint64_t dst_cell = cell_key_of((float)seg.dst_pt.lat, (float)seg.dst_pt.lon);
                if (src_cell == dst_cell)
                {
                    cell_segments[src_cell].push_back(seg);
                    return;
                }

                // Find the nearest cell boundary between src and dst.
                // Cell step = 0.05° = 1/20°.  Boundary is at multiples of 0.05°.
                constexpr double STEP = 1.0 / 20.0;   // 0.05°

                double lat1 = seg.src_pt.lat, lon1 = seg.src_pt.lon;
                double lat2 = seg.dst_pt.lat, lon2 = seg.dst_pt.lon;

                // t* = parameter in [0,1] where the segment first exits the src cell
                double t_best = 2.0;

                // Lat boundaries crossed
                if (lat2 != lat1)
                {
                    double lat_lo = std::floor(lat1 / STEP) * STEP;
                    double lat_hi = lat_lo + STEP;
                    double boundary = (lat2 > lat1) ? lat_hi : lat_lo;
                    double t = (boundary - lat1) / (lat2 - lat1);
                    if (t > 1e-9 && t < t_best) t_best = t;
                }

                // Lon boundaries crossed
                if (lon2 != lon1)
                {
                    double lon_lo = std::floor(lon1 / STEP) * STEP;
                    double lon_hi = lon_lo + STEP;
                    double boundary = (lon2 > lon1) ? lon_hi : lon_lo;
                    double t = (boundary - lon1) / (lon2 - lon1);
                    if (t > 1e-9 && t < t_best) t_best = t;
                }

                if (t_best > 1.0)
                {
                    // Numerically the cells differ but no boundary found — emit as-is.
                    cell_segments[src_cell].push_back(seg);
                    return;
                }

                // Interpolate split point
                Point split_pt;
                split_pt.lat = lat1 + t_best * (lat2 - lat1);
                split_pt.lon = lon1 + t_best * (lon2 - lon1);

                float head_dist = seg.dist_m * (float)t_best;
                float tail_dist = seg.dist_m * (float)(1.0 - t_best);

                int64_t split_id = synthetic_id--;

                // Head segment: src → split_pt  (stays in src_cell)
                SegmentData head = seg;
                head.dst_osm_id  = split_id;
                head.dst_pt      = split_pt;
                head.dist_m      = head_dist;
                cell_segments[src_cell].push_back(head);

                // Tail segment: split_pt → dst  (may still cross more boundaries → loop)
                seg.src_osm_id = split_id;
                seg.src_pt     = split_pt;
                seg.dist_m     = tail_dist;
            }
        };

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

                    emit_segment(seg);

                    if (wd.oneway != 1)
                    {
                        SegmentData rev = seg;
                        std::swap(rev.src_osm_id, rev.dst_osm_id);
                        std::swap(rev.src_pt,     rev.dst_pt);
                        rev.oneway = (wd.oneway == 2) ? 1 : 0;
                        emit_segment(rev);
                    }

                    seg_start = i;
                    seg_dist  = 0.0f;
                }
            }
        }

        // Pass 2: assign a global node index to every unique OSM node id across all cells.
        // Rule: a node is registered in the cell where it first appears as a src_osm_id.
        // Nodes that only appear as dst_osm_id (pure sinks in this cell) are registered
        // after all src nodes of their cell so they still get a valid global index, but
        // they must not block space in a cell whose src list never references them.
        // This ensures that for every segment in cell C, src_global is always within
        // C's CellNodeRange — the fundamental invariant for Pass 3 edge emission.
        std::vector<uint64_t> cell_keys;
        cell_keys.reserve(cell_segments.size());
        for (const auto& [key, _] : cell_segments)
            cell_keys.push_back(key);
        std::sort(cell_keys.begin(), cell_keys.end());

        std::unordered_map<int64_t, uint32_t> global_node_index;
        struct CellNodeRange { uint32_t base; uint32_t count; };
        std::unordered_map<uint64_t, CellNodeRange> cell_node_ranges;
        std::vector<std::pair<float,float>> global_coords;

        // First sub-pass: register each node in the cell where it appears as src.
        // This guarantees src_global ∈ [range.base, range.base+range.count) for all segs.
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
            }
            uint32_t cnt = (uint32_t)global_coords.size() - base;
            cell_node_ranges[key] = {base, cnt};
        }

        // Second sub-pass: register dst nodes that were not yet seen as src anywhere.
        // These are true terminals (dead-ends or cross-cell targets already registered).
        for (uint64_t key : cell_keys)
        {
            const auto& segs = cell_segments.at(key);
            for (const auto& seg : segs)
            {
                if (!global_node_index.count(seg.dst_osm_id))
                {
                    global_node_index[seg.dst_osm_id] = (uint32_t)global_coords.size();
                    global_coords.push_back({(float)seg.dst_pt.lat, (float)seg.dst_pt.lon});
                }
            }
        }

        // Pass 3: build per-cell edge lists
        std::vector<CellData> cells;
        cells.reserve(cell_keys.size());

        for (uint64_t key : cell_keys)
        {
            int32_t lat_e4 = (int32_t)(key >> 32);
            int32_t lon_e4 = (int32_t)(key & 0xFFFFFFFF);

            const auto& segs = cell_segments.at(key);
            const CellNodeRange& range = cell_node_ranges.at(key);

            std::unordered_map<uint32_t, std::vector<RouteEdge>> node_edges;
            uint32_t oneway_count = 0;
            uint32_t class_count[7] = {};

            for (const SegmentData& seg : segs)
            {
                uint32_t src_global = global_node_index.at(seg.src_osm_id);
                uint32_t dst_global = global_node_index.at(seg.dst_osm_id);
                if (src_global == dst_global) continue;

                float    spd   = speed_ms(seg.hw_class, seg.maxspeed);
                if (spd <= 0.0f) continue;   // inaccessible for this profile
                uint32_t cost  = (uint32_t)((seg.dist_m / spd) * 10.0f);

                RouteEdge e{};
                e.dst_node = dst_global;
                e.cost     = cost;
                node_edges[src_global].push_back(e);

                if (seg.oneway == 1) oneway_count++;
                class_count[seg.hw_class & 0x07]++;
            }

            CellData cd;
            cd.lat_e4 = lat_e4;
            cd.lon_e4 = lon_e4;
            cd.nodes.resize(range.count);
            for (uint32_t li = 0; li < range.count; ++li)
            {
                uint32_t gi = range.base + li;
                cd.nodes[li].lat         = global_coords[gi].first;
                cd.nodes[li].lon         = global_coords[gi].second;
                cd.nodes[li].edge_offset = 0;
            }

            uint32_t offset = 0;
            for (uint32_t li = 0; li < range.count; ++li)
            {
                uint32_t gi = range.base + li;
                cd.nodes[li].edge_offset = (uint16_t)offset;
                auto it = node_edges.find(gi);
                if (it != node_edges.end())
                {
                    for (const auto& e : it->second)
                        cd.edges.push_back(e);
                    offset += (uint32_t)it->second.size();
                }
            }

            if (cd.nodes.empty()) continue;
            print_stats(lat_e4, lon_e4, cd.nodes, cd.edges, range.base,
                        oneway_count, class_count);
            cells.push_back(std::move(cd));
        }

        if (cells.empty()) return;
        write_route_bin(cells);
    }

private:
    struct CellData
    {
        int32_t                lat_e4;
        int32_t                lon_e4;
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
    };

    struct SegmentData
    {
        int64_t src_osm_id;
        int64_t dst_osm_id;
        Point   src_pt;
        Point   dst_pt;
        float   dist_m;
        uint8_t hw_class;
        uint8_t oneway;
        uint8_t maxspeed;
    };

    std::unordered_map<int64_t, int> node_ref_count_;
    std::vector<WayData>             ways_;
    std::string                      output_dir_;
    RoutingProfile                   profile_;

    static uint64_t cell_key_of(float lat, float lon)
    {
        int32_t lat_e4 = (int32_t)std::floor(lat * 20.0f) * 500;
        int32_t lon_e4 = (int32_t)std::floor(lon * 20.0f) * 500;
        return ((uint64_t)(uint32_t)lat_e4 << 32) | (uint32_t)lon_e4;
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

    // Returns speed in m/s for the active profile. Returns 0 if inaccessible.
    float speed_ms(uint8_t hw_class, uint8_t maxspeed)
    {
        int pidx = (int)profile_;
        float base_kmh = PROFILE_SPEED_KMH[pidx][hw_class & 0x07];
        if (base_kmh <= 0.0f) return 0.0f;
        // maxspeed tag overrides base speed only for car profile
        if (profile_ == RoutingProfile::Car && maxspeed > 0 && maxspeed < 255)
            base_kmh = (float)maxspeed;
        return base_kmh / 3.6f;
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

    void print_stats(int32_t lat_e4, int32_t lon_e4,
                     const std::vector<RouteNode>& nodes,
                     const std::vector<RouteEdge>& edges,
                     uint32_t global_base,
                     uint32_t oneway_count,
                     const uint32_t class_count[7])
    {
        uint32_t n = (uint32_t)nodes.size();
        uint32_t e = (uint32_t)edges.size();
        if (n == 0) return;

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

        printf("[GRAPH] Cell E4:%d_%d: %u nodes, %u edges\n",
               lat_e4, lon_e4, n, e);
        printf("[GRAPH]   Giant component : %u / %u (%.1f%%)\n",
               reachable, n, 100.0f * reachable / n);
        printf("[GRAPH]   Oneway edges    : %u (%.1f%%)\n",
               oneway_count, e > 0 ? 100.0f * oneway_count / e : 0.0f);
        printf("[GRAPH]   Classes (0-6)   :");
        for (int i = 0; i < 7; i++) printf(" %u", class_count[i]);
        printf("\n");

        if (reachable < n * 0.95f)
            fprintf(stderr, "[GRAPH] WARNING: giant component < 95%% in E4:%d_%d"
                            " — check intersection detection\n",
                    lat_e4, lon_e4);
    }

    const char* profile_subdir() const
    {
        switch (profile_)
        {
            case RoutingProfile::Pedestrian: return "WALK";
            case RoutingProfile::Bike:       return "BIKE";
            default:                         return "CAR";
        }
    }

    void write_route_bin(const std::vector<struct CellData>& cells)
    {
        char dir_buf[256];
        snprintf(dir_buf, sizeof(dir_buf), "%s/ROUTE/%s", output_dir_.c_str(), profile_subdir());
        std::filesystem::create_directories(dir_buf);

        char path_buf[256];
        snprintf(path_buf, sizeof(path_buf), "%s/ROUTE/%s/ROUTE.bin", output_dir_.c_str(), profile_subdir());

        FILE* f = fopen(path_buf, "wb");
        if (!f)
        {
            fprintf(stderr, "[GRAPH] Cannot open %s for writing\n", path_buf);
            return;
        }

        // Build index: node_offset is the global node index (for cellForNode/nearestNode);
        // data_offset is the byte offset within the data block where [nodes][edges] for
        // this cell start — allows a single seek+read per cell in the firmware loader.
        std::vector<CellIndexEntry> index;
        index.reserve(cells.size());
        uint32_t node_off  = 0;
        uint32_t data_off  = 0;
        uint32_t total_edges = 0;
        for (const auto& cd : cells)
        {
            if (cd.nodes.size() > UINT16_MAX || cd.edges.size() > UINT16_MAX)
                fprintf(stderr, "[GRAPH] WARNING: cell E4:%d_%d exceeds uint16 limit"
                                " (%zu nodes, %zu edges) — counts/edge_offset will wrap\n",
                        cd.lat_e4, cd.lon_e4, cd.nodes.size(), cd.edges.size());

            CellIndexEntry ie{};
            ie.lat_e4      = cd.lat_e4;
            ie.lon_e4      = cd.lon_e4;
            ie.node_offset = node_off;
            ie.node_count  = (uint16_t)cd.nodes.size();
            ie.data_offset = data_off;
            ie.edge_count  = (uint16_t)cd.edges.size();
            index.push_back(ie);
            node_off    += ie.node_count;
            data_off    += (uint32_t)(cd.nodes.size() * sizeof(RouteNode)
                                    + cd.edges.size() * sizeof(RouteEdge));
            total_edges += ie.edge_count;
        }

        RouteFileHeader hdr{};
        memcpy(hdr.magic, "ROUT", 4);
        hdr.sub_step_e4 = 500;
        hdr.cell_count  = (uint32_t)cells.size();
        fwrite(&hdr, sizeof(hdr), 1, f);

        fwrite(index.data(), sizeof(CellIndexEntry), index.size(), f);

        // Write each cell's nodes immediately followed by its edges.
        for (const auto& cd : cells)
        {
            if (!cd.nodes.empty())
                fwrite(cd.nodes.data(), sizeof(RouteNode), cd.nodes.size(), f);
            if (!cd.edges.empty())
                fwrite(cd.edges.data(), sizeof(RouteEdge), cd.edges.size(), f);
        }

        fclose(f);
        printf("[GRAPH] Written: %s  (%zu cells, %u nodes, %u edges)\n",
               path_buf, cells.size(), node_off, total_edges);
    }
};

} // namespace nav
