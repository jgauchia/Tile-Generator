#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <chrono>
#include "psram_shim.hpp"

#pragma pack(push, 1)
struct RouteFileHeader
{
    char     magic[4];
    uint8_t  version;
    uint8_t  reserved[3];
    uint32_t cell_count;
    uint32_t reserved2[4];
};
struct CellIndexEntry
{
    int16_t  lat_floor;
    int16_t  lon_floor;
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t edge_offset;
    uint32_t edge_count;
};
struct RouteNode
{
    float    lat;
    float    lon;
    uint32_t edge_offset;
};
struct RouteEdge
{
    uint32_t dst_node;
    uint32_t cost;
    uint16_t dist_m;
    uint8_t  flags;
    uint16_t name_idx;
    uint8_t  reserved;
};
#pragma pack(pop)

static_assert(sizeof(RouteFileHeader) == 32, "RouteFileHeader size mismatch");
static_assert(sizeof(CellIndexEntry)  == 20, "CellIndexEntry size mismatch");
static_assert(sizeof(RouteNode)       == 12, "RouteNode size mismatch");
static_assert(sizeof(RouteEdge)       == 14, "RouteEdge size mismatch");

struct Graph
{
    uint32_t node_count = 0;
    uint32_t edge_count = 0;
    std::vector<CellIndexEntry>                  cell_index;
    std::vector<RouteNode, PsramAllocator<RouteNode>> nodes;
    std::vector<RouteEdge, PsramAllocator<RouteEdge>> edges;
};

// Load all cells from ROUTE.bin — suitable for small regions like Andorra.
static bool load_graph(const char* path, Graph& g)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    RouteFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return false; }
    if (memcmp(hdr.magic, "ROUT", 4) != 0 || hdr.version != 2)
    {
        fprintf(stderr, "Bad magic or version (expected ROUT v2)\n");
        fclose(f); return false;
    }

    g.cell_index.resize(hdr.cell_count);
    fread(g.cell_index.data(), sizeof(CellIndexEntry), hdr.cell_count, f);

    g.node_count = 0;
    g.edge_count = 0;
    for (const auto& c : g.cell_index)
    {
        g.node_count += c.node_count;
        g.edge_count += c.edge_count;
    }

    g.nodes.resize(g.node_count);
    g.edges.resize(g.edge_count);
    fread(g.nodes.data(), sizeof(RouteNode), g.node_count, f);
    fread(g.edges.data(), sizeof(RouteEdge), g.edge_count, f);
    fclose(f);
    return true;
}

static uint32_t nearest_node(const Graph& g, float lat, float lon)
{
    uint32_t best = 0;
    float best_d = 1e30f;
    for (uint32_t i = 0; i < g.hdr.node_count; ++i)
    {
        float dlat = g.nodes[i].lat - lat;
        float dlon = (g.nodes[i].lon - lon)
                   * cosf((g.nodes[i].lat + lat) * 0.5f * 3.14159265f / 180.f);
        float d = dlat*dlat + dlon*dlon;
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

static float heuristic(const Graph& g, uint32_t a, uint32_t b)
{
    float dlat = g.nodes[b].lat - g.nodes[a].lat;
    float dlon = (g.nodes[b].lon - g.nodes[a].lon)
               * cosf((g.nodes[a].lat + g.nodes[b].lat) * 0.5f * 3.14159265f / 180.f);
    float dist_m = sqrtf(dlat*dlat + dlon*dlon) * 111319.f;
    return (dist_m / 36.1f) * 10.f;
}

struct AStarState
{
    uint32_t f, g, node;
    bool operator>(const AStarState& o) const { return f > o.f; }
};

static std::vector<uint32_t> astar(const Graph& g, uint32_t src, uint32_t dst,
                                   uint32_t& nodes_visited)
{
    uint32_t N = g.hdr.node_count;
    const uint32_t INF = UINT32_MAX;

    std::vector<uint32_t, PsramAllocator<uint32_t>> gcost(N, INF);
    std::vector<uint32_t, PsramAllocator<uint32_t>> prev(N, UINT32_MAX);
    std::vector<bool> visited(N, false);

    std::priority_queue<AStarState,
        std::vector<AStarState, PsramAllocator<AStarState>>,
        std::greater<AStarState>> pq;

    gcost[src] = 0;
    pq.push({(uint32_t)heuristic(g, src, dst), 0, src});
    nodes_visited = 0;

    while (!pq.empty())
    {
        auto [f, gc, u] = pq.top(); pq.pop();
        if (visited[u]) continue;
        visited[u] = true;
        nodes_visited++;
        if (u == dst) break;

        uint32_t e_end = (u + 1 < N) ? g.nodes[u+1].edge_offset : g.hdr.edge_count;
        for (uint32_t ei = g.nodes[u].edge_offset; ei < e_end; ++ei)
        {
            const RouteEdge& e = g.edges[ei];
            uint32_t ng = gcost[u] + e.cost;
            if (ng < gcost[e.dst_node])
            {
                gcost[e.dst_node] = ng;
                prev[e.dst_node]  = u;
                uint32_t h = (uint32_t)heuristic(g, e.dst_node, dst);
                pq.push({ng + h, ng, e.dst_node});
            }
        }
    }

    std::vector<uint32_t> path;
    if (gcost[dst] == INF) return path;
    for (uint32_t cur = dst; cur != UINT32_MAX; cur = prev[cur])
        path.push_back(cur);
    std::reverse(path.begin(), path.end());
    return path;
}

static void run_route(const Graph& g,
                      float src_lat, float src_lon,
                      float dst_lat, float dst_lon,
                      const char* label)
{
    uint32_t src = nearest_node(g, src_lat, src_lon);
    uint32_t dst = nearest_node(g, dst_lat, dst_lon);

    uint32_t visited = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto path = astar(g, src, dst, visited);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("\n[%s]\n", label);
    printf("  src nodo %u (%.5f, %.5f)\n", src, g.nodes[src].lat, g.nodes[src].lon);
    printf("  dst nodo %u (%.5f, %.5f)\n", dst, g.nodes[dst].lat, g.nodes[dst].lon);
    if (path.empty())
    {
        printf("  Sin camino\n");
        return;
    }

    float dist = 0;
    for (size_t i = 1; i < path.size(); ++i)
    {
        float dlat = g.nodes[path[i]].lat - g.nodes[path[i-1]].lat;
        float dlon = (g.nodes[path[i]].lon - g.nodes[path[i-1]].lon)
                   * cosf((g.nodes[path[i]].lat + g.nodes[path[i-1]].lat)
                          * 0.5f * 3.14159265f / 180.f);
        dist += sqrtf(dlat*dlat + dlon*dlon) * 111319.f;
    }

    printf("  Nodos ruta     : %zu\n", path.size());
    printf("  Distancia      : %.0f m (%.1f km)\n", dist, dist / 1000.f);
    printf("  Nodos visitados: %u / %u (%.1f%%)\n",
           visited, g.hdr.node_count, 100.f * visited / g.hdr.node_count);
    printf("  Tiempo A*      : %.2f ms  %s\n",
           ms, ms < 10.0 ? "OK" : "LENTO >10ms");
}

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "NAVMAP/ROUTE/R42_1.bin";

    Graph g;
    if (!load_graph(path, g)) return 1;

    printf("Grafo : %u nodos, %u aristas\n", g.hdr.node_count, g.hdr.edge_count);
    printf("BBox  : %.3f,%.3f -> %.3f,%.3f\n",
           g.hdr.bbox_min_lat, g.hdr.bbox_min_lon,
           g.hdr.bbox_max_lat, g.hdr.bbox_max_lon);

    run_route(g, 42.5069f, 1.5218f, 42.5426f, 1.7336f,
              "Andorra la Vella -> Pas de la Casa");
    run_route(g, 42.5731f, 1.5292f, 42.4934f, 1.4859f,
              "La Massana -> Sant Julia de Loria");
    run_route(g, 42.5069f, 1.5218f, 42.6435f, 1.4737f,
              "Andorra la Vella -> Ordino");

    return 0;
}
