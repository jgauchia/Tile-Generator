/**
 * @file route_main.cpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Standalone routing graph generator. Reads an OSM PBF and writes ROUTE/R{lat}_{lon}.bin.
 * @version 0.9.0
 * @date 2026-06
 */

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include "nav_types.hpp"
#include "graph_builder.hpp"

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

// All highway values the tool can ever process (superset across all profiles)
static const std::unordered_set<std::string> ROUTABLE_HIGHWAY = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street",
    "service", "track", "road",
    "footway", "path", "cycleway", "pedestrian", "steps",
};

// Returns true if a highway tag is accessible for the given profile.
// Inaccessible ways are skipped entirely (no edges generated in GraphBuilder).
static bool is_accessible(const std::string& hw, nav::RoutingProfile profile)
{
    if (profile == nav::RoutingProfile::Pedestrian)
        return hw != "motorway" && hw != "motorway_link"
            && hw != "trunk"    && hw != "trunk_link";

    if (profile == nav::RoutingProfile::Bike)
        return hw != "motorway" && hw != "motorway_link"
            && hw != "trunk"    && hw != "trunk_link"
            && hw != "footway"  && hw != "steps";

    // Car: footways, cycleways and pedestrian zones are off-limits
    return hw != "footway"  && hw != "cycleway"
        && hw != "pedestrian" && hw != "steps" && hw != "path";
}

class RestrictionHandler : public osmium::handler::Handler
{
public:
    // type=restriction relations with a single `via` node. via-way restrictions
    // are skipped (0.6% on real maps) and documented in route_generator.md.

    void relation(const osmium::Relation& r)
    {
        const char* type = r.tags().get_value_by_key("type");
        if (!type || std::string(type) != "restriction")
            return;
        const char* rtype = r.tags().get_value_by_key("restriction");
        if (!rtype)
            return;
        std::string rt(rtype);

        // Only the directional restrictions we can honour in A*.
        static const std::unordered_set<std::string> SUPPORTED = {
            "no_left_turn", "no_right_turn", "only_straight_on",
            "only_right_turn", "only_left_turn", "no_straight_on",
            "no_u_turn",
        };
        if (!SUPPORTED.count(rt))
            return;

        // Via must be a single node.
        const osmium::RelationMember* via = nullptr;
        for (const auto& m : r.members())
            if (std::string(m.role()) == "via")
            {
                if (via) return;         // multiple via members — skip
                via = &m;
            }
        if (!via || via->type() != osmium::item_type::node)
            return;

        // from/to must be ways (via-node restrictions always reference ways).
        const osmium::RelationMember* from = nullptr;
        const osmium::RelationMember* to   = nullptr;
        for (const auto& m : r.members())
        {
            const char* role = m.role();
            if (std::string(role) == "from")  from = &m;
            else if (std::string(role) == "to") to = &m;
        }
        if (!from || !to)
            return;
        if (from->type() != osmium::item_type::way || to->type() != osmium::item_type::way)
            return;

        // from/to way ids are resolved to global edges in GraphBuilder via the
        // way_id carried on each segment; stored as OSM way ids here.
        from_way[via->ref()].insert(from->ref());
        to_way[via->ref()].insert(to->ref());
    }

    std::unordered_map<int64_t, std::unordered_set<int64_t>> from_way;
    std::unordered_map<int64_t, std::unordered_set<int64_t>> to_way;
};

class RouteHandler : public osmium::handler::Handler
{
public:
    nav::RoutingProfile       profile = nav::RoutingProfile::Car;
    std::vector<nav::Feature> road_ways;
    size_t stats_ways     = 0;
    size_t stats_filtered = 0;

    void way(const osmium::Way& w)
    {
        stats_ways++;

        const char* hw = w.tags().get_value_by_key("highway");
        if (!hw || !ROUTABLE_HIGHWAY.count(hw))        { stats_filtered++; return; }
        if (!is_accessible(std::string(hw), profile))  { stats_filtered++; return; }

        // Keep node_ids and points in sync — only include nodes with valid location
        std::vector<nav::Point> pts;
        std::vector<int64_t>    node_ids;
        for (const auto& n : w.nodes())
        {
            if (n.location().valid())
            {
                node_ids.push_back(n.ref());
                pts.push_back({n.lon(), n.lat()});
            }
        }
        if (pts.size() < 2) { stats_filtered++; return; }

        nav::Feature f;
        f.id           = w.id();
        f.highway_type = hw;
        f.points       = std::move(pts);
        f.osm_node_ids = std::move(node_ids);

        const char* ow = w.tags().get_value_by_key("oneway");
        if (ow)
        {
            std::string s(ow);
            if (s == "yes" || s == "1" || s == "true")       f.oneway = 1;
            else if (s == "-1" || s == "reverse")             f.oneway = 2;
            else if (s == "no")                               f.oneway = 0;
        }
        else
        {
            // motorway and motorway_link are oneway by default in OSM
            std::string hws(hw);
            if (hws == "motorway" || hws == "motorway_link")  f.oneway = 1;
        }

        const char* ms = w.tags().get_value_by_key("maxspeed");
        if (ms)
        {
            try { f.maxspeed = (uint8_t)std::min(std::stoi(ms), 255); }
            catch (...) {}
        }

        // Surface quality → 3 bits in RouteEdge.flags (0=unknown, 1=paved, 2=unpaved,
        // 3=gravel, 4=dirt, 5=trail, 6=sand). Only used by BIKE/PED profiles.
        const char* surf = w.tags().get_value_by_key("surface");
        if (surf)
        {
            std::string s(surf);
            if      (s == "asphalt" || s == "paved" || s == "concrete" || s == "concrete:plates" || s == "paving_stones" || s == "sett") f.surface = 1;
            else if (s == "unpaved" || s == "compacted" || s == "fine_gravel" || s == "ground")                                      f.surface = 2;
            else if (s == "gravel" || s == "pebblestone")                                                                               f.surface = 3;
            else if (s == "dirt" || s == "earth" || s == "mud" || s == "clay")                                                       f.surface = 4;
            else if (s == "grass" || s == "grass_paver" || s == "wood" || s == "sand" || s == "gravel" )                            f.surface = (s == "sand") ? 6 : 5;
        }
        else
        {
            // smoothness fallback — rough implies unpaved/dirt for bike/ped
            const char* sm = w.tags().get_value_by_key("smoothness");
            if (sm)
            {
                std::string smv(sm);
                if (smv == "bad" || smv == "very_bad" || smv == "horrible" || smv == "very_horrible" || smv == "impassable")
                    f.surface = 4; // dirt — worst paved-equivalent
                else if (smv == "intermediate")
                    f.surface = 3; // gravel
            }
        }

        const char* nm = w.tags().get_value_by_key("name");
        if (nm) f.name = nm;
        else
        {
            const char* ref = w.tags().get_value_by_key("ref");
            if (ref) f.name = ref;
        }

        road_ways.push_back(std::move(f));
    }
};

static void print_usage()
{
    std::cout << "Usage: route_generator <input.pbf> <output_dir>" << std::endl;
    std::cout << "  Generates ROUTE/CAR/ROUTE.bin, ROUTE/BIKE/ROUTE.bin, ROUTE/PEDESTRIAN/ROUTE.bin" << std::endl;
}

static const char* profile_name(nav::RoutingProfile p)
{
    switch (p)
    {
        case nav::RoutingProfile::Pedestrian: return "pedestrian";
        case nav::RoutingProfile::Bike:       return "bike";
        default:                              return "car";
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        print_usage();
        return 1;
    }

    std::string input_pbf  = argv[1];
    std::string output_dir = argv[2];

    std::cout << "Route generator" << std::endl;
    std::cout << "Input  : " << input_pbf << " ("
              << std::fixed << std::setprecision(1)
              << (std::filesystem::file_size(input_pbf) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Output : " << output_dir << "/ROUTE/{CAR,BIKE,WALK}/ROUTE.bin" << std::endl;

    try
    {
        std::error_code ec;
        if (!std::filesystem::exists(output_dir, ec))
            std::filesystem::create_directories(output_dir, ec);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error preparing output directory: " << e.what() << std::endl;
        return 1;
    }

    auto total_start = std::chrono::steady_clock::now();

    // Extract turn restrictions once (shared by all profiles).
    std::cout << "Scanning turn restrictions..." << std::endl;
    RestrictionHandler restr_handler;
    {
        osmium::io::Reader reader{input_pbf, osmium::osm_entity_bits::relation};
        osmium::apply(reader, restr_handler);
        reader.close();
    }
    std::cout << "Turn restrictions found: " << restr_handler.from_way.size() << " via-nodes" << std::endl;

    // Build resolution input (OSM ids) shared by all profiles.
    std::vector<nav::TurnRestrictionRef> turn_refs;
    {
        std::unordered_map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> merged;
        for (const auto& [via, froms] : restr_handler.from_way)
        {
            auto& [f, t] = merged[via];
            f.assign(froms.begin(), froms.end());
        }
        for (const auto& [via, tos] : restr_handler.to_way)
        {
            auto& [f, t] = merged[via];
            t.assign(tos.begin(), tos.end());
        }
        turn_refs.reserve(merged.size());
        for (auto& [via, ft] : merged)
        {
            nav::TurnRestrictionRef r;
            r.via_osm    = via;
            r.from_ways  = std::move(ft.first);
            r.to_ways    = std::move(ft.second);
            turn_refs.push_back(std::move(r));
        }
    }

    static const nav::RoutingProfile ALL_PROFILES[] = {
        nav::RoutingProfile::Car,
        nav::RoutingProfile::Bike,
        nav::RoutingProfile::Pedestrian,
    };

    for (nav::RoutingProfile profile : ALL_PROFILES)
    {
        std::cout << "\n=== Profile: " << profile_name(profile) << " ===" << std::endl;

        auto start_time = std::chrono::steady_clock::now();

        try
        {
            index_type index;
            location_handler_type location_handler{index};
            RouteHandler route_handler;
            route_handler.profile = profile;

            std::cout << "Pass 1: Scanning relations..." << std::endl;
            osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{
                osmium::area::Assembler::config_type{}};
            osmium::io::Reader reader1{input_pbf, osmium::osm_entity_bits::relation};
            osmium::apply(reader1, mp_manager);
            reader1.close();
            mp_manager.prepare_for_lookup();

            std::cout << "Pass 2: Extracting road ways..." << std::endl;
            osmium::io::Reader reader2{input_pbf};
            osmium::apply(reader2, location_handler, route_handler,
                          mp_manager.handler([](osmium::memory::Buffer&&) {}));
            reader2.close();

            std::cout << "Ways processed : " << route_handler.stats_ways << std::endl;
            std::cout << "Road ways found: " << route_handler.road_ways.size()
                      << " (filtered: " << route_handler.stats_filtered << ")" << std::endl;

            std::unordered_map<std::string, size_t> hw_counts;
            for (const auto& f : route_handler.road_ways)
                hw_counts[f.highway_type]++;
            for (const auto& [hw, cnt] : hw_counts)
                std::cout << "  " << hw << ": " << cnt << std::endl;

            std::cout << "Building routing graph..." << std::endl;
            nav::GraphBuilder graph_builder(output_dir, profile);
            for (const auto& f : route_handler.road_ways)
                graph_builder.add_way(f);
            graph_builder.build_and_write(turn_refs);

            auto end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            int total_sec = static_cast<int>(elapsed.count());
            int m = total_sec / 60;
            int s = total_sec % 60;
            std::cout << "Profile done in ";
            if (m > 0) std::cout << m << "m ";
            std::cout << s << "s" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error [" << profile_name(profile) << "]: " << e.what() << std::endl;
            return 1;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    int total_sec = static_cast<int>(total_elapsed.count());
    int m = total_sec / 60;
    int s = total_sec % 60;
    std::cout << "\nAll profiles done in ";
    if (m > 0) std::cout << m << "m ";
    std::cout << s << "s" << std::endl;

    return 0;
}
