/**
 * @file route_main.cpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Standalone routing graph generator. Reads an OSM PBF and writes ROUTE/R{lat}_{lon}.bin.
 * @version 0.7.0
 * @date 2026-05
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
            graph_builder.build_and_write();

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
