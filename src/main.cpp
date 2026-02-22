/**
 * @file main.cpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Entry point for the NAV Tile Generator C++ implementation with packed container support.
 * @version 0.5.0
 * @date 2026-02
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include "nav_types.hpp"
#include "config_manager.hpp"
#include "osmium_handler.hpp"
#include "tile_processor.hpp"
#include "utils.hpp"
#include "mapped_store.hpp"

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

void print_usage()
{
    std::cout << "Usage: nav_generator <input.pbf> <output_dir> <features.json> [--zoom min-max]" << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        print_usage();
        return 1;
    }
    std::string input_pbf = argv[1];
    std::string output_dir = argv[2];
    std::string config_file = argv[3];
    int min_zoom = 6, max_zoom = 17;
    for (int i = 4; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--zoom" && i + 1 < argc)
        {
            std::string zoom_val = argv[++i];
            size_t dash = zoom_val.find('-');
            if (dash != std::string::npos)
            {
                min_zoom = std::stoi(zoom_val.substr(0, dash));
                max_zoom = std::stoi(zoom_val.substr(dash + 1));
            }
            else
                min_zoom = max_zoom = std::stoi(zoom_val);
        }
    }
    std::cout << "Loading configuration from " << config_file << std::endl;
    nav::ConfigManager config;
    if (!config.load(config_file))
    {
        std::cerr << "Error: Could not load config file " << config_file << std::endl;
        return 1;
    }
    std::cout << "Processing PBF file: " << input_pbf << " ("
              << std::fixed << std::setprecision(1) << (std::filesystem::file_size(input_pbf) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Zoom range: " << min_zoom << "-" << max_zoom << std::endl;
    std::cout << "Output format: packed binary containers (Zxx.nav)" << std::endl;
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
    class BoundaryScanner : public osmium::handler::Handler
    {
    public:
        BoundaryScanner(std::unordered_map<osmium::unsigned_object_id_type, uint8_t>& w2b) : way_to_boundary(w2b) {}
        void relation(const osmium::Relation& rel)
        {
            const char* boundary = rel.tags().get_value_by_key("boundary");
            if (boundary && std::string(boundary) == "administrative")
            {
                const char* level_str = rel.tags().get_value_by_key("admin_level");
                int level = level_str ? std::atoi(level_str) : 10;
                if (level <= 8)
                {
                    for (const auto& member : rel.members())
                    {
                        if (member.type() == osmium::item_type::way)
                        {
                            auto it = way_to_boundary.find(member.ref());
                            if (it == way_to_boundary.end() || level < it->second)
                                way_to_boundary[member.ref()] = (uint8_t)level;
                        }
                    }
                }
            }
        }
    private:
        std::unordered_map<osmium::unsigned_object_id_type, uint8_t>& way_to_boundary;
    };
    std::cout << "Processing OSM data..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    try
    {
        index_type index;
        location_handler_type location_handler{index};
        nav::MappedStore store("nav_features.tmp");
        nav::OSMHandler osm_handler{config, store, min_zoom, max_zoom};
        std::cout << "  Pass 1: Scanning relations..." << std::endl;
        osmium::area::MultipolygonManager<osmium::area::Assembler> mp_manager{osmium::area::Assembler::config_type{}};
        BoundaryScanner boundary_scanner(osm_handler.way_to_boundary);
        osmium::io::Reader reader1{input_pbf, osmium::osm_entity_bits::relation};
        osmium::apply(reader1, mp_manager, boundary_scanner);
        reader1.close();
        std::cout << "  Preparing relations..." << std::endl;
        mp_manager.prepare_for_lookup();
        std::cout << "  Pass 2: Extracting features..." << std::endl;
        osmium::io::Reader reader2{input_pbf};
        osmium::apply(reader2, location_handler, osm_handler, mp_manager.handler([&osm_handler](osmium::memory::Buffer&& buffer) {
            osmium::apply(buffer, osm_handler);
        }));
        reader2.close();
        auto end_extract = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end_extract - start_time;
        std::cout << "\nProcessing completed in " << std::fixed << std::setprecision(2) << elapsed.count() << "s" << std::endl;
        std::cout << "Statistics:" << std::endl;
        std::cout << "  Nodes visited:      " << osm_handler.stats_nodes << std::endl;
        std::cout << "  Ways processed:     " << osm_handler.stats_ways << std::endl;
        size_t total_features = 0;
        for (int i = 0; i < 18; ++i)
            total_features += osm_handler.features_by_zoom[i].size();
        std::cout << "  Features extracted: " << total_features << std::endl;
        std::cout << "  Features filtered:  " << osm_handler.stats_filtered << std::endl;
        std::cout << "  Point features:     " << osm_handler.stats_points << std::endl;
        std::cout << "  Text labels:        " << osm_handler.stats_text_labels << std::endl;

        std::cout << "Generating packed binary containers..." << std::endl;
        nav::TileProcessor processor{output_dir};
        processor.process_all(osm_handler.features_by_zoom, store, min_zoom, max_zoom,
                              osm_handler.text_features_vec, osm_handler.point_features);

        auto end_all = std::chrono::steady_clock::now();
        std::chrono::duration<double> total_elapsed = end_all - start_time;
        uint64_t total_bytes = processor.get_total_bytes();
        size_t file_count = processor.get_total_files();
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Conversion Summary" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << "Input:            " << input_pbf << std::endl;
        std::cout << "Output directory: " << output_dir << std::endl;
        std::cout << "Format:           packed binary containers (Zxx.nav)" << std::endl;
        std::cout << "Total packs:      " << file_count << std::endl;
        std::cout << "Total size:       " << std::fixed << std::setprecision(2) << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        int total_sec = static_cast<int>(total_elapsed.count());
        int h = total_sec / 3600;
        int m = (total_sec % 3600) / 60;
        int s = total_sec % 60;
        std::cout << "Total time:       ";
        if (h > 0)
            std::cout << h << "h ";
        if (m > 0 || h > 0)
            std::cout << m << "m ";
        std::cout << s << "s" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error during PBF processing: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
