/**
 * @file osmium_handler.hpp
 * @brief OSM PBF extractor using Osmium library.
 */

#pragma once
#include <osmium/osm/way.hpp>
#include <osmium/osm/area.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/handler.hpp>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <algorithm>

#include "nav_types.hpp"
#include "config_manager.hpp"
#include "utils.hpp"

namespace nav {

/**
 * @class OSMHandler
 * @brief Processes OSM entities (Ways and Areas) and stores them as Features.
 * 
 * This class inherits from osmium::handler::Handler and implements the
 * way() and area() callbacks required by osmium::apply.
 */
class OSMHandler : public osmium::handler::Handler
{
public:
    /**
     * @brief Constructor
     * @param cfg Style configuration manager.
     * @param min_z Global minimum zoom level.
     * @param max_z Global maximum zoom level.
     */
    OSMHandler(const ConfigManager& cfg, int min_z, int max_z) 
        : config(cfg), min_zoom_range(min_z), max_zoom_range(max_z) {}

    /** @brief Callback for OSM Nodes (currently only counts them) */
    void node(const osmium::Node&)
    {
        stats_nodes++;
    }

    /**
     * @brief Callback for OSM Ways. Extracts linear features and simple polygons.
     * @param w The OSM way to process.
     */
    void way(const osmium::Way& w)
    {
        stats_ways++;
        
        std::unordered_map<std::string, std::string> tags;
        bool interesting = false;
        for (const auto& t : w.tags())
        {
            tags[t.key()] = t.value();
            if (config.is_interesting(t.key(), t.value()))
                interesting = true;
        }

        if (!interesting)
        {
            stats_filtered++;
            return;
        }

        std::string layer = get_layer(tags);
        if (layer.empty())
        {
            stats_filtered++;
            return;
        }

        FeatureConfig f_cfg = config.get_config(tags);

        Feature feat;
        feat.id = w.id();
        feat.color_rgb565 = f_cfg.color_rgb565;

        std::vector<Point> way_points;
        for (const auto& n : w.nodes())
        {
            if (n.location().valid())
                way_points.push_back(Point{n.lon(), n.lat()});
        }

        if (way_points.size() < 2)
        {
            stats_filtered++;
            return;
        }

        bool is_closed = way_points.size() >= 4 && 
                         way_points.front().lon == way_points.back().lon && 
                         way_points.front().lat == way_points.back().lat;
        
        bool has_area_tag = tags.count("building") || tags.count("landuse") || 
                           tags.count("water") || tags.count("amenity") || 
                           tags.count("leisure") || tags.count("natural") ||
                           tags.count("waterway") || tags.count("man_made") ||
                           tags.count("aeroway");

        bool is_area = is_closed && (has_area_tag || (tags.count("area") && tags.at("area") == "yes"));

        int layer_base = get_layer_priority(layer);
        int combined_priority;

        if (is_area && !tags.count("highway") && !tags.count("place"))
        {
            feat.geom_type = GEOM_POLYGON;
            feat.width_meters = 0;
            processed_areas.insert(w.id());
            combined_priority = layer_base + (f_cfg.priority % 10);
        }
        else
        {
            feat.geom_type = GEOM_LINESTRING;
            feat.width_meters = get_width(tags);
            
            // Lower priority for centerlines in water layer to stay behind polygons
            if (layer == "water")
                combined_priority = layer_base + (f_cfg.priority % 5);
            else
                combined_priority = layer_base + (f_cfg.priority % 10);
        }

        feat.points = std::move(way_points);
        feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
        feat.zoom_priority = utils::pack_zoom_priority(f_cfg.min_zoom, combined_priority);
        feat.zoom_widths = f_cfg.zoom_widths; // Transfer dynamic widths
        features_by_zoom[f_cfg.min_zoom].push_back(std::move(feat));
    }

    /**
     * @brief Callback for OSM Areas. Handles complex multipolygons.
     * @param a The OSM area to process.
     */
    void area(const osmium::Area& a)
    {
        stats_areas++;
        if (a.from_way() && processed_areas.count(a.orig_id()))
            return;
        
        std::unordered_map<std::string, std::string> tags;
        bool interesting = false;
        for (const auto& t : a.tags())
        {
            tags[t.key()] = t.value();
            if (config.is_interesting(t.key(), t.value()))
                interesting = true;
        }

        if (!interesting || tags.count("highway") || tags.count("place"))
            return;

        std::string layer = get_layer(tags);
        if (layer.empty())
            return;

        FeatureConfig f_cfg = config.get_config(tags);
        int layer_base = get_layer_priority(layer);
        uint8_t zoom_prio = utils::pack_zoom_priority(f_cfg.min_zoom, layer_base + (f_cfg.priority % 10));

        // In Osmium, an area can have multiple outer rings (MultiPolygon)
        // We separate them into distinct features to simplify processing
        for (const auto& outer_ring : a.outer_rings())
        {
            Feature feat;
            feat.id = static_cast<int64_t>(a.id());
            feat.geom_type = GEOM_POLYGON;
            feat.color_rgb565 = f_cfg.color_rgb565;
            feat.zoom_priority = zoom_prio;
            feat.width_meters = 0;
            feat.zoom_widths = f_cfg.zoom_widths; // Transfer dynamic widths

            for (const auto& n : outer_ring)
                feat.points.push_back(Point{n.lon(), n.lat()});
            
            if (feat.points.size() < 3)
                continue;
            
            feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));

            for (const auto& inner_ring : a.inner_rings(outer_ring))
            {
                size_t pts_before = feat.points.size();
                for (const auto& n : inner_ring)
                    feat.points.push_back(Point{n.lon(), n.lat()});
                
                if (feat.points.size() - pts_before >= 3)
                    feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
                else
                    feat.points.resize(pts_before); // Rollback invalid small ring
            }

            features_by_zoom[f_cfg.min_zoom].push_back(std::move(feat));
        }
    }

    std::vector<Feature> features_by_zoom[18]; ///< Features grouped by minimum visibility zoom level (0-17)
    std::unordered_set<int64_t> processed_areas; ///< IDs of ways already processed as polygons
    size_t stats_nodes = 0;     ///< Total nodes visited
    size_t stats_ways = 0;      ///< Total ways processed
    size_t stats_areas = 0;     ///< Total areas processed
    size_t stats_filtered = 0;  ///< Total features filtered out

private:
    const ConfigManager& config;
    int min_zoom_range, max_zoom_range;

    std::string get_layer(const std::unordered_map<std::string, std::string>& tags)
    {
        if (tags.count("natural") || tags.count("waterway") || tags.count("water"))
            return "water";
        if (tags.count("landuse") || tags.count("leisure"))
            return "landuse";
        if (tags.count("highway"))
            return "roads";
        if (tags.count("railway"))
            return "railways";
        if (tags.count("building"))
            return "buildings";
        if (tags.count("amenity"))
            return "amenities";
        if (tags.count("bridge") || tags.count("tunnel") || tags.count("aeroway") || (tags.count("man_made") && tags.at("man_made") == "bridge"))
            return "infrastructure";
        if (tags.count("place"))
            return "places";
        return "";
    }

    int get_layer_priority(const std::string& layer)
    {
        if (layer == "landuse") return 10;
        if (layer == "terrain") return 20;
        if (layer == "water") return 30;
        if (layer == "amenities") return 35;
        if (layer == "railways") return 40;
        if (layer == "roads") return 50;
        if (layer == "infrastructure") return 60;
        if (layer == "buildings") return 70;
        if (layer == "places") return 90;
        return 50;
    }

    float get_width(const std::unordered_map<std::string, std::string>& tags)
    {
        if (tags.count("width"))
        {
            try { return std::stof(tags.at("width")); } catch (...) {}
        }
        if (tags.count("lanes"))
        {
            try { return std::stoi(tags.at("lanes")) * 3.5f; } catch (...) {}
        }
        return 0.0f;
    }
};

} // namespace nav