/**
 * @file osmium_handler.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief OSM PBF extractor using Osmium library with mapped storage support.
 * @version 0.4.0
 * @date 2026-02
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
#include "mapped_store.hpp"

namespace nav {

/**
 * @class OSMHandler
 * @brief Processes OSM entities and stores them as offsets in a MappedStore.
 */
class OSMHandler : public osmium::handler::Handler
{
public:
    OSMHandler(const ConfigManager& cfg, MappedStore& st, int min_z, int max_z) 
        : config(cfg), store(st), min_zoom_range(min_z), max_zoom_range(max_z) {}

    void node(const osmium::Node&)
    {
        stats_nodes++;
    }

    void way(const osmium::Way& w)
    {
        stats_ways++;
        std::unordered_map<std::string, std::string> tags;
        bool is_boundary = false;
        int boundary_level = 0;
        if (way_to_boundary.count(w.id()))
        {
            boundary_level = way_to_boundary.at(w.id());
            tags["admin_level"] = std::to_string(boundary_level);
            is_boundary = true;
        }
        bool interesting = is_boundary;
        for (const auto& t : w.tags())
        {
            if (std::string(t.key()) == "boundary")
                continue;
            tags[t.key()] = t.value();
            if (!interesting && config.is_interesting(t.key(), t.value()))
                interesting = true;
        }
        if (!interesting)
        {
            stats_filtered++;
            return;
        }
        if (!is_boundary)
        {
            if (tags.count("tunnel") && tags.at("tunnel") == "yes")
            {
                stats_filtered++;
                return;
            }
            if (tags.count("railway") && tags.at("railway") == "subway")
            {
                stats_filtered++;
                return;
            }
        }
        std::string layer;
        if (is_boundary)
            layer = (boundary_level >= 8) ? "places" : "boundaries";
        else
            layer = get_layer(tags);
        if (layer.empty())
        {
            stats_filtered++;
            return;
        }
        FeatureConfig f_cfg;
        if (is_boundary)
        {
            std::unordered_map<std::string, std::string> b_tags;
            b_tags["admin_level"] = tags["admin_level"];
            f_cfg = config.get_config(b_tags);
        }
        else
            f_cfg = config.get_config(tags);
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
        if (is_area && !tags.count("highway") && !tags.count("place") && layer != "boundaries" && layer != "places")
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
            if (layer == "water")
                combined_priority = layer_base + (f_cfg.priority % 5);
            else
                combined_priority = layer_base + (f_cfg.priority % 10);
        }
        feat.points = std::move(way_points);
        feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
        feat.zoom_priority = utils::pack_zoom_priority(f_cfg.min_zoom, combined_priority);
        feat.zoom_widths = f_cfg.zoom_widths;
        features_by_zoom[f_cfg.min_zoom].push_back(store.append(feat));
    }

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
        if (!interesting || tags.count("highway") || tags.count("place") || tags.count("boundary"))
            return;
        std::string layer = get_layer(tags);
        if (layer.empty() || layer == "boundaries")
            return;
        FeatureConfig f_cfg = config.get_config(tags);
        int layer_base = get_layer_priority(layer);
        uint8_t zoom_prio = utils::pack_zoom_priority(f_cfg.min_zoom, layer_base + (f_cfg.priority % 10));
        for (const auto& outer_ring : a.outer_rings())
        {
            Feature feat;
            feat.id = static_cast<int64_t>(a.id());
            feat.geom_type = GEOM_POLYGON;
            feat.color_rgb565 = f_cfg.color_rgb565;
            feat.zoom_priority = zoom_prio;
            feat.width_meters = 0;
            feat.zoom_widths = f_cfg.zoom_widths;
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
                    feat.points.resize(pts_before);
            }
            features_by_zoom[f_cfg.min_zoom].push_back(store.append(feat));
        }
    }

    std::vector<size_t> features_by_zoom[18];
    std::unordered_set<int64_t> processed_areas;
    std::unordered_map<osmium::unsigned_object_id_type, uint8_t> way_to_boundary;
    size_t stats_nodes = 0;
    size_t stats_ways = 0;
    size_t stats_areas = 0;
    size_t stats_filtered = 0;

private:
    const ConfigManager& config;
    MappedStore& store;
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
        if (layer == "landuse")
            return 10;
        if (layer == "terrain")
            return 20;
        if (layer == "water")
            return 30;
        if (layer == "boundaries")
            return 35;
        if (layer == "railways")
            return 40;
        if (layer == "amenities")
            return 42;
        if (layer == "roads")
            return 50;
        if (layer == "infrastructure")
            return 60;
        if (layer == "buildings")
            return 70;
        if (layer == "places")
            return 90;
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
