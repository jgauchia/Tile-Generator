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
#include <cmath>
#include "nav_types.hpp"
#include "config_manager.hpp"
#include "utils.hpp"
#include "mapped_store.hpp"
#include "constants.hpp"

namespace nav {

class OSMHandler : public osmium::handler::Handler
{
public:
    OSMHandler(const ConfigManager& cfg, MappedStore& st, int min_z, int max_z)
        : config(cfg), store(st), min_zoom_range(min_z), max_zoom_range(max_z) {}

    void node(const osmium::Node& n)
    {
        stats_nodes++;
        if (!n.location().valid()) return;

        double lon = n.location().lon(), lat = n.location().lat();
        if (lon < bbox_min_lon) bbox_min_lon = lon;
        if (lon > bbox_max_lon) bbox_max_lon = lon;
        if (lat < bbox_min_lat) bbox_min_lat = lat;
        if (lat > bbox_max_lat) bbox_max_lat = lat;

        std::unordered_map<std::string, std::string> tags;
        for (const auto& t : n.tags())
            tags[t.key()] = t.value();

        // Point symbols (peaks, volcanoes)
        for (const auto& [k, v] : tags)
        {
            std::string fk = k + "=" + v;
            auto pit = constants::point_features().find(fk);
            if (pit != constants::point_features().end())
            {
                FeatureConfig f_cfg = config.get_config({{k, v}});
                int min_zoom = f_cfg.min_zoom;
                if (min_zoom > max_zoom_range) return;

                Feature feat;
                feat.id = n.id();
                feat.geom_type = GEOM_POINT;
                feat.color_rgb565 = f_cfg.color_rgb565;
                feat.zoom_priority = utils::pack_zoom_priority(min_zoom, 15);
                feat.width_meters = 0;
                feat.shape = pit->second.shape;
                feat.layer = "terrain";
                feat.points.push_back({lon, lat});
                feat.ring_ends.push_back(1);
                point_features.push_back(std::move(feat));
                stats_points++;
                return;
            }

            // Text labels (places)
            auto tit = constants::text_features().find(fk);
            if (tit != constants::text_features().end())
            {
                std::string name = tags.count("name") ? tags.at("name") : "";
                if (name.empty()) return;

                const auto& text_cfg = tit->second;
                FeatureConfig f_cfg = config.get_config({{k, v}});

                int population = 0;
                if (tags.count("population"))
                {
                    std::string ps = tags.at("population");
                    ps.erase(std::remove(ps.begin(), ps.end(), ','), ps.end());
                    ps.erase(std::remove(ps.begin(), ps.end(), ' '), ps.end());
                    try { population = std::stoi(ps); } catch (...) {}
                }

                int min_zoom = text_cfg.zoom_rules.back().second;
                for (const auto& [min_pop, z] : text_cfg.zoom_rules)
                {
                    if (population >= min_pop) { min_zoom = z; break; }
                }
                if (min_zoom > max_zoom_range) return;

                int nibble = 12;
                if (population >= constants::POPULATION_MAJOR_CITY) nibble = 15;
                else if (population >= constants::POPULATION_LARGE_CITY) nibble = 14;
                else if (population >= constants::POPULATION_TOWN) nibble = 13;

                name = utils::split_place_name(name, constants::PLACE_NAME_BREAK_THRESHOLD);
                std::string name_trunc = name.substr(0, 255);

                Feature feat;
                feat.id = n.id();
                feat.geom_type = GEOM_TEXT;
                feat.color_rgb565 = f_cfg.color_rgb565;
                feat.zoom_priority = utils::pack_zoom_priority(min_zoom, nibble);
                feat.font_size = text_cfg.font_size;
                feat.text.assign(name_trunc.begin(), name_trunc.end());
                feat.population = population;
                feat.layer = "places";
                feat.points.push_back({lon, lat});
                feat.ring_ends.push_back(1);
                text_features_vec.push_back(std::move(feat));
                stats_text_labels++;
                return;
            }
        }
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
            if (std::string(t.key()) == "boundary") continue;
            tags[t.key()] = t.value();
            if (!interesting && config.is_interesting(t.key(), t.value()))
                interesting = true;
        }
        if (!interesting) { stats_filtered++; return; }

        std::string layer;
        if (is_boundary)
            layer = (boundary_level >= 8) ? "places" : "boundaries";
        else
            layer = get_layer(tags);
        if (layer.empty()) { stats_filtered++; return; }

        FeatureConfig f_cfg;
        if (is_boundary)
        {
            std::unordered_map<std::string, std::string> b_tags;
            b_tags["admin_level"] = tags["admin_level"];
            f_cfg = config.get_config(b_tags);
        }
        else
            f_cfg = config.get_config(tags);

        int min_zoom = f_cfg.min_zoom;

        // Railway service tracks → push to z13
        if (tags.count("railway") && tags.count("service"))
        {
            const std::string& svc = tags.at("service");
            if (svc == "yard" || svc == "siding" || svc == "spur" || svc == "crossover")
            {
                min_zoom = std::max(min_zoom, 13);
                if (min_zoom > max_zoom_range) { stats_filtered++; return; }
            }
        }

        Feature feat;
        feat.id = w.id();
        feat.color_rgb565 = f_cfg.color_rgb565;
        feat.layer = layer;
        feat.name = tags.count("name") ? tags.at("name") : "";

        std::vector<Point> way_points;
        for (const auto& n : w.nodes())
        {
            if (n.location().valid())
                way_points.push_back(Point{n.lon(), n.lat()});
        }
        if (way_points.size() < 2) { stats_filtered++; return; }

        bool is_closed = way_points.size() >= 4 &&
                         way_points.front().lon == way_points.back().lon &&
                         way_points.front().lat == way_points.back().lat;
        bool has_area_tag = tags.count("building") || tags.count("landuse") ||
                           tags.count("water") || tags.count("amenity") ||
                           tags.count("leisure") || tags.count("natural") ||
                           tags.count("waterway") || tags.count("man_made") ||
                           tags.count("aeroway");
        bool is_area = is_closed && (has_area_tag || (tags.count("area") && tags.at("area") == "yes"));

        bool is_aeroway_line = tags.count("aeroway") &&
                              (tags.at("aeroway") == "runway" || tags.at("aeroway") == "taxiway" ||
                               tags.at("aeroway") == "helipad");
        
        // Optimize: Allow highway areas (pedestrian squares, parking lots) but NOT roads
        bool is_road_transit = tags.count("highway") && 
                              (tags.at("highway") == "motorway" || tags.at("highway") == "trunk" ||
                               tags.at("highway") == "primary" || tags.at("highway") == "secondary" ||
                               tags.at("highway") == "tertiary");

        if (is_area && !is_road_transit && !tags.count("place") &&
            !is_aeroway_line && layer != "boundaries" && layer != "places")
        {
            // Water ways that belong to multipolygon relations: skip here,
            // let area_callback handle them to preserve inner rings
            if (layer == "water" && water_relation_ways.count(w.id()))
                return;

            feat.geom_type = GEOM_POLYGON;
            feat.width_meters = 0;
            feat.is_building = (layer == "buildings");
            processed_areas.insert(w.id());

            // Skip underground polygons (metro platforms, etc.)
            // Only filter when explicitly underground (tunnel/covered), not layer alone
            // (layer=-5 on wetlands, riverbanks etc. means elevation, not underground)
            bool poly_underground = (tags.count("level") && std::atoi(tags.at("level").c_str()) < 0)
                || ((tags.count("tunnel") || tags.count("covered"))
                    && (tags.count("layer") && std::atoi(tags.at("layer").c_str()) < 0));
            if (poly_underground)
            {
                stats_filtered++;
                return;
            }

            int nibble = get_polygon_nibble(tags, layer);
            feat.zoom_priority = utils::pack_zoom_priority(min_zoom, nibble);
            feat.points = std::move(way_points);
            feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
            features_by_zoom[min_zoom].push_back(store.append(feat));
            return;
        }

        // Linestring
        feat.geom_type = GEOM_LINESTRING;
        feat.layer = layer;
        feat.width_meters = get_width(tags);
        feat.highway_type = get_highway_type(tags);
        feat.ref = tags.count("ref") ? tags.at("ref") : "";
        feat.old_ref = tags.count("old_ref") ? tags.at("old_ref") : "";

        int nibble = f_cfg.priority;

        feat.is_bridge = (tags.count("bridge") && (tags.at("bridge") == "yes" || tags.at("bridge") == "viaduct"));
        if (feat.is_bridge)
            nibble = 15;

        bool is_underground = (tags.count("tunnel") && (tags.at("tunnel") == "yes" || tags.at("tunnel") == "culvert"))
            || (tags.count("layer") && std::atoi(tags.at("layer").c_str()) < 0)
            || (tags.count("level") && std::atoi(tags.at("level").c_str()) < 0);
        if (is_underground)
        {
            if (layer == "water")
            {
                stats_filtered++;
                return;
            }
            std::string hw = tags.count("highway") ? tags.at("highway") : "";
            if (hw == "pedestrian" || hw == "footway" || hw == "cycleway" || hw == "steps" || hw == "platform")
            {
                stats_filtered++;
                return;
            }
            feat.color_rgb565 = 0xD69A; // #D0D0D0 light grey for underground
        }

        feat.zoom_priority = utils::pack_zoom_priority(min_zoom, nibble);

        // Densify curves (not aeroways)
        // bool is_aeroway = tags.count("aeroway") && !tags.at("aeroway").empty();
        // if (!feat.highway_type.empty() && !is_aeroway && way_points.size() >= 2)
        //    way_points = utils::densify_linestring(way_points, 0.0001);

        feat.points = std::move(way_points);
        feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
        features_by_zoom[min_zoom].push_back(store.append(feat));

        // Road labels
        create_road_label(feat.points, feat.ref, feat.old_ref, feat.highway_type, feat.color_rgb565);

        // Waterway labels
        if (tags.count("waterway"))
            create_waterway_label(feat.points, tags.at("waterway"), feat.name);
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
        bool is_place_area = tags.count("place") && (tags.at("place") == "island" || tags.at("place") == "islet");
        if (!interesting || tags.count("highway") || tags.count("boundary"))
            return;
        if (tags.count("place") && !is_place_area)
            return;
        std::string layer = get_layer(tags);
        if (layer.empty() || layer == "boundaries")
            return;

        // Force buildings layer if building tag present
        if (tags.count("building"))
            layer = "buildings";

        // Skip underground areas (metro platforms, etc.)
        bool area_underground = (tags.count("level") && std::atoi(tags.at("level").c_str()) < 0)
            || ((tags.count("tunnel") || tags.count("covered"))
                && (tags.count("layer") && std::atoi(tags.at("layer").c_str()) < 0));
        if (area_underground)
            return;

        FeatureConfig f_cfg = config.get_config(tags);
        int min_zoom = f_cfg.min_zoom;
        int nibble = get_polygon_nibble(tags, layer);

        for (const auto& outer_ring : a.outer_rings())
        {
            Feature feat;
            feat.id = static_cast<int64_t>(a.id());
            feat.geom_type = GEOM_POLYGON;
            feat.color_rgb565 = f_cfg.color_rgb565;
            feat.zoom_priority = utils::pack_zoom_priority(min_zoom, nibble);
            feat.width_meters = 0;
            feat.layer = layer;
            feat.is_building = (layer == "buildings");
            feat.name = tags.count("name") ? tags.at("name") : "";

            for (const auto& n : outer_ring)
                feat.points.push_back(Point{n.lon(), n.lat()});
            if (feat.points.size() < 3) continue;
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
            features_by_zoom[min_zoom].push_back(store.append(feat));
        }
    }

    std::vector<size_t> features_by_zoom[18];
    std::unordered_set<int64_t> processed_areas;
    std::unordered_map<osmium::unsigned_object_id_type, uint8_t> way_to_boundary;
    std::unordered_set<osmium::unsigned_object_id_type> water_relation_ways;

    std::vector<Feature> text_features_vec;
    std::vector<Feature> point_features;

    size_t stats_nodes = 0;
    size_t stats_ways = 0;
    size_t stats_areas = 0;
    size_t stats_filtered = 0;
    size_t stats_points = 0;
    size_t stats_text_labels = 0;
    size_t stats_waterway_labels = 0;

    double bbox_min_lon = 180, bbox_min_lat = 90;
    double bbox_max_lon = -180, bbox_max_lat = -90;

private:
    const ConfigManager& config;
    MappedStore& store;
    int min_zoom_range, max_zoom_range;
    std::unordered_map<std::string, int> road_label_counters;
    std::unordered_map<std::string, int> waterway_label_counters;

    std::string get_highway_type(const std::unordered_map<std::string, std::string>& tags)
    {
        if (tags.count("highway")) return tags.at("highway");
        if (tags.count("railway")) return tags.at("railway");
        if (tags.count("aeroway")) return tags.at("aeroway");
        if (tags.count("aerialway")) return tags.at("aerialway");
        return "";
    }

    std::string get_layer(const std::unordered_map<std::string, std::string>& tags)
    {
        // Water
        if ((tags.count("natural") && (tags.at("natural") == "water" || tags.at("natural") == "bay")) ||
            tags.count("waterway") || tags.count("water") ||
            (tags.count("landuse") && tags.at("landuse") == "reservoir"))
            return "water";

        // Roads / railways (before boundary check — roads can carry boundary tags)
        if (tags.count("highway"))
            return "roads";
        if (tags.count("railway") || tags.count("aerialway"))
            return "roads";

        // Boundaries / places — skip abstract features
        if (tags.count("boundary") || tags.count("admin_level"))
            return "";
        if (tags.count("place"))
        {
            auto p = tags.at("place");
            if (p == "island" || p == "islet") return "islands";
            return "places";
        }

        // Buildings
        if (tags.count("building") || (tags.count("aeroway") && tags.at("aeroway") == "hangar")
            || (tags.count("man_made") && tags.at("man_made") == "tower"))
            return "buildings";

        // Aeroways (aerodrome only — other aeroway goes to infrastructure)
        if (tags.count("aeroway") && tags.at("aeroway") == "aerodrome")
            return "aeroways";

        // Parking
        if (tags.count("amenity") && (tags.at("amenity") == "parking" || tags.at("amenity") == "parking_space"))
            return "parking";

        // Amenities
        if (tags.count("amenity"))
            return "amenities";

        // Pitch
        if (tags.count("leisure") && tags.at("leisure") == "pitch")
            return "pitch";

        // Leisure
        if (tags.count("leisure"))
            return "leisure";

        // Surface (grassland/grass/meadow)
        if ((tags.count("natural") && tags.at("natural") == "grassland") ||
            (tags.count("landuse") && (tags.at("landuse") == "grass" || tags.at("landuse") == "meadow")))
            return "surface";

        // Landuse
        if (tags.count("landuse"))
            return "landuse";

        // Infrastructure (bridges, tunnels, aeroway lines, piers, dams)
        if (tags.count("bridge") || tags.count("tunnel") || tags.count("aeroway") ||
            (tags.count("man_made") && (tags.at("man_made") == "bridge" || tags.at("man_made") == "embankment" || tags.at("man_made") == "pier")))
            return "infrastructure";

        // Terrain (natural features not matched above)
        if (tags.count("natural"))
            return "terrain";

        return "";
    }

    int get_polygon_nibble(const std::unordered_map<std::string, std::string>& tags, const std::string& layer)
    {
        // 0:bg 1:aerodrome 2:landuse base 3:landuse specific 4:surfaces
        // 5:forest/wood 6:infrastructure 7:buildings+water 8-14:roads 15:rail/bridges
        int nibble = 2;
        if (layer == "aeroways") nibble = 1;
        else if (layer == "landuse" || layer == "terrain") nibble = 2;
        else if (layer == "parking") nibble = 2;
        else if (layer == "leisure" || layer == "amenities") nibble = 3;
        else if (layer == "pitch" || layer == "surface") nibble = 4;
        else if (layer == "infrastructure") nibble = 6;
        else if (layer == "buildings") nibble = 7;
        else if (layer == "water") nibble = 7;

        if (tags.count("landuse") && tags.at("landuse") == "commercial")
            nibble = 1;
        if (tags.count("landuse") && tags.at("landuse") == "retail")
            nibble = 3;
        if (tags.count("landuse") && (tags.at("landuse") == "residential" || tags.at("landuse") == "industrial"
            || tags.at("landuse") == "brownfield" || tags.at("landuse") == "construction"))
            nibble = 1;
        if (tags.count("landuse") && (tags.at("landuse") == "farmland" || tags.at("landuse") == "farmyard"))
            nibble = 4;
        if (tags.count("landuse") && tags.at("landuse") == "garages")
            nibble = 3;
        if ((tags.count("landuse") && tags.at("landuse") == "cemetery") ||
            (tags.count("amenity") && tags.at("amenity") == "grave_yard"))
            nibble = 3;
        if (tags.count("leisure") && (tags.at("leisure") == "park" || tags.at("leisure") == "nature_reserve"))
            nibble = 1;
        if (tags.count("leisure") && tags.at("leisure") == "playground")
            nibble = 4;
        if (tags.count("natural") && tags.at("natural") == "beach")
            nibble = 4;
        if (tags.count("natural") && tags.at("natural") == "wetland")
            nibble = 4;
        if (tags.count("landuse") && (tags.at("landuse") == "grass" || tags.at("landuse") == "meadow"
            || tags.at("landuse") == "village_green"))
            nibble = 5;
        if (tags.count("natural") && (tags.at("natural") == "grassland" || tags.at("natural") == "scrub"))
            nibble = 5;
        if ((tags.count("natural") && tags.at("natural") == "wood") ||
            (tags.count("landuse") && tags.at("landuse") == "forest"))
            nibble = 5;
        if (tags.count("amenity") && (tags.at("amenity") == "school" ||
            tags.at("amenity") == "college" || tags.at("amenity") == "university"))
            nibble = 3;
        if (tags.count("aeroway") && tags.at("aeroway") == "aerodrome")
            nibble = 1;
        if (tags.count("aeroway") && tags.at("aeroway") == "apron")
            nibble = 2;
        if (tags.count("leisure") && tags.at("leisure") == "track")
            nibble = 6;
        if ((tags.count("bridge") && (tags.at("bridge") == "yes" || tags.at("bridge") == "viaduct")) ||
            (tags.count("man_made") && tags.at("man_made") == "bridge"))
            nibble = 9;

        return nibble;
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

    void create_road_label(const std::vector<Point>& coords, const std::string& ref,
                           const std::string& old_ref, const std::string& highway_type,
                           uint16_t color_rgb565)
    {
        if (ref.empty()) return;
        if (highway_type != "motorway" && highway_type != "trunk" &&
            highway_type != "primary" && highway_type != "secondary")
            return;

        bool should_create = false;
        if (ref[0] == 'A' || ref[0] == 'N')
            should_create = true;
        else if (ref[0] == 'D')
        {
            try
            {
                int d_number = std::stoi(ref.substr(1));
                if (d_number >= 1000 && d_number <= 1999 && !old_ref.empty() && old_ref[0] == 'N')
                    should_create = true;
            }
            catch (...) {}
        }
        if (!should_create) return;

        road_label_counters[ref]++;
        if (road_label_counters[ref] % constants::ROAD_LABEL_SPACING != 1)
            return;

        int label_index = road_label_counters[ref] / constants::ROAD_LABEL_SPACING;
        int min_zoom;
        if (label_index % 3 == 0) min_zoom = 10;
        else if (label_index % 2 == 0) min_zoom = 12;
        else min_zoom = 13;

        std::vector<Point> candidates;
        for (double ratio : {0.25, 0.5, 0.75})
        {
            int idx = static_cast<int>(coords.size() * ratio);
            idx = std::min(idx, (int)coords.size() - 1);
            candidates.push_back(coords[idx]);
        }

        Feature feat;
        feat.id = 0;
        feat.geom_type = GEOM_TEXT;
        feat.color_rgb565 = utils::darken_rgb565(color_rgb565);
        feat.bg_color_rgb565 = utils::lighten_rgb565(color_rgb565);
        feat.border_color_rgb565 = color_rgb565;
        feat.zoom_priority = utils::pack_zoom_priority(min_zoom, 15);
        feat.font_size = 2;
        std::string ref_trunc = ref.substr(0, 32);
        feat.text.assign(ref_trunc.begin(), ref_trunc.end());
        feat.population = 0;
        feat.points.push_back(candidates[1]); // default: 50%
        feat.ring_ends.push_back(1);
        feat.coords_candidates = candidates;
        text_features_vec.push_back(std::move(feat));
    }

    void create_waterway_label(const std::vector<Point>& coords,
                               const std::string& waterway_type,
                               const std::string& name)
    {
        if (name.empty()) return;
        if (waterway_type != "river" && waterway_type != "stream" && waterway_type != "canal")
            return;

        int spacing;
        if (waterway_type == "river") spacing = 4;
        else if (waterway_type == "canal") spacing = 8;
        else spacing = 12;

        waterway_label_counters[name]++;
        if (waterway_label_counters[name] % spacing != 1)
            return;

        int label_idx = waterway_label_counters[name] / spacing;

        int min_zoom;
        if (waterway_type == "river") min_zoom = 10;
        else if (waterway_type == "canal") min_zoom = 12;
        else min_zoom = 14;

        if (min_zoom > max_zoom_range) return;
        if (coords.size() < 2) return;

        // Alternate sub-segment position to spread labels along the waterway
        double start_ratio, end_ratio;
        switch (label_idx % 3)
        {
            case 0: start_ratio = 0.10; end_ratio = 0.55; break;
            case 1: start_ratio = 0.45; end_ratio = 0.90; break;
            default: start_ratio = 0.25; end_ratio = 0.75; break;
        }
        size_t i_start = static_cast<size_t>(coords.size() * start_ratio);
        size_t i_end = static_cast<size_t>(coords.size() * end_ratio);
        if (i_end <= i_start) i_end = i_start + 1;
        if (i_end >= coords.size()) i_end = coords.size() - 1;
        std::vector<Point> path(coords.begin() + i_start, coords.begin() + i_end + 1);
        if (path.size() < 2) return;

        Feature feat;
        feat.id = 0;
        feat.geom_type = GEOM_TEXT_LINE;
        feat.color_rgb565 = 0x4C16; // #4a80b0 dark blue
        feat.bg_color_rgb565 = 0;
        feat.border_color_rgb565 = 0;
        feat.zoom_priority = utils::pack_zoom_priority(min_zoom, 12);
        feat.font_size = 0;
        std::string name_trunc = name.substr(0, 255);
        feat.text.assign(name_trunc.begin(), name_trunc.end());
        feat.population = 0;
        feat.points = std::move(path);
        feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
        text_features_vec.push_back(std::move(feat));
        stats_waterway_labels++;
    }
};

} // namespace nav
