/**
 * @file config_manager.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Manages feature styling and zoom configuration from JSON.
 * @version 1.0.3
 * @date 2026-10
 */

#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <unordered_map>
#include <iostream>
#include "utils.hpp"

namespace nav {

/**
 * @struct FeatureConfig
 * @brief Style and zoom rules for a specific OSM tag.
 */
struct FeatureConfig
{
    int min_zoom = 6;           ///< Minimum visibility zoom level
    uint16_t color_rgb565 = 0xFFFF; ///< Color in RGB565 format
    int priority = 50;          ///< Rendering priority (higher = front)
    float width_meters = 0.0f;  ///< Fixed width in meters (if present)
    std::map<int, uint8_t> zoom_widths; ///< Table of minimum pixel widths per zoom
    // Which fields were explicitly present in the JSON entry. Qualifier tags
    // (bicycle, mtb:scale, ...) only patch the fields they declare, so an
    // entry with just "widths" bumps the line thickness without recoloring.
    bool has_zoom = false;
    bool has_color = false;
    bool has_priority = false;
    bool has_width_meters = false;
};

/**
 * @class ConfigManager
 * @brief Loads and provides styling configurations for OSM features.
 */
class ConfigManager
{
public:
    /**
     * @brief Loads configuration from a JSON file.
     * @param path Path to the features.json file.
     * @return true if loaded successfully, false otherwise.
     */
    bool load(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open())
            return false;
        
        nlohmann::json j;
        f >> j;
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            if (it.key().substr(0, 1) == "_")
                continue;
            
            FeatureConfig cfg;
            if (it.value().contains("zoom"))
            {
                cfg.min_zoom = it.value()["zoom"];
                cfg.has_zoom = true;
            }
            if (it.value().contains("color"))
            {
                cfg.color_rgb565 = utils::hex_to_rgb565(it.value()["color"]);
                cfg.has_color = true;
            }
            if (it.value().contains("priority"))
            {
                cfg.priority = it.value()["priority"];
                cfg.has_priority = true;
            }
            if (it.value().contains("width"))
            {
                cfg.width_meters = it.value()["width"];
                cfg.has_width_meters = true;
            }
            
            if (it.value().contains("widths"))
            {
                for (auto& w_it : it.value()["widths"].items())
                    cfg.zoom_widths[std::stoi(w_it.key())] = static_cast<uint8_t>(w_it.value());
            }
            
            config_map[it.key()] = cfg;
        }
        std::cout << "Loaded " << config_map.size() << " features from config." << std::endl;
        return true;
    }

    /**
     * @brief Finds the best matching configuration for a set of tags.
     * @param tags Dictionary of OSM tags (key=value).
     * @return FeatureConfig with the style and zoom settings.
     */
    FeatureConfig get_config(const std::unordered_map<std::string, std::string>& tags) const
    {
        // Tag key priority: most specific wins over generic qualifiers
        static const std::vector<std::string> key_priority = {
            "highway", "railway", "aeroway", "aerialway",
            "waterway", "water", "natural", "landuse",
            "building", "amenity", "leisure", "man_made",
            "boundary", "admin_level", "place",
            "bridge", "tunnel", "surface", "layer"
        };

        FeatureConfig best_cfg;
        bool found = false;

        // First pass: find best base config from priority keys (don't return early,
        // so qualifier tags can override afterwards)
        for (const auto& key : key_priority)
        {
            auto it = tags.find(key);
            if (it == tags.end()) continue;
            std::string exact = key + "=" + it->second;
            if (config_map.count(exact))
            {
                best_cfg = config_map.at(exact);
                found = true;
                break; // most specific primary key wins
            }
            if (!found && config_map.count(key))
            {
                best_cfg = config_map.at(key);
                found = true;
            }
        }

        // Second pass: qualifier tags override primary styling.
        // Deterministic priority order (earlier = higher priority, wins later):
        static const std::vector<std::string> qualifier_priority = {
            "bicycle", "mtb:scale:uphill", "mtb:scale:imba", "mtb:scale", "sac_scale", "trail_visibility"
        };

        // Collect matching qualifier configs with their priority rank
        struct QualMatch { int rank; FeatureConfig cfg; };
        std::vector<QualMatch> matches;

        int qi = 0;
        for (const auto& qkey : qualifier_priority)
        {
            ++qi;
            auto it = tags.find(qkey);
            if (it == tags.end()) continue;
            std::string exact = qkey + "=" + it->second;
            if (config_map.count(exact))
            {
                matches.push_back({qi, config_map.at(exact)});
            }
            else if (config_map.count(qkey))
            {
                matches.push_back({qi, config_map.at(qkey)});
            }
        }

        // Highest priority (lowest rank number) wins per field.
        // All matching qualifiers are applied in rank order (later ranks
        // first, earlier ranks last so they win), patching only the fields
        // they declare. An entry that sets only "widths" (e.g. thicker
        // bike-allowed trails) keeps the base color, and a way matching both
        // "mtb:scale=X" (color) and "bicycle=yes" (width) gets both patches.
        for (auto it = matches.rbegin(); it != matches.rend(); ++it)
        {
            const FeatureConfig& q = it->cfg;
            if (q.has_zoom)          best_cfg.min_zoom = q.min_zoom;
            if (q.has_color)         best_cfg.color_rgb565 = q.color_rgb565;
            if (q.has_priority)      best_cfg.priority = q.priority;
            if (q.has_width_meters)  best_cfg.width_meters = q.width_meters;
            if (!q.zoom_widths.empty()) best_cfg.zoom_widths = q.zoom_widths;
        }

        return best_cfg;
    }

    /**
     * @brief Checks if at least one tag key or key=value pair exists in the config.
     * @param key Tag key.
     * @param value Tag value.
     * @return true if the feature is defined in the configuration.
     */
    bool is_interesting(const std::string& key, const std::string& value) const
    {
        return config_map.count(key) || config_map.count(key + "=" + value);
    }

private:
    std::unordered_map<std::string, FeatureConfig> config_map;
};

} // namespace nav
