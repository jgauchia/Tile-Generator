/**
 * @file config_manager.hpp
 * @brief Manages feature styling and zoom configuration from JSON.
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
                cfg.min_zoom = it.value()["zoom"];
            if (it.value().contains("color"))
                cfg.color_rgb565 = utils::hex_to_rgb565(it.value()["color"]);
            if (it.value().contains("priority"))
                cfg.priority = it.value()["priority"];
            if (it.value().contains("width"))
                cfg.width_meters = it.value()["width"];
            
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
        FeatureConfig best_cfg;

        for (const auto& [k, v] : tags)
        {
            std::string exact = k + "=" + v;
            if (config_map.count(exact))
                return config_map.at(exact);
            if (config_map.count(k))
                best_cfg = config_map.at(k);
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
