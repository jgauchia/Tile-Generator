/**
 * @file shapefile_loader.hpp
 * @author Jordi Gauchía (jgauchia @jgauchia.com)
 * @brief Loads pre-computed ocean water polygons from shapefiles (osmdata.openstreetmap.de).
 * @version 1.0.3
 * @date 2026-10
 */

#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>
#include "nav_types.hpp"
#include "mapped_store.hpp"
#include "utils.hpp"

namespace nav {

inline size_t load_water_polygons(const std::string& shapefile_path,
                                  MappedStore& store,
                                  std::vector<size_t> (&features_by_zoom)[18],
                                  uint16_t water_color,
                                  uint8_t min_zoom,
                                  double bbox_min_lon, double bbox_min_lat,
                                  double bbox_max_lon, double bbox_max_lat)
{
    GDALAllRegister();

    GDALDataset* ds = (GDALDataset*)GDALOpenEx(
        shapefile_path.c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY,
        nullptr, nullptr, nullptr);
    if (!ds)
    {
        std::cerr << "Error: Could not open shapefile: " << shapefile_path << std::endl;
        return 0;
    }

    OGRLayer* layer = ds->GetLayer(0);
    if (!layer)
    {
        std::cerr << "Error: No layer found in shapefile" << std::endl;
        GDALClose(ds);
        return 0;
    }

    if (bbox_min_lon >= bbox_max_lon || bbox_min_lat >= bbox_max_lat)
    {
        std::cerr << "Error: invalid water clip bbox: " << bbox_min_lon << "," << bbox_min_lat
                  << " -> " << bbox_max_lon << "," << bbox_max_lat << std::endl;
        GDALClose(ds);
        return 0;
    }

    // Spatial filtering only selects whole features: ocean polygons can still
    // span the world. Clip them to the PBF bbox before they reach the tile grid.
    OGRLinearRing bbox_ring;
    bbox_ring.addPoint(bbox_min_lon, bbox_min_lat);
    bbox_ring.addPoint(bbox_max_lon, bbox_min_lat);
    bbox_ring.addPoint(bbox_max_lon, bbox_max_lat);
    bbox_ring.addPoint(bbox_min_lon, bbox_max_lat);
    bbox_ring.addPoint(bbox_min_lon, bbox_min_lat);
    OGRPolygon bbox_poly;
    bbox_poly.addRing(&bbox_ring);

    layer->SetSpatialFilterRect(bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat);
    layer->ResetReading();
    size_t count = 0;
    static int64_t ocean_id_base = -1000000000;

    auto process_polygon = [&](OGRPolygon* poly)
    {
        Feature feat;
        feat.id = ocean_id_base--;
        feat.geom_type = GEOM_POLYGON;
        feat.color_rgb565 = water_color;
        feat.width_meters = 0;
        feat.layer = "water";
        feat.zoom_priority = utils::pack_zoom_priority(min_zoom, 7);

        OGRLinearRing* ext = poly ? poly->getExteriorRing() : nullptr;
        if (!ext || ext->getNumPoints() < 4)
            return;

        for (int i = 0; i < ext->getNumPoints(); ++i)
            feat.points.push_back({ext->getX(i), ext->getY(i)});
        feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));

        for (int r = 0; r < poly->getNumInteriorRings(); ++r)
        {
            OGRLinearRing* inner = poly->getInteriorRing(r);
            if (!inner || inner->getNumPoints() < 4)
                continue;
            for (int i = 0; i < inner->getNumPoints(); ++i)
                feat.points.push_back({inner->getX(i), inner->getY(i)});
            feat.ring_ends.push_back(static_cast<uint32_t>(feat.points.size()));
        }

        if (min_zoom < 18)
            features_by_zoom[min_zoom].push_back(store.append(feat));
        ++count;
    };

    std::function<void(OGRGeometry*)> process_geometry = [&](OGRGeometry* geom)
    {
        if (!geom || geom->IsEmpty())
            return;
        OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
        if (gtype == wkbPolygon)
        {
            process_polygon(static_cast<OGRPolygon*>(geom));
        }
        else if (gtype == wkbMultiPolygon || gtype == wkbGeometryCollection)
        {
            OGRGeometryCollection* collection = static_cast<OGRGeometryCollection*>(geom);
            for (int i = 0; i < collection->getNumGeometries(); ++i)
                process_geometry(collection->getGeometryRef(i));
        }
    };

    OGRFeature* ogr_feat;
    while ((ogr_feat = layer->GetNextFeature()) != nullptr)
    {
        OGRGeometry* geom = ogr_feat->GetGeometryRef();
        if (!geom)
        {
            OGRFeature::DestroyFeature(ogr_feat);
            continue;
        }

        OGRwkbGeometryType gtype = wkbFlatten(geom->getGeometryType());
        if (gtype != wkbPolygon && gtype != wkbMultiPolygon)
        {
            OGRFeature::DestroyFeature(ogr_feat);
            continue;
        }

        OGRGeometry* clipped = geom->Intersection(&bbox_poly);
        if (clipped && !clipped->IsEmpty())
            process_geometry(clipped);
        delete clipped;

        OGRFeature::DestroyFeature(ogr_feat);
    }

    GDALClose(ds);
    return count;
}

} // namespace nav
