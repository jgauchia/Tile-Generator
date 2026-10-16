# Base Map (`basemap.geojson`)

The page draws its base map from `web/static/basemap.geojson`, a vendored **16.3 MB** GeoJSON of
[Natural Earth](https://www.naturalearthdata.com/) 1:10m (public domain): **10,704 features** in three
layers.

| `layer` | Features | Drawn as | When |
|---|---|---|---|
| `land` | 11 | filled polygons | always |
| `country` | 515 | solid lines | always |
| `region` | 10,178 | dashed lines | from the `min_zoom` Natural Earth gives each one (2–11) |

Antarctica is left out (everything south of −60° is dropped), so nobody tries to generate a map down
there. The file is committed and served as a static asset by Flask; **nothing downloads or rebuilds it
at run time, and no setup step needs it**. Rebuilding it is only needed to pick up a new Natural Earth
release or to change a layer, and it writes the file this repository ships.

The Geofabrik region polygons are **not** used for the base map: they follow administrative borders
including territorial waters and are simplified for a world view, so they drift several km at high zoom.
They are painted as selectable regions, never outlined, and the base map stays generalised.

---

## Rebuilding it

Needs `gdal-bin` (`ogr2ogr`) and Python 3. Run it from a scratch directory: the intermediate files land
there and only the last line writes into the repository.

```bash
base=https://naciscdn.org/naturalearth
curl -O $base/10m/physical/ne_10m_land.zip
curl -O $base/10m/cultural/ne_10m_admin_0_boundary_lines_land.zip
curl -O $base/10m/cultural/ne_10m_admin_1_states_provinces_lines.zip
unzip 'ne_10m_*.zip'
ogr2ogr -f GeoJSON -lco COORDINATE_PRECISION=3 land.geojson ne_10m_land.shp
ogr2ogr -f GeoJSON -lco COORDINATE_PRECISION=3 country.geojson ne_10m_admin_0_boundary_lines_land.shp
ogr2ogr -f GeoJSON -lco COORDINATE_PRECISION=3 -select MIN_ZOOM \
  region.geojson ne_10m_admin_1_states_provinces_lines.shp
python3 - <<'EOF'
import json


def north_of(part):
    """Highest latitude of a polygon part (a list of rings) or of a line."""
    points = [point for ring in part for point in ring] if isinstance(part[0][0], list) else part
    return max(point[1] for point in points)


def without_antarctica(geometry):
    """Antarctica is left out: nobody generates a map down there."""
    if geometry["type"] in ("Polygon", "LineString"):
        return geometry if north_of(geometry["coordinates"]) > -60 else None
    parts = [part for part in geometry["coordinates"] if north_of(part) > -60]
    if not parts:
        return None
    if len(parts) == 1:
        return {"type": "LineString" if geometry["type"] == "MultiLineString" else "Polygon",
                "coordinates": parts[0]}
    return {"type": geometry["type"], "coordinates": parts}


features = []
for layer, path, hasZoom in (("land", "land.geojson", False),
                             ("country", "country.geojson", False),
                             ("region", "region.geojson", True)):
    for feature in json.load(open(path))["features"]:
        geometry = feature.get("geometry")
        if not geometry or geometry.get("coordinates") is None:
            continue
        properties = {"layer": layer}
        if hasZoom:
            # Each province line is drawn from the zoom Natural Earth gives it.
            properties["min_zoom"] = int(feature["properties"]["MIN_ZOOM"])
        geometry = without_antarctica(geometry)
        if geometry is None:
            continue
        features.append({"type": "Feature", "properties": properties,
                         "geometry": geometry})
out = {"type": "FeatureCollection", "features": features}
open("web/static/basemap.geojson", "w").write(json.dumps(out, separators=(",", ":")))
EOF
```

The three layers become one `FeatureCollection`; `land` and `country` are drawn by the page at every
zoom and `region` only from its own `min_zoom` (`web/static/app.js`).
