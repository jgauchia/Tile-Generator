# OSM Vector Tile Generator

This repository contains a Python script for generating vector map tiles from OpenStreetMap (OSM) data in a custom binary format.  
The generated tiles are highly compact and optimized for fast rendering in custom map applications.  
Features and styling are controlled via a JSON configuration file.

---

## What Does the Script Do?

- Extracts relevant geometries and attributes from OSM PBF files using `ogr2ogr`.
- Filters and merges data into a simplified GeoJSON.
- Processes the GeoJSON in streaming mode to avoid memory overhead.
- Assigns features to tiles for a range of zoom levels.
- Writes tile data in a custom binary format for fast decoding and rendering.
- Uses a user-defined `features.json` for styling and feature selection.

---

## Binary Tile File Format

Generated `.bin` files contain encoded drawing commands and geometry for each tile.

- See full specification: [/docs/bin_tile_format.md](/docs/bin_tile_format.md)

---

## Features JSON Format

The configuration file (`features.json`) defines feature selection, styling, and priorities.

- See full specification: [/docs/features_json_format.md](/docs/features_json_format.md)

---

## Script Usage

### Main Script

The main script is `tile_generator.py`.  
It takes three required arguments:

- OSM PBF file
- Output directory for the tiles
- JSON configuration file (`features.json`)

**Example:**
```sh
python tile_generator.py planet.osm.pbf tiles/ features.json --zoom 6-17 --max-file-size 128
```

---

### Arguments

| Argument           | Description                                                    |
|--------------------|----------------------------------------------------------------|
| pbf_file           | Path to the input OSM PBF file                                 |
| output_dir         | Directory where generated tiles will be saved                  |
| config_file        | Path to the features JSON configuration                        |
| --zoom             | Zoom level or range (e.g. `12` or `6-17`)                      |
| --max-file-size    | Max tile file size in KB (default: 128 KB, optional)           |

---

## Dependencies

The following Python packages are required:

- `osmium`  
- `shapely`  
- `fiona`  
- `ijson`  
- `tqdm`  
- `psutil`  

You also need the command-line tool `ogr2ogr` (from GDAL).

### Installing Python Dependencies

You can install all required Python packages with:

```sh
pip install osmium shapely fiona ijson tqdm psutil
```

### Installing GDAL (`ogr2ogr`)

- **On Ubuntu/Debian:**
    ```sh
    sudo apt-get update
    sudo apt-get install gdal-bin
    ```
- **On MacOS (Homebrew):**
    ```sh
    brew install gdal
    ```
- **On Windows:**  
    Download [GDAL binaries](https://gdal.org/download.html) and add `ogr2ogr.exe` to your PATH.

---

## How to Use

1. Prepare your `features.json` configuration (see [/docs/features_json_format.md](/docs/features_json_format.md)).
2. Run the script with your OSM PBF file and desired zoom range.
3. Find the generated tiles in the output directory, organized as `{zoom}/{x}/{y}.bin`.

---

## Viewer Script

A viewer script `tile_viewer.py` is included for visualizing the generated map tiles.  
This script can display both vector tiles (`.bin`) and raster tiles (`.png`).  
See its documentation for usage and details:

- [/docs/tile_viewer.md](/docs/tile_viewer.md)

---

## Documentation

- [Binary Tile File Format](/docs/bin_tile_format.md)
- [Features JSON Format](/docs/features_json_format.md)
- [Tile Viewer Documentation](/docs/tile_viewer.md)

---

## Example

**Sample invocation:**
```sh
python tile_generator.py spain-latest.osm.pbf tiles/ features.json --zoom 8-16
```

---

## License

This project is released under the MIT License.

---