# NAV Generator C++ (High Performance)

This is the high-performance implementation of the NAV tile generator, designed to process large OSM PBF files (countries or continents) with minimal RAM usage and maximum CPU utilization.

## Performance vs Python

Based on the Catalunya (244 MB PBF) benchmark:
- **RAM Usage:** 2 GB (C++) vs 20 GB (Python + Swap).
- **Time (Z6-17):** ~40-60 mins (C++ with GEOS Merging) vs 7+ hours (Python).
- **Parity:** 100% feature parity including complex multipolygons and polygon merging.

## Dependencies

Install the required libraries on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake libosmium2-dev libgeos-dev \
                 nlohmann-json3-dev libbz2-dev zlib1g-dev \
                 libexpat1-dev libprotozero-dev
```

## Building

The project uses CMake and requires a C++17 compatible compiler.

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Usage

The C++ version maintains full CLI compatibility with the Python version:

```bash
./nav_generator <input.pbf> <output_dir> <features.json> [--zoom 6-17]
```

## Key Features

- **Multi-pass Extraction:** Correctly handles OSM Relations and Multipolygons.
- **GEOS Integration:** Professional-grade polygon merging (UnaryUnion) and perfect tile clipping.
- **Douglas-Peucker:** Intelligent geometry simplification based on zoom level.
- **Optimized Multithreading:** Parallel tile processing with thread-local GEOS handles.
- **Compact Binary:** Full support for Delta Encoding, VarInt, and ZigZag compression.

## Current Status

- [x] High-speed PBF Parsing (Osmium)
- [x] Feature Deduplication (Way/Area)
- [x] Complex Multipolygon support
- [x] Spatial Merging (GEOS)
- [x] Perfect Clipping (Lines & Polygons)
- [x] Zoom-dependent Simplification
- [x] Real-time Progress Monitoring (t/s, MB, %)