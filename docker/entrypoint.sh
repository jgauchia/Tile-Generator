#!/bin/bash
set -e

# build/ belongs to the host, so the container builds in a directory of its own
# (web/pipeline.py reads it from TILE_GENERATOR_BUILD_DIR).
BUILD_DIR="$PWD/build-docker"

# Compile both generators on the first run when they are missing
if [ ! -x "$BUILD_DIR/nav_generator" ] || [ ! -x "$BUILD_DIR/route_generator" ]; then
    echo "Building nav_generator and route_generator into $BUILD_DIR..."
    cmake -S . -B "$BUILD_DIR"
    cmake --build "$BUILD_DIR" -j"$(nproc)"
fi

export TILE_GENERATOR_BUILD_DIR="$BUILD_DIR"
export PATH="$BUILD_DIR:$PATH"
exec "$@"
