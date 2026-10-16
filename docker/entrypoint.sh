#!/bin/bash
set -e

# Compile both generators on the first run when they are missing
if [ ! -x build/nav_generator ] || [ ! -x build/route_generator ]; then
    echo "Building nav_generator and route_generator..."
    cmake -S . -B build
    cmake --build build -j"$(nproc)"
fi

export PATH="$PWD/build:$PATH"
exec "$@"
