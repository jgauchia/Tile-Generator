cmake_minimum_required(VERSION 3.10)
project(nav_generator_cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ZLIB REQUIRED)
find_package(EXPAT REQUIRED)
find_package(BZip2 REQUIRED)
find_package(Threads REQUIRED)
find_package(GDAL REQUIRED)

# Manually find GEOS
find_path(GEOS_INCLUDE_DIR geos_c.h)
find_library(GEOS_C_LIBRARY geos_c)

if(NOT GEOS_INCLUDE_DIR OR NOT GEOS_C_LIBRARY)
    message(FATAL_ERROR "GEOS (geos_c) not found. Please install libgeos-dev")
endif()

message(STATUS "Found GEOS include: ${GEOS_INCLUDE_DIR}")
message(STATUS "Found GEOS library: ${GEOS_C_LIBRARY}")

# nlohmann-json
find_package(nlohmann_json QUIET)

# Include directories
include_directories(
    ${GEOS_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    ${EXPAT_INCLUDE_DIRS}
    ${BZIP2_INCLUDE_DIR}
    ${GDAL_INCLUDE_DIRS}
    src
)

# Main executable
add_executable(nav_generator
    src/main.cpp
)

# Link libraries
target_link_libraries(nav_generator
    PRIVATE
    ${GEOS_C_LIBRARY}
    ${ZLIB_LIBRARIES}
    ${EXPAT_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${GDAL_LIBRARIES}
)

if(nlohmann_json_FOUND)
    target_link_libraries(nav_generator PRIVATE nlohmann_json::nlohmann_json)
endif()

# Compiler optimization flags
if(MSVC)
    target_compile_options(nav_generator PRIVATE /O2 /W3)
else()
    target_compile_options(nav_generator PRIVATE -Wall -Wextra -O3 -march=native)
endif()
