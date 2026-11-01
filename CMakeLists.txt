cmake_minimum_required(VERSION 3.14)
project(nav_generator_cpp VERSION 1.0.2)

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

# nlohmann-json: src/config_manager.hpp always needs it, so fall back to a
# fetched copy when the system package is not installed.
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found, fetching it with FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

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

target_link_libraries(nav_generator PRIVATE nlohmann_json::nlohmann_json)

# Compiler optimization flags
if(MSVC)
    target_compile_options(nav_generator PRIVATE /O2 /W3)
else()
    target_compile_options(nav_generator PRIVATE -Wall -Wextra -O3 -march=native)
endif()

# Route graph generator
add_executable(route_generator
    src/route_main.cpp
)

target_link_libraries(route_generator
    PRIVATE
    ${GEOS_C_LIBRARY}
    ${ZLIB_LIBRARIES}
    ${EXPAT_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${GDAL_LIBRARIES}
)

target_link_libraries(route_generator PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(route_generator PRIVATE /O2 /W3)
else()
    target_compile_options(route_generator PRIVATE -Wall -Wextra -O3 -march=native)
endif()
