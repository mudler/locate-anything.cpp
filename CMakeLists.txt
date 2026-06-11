cmake_minimum_required(VERSION 3.18)
project(locate_anything VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LA_BUILD_TESTS    "Build tests"            OFF)
option(LA_BUILD_CLI      "Build locate-anything-cli"                       ON)
option(LA_SHARED         "Build liblocate_anything as a shared library"    OFF)
option(LA_GGML_CUDA      "ggml CUDA backend"      OFF)
option(LA_GGML_METAL     "ggml Metal backend"     OFF)
option(LA_GGML_VULKAN    "ggml Vulkan backend"    OFF)

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Build ggml as static archives. With CMAKE_POSITION_INDEPENDENT_CODE=ON these
# link straight into liblocate_anything.so (LA_SHARED=ON) so the resulting .so
# has no external libggml runtime dependency. Our own targets use explicit
# STATIC/SHARED keywords below, so this does not change locate_anything's type.
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
# LA_GGML_* are authoritative in both directions: toggling an option OFF on a
# reconfigure also forces the ggml cache variable back OFF.
set(GGML_CUDA   ${LA_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL  ${LA_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN ${LA_GGML_VULKAN} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---------- liblocate_anything ----------
set(LA_SOURCES
    src/model_loader.cpp
    src/tokenizer.cpp
    src/prompt.cpp
    src/backend.cpp
    src/vit_rope.cpp
    src/vit_posemb.cpp
    src/vit_encoder.cpp
    src/projector.cpp
    src/lm.cpp
    src/qwen2.cpp
    src/boxes.cpp
    src/mtp.cpp
    src/image_io.cpp
    src/pil_resize.cpp
    src/visualize.cpp
    src/engine.cpp
    src/la_capi.cpp
    src/cli.cpp
    src/quantize.cpp
)
# Sources are created across M2 tasks; until they all exist, stub .cpp files keep the lib building.
if(LA_SHARED)
    add_library(locate_anything SHARED ${LA_SOURCES})
else()
    add_library(locate_anything STATIC ${LA_SOURCES})
endif()
target_include_directories(locate_anything
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_SOURCE_DIR}/third_party/stb)
target_link_libraries(locate_anything PUBLIC ggml)

if(LA_BUILD_CLI AND EXISTS ${CMAKE_SOURCE_DIR}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()

if(LA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
message(STATUS "locate-anything.cpp configured (lib + tests)")
