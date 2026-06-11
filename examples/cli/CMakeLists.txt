add_executable(locate-anything-cli main.cpp)
target_link_libraries(locate-anything-cli PRIVATE locate_anything ggml)
target_include_directories(locate-anything-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party/stb)
