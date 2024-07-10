

cmake_minimum_required(VERSION 3.17)

project(octree-miniapp CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

enable_language(CUDA)
find_package(CUDAToolkit)
set(CMAKE_CUDA_STANDARD 17)

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_executable(neighbor_search neighbor_search.cu)

add_executable(octree_gpu octree.cu)
set_source_files_properties(octree.cu PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)

add_executable(octree_cpu octree.cpp)
target_link_libraries(octree_cpu PRIVATE OpenMP::OpenMP_CXX)
