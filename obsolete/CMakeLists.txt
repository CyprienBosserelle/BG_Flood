# CMake build for BG-Flood
# Based on: https://devblogs.nvidia.com/building-cuda-applications-cmake/

cmake_minimum_required(VERSION 3.9.1 FATAL_ERROR)
project(bggpu LANGUAGES CXX CUDA)

# option to use Catalyst (enable with cmake option "-DUSE_CATALYST=ON")
option(USE_CATALYST "Link with Catalyst" OFF)

# option for running tests - include(CTest) adds the option -DBUILD_TESTING
# that is set to ON by default (to disable tests use -DBUILD_TESTING=OFF)
include(CTest)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


# default to a release (optimised) build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# compile device code in 64-bit mode
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64")

# There isn't a really good way to find NetCDF with CMake
# This should automatically find NetCDF on NeSI platforms (using EasyBuild) and on systems with
#   "nc-config" in PATH
# If NetCDF can't be found specify "-DNETCDF_DIR=/path/to/netcdf/installation" to the directory
#   containing include/netcdf.h and lib/libnetcdf.*
set(NETCDF_DIR "" CACHE PATH "Directory containing the NetCDF installation")
if (NETCDF_DIR STREQUAL "")
    # if NETCDF_DIR isn't set, first look for EBROOTNETCDF environment variable, then look for
    #   nc-config program
    if ("$ENV{EBROOTNETCDF}" STREQUAL "")
        find_program(NC_CONFIG nc-config)
        if (NOT NC_CONFIG)
            message(FATAL_ERROR "Could not find NetCDF library: try setting -DNETCDF_DIR or adding nc-config to your PATH")
        endif()
        execute_process(COMMAND "${NC_CONFIG} --prefix" OUTPUT_VARIABLE NETCDF_DIR)
        message(STATUS "NetCDF directory from nc-config: ${NETCDF_DIR}")
    else()
        set(NETCDF_DIR "$ENV{EBROOTNETCDF}")
        message(STATUS "NetCDF directory from easybuild: ${NETCDF_DIR}")
    endif()
else()
    # NETCDF_DIR already set
    message(STATUS "Using NetCDF directory: ${NETCDF_DIR}")
endif()

# look for NetCDF library
find_library(NETCDF_LIBRARIES NAMES netcdf PATHS ${NETCDF_DIR}/lib)
if (NOT NETCDF_LIBRARIES)
    message(FATAL_ERROR "Could not find NetCDF library: try setting the path to NetCDF with: -DNETCDF_DIR=/path/to/netcdf")
endif()

# look for NetCDF include dir
find_path(NETCDF_INCLUDE_DIR NAMES netcdf.h PATHS ${NETCDF_DIR}/include)
if (NOT NETCDF_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find NetCDF include dir: try setting the path to NetCDF with: -DNETCDF_DIR=/path/to/netcdf")
endif()

# BG-Flood source files
set(bggpu_SRCS Bas_Cart_CPU.cpp BasCart_gpu.cu read_input.cpp writenetcdf.cpp)

# catalyst source files
if(USE_CATALYST)
  find_package(ParaView 5.5 REQUIRED COMPONENTS vtkPVPythonCatalyst)
  include("${PARAVIEW_USE_FILE}")
  set(catalyst_SRCS vtkCPVTKOutputPipeline.cpp catalyst_adaptor.cpp)
  add_library(BG-Flood_catalyst ${catalyst_SRCS})
  target_compile_definitions(BG-Flood_catalyst PUBLIC "USE_CATALYST")
  target_link_libraries(BG-Flood_catalyst PUBLIC vtkPVPythonCatalyst ${VTK_LIBRARIES})
endif()

# need to set these files as CUDA to get it to build with CUDA 8 and GCC 5.4 (Mahuika)
set_source_files_properties(read_input.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(writenetcdf.cpp PROPERTIES LANGUAGE CUDA)


# add the executable
add_executable(BG-Flood ${bggpu_SRCS})

# linking with netCDF
target_link_libraries(BG-Flood ${NETCDF_LIBRARIES})
target_include_directories(BG-Flood PRIVATE ${NETCDF_INCLUDE_DIR})

# linking cuda libraries
target_include_directories(
    BG-Flood
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


# linking with catalyst
if(USE_CATALYST)
  target_link_libraries(BG-Flood BG-Flood_catalyst)
endif()

# Note: these lines copied from https://devblogs.nvidia.com/building-cuda-applications-cmake/
#       not sure if they are actually needed as I don't have a Mac with Nvidia GPU to test
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath,
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET BG-Flood
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

if (BUILD_TESTING)
  add_subdirectory(test)
endif()
