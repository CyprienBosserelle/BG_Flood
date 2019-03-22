cmake_minimum_required(VERSION 3.12.2)

project(bggpu CXX CUDA)

set(USE_CATALYST ON CACHE BOOL "Link with Catalyst")

if(USE_CATALYST)
  find_package(ParaView 5.5 REQUIRED COMPONENTS vtkPVPythonCatalyst)
  include("${PARAVIEW_USE_FILE}")
  set(bggpu_SRCS Bas_Cart_CPU.cpp BasCart_gpu.cu read_input.cpp writenetcdf.cpp catalyst_adaptor.cpp)
  add_definitions("-DUSE_CATALYST")
else()
  set(bggpu_SRCS Bas_Cart_CPU.cpp BasCart_gpu.cu read_input.cpp writenetcdf.cpp)
endif()

add_executable(BasCGPU ${bggpu_SRCS})
set_property(TARGET BasCGPU PROPERTY CXX_STANDARD 11)

find_library(nclib libnetcdf.so PATHS $EBROOTNETCDF/lib)

target_link_libraries(BasCGPU ${nclib})

if(USE_CATALYST)
  target_link_libraries(BasCGPU vtkPVPythonCatalyst)
endif()
