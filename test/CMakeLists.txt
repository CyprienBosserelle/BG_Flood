cmake_minimum_required(VERSION 3.9.1 FATAL_ERROR)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(USE_CATALYST)
  find_package(ParaView 5.5 REQUIRED COMPONENTS vtkPVPythonCatalyst)
  include("${PARAVIEW_USE_FILE}")
  set(Test_SRCS catch_main.cpp catalyst_adaptor_UnitTest.cpp ../catalyst_adaptor.cpp ../vtkCPVTKOutputPipeline.cpp)
  add_definitions("-DUSE_CATALYST")
  add_executable (catalystTests ${Test_SRCS})
  target_include_directories(catalystTests PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(catalystTests vtkPVPythonCatalyst ${VTK_LIBRARIES})
endif()

option(BUILD_TESTING "Build Testing" OFF)
if(BUILD_TESTING)
  include(CTest)
  if(USE_CATALYST)
    add_test (NAME Tests COMMAND catalystTests)
  endif()
endif()
