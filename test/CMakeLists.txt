# include Catch CMake integration module
include("${CMAKE_CURRENT_SOURCE_DIR}/Catch2/cmake/Catch.cmake")

if(USE_CATALYST)
  set(Test_SRCS catch_main.cpp catalyst_adaptor_UnitTest.cpp vtkCPVTKOutputPipeline_UnitTest.cpp)
  add_executable(catalystTests ${Test_SRCS})
  target_include_directories(catalystTests PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(catalystTests BasCGPU_catalyst)
  catch_discover_tests(catalystTests)
endif()
