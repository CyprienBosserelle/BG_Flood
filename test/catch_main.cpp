#define CATCH_CONFIG_RUNNER
#include "Catch2/catch.hpp"
#include "catalyst_adaptor.h"
#include <iostream>

int main( int argc, char * argv[] )
{
  // Construct Catalyst Adaptor singleton here - the adaptor can only
  // be initialised once
  catalystAdaptor& adaptor = catalystAdaptor::getInstance();
  if (adaptor.initialiseVTKOutput(1, 1.0, "test"))
  {
    std::cerr <<  "main::initialiseVTKOutput failed\n";
    return 1;
  }
  const int result = Catch::Session().run( argc, argv );
  return result;
}
