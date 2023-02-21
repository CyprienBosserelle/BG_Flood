#include "Catch2/catch.hpp"
#include "catalyst_adaptor.h"

// ------------------------------------------------------------------------------------------

TEST_CASE( "Initialisation", "[catalyst_adaptor]" )
{
  // Retrieve instance of catalystAdaptor singleton
  catalystAdaptor& adaptor = catalystAdaptor::getInstance();

  SECTION( "CoProcessor With VTK Output Fails To Initialise Twice" )
  {
    REQUIRE ( adaptor.initialiseVTKOutput(1, 1.0, "test") == 1 );
  }

  SECTION( "Cannot Re-initialise CoProcessor With Python Pipeline" )
  {
    REQUIRE ( adaptor.initialiseWithPython("testscript.py") == 1 );
  }

  SECTION( "Verify Singleton Instantiation" )
  {
    catalystAdaptor& adaptor2 = catalystAdaptor::getInstance();
    // Check that both variables reference the same object in memory
    REQUIRE ( &adaptor2 == &adaptor );
  }
}

// ------------------------------------------------------------------------------------------

TEST_CASE( "Grid Patch Handling ", "[catalyst_adaptor]" )
{
  catalystAdaptor& adaptor = catalystAdaptor::getInstance();

  const int level = 0;
  const int nx = 16;
  const int ny = 16;
  const double dx = 1.0;
  const double dy = 1.0;
  const double x0 = 0.0;
  const double y0 = 0.0;

  SECTION( "Adding And Removing Correctly Defined Patch Works" )
  {
    REQUIRE ( adaptor.addPatch(0, level, nx, ny, dx, dy, x0, y0) == 0 );
    REQUIRE ( adaptor.removePatch(0) == 0 );
  }

  SECTION( "Removing Undefined Patch Does Not Work" )
  {
    REQUIRE ( adaptor.removePatch(0) == 1 );
  }

  SECTION( "Adding Two Patches With Same ID Is Not Successful" )
  {
    REQUIRE ( adaptor.addPatch(0, level, nx, ny, dx, dy, x0, y0) == 0 );
    REQUIRE ( adaptor.addPatch(0, level, nx, ny, dx, dy, x0, y0) == 1 );
    REQUIRE ( adaptor.removePatch(0) == 0 );
  }

  SECTION( "Adding Two Patches With Different IDs Works" )
  {
    REQUIRE ( adaptor.addPatch(0, level, nx, ny, dx, dy, x0, y0) == 0 );
    REQUIRE ( adaptor.addPatch(7, 6, 24, 36, 6.0, 3.1, -5.2, 8.7) == 0 );
    REQUIRE ( adaptor.removePatch(0) == 0 );
    REQUIRE ( adaptor.removePatch(7) == 0 );
  }

  SECTION( "Adding Large Number Of Patches Works" )
  {
    const int npatches = 100;
    int status = 0;
    for (int patchId = 0; patchId < npatches; patchId++)
    {
      status += adaptor.addPatch(patchId, level, nx, ny, dx, dy, x0, y0);
    }
    REQUIRE ( status == 0 );
    status = 0;
    for (int patchId = 0; patchId < npatches; patchId++)
    {
      status += adaptor.removePatch(patchId);
    }
    REQUIRE ( status == 0 );
  }
}

// ------------------------------------------------------------------------------------------

TEST_CASE( "Updating Fields", "[catalyst_adaptor]" )
{
  catalystAdaptor& adaptor = catalystAdaptor::getInstance();

  const int level = 3;
  const int nx = 16;
  const int ny = 16;
  const double dx = 1.0;
  const double dy = 1.0;
  const double x0 = 0.0;
  const double y0 = 0.0;
  const int nIterations = 100;

  // Create dummy data fields in single and double precision
  std::vector<float> testFieldSingle;
  testFieldSingle.assign(nx*ny, 1.2345);

  std::vector<double> testFieldDouble;
  testFieldDouble.assign(nx*ny, 1.2345);

  // Create arrays of fields as well
  std::vector<std::vector<float>> testFieldSingleArray;
  testFieldSingleArray.resize(nIterations);
  for (std::vector<float>& thisField : testFieldSingleArray)
  {
    thisField.assign(nx*ny,1.0);
  }

  std::vector<std::vector<double>> testFieldDoubleArray;
  testFieldDoubleArray.resize(nIterations);
  for (std::vector<double>& thisField : testFieldDoubleArray)
  {
    thisField.assign(nx*ny,1.0);
  }

  SECTION( "Updating Fields On Non-Existent Patch Does Not Work" )
  {
    REQUIRE ( adaptor.updateFieldSingle(0, "testFieldSingle", testFieldSingle.data()) == 1 );
    REQUIRE ( adaptor.updateFieldDouble(0, "testFieldDouble", testFieldDouble.data()) == 1 );
  }

  SECTION( "Updating Fields On Correctly Defined Patch Works" )
  {
    const int patchId = 5;
    REQUIRE ( adaptor.addPatch(patchId, level, nx, ny, dx, dy, x0, y0) == 0 );
    REQUIRE ( adaptor.updateFieldSingle(patchId, "testFieldSingle", testFieldSingle.data()) == 0 );
    REQUIRE ( adaptor.updateFieldDouble(patchId, "testFieldDouble", testFieldDouble.data()) == 0 );
    REQUIRE ( adaptor.removePatch(patchId) == 0 );
  }

  SECTION( "Updating Fields On Same Patch Iteratively Works" )
  {
    const int patchId = 199;
    REQUIRE ( adaptor.addPatch(patchId, level, nx, ny, dx, dy, x0, y0) == 0 );
    int status = 0;
    for (std::vector<float>& thisField : testFieldSingleArray)
    {
      status += adaptor.updateFieldSingle(patchId, "testFieldSingle", thisField.data());
    }
    REQUIRE ( status == 0 );
    status = 0;
    for (std::vector<double>& thisField : testFieldDoubleArray)
    {
      status += adaptor.updateFieldDouble(patchId, "testFieldDouble", thisField.data());
    }
    REQUIRE ( status == 0 );
    REQUIRE ( adaptor.removePatch(patchId) == 0 );
  }

  SECTION( "Adding Patches And Updating Fields Iteratively Works" )
  {
    const int patchId = 63;
    int status = 0;
    for (std::vector<float>& thisField : testFieldSingleArray)
    {
      status += adaptor.addPatch(patchId, level, nx, ny, dx, dy, x0, y0);
      status += adaptor.updateFieldSingle(patchId, "testFieldSingle", thisField.data());
      status += adaptor.removePatch(patchId);
    }
    REQUIRE ( status == 0 );
    status = 0;
    for (std::vector<double>& thisField : testFieldDoubleArray)
    {
      status += adaptor.addPatch(patchId, level, nx, ny, dx, dy, x0, y0);
      status += adaptor.updateFieldDouble(patchId, "testFieldDouble", thisField.data());
      status += adaptor.removePatch(patchId);
    }
    REQUIRE ( status == 0 );
  }
}

// ------------------------------------------------------------------------------------------

TEST_CASE( "requestDataDescription", "[catalyst_adaptor]" )
{
  catalystAdaptor& adaptor = catalystAdaptor::getInstance();

  SECTION( "Coprocessor Should Run At Positive Times And Timesteps" )
  {
    REQUIRE ( adaptor.requestDataDescription(1.0, 1) == true );
    REQUIRE ( adaptor.requestDataDescription(2.0, 2) == true );
    REQUIRE ( adaptor.requestDataDescription(3.0, 3) == true );
    REQUIRE ( adaptor.requestDataDescription(7.0, 7) == true );
  }
}
