#include "Catch2/catch.hpp"
#include <vtkCPInputDataDescription.h>
#include <vtkNonOverlappingAMR.h>
#include "vtkCPVTKOutputPipeline.h"

// ------------------------------------------------------------------------------------------

TEST_CASE( "Basic Class Tests", "[vtkCPVTKOutputPipeline]" )
{

  vtkCPVTKOutputPipeline * pipeline = vtkCPVTKOutputPipeline::New();

  SECTION( "SetOutputFrequency/GetOutputFrequency Methods Work" )
  {
    const int frequency = 7;
    pipeline->SetOutputFrequency(frequency);
    REQUIRE( pipeline->GetOutputFrequency() == frequency );
  }

  SECTION( "SetOutputTimeInterval/GetOutputTimeInterval Methods Work" )
  {
    const double time = 11.456;
    pipeline->SetOutputTimeInterval(time);
    REQUIRE( pipeline->GetOutputTimeInterval() == time );
  }

  SECTION( "SetFileName/GetFileName Methods Work" )
  {
    const std::string fileName = "my_testname";
    pipeline->SetFileName(fileName);
    REQUIRE( pipeline->GetFileName() == fileName );
  }

  pipeline->Delete();
}

// ------------------------------------------------------------------------------------------

TEST_CASE( "RequestDataDescription", "[vtkCPVTKOutputPipeline]" )
{

  vtkCPVTKOutputPipeline * pipeline = vtkCPVTKOutputPipeline::New();
  vtkCPDataDescription* dataDescription = vtkCPDataDescription::New();

  // Set both output criteria to no output
  pipeline->SetOutputFrequency(0);
  pipeline->SetOutputTimeInterval(0.0);

  SECTION( "Negative Reply Without DataDescription Object" )
  {
    vtkCPDataDescription* noDataDescription = nullptr;
    REQUIRE( pipeline->RequestDataDescription(noDataDescription) == 0 );
  }

  SECTION( "Negative Reply Without InputDescription" )
  {
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply With More Than One InputDescription" )
  {
    dataDescription->AddInput("input1");
    dataDescription->AddInput("input2");
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Positive Reply With Forced Output" )
  {
    dataDescription->AddInput("input");
    dataDescription->SetForceOutput(true);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 1 );
  }

  SECTION( "Negative Reply With Zero Output Frequency" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(0);
    dataDescription->SetTimeData(1.0, 1);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply With Zero Output Time Interval" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputTimeInterval(0.0);
    dataDescription->SetTimeData(1.0, 1);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply At Negative Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(1);
    dataDescription->SetTimeData(0.0, -1);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply At Negative Time" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputTimeInterval(1.0);
    dataDescription->SetTimeData(-1.0, 0);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply At Zero Time And Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(1);
    dataDescription->SetTimeData(0.0, 0);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Positive Reply At Requested Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(5);
    dataDescription->SetTimeData(0.0, 5);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 1 );
  }

  SECTION( "Positive Reply At Requested Time" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputTimeInterval(3.14159);
    dataDescription->SetTimeData(3.14159, 0);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 1 );
  }

  SECTION( "Positive Reply At Requested Time And Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(3);
    pipeline->SetOutputTimeInterval(3.0);
    dataDescription->SetTimeData(3.0, 3);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 1 );
  }

  SECTION( "Negative Reply At Non-requested Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(5);
    dataDescription->SetTimeData(0.0, 4);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply At Non-requested Time" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputTimeInterval(5.0);
    dataDescription->SetTimeData(4.0, 0);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  SECTION( "Negative Reply At Non-requested Time And Timestep" )
  {
    dataDescription->AddInput("input");
    pipeline->SetOutputFrequency(5);
    pipeline->SetOutputTimeInterval(5.0);
    dataDescription->SetTimeData(4.0, 4);
    REQUIRE( pipeline->RequestDataDescription(dataDescription) == 0 );
  }

  dataDescription->Delete();
  pipeline->Delete();
}

// ------------------------------------------------------------------------------------------

TEST_CASE( "CoProcess", "[vtkCPVTKOutputPipeline]" )
{

  vtkCPVTKOutputPipeline* pipeline = vtkCPVTKOutputPipeline::New();
  vtkCPDataDescription* dataDescription = vtkCPDataDescription::New();

  SECTION( "Negative Reply Without DataDescription Object" )
  {
    vtkCPDataDescription* noDataDescription = nullptr;
    REQUIRE( pipeline->CoProcess(noDataDescription) == 0 );
  }

  SECTION( "Negative Reply Without Valid Filename" )
  {
    std::string fileName;
    fileName.clear();
    pipeline->SetFileName(fileName);
    REQUIRE( pipeline->CoProcess(dataDescription) == 0 );
  }

  SECTION( "Negative Reply Without InputDescription" )
  {
    pipeline->SetFileName("test");
    REQUIRE( pipeline->CoProcess(dataDescription) == 0 );
  }

  SECTION( "Negative Reply With Wrong InputDescription" )
  {
    pipeline->SetFileName("test");
    dataDescription->AddInput("wronginput");
    REQUIRE( pipeline->CoProcess(dataDescription) == 0 );
  }

  SECTION( "Negative Reply Without Grid" )
  {
    pipeline->SetFileName("test");
    dataDescription->AddInput("input");
    vtkNonOverlappingAMR* grid = nullptr;
    dataDescription->GetInputDescriptionByName("input")->SetGrid(grid);
    REQUIRE( pipeline->CoProcess(dataDescription) == 0 );
  }

  dataDescription->Delete();
  pipeline->Delete();
}
