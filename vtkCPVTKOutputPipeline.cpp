#include "vtkCPVTKOutputPipeline.h"

#include <vtkCPInputDataDescription.h>
#include <vtkNonOverlappingAMR.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkPVTrivialProducer.h>
#include <vtkXMLUniformGridAMRWriter.h>

vtkStandardNewMacro(vtkCPVTKOutputPipeline);

//------------------------------------------------------

vtkCPVTKOutputPipeline::vtkCPVTKOutputPipeline()
{
  this->OutputFrequency = 0;
  this->FileName.clear();
}

//------------------------------------------------------

vtkCPVTKOutputPipeline::~vtkCPVTKOutputPipeline()
{
  this->FileName.clear();
}

//------------------------------------------------------

int vtkCPVTKOutputPipeline::RequestDataDescription(vtkCPDataDescription* dataDescription)
{
  if(!dataDescription)
  {
    vtkWarningMacro("vtkCPVTKOutputPipeline::RequestDataDescription dataDescription object is undefined." << endl);
    return 0;
  }

  if(dataDescription->GetNumberOfInputDescriptions() != 1)
  {
    vtkWarningMacro("vtkCPVTKOutputPipeline::RequestDataDescription Expected exactly 1 input description.");
    return 0;
  }

  // Check if we need to produce output (either forced or via regular timestep request)
  if(dataDescription->GetForceOutput() == true ||
    (this->OutputFrequency > 0 && dataDescription->GetTimeStep() >= 0 &&
     dataDescription->GetTimeStep() % this->OutputFrequency == 0) )
  {
    // Include all fields by default
    dataDescription->GetInputDescription(0)->AllFieldsOn();
    return 1;
  }
  return 0;
}

//------------------------------------------------------

int vtkCPVTKOutputPipeline::CoProcess(vtkCPDataDescription* dataDescription)
{
  if(!dataDescription) {
    vtkWarningMacro("vtkCPVTKOutputPipeline::CoProcess dataDescription object is undefined." << endl);
    return 0;
  }

  if(this->FileName.empty())
  {
    vtkWarningMacro("vtkCPVTKOutputPipeline::RequestDataDescription No output filename was set." << endl);
    return 0;
  }

  // Try to get grid, assume that data sources uses default name "input"
  vtkCPInputDataDescription * inputDataDescription = dataDescription->GetInputDescriptionByName("input");
  if(!inputDataDescription)
  {
    vtkWarningMacro("vtkCPVTKOutputPipeline::CoProcess dataDescription is missing inputDataDescription with name input.");
    return 0;
  }

  vtkNonOverlappingAMR* grid = vtkNonOverlappingAMR::SafeDownCast(inputDataDescription->GetGrid());
  if(!grid)
  {
    vtkWarningMacro("vtkCPVTKOutputPipeline::CoProcess inputDataDescription is missing grid.");
    return 0;
  }

  vtkNew<vtkPVTrivialProducer> producer;
  producer->SetOutput(grid);

  vtkNew<vtkXMLUniformGridAMRWriter> writer;
  writer->SetInputConnection(producer->GetOutputPort());
  // File compression
  writer->SetCompressorTypeToZLib();
  // Filename
  std::ostringstream o;
  o << dataDescription->GetTimeStep();
  std::string name = this->FileName + o.str() + ".vti";
  writer->SetFileName(name.c_str());
  writer->Update();

  return 1;
}

//------------------------------------------------------

void vtkCPVTKOutputPipeline::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "OutputFrequency: " << this->OutputFrequency << "\n";
  os << indent << "FileName: " << this->FileName << "\n";
}
