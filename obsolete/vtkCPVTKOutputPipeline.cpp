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
  this->OutputTimeInterval = 0.0;
  this->LastOutputTime = 0.0;
  this->TimeTrigger = false;
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

  double currentTime = dataDescription->GetTime();
  vtkIdType currentTimeStep = dataDescription->GetTimeStep();

  // Check if simulation time and time step criteria are independently fulfilled
  bool outputThisTime = (this->OutputTimeInterval > 0) && 
                        ((currentTime - this->LastOutputTime) >= this->OutputTimeInterval);
  bool outputThisTimeStep = (this->OutputFrequency > 0) && (currentTimeStep > 0) &&
                            (currentTimeStep % this->OutputFrequency == 0);

  // Keep time trigger to reset output timer correctly
  this->TimeTrigger = outputThisTime;

  if(dataDescription->GetForceOutput() || outputThisTime || outputThisTimeStep)
  {
    // Include all fields by default - the pipeline is handed same set of fields as netCDF output
    dataDescription->GetInputDescription(0)->AllFieldsOn();
    return 1;
  }
  else
  {
    // No output; make sure that no fields are requested from simulation
    dataDescription->GetInputDescription(0)->AllFieldsOff();
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
  // Use .vth suffix for hierarchical box data files, this is closest to AMR grids
  std::string name = this->FileName + o.str() + ".vth";
  writer->SetFileName(name.c_str());
  writer->Update();

  // Reset output time if time criterion was triggered - this needs to be AFTER the CoProcess
  // method has been called, as RequestDataDescription method can be called more than once
  // in one timestep
  if(this->TimeTrigger)
  {
    this->LastOutputTime = dataDescription->GetTime();
    this->TimeTrigger = false;
  }

  return 1;
}

//------------------------------------------------------

void vtkCPVTKOutputPipeline::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "OutputFrequency: " << this->OutputFrequency << "\n";
  os << indent << "OutputTimeInterval: " << this->OutputTimeInterval << "\n";
  os << indent << "FileName: " << this->FileName << "\n";
}
