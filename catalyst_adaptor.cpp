#include "catalyst_adaptor.h"

#include "vtkCPVTKOutputPipeline.h"

#include <vtkCPPythonScriptPipeline.h>
#include <vtkCPDataDescription.h>
#include <vtkCPInputDataDescription.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>

#include <vector>

catalystAdaptor::catalystAdaptor()
{
  this->Processor = nullptr;
  this->dataDescription = nullptr;
  this->patches.clear();
}

//------------------------------------------------------

catalystAdaptor::~catalystAdaptor()
{
  this->Processor->Delete();
  this->Processor = nullptr;
  this->dataDescription->Delete();
  this->dataDescription = nullptr;
  this->patches.clear();
}

//------------------------------------------------------

catalystAdaptor& catalystAdaptor::getInstance()
{
  // Just return reference to adaptor, will be
  // constructed on first use
  static catalystAdaptor adaptor;
  return adaptor;
}

//------------------------------------------------------

int catalystAdaptor::initialiseWithPython(const std::string& scriptName)
{
  if (this->Processor != nullptr)
  {
    vtkGenericWarningMacro("catalystAdaptor::initialiseWithPython Processor is already initialised." << endl);
    return 1;
  }

  if (scriptName.empty())
  {
    vtkGenericWarningMacro("catalystAdaptor::initialiseWithPython No filename provided for Python script." << endl);
    return 1;
  }

  // Initialise Catalyst
  this->Processor = vtkCPProcessor::New();
  this->Processor->Initialize();

  // Initialise Python script pipeline
  vtkSmartPointer<vtkCPPythonScriptPipeline> pipeline = vtkSmartPointer<vtkCPPythonScriptPipeline>::New();
  pipeline->Initialize(scriptName.c_str());
  this->Processor->AddPipeline(pipeline);

  // Create data description object that keeps track of simulation data
  // "input" is the default name for the VTK data source
  this->dataDescription = vtkCPDataDescription::New();
  this->dataDescription->AddInput("input");

  return 0;
}

//------------------------------------------------------

int catalystAdaptor::initialiseVTKOutput(const int frequency, const double time, const std::string& fileName)
{
  if (this->Processor != nullptr)
  {
    vtkGenericWarningMacro("catalystAdaptor::initialiseVTKOutput Processor is already initialised." << endl);
    return 1;
  }

  if (frequency < 0)
  {
    vtkGenericWarningMacro("catalystAdaptor::initialiseVTKOutput VTK output frequency must 0 or >0." << endl);
    return 1;
  }

  if (time < 0.0)
  {
    vtkGenericWarningMacro("catalystAdaptor::initialiseVTKOutput VTK output time interval must 0.0 or >0.0." << endl);
    return 1;
  }

  this->Processor = vtkCPProcessor::New();
  this->Processor->Initialize();

  // Initialise VTK output pipeline
  vtkSmartPointer<vtkCPVTKOutputPipeline> pipeline = vtkSmartPointer<vtkCPVTKOutputPipeline>::New();
  pipeline->SetOutputFrequency(frequency);
  pipeline->SetOutputTimeInterval(time);
  if (!fileName.empty()) pipeline->SetFileName(fileName);
  this->Processor->AddPipeline(pipeline);

  this->dataDescription = vtkCPDataDescription::New();
  this->dataDescription->AddInput("input");

  return 0;
}

//------------------------------------------------------

int catalystAdaptor::addPatch(const int patchId, const int level,
                              const int nx, const int ny,
                              const double dx, const double dy,
                              const double x0, const double y0)
{
  // Check if a patch with this ID exists already
  std::unordered_map<int,gridPatch>::const_iterator it = this->patches.find(patchId);
  if (it == this->patches.end())
  {
    // Set up new grid patch as a uniform 2D grid
    gridPatch patch;
    patch.level = level;
    patch.VTKGrid = vtkSmartPointer<vtkUniformGrid>::New();
    patch.VTKGrid->SetSpacing(dx, dy, 0.0);
    patch.VTKGrid->SetExtent(0, nx, 0, ny, 0, 0);
    patch.VTKGrid->SetOrigin(x0, y0, 0.0);
    this->patches.insert({patchId, patch});
    return 0;
  }
  else
  {
    // Patch exists already, so report an error
    return 1;
  }
}

//------------------------------------------------------

int catalystAdaptor::removePatch(const int patchId)
{
  // Check if a patch with this ID exists
  std::unordered_map<int,gridPatch>::const_iterator it = this->patches.find(patchId);
  if (it == this->patches.end())
  {
    // No patch - report error
    return 1;
  }
  else
  {
    this->patches.erase(it);
    return 0;
  }
}

//------------------------------------------------------

void catalystAdaptor::setAMRPatches(vtkSmartPointer<vtkNonOverlappingAMR> AMRGrid)
{
  // Work out how many levels we have in the AMR grid
  int numberOfLevels = 0;
  for (std::pair<int,gridPatch> const& patch : this->patches)
  {
    // Coarsest level is level 0
    if ((patch.second.level+1) > numberOfLevels) numberOfLevels = patch.second.level+1;
  }

  // Work out how many blocks per level we have in the AMR grid
  std::vector<int> blocksPerLevel;
  blocksPerLevel.resize(numberOfLevels);
  blocksPerLevel.assign(numberOfLevels, 0);
  for (std::pair<int,gridPatch> const& patch : this->patches)
  {
    blocksPerLevel[patch.second.level] += 1;
  }

  // Create new container grid
  AMRGrid->Initialize(numberOfLevels, blocksPerLevel.data());

  // Add patches to the container grid
  std::vector<int> blockId;
  blockId.resize(numberOfLevels);
  blockId.assign(numberOfLevels, 0);
  for (std::pair<int,gridPatch> const& patch : this->patches)
  {
    // Each grid patch has a level and ID - we create a new ID for each patch
    // at given level in order of appearance
    AMRGrid->SetDataSet(patch.second.level, blockId[patch.second.level], patch.second.VTKGrid);
    blockId[patch.second.level] += 1;
  }
}

//------------------------------------------------------

int catalystAdaptor::updateFieldSingle(const int patchId, const std::string& name, float* data)
{
  // Check if a patch with this ID exists
  std::unordered_map<int,gridPatch>::const_iterator it = this->patches.find(patchId);
  if (it == this->patches.end())
  {
    // No patch - do nothing and report error
    return 1;
  }
  else
  {
    if (this->dataDescription == nullptr)
    {
      vtkGenericWarningMacro("catalystAdaptor::updateFieldSingle DataDescription not initialised.");
      return 1;
    }

    // Check if this field has been requested by the pipeline. If not, remove array from the
    // grid if it has been added before, and exit silently
    if (this->dataDescription->GetInputDescription(0)->IsFieldNeeded(name.c_str()))
    {
      // Get number of cells of this grid patch
      vtkIdType numberOfCells = it->second.VTKGrid->GetNumberOfCells();

      vtkSmartPointer<vtkFloatArray> field = vtkSmartPointer<vtkFloatArray>::New();
      // Store all fields as single component at this point
      // We could store vector fields as multi-component fields in the future
      field->SetNumberOfComponents(1);
      field->SetNumberOfTuples(numberOfCells);
      field->SetName(name.c_str());
      // Last parameter "1" prevents VTK from deallocating data array
      field->SetVoidArray(data, numberOfCells, 1);
      it->second.VTKGrid->GetCellData()->AddArray(field);
    }
    else if (it->second.VTKGrid->GetCellData()->HasArray(name.c_str()))
    {
      it->second.VTKGrid->GetCellData()->RemoveArray(name.c_str());
    }
    return 0;
  }
}

//------------------------------------------------------

int catalystAdaptor::updateFieldDouble(const int patchId, const std::string& name, double* data)
{
  std::unordered_map<int,gridPatch>::const_iterator it = this->patches.find(patchId);
  if (it == this->patches.end())
  {
    return 1;
  }
  else
  {
    if (this->dataDescription == nullptr)
    {
      vtkGenericWarningMacro("catalystAdaptor::updateFieldSingle DataDescription not initialised.");
      return 1;
    }
    if (this->dataDescription->GetInputDescription(0)->IsFieldNeeded(name.c_str()))
    {
      vtkIdType numberOfCells = it->second.VTKGrid->GetNumberOfCells();
      vtkSmartPointer<vtkDoubleArray> field = vtkSmartPointer<vtkDoubleArray>::New();
      field->SetNumberOfComponents(1);
      field->SetNumberOfTuples(numberOfCells);
      field->SetName(name.c_str());
      field->SetVoidArray(data, numberOfCells, 1);
      it->second.VTKGrid->GetCellData()->AddArray(field);
    }
    else if (it->second.VTKGrid->GetCellData()->HasArray(name.c_str()))
    {
      it->second.VTKGrid->GetCellData()->RemoveArray(name.c_str());
    }
    return 0;
  }
}

//------------------------------------------------------

bool catalystAdaptor::requestDataDescription(const double time, const unsigned int timeStep)
{
  if (this->dataDescription == nullptr)
  {
    vtkGenericWarningMacro("catalystAdaptor::requestDataDescription DataDescription not initialised.");
    return false;
  }
  // Tell coprocessor what time it is and check if we need to do anything
  this->dataDescription->SetTimeData(time, timeStep);
  if (this->Processor->RequestDataDescription(this->dataDescription))
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------

int catalystAdaptor::runCoprocessor()
{
  if (this->Processor == nullptr)
  {
    vtkGenericWarningMacro("catalystAdaptor::runCoprocessor Processor not initialised.");
    return 1;
  }
  if (this->dataDescription == nullptr)
  {
    vtkGenericWarningMacro("catalystAdaptor::runCoprocessor DataDescription not initialised.");
    return 1;
  }

  // Recreate VTK AMR grid container from list of grid patches
  vtkSmartPointer<vtkNonOverlappingAMR> AMRGrid = vtkSmartPointer<vtkNonOverlappingAMR>::New();
  setAMRPatches(AMRGrid);
  this->dataDescription->GetInputDescription(0)->SetGrid(AMRGrid);

  if (this->Processor->CoProcess(this->dataDescription) != 1)
  {
    vtkGenericWarningMacro("catalystAdaptor::runCoprocessor Coprocessor reported failure.");
    return 1;
  }
  else
  {
    return 0;
  }
}
