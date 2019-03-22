#ifndef CATALYST_ADAPTOR_H
#define CATALYST_ADAPTOR_H

#include <vtkUniformGrid.h>
#include <vtkUniformGridAMR.h>
#include <vtkCPProcessor.h>
#include <vtkSmartPointer.h>

#include <unordered_map>

// We need this struct to store both a pointer to the
// VTK grid object for a patch and its AMR refinement level
struct gridPatch
{
  int level;
  vtkSmartPointer<vtkUniformGrid> VTKGrid;
};

// This class is implemented using the "singleton" design pattern,
// we need to make sure that only one instance of a Catalyst
// processor can be created
class catalystAdaptor
{
 public:

  // Replaces the constructor to obtain "singleton" behaviour
  static catalystAdaptor& getInstance();

  ~catalystAdaptor();

  // Catalyst can run in two modes - full visualisation using Python script,
  // or just output in VTK file format. Note that only one can be used
  // in a given simulation run.
  const int initialiseWithPython(const std::string& scriptName);
  const int initialiseVTKOutput(const int frequency, const std::string& filePath);

  // Add a new vtkUniformGrid patch with given parameters
  // Patch IDs must be handled by the simulation program
  // to make sure that they are consistent with grid
  // patches used by the solver
  const int addPatch(const int patchId, const int level,
                     const int nx, const int ny,
                     const double dx, const double dy,
                     const double x0, const double y0);

  // Remove patch with given ID
  const int removePatch(const int patchId);

  // Add simulation data - must be called in each time step and before
  // running the coprocessor
  // IMPORTANT: These methods use "zero copy", Catalyst will access the
  // original data array for visualisation. "data" must therefore still
  // be in scope and in consistent state!
  const int addFieldSingle(const int patchId, const std::string& name, float* data);
  const int addFieldDouble(const int patchId, const std::string& name, double* data);

  // Run the visualisation pipeline
  const int runCoprocessor(const double time, const unsigned int timeStep);

  // Prevent copying and moving
  catalystAdaptor(const catalystAdaptor&) = delete;
  catalystAdaptor(catalystAdaptor&&) = delete;
  catalystAdaptor& operator=(const catalystAdaptor&) = delete;
  catalystAdaptor& operator=(catalystAdaptor&&) = delete;

 private:

  // Constructor is private to obtain singleton behaviour
  catalystAdaptor();

  // Construct VTK AMR grid container out of the grid patches
  void setAMRPatches(vtkSmartPointer<vtkUniformGridAMR> AMRGrid);

  // Stores grid patches, to avoid having to reconstruct them in every
  // refinement step
  std::unordered_map<int,gridPatch> patches;

  // Pointers to the Catalyst processor and the data structure
  // that describes simulation grid and field data
  vtkCPProcessor* Processor;
  vtkCPDataDescription* dataDescription;

};

#endif
