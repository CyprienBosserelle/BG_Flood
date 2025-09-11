#ifndef VTKCPVTKOUTPUTPIPELINE_H
#define VTKCPVTKOUTPUTPIPELINE_H

#include <vtkCPPipeline.h>
#include <vtkCPDataDescription.h>

#include <string>

// Implements a simple ParaView Catalyst pipeline that
// writes AMR grid output in VTK file format. These
// types of grids are not currently supported by
// built-in vtkCPXMLPWriterPipeline
class vtkCPVTKOutputPipeline: public vtkCPPipeline
{
 public:

  // Standard VTK members
  static vtkCPVTKOutputPipeline* New();
  void PrintSelf(ostream& os, vtkIndent indent) override;
  vtkTypeMacro(vtkCPVTKOutputPipeline,vtkCPPipeline);

  // Callback function: work out if we should produce output for this call
  // and let Catalyst know by returning 1 (yes) or 0 (no)
  int RequestDataDescription(vtkCPDataDescription* dataDescription) override;
  int CoProcess(vtkCPDataDescription* dataDescription) override;

  vtkSetMacro(OutputFrequency, int);
  vtkGetMacro(OutputFrequency, int);

  vtkSetMacro(OutputTimeInterval, double);
  vtkGetMacro(OutputTimeInterval, double);

  vtkSetMacro(FileName, std::string);
  vtkGetMacro(FileName, std::string);

  vtkCPVTKOutputPipeline(const vtkCPVTKOutputPipeline&) = delete;
  vtkCPVTKOutputPipeline& operator=(const vtkCPVTKOutputPipeline&) = delete;

 protected:

  vtkCPVTKOutputPipeline();
  virtual ~vtkCPVTKOutputPipeline();

 private:

  int OutputFrequency;
  double OutputTimeInterval;
  double LastOutputTime;
  bool TimeTrigger;
  std::string FileName;
};

#endif
