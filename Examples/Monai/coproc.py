# Copyright (c) 2005-2018 National Technology & Engineering Solutions
# of Sandia, LLC (NTESS), Kitware Inc. All rights reserved.
# See http://www.paraview.org/HTML/Copyright.html for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the above copyright notice for more information.

import paraview.simple as pvs
from paraview import coprocessing as cp

#
# Pipeline parameters
#
writeVtkOutput = False
outputFrequency = 30
fieldName = "hh"

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:

      # Create data source "input" (provides simulation fields)
      simData = coprocessor.CreateProducer( datadescription, "input" )

      # Write VTK output if requested
      if writeVtkOutput:
        fullWriter = pvs.XMLHierarchicalBoxDataWriter(Input=simData, DataMode="Appended",
                                                      CompressorType="ZLib")
        coprocessor.RegisterWriter(fullWriter, filename='bg_out_%t.vti', freq=outputFrequency)

      # Create a new render view to generate images
      renderView = pvs.CreateView('RenderView')
      renderView.ViewSize = [1500, 768]
      renderView.InteractionMode = '2D'
      renderView.AxesGrid = 'GridAxes3DActor'
      renderView.CenterOfRotation = [2.8, 1.7, 0.0]
      renderView.StereoType = 0
      renderView.CameraPosition = [2.8, 1.7, 10000.0]
      renderView.CameraFocalPoint = [2.8, 1.7, 0.0]
      renderView.CameraParallelScale = 3.386
      renderView.Background = [0.32, 0.34, 0.43]
      renderView.ViewTime = datadescription.GetTime()

      # Register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView, filename='bg_out_%t.png', freq=outputFrequency,
                               fittoscreen=1, magnification=1, width=1500, height=768,
                               cinema={})

      # Create colour transfer function for field
      LUT = pvs.GetColorTransferFunction(fieldName)
      LUT.RGBPoints = [9.355e-05, 0.231, 0.298, 0.753, 0.0674, 0.865, 0.865, 0.865, 0.135, 0.706, 0.0157, 0.149]
      LUT.ScalarRangeInitialized = 1.0

      # Render data field and colour by field value using lookup table
      fieldDisplay = pvs.Show(simData, renderView)
      fieldDisplay.Representation = 'Surface'
      fieldDisplay.ColorArrayName = ['CELLS', fieldName]
      fieldDisplay.LookupTable = LUT

    return Pipeline()

  class CoProcessor(cp.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  freqs = {'input': [outputFrequency]}
  coprocessor.SetUpdateFrequencies(freqs)
  return coprocessor

#--------------------------------------------------------------
# Global variables that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView
coprocessor.EnableLiveVisualization(False)


# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # Output all fields and meshes if forced
    if datadescription.GetForceOutput() == True:
        for i in range(datadescription.GetNumberOfInputDescriptions()):
            datadescription.GetInputDescription(i).AllFieldsOn()
            datadescription.GetInputDescription(i).GenerateMeshOn()
        return

    # Default implementation, uses output frequencies set in pipeline
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=False)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22224)
