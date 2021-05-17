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

# Turn on additional output of field data in VTK format
writeVtkOutput = False

# Disable either output criterion by setting it to 0
outputFrequency = 0
outputTimeInterval = 1.0

# Fields to be visualised - these will be requested from the simulation
fieldNames = ['hh','uu','vv']

#
# Internal variables
#

# Timekeeping for visualisation output
lastOutputTime = 0.0

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
        # Set freq=1 to ensure that output is written whenever the pipeline runs
        coprocessor.RegisterWriter(fullWriter, filename='bg_out_%t.vth', freq=1)

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

      # Show simulation time with 1 digit after decimal point
      annotateTime = pvs.AnnotateTime()
      annotateTime.Format = 'time: %.1f'
      timeDisplay = pvs.Show(annotateTime, renderView)

      # Combine uu and vv components into velocity vector field
      calculatorVelField = pvs.Calculator(Input=simData)
      calculatorVelField.AttributeType = 'Cell Data'
      calculatorVelField.ResultArrayName = 'velocity'
      calculatorVelField.Function = 'uu*iHat+vv*jHat'

      # Compute velocity field magnitudes
      calculatorVelMag = pvs.Calculator(Input=calculatorVelField)
      calculatorVelMag.AttributeType = 'Cell Data'
      calculatorVelMag.ResultArrayName = 'mag'
      calculatorVelMag.Function = 'mag(velocity)'

      # Remove cells with vanishing velocity magnitude
      velMagThreshold = pvs.Threshold(calculatorVelMag)
      velMagThreshold.Scalars = ['CELLS', 'mag']
      velMagThreshold.ThresholdRange = [1.0e-6, 1.0e30]

      # Visualise remaining velocity vector field using arrows with fixed length
      # and skipping cells to avoid crowding
      glyphs = pvs.Glyph(Input=velMagThreshold, GlyphType='Arrow')
      glyphs.Vectors = ['CELLS', 'velocity']
      glyphs.ScaleFactor = 0.2
      glyphs.GlyphMode = 'Every Nth Point'
      glyphs.Stride = 100
      glyphs.GlyphTransform = 'Transform2'

      # Register the view with coprocessor and provide it with information such as
      # the filename to use. Set freq=1 to ensure that images are rendered whenever
      # the pipeline runs
      coprocessor.RegisterView(renderView, filename='bg_out_%t.png', freq=1,
                               fittoscreen=0, magnification=1, width=1500, height=768,
                               cinema={})

      # Create colour transfer function for field
      LUT = pvs.GetColorTransferFunction('hh')
      LUT.RGBPoints = [9.355e-05, 0.231, 0.298, 0.753, 0.0674, 0.865, 0.865, 0.865, 0.135, 0.706, 0.0157, 0.149]
      LUT.ScalarRangeInitialized = 1.0

      # Render data field and colour by field value using lookup table
      fieldDisplay = pvs.Show(simData, renderView)
      fieldDisplay.Representation = 'Surface'
      fieldDisplay.ColorArrayName = ['CELLS', 'hh']
      fieldDisplay.LookupTable = LUT

      # Add velocity field visualisation
      velfieldDisplay = pvs.Show(glyphs, renderView)

    return Pipeline()

  class CoProcessor(cp.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

    # Override default method here to implement custom criterion to run pipeline and only
    # ask for fields that are actually needed
    def LoadRequestedData(self, datadescription):
      global lastOutputTime
      currentTime = datadescription.GetTime()
      currentTimeStep = datadescription.GetTimeStep()

      # Check if either criterion is independently fulfilled
      outputThisTime = outputTimeInterval > 0 and (currentTime - lastOutputTime) >= outputTimeInterval
      outputThisTimeStep = outputFrequency > 0 and currentTimeStep > 0 and \
                           currentTimeStep % outputFrequency == 0

      if outputThisTime or outputThisTimeStep:
        # Ask simulation for fields required by pipeline
        for fieldName in fieldNames:
          datadescription.GetInputDescription(0).AddCellField(fieldName)
        if outputThisTime:
          lastOutputTime = currentTime
      else:
        # No output; make sure that no fields are requested from simulation
        datadescription.GetInputDescription(0).AllFieldsOff()

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
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
