/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "BG_Flood", "index.html", [
    [ "Introduction", "index.html#autotoc_md0", [
      [ "Model development stage", "index.html#autotoc_md2", null ],
      [ "How to install", "index.html#autotoc_md4", [
        [ "From precompiled binaries (Win10 only)", "index.html#autotoc_md5", null ],
        [ "On Linux machines", "index.html#autotoc_md6", null ]
      ] ],
      [ "How to use the model", "index.html#User", null ],
      [ "How to use/change the code", "index.html#Developer", null ]
    ] ],
    [ "Installation", "How_to_install.html", null ],
    [ "Paramters and Forcings list", "ParameterList.html", [
      [ "List of the Parameters' input", "ParameterList.html#autotoc_md10", [
        [ "General parameters", "ParameterList.html#autotoc_md11", null ],
        [ "Grid parameters", "ParameterList.html#autotoc_md12", null ],
        [ "Adaptation", "ParameterList.html#autotoc_md13", null ],
        [ "Timekeeping", "ParameterList.html#autotoc_md14", null ],
        [ "Initialisation", "ParameterList.html#autotoc_md15", null ],
        [ "Outputs", "ParameterList.html#autotoc_md16", null ],
        [ "Netcdf parameters", "ParameterList.html#autotoc_md17", null ],
        [ "ParaView Catalyst parameters (SPECIAL USE WITH PARAVIEW)", "ParameterList.html#autotoc_md18", null ]
      ] ],
      [ "List of the Forcings' inputs", "ParameterList.html#autotoc_md19", null ],
      [ "List of the non-identified inputs", "ParameterList.html#autotoc_md20", null ]
    ] ],
    [ "Manual", "Manual.html", [
      [ "Model controls", "Manual.html#autotoc_md21", [
        [ "BG_param.txt", "Manual.html#autotoc_md22", [
          [ "How to use the BG_param.txt file", "Manual.html#autotoc_md23", null ],
          [ "General comment about input files", "Manual.html#autotoc_md24", null ]
        ] ],
        [ "List of Parameters", "Manual.html#autotoc_md25", null ],
        [ "Bathymetry/topography files", "Manual.html#autotoc_md26", null ],
        [ "Conputational mesh", "Manual.html#autotoc_md27", [
          [ "Masking", "Manual.html#autotoc_md28", null ]
        ] ],
        [ "Boundaries", "Manual.html#autotoc_md29", [
          [ "Boundary file (for type 2 or 3)", "Manual.html#autotoc_md30", [
            [ "Uniform boundary file", "Manual.html#autotoc_md31", null ],
            [ "Variable boundary file", "Manual.html#autotoc_md32", null ]
          ] ]
        ] ],
        [ "Bottom friction", "Manual.html#autotoc_md33", null ],
        [ "Rivers and Area discharge", "Manual.html#autotoc_md34", null ],
        [ "Wind atm pressure forcing", "Manual.html#autotoc_md35", [
          [ "Wind forcing (may contain bugs)", "Manual.html#autotoc_md36", [
            [ "spatially uniform txt file:", "Manual.html#autotoc_md37", null ],
            [ "Spatially and time varying input", "Manual.html#autotoc_md38", null ]
          ] ],
          [ "Atmospheric pressure forcing", "Manual.html#autotoc_md39", null ]
        ] ],
        [ "Outputs", "Manual.html#autotoc_md40", [
          [ "Map outputs", "Manual.html#autotoc_md41", [
            [ "Default snapshot outputs", "Manual.html#autotoc_md42", null ],
            [ "Complementary variables", "Manual.html#autotoc_md43", null ],
            [ "Mean/averaged output between output steps", "Manual.html#autotoc_md44", null ],
            [ "Max output", "Manual.html#autotoc_md45", null ],
            [ "Risk assesment related output", "Manual.html#autotoc_md46", null ],
            [ "Model related outputs", "Manual.html#autotoc_md47", null ],
            [ "Other gradients and intermediate terms of the equations", "Manual.html#autotoc_md48", null ]
          ] ],
          [ "Point or Time-Serie output", "Manual.html#autotoc_md49", null ]
        ] ],
        [ "Adaptative grid", "Manual.html#autotoc_md50", null ]
      ] ]
    ] ],
    [ "Modules", "Modules.html", [
      [ "Halo and gradient", "HaloGradient.html", null ],
      [ "Ground infiltration: Initial Loss - Continuous Loss", "ILCL.html", null ],
      [ "Conserve Elevation", "WetdryfixConservelevation.html", null ]
    ] ],
    [ "Tutorials", "Tutorials.html", [
      [ "River flooding tutorial", "tutorialRiver.html", [
        [ "Param file", "tutorialRiver.html#autotoc_md73", null ],
        [ "Preparation of the topography/bathymetry (DEM: Digital Elevation Model)", "tutorialRiver.html#autotoc_md74", null ],
        [ "Basic fluvial flooding set-up", "tutorialRiver.html#autotoc_md75", [
          [ "River discharge", "tutorialRiver.html#autotoc_md76", null ],
          [ "Timekeeping parameters", "tutorialRiver.html#autotoc_md77", null ],
          [ "Outputs", "tutorialRiver.html#autotoc_md78", [
            [ "Map outputs", "tutorialRiver.html#autotoc_md79", null ],
            [ "Time-Serie outputs", "tutorialRiver.html#autotoc_md80", null ]
          ] ],
          [ "Resolution", "tutorialRiver.html#autotoc_md81", null ],
          [ "Basic fluvial innundation results", "tutorialRiver.html#autotoc_md82", null ]
        ] ],
        [ "Completing the set-up", "tutorialRiver.html#autotoc_md83", [
          [ "Adding boundary conditions", "tutorialRiver.html#autotoc_md84", null ],
          [ "Bottom friction", "tutorialRiver.html#autotoc_md85", null ],
          [ "Initialisation", "tutorialRiver.html#autotoc_md86", null ],
          [ "Model controls", "tutorialRiver.html#autotoc_md87", null ],
          [ "Outputs", "tutorialRiver.html#autotoc_md88", null ]
        ] ],
        [ "... Adding the rain", "tutorialRiver.html#autotoc_md89", [
          [ "Rain forcing", "tutorialRiver.html#autotoc_md90", null ]
        ] ],
        [ "Refining the grid in area of interest", "tutorialRiver.html#autotoc_md91", [
          [ "Results:", "tutorialRiver.html#autotoc_md92", null ],
          [ "Ground infiltration losses (Basic ILCL model)", "tutorialRiver.html#autotoc_md93", null ]
        ] ]
      ] ],
      [ "Jet tutorial with Julia", "TutorialJetJulia.html", [
        [ "Gaussian Wave", "TutorialJetJulia.html#autotoc_md58", [
          [ "Make a bathymetry", "TutorialJetJulia.html#autotoc_md51", null ],
          [ "Make Bnd files", "TutorialJetJulia.html#autotoc_md52", [
            [ "<tt>right.bnd</tt>", "TutorialJetJulia.html#autotoc_md53", null ],
            [ "<tt>left.bnd</tt>", "TutorialJetJulia.html#autotoc_md54", null ]
          ] ],
          [ "Set up the <tt>BG_param.txt</tt> file", "TutorialJetJulia.html#autotoc_md55", null ],
          [ "Run the model", "TutorialJetJulia.html#autotoc_md56", null ],
          [ "Things to try", "TutorialJetJulia.html#autotoc_md57", null ],
          [ "Bathymetry", "TutorialJetJulia.html#autotoc_md59", null ],
          [ "Hortstart", "TutorialJetJulia.html#autotoc_md60", null ],
          [ "Make Bnd files", "TutorialJetJulia.html#autotoc_md61", null ],
          [ "Set up the <tt>BG_param.txt</tt> file", "TutorialJetJulia.html#autotoc_md62", null ],
          [ "Run the model", "TutorialJetJulia.html#autotoc_md63", null ],
          [ "Things to try:", "TutorialJetJulia.html#autotoc_md64", null ]
        ] ],
        [ "Transpacific tsunami", "TutorialJetJulia.html#autotoc_md65", [
          [ "Bathy and domain definition", "TutorialJetJulia.html#autotoc_md66", null ],
          [ "Initial tsunami wave", "TutorialJetJulia.html#autotoc_md67", null ],
          [ "Boundary", "TutorialJetJulia.html#autotoc_md68", null ],
          [ "Time Keeping", "TutorialJetJulia.html#autotoc_md69", null ],
          [ "Outputs", "TutorialJetJulia.html#autotoc_md70", null ],
          [ "Things to try", "TutorialJetJulia.html#autotoc_md71", null ]
        ] ],
        [ "River + Rain = Waikanae example", "TutorialJetJulia.html#autotoc_md72", null ]
      ] ],
      [ "Monai tutorial with Julia", "TutorialMonaiJulia.html", null ]
    ] ],
    [ "Examples", "Test-and-Examples.html", [
      [ "Gaussian wave verification", "GaussianWave.html", [
        [ "Model controls", "GaussianWave.html#autotoc_md118", null ],
        [ "Flow parameters", "GaussianWave.html#autotoc_md119", null ],
        [ "Timekeeping parameters", "GaussianWave.html#autotoc_md120", null ]
      ] ],
      [ "Monai test Case", "Monai.html", null ],
      [ "Rain on grid", "RainOnGrid.html", null ]
    ] ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ]
      ] ]
    ] ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", "functions_vars" ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", "globals_func" ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"AdaptCriteria_8cu.html",
"ConserveElevation_8h.html#afebcff1a2c87ddb0c67afdea7185cc23",
"Halo_8cu.html#a0dfd1652d58d54f5d3fd66d6d30ed2dd",
"InitialConditions_8h.html",
"MemManagement_8h.html#aece56a33f66f19a260b6ff71d5ed14bf",
"Setup__GPU_8cu.html#a10fde28f3704462d7060bc37c2b1f5eb",
"Util__CPU_8cu.html#ab7b415cd8258f95d547db14856841c40",
"classdeformmap.html#a140c770679bc3072a11437d365ac3e15",
"structForcing.html#a46d192e153e4002ff84aa88c2e688b8e"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';