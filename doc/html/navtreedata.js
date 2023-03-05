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
    [ "Installation", "How_to_install.html", "How_to_install" ],
    [ "How to create continuous losses for NZ", "md_doc_modules_NZ_Conductivity.html", null ],
    [ "Paramters and Forcings list", "ParameterList.html", [
      [ "List of the Parameters' input", "ParameterList.html#autotoc_md92", [
        [ "General parameters", "ParameterList.html#autotoc_md93", null ],
        [ "Grid parameters", "ParameterList.html#autotoc_md94", null ],
        [ "Adaptation", "ParameterList.html#autotoc_md95", null ],
        [ "Timekeeping", "ParameterList.html#autotoc_md96", null ],
        [ "Initialisation", "ParameterList.html#autotoc_md97", null ],
        [ "Outputs", "ParameterList.html#autotoc_md98", null ],
        [ "Netcdf parameters", "ParameterList.html#autotoc_md99", null ],
        [ "ParaView Catalyst parameters (SPECIAL USE WITH PARAVIEW)", "ParameterList.html#autotoc_md100", null ]
      ] ],
      [ "List of the Forcings' inputs", "ParameterList.html#autotoc_md101", null ],
      [ "List of the non-identified inputs", "ParameterList.html#autotoc_md102", null ]
    ] ],
    [ "Manual", "Manual.html", [
      [ "Input Parameters", "Manual.html#autotoc_md105", [
        [ "<tt>BG_param.txt</tt>", "Manual.html#autotoc_md106", [
          [ "How to use the <tt>BG_param.txt</tt> file", "Manual.html#autotoc_md107", null ],
          [ "General comment about input files", "Manual.html#autotoc_md108", null ]
        ] ],
        [ "List of Parameters", "Manual.html#autotoc_md109", [
          [ "Input", "Manual.html#autotoc_md110", null ],
          [ "Forcing/Boundary", "Manual.html#autotoc_md111", null ],
          [ "Hydrodynamics", "Manual.html#autotoc_md112", null ],
          [ "Time keeping", "Manual.html#autotoc_md113", null ],
          [ "Output", "Manual.html#autotoc_md114", null ],
          [ "Miscelanious", "Manual.html#autotoc_md115", null ]
        ] ],
        [ "Bathymetry/topography files", "Manual.html#autotoc_md116", null ],
        [ "Conputational mesh", "Manual.html#autotoc_md117", [
          [ "Adaptative mesh", "Manual.html#autotoc_md118", null ],
          [ "Masking", "Manual.html#autotoc_md119", null ]
        ] ],
        [ "Boundaries", "Manual.html#autotoc_md120", [
          [ "Boundary file (for type 2 or 3)", "Manual.html#autotoc_md121", [
            [ "Uniform boundary file", "Manual.html#autotoc_md122", null ],
            [ "Variable boundary file", "Manual.html#autotoc_md123", null ]
          ] ]
        ] ],
        [ "Bottom friction", "Manual.html#autotoc_md124", null ],
        [ "Rivers and Area discharge", "Manual.html#autotoc_md125", null ],
        [ "Wind atm pressure forcing", "Manual.html#autotoc_md126", [
          [ "Wind forcing (may contain bugs)", "Manual.html#autotoc_md127", [
            [ "spatially uniform txt file: <tt>windfiles=mywind.txt</tt>", "Manual.html#autotoc_md128", null ],
            [ "Spatially and time varying input <tt>windfiles=mywind.nc?uw,mywind.nc?vw</tt>", "Manual.html#autotoc_md129", null ]
          ] ],
          [ "Atmospheric pressure forcing", "Manual.html#autotoc_md130", null ]
        ] ],
        [ "Output variables (Not up to date!!!)", "Manual.html#autotoc_md131", [
          [ "Snapshot outputs", "Manual.html#autotoc_md132", null ],
          [ "Mean/averaged output between output steps", "Manual.html#autotoc_md133", null ],
          [ "Max output", "Manual.html#autotoc_md134", null ],
          [ "Risk assesment related output", "Manual.html#autotoc_md135", null ],
          [ "Infiltration outputs", "Manual.html#autotoc_md136", null ],
          [ "Other gradients and intermediate terms of the equations", "Manual.html#autotoc_md137", null ]
        ] ]
      ] ]
    ] ],
    [ "Modules", "Modules.html", "Modules" ],
    [ "Tutorials", "Tutorials.html", "Tutorials" ],
    [ "Examples", "Test-and-Examples.html", "Test-and-Examples" ],
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
"ConserveElevation_8h.html#afc1cff5d5122df41b6b03e393ac7d8fa",
"GridManip_8h_source.html",
"InitialConditions_8cu.html#a263be1815f9718565688c7df2e501586",
"MemManagement_8cu.html#ad1f3c3382ef4419a772a92a0721b7d72",
"Read__netcdf_8h.html#a709e031ea8e9e656b958e7c984e7704a",
"Updateforcing_8h.html#ad13e319a421a8ddb774a162050b0c1d1",
"classRiver.html#a423c8984c0fb44357a2c3a9c9b36f5dc",
"structBndblockP.html#a19378d34bff2a9e54b3f3567848d5ea1"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';