title: 'BG_Flood: Adaptive GPU-capable hydrodynamics model for flood and inundation '
tags:
  - C++
  - CUDA
  - flood
  - inundation
  - tsunami
  - storm-surge
  - adaptive mesh refinement
  - GPU

authors:
  - name: Cyprien Bosserelle
 q  corresponding: true # (This is how to denote the corresponding author)
    orcid: 0000-0001-8756-5247
    equal-contrib: true
    affiliation: 1
  - name: Alice Harang
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0002-5906-213X
    affiliation: 1
  - name: Emily Lane
    orcid: 0000-0002-9261-6640
    affiliation: 1
affiliations:
 - name: Earth Sciences New Zealand
   index: 1
   
 
date: 1 July 2025
bibliography: paper.bib

# Summary
`BG_Flood` is a numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinement type grid. The model was designed with the goal of simulating inundation (River, Storm surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU with adaptive mesh being generated at the start of the simulation.

The core SWE engines and adaptivity has been inspired and taken from St Venant solver from `Basilisk` (Popinet 2011) and the CUDA GPU memory model has been inspired by the work from (Vacondio et al. 2017). The model is implemented in CUDA-C to maximise GPU usage. 


# Statement of need
Flood hazard assessment and forecasting often require physics-based simulations to accurately evaluate areas exposed to hazard. These simulations are often completed using hydrodynamics models with high-resolution to capture small landscape and flow features (e.g. small drains, hydraulic jump), but also on a large enough domain to capture large scale features where the hazard forms and or amplifies (e.g. river catchment, continental shelf).

Effectively capturing this cascade of scale is only achievable with unstructured or adaptive mesh. Unstructured mesh generation rely heavily on user's input for generating a suitable mesh and do not generally offer the flexibility of modifying the mesh as the simulation progresses.

Adaptive mesh on the other hand can be generated easily with minimal user input and adapted as the computation progresses but this flexibility comes with an increase in computational cost. Adaptive mesh allow a user to iteratively increase mesh complexity and resolution, starting from coarse simple fast model to detailed but efficient models. 

For detailed large-scale flood assessment, runtime is a significant limitation either of the highest resolution that can be achieved or on the number of scenarios to be explored. Codes compatible with Graphics Processing Units offer a competitive computation time.

`BG_Flood` aims to offer a GPU native model with a simple interface that can automatically generate mesh that allow user to quickly build both simple and complex models for environmental hydrodynamics simulations. Most available open-source model with similar features are closed-source or research code that require significant coding. 

# Features
BG_Flood interface is through a text-based instruction file that is both easy to read by a human and easy to generate via scripts. 
- Text based for 1-dimensional input/output (e.g. timeseries)
- NetCDF based input/output for 2D and 3D input/outputs

All gridded inputs are automatically interpolated to the model mesh by the model. The model also supports multiple forcing types:

BG_Flood is designed by a team of scientists for various flood hazard processes including:
- Tsunami / land/seafloor deformation (needs to be precomputed) at any instant for any duration
- Wind forcing (either uniform or space-time varying)
- Atmospheric pressure forcing (space-time varying NetCDF)
- Rainfall
- River "injection" as vertical discharge
- Absorbing boundary for water level (either uniform or space-time varying)
- Segment based boundary definition
- Non-rectangular domains
 





# Citations

# Acknowledgements

# References