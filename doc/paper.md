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
`BG_Flood` is a numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinment type grid. The model was designed with the goal of simulating inundation (River, Storm surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU with adaptive mesh being generated at the start of tee simulation.

The core SWE engines and adaptivity has been inspired and taken from St Venant solver from `Basilisk` (Popinet 2011) and the CUDA GPU memory model has been inspired by the work from (Vacondio et al. 2017). The model is implemented in CUDA-C to maximise GPU usage. 


# Statement of need
Flood hazard assessment and forecasting aften require physics-based simulations to accurately evalute areas exposed to hazard. These simulation are often completted using hydrodynamics models using high-resolution to capture small landscape and flow features (e.g. small drains, hydraulic jump), but also a large enough domain to capture large scale features where the hazard forms and or amplifies (e.g. river catchment, continental shelf).



Effectivly capturing this cascade of scale is is only acheivable with unstructured or adaptive mesh. Unstructured mesh generation rely heavily on user's input for generating a suitable mesh. These model also do not generally offer the flexibility in of Adaptive mesh. Adaptive mesh on the other hand can be generated easily with small usier input and adapted as the computation progresses but this flixibility is linked to a significant increase in computationalcost. 

`BG_Flood` aimes to offer a balance between 


Most available open-source model also do not offer GPU enabled code that is well optimised and instead for the few that can offer  

Enabling GPU and semi-automatic mesh refinement is critical to enable rapid devlopement of flood assessment that doesn't compromise physics simulated. 

BG_Flood interface is through txt-based instruction and NetCDF ass well as txt file input to allow the model to be used in batch and automated systems 



# Citations

# Acknowledgements

# References