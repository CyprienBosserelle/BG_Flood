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
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
  - name: Alice Harang
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Emily Lane
    affiliation: 1
affiliations:
 - name: Earth Sciences New Zealand
   index: 1
   
 
date: 1 July 2025
bibliography: paper.bib

# Summary
Flood hazard assessment and forecasting aften require physics-based simulations. These simulation are often completted using hydrodynamics models using high resolution to capture small landscape and flow features (e.g. small drains, hydraulic jump), but also capture large scale domain where the hazard forms and or amplifies (e.g. river catchment, continental shelf). this is only acheivable with unstructured and adaptive mesh. Unfortunatly most available open-source codes only provide unstructured mest that rely heavily on user's input for generating a suitable mesh. These model also do not generally offer a flexibility in reverting mesh generation. Most available open-source model also do not offer GPU enabled code that is well optimised and instead for the few that can offer  

Enabling GPU and semi-automatic mesh refinement is critical to enable rapid devlopement of flood assessment that doesn't compromise physics simulated.     

# Statement of need
BG_Flood is a numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinment type grid. The model was designed with the goal of simulating inundation (River, Storm surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU with adaptive mesh being generated at the start of tee simulation.

The core SWE engines and adaptivity has been inspired and taken from St Venant solver from Basilisk (Popinet XXXX) and the CUDA GPU memory model has been inspired by the work from (Vacondio et al. 2017). The rest of the implementation




