# BG Flood
Numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinement type grid. The model was designed with the goal of simulating inundation (Rain, river, storm-surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU using variable resolution grid.

The core SWE engine and adaptivity has been inspired and taken from St Venant solver from [Basilisk](http://basilisk.fr/) and the CUDA GPU memory model has been inspired by the work from [Vacondio _et al._2017](https://dl.acm.org/citation.cfm?id=3031292))

The model is under constant development with features added to extent the processes captured and simplify user inputs. current developement include:
* New engine
* Culverts
* groundwater module
 
The code is open source under a GPL License:
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)



## Documentation and manual

[![DocumentationNNN](https://img.shields.io/badge/Documentation-MkDocs-blue?logo=materialformkdocs)](https://cyprienbosserelle.github.io/BG_Flood/)



## Installation & Compiling
For Windows users we make binaries/executable available for download in the [release](https://github.com/CyprienBosserelle/BG_Flood/releases/tag/v1.0) section.

MacOS and Linux users need to compile the code. Find compile guide [here.](https://cyprienbosserelle.github.io/BG_Flood/Install/#linux)


## Basic usage
The simplest usage of BG_Flood is to move the executable (an accompanying DLLs for Windows users) to a working folder (e.g. BG_Flood\Examples\SimpleRain) and either launch from CLI:
```
./BG_FLood BG_param.txt
```

or by double clicking the excecutable (BG_Flood will automatically look for ```BG_param.txt```)

Refer to [Manual](https://cyprienbosserelle.github.io/BG_Flood/) and examples to understand how to use the param file and  parameters.



## Testing and validation

BG_Flood has 12+ internal tests to make sure the hydrodynamics engine and input work as expected. 

Below is the result of all the tests (development branch):

![Test](https://github.com/CyprienBosserelle/BG_Flood/actions/workflows//main.yml/badge.svg?branch=development)

To run the test yourself add ```test = xx``` where xx is the test number (1 to 14) to your param file. 

## Cite BG_Flood

Bosserelle C., Lane E., Harang A., (2021) BG-Flood: A GPU adaptive, open-source, general inundation hazard model. Proceedings of the Australasian Coasts & Ports 2021 Conference. [PDF](docs/150_bosserelle_finalpaper.pdf)

## Other publications
Harang, A., Lane, E. M., Bosserelle, C., Dean, S., Cattoën, C., Pearson, R., Carey-Smith, T. Srinivasan, R. Shiona, H. Wilkins, M., Smart, G, Flood Hazard in Aotearoa New Zealand under Current and Future Climates (in press)

Pozo, A., Wilson, M., Katurji, M.,  Méndez, F. J., Bosserelle, C., Lane, E. (2026) Hybrid Hydrodynamic-Machine Learning Modelling for Rapid Flood Scenario Assessment: A Case Study in Aotearoa New Zealand. Journal of Flood Risk Management19, no. 2: e70206. https://doi.org/10.1111/jfr3.70206.

Paulik, R., Hosse, L., Pelmard, J., Bosserelle, C., Harang, A., Powell, J., Pearson, R., Carey-Smith, T., Lane, E., Scheele, F., Zorn, C., Wotherspoon, L., Foster, L. (2026) Evaluating New Zealand’s building risk to fluvial and pluvial flooding. Discover Hazards 2, 2 

Xu Z., Bosserelle C., Lane E.,(2024) Nearfield effects of the 2022 Hunga-Tonga volcanic tsunami and implications for a volcanic eruption near the coast. Ocean Engineering 321, 120465

Welsh R., Williams S., Bosserelle C., Paulik R., Chan Ting J., Wild A., Talia L. (2023) Sea-Level Rise Effects on Changing Hazard Exposure to Far-Field Tsunamis in a Volcanic Pacific Island. J. Mar. Sci. Eng. 2023, 11, 945.

Sischka L.; Bosserelle C.; Williams S.; Ting J.C.; Paulik R.; Whitworth, M.; Talia L.; Viskovic P. (2022) Reconstructing the 26 June 1917 Samoa Tsunami Disaster. Appl. Sci. 2022, 12, 3389. https://doi.org/10.3390/app12073389

Bosserelle C., Williams S., Cheung K. F., Lay T., Yamazaki Y., Simi T., et al. (2020). Effects of source faulting and fringing reefs on the 2009 South Pacific Tsunami inundation in southeast Upolu, Samoa. Journal of Geophysical Research: Oceans, 125, e2020JC016537. https://doi.org/10.1029/2020JC016537
