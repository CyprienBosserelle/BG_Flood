# BG Flood
Numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinement type grid. The model was designed with the goal of simulating inundation (Rain, river, storm-surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU using variable resolution grid.Developent of .

The core SWE engine and adaptivity has been inspired and taken from St Venant solver from [Basilisk](http://basilisk.fr/) and the CUDA GPU memory model has been inspired by the work from [Vacondio _et al._2017](https://dl.acm.org/citation.cfm?id=3031292))

The adaptive version of the model is currently under heavy development. Makefile in the master branch is not operational and general process may be broken in other branches as well. 



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
## Reference

Bosserelle C., Lane E., Harang A., (2021) BG-Flood: A GPU adaptive, open-source, general inundation hazard model. Proceedings of the Australasian Coasts & Ports 2021 Conference. [PDF](docs/150_bosserelle_finalpaper.pdf)


## Testing and validation

BG_Flood has 12+ internal tests to make sure the hydrodynamics engine and input work as expected. 

Below is the result of all the tests (development branch):

![Test](https://github.com/CyprienBosserelle/BG_Flood/actions/workflows//main.yml/badge.svg?branch=development)

To run the test yourself add ```test = xx``` where xx is the test number (1 to 14) to your param file. 