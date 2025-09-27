# BG Flood
Numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinement type grid. The model was designed with the goal of simulating inundation (River, Storm surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU but the adaptive/multi-resolution/AMR is being implemented and not yet operational.

The core SWE engine and adaptivity has been inspired and taken from St Venant solver from [Basilisk](http://basilisk.fr/) and the CUDA GPU memory model has been inspired by the work from [Vacondio _et al._2017](https://dl.acm.org/citation.cfm?id=3031292))

The adaptive version of the model is currently under heavy development. Makefile in the master branch is not operational and general process may be broken in other branches as well. 



[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)


[![Documentation (Work in progress)](https://img.shields.io/badge/doc-Doxygen-blue)](https://aliceharang.github.io/)

## Reference

Bosserelle C., Lane E., Harang A., (2021) BG-Flood: A GPU adaptive, open-source, general inundation hazard model. Proceedings of the Australasian Coasts & Ports 2021 Conference. 


## Testing
CI test (development branch):![Test](https://github.com/CyprienBosserelle/BG_Flood/actions/workflows//main.yml/badge.svg?branch=development)
