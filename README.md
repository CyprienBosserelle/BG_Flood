# BG Flood
Numerical model for simulating shallow water hydrodynamics on the GPU using an Adaptive Mesh Refinement type grid. The model was designed with the goal of simulating inundation (River, Storm surge or tsunami). The model uses a Block Uniform Quadtree approach that runs on the GPU but the adaptive/multi-resolution/AMR is being implemented and not yet operational.

The core SWE engine and adaptivity has been inspired and taken from St Venant solver from [Basilisk](http://basilisk.fr/) and the CUDA GPU memory model has been inspired by the work from [Vacondio _et al._2017](https://dl.acm.org/citation.cfm?id=3031292))

The adaptive version of the model is currently under heavy development. Makefile in the master branch is not operational and general process may be broken in other branches as well. 



[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8d871cf493e94a6eb474eaa30f573583)](https://www.codacy.com/project/CyprienBosserelle/Basil_Cart_StV/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CyprienBosserelle/Basil_Cart_StV&amp;utm_campaign=Badge_Grade_Dashboard)
[![Documentation](https://img.shields.io/badge/doc-Doxygen-blue)](https://cyprienbosserelle.github.io/BG_Flood/)
[![Documentation](https://img.shields.io/badge/doc-Doxygen-blue)](https://aliceharang.github.io/)

## Build status
Last developments are in the "development" branch: [![Build Status](https://travis-ci.com/CyprienBosserelle/BG_Flood.svg?branch=development)](https://travis-ci.com/CyprienBosserelle/BG_Flood)

## Testing
CI test (development branch):![Test](https://github.com/CyprienBosserelle/BG_Flood/actions/workflows//main.yml/badge.svg?branch=development)
