@mainpage

Welcome to the BG-Flood documentation !
# Introduction
BG-Flood is a Shallow Water Equation (SWE) model based on the St Venant formulation of [Basilisk](http://basilisk.fr/) but applied to a Block Uniform Quadtree memory structure on the GPU (based on [Vacondio _et al._](https://dl.acm.org/citation.cfm?id=3031292)).

The model is designed to simulate [flood/inundation](https://english.stackexchange.com/questions/131195/difference-between-flooding-and-inundation) caused by tsunami, river, rain, tide or storm surge (from wind stress or atm pressure differential) or any combination of these. The model does not (yet) include the effect of wind waves. The model is not restricted to simulating inundation hazard and can also be used for depth averaged circulation simulation. 

This wiki is to help users getting started on using the model. This is work in progress... If you have any issue/question post it on the issue board. While the model is fairly user friendly, it was designed for users with a basic understanding of hydrodynamics and some experience in modelling. The wiki (and the model) expect that users have some experience in basic file manipulations and are comfortable with handling NetCDF files for input and output.

=======================

## Model development stage
BG-Flood is still in early development stage but is already useful for real inundation simulation.       

The master branch is at stage 4 of development with 5 stage planned as below
1. ~~Cartesian naive formulation~~
2. ~~Same but with some ease of use for modelling studies~~
3. ~~Block Uniform formulation at constant level with masked blocks (with a full swing of _user friendly_ capabilities)~~
4. ~~Static Block uniform Quad-tree formulation (i.e. adapted mesh at the first step only )~~
5. fully adaptive Block uniform Quad-tree formulation  (i.e. basilisk style wavelet refine)

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-brightgreen.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8d871cf493e94a6eb474eaa30f573583)](https://www.codacy.com/project/CyprienBosserelle/Basil_Cart_StV/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CyprienBosserelle/Basil_Cart_StV&amp;utm_campaign=Badge_Grade_Dashboard)
[![Documentation](https://img.shields.io/badge/doc-Doxygen-blue)](https://cyprienbosserelle.github.io/BG_Flood/)


**Testing**
CI test (development branch):![Test](https://github.com/CyprienBosserelle/BG_Flood/actions/workflows//main.yml/badge.svg?branch=development) 

=======================

## How to install

### From precompiled binaries (Win10 only)
1. download the zip from any release
2. unzip into preferred path
3. Add path to "Path" in environment variables (search for "Edit environment variables for your account")
4. When updating/upgrading BG_Flood simply repeat 1 and 2

### On Linux machines

* @ref Compile-under-linux
* @ref Use-On-SuperComputer

=======================

## How to use the model {#User}

* @ref Manual
* @ref Modules
* @ref Tutorials 
* @ref Test-and-Examples 

===========================================

## How to use/change the code {#Developer}
* @ref Code-map

