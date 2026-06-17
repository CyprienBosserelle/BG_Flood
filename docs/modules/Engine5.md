# Multi-layer solver
BG_Flood started the implementration of the Basilisk multilayer solver ([Popinet 2020](https://hal.science/hal-02365730)). Despite the name, the implementation was only done for a single layer. multi-layer development is underway.

## Motivation
Adopting the Basilisk multilayer solver is driven by two main goals
* having a non-hydrostatic solver for Tsunami and swell waves
* consider the impact of density driven flow in estuaries

BG_Flood development is very much focus on inuindations drivers and not so much on larger spatial and temporal scale oceanographic processes. While this solver should open the door for simulating some of these processes we will not really focus on these and users may find more user friendly features in ROMS.

### Nice side effect
The engine is claimed to be superior to other engine/solvers/ reconstruction already implemented in BG_Flood. Our experience so far shows:

* Much improvement for rain on steep catchment (see below)
* Sharper shock capture with 1 layer but this may be minor
* More prone to instability

# Usage
If you want to use this engine, simply specify the engine in your param file like this:

```
engine = 5
```

We are still experimenting with conflict with other solver default options and fixes and for now make sure you have the option `wetdryfix=false` in your param file too. 

# Test
There is a test for the solver as part of `test=1` note this is only tested for the selected engine.


# Validation 
Below are a series of validation of the solver for rain-on-grid, dam break and tsunami.

## Rain on grid validation
### Simple mass-conservation test
The simple mass conservation test is great for checking engine flaws when using rain on grid. Here we push it to the extreme by throwing 100 m<sup>3</sup>/s for 100 s over 1 km<sup>2</sup> (equivalent 10 mm of rainfall falling in 100 s which is equivalent to a rate of 360 mm/h for 1.5-ish minutes ). Note that there are a few tricks to  There are a few trick to force the Buttinger to conserve mass but we decided not to turn them on for these tests. 

When topography is gentle mass conservation is a given but when using steep topography things start to break down. This is the extreme bit. Clearly the Kurganov is better at conserving Mass and the new engine is not bad either.

| Test | Theory | Kurganov Engine | Buttinger Engine | 1-layer MLH Engine |
|---|---|---|---|---|
| No Topo, Wet ground | 10000.0 | -0.00071% | -0.00290% | 0.00351% |
| No Topo Dry ground | 10000.0 |  0.00027% | 0.00023% | 0.00022% |
| Real Topo Dry ground | 10000.0 | 2.24% | -13.03% | 3.10% |

The table above show that the Kurganov (the original) engine conserves mass better but the fighures below show that it actually does that by keeping water up the catchment and not letting it flow down (Bad). Buttinger model looks like it is missing a huge amount of mass (Bad) but in reality it is because it lets the water down (Good) but as a consequence it leaves negative water depth upstream (Bad). Thew new engine as a good mass balance and lets the water flow nicely...
 
Water depth after 500 s, Kurganov engine. Note the black pixels are negative values.
<img width="711" height="594" alt="image" src="https://github.com/user-attachments/assets/1ccf8880-6ffa-4b30-bb61-32883654e4d4" />

Water depth after 500 s, Buttinger engine. Note the black pixels are negative values.
<img width="690" height="591" alt="image" src="https://github.com/user-attachments/assets/09d86ae7-d381-4a34-a46b-1817d20d55e3" />

Water depth after 500 s, 1-layer MLH Engine. No negative values and the water has accumulated in the rivers instead of upstream of the steep slopes.
<img width="700" height="598" alt="image" src="https://github.com/user-attachments/assets/007e5a13-9974-480c-93f4-111eae85727d" />

## Cea et al. test
This is one of my favourite test for Rain on grid even though the lab test has some flaw itself. This shows that the Buttinger and 1-layer MLH engines basically perform the same. and despite over-predicting the peak capture the both raising and falling limb really well.

<img width="685" height="483" alt="image" src="https://github.com/user-attachments/assets/109a7ce1-4485-4d20-87ae-94b2abe1452e" />

### Rain-on-grid Summary
The new engine is performing very well and looks like is able to dramatically improve results using rain on grid. In the past BG_Flood didn't see much improvement using old engine with no-infiltration, the new engine might require a tighter use of infiltration for rain-on-grid.

## Dam Break
Dam break lab experiments are very difficult to replicate but they are widely use for model comparison so this is a great to compare engine and compare with other hydrodynamic packages (we won't be doing that explicitly here...).

The Dam break experiment we are replicating here is the classic _dam-break against an isolated obstacle_ from [Soares-Frazão and Zech (2002)](https://www.tandfonline.com/doi/abs/10.1080/00221686.2007.9521830). This test is used in the [_Benchmarking of 2D Hydraulic Modelling Packages_](https://assets.publishing.service.gov.uk/media/5a75765c40f0b6360e4744dd/scho0510bsno-e-e.pdf) from the UK Environment Agency.

### Experimental setup
<img width="1259" height="813" alt="Experimental setup" src="https://github.com/user-attachments/assets/d41ff558-62b6-43a1-a38e-cd7903682abe" />

As expected the model is doing sorta OK against the lab data. And more importantly both Kurganov solver and the 1-layered MLH are doing pretty similar at all gauges. 

### _BG_Flood_ results

### Gauge 1

The model capture the initial shock OK but then struggles because of the breaking of the shock forms rollers.
<img width="702" height="464" alt="image" src="https://github.com/user-attachments/assets/810c3797-6ce3-4f14-86d5-87a7d916acf3" />

<img width="685" height="470" alt="image" src="https://github.com/user-attachments/assets/896ba5a1-4aa3-4d7c-89c8-8ebefb88b729" />


### Gauge 2

<img width="700" height="474" alt="image" src="https://github.com/user-attachments/assets/55acc76d-93f2-4f96-8f0f-e31c67e20a00" />

<img width="685" height="469" alt="image" src="https://github.com/user-attachments/assets/93d07569-8646-4f94-8521-396693aeb2d9" />

### Gauge 6
<img width="686" height="471" alt="image" src="https://github.com/user-attachments/assets/df328c56-38ba-45b8-9aa3-0653dfd93db7" />


### Velocity field 
While the velocity figure above are not too promising the plot below show modelled velocity patterns and amplitude with some canning similarity to the measurements.

#### Downstream velocity
<img width="698" height="809" alt="image" src="https://github.com/user-attachments/assets/db3a3874-7db5-453e-950c-da01ac0aa1ab" />

#### Cross-stream velocity
<img width="703" height="796" alt="image" src="https://github.com/user-attachments/assets/e942ee36-7e1a-4397-8775-756580231f07" />

### Other models
For reference this is how other models did.
<img width="805" height="665" alt="image" src="https://github.com/user-attachments/assets/1009807a-917e-45e6-9f25-1d758b5dcfac" />

### Dam Break summary
The new engine perform at least as well as the previous engine and is as satisfactory as a 2D solver can be for dam break problems.

## Tsunami Validation

This post is to check the validity and performance of the new engine for tsunami problem
### Monai example
One of the hope is that the new engine handles tsunami wave propagation and inundation. The Monai benchmark test is a great test for that.
<img width="1041" height="689" alt="image" src="https://github.com/user-attachments/assets/285556bb-b203-4838-959b-0a5942e211bb" />
The new engine (5) process very similar results as the other engine (1). 

### Kamchatka 2025 earthquake
Here we plug the SIFT inversion of the tsunami for deformation and run the model across the Pacific. We only plot the comparison for a single DART buoys below but we have looked at a bunch and there is only minor difference between the two engine tested.
It's actually quite hard to see the difference between the two model. this is actually a good thing that two fairly different engine produces such similar results.

<img width="1034" height="699" alt="image" src="https://github.com/user-attachments/assets/deccfe3a-4751-474c-9827-281b8d7f5998" />

<img width="1038" height="693" alt="image" src="https://github.com/user-attachments/assets/30e5182c-f485-4a6f-b108-92ed418afc69" />

### Tsunami summary
The new engine shows better shock capture but that might be only helpful for nearshore processes (this is still very good).
