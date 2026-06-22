# Basic validation and checks

 
## Comparison with Basilisk
Engines in BG_Flood are not written from scratch, instead they are translated from Basilisk to BG_Flood. It is easy to check if the models agree. 

To test the comparison we release a gaussian bump on a shallow bassin and track it evolution in Basilisk and BG_Flood.

### Base setup
a simple model it setp with a DEM matrix of 256x256 with all values set to 0.0. and a side width of 1.0m. 
a Gaussian wave with an amplitude of 1.0 m and a characteristic radius of 200 is setup:

<img width="893" height="781" alt="image" src="https://github.com/user-attachments/assets/279e2da5-91ca-4f2c-88e9-20658ca088b9" />

The model is then run for 0.1s an

<img width="885" height="781" alt="image" src="https://github.com/user-attachments/assets/3945b86b-56fb-41e1-9745-cc6ae7a0c8d4" />

The test succeed if the solution is whith 10-6 of the original Basilisk simulation at 8 point along that transect.

<img width="883" height="586" alt="image" src="https://github.com/user-attachments/assets/2c466068-f339-4d25-8ab3-2f92a28f9a73" />

### Kurganov engine

| ix | 15 |  31 | 47 | 63 | 79 | 95 | 111 | 127 |
|--|--|--|--|--|--|--|--|--|
| Basilisk |0.1000 | 0.1000 | 0.1001 | 0.1950 | 0.1367 | 0.0848 | 0.0663 | 0.0637 |
| BG_Flood |0.1000 | 0.1000 | 0.1001 | 0.1950 | 0.1367 | 0.0848 | 0.0663 | 0.0637 |

### Buttinger reconstruction

| ix | 15 |  31 | 47 | 63 | 79 | 95 | 111 | 127 |
|--|--|--|--|--|--|--|--|--|
| Basilisk |0.1000 | 0.1000 | 0.1001 | 0.1951 | 0.1368 | 0.0851 | 0.0663 | 0.0637 |
| BG_Flood |0.1000 | 0.1000 | 0.1001 | 0.1951 | 0.1368 | 0.0851 | 0.0663 | 0.0637 |

### Multi-layer solver (1 layer)

| ix | 15 |  31 | 47 | 63 | 79 | 95 | 111 | 127 |
|--|--|--|--|--|--|--|--|--|
| Basilisk | 0.1000 | 0.1000 | 0.1001 | 0.1911 | 0.1345 | 0.0854 | 0.0666 | 0.0639 |
| BG_Flood | 0.1000 | 0.1000 | 0.1001 | 0.1911 | 0.1345 | 0.0854 | 0.0666 | 0.0639 |

## Lake-at-rest
Lake-at-rest test is the basic test where a lake is initialised in the model with no external forcing. The goal is that velocity remain close to zero and water-level remains identical. This checks that there are no issues in the engine hydrostatic reconstruction.  

Here is a water level after long enough wait:
<img width="870" height="555" alt="image" src="https://github.com/user-attachments/assets/9ed44e70-404c-4ba7-9cfb-7de8cf7593f7" />

Here is the velocity:

<img width="868" height="570" alt="image" src="https://github.com/user-attachments/assets/1bc9756d-96c4-441d-964b-9723bcbd1b6e" />

It is normal for the velocity to be non-zero. this is due to the round off error of computer calculations (amount to 10<sup>-6</sup>). In this example we used floating point precision math (default in BGFlood) which is faster but less precise. if we tell BG_Flood to use double precision ```doubleprecision = 1``` (note it is turned on by default when using rain-on-grid), we get much smaller round off errors (10<sup>-15</sup>):


<img width="869" height="565" alt="image" src="https://github.com/user-attachments/assets/e26355c5-d671-495f-aed4-494107a67b71" />


## Mass conservation 
Mass should be conserved in a model except when water is meant to leave the domain. Mass conservation is a basic test where whe check if mass of water already in the model and what is meant to be injected (river and/or rain) stay in the model (with wall bnds). This test is basic and trivial test with mild slope and already wet environment. It is, however a surprisingly hard test to suceed on steep dry slopes. 

The simple mass conservation test is great for checking engine flaws when using rain on grid. Here we push it to the extreme by throwing 100 m<sup>3</sup>/s for 100 s over 1 km<sup>2</sup> (equivalent 10 mm of rainfall falling in 100 s which is equivalent to a rate of 360 mm/h for 1.5-ish minutes ). Note that there are a few tricks to to force the Buttinger to conserve mass but we decided not to turn them off for these tests. 

When topography is gentle mass conservation is a given but when using steep topography things start to break down. This is the extreme bit. Clearly the Kurganov is better at conserving Mass and the new engine is not bad either.

| Test | Theory [m<sup>3</sup>] | Kurganov Engine | Buttinger Engine | 1-layer Multi-layer Engine |
|---|---|---|---|---|
| No Topo, Wet ground | 10000.0 | -0.00071% | -0.00290% | 0.00351% |
| No Topo Dry ground | 10000.0 |  0.00027% | 0.00023% | 0.00022% |
| Real Topo Dry ground | 10000.0 | 2.24% | -13.03% | 3.10% |



### River discharge and Rainfall

The test above confirm that rainfall forcing is consistent with theory (even when using insane amount of rainfall). River discharge mass conservation was not tested above but a simple test is implemented internally. It is good to check that the injection of water is same as expected.




## Boundary check
We have checks to ensure that boundary work. 

### Wall boundary are impermable
IN this test we run a river down a steep slope and checks the mass inside the model is conserved. We do that along each side of the model.


## Adaptive
Eahc of the test are run in a uniform resolution or with a variable resolution. for the mass conservation test the goal is the same but for propagation the results need to be different. More importantly in the variable resolution we chack that the solution are symetrical and that mesh structure do not impede wave propagation.



