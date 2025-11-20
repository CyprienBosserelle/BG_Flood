@page GaussianWave Gaussian wave verification


### Intro
This is a basic but common test to verify the integrity of the model equations. This test is also good to show different type of boundaries reflect waves. 
### Goal
* Check that the solution produces symmetrical results
* Test hotstart condition
* Show how different boundary formulation absorb wave

### Settings
#### You will need:
* Flat bottom bathymetry file. (See below) 
* Hotstart file. here only water level is needed to hotstart the water surface (zs)

#### Bathy

#### Initial conditon
Here we follow a similar initial condition as in [Basilisk](http://basilisk.fr/Tutorial)
#### BG_param.txt
    bathy = bathy.nc?z;
    # Model controls
    gpudevice = 0;
    spherical = 0;
    theta=1.3;
    doubleprecision=1
    zsinit=0.0
    hotstartfile=zsinit.nc
    # Flow parameters
    eps = 0.00010000;
    cf=0.0000;
    frictionmodel = 0
    # Timekeeping parameters
    CFL = 0.500000;
    outputtimestep = 600.0;
    outvars = zb, hh, uu, vv, hhmax, zs, zsmax;
    endtime = 18000.0;
    left=2;
    right=2;
    top=2;
    bot=2;
    topbndfile=bnbzero.txt;
    botbndfile=bnbzero.txt;
    rightbndfile=bnbzero.txt;
    leftbndfile=bnbzero.txt;
    smallnc=0;
    outfile = Testbed-Gaussian.nc;

### Result

### Run times

## Conical Island test case
### Goals
* Check Boundary forcing
* Check wave propagation and runup against experimental data


## Monai test case
### Goals
* Check Boundary forcing
* Check wave propagation and runup on complex topography against experimental data

### Status
Success
### Settings

### Results
Model behaves similar to other codes and ca reproduce the bulk of the tsunami waves. Results are comparable to the same class of models. The double precision simulation is virtually identical to the Double precision run confirming that in this case there is not much point on using the Double precision.

skill assessment:
**Single precision**

| ------------- | RMS error (m) | Will-corr | BSS | 
| ------------- |:-------------:|  --------:| -----:|
| Gauge 1 | 0.003391 | 0.978040 | 0.910979 |
| Gauge 2 | 0.003735 | 0.965899 | 0.870943 |
| Gauge 3 | 0.003284 | 0.976699 | 0.884080 |

**Double precision**

| ------------- | RMS error (m) | Will-corr | BSS | 
| ------------- |:-------------:|  --------:| -----:|
| Gauge 1 | 0.003391 | 0.978039 | 0.910981 |
| Gauge 2 | 0.003733 | 0.965948 | 0.871093 |
| Gauge 3 | 0.003283 | 0.976704 | 0.884097 |


![](https://github.com/CyprienBosserelle/Basil_Cart_StV/blob/master/Examples/Monai/Results/newbathy3_Gauges.png)
**Figure 1.** Measured and simulated timeseries for the Monai benchmark


### Run times
| GPU | Quadro K620 | GeForce GTS 450 |other GPU |
| ------------- |:-------------:| -----:| -----:|
| **Single Precision (s)** | 27 | 20 | XX |
| **Double Precision (s)** | XX | 34 | XX |

## Rain-on-grid
### Goals
* Check rainfall forcing
* Check model mass conservation
* Compare model performance to experimental data
### Settings

### Results
![Rain on grid model vs measured fluxes](https://github.com/CyprienBosserelle/Basil_Cart_StV/blob/master/Examples/Rain_Cea2008/Results/test_Rainongrid.png)
# Examples
## Jet

## 2010 Maule tsunami propagation

## Nadi 2012 flood

