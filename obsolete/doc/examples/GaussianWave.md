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
