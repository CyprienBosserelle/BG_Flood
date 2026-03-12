# Transpacific tsunami
This tutorial is the next stat from the Gaussian wave. Here we produce a realistic tsunami and let it propagate across the Pacific.

## Bathy and domain definition
The bathymetry file we are using was extracted from the GEBCO gobal bathymetry using GMT command grdcut / grdsample. This section needs a tutorial of its own. Here in the `BG_param.txt` we specify the file name and that it is a spherical model domain.
``` txt title="BG_param.txt"
    # Bathymetry file
    bathy = Tpac_big.asc;
    spherical = 1;
```
The file covers a bigger area than we want to use for the simulation so we restict the domain:
``` txt
    ymin=-78.0
    ymax=14.32
    
    dx=0.08;
```
Also we do not want to simulate Blocks that are entirely covered in land where the elevation is bigger than say 30.0m above the datum
``` txt
    mask = 30.0;
```

## Initial tsunami wave
See matlab file in the folder or simply use:
``` txt
    hotstartfile=Maule_zs_init_simpleflt.nc;
```
## Boundary
In the previous tutorial you have seen how different boundary type let's wave through with minimal reflection. You choose.

## Time Keeping
``` txt
    totaltime = 0.000000; # Start time
    endtime = 54000.000000;
    outputtimestep = 600.000000;
```
## Outputs
``` txt 
    outvars = hh,zs,zsmax;
    # Files
    outfile = Output_simple_fault.nc;
    smallnc = 0; #if smallnc==1 all Output are scaled and saved as a short int
    
    TSnode = -86.374,-17.984
    TSOfile = SW_Lima.txt
```

## Things to try
* Try changing the model domain and resolution. What happens if part of the domain is outside of the area covered by the bathymetry?
* Try outputing a time series near Christchurch

