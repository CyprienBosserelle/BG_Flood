# Gaussian Wave
This is a great example to test whether there are bugs in the model and how the boundary work.

## Bathymetry
We start with a flat bathymetry with 0.0 everywhere. You still need a file for that!

Here a couple of suggestion on making the file:

- Using GMT:
``` bash
grdmath -R-5/5/-5/5 -I0.03921 0 MUL = bathy.nc
```

Using Julia: See section below with the hotstart file.

In any case you can pick up the file in the example folder.

## Hortstart
We want to setup a bump in the water level centered in the middle of the bathy. IN the example below this is done using Julia, but it should be easily done in Matlab or Python. Note that the script below also generates a bathymetry file.

``` julia

    using GMT
    
    xo=-5;
    yo=-5;
    
    nx=16*16;
    ny=16*16;
    
    len=10.0;
    
    dx=len/(nx-1);
    
    xx=collect(xo:len/(nx-1):(len+xo));
    yy=collect(yo:len/(ny-1):(len+yo));
    
    # Make a bathy file
    zb=zeros(nx,ny);
    G = mat2grid(transpose(zb), 1,[xx[1] xx[end] yy[1] yy[end] minimum(zb) maximum(zb) 1 dx dx])
    gmtwrite("bathy.asc", G; id="ef");
    
    #make the hotstart file
    hh=zeros(nx,ny);
    for i=1:nx
        for j=1:ny
            hh[i,j] = 1.0.+ 1.0.*exp.(-1.0*(xx[i].*xx[i] .+ yy[j].*yy[j]));
            #hh[i,j] =
        end
    end
    
    G = mat2grid(transpose(hh), 1,[xx[1] xx[end] yy[1] yy[end] minimum(hh) maximum(hh) 1 dx dx])
    cmap = grd2cpt(G);      # Compute a colormap with the grid's data range
    grdimage(G, lw=:thinnest, color=cmap, fmt=:png, show=true)
    
    gmtwrite("gauss.asc", G; id="ef");
    gmtwrite("gauss.nc", G);
    # GMT netcdf variable is "z" by default but the hotstart file needs "zs" for water surface
    gmt("grdmath gauss.nc 1.0 MUL = gauss_zs.nc?zs");

```

## Make Bnd files
We are going to set the water level to 0.0m on all 4 sides and keep it constant. To do that we create 1 files `zero.txt`:
``` txt
    # This is the a boundary
    0.0 0.0
    3600.0 0.0
```

## Set up the `BG_param.txt` file
First it is good practice to document your parameter file:
``` txt title="BG_param.txt"
    ##############################
    ## Gaussian bump demo
    # CB 04/05/2019
```
Then specify the bathymetry file to use. We will use the entire domain and resolution of the file so no need for other parameter for specifying the domain.
``` txt
    bathy=bathy.asc
```
This is a relatively small model so we can force the netcdf variable to be saved as floats.
``` txt
    smallnc=0
```
Specify the hotstart file:
``` txt
    hotstartfile=gauss_zs.nc;
```
Boundary conditions are all the same :
``` txt
    right = 3; # Absorbing bnd
    rightbndfile = zeros.txt
    
    top = 3; # Absorbing bnd
    topbndfile = zeros.txt
    
    bot = 3; # Absorbing bnd
    botbndfile = zeros.txt
    
    left=3; # Absorbing bnd
    leftbndfile=zeros.txt
```
Time keeping:
``` txt
    endtime=20;
    outtimestep=1;
```
Output parameters:
``` txt
    # Netcdf file for snapshot of the model domain
    outfile=Gauss_demo.nc
    outvars=zb,uu,vv,zs,vort;
    
    # Outpout a single txt file with all the model steps at the nearest node to location x=0.0, y=-4.0
    # This file will contain 5 column: time,zs,hh,uu,vv
    TSOfile=Southside.txt;
    TSnode=0.0,-4.0;
```
## Run the model
Plot Southside.txt in your favorite tool.

## Things to try

- What happens with different boundary types
- Try running with double precision. What is the difference?