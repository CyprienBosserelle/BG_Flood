# Jet tutorial with Julia


This is a very simple example that shows the model stability in transitional flows.

## Make a bathymetry
Below is a Julia script to make the bathymetry.
``` julia
    using GMT
    
    ny=16*16;
    nx=16*32;
    
    dx=5.0;
    
    xmin=0;
    xmax=nx*dx;

    ymin=0;
    ymax=ny*dx;
    
    Bathy=zeros(nx,ny);

    Bathy.=-5.0;

    Bathy[170:172,:].=5.0;
    Bathy[170:172,127:129].=-5.0;

    Bathy[:,1:2].=5.0;
    Bathy[:,(end-1):end].=5.0;

    G = mat2grid(transpose(Bathy), 1,[xmin xmax ymin ymax -5.0 5.0 1 dx dx])
    cmap = grd2cpt(G);      # Compute a colormap with the grid's data range
    grdimage(G, lw=:thinnest, color=cmap, fmt=:png, show=true)

    gmtwrite("bathy.asc", G; id="ef");
```

The result is a grid domain looking like this:
![Bathy file](https://github.com/CyprienBosserelle/BG/blob/development/Examples/Jet/bathy_jet.png)
You can also use the bathy.asc file in the example folder.

## Make Bnd files
We are going to set the water level to 0.0m on the right and 1.0 on the left and keep it constant. To do that we create 2 files `right.txt` and `left.bnd`

``` text title="right.bnd"
# This is the right boundary
0.0 0.0
3600.0 0.0
```

``` text title="left.bnd"
# This is the left boundary
0.0 1.0
3600.0 1.0
```

## Set up the `BG_param.txt` file
First it is good practice to document your parameter file:

``` text title="BG_param.txt"

    ##############################
    ## Jet demo
    # CB 04/05/2019
```
Then specify the bathymetry file to use. We will use the entire domain and resolution of the file so no need for other parameter for specifying the domain.
``` text
    bathy=bathy.asc
```
Specify the parameter theta to control the numerical diffusion of the model. but first let's leave it to the default.
``` txt
    theta=1.3;
```
This is a relatively small model so we can force the netcdf variable to be saved as floats.
``` txt
    smallnc=0
```
Sepcify the model duration, output timestep and output file name and variables
``` txt
    endtime=1800
    outtimestep=10
    outfile=Jet_demo.nc
    outvars=zb,uu,vv,zs,vort;
```
Specify absorbing boundaries for left and right (There is a wal at the top and bottom so no need to specify any boundary there).
``` txt
    right = 3; # Absorbing bnd
    rightbndfile = right.bnd

    left=3; # Absorbing bnd
    leftbndfile = left.txt
```

## Run the model
 If you are on windows simply double-click on the executable and on linux launch the binary.

Vorticity output (`vort` variable) should look like this:

<!--- <video src="https://github.com/CyprienBosserelle/BG/blob/development/Examples/Jet/anim_01.mp4" width="320" height="200" controls preload></video>
[Video](https://youtu.be/f1A7EeeMlls)
--->
![type:video](../videos/jet_vort.mp4)

## Things to try

- What happens when using a different value for theta (1-2.0)
- What happens when specifying a different type of boundary on the right 
- What happens when you bring the boundary close to the jet
- Why is the jet so unstable/asymmetrical? (what are the initial condition like?)

