\defgroup GrpJetJ Jet tutorial
@{
    \ingroup GrpTutorial

# Jet
This is a very simple example that shows the model stability in transitional flows.

## Make a bathymetry
Below is a Julia script to make the bathymetry.

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

The result is a grid domain looking like this:
![Bathy file](https://github.com/CyprienBosserelle/BG/blob/development/Examples/Jet/bathy_jet.png)
You can also use the bathy.asc file in the example folder.

## Make Bnd files
We are going to set the water level to 0.0m on the right and 1.0 on the left and keep it constant. To dio that we create 2 files `right.txt` and `left.bnd`

### `right.bnd`

    # This is the right boundary
    0.0 0.0
    3600.0 0.0

### `left.bnd`

    # This is the right boundary
    0.0 1.0
    3600.0 1.0

## Set up the `BG_param.txt` file
First it is good practice to document your parameter file:

    ##############################
    ## Jet demo
    # CB 04/05/2019

Then specify the bathymetry file to use. We will use the entire domain and resolution of the file so no need for other parameter for specifying the domain.

    bathy=bathy.asc

Specify the parameter theta to control the numerical diffusion of the model. but first let's leave it to the default.

    theta=1.3;

This is a relatively small model so we can force the netcdf variable to be saved as floats.

    smallnc=0

Sepcify the model duration, output timestep and output file name and variables

    endtime=1800
    outtimestep=10
    outfile=Jet_demo.nc
    outvars=zb,uu,vv,zs,vort;

Specify absorbing boundaries for left and right (There is a wal at the top and bottom so no need to specify any boundary there).

    right = 3; # Absorbing bnd
    rightbndfile = right.bnd

    left=3; # Absorbing bnd
    leftbndfile = left.txt

## Run the model
 If you are on windows simply double-click on the executable and in linux launch the binary.

Vorticity output should look like this:

<video src="https://github.com/CyprienBosserelle/BG/blob/development/Examples/Jet/anim_01.mp4" width="320" height="200" controls preload></video>
[Video](https://youtu.be/f1A7EeeMlls)

## Things to try
* What happens when using a different value for theta (1-2.0)
* What happens when specifying a different type of boundary on the right 
* What happens when you bring the boundary close to the jet
* Why is the jet so unstable/asymmetrical? (what are the initial condition like?)

# Gaussian Wave
This is a great example to test whether there are bugs in the model and how the boundary work.

## Bathymetry
We start with a flat bathymetry with 0.0 everywhere. You still need a file for that!

Here a couple of suggestion on making the file:
Using GMT:
`grdmath -R-5/5/-5/5 -I0.03921 0 MUL = bathy.nc`

Using Julia: See section below with the hotstart file.

In any case you can pick up the file in the example folder.

## Hortstart
We want to setup a bump in the water level centered in the middle of the bathy. IN the example below this is done using Julia, but it should be easily done in Matlab or Python. Note that the script below also generates a bathymetry file.



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

## Make Bnd files
We are going to set the water level to 0.0m on all 4 sides and keep it constant. To do that we create 1 files `zero.txt`:

    # This is the a boundary
    0.0 0.0
    3600.0 0.0

## Set up the `BG_param.txt` file
First it is good practice to document your parameter file:

    ##############################
    ## Gaussian bump demo
    # CB 04/05/2019

Then specify the bathymetry file to use. We will use the entire domain and resolution of the file so no need for other parameter for specifying the domain.

    bathy=bathy.asc

This is a relatively small model so we can force the netcdf variable to be saved as floats.

    smallnc=0

Specify the hotstart file:

    hotstartfile=gauss_zs.nc;

Boundary conditions are all the same :

    right = 3; # Absorbing bnd
    rightbndfile = zeros.txt
    
    top = 3; # Absorbing bnd
    topbndfile = zeros.txt
    
    bot = 3; # Absorbing bnd
    botbndfile = zeros.txt
    
    left=3; # Absorbing bnd
    leftbndfile=zeros.txt

Time keeping:

    endtime=20;
    outtimestep=1;

Output parameters:

    # Netcdf file for snapshot of the model domain
    outfile=Gauss_demo.nc
    outvars=zb,uu,vv,zs,vort;
    
    # Outpout a single txt file with all the model steps at the nearest node to location x=0.0, y=-4.0
    # This file will contain 5 column: time,zs,hh,uu,vv
    TSOfile=Southside.txt;
    TSnode=0.0,-4.0;

## Run the model
Plot Southside.txt in your favorite tool.

## Things to try:
* What happens with different boundary types
* Try running with double precision. What is the difference?

# Transpacific tsunami
This tutorial is the next stat from the Gaussian wave. Here we produce a realistic tsunami and let it propagate across the Pacific.

## Bathy and domain definition
The bathymetry file we are using was extracted from the GEBCO gobal bathymetry using GMT command grdcut / grdsample. This section needs a tutorial of its own. Here in the `BG_param.txt` we specify the file name and that it is a spherical model domain.

    # Bathymetry file
    bathy = Tpac_big.asc;
    spherical = 1;

The file covers a bigger area than we want to use for the simulation so we restict the domain:

    ymin=-78.0
    ymax=14.32
    
    dx=0.08;

Also we do not want to simulate Blocks that are entirely covered in land where the elevation is bigger than say 30.0m above the datum

    mask = 30.0;


## Initial tsunami wave
See matlab file in the folder or simply use:

    hotstartfile=Maule_zs_init_simpleflt.nc;

## Boundary
In the previous tutorial you have seen how different boundary type let's wave through with minimal reflection. You choose.

## Time Keeping

    totaltime = 0.000000; # Start time
    endtime = 54000.000000;
    outputtimestep = 600.000000;

## Outputs

    outvars = hh,zs,zsmax;
    # Files
    outfile = Output_simple_fault.nc;
    smallnc = 0; #if smallnc==1 all Output are scaled and saved as a short int
    
    TSnode = -86.374,-17.984
    TSOfile = SW_Lima.txt

## Things to try
* Try changing the model domain and resolution. What happens if part of the domain is outside of the area covered by the bathymetry?
* Try outputing a time series near Christchurch
  
# River + Rain = Waikanae example
Here we setup a model of the Waikanae Catchment (on the Kapiti Coast in New Zealand) force the tide on one of the open boundary 


@}