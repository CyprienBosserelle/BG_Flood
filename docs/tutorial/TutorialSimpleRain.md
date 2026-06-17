# Basic rain flood tutorial

The objective of this tutorial is to explain the basic use of BG_Flood and starting a simple model from scratch.

## START SIMPLE !
It is easy to get excited about starting a new model and adding lots of complex features (spatially variable rainfall, a friction map, a tide boundary, and a bunch of streams and culverts). But it is important to start simple. It is a lot harder to find what is going wrong with a complex model.

## Tools
BG_Flood is not a one-stop shop for making inundation models, and you will need to have a decent background on GIS data manipulation to develop complex models. A bit of knowledge on scripting languages does help too. In this tutorial we use only BG_Flood, but in my everyday model development I use:

* Scripting tools/notebooks make everything easier and easier to repeat operations. I use Julia, but many of my colleagues use Python.
* QGIS for quick visualisation of input and outputs.
* GMT for manipulating grids.

Don't have or like these tools? There are plenty of alternative ways to do these operations. Ask your favourite AI agent or search engine for help. 

## DEM 
A good inundation model requires a good topography and bathymetry. This requires a tutorial of its own. To keep this tutorial simple, we provided a ready-to-use DEM of Port Charles in Coromandel, Aotearoa New Zealand.

![dem](../../Examples/SimpleRain/DEM_figure.png)


## Make it rain!
This tutorial is about rain on grid, so we need to put rain. Again keeping it simple, we just generate a text file with a timeseries of rainfall intensities (in mm/h). 

### Go big or go home
When developing a new model, it is useful to push the model to some extreme. This usually helps find problematic areas where the conditioning may not be optimal. So here I'm putting 100 mm/h of rain for an hour. In tropical areas, that would be about your 1% AEP 1-hour duration event. In NZ, this is a bit more extreme.

Here is my file: `rain.txt`

```
# time  rain_int
0.0   100.0
3600.0  100.0
3601.0  0.0
36000.0 0.0
```

> **Note:** BG_Flood reads time either as calendar datetime or seconds from an arbitrary date. Importantly, time steps do not need to be uniform; BG_Flood will linearly interpolate between them. So here, I want rain to go for one hour and then stop.

> **Important:** You need 0.0 rainfall after the rain stops to be in the rain file because the model will refuse to go beyond the last time in forcing! So here, I will not be able to (and shouldn't) run the model beyond 10 hours. 


## Make a parameter file
Yes, rainfall and a DEM are all you need to get a simple model started. Again, don't expect much from such a simple model, but it is a basic building block for a more complex and realistic model.

### General good practice for your parameter files

Add comments about what the model is meant to do and why. This makes it easier to modify when you revisit a model. I like to put a header on my files, but sometimes get a bit lazy.

BG_Flood will ignore any line starting with `#`.

```
################
## My BG_Flood simple rain model
#################

# Port Charles stress test 
```

### Tell the model about the DEM

BG_Flood parameters are always on the left of an `=` sign and the value of the parameters on the right. Beware that some parameters expect several values (normally separated with a comma).

For NetCDF input like a DEM, you need to specify the name of the variable in the NetCDF file (often `z` or `band1`). This is done by appending a `?` to the file name followed by the variable name.

```
##########
# DEM
##########

dem = PortCharles_DEM.nc?z
```

CF-compliant NetCDF files are self-describing, so BG_Flood takes care of the rest.

> **Tip:** BG_Flood can accept multiple DEMs to be superposed on top of each other. One classic example is a mildly coarse model with bathymetry superposed with a high-resolution DEM of just the land. 

### Defining the model domain
By default, BG_Flood will use the extent and resolution of the first `dem` parameter given by the user and will build a uniform grid with these parameters. 

> **Larger domain:** If you leave this default behaviour, you may notice the actual domain is slightly extended to the right and top of your DEM domain. This is because BG_Flood builds quadtree blocks that are 16x16 cells, and if the domain cuts across one block, it is automatically extended to fit the block. BG_Flood then repeats the values from the innermost cell to extrapolate.

Because this is a simple test, I do not want to run at full DEM resolution (4 m), and you shouldn't start any model at high resolution. BG_Flood makes it easy to build a coarse model, and then, once you are confident this is right, switch to fine resolution. The easiest way to do this is to force the base resolution of the model via:

```
##########
# Domain
##########

dx = 32.0
```

Now this is a bit coarser than my input resolution. BG_Flood will automatically average 4x4 cells to make the 32m cells.

### Specify your rain file

```
##########
# Forcing
##########

rain = rain.txt
```

### Adding a realistic friction
BG_Flood includes several roughness formulations. You can used Mannings (very commonly used but somewhat flawed) `frictionmodel = -1`. You can use a quadratic friction (The default `frictionmodel = 0`, quick and easy but not very realistic). You can also use the Smart friction which requires a roughness length (not as common as manning but less flawed) `frictionmodel = 1`.

For this example we are using a zo of 0.04 with the Smart formulation:

```
##########
# Roughness
##########

frictionmodel = 1

roughness = 0.04

```

### Optional but useful

While we could stop there, it is not very useful to let BG_Flood's default behaviour run the show. In particular, we want to specify what to output, how often, how long and where.

```
##########
# output
##########

endtime = 36000.0
outputtimestep = 600.00
outvars = hmax, zsmax, hUmax, h, zs, u, v, Umax
outfile = StressTest_Port_Charles.nc
```

> **Tip:** BG_Flood will **never** overwrite an existing file. Instead, it will add an incremental number to the end of the filename each time it runs (e.g., `StressTest_Port_Charles_1.nc`).

### Use non-default engine

BG_Flood default engine is great for many use but doesn't do too well with heavy rain on steep catchment (i.e. this tutorial). so to get better results we will change the engine to a more suitable one. Also, when using Rain-on-grid, steep topography can lead to waterfalls and out-of-control flow velocity. We want to prevent that by adding the `vmax` parameter that will prevent flow velocity exceeding 16 m/s in our case.

```
#################
## Use the engine 5 for rain-on-grid
#################

engine = 5

vmax = 16.0

```

## Full BG_param.txt

```
################
## My BG_Flood simple rain model
#################

# Port Charles stress test 

##########
# DEM
##########

dem = PortCharles_DEM.nc?z

##########
# Domain
##########

dx = 32.0

##########
# Forcing
##########

rain = rain.txt

##########
# Roughness
##########

frictionmodel = 1

roughness = 0.04

#################
## Use the multi-layer engine with just 1 layer
#################

engine = 5

vmax = 16.0


##########
# output
##########

endtime = 36000.0
outputtimestep = 600.00
outvars = hmax, zsmax, hUmax, h, zs, u, v, Umax

outfile = StressTest_Port_Charles.nc
```

## Run the model

To run BG_Flood simply launch the executable by double click or via a terminal. 

> **Tip:** BG_Flood looks for a `BG_param.txt` file by default, but when launching via a terminal you can use whatever file name you want just specify it as argument to BG_Flood when you launch. This is useful when keeping several setof parameter for a similar area or for running scenarios.

### Screen log and file log

After launching BG_Flood you should see a bunch of info scrolling on the screen. an almost identical amount of information is generated in a `BG_log.txt` file. This should be your go-to file for info when things do go as planned. if there is an error you should get your clues here. My screen shows this:

```
#################################
BG_Flood v0.95
#################################
#################################
#

Reading parameter file: BG_param.txt ...

Reading bathymetry grid data...
Reading forcing metadata. file: PortCharles_DEM.nc extension: nc
Forcing grid info: nx=3152 ny=2127 dx=4.000000 dy=4.000000 grdalpha=0.000000 xo=1815197.000000 xmax=1827801.000000 yo=5950097.000000 ymax=5958601.000000
Reading CRS information from forcing metadata (file: PortCharles_DEM.nc)
grid info detected: grid_mapping
CRS_info: PROJCS["NZGD2000 / New Zealand Transverse Mercator 2000",GEOGCS["NZGD2000",DATUM["New_Zealand_Geodetic_Datum_2000",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6167"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4167"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",173],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",1600000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","2193"]]

Reading boundary data...

Preparing Rain forcing

Adjusted model domain (xo/xmax/yo/ymax):
        1815195.000000/1827995.000000/5950095.000000/5958799.000000
         Initial resolution (level 0) = 32.000000
Reference time: 2000-01-01T00:00:00
Model Initial time: 0.000000 ;

Model Initial time: 0.000000 ;
Model end time: 36000.000000
There are 1 GPU devices on this machine
Using Device: NVIDIA T1000 8GB

Initializing mesh
        Initial number of blocks: 425; Will be allocating 447 in memory.

Initial condition:
        Cold start

Boundary Segment 0 :  has 80 blocks
Output Times:
0.000000e+00, 6.000000e+02, 1.200000e+03, 1.800000e+03, 2.400000e+03, 3.000000e+03, 3.600000e+03, 4.200000e+03, 4.800000e+03, 5.400000e+03, 6.000000e+03, 6.600000e+03, 7.200000e+03, 7.800000e+03, 8.400000e+03, 9.000000e+03, 9.600000e+03, 1.020000e+04, 1.080000e+04, 1.140000e+04, 1.200000e+04, 1.260000e+04, 1.320000e+04, 1.380000e+04, 1.440000e+04, 1.500000e+04, 1.560000e+04, 1.620000e+04, 1.680000e+04, 1.740000e+04, 1.800000e+04, 1.860000e+04, 1.920000e+04, 1.980000e+04, 2.040000e+04, 2.100000e+04, 2.160000e+04, 2.220000e+04, 2.280000e+04, 2.340000e+04, 2.400000e+04, 2.460000e+04, 2.520000e+04, 2.580000e+04, 2.640000e+04, 2.700000e+04, 2.760000e+04, 2.820000e+04, 2.880000e+04, 2.940000e+04, 3.000000e+04, 3.060000e+04, 3.120000e+04, 3.180000e+04, 3.240000e+04, 3.300000e+04, 3.360000e+04, 3.420000e+04, 3.480000e+04, 3.540000e+04, 3.600000e+04,
Setting up GPU

Model setup complete
#################################
Initialising model main loop
Create netCDF output file...
Warning! Output file name already exist
                Completed
Model Running...

```

And if I'm patient enough (about 3 min on my T1000) the model is finished and I can look at the results.

## Looking at results

For this part I normally use GMT and QGIS to look at the result and make a quick screenshot. There are plenty of other option for reading CCF-compliant NetCDF. pick your favorite.

### Convert the results to plot ready geotiff

You can drage and drop BG_FLood results in your favorite netcdf viewer and visualise the results but for making the results more portable I like to export the results to Geotiff. With GMT it is just one line of code and another line to make the results look better.

First extract the maximum flood depth but ignore any flooding under 10 cm. 

```
gmt grdmath myBGF_outputfile.nc?hmax_P0[9] DUP 0.1 GT 0 NAN MUL = maxflood.nc

gmt grdconvert maxflood.nc -Gmaxflood.tif=gd:GTIFF
```

 > **Tip:** With Rain-on-grid there is water everywhere that's why it is important to remove small flood depth from visualisation.

### What's with the variable name and `_P0`

BG_Flood can handle variable grid size. This is tricky can be annoying to handle in Netcdf format. One way is to save the data in a unstructured format but that would not be easy to visualise and manipulate. Instead we save each level of refinement in a separate variable with it's own grid. that way each level of refinement is a raster in itself and can be stacked with other level of refinement allowing the display of the true grid as a stack of regular grids. Clever... but that requires multiple output variable for the same model variable. the simplest way is to extend the model variable name with its level of refinement `_P0` is level 0, `_P1` is level 1, `_N1` is level -1 etc...
