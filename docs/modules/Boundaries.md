# Boundaries
Making sure you understand boundaries is critical to make a better model.

when specifying boumndary you need to specify where is your boundary, what type of boundary you want (see __type__ below) and water level you want to specify.

Boundaries are applied to the open side of all block that have a face with no neighbour. Selecting blocks to applie a boundary is through a polygon.

# usage

For specifying 

`bndseg = type,polygonfile,boundarydata`

or

`bnd = type,polygonfile,boundarydata`

Where __type__ is:
* type 0: Wall
* type 1: Neumann boundaries (default)
* type 2: Dirichtlet water level boundary
* type 3: Absorbing boundary water level boundary
* type 4: read from other model (currently deprecated)

__polygonfile__ is a file with 2 columns for coordinate drawing a polygon (last line should be the same as first line). All the boundary blocks inside (or at least touches) that polygon will apply the boundary (only for the sides with no neighbour)


Example of polygon file:
```
0.0 0.0
1.0 0.0
1.0 1.0
0.0 1.0
0.0 0.0

```

__boundarydata__ is the data that is applied to the boundary (only 2 and 3). This can be
* Timeseries text file of water level: the values will apply to all the open side in the selected blocks
Example of boundarydata text file:
```
0.0 0.0
3600.0 1.0
36000.0 1.0

```
* NetCDF file with a variable with 3 dimensions (x, y, time)

Specify the variable name in the NetCDF file using the `?` delimiter

`bndseg = 3, mypolyfile.txt, my3dfile.nc?myvar`


## Short hand for side boundary 
For simple rectangular domains, it is not always practical to specify a polygon file. We created a short-hand for boundary block that lies strictly to the edge of the domain (i.e. touching the bounding box). In that case you can specify the side you want (`left`, `right`, `top`, `bot`) and you don''t have to specify a polygon file (one is created internally).e.g.:

`left=3,mybnd.txt`


## Area of interest bnd
In complex domain (e.g. a catchment outline) it may not be very productive to specify a boundary segment. by you may still want to ontrol what happens there. 
`aoibnd = bndtype` allow the user to specify the type of bnd for the blocks that do not fall in any boundary segments. if specifying type 2 or 3 the water level will be forced by the `zsinit` value.

## Overlaping boundary segments
When specifying boundary segment they may overlap. In this case the order of the given bndseg matters. The last bndseg will be assigned to overlapped blocks. Internally to the model, each bndseg operation is independent and for a given block will be overriden by any subsequent call. 


# Bnd Type details

## Wall
Normal flux to the wall will be set to 0.0. This is very reflective but does not leak and remains stable even with high froud numbers. This is usually reserved for catchment edges or for replicating lab test.


## Neumann
Normal flux to the boundary will be assigned the same value as the oposite face of the cell. This will let water through but is still relatively reflective. Critically, water can leak back in the model.

Because it leaks, it shouldn't be used at rivers but are usually OK for ocean next to an absorbing boundary.

## Dirichlet
Dirichlet boundary requires a specified water level timeseries and calculates the flux at the boundary to enforce it. This is the prefered boundary when specifying tsunami and sharp changes in water level but will reflect off incoming waves. It can also trap eddies and cause instabilities.

## Absorbing
absorbing boundary is the prefered type of boundary for most applications. It requires a specified water level timeseries but is generally very stable and does not reflect incoming waves and eddies. Users can control (somewhat) the timescale of the absorbing capabilities by modifying the relaxation time and filtering time. default values are:

```
bndrelaxtime = 3600.0;
bndfiltertime = 60.0;
```
## Nesting
This is currently deprecated but will be rebuilt at some point. in the mean time you can nest water level with type 2 or 3 by specifying a NetCDF boundary data. 

# Other boundary effect

## Tapering
When cold-starting a model it may be desirable to slowly transition from initial condition to what is imposed on the boundary. A taper time (in seconds) can be applied to smooth the transition. e.g.:

`bndtaper = 600.0`

## zsoffset

`zsoffset` paramter is useful to automatically shift all imput water level by a given offset. This can be used for sea-level rise scenario or for changing datum when water level and DEM are kept is separate datums.

>Note that zsoffset only applies to water levels in forcing and initial condition. it does not apply to DEM or other elevations.

## Atm pressure
Atmospheric pressure impact is calculated and applied to boundary water level. the wate level specified are expected to have none of the local atmospheric effect.

