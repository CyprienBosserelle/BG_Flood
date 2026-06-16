# Breaching stop banks
 Breaching stop-bank can dramatically increase local flooding or change the dynamics of floods. While BG_Flood does not have a morphoodynamics model, it has internal process that can approximate a breach (or many of them). This was critical when trying to replicate the exTC Gabrielle flood in Heretaunga Plains where more than 10 stop-bank breaches occured.

## How
BG_Flood can deform the ground level at anytime using the `deform` parameter. This was initially designed for tsunami generation from co-seismic seafloor deformation. However, it can be applied to any change on the landscape.   

* `deform` input needs to be the change to landscape to be applied.   

* `deform`gridded input can be any size and can be much smaller than the model domain. you could have a separate grid for each breach

* You can control the start time of the break and the duration of the breach formation with the two paramter attached to deform

# Example

```
deform = myfirstbreach.nc?z,73000.,60.0
deform = my2ndbreach.nc?z,34000.,45.0

```


