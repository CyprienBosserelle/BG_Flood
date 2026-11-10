# Atmospheric pressure forcing
 
## Motivation
Atmospheric pressure change can create elevated water levels in oceans (via the inverse barometer effect) and small moving pressure gradient can create meteotsunami. this effect can be captured in BG_Flood by specifying an atmospheric pressure field in the parameter file. 

# Usage

A pressure field (i.e. map-series, spatially and temporally varying) is expected for forcing. This requires a NetCDF file: 

`Atmp=AtmosphericPressure.nc?p`

By default the absolute atm pressure field is expected in Pa. However the model computes the pressure anomaly by removing the reference pressure. Hence by default the reference is in Pa:

`Paref = 101300.0;`

If unit is hPa then user should use 1013.0

Conversion between atmospheric pressure changes to water level changes in Pa is specified by the `Pa2m` parameter 

`Pa2m = 0.00009916`

If unit is hPa then user should use 0.009916.

## Impact on boundaries
When specifying water level boundaries, the model will add-on the calculated impact of atmospheric pressure. This is usually easier for the user but this cannot be overruled.




# Video 
<!-- <video src="https://cyprienbosserelle.github.io/BG_Flood/videos/Tsunami_AtmosphericP.mp4" controls="" loop="" autoplay="" muted="" style="width:80%;">
  Your browser does not support the video tag.
</video> -->