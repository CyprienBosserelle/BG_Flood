# Culverts
A _simple_ implementation of culvert flow is available in BG_Flood. This document explains the motivation and choice of equation and the usage as well as validations. 
## Motivation
Culverts are key for allowing flow under infrastructures. The implementation in BG_Flood is a simple engineering implementation that calculates inlet and outlet flow without accounting for the flow transiting inside the culverts.

### Culvert types

Culverts are very diverse and all the different culverts behave slightly differently.while this is mean to be a simple implementation we have implemented 3 type of _culverts_:

* **Pumps**: Yes they are not technically culverts but mathematically they are a simple pipe with a fix flow
* **Gated culverts**: culvert with 1-way flow they flow from inlt to outlet but stop if the oultet headwater is higher than the inlet.
* **2-way culverts**: Yopur traditional (short) pipe with flow driven by head difference between either side. There are options to cover different type of pipes shape and inlet and outlet flow restrictions.

### Limitation. This is not a full stormwater network
The culvert implementation in BG_Flood is not a proper stormwater system implementation. The culvert implementation teleports the water from the inlet to the outlet. when timestepos are really small this is not a big deal as long as the culvert is relatively short.  

The implementation conserves mass but the can do funky things if you are not careful. See the **Beware** section below for various gotchas.

# Usage
A culvert is by using the `culvert =` parameter. Culverts takes at least 6 (comma separated) and up to 14 parameters .


## Simple culvert

```
culvert = type,x1,y1,x2,y2,width|max flow
```

type is :
* __0__ for pumps
* __1__ for 1-way culverts
* __2__ for 2-way culverts

`x1,y1` is the inlet coordinate and `x2,y2` is the outlet coordinate. 

if using __type 0__ (pumps) the 6th parameter is the maximum pump capacity of the pump. For __type 1 and 2__ the 6th parameter is for the culvert width.

## Full culvert
More information can be given about the culvert and this is the recommended way.

```
culvert = type, x1, y1, x2, y2, width, fac, shape, n, k_ex, k_en, Cd, zb1, zb2
```
`fac` is a scale factor. It can be used to limit the culvert flow or for multipliying the number of barrels. `shape` is either `0` for circular or `1` for rectangular. `n`is the Manning's friction of the culvert. `k_ex` is the exit loss coefficient. `k_en` is the entrance loss coefficient. `Cd` is the Discharge coefficient for the submerged culvert. `zb1, zb2` are the invert elevation of the inlet and outlet respectively.

You can repeat the culvert param for as many culver as you have.

## example
```
# this culvert I know well. 1 barrel circular 2m diameter 
Culvert = 2,415169.26,7655100.06,415145.79,7655102.49,2.00,1.00,0,0.01,0.70,0.50,1.0,0.40,0.30

# this one has two barrels
# I don't know the cd paramter (-999 will be corrected to default(1.0 for circular or 0.62 for rectangular))
Culvert = 2,415407.06,7655083.92,415405.44,7655095.28,1.20,2.00,1,0.01,0.70,0.50,-999.00,0.80,0.50

# more culverts
Culvert = 2,416866.91,7655470.24,416854.51,7655532.68,1.50,1.00,0,0.01,0.70,0.50,-999.00,0.80,0.80
Culvert = 2,417401.38,7655566.33,417393.40,7655598.77,3.00,1.00,0,0.01,0.70,0.50,-999.00,0.80,0.80
Culvert = 2,415364.88,7654615.17,415350.60,7654625.46,1.20,1.00,1,0.01,0.70,0.50,-999.00,5.00,4.60
Culvert = 2,417492.49,7654975.29,417501.13,7654994.55,3.00,1.00,0,0.01,0.70,0.50,-999.00,4.30,3.80
Culvert = 2,418565.87,7654389.67,418574.95,7654407.16,2.00,1.00,0,0.01,0.70,0.50,-999.00,4.60,4.20
Culvert = 2,419216.70,7654362.65,419226.22,7654378.60,5.00,1.00,0,0.01,0.70,0.50,-999.00,5.50,5.00
Culvert = 2,419648.66,7654704.73,419631.17,7654722.22,2.00,1.00,0,0.01,0.70,0.50,-999.00,0.50,0.50

# this one I don't know much so we use default values and assume the invert elevation is at DEM level
Culvert = 2,419863.03,7654624.69,419850.63,7654626.61,2.00,1.00
```



# Beware
 * only 1 cell is used as inlet and a single cell as outlet
 * Ideally culvert width is smaller than the cell width. the model doesn't check that so beware, putting a 10m culvert in a 2m resolution cell will cause some serious instabilities.
 * When inlet and outlet (invert) elevation is specified, BG_Flood enforce makes sure the ground levvel is at least as low as this. 
 * Culverts don't conserve velocity! depression and water level  humps may be exagerated.
 
# More ressource
if you are looking for info on what parameter to use for you culvert, refer to the following documents. Note that the formulations used in BG_Flood are closer to the fhwa methods. Beware that BG_Flood uses metric unit system. 

* [fhwa](https://www.fhwa.dot.gov/engineering/hydraulics/pubs/12026/hif12026.pdf)
* [USGS culvert bible](https://pubs.usgs.gov/twri/twri3-a3/)

# Test
Cuvert formulation is tested with a double internal test for inlet control and outlet control similar to those used in the validation.
Use `test=16` to run those tests.

# Validation 
## Sanity check: Qualitative validation
I've been running some sanity check test. All successful:
- 2-way culverts do flow both ways!
- A wider culvert flow more water than small ones
- culver start to flow when water level reach their invert elevation
- in outlet control, a rise in downstream water level result in a similar raise in upstream water level

## Quantitative validation
For validation, calculating the "inverse" problem of the [USGS example](https://pubs.usgs.gov/twri/twri3-a3/pdf/TWRI_3-A3.pdf) on culvert is a good idea. It's what HecRas [does](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/modeling-culverts/culvert-hydraulics/comparison-to-the-usgs-culvert-procedures) too.

In the USGS exercice the heads are imposed and the user calculates the flow. In BG_Flood that is not easy to do since culvert flow rates are not an output of the model. Instead we impose the flow rate on one side and the head downstream and calculate at what level the downstream water level settles.


### Example 6. Type 4 flow through a concrete pipe culvert. Outlet controlled 
To replicate this example we impose a flat water level downstream of 1.524m (5ft), an input flow upstream of the culvert of 3.54m^3/s (125cfs) and a culvert with a round shape 1.22m wide (4ft) and 15.24 m long (50ft). 
```
culvert=2,96.0,100.0,111.24,100.0,1.2192,1.0,1,0.012
river=steadyflowExc6.txt,80.0,98.0,90.0,110.0
right=3,steadylevelExc6.txt
```
USGS simple calculation suggest the headwater needs to reach 2.13m (7ft) giving a head difference of 0.6m. BG_Flood behaves a bit different. it simulates a drawdown (2.34m (1m res) and 2.26m (2m res)) at the intake and super elevated water at the outlet (1.66 at 1m or 1.60m at 2m). so while the headwater seem to be a bit off (~13cm too high), the head different is not as bad (8cm too high at 1m or 6cm too high at 2m). This could be tuned with adjusting K_ex to 0.65 instead of the default 0.7 but that seems unnecessary.

### Example 1:Type I Flow through a corrugated-metal pipe culvert. Inlet controlled

This example is for a inlet controlled flow. A large pipe (~3m) 30m long. Here, USGS suggest a 3.66m (12ft) headwater solution. BG_Flood produces a 3.66 m exactly. showing the solution is stable and comparable to USGS method for predicting culvert flow.