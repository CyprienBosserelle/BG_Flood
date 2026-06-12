# Culverts
A _simple_ implementation of culvert flow is available in BG_Flood. This document explains the motivation and choice of equation and the usage as well as validations. 
## Motivation
Culverts are key for allowing flow under infrastructures. The implementation in BG_Flood is a simple engineering implementation that calculates inlet and outlet flow without accounting for the flow transiting inside the culverts.

### Culvert types

Culverts are very diverse and all the different culverts behave slightly differently.while this is mean to be a simple implementation we have implemented 3 type of _culverts_:

* **Pumps**: Yes they are not technically culverts but mathematically they are a simple pipe with a fix flow
* **Gated culverts**: culvert with 1-way flow they flow from inlt to outlet but stop if the oultet headwater is higher than the inlet.
* **2-way culverts**: Yopur traditional (short) pipe with flow driven by head difference between either side. There are options to cover different type of pipes shape and inlet and outlet flow restrictions.

### Limitation, This is not a full stormwater network
The culvert implementation in BG_Flood is not a proper stormwater system implementation. The culvert implementation teleports the water from the inlet to the outlet. when timestepos are really small this is not a big deal as long as the culvert is relatively short.  

The implementation conserves mass but the can do funky things if you are not careful. See the **Beware** section below for various gotchas.

# Usage


# Beware

# More ressource

# Test


# Validation 
## Sanity check: Qualitative validation
I've been running some sanity check test. All successful:
- 2-way culverts do flow both
-  wider culvert flow more water than small ones
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