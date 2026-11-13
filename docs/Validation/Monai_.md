# Tsunami Runup onto a Complex Three-dimensional Beach; Monai Valley (Liu et al. 2008)


This is the most famous lab comparison for tsunami inundation modelling. This lab example is inspired from a real event. The Hokkaido-Nansei-Oki tsunami of 1993 that struck Okushiri Island, Japan, with extreme runup height of 30 m a discovered at the tip of a very narrow gulley within a small cove at Monai. This 1:400 laboratory model of Monai was constructed as part of the 

## Model Setup
### Domain
The model is relatively simple to setup but the precise extent of lab experiment Need to be carefully managed in BG_Flood. BG_Flood attempt to extend domain to satify filling all the blocks in the domain. if the domain dimension is not a nice factor of 16 x dx of the domain, BG_Flood pads the domain with repeating values to the right and top to fill the missing area. in this can we need to make sure this is a wall to the top boundary. in our grid we add an extra line at the top of the domain with values at -1. 

No adaptivity is ued as we want to remain at the native resolution and as close as possible to the lab experiment. 

### Boundary
We force the left boundary with the provided model input. BG_Flood interpolates linearly to what it needs.We are using a Dirichlet boundary (type = 2) because absorbing boundary are not suitable for tsunami wave forcing.


## Gauge Comparison 



All the available engines in BG_Flood perform really well. Note that the second peak for gauge 2 is nicely captured (not all model reproduce this one). Engine 5 produce more oscillations. See in-depth for discussion on that. 

## Reference
Liu, P.L.-F., H. Yeh, and C. Synolakis (2008): Advanced Numerical Models for Simulating Tsunami Waves and Runup. Advances in Coastal and Ocean Engineering, 10, 250 pp.

[PMEL webpage](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html)