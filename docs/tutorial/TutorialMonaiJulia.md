Monai tutorial with Julia

This is a tutorial realised for BG_Flood on Julia.

--8<-- "Monai.jl.html"
<!--<iframe src="Monai.jl.html" width="100%" height="600px"></iframe>-->


### Goals

- Check Boundary forcing
- Check wave propagation and runup on complex topography against experimental data

### Status
Success
### Settings

### Results
Model behaves similar to other codes and ca reproduce the bulk of the tsunami waves. Results are comparable to the same class of models. The double precision simulation is virtually identical to the Double precision run confirming that in this case there is not much point on using the Double precision.

skill assessment:
**Single precision**

| ------------- | RMS error (m) | Will-corr | BSS | 
| ------------- |:-------------:|  --------:| -----:|
| Gauge 1 | 0.003391 | 0.978040 | 0.910979 |
| Gauge 2 | 0.003735 | 0.965899 | 0.870943 |
| Gauge 3 | 0.003284 | 0.976699 | 0.884080 |

**Double precision**

| ------------- | RMS error (m) | Will-corr | BSS | 
| ------------- |:-------------:|  --------:| -----:|
| Gauge 1 | 0.003391 | 0.978039 | 0.910981 |
| Gauge 2 | 0.003733 | 0.965948 | 0.871093 |
| Gauge 3 | 0.003283 | 0.976704 | 0.884097 |


![](https://github.com/CyprienBosserelle/Basil_Cart_StV/blob/master/Examples/Monai/Results/newbathy3_Gauges.png)
**Figure 1.** Measured and simulated timeseries for the Monai benchmark


### Run times
| GPU | Quadro K620 | GeForce GTS 450 |other GPU |
| ------------- |:-------------:| -----:| -----:|
| **Single Precision (s)** | 27 | 20 | XX |
| **Double Precision (s)** | XX | 34 | XX |





