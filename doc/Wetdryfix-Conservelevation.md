
# Conserve Elevation 

At the interface of coarse and fine blocks we often cannot **strictly** conserve both mass (h) and water elevation. The difference in most cases is relatively small but near the wet/dry interface, trying to conserve mass leads to instabilities. Since these instabilities are a bit annoying by default BG_Flood enforces the conservation of elevation at the interface between coarse and fine blocks where wet and dry cells are present. This might lead to a mismatch between the expected volume and actual volume when simulating rain or river flooding. Similarly, ignoring the conserve elevation requirement when simulating a tsunami can lead to a underestimate of the tsunami. The impact of these switch is small and we can show how small they are below.

Below are two examples of basic flood model to test the impact of the wet/dry instability fix (hereafter wetdryfix) on mass conservation. Further below is to test the impact of the wetdryfix and conserve elevation routimne (hereafter conserveElevation) on a tsunami wave propagating on a real steep bathymetry (Samoa).



## Mass conservation implication
Using the Waikanae topo with cst rain at 50mm/h for 1 hour. using the rainbnd option and wall bnd on all side.
Results are undestinguishable from each other and after 1 hr the model has **100.48%** of the theoretical volume of water in both with and without the wetdryfix.  

For comparison the current **Dev** branch has 100.19% of the theoretical volume. While this is somehow better it does not really undermine the new branch (i.e. the dev branch might get closer to the theory for the wrong raisons), the bugfixes and instability improvement of this branch are totally justified. 

results
|   | % of theoretical volume | runtime |
| ------------- | ------------- | ------------- |
| Wetdry fix  | 100.48%  | 97s  |
| **NO** Wetdryfix  | 100.48%  | 96s  |


## Tsumani test results

The impact of the wet/dry fix should be much more obvious in tsunami simulation but here conserve elevation should be used in most tsunami cases.

### No wetdryfix
switching off the wetdryfix causes instability that really have an impact on tsunami wave. Here the instability are clearly visible in the map and in the comparison are seen as 0.01 m waves before the tsunami wave arrive.

![image](https://user-images.githubusercontent.com/3713631/213352941-31b5ff25-ca1e-4b80-b0d4-cba7d1aa5496.png)

### Wetdryfix
When using the wetdryfix the instabilities disappear and the solution is smoother.

![image](https://user-images.githubusercontent.com/3713631/213353179-2d7c133b-057d-4fe9-9abc-d0a02fb64dcb.png)

### Comparison Wetdryfix vs no Wetdryfix
The instabilities are more obvious when we look at a transect diagonally from the plot above. These instabilities have a notable affect the height of the tsunami wave... a lot! This justifies that we should have the wetdryfix switched on by default.

![image](https://user-images.githubusercontent.com/3713631/213353988-e17c1dac-4f68-485b-a9c3-aa8f3baee954.png)


### Conserve Elevation
When running tsunami/storm surge simulation without rivers or rain then the conserve elevation should be switched on. This doesn't make a considerable impact on the tsunami wave (see transect above the red and blue line are indistinguishable) I'm not sure why the difference is so small but there is a difference. Below is the difference between the wetdryfix and conservelevation routine. it produces a difference of O(10-5).

 
![image](https://user-images.githubusercontent.com/3713631/213354080-d71a6e49-7085-41ed-93d6-2afb724cac10.png)



