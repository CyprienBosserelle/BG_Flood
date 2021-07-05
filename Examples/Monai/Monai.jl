### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 9f6a4682-dd4b-11eb-29b9-0311850f7de6
using GMT, DelimitedFiles, Printf

# ╔═╡ 8fd3f310-dd4b-11eb-13f2-634f6f98c198


# ╔═╡ 949761a0-dd16-11eb-32de-4b235b067893
md"""
# Monai Valley Example
This example replicates the "Tsunami runup onto a complex three-dimensional beach; Monai Valley" benchmerk from NOAA.

This notebook is presented to show how I use the model and also how the model performs against standart benchmarks.

## Some background
This lab benchmark is an experimental setup to replicate the extreme runup observed during the Hokkaido-Nansei-Oki tsunami of 1993 that struck Okushiri Island, Japan. The extreme runup was located on a small cove (Monai Cove) surrounded by a complex bathymetry.

Details of the experiements and the benchmark data are available at the  [NOAA webpage](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html) and are referenced in [Synolakis et al. 2007](https://nctr.pmel.noaa.gov/benchmark/SP_3053.pdf) 

## To run this notebook
This notebook requires **GMT 6.2** installed and the GMT package. Downloads is part of base in Julia version 1.6 but may need to be added. also please add Printf for writing neat looking txt files.

"""


# ╔═╡ a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
md"""
##  Experimental data
We download the bathymetry, input wave and gauge reading from the internet and use GMT to make a NetCDF grid.
"""

# ╔═╡ 92e31b39-e80c-4dbb-8daa-b624abcf18bb
md"""
### Bathymetry
"""

# ╔═╡ dda506d2-f026-44c5-aed1-47f7b174d533
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_Bathymetry.txt","bathy.xyz")

# ╔═╡ 1a056de8-5984-4e93-95f5-ef384b03a367
Bathy=xyz2grd("bathy.xyz",region=(0.0,5.488,0.0,3.402), inc=0.014)

# ╔═╡ 4ac6b170-adf8-46cb-8ece-6d7783229eb9
md"""
This bathy is _positive down_. This is not an issue for BG_Flood but we need to keep that in mind. Before ploting we create a custom colormap. In the plot we will also include the gauges locations.
"""

# ╔═╡ 79fc30ee-1de4-455b-ab4e-f99fe8899f25
cpt=makecpt(cmap=:geo, inverse=true, range=(-0.13,0.13));

# ╔═╡ f99d1880-dd4b-11eb-3def-57d4ec93c373
begin
	grdimage(Bathy, cmap=cpt);
	scatter!([4.521 1.196; 4.521 1.696; 4.521 2.196],mc=:black,show=true)
	
end

# ╔═╡ c1ef1d6f-367a-499a-81cf-5933af1202cf
md"""
#### Modifying the bathy
We need to modify slighly the bathymetry so that when BG_Flood creates the mesh, it stricly preserves the shape of the lab tank
##### Why do we need to do that?
BG\_Flood creates a mesh by assembling blocks of 16x16 cells together. When the input DEM/Bathy extent or a manualy specified extend does quite fit in the layout of blocks, BG\_Flood extends the last row/column to fill the space. In real world application tha's not a big deal because you can ask BG\_Flood to make a computational domain slightly smaller then your DEM but here in a small tank that will mean a bit more water. Although probably not a huge deal for teh experiment we need to get it right so we put _Physical_ walls on the top and bottom side.
"""

# ╔═╡ 5590a60d-54f2-4c7d-8bc6-e053c48420b7
BathyMod=xyz2grd("bathy.xyz",region=(0.0,5.488,-0.014,3.418), inc=0.014);

# ╔═╡ 6459595b-fa47-4e69-8c72-d38d612abafe
BathyModB=grdmath("? -0.125 DENAN =",BathyMod);

# ╔═╡ 53645d59-53c1-441f-b31c-15e841c77413
md"""
### Input wave
"""

# ╔═╡ acb05c81-db01-42b2-ae3f-48bdae935a36
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_InputWave.txt","InputWave.txt")

# ╔═╡ 94e8ff56-ac70-4b9c-9c49-bbec52f2fb33
md"""
The file has a non-standart hearder and non consistent delimiter so we better read it and rewrite it nicely so not to confuse BG_flood. Actually this file is not too bad and you could manually redo the first 2 lines and add a # to the front of the header line.
"""

# ╔═╡ 9476801d-8f0a-4f5c-afea-6d195da60e43

data=readdlm("InputWave.txt", '\t',  skipstart=3)


# ╔═╡ 75e1fa4e-a598-4aa4-a736-c3caab94a2b8
plot(data,show=true)

# ╔═╡ 42f760b3-93c5-4aa7-8855-991318dd0b2e
open("Input_Wave_clean.txt","w") do io
     for i=1:first(size(data))
         Printf.@printf(io,"%f\t%f\n",data[i,1],data[i,2]);

     end

end


# ╔═╡ 38788e11-be78-42bf-9b10-50ee8a0b4d8c
md"""
### Gauge data
"""

# ╔═╡ a05b1574-d37a-4635-9984-bf070e2f2b62
download("https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/MonaiValley_WaveGages.xls","WaveGages.xls")

# ╔═╡ c47e94f5-baa6-472c-8034-6da7b088289d


# ╔═╡ Cell order:
# ╠═8fd3f310-dd4b-11eb-13f2-634f6f98c198
# ╟─949761a0-dd16-11eb-32de-4b235b067893
# ╠═9f6a4682-dd4b-11eb-29b9-0311850f7de6
# ╟─a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
# ╠═92e31b39-e80c-4dbb-8daa-b624abcf18bb
# ╠═dda506d2-f026-44c5-aed1-47f7b174d533
# ╠═1a056de8-5984-4e93-95f5-ef384b03a367
# ╟─4ac6b170-adf8-46cb-8ece-6d7783229eb9
# ╠═79fc30ee-1de4-455b-ab4e-f99fe8899f25
# ╠═f99d1880-dd4b-11eb-3def-57d4ec93c373
# ╟─c1ef1d6f-367a-499a-81cf-5933af1202cf
# ╠═5590a60d-54f2-4c7d-8bc6-e053c48420b7
# ╠═6459595b-fa47-4e69-8c72-d38d612abafe
# ╟─53645d59-53c1-441f-b31c-15e841c77413
# ╠═acb05c81-db01-42b2-ae3f-48bdae935a36
# ╟─94e8ff56-ac70-4b9c-9c49-bbec52f2fb33
# ╠═9476801d-8f0a-4f5c-afea-6d195da60e43
# ╠═75e1fa4e-a598-4aa4-a736-c3caab94a2b8
# ╠═42f760b3-93c5-4aa7-8855-991318dd0b2e
# ╟─38788e11-be78-42bf-9b10-50ee8a0b4d8c
# ╠═a05b1574-d37a-4635-9984-bf070e2f2b62
# ╠═c47e94f5-baa6-472c-8034-6da7b088289d
