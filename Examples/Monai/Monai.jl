### A Pluto.jl notebook ###
# v0.12.12

using Markdown
using InteractiveUtils

# ╔═╡ 9f6a4682-dd4b-11eb-29b9-0311850f7de6
using GMT

# ╔═╡ 8fd3f310-dd4b-11eb-13f2-634f6f98c198


# ╔═╡ 949761a0-dd16-11eb-32de-4b235b067893
md"""
# Monai Valley Example
This example replicates the "Tsunami runup onto a complex three-dimensional beach; Monai Valley" benchmerk from NOAA.

## Some background
This lab benchmark is an experimental setup to replicate the extreme runup observed during the Hokkaido-Nansei-Oki tsunami of 1993 that struck Okushiri Island, Japan. The extreme runup was located on a small cove (Monai Cove) surrounded by a complex bathymetry.

Details of the experiements and the benchmark data are available at the  [NOAA webpage](https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html) and are referenced in [Synolakis et al. 2007](https://nctr.pmel.noaa.gov/benchmark/SP_3053.pdf) 

## To run this notebook
**needs GMT**

"""


# ╔═╡ a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
md"""
## Bathymetry
We alreaddy prepared the bathymetry from the original file. Note that is it slighly different because we need to force BG_flood to stick with the lab extent even if the lab extent will be smaller than the number of block needed.
"""

# ╔═╡ f99d1880-dd4b-11eb-3def-57d4ec93c373
grdimage("Bathy_D.nc", cmap=:geo; shade=true, show=true)

# ╔═╡ Cell order:
# ╠═8fd3f310-dd4b-11eb-13f2-634f6f98c198
# ╟─949761a0-dd16-11eb-32de-4b235b067893
# ╠═9f6a4682-dd4b-11eb-29b9-0311850f7de6
# ╟─a81a6580-dd4b-11eb-2c2c-1f7e0e684d59
# ╠═f99d1880-dd4b-11eb-3def-57d4ec93c373
