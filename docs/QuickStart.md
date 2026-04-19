# Quick Start

This page will get you running your first BG_Flood simulation in minutes.

## Prerequisites

- BG_Flood executable: either download the latest [release](https://github.com/CyprienBosserelle/BG_Flood/releases/latest) (Windows) or [build from source](Install.md) (Linux)
- A bathymetry/topography file (`.asc` or `.nc` format)

!!! tip "No data yet?"
    You can use the example files bundled in the [Examples/Jet](https://github.com/CyprienBosserelle/BG_Flood/tree/master/Examples/Jet) folder of the repository.

## Step 1 — Create a parameter file

Create a text file called `BG_param.txt` in the same directory as the BG_Flood executable (or wherever you want to run). This file tells BG_Flood everything it needs to know.

Here is a minimal working example using the Jet example bathymetry:

``` txt title="BG_param.txt"
# Minimal BG_Flood parameter file

# Bathymetry file (the only required parameter)
bathy = Slit_Bathy.asc

# Simulation end time in seconds
endtime = 600.0

# Output a snapshot every 10 seconds
outputtimestep = 10.0

# Output file name
outfile = MyFirstRun.nc

# Variables to save
outvars = zs,h,u,v,zb

# Save as float (easier to inspect; default is short integer)
smallnc = 0

# Boundary conditions
# 0: Wall, 1: Neumann (default), 2: Dirichlet, 3: Absorbing
left = 3
leftbndfile = Leftbnd.txt
right = 3
rightbndfile = Rightbnd.txt
```

The boundary files are simple two-column text files (time in seconds, water level in metres):

``` txt title="Leftbnd.txt"
0.0  1.0
3600.0  1.0
```

``` txt title="Rightbnd.txt"
0.0  0.0
3600.0  0.0
```

## Step 2 — Run BG_Flood

=== "Windows"

    Open a terminal in the folder containing `BG_param.txt` and run:
    ```
    BG_Flood.exe
    ```
    Or simply double-click the executable if `BG_param.txt` is in the same folder.

=== "Linux"

    ```bash
    ./BG_Flood
    ```

You can also specify a custom parameter file name:
```bash
./BG_Flood my_custom_params.txt
```

BG_Flood will print progress to the terminal and create a log file (`BG_log.txt`).

## Step 3 — Inspect the results

The output is a NetCDF file (`MyFirstRun.nc`). You can open it with:

- **Python**: `xarray`, `netCDF4`, or `matplotlib`
- **QGIS**: drag-and-drop the `.nc` file
- **Panoply** or **ncview**: lightweight NetCDF viewers

A quick Python example:

```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("MyFirstRun.nc")

# Plot the last timestep of water surface elevation
ds["zs"].isel(time=-1).plot()
plt.title("Water surface at final timestep")
plt.show()
```

## What's in the output?

The default output variables are:

| Variable | Description | Unit |
|----------|-------------|------|
| `zb` | Bottom/ground elevation | m |
| `zs` | Water surface elevation | m |
| `h`  | Water depth | m |
| `u`  | Velocity in x-direction | m/s |
| `v`  | Velocity in y-direction | m/s |

See the full [Parameters list](ParametersList-py.md) for all available output variables and options.

## Next steps

- Read the [Manual](Manual.md) for a detailed description of all features
- Follow the [River flooding tutorial](tutorial/TutorialRiver.md) for a realistic setup with tides, rivers, and rain
- Try the [Jet tutorial](tutorial/TutorialJetJulia.md) or [Gaussian wave tutorial](tutorial/Gaussian_Wave_Julia.md) for benchmark examples
- Browse the [Parameters list](ParametersList-py.md) for all available keywords

!!! warning "Unrecognized parameters"
    If you misspell a keyword in `BG_param.txt`, BG_Flood will now print a warning: `WARNING: Unrecognized parameter keyword "..."`. Check the spelling against the [Parameters list](ParametersList-py.md).
