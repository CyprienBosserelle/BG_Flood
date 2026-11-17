#!/usr/bin/env python3
"""
Set-up for the tsunami boundary-reflection benchmark.

A Gaussian hump of water is released (at rest) in the middle of a flat-bottom,
frictionless rectangular channel. It splits into two identical pulses of half
the initial amplitude that travel left and right. Each pulse travels
`NWAVE` wavelengths before hitting the open boundary at the end of the channel.

Two kind of channels are generated:

* the "short" channel, whose left/right boundaries are the ones being tested
  (one run per boundary type),
* the "long" benchmark channel, identical but `LONGFACTOR` times longer, so
  that whatever reflects off its end arrives at the gauges long after the end
  of the simulation. Within the simulated time this is a reflection-free
  solution, carrying the same numerical diffusion/dispersion as the tests.

The reflected signal of a test run is then simply  eta_test - eta_benchmark.

Everything is written in ./runs/<scenario>/ (bathymetry, hot-start file,
BG_param.txt). Run with ./run_all.sh then analyse with analyse.py.

Requires: numpy, netCDF4
"""
import json
import os

import numpy as np
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Physical/numerical set-up (SI units)
# ---------------------------------------------------------------------------
G = 9.81
H0 = 100.0          # still water depth [m]
AMP = 0.5           # initial Gaussian amplitude [m] (each pulse is AMP/2); AMP/H0 << 1 -> linear
SIGMA = 1000.0      # Gaussian standard deviation [m]
WAVELENGTH = 6.0 * SIGMA   # "wavelength" of the pulse: +/-3 sigma, i.e. its full extent [m]
NWAVE = 10          # number of wavelengths travelled before meeting the boundary
DX = 100.0          # grid resolution [m] (10 cells per sigma, 60 per wavelength)
BLKW = 16           # BG_Flood block width
NBLKY = 1           # domain width in blocks
NLAND = 1           # rows of dry land along the top and bottom of the domain (lateral walls)
ZLAND = 10.0        # elevation of the land rows [m]
LONGFACTOR = 2      # benchmark channel is LONGFACTOR times longer: its own end reflection
                    # only re-enters the short-channel extent after (20*LONGFACTOR-10) wavelengths
                    # of travel time, i.e. after the end of the run

C0 = np.sqrt(G * H0)            # long-wave celerity
TWAVE = WAVELENGTH / C0         # time for the pulse to travel one wavelength
HALFLEN = NWAVE * WAVELENGTH    # distance from the centre to the tested boundary

# Run until the reflection from the tested boundary is back past the centre:
# pulse front reaches the boundary at ~NWAVE*TWAVE, reflection back at the centre
# at ~2*NWAVE*TWAVE; add 2 wavelengths margin.
ENDTIME = (2 * NWAVE + 2) * TWAVE
OUTSTEP = TWAVE / 4.0

# Gauges (x position, in wavelengths from the centre). y is the channel centre-line.
GAUGES = {
    "G00": 0.0,
    "GR5": 5.0, "GL5": -5.0,     # halfway: incident and reflected pulse well separated
    "GR9": 9.0, "GL9": -9.0,     # one wavelength from the boundary
}

# Scenarios: name -> (channel, boundary type at left/right, extra parameters)
#   BG_Flood bnd types: 0 wall, 1 Neumann, 2 Dirichlet (zs = zsinit), 3 absorbing 1D
SCENARIOS = {
    "benchmark_long": dict(channel="long", bnd=3, extra={}),
    "wall": dict(channel="short", bnd=0, extra={}),
    "neumann": dict(channel="short", bnd=1, extra={}),
    "dirichlet": dict(channel="short", bnd=2, extra={}),
    "absorbing": dict(channel="short", bnd=3, extra={}),
    # sensitivity of the absorbing bnd to its filtering time (default 60 s)
    "absorbing_filt10": dict(channel="short", bnd=3, extra={"bndfiltertime": 10.0}),
    "absorbing_filt600": dict(channel="short", bnd=3, extra={"bndfiltertime": 600.0}),
}

# BG_Flood options common to all runs (edit to test other engines/settings)
COMMON = {
    "gpudevice": int(os.environ.get("BG_GPU", 0)),  # -1 for CPU (export BG_GPU=-1)
    "doubleprecision": 1,
    "engine": 1,
    "spherical": 0,
    "cf": 0.0,
    "frictionmodel": 0,
    "CFL": 0.5,
    "zsinit": 0.0,           # also the target level of Dirichlet / absorbing bnd
}


def write_grid(fname, x, y, varname, data):
    with Dataset(fname, "w") as nc:
        nc.createDimension("x", len(x))
        nc.createDimension("y", len(y))
        vx = nc.createVariable("x", "f8", ("x",))
        vy = nc.createVariable("y", "f8", ("y",))
        vz = nc.createVariable(varname, "f8", ("y", "x"))
        vx[:] = x
        vy[:] = y
        vz[:] = data


def make_channel(path, halflen):
    nx = int(round(2 * halflen / DX))
    assert nx % BLKW == 0, "channel length must be a multiple of the block width"
    ny = NBLKY * BLKW
    # cell centres
    x = -halflen + DX * (np.arange(nx) + 0.5)
    y = DX * (np.arange(ny) + 0.5)
    X, _ = np.meshgrid(x, y)
    # The channel is bounded laterally by dry land rather than by top/bot boundary
    # conditions: BG_Flood assigns each boundary *block* to a single boundary
    # segment (the last one listed that touches it), so in a channel only one
    # block wide, a "bot"/"top" segment would hijack the end blocks and their
    # left/right faces. With dry land rows the top/bot faces are inert and only
    # the left/right boundaries being tested act on the flow.
    zb = -H0 * np.ones_like(X)
    zb[:NLAND, :] = ZLAND
    zb[-NLAND:, :] = ZLAND
    write_grid(os.path.join(path, "bathy.nc"), x, y, "z", zb)
    write_grid(os.path.join(path, "zsinit.nc"), x, y, "zs",
               AMP * np.exp(-X**2 / (2.0 * SIGMA**2)))
    return x, y


def write_param(path, sc, y):
    p = dict(COMMON)
    p.update(sc["extra"])
    ymid = y[len(y) // 2]  # a cell centre on the channel centre-line
    lines = ["# BG_Flood tsunami boundary reflection test (generated by make_case.py)",
             "bathy = bathy.nc?z;",
             "hotstartfile = zsinit.nc;"]
    lines += [f"{k} = {v};" for k, v in p.items()]
    lines += [f"endtime = {ENDTIME:.3f};",
              f"outputtimestep = {OUTSTEP:.3f};",
              "outvars = zs,h,u,v;",
              "outfile = output.nc;",
              "smallnc = 0;",
              f"left = {sc['bnd']};",
              f"right = {sc['bnd']};"]
    for g, xg in GAUGES.items():
        lines.append(f"TSOutput = {g}.txt,{xg * WAVELENGTH:.3f},{ymid:.3f};")
    with open(os.path.join(path, "BG_param.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    for name, sc in SCENARIOS.items():
        path = os.path.join(here, "runs", name)
        os.makedirs(path, exist_ok=True)
        halflen = HALFLEN * (LONGFACTOR if sc["channel"] == "long" else 1)
        _, y = make_channel(path, halflen)
        write_param(path, sc, y)
        print(f"{name:20s} channel={sc['channel']:5s} L={2*halflen/1e3:6.0f} km  bnd={sc['bnd']}")

    meta = dict(g=G, h0=H0, amp=AMP, sigma=SIGMA, wavelength=WAVELENGTH, nwave=NWAVE,
                dx=DX, c0=C0, twave=TWAVE, halflen=HALFLEN, longfactor=LONGFACTOR,
                endtime=ENDTIME, gauges={g: xg * WAVELENGTH for g, xg in GAUGES.items()},
                scenarios={k: dict(channel=v["channel"], bnd=v["bnd"], extra=v["extra"])
                           for k, v in SCENARIOS.items()})
    with open(os.path.join(here, "runs", "setup.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"c0 = {C0:.2f} m/s, pulse travel time per wavelength = {TWAVE:.1f} s, endtime = {ENDTIME:.0f} s")


if __name__ == "__main__":
    main()
