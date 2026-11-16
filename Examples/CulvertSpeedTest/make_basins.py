#!/usr/bin/env python3
"""
Culvert speed-up benchmark for BG_Flood.

Builds a synthetic topography of NBX x NBY square basins separated by walls and
connected in a "snake" chain by culverts, plus the matching BG_param.txt.

  - Basin k (snake order) has its floor at -k*dz, so water cascades down the chain.
  - Culvert k (k = 0 .. N-2) is a 2-way culvert crossing the wall from basin k to basin k+1.
  - Culvert N-1 is a pump from the last basin back to basin 0, closing the loop
    (so there are exactly N culverts for N basins and the system keeps running).
  - A river injects a constant discharge into basin 0.

Default: 8 x 8 = 64 basins and 64 culverts. Scale up either with more basins
(--nbx 32 --nby 32 gives 1024 culverts on a bigger DEM) or with more culverts per
wall on the same DEM (--per-wall 16 gives 63*16 culverts + 16 return pumps = 1024).

With --share-cells the outlet of culvert k is placed on the same cell as the inlet
of culvert k+1. That forces the GPU injection to use 2 sequential passes
(race-condition handling) instead of 1.

Requires numpy and netCDF4.
"""
import argparse
import numpy as np
from netCDF4 import Dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nbx", type=int, default=8, help="number of basins in x")
    p.add_argument("--nby", type=int, default=8, help="number of basins in y")
    p.add_argument("--L", type=int, default=40, help="basin interior size [cells]")
    p.add_argument("--W", type=int, default=4, help="wall thickness [cells]")
    p.add_argument("--dx", type=float, default=1.0, help="grid resolution [m]")
    p.add_argument("--zwall", type=float, default=10.0, help="wall crest elevation [m]")
    p.add_argument("--dz", type=float, default=0.05, help="floor drop between consecutive basins [m]")
    p.add_argument("--Q", type=float, default=2.0, help="river inflow into basin 0 [m3/s]")
    p.add_argument("--width", type=float, default=0.8, help="culvert width/diameter [m]")
    p.add_argument("--endtime", type=float, default=3600.0, help="simulation end time [s]")
    p.add_argument("--gpu", type=int, default=0, help="GPU device (-1 for CPU)")
    p.add_argument("--per-wall", type=int, default=1, help="parallel culverts across each wall (and number of return pumps)")
    p.add_argument("--param", default=None, help="name of the param file (default <prefix>BG_param.txt)")
    p.add_argument("--share-cells", action="store_true", help="outlet of culvert k == inlet of culvert k+1")
    p.add_argument("--prefix", default="", help="prefix for output file names")
    a = p.parse_args()

    nbx, nby, L, W, dx = a.nbx, a.nby, a.L, a.W, a.dx
    nb = nbx * nby
    nx = nbx * (L + W) + W
    ny = nby * (L + W) + W

    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    z = np.full((ny, nx), a.zwall, dtype=np.float32)

    # snake ordering of the basins
    order = []  # list of (row, col) in snake order
    for r in range(nby):
        cols = range(nbx) if r % 2 == 0 else range(nbx - 1, -1, -1)
        for c in cols:
            order.append((r, c))

    def basin_box(r, c):
        """index range of basin interior (i0:i1, j0:j1)"""
        i0 = W + c * (L + W)
        j0 = W + r * (L + W)
        return i0, i0 + L, j0, j0 + L

    for k, (r, c) in enumerate(order):
        i0, i1, j0, j1 = basin_box(r, c)
        z[j0:j1, i0:i1] = -k * a.dz

    def cell_xy(i, j):
        return x[i], y[j]

    off = 2  # distance of inlet/outlet cell from the wall [cells]
    mid = L // 2

    npw = a.per_wall
    if npw < 1 or npw > L:
        raise SystemExit("--per-wall must be between 1 and L")
    # positions along the wall of the parallel culverts (npw = 1 -> middle of the wall)
    along = [int((m + 0.5) * L / npw) for m in range(npw)]

    def port(r, c, towards, s):
        """cell next to the wall of basin (r,c) facing direction 'towards' (dr,dc), at position s along the wall"""
        i0, i1, j0, j1 = basin_box(r, c)
        dr, dc = towards
        if dc == 1:
            return i1 - 1 - off, j0 + s
        if dc == -1:
            return i0 + off, j0 + s
        if dr == 1:
            return i0 + s, j1 - 1 - off
        return i0 + s, j0 + off

    chain = []  # chain[k][m]: m-th parallel culvert from basin k to basin k+1
    for k in range(nb - 1):
        (r1, c1), (r2, c2) = order[k], order[k + 1]
        d = (r2 - r1, c2 - c1)
        row = []
        for s in along:
            ia, ja = port(r1, c1, d, s)
            ib, jb = port(r2, c2, (-d[0], -d[1]), s)
            row.append([2, cell_xy(ia, ja), cell_xy(ib, jb)])
        chain.append(row)

    if a.share_cells:
        # move the outlet of culvert (k, m) onto the inlet cell of culvert (k+1, m)
        for k in range(nb - 2):
            for m in range(npw):
                chain[k][m][2] = chain[k + 1][m][1]

    culverts = [tuple(c) for row in chain for c in row]

    # closing pumps: last basin -> basin 0 (around the centre of the basins)
    (rl, cl), (r0, c0) = order[-1], order[0]
    il0, _, jl0, _ = basin_box(rl, cl)
    ib0, _, jb0, _ = basin_box(r0, c0)
    if npw == 1:
        pumps = [((mid + 5, mid + 5), (mid, mid))]
    else:
        ng = int(np.ceil(np.sqrt(npw)))
        grid = [(mid + 3 * (gi - ng // 2), mid + 3 * (gj - ng // 2)) for gj in range(ng) for gi in range(ng)][:npw]
        pumps = [(g, g) for g in grid]
    for (pi, pj), (qi, qj) in pumps:
        culverts.append((0, cell_xy(il0 + pi, jl0 + pj), cell_xy(ib0 + qi, jb0 + qj)))

    # river area in basin 0 (5 x 5 m patch away from the culvert ports)
    i0, _, j0, _ = basin_box(r0, c0)
    rx0, ry0 = x[i0 + 5] - 0.5 * dx, y[j0 + 5] - 0.5 * dx
    rx1, ry1 = rx0 + 5 * dx, ry0 + 5 * dx

    pre = a.prefix
    bathyfile = f"{pre}basins.nc"
    riverfile = f"{pre}river_inflow.txt"
    paramfile = a.param if a.param else f"{pre}BG_param.txt"

    with Dataset(bathyfile, "w") as nc:
        nc.createDimension("x", nx)
        nc.createDimension("y", ny)
        vx = nc.createVariable("x", "f8", ("x",))
        vy = nc.createVariable("y", "f8", ("y",))
        vz = nc.createVariable("z", "f4", ("y", "x"))
        vx[:] = x
        vy[:] = y
        vz[:] = z
        vz.units = "m"
        nc.title = f"{nb} basins connected by {len(culverts)} culverts (BG_Flood culvert benchmark)"

    with open(riverfile, "w") as f:
        # BG_Flood needs the river time series to extend past the end time
        f.write(f"0.0 {a.Q}\n{10 * a.endtime + 1e6:.1f} {a.Q}\n")

    with open(paramfile, "w") as f:
        f.write(f"# Culvert benchmark: {nbx}x{nby} = {nb} basins, {len(culverts)} culverts ({npw} per wall)"
                f"{' (shared cells -> 2 injection passes)' if a.share_cells else ''}\n")
        f.write(f"# Generated by make_basins.py; domain {nx} x {ny} cells at dx = {dx} m\n\n")
        f.write(f"bathy = {bathyfile}?z;\n")
        f.write(f"dx = {dx};\n")
        f.write(f"gpu = {a.gpu};\n")
        f.write("doubleprecision = 0;\n")
        f.write("zsinit = -10.0;  # all basins start dry\n")
        f.write("cf = 0.01;\n\n")
        f.write(f"endtime = {a.endtime};\n")
        f.write(f"outputtimestep = {a.endtime};  # keep output light for timing\n")
        f.write("outvars = zs,h,u,v,zb;\n")
        f.write(f"outfile = {pre}culvert_benchmark.nc;\n\n")
        f.write(f"river = {riverfile},{rx0},{rx1},{ry0},{ry1};\n\n")
        f.write("# culvert = type,x1,y1,x2,y2,width|Qmax\n")
        for t, (x1, y1), (x2, y2) in culverts:
            w = a.Q / npw if t == 0 else a.width
            f.write(f"culvert = {t},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{w};\n")

    print(f"{bathyfile}: {nx} x {ny} cells, {nb} basins")
    print(f"{paramfile}: {len(culverts)} culverts")


if __name__ == "__main__":
    main()
