# Culvert speed-up benchmark

Synthetic test: 8 x 8 = 64 basins (40 m square, 4 m walls, 1 m grid, 356 x 356 cells) connected in a snake chain by 63 two-way culverts. A 64th culvert, a pump, links the last basin back to basin 0 and closes the loop. Each basin floor is 5 cm lower than the one before it (0 to -3.15 m). A 2 m3/s river feeds basin 0, so water cascades through every culvert.

Files:
- `basins.nc`: topography (`z`, with `x`/`y` at cell centres)
- `BG_param.txt`: matching parameter file (64 `culvert =` lines, 1 river)
- `river_inflow.txt`: constant inflow
- `make_basins.py`: generator (needs numpy and netCDF4)

## Regenerate / scale up

```bash
python make_basins.py                       # default 8x8 = 64 culverts
python make_basins.py --nbx 32 --nby 32     # 1024 culverts (1412 x 1412 cells)
python make_basins.py --share-cells         # outlet of culvert k = inlet of culvert k+1 -> 2 GPU injection passes
python make_basins.py --prefix big_ --nbx 16 --nby 16   # write files with a prefix
```

## Timing old vs new

Run the same `BG_param.txt` with a build of each commit and compare `Model runtime` in `BG_log.txt`:

- old (one kernel launch per culvert): `e513787`
- new (batched culverts): `FasterCulverts` HEAD

To isolate the cost of the culverts, run once more with the `culvert =` lines removed. That run gives the baseline cost of the river and the flow solver.
