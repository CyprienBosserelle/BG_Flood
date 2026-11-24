#!/usr/bin/env python
"""
bg_html.py: build a self-contained HTML viewer for BG_Flood output
===================================================================

    python bg_html.py results.nc                       -> results.html
    python bg_html.py results.nc -o run12.html --vars hmax,zsmax,Umax --steps last
    python bg_html.py results.nc --region 1.75e6,1.76e6,5.42e6,5.43e6 --eps 0.01

The HTML needs no Python and no server: data, Leaflet and proj4 are all
inside the file. Open it in a browser, e-mail it, or put it on a web page.

How the data is stored
  BG_Flood writes one (time, y, x) array per refinement level, named
  <var>_P<n> / <var>_N<n>. Each level spans the whole domain, but only the
  16 x 16-cell blocks that are active at that level hold data. This script
  reads only the chunks that were actually written and keeps every active
  block at its own resolution. Blocks fully covered by finer blocks are
  dropped. Each field is then quantised to 16 bits and gzip-compressed.
  The page draws every block at its native cell size, reprojected to
  Web Mercator so it sits on a basemap.

Page features
  * variable drop-down, time slider with play / pause and speed
  * colour bars: Red -> Blue, Jet, seminf-haxby, GMT haxby (+ reverse),
    editable range
  * "Mesh blocks" tick box (outline of every active block)
  * dry masking: cells with water depth < eps (h, or hmax for *max
    variables) shown as missing; eps editable in the page
  * basemaps when the CRS is known: OpenStreetMap, LINZ aerial / topo (free
    LINZ key), Google satellite / roadmap / hybrid (Google Map Tiles API key)
  * hover readout, click a point for its time series, mesh summary, KPIs

CRS: read from the file (crs_wkt / spatial_ref), or --crs; EPSG:2193 if absent.

Requirements: numpy, h5py (or netCDF4), pyproj
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import gzip
import io
import json
import os
import re
import sys
import tarfile
import urllib.request

import numpy as np
import warnings
warnings.filterwarnings("ignore", message="You will likely lose important projection")

BW = 16                      # BG_Flood block width (cells)
DEFAULT_EPSG = 2193
PREFERRED_ORDER = ["hmax", "zsmax", "hUmax", "Umax", "h", "zs", "u", "v", "hU", "zb"]
NO_MASK = {"zb"}
SIGNED = {"u", "v", "hu", "hv", "dhdt"}

########################################################################
# Small helpers


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return _to_str(v.reshape(-1)[0])
        return str(v.tolist())
    return str(v)


def _attr(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.size == 1:
        x = v.reshape(-1)[0]
        return x.item() if hasattr(x, "item") and v.dtype.kind in "iuf" else _to_str(x)
    return v


def level_suffix(level):
    return f"{'N' if level < 0 else 'P'}{abs(level)}"


_LEVEL_RE = re.compile(r"^(?P<var>.+)_(?P<s>[PN])(?P<n>\d+)$")


def _parse_since(units):
    m = re.search(r"since\s+(.+)$", units or "")
    if not m:
        return None
    s = m.group(1).strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(s[:19], fmt)
        except ValueError:
            pass
    return None


########################################################################
# File access


class H5Backend:
    def __init__(self, path):
        import h5py
        self.f = h5py.File(path, "r", rdcc_nbytes=256 * 2**20, rdcc_nslots=10007)

    def names(self):
        return [k for k, v in self.f.items() if hasattr(v, "shape")]

    def shape(self, n):
        return self.f[n].shape

    def read1d(self, n):
        return np.asarray(self.f[n][...])

    def read(self, n, t, js, is_):
        return self.f[n][t, js, is_]

    def var_attrs(self, n):
        if n not in self.f:
            return {}
        return {k: _attr(v) for k, v in self.f[n].attrs.items()
                if k not in ("DIMENSION_LIST", "REFERENCE_LIST", "CLASS", "NAME")}

    def global_attrs(self):
        return {k: _attr(v) for k, v in self.f.attrs.items()}

    def written_chunks(self, n):
        """List of (t0, t1, y0, y1, x0, x1) for chunks that hold data, or None if unknown."""
        ds = self.f[n]
        if ds.chunks is None:
            return None
        ct, cy, cx = ds.chunks
        out = []

        def add(off):
            out.append((off[0], off[0] + ct, off[1], off[1] + cy, off[2], off[2] + cx))
        try:
            ds.id.chunk_iter(lambda info: add(info.chunk_offset))
        except Exception:
            try:
                for k in range(ds.id.get_num_chunks()):
                    add(ds.id.get_chunk_info(k).chunk_offset)
            except Exception:
                return None
        return out


class NCBackend:
    def __init__(self, path):
        import netCDF4
        self.f = netCDF4.Dataset(path, "r")
        self.f.set_auto_maskandscale(False)

    def names(self):
        return list(self.f.variables)

    def shape(self, n):
        return self.f.variables[n].shape

    def read1d(self, n):
        return np.asarray(self.f.variables[n][:])

    def read(self, n, t, js, is_):
        return np.asarray(self.f.variables[n][t, js, is_])

    def var_attrs(self, n):
        if n not in self.f.variables:
            return {}
        v = self.f.variables[n]
        return {k: v.getncattr(k) for k in v.ncattrs()}

    def global_attrs(self):
        return {k: self.f.getncattr(k) for k in self.f.ncattrs()}

    def written_chunks(self, n):
        return None


def open_backend(path):
    try:
        return H5Backend(path)
    except Exception:
        return NCBackend(path)


class Level:
    def __init__(self, lev, xx, yy):
        self.lev = lev
        self.nx, self.ny = len(xx), len(yy)
        self.dx = float(xx[1] - xx[0]) if self.nx > 1 else 1.0
        self.dy = float(yy[1] - yy[0]) if self.ny > 1 else self.dx
        self.x0 = float(xx[0]) - 0.5 * self.dx      # outer edge
        self.y0 = float(yy[0]) - 0.5 * self.dy
        self.nbx = -(-self.nx // BW)
        self.nby = -(-self.ny // BW)


class BGSource:
    def __init__(self, path, crs_override=None):
        self.path = path
        self.b = open_backend(path)
        names = set(self.b.names())
        self.levels = {}
        for n in names:
            m = _LEVEL_RE.match(n)
            if m and m["var"] == "xx":
                lev = int(m["n"]) * (-1 if m["s"] == "N" else 1)
                if "yy_" + level_suffix(lev) in names:
                    self.levels[lev] = Level(lev, self.b.read1d(n), self.b.read1d("yy_" + level_suffix(lev)))
        if not self.levels:
            raise SystemExit("No xx_<level>/yy_<level> axes found: is this a BG_Flood output file?")
        vars_ = {}
        for n in names:
            m = _LEVEL_RE.match(n)
            if not m or m["var"] in ("xx", "yy"):
                continue
            lev = int(m["n"]) * (-1 if m["s"] == "N" else 1)
            if lev in self.levels and len(self.b.shape(n)) == 3:
                vars_.setdefault(m["var"], []).append(lev)
        self.vars = {k: sorted(v) for k, v in vars_.items()}
        if not self.vars:
            raise SystemExit("No <var>_<level> data variables found.")
        self.levlist = sorted(self.levels)

        if "time" in names:
            self.time = np.atleast_1d(self.b.read1d("time")).astype(float)
            self.time_units = _to_str(self.b.var_attrs("time").get("units", ""))
        else:
            self.time = np.array([0.0])
            self.time_units = ""
        # number of steps actually present in the data arrays
        any_var = self.varname(next(iter(self.vars)), self.vars[next(iter(self.vars))][0])
        self.nt = min(len(self.time), self.b.shape(any_var)[0])
        self._attrs, self._chunks = {}, {}
        self.crs, self.crs_desc, self.crs_source = self._read_crs(crs_override)

    # ---- metadata
    def varname(self, v, lev):
        return f"{v}_{level_suffix(lev)}"

    def attrs(self, n):
        if n not in self._attrs:
            self._attrs[n] = self.b.var_attrs(n)
        return self._attrs[n]

    def var_info(self, v):
        a = self.attrs(self.varname(v, self.vars[v][0]))
        return _to_str(a.get("long_name", v)), _to_str(a.get("units", ""))

    def _read_crs(self, override):
        from pyproj import CRS
        if override:
            crs = CRS.from_user_input(override)
            return crs, _crs_name(crs), "given on the command line"
        wkt = None
        for vn in ("crs", "spatial_ref", "projection"):
            a = self.b.var_attrs(vn)
            wkt = a.get("crs_wkt") or a.get("spatial_ref")
            if wkt:
                break
        if not wkt:
            g = self.b.global_attrs()
            wkt = g.get("crs_wkt") or g.get("spatial_ref") or g.get("crs")
        if wkt:
            try:
                crs = CRS.from_user_input(_to_str(wkt))
                return crs, _crs_name(crs), "read from the file"
            except Exception:
                pass
        crs = CRS.from_epsg(DEFAULT_EPSG)
        return crs, _crs_name(crs), f"not in the file, EPSG:{DEFAULT_EPSG} assumed"

    def time_label(self, t):
        v = float(self.time[t])
        t0 = _parse_since(self.time_units)
        mult = 1
        u = self.time_units.lower()
        for key, s in (("sec", 1), ("min", 60), ("hour", 3600), ("day", 86400)):
            if u.startswith(key):
                mult = s
        if t0 is not None:
            return (t0 + _dt.timedelta(seconds=v * mult)).strftime("%Y-%m-%d %H:%M"), v * mult
        return f"{v:g} {self.time_units}".strip(), v * mult

    # ---- data
    def _unpack(self, name, raw):
        a = self.attrs(name)
        fill = a.get("_FillValue", a.get("missing_value", a.get("missingvalue")))
        z = raw.astype(np.float32, copy=True)
        bad = ~np.isfinite(z) | (np.abs(z) > 1e30)
        if fill is not None:
            bad |= raw == raw.dtype.type(np.asarray(fill).reshape(-1)[0])
        if a.get("scale_factor") is not None:
            z *= np.float32(a["scale_factor"])
        if a.get("add_offset") is not None:
            z += np.float32(a["add_offset"])
        z[bad] = np.nan
        return z

    def _windows(self, name, t, L, bbox):
        """Block-aligned (y0,y1,x0,x1) windows to read for level L at step t."""
        if name not in self._chunks:
            self._chunks[name] = self.b.written_chunks(name)
        ch = self._chunks[name]
        by0, by1, bx0, bx1 = bbox
        if ch is None:
            step = max(BW, (4_000_000 // max(L.nx, 1)) // BW * BW)
            rects = [(y, min(y + step, L.ny), bx0 * BW, min(bx1 * BW, L.nx))
                     for y in range(by0 * BW, min(by1 * BW, L.ny), step)]
        else:
            rects = [(y0, min(y1, L.ny), x0, min(x1, L.nx)) for (t0, t1, y0, y1, x0, x1) in ch if t0 <= t < t1]
        out = []
        for (y0, y1, x0, x1) in rects:
            a, b_ = max(y0 // BW, by0), min(-(-y1 // BW), by1)
            c, d = max(x0 // BW, bx0), min(-(-x1 // BW), bx1)
            if a < b_ and c < d:
                out.append((a, b_, c, d))
        return out

    def level_blocks(self, v, t, lev, bbox):
        """{(bj, bi): 16x16 float32 array} for every block of `v` at `lev` with any data."""
        L = self.levels[lev]
        name = self.varname(v, lev)
        res = {}
        for (a, b_, c, d) in self._windows(name, t, L, bbox):
            y0, y1 = a * BW, min(b_ * BW, L.ny)
            x0, x1 = c * BW, min(d * BW, L.nx)
            z = self._unpack(name, self.b.read(name, t, slice(y0, y1), slice(x0, x1)))
            ph, pw = (b_ - a) * BW - z.shape[0], (d - c) * BW - z.shape[1]
            if ph or pw:
                z = np.pad(z, ((0, ph), (0, pw)), constant_values=np.nan)
            zb = z.reshape(b_ - a, BW, d - c, BW).transpose(0, 2, 1, 3)
            good = np.isfinite(zb).any(axis=(2, 3))
            for j, i in zip(*np.nonzero(good)):
                k = (a + j, c + i)
                if k not in res:
                    res[k] = zb[j, i].copy()
        return res


def _crs_name(crs):
    epsg = crs.to_epsg(min_confidence=60)
    name = crs.name if crs.name and crs.name.lower() != "unknown" else None
    if not name:
        try:
            p = crs.to_dict()
            name = f"{p.get('proj', 'custom')} (lon_0={p.get('lon_0', '?')}, x_0={p.get('x_0', '?')}, y_0={p.get('y_0', '?')})"
        except Exception:
            name = "custom CRS"
    return name + (f" · EPSG:{epsg}" if epsg else "")


########################################################################
# Build the data package


def quantise(a):
    """float32 array -> (gz bytes of shuffled uint16, lo, step). q=0 means missing."""
    fin = np.isfinite(a)
    q = np.zeros(a.size, dtype=np.uint16)
    lo, st = 0.0, 1.0
    if fin.any():
        lo = float(a[fin].min())
        hi = float(a[fin].max())
        st = (hi - lo) / 65533.0 if hi > lo else 1.0
        q[fin.ravel()] = (np.round((a.ravel()[fin.ravel()] - lo) / st) + 1).astype(np.uint16)
    raw = q.view(np.uint8).reshape(-1, 2)
    shuffled = np.concatenate([raw[:, 0], raw[:, 1]]).tobytes()
    return gzip.compress(shuffled, 9), lo, st


def _avg_up(children):
    """{(bj,bi): 16x16} at one level -> {(pj,pi): 16x16} one level coarser (NaN-aware 2x2 mean).
    Only parents whose 4 children are all present are built."""
    groups = {}
    for (bj, bi), a in children.items():
        groups.setdefault((bj >> 1, bi >> 1), []).append((bj & 1, bi & 1, a))
    h = BW // 2
    out = {}
    for p, lst in groups.items():
        if len(lst) != 4:
            continue
        blk = np.empty((BW, BW), np.float32)
        for sj, si_, a in lst:
            r = a.reshape(h, 2, h, 2)
            fin = np.isfinite(r)
            n = fin.sum(axis=(1, 3))
            sm = np.where(fin, r, 0).sum(axis=(1, 3))
            blk[sj * h:(sj + 1) * h, si_ * h:(si_ + 1) * h] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        out[p] = blk
    return out


def build_package(src, var_list, steps, eps, region=None, max_level=None, verbose=True):
    levs = src.levlist
    # block index bounding box per level (region crop)
    bbox = {}
    for lev in levs:
        L = src.levels[lev]
        if region is None:
            bbox[lev] = (0, L.nby, 0, L.nbx)
        else:
            x0, x1, y0, y1 = region
            bw, bh = BW * L.dx, BW * L.dy
            bbox[lev] = (max(0, int((y0 - L.y0) // bh)), min(L.nby, int(-(-(y1 - L.y0) // bh))),
                         max(0, int((x0 - L.x0) // bw)), min(L.nbx, int(-(-(x1 - L.x0) // bw))))

    ref = next((v for v in ("h", "zs", "hmax", "zsmax") if v in src.vars), var_list[0])
    ref_levs = src.vars[ref]
    Lm = ref_levs[-1] if max_level is None else min(max(max_level, ref_levs[0]), ref_levs[-1])
    blobs, meshes, mesh_index, step_mesh, fields = {}, [], {}, [], {v: [] for v in var_list}
    samples = {v: [] for v in var_list}
    depth_of = {}
    for v in var_list:
        if v in NO_MASK:
            depth_of[v] = None
        else:
            pref = ["hmax", "h"] if v.lower().endswith("max") else ["h", "hmax"]
            depth_of[v] = next((p for p in pref if p in src.vars), None)
    # depth variables first, so their arrays are ready for masking the range samples
    order = [v for v in ("h", "hmax") if v in var_list] + [v for v in var_list if v not in ("h", "hmax")]
    NANB = np.full((BW, BW), np.nan, np.float32)

    for si, t in enumerate(steps):
        if verbose:
            print(f"  step {t + 1}/{src.nt} ({src.time_label(t)[0]})", flush=True)
        # --- mesh: active blocks of the reference variable; a block fully covered by finer blocks is dropped
        ref_blocks, act = {}, {}
        for lev in ref_levs:
            ref_blocks[lev] = src.level_blocks(ref, t, lev, bbox[lev])
            act[lev] = set(ref_blocks[lev])
        covered = {lev: set() for lev in levs}
        for lev in reversed(ref_levs[:-1]):
            child = act.get(lev + 1, set()) | covered.get(lev + 1, set())
            cnt = {}
            for (bj, bi) in child:
                p = (bj >> 1, bi >> 1)
                cnt[p] = cnt.get(p, 0) + 1
            covered[lev] = {p for p, n in cnt.items() if n == 4}
            act[lev] -= covered[lev]
        # with --max-level, everything finer than Lm is averaged into level-Lm blocks
        mesh_sets = {lev: set(act[lev]) for lev in ref_levs if lev < Lm}
        mesh_sets[Lm] = set(act[Lm]) | (covered[Lm] if Lm < ref_levs[-1] else set())
        keys = [(lev, bj, bi) for lev in sorted(mesh_sets) for (bj, bi) in sorted(mesh_sets[lev])]
        if not keys:
            step_mesh.append(None)
            for v in var_list:
                fields[v].append(None)
            continue
        lev_a = np.array([k[0] for k in keys], np.int8)
        bj_a = np.array([k[1] for k in keys], np.int32)
        bi_a = np.array([k[2] for k in keys], np.int32)
        sig = hash((lev_a.tobytes(), bj_a.tobytes(), bi_a.tobytes()))
        if sig not in mesh_index:
            mid = len(meshes)
            mesh_index[sig] = mid
            name = f"mesh{mid}"
            blobs[name] = gzip.compress(bi_a.tobytes() + bj_a.tobytes() + lev_a.tobytes(), 9)
            counts = {int(l): int((lev_a == l).sum()) for l in np.unique(lev_a)}
            meshes.append({"file": name, "nb": len(keys), "counts": counts})
        step_mesh.append(mesh_index[sig])

        # --- fields on this mesh, one variable at a time (keeps memory low)
        depth_arrays = {}
        for v in order:
            def read(lev):
                if lev not in src.vars[v]:
                    return {}
                return ref_blocks[lev] if v == ref else src.level_blocks(v, t, lev, bbox[lev])
            got = {}
            for lev in sorted(mesh_sets):
                lb = read(lev)
                for k in mesh_sets[lev]:
                    if k in lb:
                        got[(lev,) + k] = lb[k]
            if Lm < ref_levs[-1] and covered[Lm]:
                up = {}
                for lev in range(ref_levs[-1], Lm, -1):
                    lb = read(lev)
                    cur = {k: lb.get(k, NANB) for k in act[lev]}
                    cur.update(up)
                    up = _avg_up(cur)
                for k in covered[Lm]:
                    if k in up:
                        got[(Lm,) + k] = up[k]
            arr = np.full((len(keys), BW, BW), np.nan, np.float32)
            for i, k in enumerate(keys):
                b = got.get(k)
                if b is not None:
                    arr[i] = b
            del got
            gz, lo, st = quantise(arr)
            name = f"f_{v}_{si}"
            blobs[name] = gz
            fields[v].append({"file": name, "lo": lo, "st": st, "n": int(arr.size)})
            # sample for the default colour range (dry-masked)
            d = depth_of.get(v)
            sel = np.isfinite(arr)
            if d is not None and d in depth_arrays:
                sel &= depth_arrays[d] >= eps
            elif d == v:
                sel &= arr >= eps
            vals = arr[sel]
            if vals.size > 200_000:
                vals = vals[np.random.default_rng(0).choice(vals.size, 200_000, replace=False)]
            samples[v].append(vals)
            if v in ("h", "hmax"):
                depth_arrays[v] = arr
            del arr
        del ref_blocks, depth_arrays

    if all(m is None for m in step_mesh):
        raise SystemExit("No data found: every block of every level is empty for the selected steps "
                         "(the file may have been written before any output, or --region misses the domain).")
    return blobs, meshes, step_mesh, fields, samples, depth_of, Lm


########################################################################
# Vendor JS (inlined so the page works offline)

VENDOR = {
    "leaflet.js": ("leaflet", "1.9.4", "package/dist/leaflet.js"),
    "leaflet.css": ("leaflet", "1.9.4", "package/dist/leaflet.css"),
    "proj4.js": ("proj4", "2.11.0", "package/dist/proj4.js"),
}
CDN = {
    "leaflet.js": "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js",
    "leaflet.css": "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css",
    "proj4.js": "https://cdnjs.cloudflare.com/ajax/libs/proj4js/2.11.0/proj4.js",
}


def vendor_text(fname, vendor_dir=None):
    cache = os.path.join(os.path.expanduser("~"), ".cache", "bg_html")
    for d in filter(None, [vendor_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"), cache]):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    pkg, ver, member = VENDOR[fname]
    url = f"https://registry.npmjs.org/{pkg}/-/{pkg}-{ver}.tgz"
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            os.makedirs(cache, exist_ok=True)
            for f2, (p2, v2, m2) in VENDOR.items():
                if p2 == pkg:
                    txt = tf.extractfile(m2).read().decode("utf-8")
                    open(os.path.join(cache, f2), "w", encoding="utf-8").write(txt)
        return open(os.path.join(cache, fname), encoding="utf-8").read()
    except Exception as e:
        print(f"  warning: could not fetch {fname} ({e}); the page will load it from a CDN instead", file=sys.stderr)
        return None


########################################################################
# Main


def parse_steps(spec, nt):
    if spec in (None, "", "all"):
        return list(range(nt))
    if spec == "last":
        return [nt - 1]
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted({s - 1 for s in out if 1 <= s <= nt})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a self-contained HTML viewer for BG_Flood output.")
    ap.add_argument("file", help="BG_Flood output NetCDF")
    ap.add_argument("-o", "--out", help="output HTML (default: <file>.html)")
    ap.add_argument("--vars", help="comma-separated variables (default: all)")
    ap.add_argument("--steps", default="all", help="1-based steps: all | last | 1,3,5 | 2-6 (default all)")
    ap.add_argument("--eps", type=float, default=0.03, help="initial dry threshold on depth, m (default 0.03)")
    ap.add_argument("--region", help="crop to xmin,xmax,ymin,ymax (model CRS)")
    ap.add_argument("--max-level", type=int, help="average levels finer than this into it (each level dropped cuts size ~4x)")
    ap.add_argument("--crs", help="override CRS, e.g. EPSG:2193")
    ap.add_argument("--title", help="page title (default: file name)")
    ap.add_argument("--linz-key", default="", help="embed a LINZ basemaps API key (visible to anyone with the file)")
    ap.add_argument("--google-key", default="", help="embed a Google Map Tiles API key (visible to anyone with the file)")
    ap.add_argument("--google-region", default="NZ", help="region code for Google tiles (default NZ)")
    ap.add_argument("--vendor-dir", help="folder holding leaflet.js, leaflet.css, proj4.js (offline builds)")
    a = ap.parse_args(argv)

    src = BGSource(a.file, a.crs)
    all_vars = [v for v in PREFERRED_ORDER if v in src.vars] + sorted(v for v in src.vars if v not in PREFERRED_ORDER)
    if a.vars:
        want = [v.strip() for v in a.vars.split(",") if v.strip()]
        miss = [v for v in want if v not in src.vars]
        if miss:
            raise SystemExit(f"Not in file: {', '.join(miss)}. Available: {', '.join(all_vars)}")
        var_list = want
        # depth variables are always included: masking and KPIs need them
        for d in ("hmax", "h"):
            if d in src.vars and d not in var_list:
                var_list.append(d)
        var_list = [v for v in all_vars if v in var_list]
    else:
        var_list = all_vars
    if src.nt == 0:
        raise SystemExit("The data arrays in this file have no time records: BG_Flood created it "
                         "but never wrote an output step (did the run stop early?).")
    steps = parse_steps(a.steps, src.nt)
    if not steps:
        raise SystemExit(f"No valid steps in --steps {a.steps!r} (file has {src.nt}).")
    region = tuple(float(x) for x in a.region.split(",")) if a.region else None

    print(f"Reading {a.file}: levels {src.levlist[0]}..{src.levlist[-1]}, "
          f"{len(var_list)} variables, {len(steps)} of {src.nt} steps")
    blobs, meshes, step_mesh, fields, samples, depth_of, Lm = build_package(src, var_list, steps, a.eps, region, a.max_level)

    # keep only steps with data
    keep = [i for i, m in enumerate(step_mesh) if m is not None]
    step_mesh = [step_mesh[i] for i in keep]
    steps_k = [steps[i] for i in keep]
    for v in var_list:
        fields[v] = [fields[v][i] for i in keep]

    # variable metadata + default colour ranges
    vmeta = {}
    for v in var_list:
        ln, un = src.var_info(v)
        vals = np.concatenate(samples[v]) if samples[v] else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98]).astype(float)
            if hi <= lo:
                lo, hi = float(vals.min()), float(vals.max()) + (1.0 if vals.max() == vals.min() else 0)
        else:
            lo, hi = 0.0, 1.0
        signed = v in SIGNED or (lo < 0 < hi and v.lower() in SIGNED)
        if signed:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        vmeta[v] = {"long": ln, "units": un, "range": [float(f"{lo:.4g}"), float(f"{hi:.4g}")],
                    "depth": depth_of.get(v), "cmap": "rdbu" if signed else "seminf"}

    levels = {str(l): {"dx": L.dx, "dy": L.dy, "x0": L.x0, "y0": L.y0} for l, L in src.levels.items()}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proj4 = src.crs.to_proj4()
    except Exception:
        proj4 = None
    tl = [src.time_label(t) for t in steps_k]
    t0s = tl[0][1] if tl else 0.0
    L0 = src.levels[src.levlist[0]]
    ext = [L0.x0, L0.x0 + L0.nx * L0.dx, L0.y0, L0.y0 + L0.ny * L0.dy]
    default_var = next((v for v in ("hmax", "h", "zsmax", "zs") if v in var_list), var_list[0])
    title = a.title or os.path.splitext(os.path.basename(a.file))[0]
    meta = {
        "title": title, "file": os.path.basename(a.file), "built": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bw": BW, "levels": levels, "levlist": src.levlist, "extent": ext,
        "crs": src.crs_desc, "crsSource": src.crs_source, "proj4": proj4,
        "vars": vmeta, "varorder": var_list, "defaultVar": default_var,
        "steps": [s + 1 for s in steps_k], "ntFile": src.nt, "tlabels": [x[0] for x in tl],
        "thours": [(x[1] - t0s) / 3600.0 for x in tl], "tunits": src.time_units,
        "stepMesh": step_mesh, "meshes": meshes, "fields": fields, "eps": a.eps,
        "keys": {"linz": a.linz_key, "google": a.google_key}, "googleRegion": a.google_region,
        "region": region, "maxLevel": (Lm if Lm < src.levlist[-1] else None),
    }

    vend = {k: vendor_text(k, a.vendor_dir) for k in VENDOR}

    def script(name):
        t = vend[name]
        return f"<script>{t}</script>" if t else f'<script src="{CDN[name]}"></script>'
    css = vend["leaflet.css"]
    css_tag = f"<style>{css}</style>" if css else f'<link rel="stylesheet" href="{CDN["leaflet.css"]}">'

    # JSON inside <script>: escape every "<" so nothing in it can end or confuse the tag
    meta_json = json.dumps(meta, separators=(",", ":")).replace("<", "\\u003c")
    page = (PAGE.replace("__TITLE__", _html_escape(title))
                .replace("<!--LEAFLET_CSS-->", css_tag)
                .replace("<!--LEAFLET_JS-->", script("leaflet.js"))
                .replace("<!--PROJ4_JS-->", script("proj4.js"))
                .replace("/*META*/", meta_json))
    head, tail = page.split("<!--DATA-->")
    out = a.out or os.path.splitext(a.file)[0] + ".html"
    # Each blob goes in its own non-executed tag, split into parts well below the
    # browser's ~512 M-character string limit, and the file is written as a stream.
    PART = 32 * 2**20
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(head)
        for k in list(blobs):
            b64 = base64.b64encode(blobs.pop(k)).decode("ascii")
            for i in range(0, max(len(b64), 1), PART):
                fh.write(f'<script type="application/octet-stream" id="d:{k}:{i // PART}">{b64[i:i + PART]}</script>\n')
            del b64
        fh.write(tail)
    nb = [m["nb"] for m in meshes]
    size = os.path.getsize(out) / 2**20
    print(f"Wrote {out}: {size:.1f} MB, {len(meshes)} mesh(es) of "
          f"{min(nb):,}-{max(nb):,} blocks, {len(steps_k)} steps x {len(var_list)} variables")
    big = max(meshes, key=lambda m: m["nb"])["counts"]
    print("  blocks per level: " + ", ".join(f"L{l} ({src.levels[l].dx:g} m): {n:,}" for l, n in sorted(big.items())))
    if size > 300:
        print(f"  warning: {size:.0f} MB is heavy for a browser. Reduce it with --max-level "
              f"{Lm - 1 if Lm > src.levlist[0] else Lm} (about 4x smaller), --vars, --steps or --region.")
    if a.linz_key or a.google_key:
        print("  note: API keys are embedded in the HTML; anyone with the file can read them.")
    return out


def _html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


########################################################################
# Page template

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · BG_Flood results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
<!--LEAFLET_CSS-->
<style>
:root{color-scheme:light}
.app{--bg:#ffffff;--panel:#f5f6f4;--ink:#0f1d2b;--ink2:#4d5965;--mute:#7c8790;--line:#dde1e3;--e1:#2a78d6;--e2:#eb6834;--land:#ebe5d4;color:var(--ink);background:var(--bg)}
.app.dark{color-scheme:dark;--bg:#11161a;--panel:#192026;--ink:#e9edf0;--ink2:#b7c0c7;--mute:#8a959e;--line:#2b353d;--e1:#3987e5;--e2:#d95926;--land:#2b2923}
*{box-sizing:border-box;min-width:0}
body{margin:0;font:16px/1.5 Barlow,system-ui,sans-serif;overflow-x:clip}
h1,h2,h3,.fig-n{font-family:"Barlow Semi Condensed",Barlow,sans-serif}
.band{padding:clamp(20px,3vw,40px) max(4vw,16px)}
.wrap{max-width:1400px;margin:0 auto}
header.band{padding-bottom:12px}
.kick{font-size:14px;color:var(--ink2);margin:0 0 6px}
h1{font-size:clamp(28px,3.4vw,44px);line-height:1.05;margin:0 0 12px;text-wrap:balance;font-weight:600}
.sub{color:var(--ink2);margin:0;max-width:90ch}
.ev1{color:var(--e1)}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:18px}
.kpi{background:var(--panel);border-radius:10px;padding:12px 14px}
.kpi h3{font-size:14px;font-weight:500;color:var(--ink2);margin:0 0 6px;font-family:Barlow,system-ui,sans-serif}
.fig-n{font-size:clamp(22px,2.2vw,30px);font-weight:600;line-height:1.15;white-space:nowrap}
.kpi .note{font-size:13px;color:var(--mute)}
.main{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:18px;align-items:start}
.card{background:var(--panel);border-radius:12px;padding:14px}
.card h2{font-size:20px;margin:0 0 4px;font-weight:600}
.card .hint{font-size:14px;color:var(--ink2);margin:0 0 10px}
.ctl{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin-bottom:10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--bg)}
.seg button{border:0;background:none;color:var(--ink);font:500 15px Barlow,sans-serif;padding:0 12px;min-height:38px;cursor:pointer;white-space:nowrap}
.seg button+button{border-left:1px solid var(--line)}
select,input[type=number],input[type=text]{font:16px Barlow,sans-serif;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:8px;min-height:38px;padding:0 8px;max-width:100%}
input[type=number]{width:5.6em}
#keyin{width:15em}
label.l{font-size:14px;color:var(--ink2);display:inline-flex;gap:6px;align-items:center;white-space:nowrap}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--e1);outline-offset:2px}
.btn{font:500 14px Barlow,sans-serif;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:8px;min-height:38px;padding:0 12px;cursor:pointer;white-space:nowrap}
.btn:disabled,select:disabled,input:disabled{opacity:.5;cursor:default}
.player{display:flex;gap:10px;align-items:center;margin:0 0 10px;padding:8px 10px;background:var(--bg);border:1px solid var(--line);border-radius:10px}
.player .btn{min-width:44px;font-size:16px}
.player input[type=range]{flex:1;accent-color:var(--e1);min-width:120px}
#tlabel{font:600 15px "Barlow Semi Condensed",Barlow,sans-serif;white-space:nowrap;font-variant-numeric:tabular-nums}
#map{height:clamp(440px,68vh,780px);border-radius:10px;background:var(--land);position:relative}
.leaflet-container{font:14px Barlow,sans-serif;background:var(--land)}
.leaflet-control-attribution{font-size:11px}
.mapover{position:absolute;z-index:700;background:color-mix(in srgb,var(--bg) 92%,transparent);color:var(--ink);border-radius:8px;padding:8px 10px;font-size:13px;line-height:1.35;pointer-events:none;box-shadow:0 1px 3px #0003}
#legend{left:10px;bottom:22px;width:250px}
.mapover canvas{width:100%;height:12px;display:block;border-radius:3px;margin:4px 0 2px}
.mapover .ends{display:flex;justify-content:space-between;color:var(--ink2);gap:6px}
#readout{right:10px;top:10px;min-width:200px;display:none}
#readout table{border-collapse:collapse}#readout td{padding:1px 4px;text-align:right;white-space:nowrap}#readout td:first-child{text-align:left;color:var(--ink2)}
#loading{position:absolute;inset:0;display:grid;place-items:center;z-index:800;color:var(--ink2);font-size:15px;pointer-events:none}
#bmnote{font-size:13px;color:var(--ink2);margin:6px 2px 0;min-height:1.3em}
.side{display:grid;gap:18px}
.gstats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 12px;font-size:14px;margin:8px 0 6px}
.gstats div span{display:block;color:var(--ink2);font-size:13px}
.gstats b{font-weight:600;white-space:nowrap}
#chart{width:100%;height:240px;display:block}
table.t{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
.t th{font-weight:600;text-align:right;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.t th:first-child,.t td:first-child{text-align:left}
.t td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:6px}
footer{font-size:13px;color:var(--mute)}
footer p{max-width:110ch}
.hide{display:none!important}
@media(max-width:1020px){.main{grid-template-columns:minmax(0,1fr)}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.kpis{grid-template-columns:minmax(0,1fr)}.gstats{grid-template-columns:repeat(2,minmax(0,1fr))}#legend{width:190px}}
</style>
</head>
<body>
<div class="app" id="app">
<header class="band"><div class="wrap">
  <p class="kick" id="kick"></p>
  <h1 id="finding">BG_Flood results</h1>
  <p class="sub" id="subline"></p>
  <div class="kpis" id="kpis"></div>
</div></header>

<section class="band" style="padding-top:8px"><div class="wrap">
 <div class="main">
  <div class="card" id="map-card">
    <h2>Results map</h2>
    <p class="hint">Each 16 × 16-cell block is drawn at the resolution it was modelled at. Hover for values and cell size; click anywhere for the time series at that point.</p>
    <div class="ctl">
      <select id="var" aria-label="Variable"></select>
      <select id="cmap" aria-label="Colour bar">
        <option value="rdbu">Colour bar: red → blue</option>
        <option value="jet">Colour bar: jet</option>
        <option value="seminf">Colour bar: seminf-haxby</option>
        <option value="haxby">Colour bar: GMT haxby</option>
      </select>
      <label class="l"><input type="checkbox" id="rev"> Reverse</label>
      <label class="l">Range <input type="number" id="rmin" step="any" aria-label="Colour minimum"> to <input type="number" id="rmax" step="any" aria-label="Colour maximum"></label>
      <button class="btn" id="rreset" title="Range over all steps">Reset</button>
      <button class="btn" id="rfit" title="2–98 % of the values at this step">Fit to step</button>
    </div>
    <div class="ctl">
      <select id="basemap" aria-label="Basemap">
        <option value="none">Basemap: none</option>
        <option value="osm">Basemap: OpenStreetMap</option>
        <option value="linz_aerial">Basemap: LINZ aerial</option>
        <option value="linz_topo">Basemap: LINZ topo</option>
        <option value="g_satellite">Basemap: Google satellite</option>
        <option value="g_hybrid">Basemap: Google hybrid</option>
        <option value="g_roadmap">Basemap: Google roadmap</option>
      </select>
      <label class="l hide" id="keywrap"><span id="keylab">Key</span> <input type="text" id="keyin" placeholder="paste API key" autocomplete="off" spellcheck="false"></label>
      <div class="seg" id="zoomseg" role="group" aria-label="Zoom to"><button data-z="all">Whole model</button><button data-z="fine">Finest mesh</button></div>
      <label class="l"><input type="checkbox" id="meshon"> Mesh blocks</label>
      <span class="l"><label class="l"><input type="checkbox" id="mask" checked> Hide dry cells, depth &lt;</label><input type="number" id="eps" step="0.005" min="0" aria-label="Dry threshold (m)"> m</span>
      <label class="l">Opacity <input type="range" id="opac" min="0.2" max="1" step="0.05" value="0.85"></label>
    </div>
    <div class="player" id="player">
      <button class="btn" id="prev" aria-label="Previous step">⏮</button>
      <button class="btn" id="play" aria-label="Play">▶</button>
      <button class="btn" id="next" aria-label="Next step">⏭</button>
      <input type="range" id="tslider" min="0" max="0" step="1" value="0" aria-label="Time step">
      <span id="tlabel"></span>
      <select id="speed" aria-label="Playback speed"><option value="1600">0.5×</option><option value="800" selected>1×</option><option value="400">2×</option><option value="200">4×</option></select>
    </div>
    <div id="map"><div id="loading">Loading model results…</div>
      <div class="mapover" id="legend"><div id="lgtitle"></div><canvas id="lgbar" width="250" height="12"></canvas><div class="ends"><span id="lgmin"></span><span id="lgmax"></span></div></div>
      <div class="mapover" id="readout"></div>
    </div>
    <p id="bmnote"></p>
  </div>
  <div class="side">
   <div class="card" id="probe-card">
    <h2 id="pname">Point</h2>
    <p class="hint" id="phint">Click the map to see how the selected variable changes through time at that point.</p>
    <div class="gstats hide" id="pstats"></div>
    <canvas id="chart" class="hide"></canvas>
   </div>
   <div class="card">
    <h2>Mesh at this time</h2>
    <p class="hint" id="meshhint"></p>
    <table class="t" id="meshtbl"></table>
   </div>
  </div>
 </div>
</div></section>

<footer class="band"><div class="wrap">
 <p id="howto"></p>
 <p>Basemaps: © OpenStreetMap contributors; LINZ Basemaps © Toitū Te Whenua LINZ, CC BY 4.0 (free API key from basemaps.linz.govt.nz); Google tiles through the Google Map Tiles API (your own API key, Google terms apply). Keys you paste are kept only in this browser. Colour bars: jet (MATLAB); seminf-haxby and haxby from GMT / cpt-city; red → blue after ColorBrewer RdBu.</p>
 <p id="srcnote"></p>
</div></footer>
</div>

<script id="meta-json" type="application/json">/*META*/</script>
<!--DATA-->
<!--LEAFLET_JS-->
<!--PROJ4_JS-->
<script>
{
const M=JSON.parse(document.getElementById('meta-json').textContent);
const $=id=>document.getElementById(id);
const app=$('app');
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
const hosted=/claude|anthropic/.test(location.hostname);
const NAN=Number.NaN,LEV='__level',BW=M.bw,BB=BW*BW;
const f=(v,d=2)=>v==null||Number.isNaN(v)?'–':v.toFixed(d);
const fn=v=>{if(v==null||Number.isNaN(v))return '–';const a=Math.abs(v);return a>=100?v.toFixed(0):a>=10?v.toFixed(1):v.toFixed(2);};
const NS=M.steps.length;
const S={v:M.defaultVar,s:NS-1,cmap:M.vars[M.defaultVar].cmap,cmapUser:false,rev:false,range:{},mesh:false,mask:true,eps:M.eps,opac:0.85,
  basemap:M.proj4?(store.get('bgv_bm')||'none'):'none',keys:{linz:store.get('bgv_linz')||M.keys.linz||'',google:store.get('bgv_google')||M.keys.google||''},
  probe:null,playing:false,interval:800};

/* ---------- theme ---------- */
let dark=false,PAL={};
function readTheme(){
  const t=document.documentElement.getAttribute('data-theme');
  dark=t==='dark'||(t!=='light'&&matchMedia('(prefers-color-scheme: dark)').matches);
  app.classList.toggle('dark',dark);document.body.style.background=dark?'#11161a':'#ffffff';
  PAL=dark?{e1:'#3987e5',ink:'#e9edf0',ink2:'#b7c0c7',line:'#2b353d',panel:'#192026',mesh:'rgba(255,255,255,0.6)'}
          :{e1:'#2a78d6',ink:'#0f1d2b',ink2:'#4d5965',line:'#dde1e3',panel:'#f5f6f4',mesh:'rgba(15,29,43,0.55)'};
}
readTheme();

/* ---------- colour bars ---------- */
const hex=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));
const rgbh=a=>'#'+a.map(v=>v.toString(16).padStart(2,'0')).join('');
function lut(stops){
  const s=stops.map((c,i)=>Array.isArray(c)?[c[0],hex(c[1])]:[i/(stops.length-1),hex(c)]),o=new Uint8Array(256*3);
  for(let i=0;i<256;i++){const t=i/255;let k=0;while(k<s.length-2&&t>s[k+1][0])k++;const u=Math.min(1,Math.max(0,(t-s[k][0])/(s[k+1][0]-s[k][0]||1)));
    for(let j=0;j<3;j++)o[i*3+j]=Math.round(s[k][1][j]+(s[k+1][1][j]-s[k][1][j])*u);}
  return o;
}
const HAXBY=[[10,0,121],[40,0,150],[20,5,175],[0,10,200],[0,25,212],[0,40,224],[26,102,240],[13,129,248],[25,175,255],[50,190,255],[68,202,255],[97,225,240],[106,235,225],[124,235,200],[138,236,174],[172,245,168],[205,255,162],[223,245,141],[240,236,121],[247,215,104],[255,189,87],[255,160,69],[244,117,75],[238,80,78],[255,90,90],[255,124,124],[255,158,158],[245,179,174],[255,196,196],[255,215,215],[255,235,235],[255,255,255]].map(rgbh);
const SEMINF=[[255,255,255],[208,216,251],[186,197,247],[143,161,241],[97,122,236],[0,39,224],[25,101,240],[12,129,248],[24,175,255],[49,190,255],[67,202,255],[96,225,240],[105,235,225],[123,235,200],[138,236,174],[172,245,168],[205,255,162],[223,245,141],[240,236,120],[247,215,103],[255,189,86],[255,160,68],[244,116,74],[238,79,77]].map(rgbh);
const JET=[[0,'#00007f'],[0.125,'#0000ff'],[0.375,'#00ffff'],[0.625,'#ffff00'],[0.875,'#ff0000'],[1,'#7f0000']];
const LUT={rdbu:lut(['#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac']),jet:lut(JET),seminf:lut(SEMINF),haxby:lut(HAXBY)};
const LEVCOL=['#0d366b','#1c5cab','#3987e5','#86b6ef','#f0b13c','#e5732a','#c1381e','#7a1d10'];
function colIdx(t){t=t<0?0:t>1?1:t;let i=Math.round(t*255);if(S.rev)i=255-i;return i*3;}

/* ---------- data ---------- */
async function bytes(name){
  const parts=[];for(let i=0;;i++){const el=document.getElementById('d:'+name+':'+i);if(!el)break;parts.push(el.textContent.trim());}
  if(!parts.length)throw new Error('data block "'+name+'" is missing from this file (was it truncated while copying?)');
  const arrs=parts.map(p=>{const s=atob(p),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;});
  let buf=arrs[0];if(arrs.length>1){buf=new Uint8Array(arrs.reduce((n,a)=>n+a.length,0));let o=0;for(const a of arrs){buf.set(a,o);o+=a.length;}}
  if(buf[0]===0x1f&&buf[1]===0x8b){
    if(typeof DecompressionStream==='undefined')throw new Error('this browser cannot unpack the data (needs DecompressionStream: current Chrome, Edge, Firefox or Safari)');
    buf=new Uint8Array(await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());}
  return buf.buffer;
}
const FQ=new Map(),fpend={};
function loadField(v,s){
  const m=M.fields[v]?.[s];if(!m)return Promise.resolve(null);const k=v+'|'+s;
  if(FQ.has(k)){const o=FQ.get(k);FQ.delete(k);FQ.set(k,o);return Promise.resolve(o);}
  return fpend[k]||(fpend[k]=bytes(m.file).then(b=>{const u=new Uint8Array(b),n=m.n,q=new Uint16Array(n);for(let i=0;i<n;i++)q[i]=u[i]|(u[n+i]<<8);
    const o={q,lo:m.lo,st:m.st};FQ.set(k,o);delete fpend[k];while(FQ.size>48)FQ.delete(FQ.keys().next().value);return o;}));
}
const F_=(v,s)=>FQ.get(v+'|'+s)||null;
const val=(F,ci)=>{if(!F||ci<0)return NAN;const q=F.q[ci];return q?F.lo+(q-1)*F.st:NAN;};

/* ---------- projection ---------- */
const R0=20037508.342789244;
let toM,fromM,toLL=null;
if(M.proj4){proj4.defs('SRC',M.proj4);const pm=proj4('SRC','EPSG:3857'),pl=proj4('SRC','EPSG:4326');toM=p=>pm.forward(p);fromM=p=>pm.inverse(p);toLL=p=>pl.forward(p);}
else{const cx=(M.extent[0]+M.extent[1])/2,cy=(M.extent[2]+M.extent[3])/2;toM=p=>[p[0]-cx,p[1]-cy];fromM=p=>[p[0]+cx,p[1]+cy];}
const m2ll=(mx,my)=>L.CRS.EPSG3857.unproject(L.point(mx,my));
const ll2nat=ll=>{const p=L.CRS.EPSG3857.project(ll);return fromM([p.x,p.y]);};

/* ---------- meshes ---------- */
const MS={},mpend={};
function loadMesh(id){
  if(MS[id])return Promise.resolve(MS[id]);
  return mpend[id]||(mpend[id]=bytes(M.meshes[id].file).then(b=>{const nb=M.meshes[id].nb;
    return MS[id]=prepMesh(id,new Int32Array(b,0,nb),new Int32Array(b,4*nb,nb),new Int8Array(b,8*nb,nb));}));
}
function prepMesh(id,bi,bj,lev){
  const nb=bi.length,W=new Float64Array(nb),E=new Float64Array(nb),So=new Float64Array(nb),No=new Float64Array(nb),dx=new Float64Array(nb),dy=new Float64Array(nb);
  const cm=new Float64Array(nb*6),mb=new Float64Array(nb*4);let H=0,X0=1e20,Y0=1e20,dmin=1e20;
  for(let k=0;k<nb;k++){const Lv=M.levels[lev[k]];dx[k]=Lv.dx;dy[k]=Lv.dy;W[k]=Lv.x0+bi[k]*BW*Lv.dx;E[k]=W[k]+BW*Lv.dx;So[k]=Lv.y0+bj[k]*BW*Lv.dy;No[k]=So[k]+BW*Lv.dy;
    H=Math.max(H,BW*Lv.dx,BW*Lv.dy);X0=Math.min(X0,W[k]);Y0=Math.min(Y0,So[k]);dmin=Math.min(dmin,Lv.dx);
    const nw=toM([W[k],No[k]]),ne=toM([E[k],No[k]]),sw=toM([W[k],So[k]]),se=toM([E[k],So[k]]);
    cm.set([nw[0],nw[1],ne[0],ne[1],sw[0],sw[1]],k*6);
    mb.set([Math.min(nw[0],sw[0]),Math.min(sw[1],se[1]),Math.max(ne[0],se[0]),Math.max(nw[1],ne[1])],k*4);}
  const hash=new Map();
  for(let k=0;k<nb;k++)for(let gx=Math.floor((W[k]-X0)/H);gx<=Math.floor((E[k]-1e-6-X0)/H);gx++)for(let gy=Math.floor((So[k]-Y0)/H);gy<=Math.floor((No[k]-1e-6-Y0)/H);gy++){const kk=gy*1e7+gx;let a=hash.get(kk);if(!a)hash.set(kk,a=[]);a.push(k);}
  const bnd=sel=>{let a=1e20,b=1e20,c=-1e20,d=-1e20;for(let k=0;k<nb;k++)if(sel(k)){a=Math.min(a,mb[4*k]);b=Math.min(b,mb[4*k+1]);c=Math.max(c,mb[4*k+2]);d=Math.max(d,mb[4*k+3]);}return L.latLngBounds(m2ll(a,b),m2ll(c,d));};
  return {id,nb,bi,bj,lev,W,E,S:So,N:No,dx,dy,cm,mb,hash,H,X0,Y0,dmin,all:bnd(()=>true),fine:bnd(k=>dx[k]===dmin)};
}
function lookup(ms,x,y){
  if(!ms)return -1;const a=ms.hash.get(Math.floor((y-ms.Y0)/ms.H)*1e7+Math.floor((x-ms.X0)/ms.H));if(!a)return -1;
  let best=-1,bd=1e20;for(const k of a)if(x>=ms.W[k]&&x<ms.E[k]&&y>=ms.S[k]&&y<ms.N[k]&&ms.dx[k]<bd){bd=ms.dx[k];best=k;}
  if(best<0)return -1;
  return best*BB+Math.min(BW-1,Math.floor((y-ms.S[best])/ms.dy[best]))*BW+Math.min(BW-1,Math.floor((x-ms.W[best])/ms.dx[best]));
}
const meshAt=s=>MS[M.stepMesh[s]];
const depthVar=v=>v===LEV?null:M.vars[v]?.depth||null;
async function ensureStep(s,vars){
  await loadMesh(M.stepMesh[s]);
  await Promise.all([...new Set(vars.filter(Boolean))].filter(v=>v!==LEV).map(v=>loadField(v,s)));
}
function maskedVal(v,s,ci){
  if(ci<0)return NAN;
  if(v===LEV){const ms=meshAt(s);return ms?ms.lev[Math.floor(ci/BB)]:NAN;}
  const x=val(F_(v,s),ci);if(!S.mask||isNaN(x))return x;
  const d=depthVar(v);if(!d)return x;const h=d===v?x:val(F_(d,s),ci);return h>=S.eps?x:NAN;
}

/* ---------- ranges ---------- */
function curRange(){
  if(S.v===LEV){return [M.levlist[0]-0.5,M.levlist[M.levlist.length-1]+0.5];}
  return S.range[S.v]||M.vars[S.v].range;
}

/* ---------- map ---------- */
const map=L.map('map',{minZoom:2,maxZoom:22,zoomSnap:0.25,attributionControl:true});
map.attributionControl.setPrefix(false);
map.createPane('meshPane').style.zIndex=410;map.createPane('ptPane').style.zIndex=640;
let ATL=null;const ACOLS=256;
function buildAtlas(){
  const s=S.s,ms=meshAt(s);if(!ms)return null;
  const nb=ms.nb,rows=Math.ceil(nb/ACOLS),Wd=Math.min(nb,ACOLS)*BW,cv=document.createElement('canvas');cv.width=Wd;cv.height=rows*BW;
  const ctx=cv.getContext('2d'),img=ctx.createImageData(Wd,rows*BW),px=img.data;
  const [lo,hi]=curRange(),span=hi-lo||1,lt=LUT[S.cmap],isLev=S.v===LEV;
  const F=isLev?null:F_(S.v,s),dv=depthVar(S.v),D=dv&&S.mask?(dv===S.v?F:F_(dv,s)):null,eps=S.eps;
  for(let k=0;k<nb;k++){
    const bx=(k%ACOLS)*BW,by=Math.floor(k/ACOLS)*BW,off=k*BB;
    let lc=null;if(isLev){const i=M.levlist.indexOf(ms.lev[k]);lc=hex(LEVCOL[Math.max(0,i)%LEVCOL.length]);}
    for(let j=0;j<BW;j++)for(let i=0;i<BW;i++){
      const ci=off+j*BW+i,p=((by+BW-1-j)*Wd+bx+i)*4;
      if(isLev){px[p]=lc[0];px[p+1]=lc[1];px[p+2]=lc[2];px[p+3]=255;continue;}
      const q=F.q[ci];if(!q)continue;
      if(D){const qd=D.q[ci];if(!qd||D.lo+(qd-1)*D.st<eps)continue;}
      const c=colIdx((F.lo+(q-1)*F.st-lo)/span);px[p]=lt[c];px[p+1]=lt[c+1];px[p+2]=lt[c+2];px[p+3]=255;
    }
  }
  ctx.putImageData(img,0,0);
  return {cv,ms};
}
function drawTile(t,c){
  const ctx=t.getContext('2d');ctx.clearRect(0,0,256,256);if(!ATL)return;
  const {cv,ms}=ATL,sc=256*Math.pow(2,Math.round(c.z))/(2*R0),ox=c.x*256,oy=c.y*256;
  const mx0=ox/sc-R0,mx1=(ox+256)/sc-R0,my1=R0-oy/sc,my0=R0-(oy+256)/sc;
  ctx.imageSmoothingEnabled=false;ctx.strokeStyle=PAL.mesh;
  for(let k=0;k<ms.nb;k++){const b=k*4;if(ms.mb[b]>mx1||ms.mb[b+2]<mx0||ms.mb[b+1]>my1||ms.mb[b+3]<my0)continue;
    const q=k*6,nwx=(ms.cm[q]+R0)*sc-ox,nwy=(R0-ms.cm[q+1])*sc-oy,nex=(ms.cm[q+2]+R0)*sc-ox,ney=(R0-ms.cm[q+3])*sc-oy,swx=(ms.cm[q+4]+R0)*sc-ox,swy=(R0-ms.cm[q+5])*sc-oy;
    const ax=(nex-nwx)/BW,ay=(ney-nwy)/BW,cx=(swx-nwx)/BW,cy=(swy-nwy)/BW,cell=Math.hypot(ax,ay),e=Math.min(0.5,0.6/cell);
    ctx.setTransform(ax,ay,cx,cy,nwx,nwy);
    ctx.drawImage(cv,(k%ACOLS)*BW,Math.floor(k/ACOLS)*BW,BW,BW,-e,-e,BW+2*e,BW+2*e);
    if(S.mesh&&cell*BW>10){ctx.lineWidth=1/cell;ctx.strokeRect(0,0,BW,BW);}
  }
  ctx.setTransform(1,0,0,1,0,0);
}
const ResLayer=L.GridLayer.extend({createTile(c){const t=L.DomUtil.create('canvas');t.width=t.height=256;drawTile(t,c);return t;}});
const resLayer=new ResLayer({pane:'meshPane',tileSize:256,maxZoom:22,opacity:S.opac,updateWhenZooming:false,keepBuffer:1}).addTo(map);
function refreshLayer(){resLayer.remove();resLayer.addTo(map);}

/* ---------- legend ---------- */
function vname(v){return v===LEV?'Mesh refinement level':M.vars[v].long;}
function vunit(v){return v===LEV?'':M.vars[v].units;}
function legend(){
  const c=$('lgbar'),x=c.getContext('2d');x.clearRect(0,0,250,12);
  if(S.v===LEV){const lv=M.levlist,w=250/lv.length;lv.forEach((l,i)=>{x.fillStyle=LEVCOL[i%LEVCOL.length];x.fillRect(i*w,0,w-1,12);});
    $('lgtitle').textContent='Cell size per level: '+lv.map(l=>`${M.levels[l].dx} m`).join(' · ');$('lgmin').textContent='coarse';$('lgmax').textContent='fine';return;}
  const [lo,hi]=curRange(),lt=LUT[S.cmap];
  for(let i=0;i<250;i++){const k=colIdx(i/249);x.fillStyle=`rgb(${lt[k]},${lt[k+1]},${lt[k+2]})`;x.fillRect(i,0,1,12);}
  $('lgtitle').textContent=vname(S.v)+(vunit(S.v)?` (${vunit(S.v)})`:'')+` · ${M.tlabels[S.s]}`;
  $('lgmin').textContent=fn(lo);$('lgmax').textContent=fn(hi);
}
function syncRange(){const [lo,hi]=curRange();$('rmin').value=+lo.toPrecision(4);$('rmax').value=+hi.toPrecision(4);
  const lv=S.v===LEV;for(const id of ['rmin','rmax','rreset','rfit','cmap','rev'])$(id).disabled=lv;}

/* ---------- render ---------- */
let tok=0;
async function render(){
  const my=++tok,s=S.s;
  await ensureStep(s,[S.v,depthVar(S.v),...kpiVars()]);
  if(my!==tok)return;
  ATL=buildAtlas();refreshLayer();legend();syncRange();syncTime();kpis();meshTable();drawProbe();
}
function syncTime(){$('tslider').value=S.s;$('tlabel').textContent=`${M.tlabels[S.s]}  ·  step ${M.steps[S.s]} of ${M.ntFile}`;}

/* ---------- KPIs ---------- */
const has=v=>v in M.vars;
const KD=has('hmax')?'hmax':has('h')?'h':null,KSP=has('Umax')?'Umax':(has('u')&&has('v'))?'uv':null,KZ=has('zsmax')?'zsmax':has('zs')?'zs':null;
function kpiVars(){const a=[KD,KZ];if(KSP==='uv')a.push('u','v','h');else a.push(KSP);if(KSP==='Umax'&&has('hmax'))a.push('hmax');return a.filter(Boolean);}
function kpis(){
  const s=S.s,ms=meshAt(s);if(!ms)return;const D=KD&&F_(KD,s),eps=S.eps;
  let dmax=-Infinity,dk=-1,area=0,smax=-Infinity,zmax=-Infinity,zk=-1;
  const Z=KZ&&F_(KZ,s),U=KSP==='Umax'?F_('Umax',s):null,Uu=KSP==='uv'?F_('u',s):null,Uv=KSP==='uv'?F_('v',s):null,Hm=KSP==='uv'?F_('h',s):(has('hmax')?F_('hmax',s):null);
  const n=ms.nb*BB;
  for(let ci=0;ci<n;ci++){
    const d=val(D,ci);if(!(d>=eps))continue;const k=(ci/BB)|0;
    area+=ms.dx[k]*ms.dy[k];if(d>dmax){dmax=d;dk=ci;}
    if(Z){const z=val(Z,ci);if(z>zmax){zmax=z;zk=ci;}}
    if(U){const u=val(U,ci);if(u>smax)smax=u;}else if(Uu){const h=val(Hm,ci);if(h>=eps){const u=Math.hypot(val(Uu,ci),val(Uv,ci));if(u>smax)smax=u;}}
  }
  const areaS=area>=1e6?f(area/1e6,2)+' km²':area>=1e4?f(area/1e4,1)+' ha':f(area,0)+' m²';
  const levs=Object.keys(M.meshes[M.stepMesh[s]].counts).map(Number).sort((a,b)=>a-b);
  const dxs=levs.map(l=>M.levels[l].dx);
  const t=(h,a,note)=>`<div class="kpi"><h3>${h}</h3><div class="fig-n ev1">${a}</div><div class="note">${note}</div></div>`;
  const dmaxS=dmax>-Infinity?f(dmax,2)+' m':'dry';
  let html='';
  if(KD)html+=t(KD==='hmax'?'Deepest water so far':'Deepest water now',dmaxS,KD==='hmax'?'maximum depth reached up to this time':'depth at this time');
  if(KD)html+=t(KD==='hmax'?'Area flooded so far':'Area wet now',areaS,`cells with ${KD} ≥ ${S.eps} m`);
  if(KSP)html+=t(KSP==='Umax'?'Fastest flow so far':'Fastest flow now',smax>-Infinity?f(smax,2)+' m/s':'–',KSP==='Umax'?'maximum speed up to this time':'speed |(u, v)| in wet cells');
  else if(KZ)html+=t(KZ==='zsmax'?'Highest water level so far':'Highest water level now',zmax>-Infinity?f(zmax,2)+' m':'–','water surface elevation, model datum');
  html+=t('Model mesh',`${ms.nb.toLocaleString()} blocks`,`${levs.length} level${levs.length>1?'s':''}, cells ${Math.max(...dxs)}–${Math.min(...dxs)} m`);
  $('kpis').innerHTML=html;
  $('finding').innerHTML=KD?(dmax>-Infinity?`By ${M.tlabels[s]}, water reaches <span class="ev1">${f(dmax,1)} m</span> deep and covers <span class="ev1">${areaS}</span>`:`No cell is wetter than ${S.eps} m at ${M.tlabels[s]}`):`${M.title} at ${M.tlabels[s]}`;
}

/* ---------- mesh table ---------- */
function meshTable(){
  const s=S.s,c=M.meshes[M.stepMesh[s]].counts,lv=Object.keys(c).map(Number).sort((a,b)=>a-b);let tot=0;
  const rows=lv.map(l=>{const L_=M.levels[l],a=c[l]*BW*BW*L_.dx*L_.dy;tot+=a;return [l,L_.dx,c[l],a];});
  $('meshtbl').innerHTML='<thead><tr><th>Level</th><th>Cell size</th><th>Blocks</th><th>Area</th></tr></thead><tbody>'+
    rows.map(([l,dx,n,a])=>`<tr><td><span class="sw" style="background:${LEVCOL[M.levlist.indexOf(l)%LEVCOL.length]}"></span>${l}</td><td>${dx} m</td><td>${n.toLocaleString()}</td><td>${f(a/1e6,2)} km²</td></tr>`).join('')+'</tbody>';
  $('meshhint').textContent=`Active 16 × 16-cell blocks at ${M.tlabels[s]}, by refinement level (finest blocks win where levels overlap). Total ${f(tot/1e6,2)} km².`+(M.meshes.length>1?` The mesh changes over time (${M.meshes.length} layouts).`:'');
}

/* ---------- readout ---------- */
map.on('mousemove',ev=>{
  const ro=$('readout'),ms=meshAt(S.s);if(!ms){ro.style.display='none';return;}
  const [x,y]=ll2nat(ev.latlng),ci=lookup(ms,x,y);
  if(ci<0){ro.style.display='none';return;}
  const k=Math.floor(ci/BB);
  let h=`<div style="margin-bottom:3px;color:var(--ink2)">E ${x.toFixed(0)} · N ${y.toFixed(0)}`+(toLL?`<br>${ev.latlng.lat.toFixed(5)}°, ${ev.latlng.lng.toFixed(5)}°`:'')+`</div><table>`;
  if(S.v!==LEV){const v=maskedVal(S.v,S.s,ci),raw=val(F_(S.v,S.s),ci);
    h+=`<tr><td>${vname(S.v)}</td><td><b>${isNaN(v)?(isNaN(raw)?'–':'dry'):fn(v)+' '+vunit(S.v)}</b></td></tr>`;
    const d=depthVar(S.v);if(d&&d!==S.v)h+=`<tr><td>${d}</td><td>${fn(val(F_(d,S.s),ci))} m</td></tr>`;}
  h+=`<tr><td>Cell</td><td>${ms.dx[k]} m (level ${ms.lev[k]})</td></tr></table>`;
  ro.innerHTML=h;ro.style.display='block';
});
map.on('mouseout',()=>$('readout').style.display='none');

/* ---------- point probe ---------- */
let pmark=null;
map.on('click',ev=>{const [x,y]=ll2nat(ev.latlng);S.probe={x,y,ll:ev.latlng};
  if(pmark)pmark.remove();pmark=L.circleMarker(ev.latlng,{pane:'ptPane',radius:7,weight:3,color:dark?'#fff':'#000',fillColor:PAL.e1,fillOpacity:1}).addTo(map);
  drawProbe(true);});
let ptok=0;
async function drawProbe(load){
  if(!S.probe)return;const my=++ptok,{x,y,ll}=S.probe,v=S.v;
  if(load||true){await Promise.all(M.steps.map((_,s)=>ensureStep(s,[v,depthVar(v)])));}
  if(my!==ptok)return;
  const ys=M.steps.map((_,s)=>maskedVal(v,s,lookup(meshAt(s),x,y)));
  const ci=lookup(meshAt(S.s),x,y),k=ci<0?-1:Math.floor(ci/BB),ms=meshAt(S.s);
  $('pname').textContent='Point '+(toLL?`${ll.lat.toFixed(4)}°, ${ll.lng.toFixed(4)}°`:`E ${x.toFixed(0)}, N ${y.toFixed(0)}`);
  $('phint').textContent=ci<0?'Outside the model domain.':`${vname(v)} at E ${x.toFixed(0)}, N ${y.toFixed(0)}; cell ${ms.dx[k]} m (level ${ms.lev[k]}).`+(depthVar(v)&&S.mask?` Gaps are dry (depth < ${S.eps} m).`:'');
  const fin=ys.map((y,i)=>[y,i]).filter(a=>Number.isFinite(a[0]));
  const mx=fin.length?fin.reduce((a,b)=>b[0]>a[0]?b:a):null,wet=fin.length;
  const u=vunit(v);
  $('pstats').innerHTML=`<div><span>Now</span><b>${fn(ys[S.s])} ${isNaN(ys[S.s])?'':u}</b></div><div><span>Maximum</span><b>${mx?fn(mx[0])+' '+u:'–'}</b></div><div><span>Time of maximum</span><b>${mx?M.tlabels[mx[1]].slice(11)||M.tlabels[mx[1]]:'–'}</b></div>`;
  $('pstats').classList.remove('hide');$('chart').classList.toggle('hide',NS<2);
  if(NS>=2)drawChart($('chart'),M.thours,ys,S.s,vname(v)+(u?` (${u})`:''));
}
function ticks(lo,hi,n){const r=hi-lo,st0=Math.pow(10,Math.floor(Math.log10(r/n))),st=[1,2,5,10].map(m=>m*st0).find(s=>r/s<=n)||st0*10;const out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)out.push(+v.toPrecision(10));return out;}
function drawChart(cv,xs,ys,cur,ylab){
  const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;
  const x=cv.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  x.font='13px Barlow, sans-serif';
  const fin=ys.filter(Number.isFinite);
  if(!fin.length){x.fillStyle=PAL.ink2;x.fillText('No data at this point (dry or outside the mesh).',8,24);return;}
  const ml=62,mr=12,mt=10,mbt=36,pw=W-ml-mr,ph=H-mt-mbt;
  let lo=Math.min(...fin),hi=Math.max(...fin);if(hi-lo<1e-9){lo-=0.5;hi+=0.5;}const pd=(hi-lo)*0.1;lo-=pd;hi+=pd;
  const X0=xs[0],X1=xs[xs.length-1]>X0?xs[xs.length-1]:X0+1;
  const sx=t=>ml+(t-X0)/(X1-X0)*pw,sy=v=>mt+(1-(v-lo)/(hi-lo))*ph;
  x.strokeStyle=PAL.line;x.fillStyle=PAL.ink2;x.lineWidth=1;x.textAlign='right';x.textBaseline='middle';
  for(const t of ticks(lo,hi,4)){const yy=Math.round(sy(t))+0.5;x.beginPath();x.moveTo(ml,yy);x.lineTo(ml+pw,yy);x.stroke();x.fillText(fn(t),ml-6,yy);}
  x.textAlign='center';x.textBaseline='top';
  for(const t of ticks(X0,X1,6)){const xx=sx(t);x.fillText(fn(t),xx,mt+ph+6);}
  x.fillText('Hours since first step shown',ml+pw/2,mt+ph+21);
  x.save();x.translate(11,mt+ph/2);x.rotate(-Math.PI/2);x.textBaseline='middle';x.fillText(ylab,0,0);x.restore();
  const cxp=sx(xs[cur]);x.strokeStyle=PAL.ink2;x.setLineDash([4,4]);x.beginPath();x.moveTo(cxp,mt);x.lineTo(cxp,mt+ph);x.stroke();x.setLineDash([]);
  x.strokeStyle=PAL.e1;x.lineWidth=2;x.beginPath();let pen=false;
  ys.forEach((v,i)=>{if(!Number.isFinite(v)){pen=false;return;}const X=sx(xs[i]),Y=sy(v);pen?x.lineTo(X,Y):x.moveTo(X,Y);pen=true;});x.stroke();
  ys.forEach((v,i)=>{if(!Number.isFinite(v))return;x.beginPath();x.arc(sx(xs[i]),sy(v),i===cur?5:3,0,7);x.fillStyle=i===cur?PAL.ink:PAL.e1;x.fill();});
}

/* ---------- basemaps ---------- */
const BM={osm:{url:'https://tile.openstreetmap.org/{z}/{x}/{y}.png',max:19,att:'© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'},
  linz_aerial:{url:'https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api={key}',max:22,att:'© <a href="https://www.linz.govt.nz/">Toitū Te Whenua LINZ</a> CC BY 4.0',key:'linz'},
  linz_topo:{url:'https://basemaps.linz.govt.nz/v1/tiles/topo-raster/WebMercatorQuad/{z}/{x}/{y}.webp?api={key}',max:18,att:'© <a href="https://www.linz.govt.nz/">Toitū Te Whenua LINZ</a> CC BY 4.0',key:'linz'},
  g_satellite:{google:'satellite',max:22,key:'google'},g_hybrid:{google:'hybrid',max:22,key:'google'},g_roadmap:{google:'roadmap',max:22,key:'google'}};
const KEYLAB={linz:'LINZ key',google:'Google Maps key'};
const GS={};
async function googleSession(type){
  const key=S.keys.google,id=type+'|'+key;if(GS[id]&&GS[id].exp>Date.now()/1000+300)return GS[id].s;
  const body={mapType:type==='hybrid'?'satellite':type,language:'en',region:M.googleRegion||'NZ'};if(type==='hybrid')body.layerTypes=['layerRoadmap'];
  const r=await fetch('https://tile.googleapis.com/v1/createSession?key='+encodeURIComponent(key),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok)throw new Error('Google refused the key ('+r.status+'): enable the Map Tiles API for it in Google Cloud.');
  const j=await r.json();GS[id]={s:j.session,exp:+j.expiry};return j.session;
}
let tiles=null,bmTok=0;
async function setBasemap(){
  const my=++bmTok;if(tiles){tiles.remove();tiles=null;}$('bmnote').textContent='';
  const b=BM[S.basemap];$('keywrap').classList.toggle('hide',!b?.key);if(b?.key){$('keylab').textContent=KEYLAB[b.key];$('keyin').value=S.keys[b.key]||'';}
  store.set('bgv_bm',S.basemap);
  if(!b)return;
  if(b.key&&!S.keys[b.key]){$('bmnote').textContent=b.key==='linz'?'LINZ tiles need a free API key: sign in at basemaps.linz.govt.nz, copy the key from an XYZ link (the text after api=) and paste it here.':'Google tiles need a Google Cloud API key with the Map Tiles API enabled. Paste it here; it stays in this browser.';return;}
  let url=b.url,att=b.att;
  if(b.google){try{const s=await googleSession(b.google);if(my!==bmTok)return;url=`https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}?session=${s}&key=${encodeURIComponent(S.keys.google)}`;att='Map data © Google';}
    catch(e){$('bmnote').textContent=hosted?'Google tiles can’t load in this hosted preview (outside requests are blocked here). Open the downloaded HTML file instead.':e.message;return;}}
  tiles=L.tileLayer(url,{key:S.keys[b.key]||'',maxNativeZoom:b.max,maxZoom:22,attribution:att}).addTo(map);
  let ok=0,err=0;tiles.on('tileload',()=>{ok++;$('bmnote').textContent='';});
  tiles.on('tileerror',()=>{err++;if(!ok&&err>3)$('bmnote').textContent=hosted?'Basemap tiles can’t load in this hosted preview (outside images are blocked here). Download the HTML and open it locally to see the basemap.':(b.key?'Tiles failed to load: check the API key and your internet connection.':'Basemap tiles failed to load: check your internet connection.');});
}

/* ---------- controls ---------- */
function staticText(){
  const levs=M.levlist.map(l=>`${M.levels[l].dx} m`).join(', ');
  $('kick').textContent=`${M.title} · BG_Flood output · ${M.crs}`;
  $('subline').textContent=`${M.varorder.length} variables over ${NS} time step${NS>1?'s':''}`+(NS<M.ntFile?` (of ${M.ntFile} in the file)`:'')+`, ${M.tlabels[0]}${NS>1?' to '+M.tlabels[NS-1]:''}. Adaptive mesh with cell sizes ${levs}.`+(M.region?' Cropped to a sub-region.':'')+(M.maxLevel!=null?` Levels finer than ${M.maxLevel} (${M.levels[M.maxLevel].dx} m) are averaged into ${M.levels[M.maxLevel].dx} m blocks.`:'');
  $('howto').textContent=`How to read this. BG_Flood stores each variable once per refinement level (suffix _P n / _N n); every location belongs to the finest level whose 16 × 16-cell block is active there, and that is what is drawn. Cells where the water depth (h, or hmax for “max” variables) is below the dry threshold are hidden when “Hide dry cells” is ticked. Values are stored to 16-bit precision (about 1/65 000 of each field’s range). The CRS ${M.crsSource}: ${M.crs}.`+(M.proj4?' Blocks are reprojected to Web Mercator in the browser to sit on the basemap.':' The CRS could not be converted, so no basemap is available and coordinates are shown in model units.');
  $('srcnote').textContent=`Built ${M.built} from ${M.file} with bg_html.py.`;
  $('var').innerHTML=[...M.varorder,LEV].map(v=>`<option value="${v}">${v===LEV?'Refinement level':`${v}: ${M.vars[v].long}${M.vars[v].units?' ('+M.vars[v].units+')':''}`}</option>`).join('');
  $('var').value=S.v;$('cmap').value=S.cmap;$('eps').value=S.eps;
  $('tslider').max=NS-1;if(NS<2)$('player').classList.add('hide');
  if(!M.proj4){$('basemap').disabled=true;$('bmnote').textContent='Basemaps are off: the model CRS is unknown.';}
  $('basemap').value=S.basemap;
}
$('var').onchange=e=>{S.v=e.target.value;if(!S.cmapUser&&S.v!==LEV){S.cmap=M.vars[S.v].cmap;$('cmap').value=S.cmap;}render();};
$('cmap').onchange=e=>{S.cmap=e.target.value;S.cmapUser=true;render();};
$('rev').onchange=e=>{S.rev=e.target.checked;render();};
const setR=()=>{const lo=parseFloat($('rmin').value),hi=parseFloat($('rmax').value);if(isFinite(lo)&&isFinite(hi)&&hi>lo){S.range[S.v]=[lo,hi];render();}};
$('rmin').onchange=setR;$('rmax').onchange=setR;
$('rreset').onclick=()=>{delete S.range[S.v];render();};
$('rfit').onclick=()=>{const ms=meshAt(S.s);if(!ms)return;const a=[],n=ms.nb*BB,stp=Math.max(1,Math.floor(n/200000));
  for(let ci=0;ci<n;ci+=stp){const v=maskedVal(S.v,S.s,ci);if(Number.isFinite(v))a.push(v);}
  if(!a.length)return;a.sort((p,q)=>p-q);let lo=a[Math.floor(a.length*0.02)],hi=a[Math.min(a.length-1,Math.floor(a.length*0.98))];
  if(hi<=lo){lo=a[0];hi=a[a.length-1]+(a[a.length-1]===a[0]?1:0);}if(M.vars[S.v].cmap==='rdbu'&&lo<0&&hi>0){const m=Math.max(-lo,hi);lo=-m;hi=m;}
  S.range[S.v]=[lo,hi];render();};
$('meshon').onchange=e=>{S.mesh=e.target.checked;refreshLayer();};
$('mask').onchange=e=>{S.mask=e.target.checked;render();};
$('eps').onchange=e=>{const v=parseFloat(e.target.value);if(isFinite(v)&&v>=0){S.eps=v;render();}};
$('opac').oninput=e=>{S.opac=+e.target.value;resLayer.setOpacity(S.opac);};
$('basemap').onchange=e=>{S.basemap=e.target.value;setBasemap();};
$('keyin').onchange=e=>{const b=BM[S.basemap];if(!b?.key)return;S.keys[b.key]=e.target.value.trim();store.set('bgv_'+b.key,S.keys[b.key]);setBasemap();};
$('zoomseg').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;const ms=meshAt(S.s);if(ms)map.fitBounds(b.dataset.z==='fine'?ms.fine:ms.all);});
$('tslider').oninput=e=>{S.s=+e.target.value;render();};
$('prev').onclick=()=>{S.s=(S.s-1+NS)%NS;render();};
$('next').onclick=()=>{S.s=(S.s+1)%NS;render();};
$('speed').onchange=e=>{S.interval=+e.target.value;};
$('play').onclick=()=>{S.playing=!S.playing;$('play').textContent=S.playing?'❚❚':'▶';$('play').setAttribute('aria-label',S.playing?'Pause':'Play');if(S.playing)loop();};
async function loop(){while(S.playing){const t0=performance.now();S.s=(S.s+1)%NS;await render();await new Promise(r=>setTimeout(r,Math.max(30,S.interval-(performance.now()-t0))));}}
function rethemed(){readTheme();if(pmark)pmark.setStyle({color:dark?'#fff':'#000',fillColor:PAL.e1});render();}
new MutationObserver(rethemed).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',rethemed);
addEventListener('resize',()=>{if(S.probe)drawProbe();});

/* ---------- boot ---------- */
staticText();
(async()=>{
  try{
    await ensureStep(S.s,[S.v,depthVar(S.v),...kpiVars()]);
    map.fitBounds(meshAt(S.s).all);
    $('loading').classList.add('hide');
    await render();setBasemap();
    // warm the other steps in the background so playback is smooth
    for(let s=0;s<NS;s++)await ensureStep(s,[S.v,depthVar(S.v)]);
  }catch(err){$('loading').textContent='Could not load results: '+err.message;console.error(err);}
})();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
