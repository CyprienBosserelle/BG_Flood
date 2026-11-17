#!/usr/bin/env python3
"""
Analysis of the tsunami boundary-reflection benchmark (see make_case.py).

For every test scenario the reflected signal is isolated by subtracting the
long-channel benchmark:  eta_refl = eta_test - eta_benchmark.

Two reflection coefficients are reported:

* R_amp  (gauges): max|eta_refl| / max|eta_incident| at the gauges located
  halfway between the centre and the boundary (GL5, GR5). Incident and
  reflected pulses are well separated in time there. The sign column tells
  whether the reflection is in phase (wall, +) or inverted (Dirichlet, -).
* R_E (energy): once the incident pulses have fully left the short channel,
  any wave energy left inside it is reflected energy.
      R_E = E_refl / E_incident,  R_amp_E = sqrt(R_E)
  with E = int( g*eta^2/2 + h*u^2/2 ) dx on the channel centre-line, computed
  on the test-minus-benchmark fields, and E_incident the benchmark energy before
  the pulses reach the boundary. This also captures long-period residuals
  (e.g. a set-down/set-up left behind by the boundary) that a peak value misses.

Outputs (in ./results/): reflection.csv, reflection.md and figures.

Requires: numpy, netCDF4, matplotlib
"""
import json
import os

import numpy as np
from netCDF4 import Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results")
BENCH = "benchmark_long"

# Fixed categorical order (validated colour-blind safe palette); benchmark is neutral ink.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BNDNAME = {0: "wall", 1: "Neumann", 2: "Dirichlet", 3: "absorbing"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.5, "figure.dpi": 150,
})


def read_gauge(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1], d[:, 3]  # time, zs, u


def read_map(path):
    with Dataset(path) as nc:
        x = nc["xx_P0"][:].data
        y = nc["yy_P0"][:].data
        t = nc["time"][:].data
        j = len(y) // 2  # channel centre-line (away from the dry land rows)
        zs = nc["zs_P0"][:, j, :].data
        u = nc["u_P0"][:, j, :].data
        h = nc["h_P0"][:, j, :].data
    return t, x, zs, u, h


def label(name, meta):
    sc = meta["scenarios"][name]
    extra = ", ".join(f"{k}={v:g}" for k, v in sc["extra"].items())
    return f"{name} (type {sc['bnd']}{', ' + extra if extra else ''})"


def main():
    meta = json.load(open(os.path.join(RUNS, "setup.json")))
    os.makedirs(OUT, exist_ok=True)
    g, h0, tw, L = meta["g"], meta["h0"], meta["twave"], meta["halflen"]
    nw = meta["nwave"]
    tests = [s for s in meta["scenarios"] if s != BENCH]
    color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(tests)}

    # ---------------- benchmark --------------------------------------------
    tb, xb, zb_, ub, hb = read_map(os.path.join(RUNS, BENCH, "output.nc"))
    inwin = np.abs(xb) <= L
    xw = xb[inwin]
    dxw = np.gradient(xw)

    def energy(eta, u):
        return np.sum((0.5 * g * eta**2 + 0.5 * h0 * u**2) * dxw, axis=-1)

    Eb = energy(zb_[:, inwin], ub[:, inwin])
    # incident energy: benchmark energy while the pulses are still well inside
    # the short-channel extent (between 2 and nw-2 wavelengths of travel)
    sel = (tb > 2 * tw) & (tb < (nw - 2) * tw)
    Einc = Eb[sel].mean()
    # window when the incident pulse has fully left (pulse half-extent = 0.5 wavelength,
    # plus 1 wavelength margin) and before the reflected pulses reach the other side
    rwin = (tb > (nw + 1.5) * tw) & (tb < (2 * nw - 1) * tw)

    gb = {k: read_gauge(os.path.join(RUNS, BENCH, f"{k}.txt")) for k in meta["gauges"]}

    rows = []
    maps = {}
    for s in tests:
        t, x, zs, u, h = read_map(os.path.join(RUNS, s, "output.nc"))
        nt = min(len(t), len(tb))
        assert np.allclose(t[:nt], tb[:nt]), "map output times differ from the benchmark"
        dz = zs[:nt] - zb_[:nt, inwin]
        du = u[:nt] - ub[:nt, inwin]
        maps[s] = (t[:nt], x, zs[:nt], dz)
        Er = energy(dz, du)
        RE = np.mean(Er[rwin[:nt]]) / Einc
        # mean water-level offset left in the channel (long-period residual)
        offset = np.mean(dz[rwin[:nt]])

        ramp = {}
        for k in ("GL5", "GR5"):
            tt, z, _ = read_gauge(os.path.join(RUNS, s, f"{k}.txt"))
            tg, zg, _ = gb[k]
            zgi = np.interp(tt, tg, zg)
            refl = z - zgi
            ainc = np.max(np.abs(zgi))
            i = np.argmax(np.abs(refl))
            ramp[k] = refl[i] / ainc
        rows.append(dict(scenario=s, bnd=meta["scenarios"][s]["bnd"],
                         extra=";".join(f"{k}={v:g}" for k, v in meta["scenarios"][s]["extra"].items()),
                         R_amp_GL5=ramp["GL5"], R_amp_GR5=ramp["GR5"],
                         R_E=RE, R_amp_E=np.sqrt(RE), offset=offset, Er=Er, t=t[:nt]))

    # ---------------- tables -------------------------------------------------
    hdr = ["scenario", "bnd type", "extra", "R_amp GL5", "R_amp GR5", "R_E (energy)",
           "sqrt(R_E)", "mean offset [m]"]
    with open(os.path.join(OUT, "reflection.csv"), "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(f"{r['scenario']},{r['bnd']},{r['extra']},{r['R_amp_GL5']:.4f},{r['R_amp_GR5']:.4f},"
                    f"{r['R_E']:.5f},{r['R_amp_E']:.4f},{r['offset']:.2e}\n")
    with open(os.path.join(OUT, "reflection.md"), "w") as f:
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write(f"| {r['scenario']} | {r['bnd']} ({BNDNAME[r['bnd']]}) | {r['extra']} | "
                    f"{r['R_amp_GL5']:+.3f} | {r['R_amp_GR5']:+.3f} | {r['R_E']:.4f} | "
                    f"{r['R_amp_E']:.3f} | {r['offset']:+.1e} |\n")
    print(open(os.path.join(OUT, "reflection.md")).read())

    # ---------------- figure 1: gauge GR5 -------------------------------------
    fig, axs = plt.subplots(len(tests), 1, figsize=(7.5, 1.35 * len(tests) + 0.6),
                            sharex=True, sharey=True)
    tg, zg, _ = gb["GR5"]
    for ax, s in zip(axs, tests):
        tt, z, _ = read_gauge(os.path.join(RUNS, s, "GR5.txt"))
        ax.plot(tg / tw, zg, color=INK2, lw=1.0, label="benchmark (long channel)")
        ax.plot(tt / tw, z, color=color[s], label=s)
        ax.plot(tt / tw, z - np.interp(tt, tg, zg), color=color[s], lw=1.0, ls="--",
                label="reflected = test - benchmark")
        ax.text(0.01, 0.92, label(s, meta), transform=ax.transAxes, va="top", color=INK)
        ax.axvline(3 * nw / 2, color=GRID, lw=0.8)
    axs[0].legend(loc="upper right", ncol=3, fontsize=7)
    axs[-1].set_xlabel("time / (wavelength / c)")
    fig.supylabel("water level at gauge GR5 [m]", fontsize=9)
    fig.suptitle("Gauge halfway to the right boundary: incident pulse at t~5, reflection at t~15",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gauge_GR5.png"))
    plt.close(fig)

    # ---------------- figure 2: x-t diagrams of the reflected signal -------------
    vmax = 0.05 * meta["amp"]
    fig, axs = plt.subplots(1, len(tests) + 1, figsize=(2.0 * (len(tests) + 1), 3.6),
                            sharey=True)
    ext = lambda xx, tt: [xx[0] / meta["wavelength"], xx[-1] / meta["wavelength"], tt[0] / tw, tt[-1] / tw]
    axs[0].imshow(zb_[:, inwin], origin="lower", aspect="auto", cmap="RdBu_r",
                  vmin=-0.5 * meta["amp"], vmax=0.5 * meta["amp"], extent=ext(xw, tb))
    axs[0].set_title("benchmark eta", fontsize=8)
    axs[0].set_ylabel("time / (wavelength / c)")
    for ax, s in zip(axs[1:], tests):
        t, x, zs, dz = maps[s]
        im = ax.imshow(dz, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=ext(xw, t))
        ax.set_title(f"{s}\ntest - benchmark", fontsize=8)
    for ax in axs:
        ax.grid(False)
        ax.set_xlabel("x / wavelength")
    fig.colorbar(im, ax=axs, shrink=0.8, label=f"reflected eta [m] (colour scale +/-{vmax:g} m = 5% of amp)")
    fig.savefig(os.path.join(OUT, "xt_reflection.png"), bbox_inches="tight")
    plt.close(fig)

    # ---------------- figure 3: reflected energy vs time --------------------------
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for r in rows:
        ax.semilogy(r["t"] / tw, np.maximum(r["Er"] / Einc, 1e-12), color=color[r["scenario"]],
                    label=r["scenario"])
    ax.axvspan((nw + 1.5), (2 * nw - 1), color=GRID, alpha=0.5, lw=0)
    ax.text(nw + 1.6, 1.5, "averaging window", color=INK2, fontsize=7)
    ax.set_ylim(1e-8, 3)
    ax.set_xlabel("time / (wavelength / c)")
    ax.set_ylabel("E(test - benchmark) / E incident")
    ax.legend(ncol=2, fontsize=7, loc="lower right")
    ax.set_title("Energy of the reflected signal inside the short channel", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reflected_energy.png"))
    plt.close(fig)

    # ---------------- figure 4: summary bars -----------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    names = [r["scenario"] for r in rows]
    vals = [r["R_amp_E"] for r in rows]
    yy = np.arange(len(rows))[::-1]
    ax.barh(yy, vals, color=[color[n] for n in names], height=0.6)
    for y_, v in zip(yy, vals):
        ax.text(v + 0.01, y_, f"{100 * v:.1f}%", va="center", color=INK, fontsize=8)
    ax.set_yticks(yy, names)
    ax.set_xlim(0, 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("amplitude reflection coefficient  sqrt(E_refl / E_inc)")
    ax.set_title("Boundary reflection of a tsunami pulse", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reflection_summary.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
