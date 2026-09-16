# Implicit Barotropic Time Integration

## Motivation
Implementing the Implicit Barotropic Time Integration is a step toward have a non-hydrostatic multi-layer solver in BG_Flood. It could be useful in the simulation of strongly barotropic problem in a single-layer model (e.g. tide model) but it is not clear if it is useful for Rain-on-grid problems.

## Overview

This module implements a time-implicit (theta-method) integration of the
barotropic (gravity-wave) term in the single-layer shallow-water solver,
ported from Basilisk's `src/layered/implicit.h`. Its purpose is to remove
the gravity-wave CFL restriction from the timestep, allowing `dt` to be
governed by advection alone rather than by the much stiffer
$\sqrt{gh}$-based wave-speed limit that constrains a fully explicit
scheme.

The pressure-gradient (barotropic) term is blended between the old and
new timestep using a weight $\theta_H$:

$$\frac{\eta^{n+1}-\eta^n}{\Delta t} = -\nabla\cdot\big[\theta_H (hu)^{n+1} + (1-\theta_H)(hu)^n\big]$$
$$\frac{(hu)^{n+1}-(hu)^n}{\Delta t} = -g\,h^{n+\theta}\big[\theta_H\nabla\eta^{n+1} + (1-\theta_H)\nabla\eta^n\big]$$

- $\theta_H = 0.5$: Crank-Nicolson — second-order accurate, non-dissipative
  for the linear gravity-wave equation.
- $\theta_H = 1.0$: backward Euler — first-order, actively damps all
  modes (L-stable).
- $\theta_H > 1.0$ : over-implicit,  trades additional accuracy for extra damping margin against the
  nonlinear/advective coupling that Crank-Nicolson alone does not control (not recommended).

Substituting the momentum equation into the continuity equation eliminates
the flux and reduces the whole system to a single scalar elliptic
(Helmholtz) equation for $\eta^{n+1}$:

$$\eta^{n+1} - \frac{G}{\Delta^2}\sum_{\text{faces}}\alpha_{\text{face}}(\eta_{\text{nb}}-\eta^{n+1}) = \text{rhs}_\eta$$

with $\alpha_{\text{face}} = -(\theta_H\Delta t)^2 h_f$ and $\text{rhs}_\eta$
built from the old $\eta$ and a predictor flux. This operator is
symmetric positive-definite, which is what makes a Conjugate-Gradient
solve viable in place of Basilisk's multigrid (see below).

## Key differences from Basilisk : PCG replaces multigrid
Basilisk solves the Helmholtz equation with `mg_solve`, using
`relax_hydro`/`residual_hydro` as the smoother/residual on its adaptive
quadtree. BG_Flood's block-based GPU memory layout (fixed-size blocks,
flat arrays, a compacted active-block list) does not map cleanly onto a
recursive coarse-grid hierarchy, so the solver was replaced with a
**Jacobi-preconditioned Conjugate Gradient** solve on the same discrete
operator. Both solve the identical linear system; they differ only in
convergence behavior: The Multigrid's Gauss-Seidel smoothing damps high-frequency error strongly
  by construction. Plain Jacobi-PCG has no such spectral bias and
  converges markedly slower in regions with sharp coefficient jumps
  ( wet/dry fronts, where $h_f\to 0$ over a few cells ?).


## Usage
The implicit solver is only available for engine 5. To use this functionality you will need:
```
 engine = 5;
 implicit = true
 ```
Ideally you will need to adjust a few additional parameters described below. 

### Time steps
BG_Flood doesn't control timestep explicitly but you could control it via the `CFL`~parameter. When using `implicit = true` the CFL parameter only applies to CFL_H (barotropic CFL) with the momentun CFL remaining at 0.5. By default, the CFL remains at `0.5` but in most cases you will want to run with CFL> 1.0. e.g.:

`CFL = 5.0`

In Basilisk is seem that the user specifies `dt`. but that is not very easy to go down that route. Maybe in the future.

Beware when using large CFL as your model will leap foreward a bit too much. Experimentation on the problem are needed.

### Controlling the solution
You can control the solution via the pressure gradient weight $\theta_H$, the maximum residual tolerance and the maximum number of iteration.
by default:
```
mg_max_iter = 100
mg_tol = 0.00001
thetaH = 0.5;
```

`thetaH` should be the preferred way to control the solution. Increasing max iteration and tolerance can significantly increase runtime.
A warning is issued if max iteration is reached for any step.

### Diagnostics
Residuals and implicit solver variables are available for output:
`p_imp,r_imp,z_imp,Ap_imp,gx_imp,gy_imp,alphax_imp,alphay_imp,eta_r,diagInv,rhs_eta,su_imp,sv_imp`  

However most of these variables above are only useful for debugging. For understanding residuals, focus on:
`eta_r, rhs_eta `

### More testing will be completed as part of the development of the multi-layering. 
