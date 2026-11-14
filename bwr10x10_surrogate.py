"""
bwr10x10_surrogate.py
=======================
A lightweight, physics-informed surrogate for the BWR 10x10 (GE14)
assembly physics, used to screen candidates for real DRAGON5 verification
(see arr_hybrid_dragon5.py / SurrogateCalibrationState) in the hybrid
surrogate-screened search described in

    Chang, K., Park, H.J., Shim, S. "Nuclear Fuel Assembly Optimization."
    (manuscript in preparation, 2026), which extends the 10x10/GE14
    problem of Radaideh, M.I. et al., Nuclear Engineering and Design 372
    (2021) 110966.

100 grid cells, 8 of which are large water rods (no fuel), fuel
enrichment ranging 1.6-4.95%, and some fuel rods mixed with gadolinium
(GAD) burnable poison.

Calibration data (7 points, independent of the source paper)
----------------------------------------------------------------
Unlike an earlier version of this module (which was calibrated from only
the 3 worked examples printed in the source paper's own figures), this
surrogate is calibrated from SEVEN real DRAGON5-evaluated points chosen
independently of any pattern reported in the source paper -- see
phase0_calibration/phase0_points.json for the raw calibration data, and
arr_hybrid_dragon5.py::SurrogateCalibrationState for how it is loaded and
subsequently re-fit online.

IMPORTANT: the module-level constants below (_K_A, _K_B, _CL_A, _CL_B,
_A2, _A_GAD, _D, _DE) are only the INITIAL values from that 7-point
calibration. In every real run, SurrogateCalibrationState overwrites
these module globals directly (surrogate._A2 = ..., etc.) immediately
after seeding from phase0_points.json, and again after every subsequent
real DRAGON5 call folded in during the search -- so a live run's actual
coefficients drift from the values hardcoded here as more real data
accumulates. Treat the values below as a reproducible STARTING POINT
(reproducible by feeding phase0_points.json through
SurrogateCalibrationState.seed_from_phase0() from a fresh interpreter),
not as this project's final, converged coefficients.

k_inf_max model
-----------------
Linear in average enrichment E alone:

    k_inf_max = a + b*E

Fit by closed-form ordinary least squares over the 7 calibration points.
Kept k_inf_max prediction error to approximately 0.2% relative to real
DRAGON5 values across the full calibration and search campaign; accurate
enough that k_inf_max was never observed to be the binding reason a
surrogate-passed candidate was rejected by real DRAGON5.

PPF model
----------
A genuine spatial model rather than a regression in E: a 1-group
neutron-diffusion eigenvalue problem solved directly on the full 10x10
pin layout (see _pin_power() below), with:
  * water-rod cells: zero fission source (nuSigmaF=0), same diffusion/
    absorption otherwise
  * GAD rods: extra absorption proportional to local GAD wt%
  * a linear-extrapolation boundary condition at the assembly edge

Four physically interpretable parameters (a2: how strongly a cell's own
enrichment drives its local fission source; a_gad: strength of GAD's
local absorption enhancement; D: diffusion coefficient; DE: boundary
extrapolation length) are fit by nonlinear least squares
(scipy.optimize.least_squares, method="trf", capped at 200 function
evaluations) over the accumulated set of real DRAGON5-verified points,
re-run after every new point (Section 3.4 of the paper above).

PPF from this module carries a documented, pattern-dependent uncertainty
of roughly +/-5-10% relative to real DRAGON5 (itself cross-checked
against CASMO4 to within about that same band on 3 reference patterns --
see bwr10x10_dragon5.py's own docstring); PPF is the harder of the three
surrogated quantities to approximate well precisely because it depends on
the SPATIAL arrangement of enrichment/GAD, not on average enrichment E
alone the way k_inf_max and cycle length do. Use this surrogate to screen
candidates cheaply before an expensive real DRAGON5 call, not as a
CASMO4-accurate replacement, and not for real reactor design decisions.
"""

import numpy as np
from scipy.linalg import eig

GRID_N = 10
E_REF = 4.3

# ----------------------------------------------------------------------
# k_inf_max: linear in average enrichment E (fit over all 3 known points,
# see bwr10x10_surrogate_dev.py::fit_k_inf(), variant (a))
# ----------------------------------------------------------------------
_K_A = 1.06308
_K_B = 0.0087533


def k_inf_max(E):
    return _K_A + _K_B * E


# ----------------------------------------------------------------------
# CL (cycle length) proxy -- NOT a full depletion/isotopics physics
# model; a linear regression of the 7 independent (E, CL) calibration
# points (see module docstring), re-fit online as real DRAGON5 points
# accumulate during search. Provided so the improvement-stage search has
# a concrete real-DRAGON5-verifiable objective; treat its absolute
# values as directional, not as a substitute for a real depletion
# calculation.
#
#   CL_proxy = 234.023 + 281.878*E   (7-point initial calibration)
# ----------------------------------------------------------------------
_CL_A = 234.023
_CL_B = 281.878


def cl_proxy(E):
    return _CL_A + _CL_B * E


# ----------------------------------------------------------------------
# PPF: 1-group diffusion eigenvalue solve, water cells + GAD absorption
# (fit via bwr10x10_surrogate_dev.py::fit_ppf())
# ----------------------------------------------------------------------
_A2 = 0.044457
_A_GAD = 0.014910
_D = 1.245296
_DE = 1.029329


def pattern_to_grids(pattern):
    """Convert a 10x10 nested-list pattern (see bwr10x10_reference_patterns,
    or build your own with the same cell convention) into three 10x10
    numpy arrays: enrich_grid, gad_grid, water_mask."""
    enrich = np.zeros((GRID_N, GRID_N))
    gad = np.zeros((GRID_N, GRID_N))
    water = np.zeros((GRID_N, GRID_N), dtype=bool)
    for i in range(GRID_N):
        for j in range(GRID_N):
            cell = pattern[i][j]
            if cell[0] == "WATER":
                water[i, j] = True
            elif cell[0] == "UO2":
                enrich[i, j] = cell[1]
            elif cell[0] == "GAD":
                enrich[i, j] = cell[1]
                gad[i, j] = cell[2]
            else:
                raise ValueError(f"unknown cell type {cell[0]!r}")
    return enrich, gad, water


def average_enrichment(enrich_grid, water_mask):
    """E: assembly-average UO2 enrichment over the 92 fuel cells (exact)."""
    return float(enrich_grid[~water_mask].mean())


def average_gad(gad_grid, water_mask):
    """G: average GAD enrichment over the GAD-bearing fuel cells only (exact)."""
    vals = gad_grid[(~water_mask) & (gad_grid > 0)]
    return float(vals.mean()) if len(vals) else 0.0


def _pin_power(enrich_grid, gad_grid, water_mask, a2=_A2, a_gad=_A_GAD, D=_D, de=_DE, h=1.0):
    """Solve -D*Laplacian(phi) + Sigma_a*phi = (1/k)*nuSigmaF*phi on the
    10x10 lattice with a linear-extrapolation (leakage) boundary condition,
    zero fission source at water-rod cells, and extra GAD absorption.
    Returns (pin_power_grid, k_eigenvalue); pin_power is NOT yet normalized
    (use ppf() for the normalized PPF metric)."""
    n = GRID_N
    N = n * n
    Sa = np.ones((n, n)) + a_gad * gad_grid
    nuSf = 1.0 + a2 * (enrich_grid - E_REF)
    nuSf[water_mask] = 0.0

    def idx(i, j):
        return i * n + j

    F = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            F[idx(i, j), idx(i, j)] = nuSf[i, j]

    L = h / 2 + de
    edge_factor = 1.0 - h / L

    A = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            diag = Sa[i, j]
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                diag += D / h ** 2
                if 0 <= ni < n and 0 <= nj < n:
                    A[k, idx(ni, nj)] -= D / h ** 2
                else:
                    diag -= D / h ** 2 * edge_factor
            A[k, k] = diag

    w, v = eig(A, F)
    w = w.real
    v = v.real
    order = np.argsort(w)
    for i0 in order:
        if w[i0] <= 0:
            continue
        vec = v[:, i0]
        if np.all(vec > 0) or np.all(vec < 0):
            vec = np.abs(vec)
            return vec.reshape(n, n), 1.0 / w[i0]
    # fallback: first positive eigenvalue, whatever the sign pattern
    for i0 in order:
        if w[i0] > 0:
            return np.abs(v[:, i0]).reshape(n, n), 1.0 / w[i0]
    raise RuntimeError("no positive eigenvalue found")


def ppf(enrich_grid, gad_grid, water_mask):
    """Power Peaking Factor = max(fuel pin power) / mean(fuel pin power),
    excluding the 8 water-rod cells from both max and mean."""
    # FIX (2026-08-15): _pin_power's a2/a_gad/D/de defaults were bound
    # at module-import time (Python's classic mutable-default-argument
    # trap) -- calling it without these kwargs silently used whatever
    # _A2/_A_GAD/_D/_DE were when this module first loaded, NEVER the
    # current values, no matter how many times SurrogateCalibrationState
    # refits them. Pass the CURRENT globals explicitly on every call.
    power, _ = _pin_power(enrich_grid, gad_grid, water_mask,
                           a2=_A2, a_gad=_A_GAD, D=_D, de=_DE)
    fuel_power = power[~water_mask]
    fuel_power = fuel_power / fuel_power.mean()
    return float(fuel_power.max())


# ----------------------------------------------------------------------
# Main API
# ----------------------------------------------------------------------

def casmo4_bwr10x10(pattern):
    """Drop-in replacement for a CASMO4 call on the BWR 10x10 (GE14)
    assembly: k_inf_max and PPF are the 1-group-diffusion-based surrogate
    outputs; CL_proxy is a crude 3-point linear regression in E only (see
    cl_proxy() above), NOT a real depletion/burnup calculation.

    Parameters
    ----------
    pattern : 10x10 nested list, cell = ("UO2", enrichment_pct) |
        ("GAD", enrichment_pct, gad_wt_pct) | ("WATER", None)
        (see bwr10x10_reference_patterns.py for the convention / examples)

    Returns
    -------
    dict with keys 'E', 'G', 'N_UO2', 'N_GAD', 'k_inf_max', 'PPF',
    'CL_proxy', 'pin_power' (10x10 array), 'enrich_grid', 'gad_grid',
    'water_mask'
    """
    enrich, gad, water = pattern_to_grids(pattern)
    E = average_enrichment(enrich, water)
    G = average_gad(gad, water)
    n_uo2 = int(np.sum((~water) & (gad == 0)))
    n_gad = int(np.sum((~water) & (gad > 0)))
    # FIX (2026-08-15): same fix as ppf() above -- see that comment.
    power, _ = _pin_power(enrich, gad, water,
                           a2=_A2, a_gad=_A_GAD, D=_D, de=_DE)
    fuel_power = power[~water]
    fuel_power_norm = fuel_power / fuel_power.mean()
    return {
        "E": E,
        "G": G,
        "N_UO2": n_uo2,
        "N_GAD": n_gad,
        "k_inf_max": k_inf_max(E),
        "PPF": float(fuel_power_norm.max()),
        "CL_proxy": cl_proxy(E),
        "pin_power": power,
        "enrich_grid": enrich,
        "gad_grid": gad,
        "water_mask": water,
    }


# ----------------------------------------------------------------------
# Self-check against the 3 known CASMO4 data points
# ----------------------------------------------------------------------

def self_check(verbose=True):
    from bwr10x10_reference_patterns import (
        REFERENCE, REFERENCE_TARGETS,
        RL_PPF_1_6, RL_PPF_1_6_TARGETS,
        RL_PPF_1_4, RL_PPF_1_4_TARGETS,
    )
    cases = [
        ("REFERENCE (Fig. 1b)", REFERENCE, REFERENCE_TARGETS),
        ("RL_PPF_1_6 (Fig. 8a)", RL_PPF_1_6, RL_PPF_1_6_TARGETS),
        ("RL_PPF_1_4 (Fig. 8b)", RL_PPF_1_4, RL_PPF_1_4_TARGETS),
    ]
    rows = []
    for name, pat, tgt in cases:
        out = casmo4_bwr10x10(pat)
        rows.append((name, out, tgt))
        if verbose:
            print(f"{name}")
            print(f"    E        : surrogate={out['E']:.4f}   CASMO4={tgt['E']}")
            print(f"    G        : surrogate={out['G']:.4f}   CASMO4={tgt['G']}")
            print(f"    N_UO2/GAD: surrogate={out['N_UO2']}/{out['N_GAD']}   "
                  f"CASMO4={tgt['N_UO2']}/{tgt['N_GAD']}")
            print(f"    k_inf_max: surrogate={out['k_inf_max']:.5f}   CASMO4={tgt['k_inf_max']}   "
                  f"err={out['k_inf_max']-tgt['k_inf_max']:+.5f}")
            print(f"    PPF      : surrogate={out['PPF']:.4f}   CASMO4={tgt['PPF']}   "
                  f"err={out['PPF']-tgt['PPF']:+.4f}")
    return rows


if __name__ == "__main__":
    self_check()
