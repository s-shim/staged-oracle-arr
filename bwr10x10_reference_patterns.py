"""
bwr10x10_reference_patterns.py
================================
Full 10x10 rod-by-rod patterns for the three concrete BWR 10x10 (GE14)
examples given in Radaideh et al. 2021 (Nuclear Engineering and Design
372, 110966), transcribed directly from the paper's figures:

  1. REFERENCE  -- Fig. 1(b), Section 2. The as-built/reference GE14
     design. Extracted via automated pixel-color classification against
     the figure's own legend swatches (11 fuel-type colors + water),
     using a least-squares-fit grid model (100 cells, mean centroid
     residual ~3px). Cross-validated: N_UO2=74, N_GAD=18, N_water=8 all
     matched the paper's stated counts exactly, and computed E=4.311%
     / G=7.722% matched the paper's stated E=4.305% / G=7.72% to within
     figure-legend rounding.
  2. RL_PPF_1_6 -- Fig. 8(a), Section 5.3. Best RL(PPO)-found pattern
     under the PPF<1.6 constraint. This figure prints every cell's
     numeric value directly (not just color), so it was transcribed by
     direct reading, not pixel classification.
  3. RL_PPF_1_4 -- Fig. 8(b), Section 5.3. Best RL(PPO)-found pattern
     under the tighter PPF<1.4 constraint. Same direct-reading method.

Grid convention
----------------
Each pattern is a 10x10 nested list, row 1 = top row of the figure,
column 1 = leftmost column, matching the figures' own orientation.
Each cell is one of:
  ("UO2", enrichment_pct)                    -- pure UO2 rod
  ("GAD", enrichment_pct, gad_wt_pct)        -- UO2+GAD rod
  ("WATER", None)                            -- large water rod cell
    (each large water rod occupies a 2x2 block of 4 grid cells)

These are the FULL physical 10x10 patterns (all 100 cells, not just the
51 symmetry-unique positions), i.e. exactly what a physics code/surrogate
would need as input. All three patterns share the same water-rod
geometry (fixed, per the paper): 2x2 blocks at rows 4-5/cols 6-7 and
rows 6-7/cols 4-5 (1-indexed).

Known target outputs (from the paper, for calibration/validation)
-------------------------------------------------------------------
REFERENCE:   k_inf_max=1.09814, PPF=1.62,  CL=1450 days, E=4.305%, G=7.72%,  N_UO2=74, N_GAD=18
RL_PPF_1_6:  k_inf_max=1.10151, PPF=1.57,  CL=1478 days, E=4.355%, G=7.875%, N_UO2=76, N_GAD=16
RL_PPF_1_4:  k_inf_max=1.10446, PPF=1.367, CL=1462 days, E=4.350%, G=7.333%, N_UO2=74, N_GAD=18
"""

W = ("WATER", None)


def U(e):
    return ("UO2", e)


def G(e, gad):
    return ("GAD", e, gad)


# ---------------------------------------------------------------------
# 1. REFERENCE GE14 design (Fig. 1b) -- extracted via pixel-color match
# ---------------------------------------------------------------------
REFERENCE = [
    [U(1.6), U(2.8), U(3.2), U(3.95), U(3.95), U(3.95), U(3.95), U(3.95), U(3.95), U(2.8)],
    [U(2.8), U(2.8), U(3.2), U(3.95), U(3.6), G(3.95, 8.0), U(3.95), G(4.4, 8.0), U(3.95), U(3.95)],
    [U(3.2), U(3.2), G(4.4, 3.0), U(4.4), G(4.9, 8.0), U(4.4), G(4.4, 8.0), U(4.4), G(4.4, 8.0), U(4.9)],
    [U(3.95), U(3.95), U(4.4), G(4.9, 8.0), U(3.95), W, W, U(4.9), U(4.9), U(4.9)],
    [U(3.95), U(3.6), G(4.9, 8.0), U(3.95), U(4.9), W, W, U(4.9), U(4.9), U(4.9)],
    [U(3.95), G(3.95, 8.0), U(4.4), W, W, U(4.9), U(4.9), U(4.9), G(4.9, 8.0), U(4.9)],
    [U(3.95), U(3.95), G(4.4, 8.0), W, W, U(4.9), U(4.9), G(4.9, 8.0), U(4.9), U(4.9)],
    [U(3.95), G(4.4, 8.0), U(4.4), U(4.9), U(4.9), U(4.9), G(4.9, 8.0), U(4.9), G(4.9, 8.0), U(4.9)],
    [U(3.95), U(3.95), G(4.4, 8.0), U(4.9), U(4.9), G(4.9, 8.0), U(4.9), G(4.9, 8.0), U(4.9), U(4.9)],
    [U(2.8), U(3.95), U(4.9), U(4.9), U(4.9), U(4.9), U(4.9), U(4.9), U(4.9), U(3.2)],
]
REFERENCE_TARGETS = dict(k_inf_max=1.09814, PPF=1.62, CL=1450, E=4.305, G=7.72, N_UO2=74, N_GAD=18)


# ---------------------------------------------------------------------
# 2. RL best pattern, PPF<1.6 (Fig. 8a) -- read directly (values printed)
# ---------------------------------------------------------------------
RL_PPF_1_6 = [
    [U(3.2), U(4.4), U(4.95), U(4.95), U(4.95), U(4.95), U(4.95), U(4.95), U(4.4), U(3.2)],
    [U(4.4), U(4.95), G(4.4, 8.0), U(4.95), U(3.6), U(3.6), U(3.6), G(4.4, 8.0), U(3.6), U(4.4)],
    [U(4.95), G(4.4, 8.0), U(4.95), U(3.6), U(3.6), G(4.4, 8.0), U(3.6), U(4.95), U(4.0), U(4.95)],
    [U(4.95), U(4.95), U(3.6), U(3.6), U(3.6), W, W, U(3.6), U(3.6), U(4.95)],
    [U(4.95), U(3.6), U(3.6), U(3.6), U(3.6), W, W, G(4.4, 8.0), U(4.95), U(4.95)],
    [U(4.95), U(3.6), G(4.4, 8.0), W, W, U(3.6), G(4.4, 8.0), U(3.6), G(4.4, 8.0), U(4.95)],
    [U(4.95), U(3.6), U(3.6), W, W, G(4.4, 8.0), U(4.95), G(4.4, 8.0), U(4.95), U(4.95)],
    [U(4.95), G(4.4, 8.0), U(4.95), U(3.6), G(4.4, 8.0), U(3.6), G(4.4, 8.0), U(4.95), G(4.95, 7.0), U(4.95)],
    [U(4.4), U(3.6), U(4.0), U(3.6), U(4.95), G(4.4, 8.0), U(4.95), G(4.95, 7.0), U(4.95), U(4.4)],
    [U(3.2), U(4.4), U(4.95), U(4.95), U(4.95), U(4.95), U(4.95), U(4.95), U(4.4), U(3.2)],
]
RL_PPF_1_6_TARGETS = dict(k_inf_max=1.10151, PPF=1.57, CL=1478, E=4.355, G=7.875, N_UO2=76, N_GAD=16)


# ---------------------------------------------------------------------
# 3. RL best pattern, PPF<1.4 (Fig. 8b) -- read directly (values printed)
# ---------------------------------------------------------------------
RL_PPF_1_4 = [
    [U(2.4), U(3.2), U(4.0), U(4.0), U(4.0), U(4.0), U(4.0), U(4.0), U(3.2), U(2.4)],
    [U(3.2), U(4.95), G(4.4, 8.0), U(4.95), U(4.95), U(4.95), U(4.95), G(4.95, 7.0), U(4.0), U(3.2)],
    [U(4.0), G(4.4, 8.0), U(4.95), U(4.95), U(4.95), G(4.95, 7.0), U(4.95), U(4.95), U(4.4), U(4.0)],
    [U(4.0), U(4.95), U(4.95), U(4.95), U(4.95), W, W, U(4.95), G(4.4, 8.0), U(4.0)],
    [U(4.0), U(4.95), U(4.95), U(4.95), U(4.95), W, W, G(4.95, 7.0), U(4.0), U(4.0)],
    [U(4.0), U(4.95), G(4.95, 7.0), W, W, U(4.95), G(4.95, 7.0), U(4.95), G(4.4, 8.0), U(4.0)],
    [U(4.0), U(4.95), U(4.95), W, W, G(4.95, 7.0), U(4.95), G(4.95, 7.0), U(4.95), U(4.0)],
    [U(4.0), G(4.95, 7.0), U(4.95), U(4.95), G(4.95, 7.0), U(4.95), G(4.95, 7.0), U(4.95), G(4.95, 7.0), U(4.0)],
    [U(3.2), U(4.0), U(4.4), G(4.4, 8.0), U(4.0), G(4.4, 8.0), U(4.95), G(4.95, 7.0), U(4.95), U(3.2)],
    [U(2.4), U(3.2), U(4.0), U(4.0), U(4.0), U(4.0), U(4.0), U(4.0), U(3.2), U(2.4)],
]
RL_PPF_1_4_TARGETS = dict(k_inf_max=1.10446, PPF=1.367, CL=1462, E=4.350, G=7.333, N_UO2=74, N_GAD=18)


def _validate(pattern, targets, name):
    flat = [c for row in pattern for c in row]
    n_water = sum(1 for c in flat if c[0] == "WATER")
    n_gad = sum(1 for c in flat if c[0] == "GAD")
    n_uo2 = sum(1 for c in flat if c[0] == "UO2")
    all_enrich = [c[1] for c in flat if c[0] in ("UO2", "GAD")]
    gad_enrich = [c[2] for c in flat if c[0] == "GAD"]
    E = sum(all_enrich) / len(all_enrich)
    Gavg = sum(gad_enrich) / len(gad_enrich) if gad_enrich else 0.0
    print(f"{name}: N_water={n_water} (want 8), N_UO2={n_uo2} (want {targets['N_UO2']}), "
          f"N_GAD={n_gad} (want {targets['N_GAD']}), E={E:.3f}% (want {targets['E']}%), "
          f"G={Gavg:.3f}% (want {targets['G']}%)")


if __name__ == "__main__":
    _validate(REFERENCE, REFERENCE_TARGETS, "REFERENCE")
    _validate(RL_PPF_1_6, RL_PPF_1_6_TARGETS, "RL_PPF_1_6")
    _validate(RL_PPF_1_4, RL_PPF_1_4_TARGETS, "RL_PPF_1_4")
