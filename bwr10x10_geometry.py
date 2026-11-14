"""
bwr10x10_geometry.py
======================
Shared geometry / decision-variable definitions for the BWR 10x10 (GE14)
optimization, used by both the ARR Phase-1 and Phase-2 drivers.

Per the user's choice, ALL 51 symmetry-unique rod positions are treated
as free variables (the paper's own Heuristics 1-4, which tie some
positions together and restrict which pin types are allowed at each, are
NOT applied here) -- every position may independently take any of the 13
pin types used anywhere in the paper's action space (Section 5.1):

    9 pure UO2 types  : 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.95 %
    4 UO2+GAD types   : 4.4%+7%GAD, 4.95%+7%GAD, 4.4%+8%GAD, 4.95%+8%GAD

POS51 (the 51 unique (row, col) grid cells, 0-indexed, lower triangle
including the diagonal, excluding the 4 water-rod cells that fall on
that side) is derived programmatically from the fixed water-rod geometry
already established in bwr10x10_reference_patterns.py, so it is
guaranteed consistent with the 3 calibration patterns and the surrogate.
"""

import numpy as np

from bwr10x10_surrogate import pattern_to_grids, GRID_N
from bwr10x10_reference_patterns import REFERENCE

N_POSITIONS = 51
N_TYPES = 13

# index -> (kind, enrichment_pct, gad_wt_pct)
PIN_TYPES = [
    ("UO2", 1.6, 0.0),
    ("UO2", 2.0, 0.0),
    ("UO2", 2.4, 0.0),
    ("UO2", 2.8, 0.0),
    ("UO2", 3.2, 0.0),
    ("UO2", 3.6, 0.0),
    ("UO2", 4.0, 0.0),
    ("UO2", 4.4, 0.0),
    ("UO2", 4.95, 0.0),
    ("GAD", 4.4, 7.0),
    ("GAD", 4.95, 7.0),
    ("GAD", 4.4, 8.0),
    ("GAD", 4.95, 8.0),
]
assert len(PIN_TYPES) == N_TYPES

# water-rod geometry is fixed and identical across all 3 calibration
# patterns; derive the boolean mask once from REFERENCE.
_, _, WATER_MASK = pattern_to_grids(REFERENCE)

POS51 = [(i, j) for i in range(GRID_N) for j in range(i + 1) if not WATER_MASK[i, j]]
assert len(POS51) == N_POSITIONS, f"expected 51 unique positions, got {len(POS51)}"


def build_grid(type_idx):
    """Expand a length-51 vector of pin-type indices (0..12, order per
    POS51) into the full 10x10 pattern (diagonally mirrored, water cells
    filled in), in the ("UO2", e) / ("GAD", e, g) / ("WATER", None)
    convention used by bwr10x10_surrogate.py."""
    type_idx = np.asarray(type_idx, dtype=int)
    if type_idx.shape[0] != N_POSITIONS:
        raise ValueError(f"expected {N_POSITIONS} values, got {type_idx.shape[0]}")

    grid_type = np.full((GRID_N, GRID_N), -1, dtype=int)
    for (i, j), t in zip(POS51, type_idx):
        grid_type[i, j] = t
        grid_type[j, i] = t

    pattern = []
    for i in range(GRID_N):
        row = []
        for j in range(GRID_N):
            if WATER_MASK[i, j]:
                row.append(("WATER", None))
            else:
                kind, e, g = PIN_TYPES[grid_type[i, j]]
                row.append(("UO2", e) if kind == "UO2" else ("GAD", e, g))
        pattern.append(row)
    return pattern
