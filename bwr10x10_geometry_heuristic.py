"""
bwr10x10_geometry_heuristic.py
=================================
Heuristic-reduced decision-variable geometry for the BWR 10x10 (GE14)
optimization -- Heuristics 1-4 of Radaideh et al. (2021), Section 5.1,
applied to shrink both the number of independent decision variables AND
the per-position choice set (unlike bwr10x10_geometry.py, which leaves
all 51 positions free with all 13 pin types each).

Deriving the heuristic groups geometrically
----------------------------------------------
Rather than re-reading rod index numbers off Fig. 1(b) (error-prone), the
four heuristic groups are derived directly from the grid geometry, and
cross-checked against the paper's own combinatorics (5*6*4=120 outer
frames, 32 interior positions):

  * Heuristic 1 (corners): the 3 corners of the lower-triangle (POS51)
    shape: (0,0), (9,0), (9,9).  5 choices: UO2 1.6/2.0/2.4/2.8/3.2%.
  * Heuristic 2 (near-corner edge rods): the boundary cells one step
    away from each corner, within the lower triangle: near (0,0) that is
    (1,0); near (9,0) that is (8,0) AND (9,1) (two cells, since (9,0) has
    two boundary directions both inside the lower triangle); near (9,9)
    that is (9,8).  4 cells total.  6 choices: UO2 2.8/3.2/3.6/4.0/4.4/4.95%.
  * Heuristic 3 (remaining edge rods): the rest of the left column
    (rows 2..7, col 0) and the rest of the bottom row (row 9, cols 2..7)
    -- 6+6 = 12 cells, ALL tied to ONE shared choice (matches the
    paper's 5*6*4=120 combinatorics, which requires H3 to be a single
    variable). 4 choices: UO2 3.6/4.0/4.4/4.95%.
  * Heuristic 4 (interior): every other POS51 cell (51-3-4-12 = 32 cells),
    each an INDEPENDENT variable. 8 choices: UO2 3.6/4.0/4.4/4.95% +
    GAD 4.4%+7%/4.95%+7%/4.4%+8%/4.95%+8% -- EXCEPT cells (4,4) and
    (5,5) (see Rule 3 note below), which get only the 4 UO2 choices.

Total independent decision variables = 1 (H1) + 1 (H2) + 1 (H3) + 32 (H4)
= 35, each mapped onto a row of the shared 13-slot global pin-type index
(see bwr10x10_geometry.PIN_TYPES) via a boolean allowed-type mask.

Rule 3 (paper's literal 2-position GAD-exclusion clause)
----------------------------------------------------------
Radaideh et al. (2021)'s Rule 3: "two GAD rods are not allowed to sit
next to each other... In addition, for the two rod positions 26 and 29,
which neighbor the large water rods, GAD rods are not allowed." The
paper's own diagonal rod-numbering (1-6-14-22-26-29-33-36-38-40) maps
these two labels to grid cells (4,4) and (5,5) 0-indexed -- geometrically
the two diagonal cells that touch BOTH water-rod blocks, verified this
project (2026-08-11 session) by cross-matching against
count_gad_violations()'s own reported violation coordinates on
REFERENCE/RL_PPF_1_6/RL_PPF_1_4 (all matched exactly). Confirmed all 3
of the paper's own patterns respect this at exactly these two cells.

The GAD-GAD-adjacency half of Rule 3 (no two GAD rods orthogonally
adjacent) is NOT enforced here -- it's a pairwise constraint between
whichever positions happen to be chosen, not a fixed per-position
choice-set restriction, so it can't be encoded via TYPE_MASK the way
this position-level restriction can. It would need a runtime constraint
check instead (out of scope for this change).
"""

import numpy as np

from bwr10x10_geometry import PIN_TYPES, N_TYPES, POS51
from bwr10x10_surrogate import GRID_N, pattern_to_grids
from bwr10x10_reference_patterns import REFERENCE

_, _, WATER_MASK = pattern_to_grids(REFERENCE)

# ---------------------------------------------------------------------
# geometric group derivation
# ---------------------------------------------------------------------
H1_CELLS = [(0, 0), (9, 0), (9, 9)]
H2_CELLS = [(1, 0), (8, 0), (9, 1), (9, 8)]
H3_CELLS = [(i, 0) for i in range(2, 8)] + [(9, j) for j in range(2, 8)]

_grouped = set(H1_CELLS) | set(H2_CELLS) | set(H3_CELLS)
H4_CELLS = [cell for cell in POS51 if cell not in _grouped]

assert len(H1_CELLS) == 3
assert len(H2_CELLS) == 4
assert len(H3_CELLS) == 12
assert len(H4_CELLS) == 32, f"expected 32 H4 cells, got {len(H4_CELLS)}"
assert len(H1_CELLS) + len(H2_CELLS) + len(H3_CELLS) + len(H4_CELLS) == 51

# ---------------------------------------------------------------------
# allowed pin-type subsets per heuristic (global type indices, see
# bwr10x10_geometry.PIN_TYPES: 0-8 = UO2 1.6..4.95%, 9-12 = the 4 GAD types)
# ---------------------------------------------------------------------


def _uo2_indices(values):
    idx = []
    for v in values:
        for k, (kind, e, g) in enumerate(PIN_TYPES):
            if kind == "UO2" and abs(e - v) < 1e-6:
                idx.append(k)
                break
    return idx


H1_ALLOWED = _uo2_indices([1.6, 2.0, 2.4, 2.8, 3.2])
H2_ALLOWED = _uo2_indices([2.8, 3.2, 3.6, 4.0, 4.4, 4.95])
H3_ALLOWED = _uo2_indices([3.6, 4.0, 4.4, 4.95])
H4_ALLOWED = _uo2_indices([3.6, 4.0, 4.4, 4.95]) + [9, 10, 11, 12]  # + all 4 GAD types

assert len(H1_ALLOWED) == 5
assert len(H2_ALLOWED) == 6
assert len(H3_ALLOWED) == 4
assert len(H4_ALLOWED) == 8

# ---------------------------------------------------------------------
# 35-variable representation: var 0=H1, var 1=H2, var 2=H3, vars 3..34=H4
# ---------------------------------------------------------------------
N_VARS = 3 + len(H4_CELLS)  # 35
assert N_VARS == 35

VAR_CELLS = [H1_CELLS, H2_CELLS, H3_CELLS] + [[cell] for cell in H4_CELLS]
VAR_ALLOWED = [H1_ALLOWED, H2_ALLOWED, H3_ALLOWED] + [H4_ALLOWED] * len(H4_CELLS)

# ---------------------------------------------------------------------
# Rule 3 override: positions 26/29 = grid cells (4,4)/(5,5) -- see the
# module docstring's "Rule 3" section above for the full derivation and
# provenance. These two H4 variables get H3_ALLOWED (UO2 only) instead
# of the general H4_ALLOWED (which includes the 4 GAD types).
# ---------------------------------------------------------------------
NO_GAD_CELLS = [(4, 4), (5, 5)]  # paper's positions 26 and 29
NO_GAD_VARS = []
for cell in NO_GAD_CELLS:
    var = 3 + H4_CELLS.index(cell)
    VAR_ALLOWED[var] = H3_ALLOWED
    NO_GAD_VARS.append(var)

TYPE_MASK = np.zeros((N_VARS, N_TYPES), dtype=bool)
for v, allowed in enumerate(VAR_ALLOWED):
    for k in allowed:
        TYPE_MASK[v, k] = True

# sanity: the two Rule-3 variables must have exactly H3_ALLOWED's 4
# UO2-only choices and NONE of the 4 GAD types (global indices 9-12)
for var in NO_GAD_VARS:
    assert set(np.nonzero(TYPE_MASK[var])[0].tolist()) == set(H3_ALLOWED)
    assert not TYPE_MASK[var, 9:13].any()


def build_grid_heuristic(type_idx):
    """Expand a length-35 vector of pin-type indices (one per variable,
    order = VAR_CELLS/VAR_ALLOWED above) into the full 10x10 pattern,
    applying each variable's chosen type to ALL of its tied physical
    cells (and their diagonal mirrors), in the ("UO2", e) / ("GAD", e, g)
    / ("WATER", None) convention used by bwr10x10_surrogate.py."""
    type_idx = np.asarray(type_idx, dtype=int)
    if type_idx.shape[0] != N_VARS:
        raise ValueError(f"expected {N_VARS} values, got {type_idx.shape[0]}")

    grid_type = np.full((GRID_N, GRID_N), -1, dtype=int)
    for v, cells in enumerate(VAR_CELLS):
        t = type_idx[v]
        for (i, j) in cells:
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


if __name__ == "__main__":
    print(f"H1 cells ({len(H1_CELLS)}): {H1_CELLS}  allowed={H1_ALLOWED}")
    print(f"H2 cells ({len(H2_CELLS)}): {H2_CELLS}  allowed={H2_ALLOWED}")
    print(f"H3 cells ({len(H3_CELLS)}): {H3_CELLS}  allowed={H3_ALLOWED}")
    print(f"H4 cells ({len(H4_CELLS)}): {H4_CELLS}  allowed={H4_ALLOWED}")
    print(f"N_VARS = {N_VARS}")
    print(f"Rule 3 (positions 26/29 = cells {NO_GAD_CELLS}, "
          f"vars {NO_GAD_VARS}): GAD excluded, allowed={H3_ALLOWED}")
    combos = 5 * 6 * 4 * (8 ** 30) * (4 ** 2)
    print(f"total combinatorial space (with Rule 3 applied) = "
          f"5*6*4*8^30*4^2 = {combos:.3e}  "
          f"(paper states outer-frame combos 5*6*4=120 exactly; "
          f"its own '832' interior figure looks like a garbled '13^32' or similar -- not reproduced here)")
