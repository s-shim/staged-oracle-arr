"""
arr_quantum_inspired.py
==========================
Quantum-inspired variant of ARR's Perturb+Round step: instead of rounding
each position independently (per-position argmax on the perturbed
centroid, then rejecting the whole candidate if constraint set 1 or the
GAD-GAD adjacency rule is violated), the perturbed centroid's argmax
scores are used as a QUBO objective, and the joint rounding across ALL
positions is solved at once by D-Wave Ocean's SimulatedAnnealingSampler
-- so E-range, N_GAD-range, and GAD-GAD-adjacency feasibility are baked
into the rounding step itself rather than checked (and possibly
rejected) after the fact.

Derivation (worked out interactively; see project notes for the full
QUBO write-up):

    Phi(x) = sum_{i,k} ptb(i,k) * x_{i,k}         (ARR's own rounding
                                                     objective -- what
                                                     per-position argmax
                                                     already maximizes)

    H(x,s) = -Phi(x)
             + P1   * H_onehot        (each position picks exactly 1 type)
             + PE   * H_E             (39100 <= S(x) <= 40112, i.e.
                                        E(x) in [4.25, 4.36]%)
             + PN   * H_N             (16 <= N_GAD(x) <= 18)
             + PGAD * H_GAD_adjacency (no two GAD rods orthogonally
                                        adjacent -- the OTHER half of
                                        Rule 3, not enforced elsewhere
                                        in this project)

    All four penalty weights only need P > |I| (I = the position set
    actually in use, e.g. 35 for the Heuristics-1-4-reduced space) --
    since 0 <= Phi(x) < |I| under one-hot, any single constraint
    violation (minimum penalty increment: 1) already costs more than
    Phi's entire possible range, regardless of how many other
    constraints are simultaneously violated. This module defaults to
    P = |I| + 1 for all four.

NOT covered: k_inf_max, PPF, CL. These have no closed form in x (real
DRAGON5/CASMO4 needed) and cannot be QUBO terms. This module produces
CONSTRAINT-SET-1-AND-GAD-ADJACENCY-FEASIBLE candidates only -- exactly
Phase 1.1's job, done jointly via annealing instead of per-position
argmax + reject. A constraint-set-2-aware hybrid (QUBO-round -> real
DRAGON5 verify, mirroring arr_hybrid_dragon5.py) is a natural next step,
not implemented here.

Requires: dimod, and one of:
    pip install dimod dwave-samplers --break-system-packages   (preferred)
    pip install dimod dwave-neal --break-system-packages       (older Ocean SDK)
"""

import numpy as np

# CHANGE (2026-11-03): dimod / the D-Wave sampler are needed ONLY by the
# QUBO/SA rounding path (build_qubo, qubo_round*, Phase 0 pool generation).
# The greedy Phase 1 / Phase 2 loops (perturb, greedy_round, move_seed,
# init_centroid, check_feasibility) never touch them, so a missing
# install must not stop those from importing. The error is raised only
# when a QUBO function is actually called.
try:
    import dimod
except ImportError:                      # pragma: no cover
    dimod = None

SimulatedAnnealingSampler = None
if dimod is not None:
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ImportError:
        try:
            from neal import SimulatedAnnealingSampler
        except ImportError:
            SimulatedAnnealingSampler = None


def _require_qubo_stack():
    if dimod is None or SimulatedAnnealingSampler is None:
        raise ImportError(
            "The QUBO/SA rounding path needs dimod and a SimulatedAnnealingSampler:\n"
            "  pip install dimod dwave-samplers\n"
            "(not needed for the greedy Phase 1 / Phase 2 loops)")


def _needs_qubo(fn):
    import functools
    @functools.wraps(fn)
    def wrapper(*a, **k):
        _require_qubo_stack()
        return fn(*a, **k)
    return wrapper

GAD_TYPE_IDS = (9, 10, 11, 12)  # global PIN_TYPES indices, see bwr10x10_geometry.py

# ---------------------------------------------------------------------
# Geometry helpers (variable-level, built once from VAR_CELLS)
# ---------------------------------------------------------------------


def compute_multiplicities(var_cells):
    """M_v = total physical rod count governed by variable v (accounts
    for both (a) on-diagonal cells having no mirror twin (m=1) vs
    off-diagonal cells having one (m=2), and (b) tied-group variables
    like Heuristics 1-3 governing MULTIPLE physical cells at once)."""
    M = []
    for cells in var_cells:
        m = sum(1 if i == j else 2 for (i, j) in cells)
        M.append(m)
    return M


def build_cell_to_var(var_cells):
    """Maps every physical (i,j) grid cell (both lower AND upper
    triangle, i.e. including mirror images) to its owning variable
    index -- mirrors build_grid_heuristic's own mirroring convention
    exactly (grid_type[i,j]=t; grid_type[j,i]=t), so it is consistent
    with it by construction."""
    cell_to_var = {}
    for v, cells in enumerate(var_cells):
        for (i, j) in cells:
            cell_to_var[(i, j)] = v
            cell_to_var[(j, i)] = v
    return cell_to_var


def build_adjacency_pairs(var_cells, water_mask, grid_n):
    """All (p, q), p<q, pairs of DIFFERENT variables whose physical
    cells are orthogonally adjacent somewhere in the full 10x10 grid
    (water cells excluded as neighbors). This is the geometric input to
    the GAD-GAD-adjacency QUBO term -- computed once, not part of the
    QUBO itself."""
    cell_to_var = build_cell_to_var(var_cells)
    pairs = set()
    for (i, j), v in cell_to_var.items():
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_n and 0 <= nj < grid_n and not water_mask[ni, nj]:
                v2 = cell_to_var.get((ni, nj))
                if v2 is not None and v2 != v:
                    pairs.add((min(v, v2), max(v, v2)))
    return pairs


# ---------------------------------------------------------------------
# ARR mechanics: perturb, decelerator, reset (self-contained
# reimplementation of the same formulas used by
# arr_optimizer_categorical_centroid05.py, for this QUBO-rounding
# variant specifically)
# ---------------------------------------------------------------------


def init_centroid(var_allowed, n_types):
    """c[i,k] = 0.5 for every allowed (i,k), 0.0 (pinned) otherwise --
    matches the project's fixed-0.5-centroid convention regardless of
    how many choices a position has."""
    c = np.zeros((len(var_allowed), n_types))
    for i, allowed in enumerate(var_allowed):
        for k in allowed:
            c[i, k] = 0.5
    return c


def perturb(c, var_allowed, rng, mode="mult"):
    """Draw ptb(i,k) for every allowed (i,k). mode='mult': c*pi (Eq.46);
    mode='add': c+(1-c)*pi (Eq.47)."""
    ptb = {}
    for i, allowed in enumerate(var_allowed):
        for k in allowed:
            pi = rng.random()
            ptb[(i, k)] = c[i, k] * pi if mode == "mult" else c[i, k] + (1 - c[i, k]) * pi
    return ptb


def greedy_round(ptb, var_allowed, adjacency_pairs, gad_type_ids=GAD_TYPE_IDS):
    """(2026-08-16) Deterministic Round step that GUARANTEES one-hot and
    GAD-GAD adjacency BY CONSTRUCTION -- no QUBO, no SA, no penalty
    terms needed for these two constraints at all. Replaces
    qubo_round/qubo_round_multi for Phase 1 specifically (Phase 2 is
    untouched and keeps using the QUBO/SA path).

    Algorithm: scan every (i,k) pair across ALL positions in
    DESCENDING order of ptb(i,k), and greedily assign x(i,k)=1 the
    first time each position i is reached (this is automatically the
    position's LARGEST ptb value among its own allowed k's, so it's
    the direct analog of classical ARR's per-position
    argmax_k ptb(i,k) rounding) -- UNLESS doing so would make i a GAD
    type while some neighboring position j (per adjacency_pairs) has
    ALREADY been assigned a GAD type earlier in this same scan (i.e.
    at a higher ptb value) -- in that case k is skipped and the scan
    continues to i's next-highest remaining ptb(i,k'), which is
    typically a non-GAD choice (var_allowed positions normally mix GAD
    and non-GAD options) and so usually succeeds immediately after.

    One-hot: guaranteed -- each i is assigned at most once (skipped
    once already assigned) and, since every (i,k) pair for every
    ALLOWED k is in the scan list, always assigned exactly once by the
    time the full list is exhausted (unless every single k allowed at
    some position is a GAD type AND all of them get blocked by
    neighbors, an extremely unlikely edge case given the geometry;
    caught explicitly below rather than silently producing an invalid
    pattern).

    GAD-GAD adjacency: guaranteed -- a position can only become GAD if
    NONE of its neighbors already are, checked fresh against the
    partial assignment built up so far at the exact moment it's
    considered.

    Returns type_idx (list of length n_vars, each an index into
    PIN_TYPES) -- never None entries; raises RuntimeError in the
    (extremely unlikely) edge case above rather than returning an
    inconsistent pattern.
    """
    n_vars = len(var_allowed)
    neighbors = [[] for _ in range(n_vars)]
    for i, j in adjacency_pairs:
        neighbors[i].append(j)
        neighbors[j].append(i)

    order = sorted(ptb.keys(), key=lambda ik: ptb[ik], reverse=True)
    type_idx = [None] * n_vars
    for (i, k) in order:
        if type_idx[i] is not None:
            continue
        if k in gad_type_ids and any(type_idx[j] in gad_type_ids for j in neighbors[i]):
            continue
        type_idx[i] = k

    unresolved = [i for i in range(n_vars) if type_idx[i] is None]
    if unresolved:
        raise RuntimeError(
            f"greedy_round: positions {unresolved} could not be assigned any "
            f"allowed type (every option was GAD and blocked by an already-GAD "
            f"neighbor). This should be geometrically rare -- if it recurs, "
            f"var_allowed's GAD/non-GAD mix at these positions needs review.")
    return type_idx


def move_seed(c, x_best, var_allowed):
    """alpha = 1/(1+exp(4*RMSD(c, 0.5))) -- shrinks the step as c drifts
    from its uniform 0.5 centroid; c <- (1-alpha)*c + alpha*x_best."""
    allowed_mask = np.zeros_like(c, dtype=bool)
    for i, allowed in enumerate(var_allowed):
        for k in allowed:
            allowed_mask[i, k] = True
    diffs = (c[allowed_mask] - 0.5)
    rmsd = float(np.sqrt(np.mean(diffs ** 2))) if diffs.size else 0.0
    alpha = 1.0 / (1.0 + np.exp(4.0 * rmsd))
    x_best_onehot = np.zeros_like(c)
    for i, k in enumerate(x_best):
        if k is not None:
            x_best_onehot[i, k] = 1.0
    c_new = (1 - alpha) * c + alpha * x_best_onehot
    c_new[~allowed_mask] = 0.0
    return c_new, alpha, rmsd


# ---------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------


def _add_squared_linear_penalty(bqm, terms, constant, weight):
    """Adds weight * (sum(coeff*var for var,coeff in terms) + constant)^2
    to bqm, using x^2=x for binary variables. `terms`: dict var -> coeff."""
    items = list(terms.items())
    for idx, (v, coeff) in enumerate(items):
        bqm.add_linear(v, weight * (coeff * coeff + 2 * coeff * constant))
        for v2, coeff2 in items[idx + 1:]:
            bqm.add_quadratic(v, v2, weight * 2 * coeff * coeff2)
    bqm.offset += weight * constant * constant


@_needs_qubo
def build_qubo(ptb, var_allowed, pin_types, adjacency_pairs, multiplicities,
               e_bounds_pct=(4.25, 4.36), ngad_bounds=(16, 18),
               gad_type_ids=GAD_TYPE_IDS, penalty=None,
               include_e_constraint=True, include_ngad_constraint=True):
    """Builds the full QUBO (as a dimod.BinaryQuadraticModel) for one
    Perturb+Round step. Variable labels are (i, k) tuples, i in
    range(len(var_allowed)), k in var_allowed[i]. Returns (bqm, slack_info).

    include_e_constraint (added 2026-08-15, default True preserves all
    prior behavior): when False, the E-range constraint (H_E below) is
    OMITTED from the QUBO entirely -- E is left to be handled as a SOFT
    term in whatever outer objective calls this (see
    run_qubo_hybrid_dragon5.run_qubo_phase1_real_feasible for why: under
    a narrow target like ppf_c=1.4, requiring SA to hit E exactly on
    EVERY round discarded many candidates that might have had genuinely
    good k_inf_max/PPF, just because E missed by a hair -- moving E into
    the same continuous penalty-minimization objective as k_inf/PPF lets
    the ARR seed-move guide toward the right E region using combined
    signal from all three, rather than requiring every single QUBO round
    to independently re-solve E exactly). When False, only N_GAD/
    adjacency/one-hot remain as hard QUBO constraints, and the P1/PGAD
    scale-boost (see scaling note below) is based on N_GAD's own natural
    coefficient scale (multiplicities directly, up to 24x) instead of
    E's -- smaller (24^2=576 vs E's ~118.8^2~14113), so the resulting
    QUBO is noticeably better-conditioned for SA when E is excluded.

    SCALING NOTE (found and fixed during testing -- read before changing
    anything below). The E-constraint's own natural coefficients --
    enrichment times a position's physical-rod multiplicity, up to 24x
    for tied Heuristic-1-3 groups -- are inherently far larger than
    one-hot's or GAD-adjacency's natural coefficients (~1). Two things
    were tried and FAILED before landing on the fix used here:
      1. A single global bqm.normalize() after combining everything with
         equal weights: compresses one-hot's now-comparatively-tiny
         signal into numerical noise. SA then reliably satisfies E/N_GAD/
         adjacency while leaving one-hot badly violated at as many as 15
         of 35 positions, at near-zero (apparently "optimal") energy.
      2. Rescaling the E constraint down into smaller (percent, not
         hundredths-of-percent) units: shrinks the gap by ~100x but the
         24x tied-group multiplicity still dominates; same failure mode,
         just somewhat less severe.
      3. Also tried: real-valued (non-power-of-2) slack weights so
         non-integer percent bounds could be hit exactly. This actually
         broke correctness in a different way -- with continuous slack,
         ANY infeasible E can find a very close slack match, so the
         provable "minimum violation costs >= 1 raw unit" guarantee
         (which relies on S(x)/L/slack sharing a common INTEGER lattice)
         no longer holds. Real-valued slack weights are still used below
         (needed for non-integer percent bounds), but see the point
         directly below for why this project's specific numbers still
         work out safely in practice.
      WHAT ACTUALLY WORKS: instead of shrinking E's penalty to match
      one-hot's scale (structurally difficult -- the 24x multiplicity is
      a genuine geometric fact, not an encoding artifact), GROW one-hot's
      and GAD-adjacency's weights (P1, PGAD) to match E's natural scale
      instead, computed dynamically as max(E-term coefficient)^2 -- i.e.
      make ALL FOUR constraint types comparably "hard" in the same
      landscape, rather than leaving E as the dominant one. Verified over
      15 independent random perturbations: 15/15 single QUBO calls
      (Q4C=1 each) landed on a fully one-hot-and-E-and-
      N_GAD-and-GAD-adjacency-feasible sample with default
      num_reads=200/num_sweeps=1000 after this fix, versus 0/many before
      it. Always still independently re-verify with check_feasibility()
      regardless -- SA remains a heuristic, not an exact solver.

    include_ngad_constraint (added 2026-08-16, default True preserves
    prior behavior): when False, the N_GAD-range constraint (H_N below)
    is ALSO omitted from the QUBO (paired with include_e_constraint=
    False -- see run_qubo_hybrid_dragon5.py's combined_penalty, which
    handles both as soft terms together). Resolves a SEPARATE, more
    serious problem than the scaling note above: once E is excluded but
    N_GAD stays, P1/PGAD still get boosted to N_GAD's own coefficient
    scale (multiplicities up to 24x) to keep one-hot/adjacency correctly
    enforced -- but this makes Phi(x) (whose own coefficients are just
    c(i,k) in [0,1], i.e. O(1)) roughly 20000x smaller than P1/PGAD.
    After normalize(), Phi's contribution to the landscape becomes
    indistinguishable from numerical noise -- confirmed directly
    (2026-08-16): even with c fully converged toward a specific x_best
    (coefficients ~0.9999/0.0001), the QUBO's own linear coefficients
    are dominated entirely by the ~20736-scale P1 term; Phi supplies
    none of the guidance it's supposed to. SA ends up sampling close to
    UNIFORMLY among constraint-satisfying candidates, not preferentially
    toward whatever c/x_best favors -- observed as centroid rmsd
    saturating at its 0.5 maximum with no further improvement for
    hundreds of rounds. Excluding N_GAD too removes the LAST source of
    multiplicity-driven large coefficients from the QUBO entirely (one-
    hot and GAD-adjacency both have naturally O(1) coefficients on their
    own) -- so no boost is needed at all: P1=PGAD=penalty (the original,
    simple P>|I| derivation from the module docstring) is sufficient
    again, and Phi's O(35)-scale signal is no longer swamped.
    """
    n_vars = len(var_allowed)
    if penalty is None:
        penalty = float(n_vars + 1)  # > |I|, see module docstring
    PE = PN = penalty

    if include_e_constraint:
        # E-term coefficients (percent units -- see scaling note point 2)
        e_terms_raw = {}
        for i, allowed in enumerate(var_allowed):
            m = multiplicities[i]
            for k in allowed:
                _, e_pct, _ = pin_types[k]
                e_terms_raw[(i, k)] = e_terms_raw.get((i, k), 0.0) + m * e_pct
        max_e_coeff = max(abs(v) for v in e_terms_raw.values())
        # scale P1/PGAD up to E's own natural squared-coefficient scale
        # (see scaling note above) -- this is the actual fix.
        scale_factor = max_e_coeff ** 2
    elif include_ngad_constraint:
        # E excluded but N_GAD still in -- N_GAD's own coefficients (raw
        # multiplicities, up to 24x) are now the largest remaining scale
        # to match against.
        scale_factor = max(multiplicities) ** 2
    else:
        # BOTH E and N_GAD excluded -- no multiplicity-driven large
        # coefficients remain anywhere in the QUBO (one-hot and GAD-
        # adjacency are naturally O(1) on their own), so NO boost is
        # needed at all -- see include_ngad_constraint's docstring above
        # for why this specifically fixes Phi(x) being swamped.
        scale_factor = 1.0
    P1 = PGAD = penalty * max(1.0, scale_factor)

    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")

    # --- objective: minimize -Phi(x) = -sum ptb(i,k)*x_{i,k} ---
    for (i, k), val in ptb.items():
        bqm.add_linear((i, k), -val)

    # --- one-hot per position: (sum_k x_{i,k} - 1)^2, x^2=x ---
    for i, allowed in enumerate(var_allowed):
        for k in allowed:
            bqm.add_linear((i, k), P1 * (-1.0))
        for a in range(len(allowed)):
            for b in range(a + 1, len(allowed)):
                bqm.add_quadratic((i, allowed[a]), (i, allowed[b]), P1 * 2.0)
        bqm.offset += P1 * 1.0

    # --- E(x) in [4.25, 4.36]% -- built in PERCENT units (not hundredths)
    # to keep coefficients smaller; L_pct/U_pct and slack weights are
    # real-valued (only the slack VARIABLES are binary; see scaling note
    # point 3 on why this is still safe here). Omitted entirely when
    # include_e_constraint=False (see docstring above). ---
    slack_e_vars = []
    if include_e_constraint:
        L_pct = sum(multiplicities) * e_bounds_pct[0]  # e.g. 92*4.25=391.0
        U_pct = sum(multiplicities) * e_bounds_pct[1]  # e.g. 92*4.36=401.12
        width_pct = U_pct - L_pct
        e_weights = _bounded_slack_weights_real(width_pct, n_bits=10)
        e_terms = dict(e_terms_raw)
        slack_e_vars = [f"sE_{r}" for r in range(len(e_weights))]
        for sv, w in zip(slack_e_vars, e_weights):
            e_terms[sv] = -float(w)
        _add_squared_linear_penalty(bqm, e_terms, -L_pct, PE)

    # --- N_GAD(x) in [16, 18] (already small-scale, no rescaling needed).
    # Omitted entirely when include_ngad_constraint=False (see docstring
    # above). ---
    slack_n_vars = []
    if include_ngad_constraint:
        Ln, Un = ngad_bounds
        n_terms = {}
        for i, allowed in enumerate(var_allowed):
            m = multiplicities[i]
            for k in allowed:
                if k in gad_type_ids:
                    n_terms[(i, k)] = n_terms.get((i, k), 0.0) + m
        slack_n_vars = [f"sN_{r}" for r in range(Un - Ln)]
        for sv in slack_n_vars:
            n_terms[sv] = -1.0
        _add_squared_linear_penalty(bqm, n_terms, -Ln, PN)

    # --- GAD-GAD adjacency: sum over adjacent (i,j) of
    #     (sum_{k in GAD, allowed(i)} x_{i,k}) * (sum_{k' in GAD, allowed(j)} x_{j,k'}) ---
    for (i, j) in adjacency_pairs:
        gad_i = [k for k in var_allowed[i] if k in gad_type_ids]
        gad_j = [k for k in var_allowed[j] if k in gad_type_ids]
        for k in gad_i:
            for kp in gad_j:
                bqm.add_quadratic((i, k), (j, kp), PGAD)

    slack_info = {"E": slack_e_vars, "N_GAD": slack_n_vars,
                  "P1": P1, "PE": PE, "PN": PN, "PGAD": PGAD,
                  "L_pct": L_pct if include_e_constraint else None,
                  "U_pct": U_pct if include_e_constraint else None}
    return bqm, slack_info


def _bounded_slack_weights_real(width, n_bits=10):
    """Real-valued (not necessarily integer) 'bounded-coefficient' slack
    encoding: like a standard 1,2,4,...-doubling slack ladder, but scaled
    to hit an arbitrary real-valued width with a FIXED bit budget
    (n_bits), so it stays valid when width is not an integer (e.g. a
    percent-unit E-window width instead of a hundredths-of-a-percent
    one). No representable gaps in [0, width] to within the finest slack
    bit's resolution (width / (2^n_bits - 1))."""
    unit = width / (2 ** n_bits - 1)
    weights = [unit * (2 ** r) for r in range(n_bits - 1)]
    weights.append(width - sum(weights))
    return weights


# ---------------------------------------------------------------------
# Solve + decode
# ---------------------------------------------------------------------


@_needs_qubo
def solve_and_decode(bqm, var_allowed, sampler=None, num_reads=200,
                      num_sweeps=1000, seed=None):
    """Runs SimulatedAnnealingSampler on the QUBO, decodes the lowest-
    energy sample back into a per-position type assignment, and reports
    whether one-hot actually held (SA is heuristic -- not guaranteed).

    IMPORTANT: the raw QUBO built by build_qubo() has extremely
    ill-conditioned coefficients -- tied-group variables (Heuristics
    1-3) have physical multiplicities up to 24x, which combined with
    the penalty weight and the ~39100-scale E-constraint constant
    produces linear/quadratic coefficients up to ~1e10 and an offset
    around 5.5e10 (confirmed by direct inspection). Generic SA's default
    temperature (beta) schedule is not calibrated for this scale and
    reliably fails to find feasible solutions even with many more reads/
    sweeps than should be needed for a problem this size (tested: still
    stuck around energy ~1e4 at 1000 reads / 10000 sweeps). Normalizing
    the BQM's coefficients into [-1, 1] before sampling (which does not
    change the argmin, only rescales all energies together) fixes this
    completely -- verified to reach ~1e-7 (i.e. fully feasible) with the
    DEFAULT num_reads=200/num_sweeps=1000 after normalizing, versus never
    getting close without it. This function normalizes a COPY of bqm
    (the caller's bqm is left untouched) and reports the NORMALIZED
    energy (near 0 <=> feasible), not the original-scale one.

    Returns (type_idx, feasible_onehot, normalized_energy, sampleset)."""
    if sampler is None:
        sampler = SimulatedAnnealingSampler()
    bqm_norm = bqm.copy()
    bqm_norm.normalize((-1.0, 1.0))
    kwargs = {"num_reads": num_reads, "num_sweeps": num_sweeps}
    if seed is not None:
        kwargs["seed"] = seed
    sampleset = sampler.sample(bqm_norm, **kwargs)
    best = sampleset.first
    sample, energy = best.sample, best.energy

    type_idx = [None] * len(var_allowed)
    feasible_onehot = True
    for i, allowed in enumerate(var_allowed):
        chosen = [k for k in allowed if sample.get((i, k), 0) == 1]
        if len(chosen) == 1:
            type_idx[i] = chosen[0]
        else:
            feasible_onehot = False  # 0 or >1 chosen -- SA didn't fully satisfy one-hot
    return type_idx, feasible_onehot, energy, sampleset


@_needs_qubo
def solve_and_decode_all(bqm, var_allowed, sampler=None, num_reads=200,
                          num_sweeps=1000, seed=None):
    """Like solve_and_decode(), but decodes and returns ALL num_reads
    samples (deduplicated by type_idx), not just the single lowest-QUBO-
    energy one. Motivation (added 2026-08-15): the QUBO's own energy
    only reflects Phi(x) (the ARR rounding-quality proxy) plus
    constraint satisfaction -- it says nothing about a caller's actual
    downstream objective when that differs (e.g. Phase 1's combined
    E/k_inf/PPF surrogate penalty, run_qubo_hybrid_dragon5.py). The
    single QUBO-optimal sample is therefore not necessarily the sample
    that's best for such a caller. Since SA already computes all
    num_reads samples regardless of how many are kept, extracting all
    of them costs NO extra SA computation -- only cheap per-sample
    decoding (deduplication and downstream evaluation costs are the
    caller's to manage).

    Returns a list of dicts, each {"type_idx", "onehot_ok", "energy",
    "num_occurrences"}, sorted by QUBO energy ascending, deduplicated by
    the DECODED type_idx (SA commonly reconverges to the same local
    optimum many times across independent reads; dedup avoids wasting
    downstream evaluation on identical candidates)."""
    if sampler is None:
        sampler = SimulatedAnnealingSampler()
    bqm_norm = bqm.copy()
    bqm_norm.normalize((-1.0, 1.0))
    kwargs = {"num_reads": num_reads, "num_sweeps": num_sweeps}
    if seed is not None:
        kwargs["seed"] = seed
    sampleset = sampler.sample(bqm_norm, **kwargs)

    seen = set()
    decoded = []
    for datum in sampleset.data(["sample", "energy", "num_occurrences"]):
        sample = datum.sample
        type_idx = [None] * len(var_allowed)
        onehot_ok = True
        for i, allowed in enumerate(var_allowed):
            chosen = [k for k in allowed if sample.get((i, k), 0) == 1]
            if len(chosen) == 1:
                type_idx[i] = chosen[0]
            else:
                onehot_ok = False
        key = tuple(type_idx)
        if key in seen:
            continue
        seen.add(key)
        decoded.append({"type_idx": type_idx, "onehot_ok": onehot_ok,
                         "energy": datum.energy,
                         "num_occurrences": datum.num_occurrences})
    decoded.sort(key=lambda d: d["energy"])  # guarantee ascending order --
    # don't rely on dimod's typical (but unenforced) sample ordering
    return decoded


def check_feasibility(type_idx, var_allowed, pin_types, multiplicities,
                       adjacency_pairs, e_bounds_pct=(4.25, 4.36),
                       ngad_bounds=(16, 18), gad_type_ids=GAD_TYPE_IDS,
                       check_e_range=True, check_ngad_range=True):
    """Independent, non-QUBO verification of a decoded candidate --
    since SA gives no formal guarantee, ALWAYS re-check in real
    arithmetic before trusting a candidate as feasible.

    check_e_range (added 2026-08-15, default True preserves prior
    behavior): when False, E is still COMPUTED and returned (callers
    treating it as a soft penalty term still need the value), but is
    NOT enforced as a hard pass/fail gate here -- pair with
    build_qubo(..., include_e_constraint=False).

    check_ngad_range (added 2026-08-16, default True preserves prior
    behavior): same idea, for N_GAD -- pair with build_qubo(...,
    include_ngad_constraint=False)."""
    if any(k is None for k in type_idx):
        return False, "one-hot violated (some position has 0 or >1 chosen type)"

    E_num = sum(multiplicities[i] * pin_types[k][1] for i, k in enumerate(type_idx))
    E_den = sum(multiplicities)
    E_pct = E_num / E_den
    if check_e_range and not (e_bounds_pct[0] <= E_pct <= e_bounds_pct[1]):
        return False, f"E={E_pct:.4f}% outside {e_bounds_pct}"

    n_gad = sum(multiplicities[i] for i, k in enumerate(type_idx) if k in gad_type_ids)
    if check_ngad_range and not (ngad_bounds[0] <= n_gad <= ngad_bounds[1]):
        return False, f"N_GAD={n_gad} outside {ngad_bounds}"

    for (i, j) in adjacency_pairs:
        if type_idx[i] in gad_type_ids and type_idx[j] in gad_type_ids:
            return False, f"GAD-GAD adjacency violated between vars {i} and {j}"

    return True, {"E": E_pct, "N_GAD": n_gad}


# ---------------------------------------------------------------------
# One full Perturb+Round step, and a Phase-1.1-style search loop
# ---------------------------------------------------------------------


@_needs_qubo
def qubo_round(c, var_allowed, pin_types, adjacency_pairs, multiplicities,
               rng, sampler=None, num_reads=200, mode="mult", penalty=None,
               include_e_constraint=True):
    """One Perturb+Round step: draw ptb from c, build the QUBO, solve,
    decode, and independently re-verify feasibility. Returns a dict with
    everything a caller (e.g. the search loop below) needs.

    include_e_constraint=False (default True): drops E from the QUBO's
    hard constraints (see build_qubo's docstring) and from
    check_feasibility's hard pass/fail gate -- E is still computed and
    available in the returned detail dict (when feasible) for callers
    that want to treat it as a soft penalty term instead."""
    ptb = perturb(c, var_allowed, rng, mode=mode)
    bqm, slack_info = build_qubo(ptb, var_allowed, pin_types, adjacency_pairs,
                                  multiplicities, penalty=penalty,
                                  include_e_constraint=include_e_constraint)
    type_idx, onehot_ok, energy, sampleset = solve_and_decode(
        bqm, var_allowed, sampler=sampler, num_reads=num_reads)
    feasible, detail = check_feasibility(
        type_idx, var_allowed, pin_types, multiplicities, adjacency_pairs,
        check_e_range=include_e_constraint)
    return {
        "type_idx": type_idx, "feasible": feasible, "detail": detail,
        "onehot_ok": onehot_ok, "energy": energy, "ptb": ptb,
    }


@_needs_qubo
def qubo_round_multi(c, var_allowed, pin_types, adjacency_pairs, multiplicities,
                      rng, sampler=None, num_reads=200, num_sweeps=1000, mode="mult",
                      penalty=None, include_e_constraint=True,
                      include_ngad_constraint=True, use_perturbation=True):
    """Like qubo_round(), but independently re-verifies and returns
    EVERY distinct decoded sample from the round (not just the single
    lowest-QUBO-energy one) -- see solve_and_decode_all()'s docstring
    for why. Costs no extra SA computation; only cheap decode/verify
    per extra sample. Returns (results, ptb) where results is a list of
    dicts, each shaped like qubo_round()'s single return value (minus
    "ptb", which is shared and returned separately since it's the same
    across every sample in one round), sorted by QUBO energy ascending.

    num_sweeps (added 2026-08-16): was previously NOT threaded through
    to solve_and_decode_all at all -- silently stuck at that function's
    own default (1000) regardless of what a caller passed elsewhere.
    Now an explicit parameter here.

    include_ngad_constraint (added 2026-08-16, default True preserves
    prior behavior): see build_qubo's docstring -- pair with
    include_e_constraint=False to eliminate Phi(x) being swamped by the
    penalty terms' scale-boost.

    use_perturbation (added 2026-08-16, default True preserves prior
    behavior): when False, skips perturb() entirely and uses the RAW
    centroid c(i,k) directly as the QUBO's linear objective
    coefficients, instead of a randomly-perturbed ptb(i,k). Motivation:
    this function already extracts up to num_reads DISTINCT candidates
    per round via SA's own internal stochasticity (independent anneals,
    not just a single perturbation-then-argmax-round) -- the additional
    randomness perturb() injects is therefore redundant on top of that;
    SA's own many independent reads already supply the diversity
    classical ARR's single-shot rounding originally needed external
    perturbation for."""
    if use_perturbation:
        ptb = perturb(c, var_allowed, rng, mode=mode)
    else:
        ptb = {(i, k): c[i, k] for i, allowed in enumerate(var_allowed) for k in allowed}
    bqm, slack_info = build_qubo(ptb, var_allowed, pin_types, adjacency_pairs,
                                  multiplicities, penalty=penalty,
                                  include_e_constraint=include_e_constraint,
                                  include_ngad_constraint=include_ngad_constraint)
    decoded_list = solve_and_decode_all(bqm, var_allowed, sampler=sampler,
                                         num_reads=num_reads, num_sweeps=num_sweeps)
    results = []
    for d in decoded_list:
        feasible, detail = check_feasibility(
            d["type_idx"], var_allowed, pin_types, multiplicities, adjacency_pairs,
            check_e_range=include_e_constraint,
            check_ngad_range=include_ngad_constraint)
        results.append({"type_idx": d["type_idx"], "feasible": feasible,
                         "detail": detail, "onehot_ok": d["onehot_ok"],
                         "energy": d["energy"],
                         "num_occurrences": d["num_occurrences"]})
    return results, ptb



@_needs_qubo
def run_qubo_phase1(var_allowed, pin_types, adjacency_pairs, multiplicities,
                     max_qubo_calls=200, num_reads=200, seed=0,
                     reset_window=20, verbose=True):
    """QUBO-rounding analogue of Phase 1.1: find the first pattern
    satisfying E-range, N_GAD-range, AND GAD-GAD adjacency jointly (the
    classical Phase 1.1 only free-checks E/N_GAD; GAD-adjacency is not
    enforced anywhere else in this project). Tracks Q4C (QUBO calls,
    the direct analogue of C4C) as its cost metric -- zero CASMO4/
    DRAGON5 calls are made by this function."""
    rng = np.random.default_rng(seed)
    sampler = SimulatedAnnealingSampler()
    n_types = max(k for allowed in var_allowed for k in allowed) + 1
    c = init_centroid(var_allowed, n_types)

    x_best, stuck = None, 0
    for q4c in range(1, max_qubo_calls + 1):
        step = qubo_round(c, var_allowed, pin_types, adjacency_pairs,
                           multiplicities, rng, sampler=sampler,
                           num_reads=num_reads)
        if verbose:
            tag = "FEASIBLE" if step["feasible"] else f"reject ({step['detail']})"
            print(f"  Q4C {q4c:3d}: onehot_ok={step['onehot_ok']!s:5s} "
                  f"energy={step['energy']:+8.3f}  {tag}")
        if step["feasible"]:
            return {"type_idx": step["type_idx"], "q4c": q4c, "detail": step["detail"]}

        if x_best is None:
            x_best = step["type_idx"]
        c, alpha, rmsd = move_seed(c, x_best, var_allowed)
        stuck = stuck + 1 if step["type_idx"] == x_best else 0
        if stuck >= reset_window and rng.random() < min(1.0, stuck / reset_window - 1 + rmsd):
            c = init_centroid(var_allowed, n_types)
            stuck = 0
    return None


if __name__ == "__main__":
    from bwr10x10_geometry_heuristic import VAR_CELLS, VAR_ALLOWED, N_VARS
    from bwr10x10_geometry import PIN_TYPES
    from bwr10x10_surrogate import GRID_N, pattern_to_grids
    from bwr10x10_reference_patterns import REFERENCE
    _, _, WATER_MASK = pattern_to_grids(REFERENCE)

    multiplicities = compute_multiplicities(VAR_CELLS)
    print(f"N_VARS={N_VARS}  total physical rods={sum(multiplicities)} (expect 92)")

    adjacency_pairs = build_adjacency_pairs(VAR_CELLS, WATER_MASK, GRID_N)
    print(f"adjacency pairs (variable-level): {len(adjacency_pairs)}")

    print("\n--- single QUBO-round smoke test (uniform 0.5 centroid) ---")
    rng = np.random.default_rng(0)
    c0 = init_centroid(VAR_ALLOWED, 13)
    result = qubo_round(c0, VAR_ALLOWED, PIN_TYPES, adjacency_pairs,
                         multiplicities, rng, num_reads=200)
    print(f"onehot_ok={result['onehot_ok']}  feasible={result['feasible']}  "
          f"detail={result['detail']}  energy={result['energy']:.3f}")

    print("\n--- Phase-1.1-style search: first fully feasible pattern ---")
    out = run_qubo_phase1(VAR_ALLOWED, PIN_TYPES, adjacency_pairs, multiplicities,
                           max_qubo_calls=30, num_reads=200, seed=0)
    if out is None:
        print("No feasible pattern found within max_qubo_calls.")
    else:
        print(f"Feasible pattern found at Q4C={out['q4c']}: {out['detail']}")
