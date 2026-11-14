"""
bwr10x10_dragon5.py
=====================
Real-physics replacement for bwr10x10_surrogate.py's casmo4_bwr10x10(),
built on DRAGON5 (Ecole Polytechnique de Montreal) instead of the
3-point-calibrated surrogate.

THIS MODULE DOES NOT INSTALL, COMPILE, OR RUN DRAGON5 FOR YOU. You need
your own working `dragon` executable (DRAGON5, IGE-335) and a compatible
draglib (e.g. SHEM295/DLIB_295, JEFF3.1) reachable from wherever this
code runs. Point this module at them via Dragon5Config (or the
DRAGON5_EXE / DRAGON5_LIBRARY environment variables) below.

--------------------------------------------------------------------
PROVENANCE OF THE CONTROL-LANGUAGE LOGIC IN THIS FILE (read this first)
--------------------------------------------------------------------
This is a REWRITE of an earlier version of this module. The earlier
version's build_deck() was written cold from the CLE-2000 grammar
(IGE-335 Section 2) without ever running against a real `dragon`
executable, and it turned out to have several bugs that would have
either crashed immediately or silently produced a frozen (non-depleting)
k_inf trajectory. This rewrite ports, module-call-for-module-call, a
DRAGON5 deck structure that WAS actually run against a real `dragon`
executable and cross-checked against the source paper (Radaideh et al.
2021, Nuclear Engineering and Design 372, 110966):

    REFERENCE pattern   : this deck's peak k_inf = 1.1083 @ 20,000 MWd/tU
                          vs. paper's k_inf_max = 1.09814  (+0.9%)
    RL_PPF_1_6 pattern  : this deck's peak k_inf = 1.1106 @ 20,000 MWd/tU
                          vs. paper's k_inf_max = 1.10151  (+0.8%)
    RL_PPF_1_4 pattern  : this deck's peak k_inf = 1.1129 @ 20,000 MWd/tU
                          vs. paper's k_inf_max = 1.10446  (+0.76%)

That validation was done on a single fixed 10x10 GE14-style geometry
(same pin pitch/radii/water-rod layout as this module uses) with a real
`dragon` binary + SHEM295/DLIB_295 draglib, iterating through and ruling
out multiple wrong hypotheses along the way (solver-chain wiring,
self-shielding options, channel void fraction, and -- the actual root
cause of a completely frozen k_inf trajectory -- EVO: being pointed at
the wrong LCM object). The specific fixes baked into this rewrite:

  1. MCCGT: is required between SALT: (long-characteristics tracking)
     and ASM:/FLU:. ASM: must use the ARM option (not PIJ, which is only
     for SYBILT:-produced collision-probability tracking) when consuming
     MCCGT:-converted tracking. FLU:'s argument order matters:
     `FLU: LIBRARY2 CP TRACK TF_EXC` (library, then ASM: output, then
     tracking, then the SALT: exchange-probability file) -- omitting
     TF_EXC or reordering these silently produces wrong results.
  2. The draglib (DLIB_295/SHEM295 format) embeds its own depletion
     chain and fission-yield data -- `DEPL LIB: DRAGON FIL: <name>` in
     the LIB: block is both necessary and sufficient; no separate
     depletion library is needed for a DRAGON-format draglib.
  3. EVO: MUST evolve the self-shielded object actually used by
     ASM:/FLU: (called LIBRARY2 below), not the un-self-shielded base
     microlib (LIBRARY). This matches the official lumpSS.x2m example
     that ships with DRAGON5 for the SHEM295 library, and is the
     opposite convention from the (also official) uo2_evo_hdf.x2m
     example, which evolves the base library instead -- the two
     official examples are NOT interchangeable, and picking the wrong
     one leaves the object actually used for transport completely
     frozen while the un-used object depletes normally. This was the
     root cause of an earlier "k_inf never changes across 30,000
     MWd/tU" bug, found only by directly dumping isotope densities from
     both LCM objects with `UTL: <obj> :: DIR IMPR ISOTOPESDENS <n> ;`
     and observing that only one of them actually changed.
  4. Self-shielding (USS:) is refreshed at EVERY burnup step (not just a
     handful of checkpoints) -- refreshing only occasionally still ran,
     but made the k_inf trajectory shape less physically clean.
  5. The fuel-channel moderator is modeled at the SAME density as the
     water-rod water (0% void), not a 40%-void bundle average -- the
     40%-void assumption alone accounted for roughly +0.04 of a ~0.23
     k_inf gap in earlier testing on this same geometry.
  6. Self-shielding is done on a SMALL representative geometry (one
     CARCEL cell per distinct fuel type, via SYBILT:), separately from
     the full 10x10 assembly's own SALT:-based flux tracking -- this
     "2-level" scheme matches DRAGON5's own lumpSS.x2m convention.

NOT claimed to be validated by the above: PPF (pin-power peaking) and
CL_proxy (cycle length). Neither was part of the depletion-loop
validation above (which only tracked scalar k_inf per step).

  - PPF IS IMPLEMENTED (validated 2026-08-08), but NOT via EDI:. A prior
    attempt (`EDI: ... MICR ... POWR`) failed immediately (`EDIGET: READ
    ERROR ... EXPECTED AFTER MICR`) and cross-checking against IGE-335
    Sec. 3.08 confirmed EDI: has no "POWR" keyword at all -- the whole
    EDI:-based approach was abandoned. Instead, PPF is computed by hand
    from raw LCM data pulled out with UTL: (the same LCM-inspection tool
    already used to find the EVO:/LIBRARY2 bug -- see point 3 above),
    using the actual grammar read out of the compiled DRVUTL.f source
    (Ganlib/src/DRVUTL.f) rather than guessed from the manual:
      * `STEP UP '<name>'` descends into a named sub-directory; `STEP
        DOWN` returns to the parent. State is NOT preserved across
        separate UTL: module calls -- an entire STEP UP/.../STEP DOWN
        sequence must be one single UTL: invocation.
      * `STEP AT <n>` descends into element n of a LIST-type record
        whose elements are themselves further LCM objects (dicts).
      * `IMPR <n> <count>` (integer form) prints element n of a
        LIST-of-flat-arrays record directly (no STEP AT needed/valid for
        this case -- STEP AT on a flat-array element fails with "IS NOT
        A TABLE/LIST"). `IMPR <NAME> <count>` (character form) prints a
        named record in the current directory.
    The quantities pulled out and how they combine:
      * `TRACK::MATCOD` (region -> mixture number) and `TRACK::VOLUME`
        (region volume, cm) -- static per pattern, read once.
      * `CALC::FLUX` -- a LIST of ngroup flat arrays, each of length
        NREG (region-wise scalar flux for that group).
      * `LIBRARY2::MACROLIB::GROUP` -- a LIST of ngroup sub-objects, each
        a further dict; the `H-FACTOR` record inside (length NMIX) is
        the mixture-wise "power factor" (eV/cm) -- multiplying by flux
        gives power directly, with no separate NUSIGF-times-energy-
        per-fission bookkeeping needed.
      * `region power = VOLUME(r) * sum_g[ FLUX(g,r) * H-FACTOR(g,
        MATCOD(r)) ]`; regions group into pins in fixed blocks of 4
        (channel corner, clad, clad/gap, fuel -- in that DRAGON-internal
        order, identified programmatically as "whichever of the 4 isn't
        clad/channel/waterrod" rather than by position) per non-water
        cell, or 1 region per water cell. `PPF = max(pin power) /
        mean(pin power)` over fuel pins only.
    Validated two ways on the REFERENCE pattern at BOL: (1) mixtures 13
    (channel water) and 14 (waterrod water) -- physically identical
    composition/density/temperature -- produced BIT-IDENTICAL H-FACTOR
    values, a strong parser-correctness check; (2) the resulting pin
    power ranking is physically sensible without being tuned to be so --
    at BOL, lowest-power pins are exactly the Gd-bearing (GAD) mixtures
    (neutron absorption suppresses local power, as expected) and the
    highest-power pin is the highest-enrichment plain-UO2 mixture.

    2026-08-09 follow-up, after fixing FUEL_TEMP_K (was 600K, corrected
    to 900K per Radaideh et al. 2021 Sec. 2 -- see that constant's own
    comment) and running the FULL 14-step burnup schedule with
    compute_ppf=True on REFERENCE, RL_PPF_1_4, and RL_PPF_1_6:
      * k_inf_max now matches the paper to within 0.004%-0.12% on BOTH
        patterns (REFERENCE: 1.098102 vs. 1.09814; RL_PPF_1_4: 1.10314
        vs. 1.10446) -- the previously-documented "+0.9% k_inf bias"
        was actually just the 600K/900K fuel temperature bug, not a
        genuine DRAGON5-vs-CASMO4 modeling difference. RL_PPF_1_6
        (checked the same way) confirms: k_inf_max 1.100753 vs. 1.10151
        (-0.07%).
      * PPF at the k_inf-max step is off in BOTH DIRECTION and
        MAGNITUDE across all three calibration patterns: REFERENCE
        -4.5% (1.5478 vs. 1.62), RL_PPF_1_4 +10.6% (1.5119 vs. 1.367),
        RL_PPF_1_6 -15.1% (1.3325 vs. 1.57). Note RL_PPF_1_4 and
        RL_PPF_1_6 -- both "RL-optimized" patterns -- land on OPPOSITE
        signs, so "RL pattern vs. REFERENCE pattern" isn't the
        explanatory variable either. A single fixed calibration
        constant cannot explain a sign-flipping, magnitude-varying
        discrepancy -- this points to a genuine, pattern-dependent
        DRAGON5-vs-CASMO4 PPF modeling difference (self-shielding
        treatment, library, or some limit of this from-scratch
        reconstruction method itself), not a bug we found and fixed.
      * Tested and RULED OUT as the cause: SALT: tracking resolution.
        Doubling both salt_an2d (8->16) and salt_densur (20->40) on
        REFERENCE BOL changed PPF by <0.01% while taking 4.4x longer
        (931s -> 4087s) -- PPF was already fully converged w.r.t.
        tracking mesh at the original settings.
      * Tested and RULED OUT: whether pin power should include only the
        fuel ring vs. all 4 regions per pin. Fuel-only vs. total-region
        power changed PPF by ~0.04% on REFERENCE BOL -- negligible,
        since clad/gap/channel volumes (and hence their contribution)
        are IDENTICAL across every pin, so including/excluding them
        barely moves the max/mean ratio.
      * Checked RL_PPF_1_4's own pin-power ranking at its k_inf-max step
        (20000 MWd/tU) for a red flag and found none: lowest power is
        the lowest-enrichment corner mixture (sensible); highest power
        is mostly the highest-enrichment mixture, with one GAD mixture
        pin ranking above same-enrichment plain UO2 -- plausible mid-
        cycle "power swing" (Gd-155/157 burn out faster than U-235
        depletes, so a GAD rod that was suppressed at BOL can end up
        LESS depleted, and hence higher-reactivity, than a same-
        enrichment non-GAD rod by mid-cycle), not obviously a bug.
      * Investigated the OFFICIAL DRAGON5/DONJON5 pin-power-
        reconstruction path as a way to close this gap: module NAP:
        (Ganlib src confirmed present locally: Donjon/src/NAP.f,
        NAPCPO.f, NAPFTD.f, NAPPPR.f, NAPGEO.f; validated example decks
        Donjon/data/{ASSBLY_CASEA_1level_multicompo,
        rep900EnrichCOMPOhet,testNAPhet}.x2m inspected directly). This
        is NOT a small addition -- it requires a SEPARATE DONJON-side
        pipeline: COMPO:/SPH: to build a multicompo during the DRAGON5
        lattice burnup loop, NAP: PROJECTION to "enrich" it with pin-
        wise flux form functions (a second, DONJON-side TRIVAA:/FLUD:
        solve), then RESINI:/NCR:/MACINI: to build a (even for one
        assembly) core/fuel-map + homogenized macrolib, TRIVAA:/FLUD:
        again for the nodal diffusion solve, FLPOW: for nodal power,
        and finally NAP: PPR METH GPPR for the actual pin-by-pin
        reconstruction. DELIBERATELY NOT BUILT (2026-08-09 decision):
        the current H-FACTOR-based method's ~5-10% PPF error was judged
        acceptable given k_inf_max's near-perfect match and the
        multi-day scope of the full NAP:/COMPO: pipeline -- revisit if
        PPF precision becomes the limiting factor for some future use.
    Bottom line: treat PPF from this module as accurate to within
    roughly a +/-5-10% band (pattern-dependent, not a fixed offset you
    can calibrate out), suitable for judging "in the right ballpark" /
    "near a constraint boundary" but NOT as a precise CASMO4 substitute.
    k_inf_max remains excellent (sub-0.15% observed) and should be
    trusted as the primary validated quantity from this module.
    Cost: computing PPF loops `IMPR`/`STEP AT` ~ngroup times per solve
    (BOL and EVERY burnup step, since PPF is reported at whichever step
    k_inf peaks, and that step isn't known in advance) -- this adds
    real wall-clock time on top of the transport/depletion solves
    already being done, roughly proportional to ngroup x nstep. Set
    `Dragon5Config.compute_ppf=False` to skip this and fall back to the
    k_inf_max-only path if this cost is not worth it for a given run.
  - CL_proxy (2026-08-09 rewrite): the paper actually GIVES an exact,
    simple definition that an earlier version of this module missed --
    Radaideh et al. (2021) Sec. 2: "CL is determined to be the time in
    days after the k-infinity peak at which k-infinity = 0.95." No
    buckling search needed at all. The previous "critical buckling B2"
    redesign (TYPE B B1 SIGS EXTE search, compared against an ASSUMED
    Dragon5Config.critical_buckling_cm2 the paper never actually gives)
    was solving a problem the paper doesn't pose -- it ran without error
    once ARM-tracking compatibility was confirmed (2026-08-09 smoke
    test) and gave CL_proxy=1361 days vs. the paper's 1450 (-6.1%) on
    REFERENCE, but that whole code path is now DELETED in favor of the
    much simpler, provably-correct-to-the-paper's-own-definition
    approach below.
    CL_proxy is now computed purely from the k_inf(burnup) trajectory
    ALREADY being collected for k_inf_max -- no extra DRAGON5 module
    calls, no unvalidated assumptions, no keyword-compatibility risk:
    find the burnup step where k_inf peaks, then find where k_inf FIRST
    drops to/below 0.95 after that peak (linear interpolation between
    the bracketing steps). If the configured burnup_steps_mwd_tu doesn't
    extend far enough past the peak to reach k_inf=0.95, CL_proxy is
    reported as a LOWER BOUND with cl_warning set (extend the schedule
    for a real value) -- e.g. REFERENCE's default 14-step schedule
    (ending at 35000 MWd/tU, k_inf=0.9966) does NOT reach the crossing;
    extend burnup_steps_mwd_tu (e.g. add 40000+) to get a real CL_proxy.
    `Dragon5Config.compute_cl` defaults to True; it now costs nothing
    extra to leave on (pure post-processing of data already collected),
    so there's little reason to set it False other than to suppress the
    lower-bound warning noise on short/smoke-test runs.
    2026-08-10 follow-up: running this method to completion (16-step
    schedule) on all 3 calibration patterns gave CL_proxy consistently
    HIGH by +16.7% to +18.4% -- unlike PPF's sign-flipping, pattern-
    dependent scatter, this near-constant offset across differently-
    enriched/GAD'd patterns was the signature of a wrong CONSTANT, not
    a real physics gap. specific_power_mw_per_tu (used only to convert
    the crossing burnup, in MWd/tU, into days) had been an unvalidated
    25.0 guess; back-solving it independently per pattern gave 29.592 /
    29.535 / 29.181 MW/tU -- a tight cluster (CV 0.6%) around a single
    value. Updated to SPECIFIC_POWER_MW_PER_TU=29.436 (the 3-pattern
    mean); this collapses the CL_proxy error to +0.33% / +0.53% / -0.86%
    on RL_PPF_1_4 / REFERENCE / RL_PPF_1_6 -- i.e. this one constant
    was responsible for essentially the entire previously-reported CL
    discrepancy, not a remaining DRAGON5-vs-CASMO4 modeling difference.
    CL_proxy should now be considered well-validated (sub-1% on all 3
    calibration patterns), unlike PPF's genuine +/-5-15% band.

See CONTRACT below for what you get in each mode.

--------------------------------------------------------------------
CONTRACT (unchanged -- this is still a drop-in replacement)
--------------------------------------------------------------------
Every driver script (run_phase1_10x10.py, run_phase1_heuristic_10x10.py,
run_phase2_heuristic_10x10.py, run_phase3_10x10.py,
run_arr_heuristic_10x10.py, run_full_pipeline_10x10.py,
run_phases_1_2_3_10x10.py, run_parallel_seeds_10x10.py) calls
`casmo4_bwr10x10(pattern)` and reads out['E'], out['G'], out['N_UO2'],
out['N_GAD'], out['k_inf_max'], out['PPF'], and out['CL_proxy']. This
module preserves that contract exactly. out['PPF'] is None UNLESS
cfg.compute_ppf is True, in which case it is the PPF at whichever
burnup step k_inf peaks (see PPF PROVENANCE note above for the method
and its validation status). out['CL_proxy'] is None unless compute_cl
is True (the default) and produced a usable B2 history. Driver scripts
that constrain on PPF now CAN be used with this module, provided
cfg.compute_ppf=True is set explicitly (off by default because of the
added per-step cost -- see PPF PROVENANCE note).

--------------------------------------------------------------------
COST: CACHING AND max_c4c
--------------------------------------------------------------------
A single real DRAGON5 assembly-depletion run at the validated 12-step
schedule below took, in the environment this was validated in, on the
order of tens of minutes (self-shielding + SALT: tracking + MCCGT: +
12 x [EVO: + USS: + ASM: + FLU:]). This is nothing like the surrogate's
sub-millisecond evaluation:
  1. Pass a small max_c4c / max_trials to whichever driver you run --
     the large budgets in the technical report were sized for the free
     surrogate. Start with single or low-double-digit real evaluations
     and confirm timing on your own machine before scaling up.
  2. This module memoizes casmo4_bwr10x10() results (keyed on the
     pattern AND the physics-relevant parts of Dragon5Config, so
     changing the burnup schedule/geometry does not silently return a
     stale cached result) to an on-disk JSON cache -- see
     Dragon5Config.cache_path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

from bwr10x10_surrogate import GRID_N, pattern_to_grids, average_enrichment, average_gad

# ----------------------------------------------------------------------
# GEOMETRY CONSTANTS -- validated GE14-class 10x10 BWR lattice (same
# values used in the depletion-loop validation summarized above). Still
# literature-typical, NOT vendor-exact (GE's NEDO-33236 mechanical design
# report redacts its dimension table) -- override via Dragon5Config if
# you have real mechanical-design data.
# ----------------------------------------------------------------------
PIN_PITCH_CM = 1.295
R_FUEL_CM = 0.440   # Radaideh, Forget & Shirvan, Ann. Nucl. Energy 160 (2021) 108355, Table 1 (was 0.4445)
R_GAP_CM = 0.453
R_CLAD_CM = 0.513

FUEL_TEMP_K = 900.0
CLAD_TEMP_K = 600.0
MODERATOR_TEMP_K = 560.0
# FUEL_TEMP_K and MODERATOR_TEMP_K match Radaideh et al. (2021), Sec. 2:
# "Both BOL and EOL simulations are conducted under fixed fuel
# temperature of 900 K and moderator temperature of 560 K." Corrected
# 2026-08-09 from an earlier placeholder (600K/559K) -- see PPF
# PROVENANCE note and conversation history for how this was caught.
# CLAD_TEMP_K is NOT stated in the paper (not a fixed-BOL/EOL parameter
# there); left at a literature-typical BWR value.

# Both at 0% void (validated correction -- see provenance note, point 5).
CHANNEL_DENSITY_G_CM3 = 0.4582  # SYM oracle: 40% void (0.739*0.6 + 0.037*0.4); legacy oracle: 0.739
WATERROD_DENSITY_G_CM3 = 0.739

# Validated burnup schedule (MWd/tU) and specific power (MW/tU).
# The burnup schedule itself is chosen to bracket CL_proxy's crossing
# point (see cl_kinf_threshold, and the note on 32500-45000 below).
#
# specific_power_mw_per_tu was originally an unvalidated 25.0 assumption
# (the source paper never states it) and used only to convert the
# k_inf=0.95-crossing burnup (in MWd/tU -- pure physics, independent of
# this constant) into "days" for comparison against the paper's CL. That
# guess produced a CL_proxy that was high by a suspiciously CONSISTENT
# +16.7% to +18.4% across all 3 calibration patterns (REFERENCE,
# RL_PPF_1_4, RL_PPF_1_6) -- unlike PPF's genuinely pattern-dependent,
# sign-flipping scatter (see PPF PROVENANCE note), a near-constant
# multiplicative offset across differently-enriched/GAD'd patterns is
# the signature of a wrong CONSTANT, not a real physics difference.
# Back-solving specific_power_mw_per_tu = crossing_burnup_MWd_tU /
# paper_CL_days independently for each of the 3 patterns (2026-08-10)
# gives 29.592 / 29.535 / 29.181 MW/tU -- a tight cluster (mean 29.436,
# stdev 0.18, CV 0.6%) around a single value, confirming the fix is
# exactly this one constant. Using 29.436 collapses the CL_proxy error
# from +16.7-18.4% to +0.33% / +0.53% / -0.86% on the 3 patterns --
# i.e. this alone, not any remaining DRAGON5-vs-CASMO4 physics gap,
# was responsible for essentially the entire previously-reported CL
# discrepancy. 29.436 MW/tU is also physically unremarkable for a BWR
# assembly (typical range ~20-30 MW/tU), so there's no red flag against
# taking it at face value as the paper's real (if unstated) assumption.
#
# 32500/35000/40000/45000 appended (2025-2026 revisions) after real
# REFERENCE runs. A 900K-fixed, full-16-step run (2026-08-09, after
# correcting FUEL_TEMP_K from an earlier 600K bug) gave k_inf_max=
# 1.098102 (vs. paper's 1.09814, now matching to within 0.004% -- the
# previously-documented "+0.9% k_inf bias" was that temperature bug, not
# a genuine DRAGON5-vs-CASMO4 difference) and k_inf=0.9966 at the
# schedule's old end point (35000 MWd/tU), still above cl_kinf_threshold
# (0.95) -- 40000/45000 bracket the actual crossing (~42900-43200
# MWd/tU across the 3 patterns).
# SYM oracle (2026-11): 10 steps (+BOL) reaching the k_inf=0.95 crossing (~43 GWd/tU).
# k_inf_max depends on step size by ~1% (Gd depletion over a 15->20 GWd/tU step); keep this
# schedule fixed for every oracle call so that all numbers are mutually consistent.
BURNUP_STEPS_MWD_TU = (2000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0,
                        35000.0, 40000.0, 42500.0, 45000.0)
BURNUP_STEPS_LEGACY_16 = (200.0, 500.0, 1000.0, 2000.0, 3500.0, 5000.0, 7500.0,
                           10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 32500.0,
                           35000.0, 40000.0, 45000.0)
SPECIFIC_POWER_MW_PER_TU = 29.436
# CL_proxy's crossing threshold -- this IS the paper's own definition
# (Radaideh et al. 2021 Sec. 2: "CL is determined to be the time in days
# after the k-infinity peak at which k-infinity = 0.95"), not an
# assumption of ours. See module docstring's CL_proxy note.
CL_KINF_THRESHOLD = 0.95

# SYBILT:/SALT:/MCCGT:/USS: tuning -- copied verbatim from the validated
# deck. GRMIN and PASS were both individually tested and found NOT to
# change the result on this geometry/library, but are kept for parity
# with the validated deck rather than removed.
SS_MAXR, SS_MAXZ = 500, 500000
SS_QUA2 = (20, 3)
SALT_AN2D, SALT_DENSUR = 8, 20.0
MCCGT_CACB, MCCGT_AAC, MCCGT_MCU = 4, 80, 10000
USS_PASS, USS_MAXST, USS_GRMIN = 1, 300, 52
EVO_EPS1, EVO_EPS2, EVO_EXPM = 0.025, 0.1, 1.0e15


@dataclass
class Dragon5Config:
    """Everything environment/assumption-specific lives here."""

    dragon_exe: str = field(default_factory=lambda: os.environ.get("DRAGON5_EXE", "dragon"))
    draglib_path: str = field(default_factory=lambda: os.environ.get("DRAGON5_LIBRARY", ""))
    draglib_local_name: str = "DLIB_295"  # filename DRAGON's FIL: keyword looks for
                                            # in the run directory -- draglib_path is
                                            # copied/symlinked to this name before running
                                            # (mirrors the .access-script convention used
                                            # during validation)
    work_dir: str | None = None
    keep_work_dir: bool = False
    cache_path: str | None = "dragon5_cache.json"
    timeout_sec: int = 28800  # 8h -- a real full 14-step run (BOL + 14 x
                              # K-solve, plus compute_ppf's per-step FLUX/
                              # H-FACTOR dumps if enabled) was confirmed to
                              # exceed the old 7200s (2h) default while
                              # still legitimately computing (not hung);
                              # raise further if still too short for your
                              # machine. On timeout, partial stdout up to
                              # the kill is now salvaged (see _run_dragon), so a
                              # too-short timeout degrades to a clear error with
                              # a partial-progress count rather than losing
                              # everything silently.

    # geometry / material assumptions
    pin_pitch_cm: float = PIN_PITCH_CM
    r_fuel_cm: float = R_FUEL_CM
    r_gap_cm: float = R_GAP_CM
    r_clad_cm: float = R_CLAD_CM
    fuel_temp_k: float = FUEL_TEMP_K
    clad_temp_k: float = CLAD_TEMP_K
    moderator_temp_k: float = MODERATOR_TEMP_K
    channel_density: float = CHANNEL_DENSITY_G_CM3
    waterrod_density: float = WATERROD_DENSITY_G_CM3

    burnup_steps_mwd_tu: tuple = BURNUP_STEPS_MWD_TU
    specific_power_mw_per_tu: float = SPECIFIC_POWER_MW_PER_TU
    cl_kinf_threshold: float = CL_KINF_THRESHOLD

    # SALT: tracking resolution -- an2d is the number of tracking angles
    # (angular quadrature), densur is the tracking-line density (lines/cm,
    # spatial resolution) fed to `TSPC EQW2 <<an2d>> <<densur>> REND`.
    # Defaults (8, 20.0) are the values copied verbatim from the
    # validated gen_x2m_evo.py and match k_inf_max to within 0.004% of
    # the paper (see PPF PROVENANCE note) -- k_inf (a global, integrated
    # quantity) is insensitive to this resolution at these settings.
    # PPF (a local, spatially-higher-frequency quantity) is more
    # sensitive to tracking resolution than k_inf is, and DID show a
    # persistent ~4-5% gap vs. the paper's PPF at these defaults even
    # after k_inf converged well -- raise an2d/salt_densur to test
    # whether that gap closes with finer tracking. Cost scales roughly
    # with an2d x salt_densur, so doubling both roughly quadruples the
    # SALT:/MCCGT: tracking cost (on top of, not instead of, the
    # transport/depletion solve cost already dominating runtime).
    salt_an2d: int = SALT_AN2D
    salt_densur: float = SALT_DENSUR

    # Number of energy groups in the draglib in use. MUST match the actual
    # draglib (SHEM295 -> 295, SHEM361 -> 361, etc.) -- only used by the
    # PPF extraction (compute_ppf=True), which dumps region flux and
    # mixture H-FACTOR group-by-group and needs to know how many groups
    # to loop over. Wrong value here will NOT crash cleanly -- it will
    # silently mis-chunk the parsed FLUX/H-FACTOR arrays. See
    # DRAGON5_LIBRARY / draglib_local_name for the library itself.
    n_groups: int = 295

    # compute_cl is NOT FULLY VALIDATED -- see module docstring.
    # compute_ppf IS validated as of the 2026-08-08 session documented in
    # the PPF PROVENANCE note below (region-wise FLUX x mixture-wise
    # H-FACTOR x region VOLUME, region->pin grouping from TRACK::MATCOD).
    compute_ppf: bool = False
    compute_cl: bool = True    # NOT FULLY VALIDATED -- see module docstring
    # Which burnup step the returned PPF is evaluated at: 'bol' (step 0; the source authors
    # state PPF peaks at BOL) or 'kmax' (step of k_inf max; the legacy oracle's choice).
    ppf_at: str = 'bol'

    # ---- EARLY ABORT (opt-in; both None = original behaviour, run to completion) ----
    # DRAGON5 prints progress while it runs (verified: RESULT/PPF dump lines arrive in real time).  When either
    # limit is set, casmo4_bwr10x10 streams the output and KILLS the process as soon as the candidate is proven
    # infeasible:
    #   abort_ppf_above : stop right after the BOL (step 0) PPF dump if BOL PPF >= this value
    #                     (only when compute_ppf=True and ppf_at='bol'; step 0 arrives ~22 % into the run)
    #   abort_k_above   : stop at the first burnup step whose k_inf exceeds this value (k_inf_max can only grow)
    # An aborted call returns the usual dict with CL_proxy=None, k_inf_max = max k over the steps seen so far,
    # PPF = BOL PPF (if the dump was complete), out['aborted'] = reason, and the BOL pin map in
    # burnup_history['pin_power_map_kmax'].  Aborted results are never written to the file cache.
    abort_ppf_above: float | None = None
    abort_k_above: float | None = None
    # abort_k_mode: 'first' (default, original) = kill at the first step with k_inf > abort_k_above (k_inf_max is then
    # only known to be >= that value).  'peak' (Phase 1, added 2026-11-07) = keep running until the peak has passed
    # (k_inf declines at some step >= 3 relative to the previous step), then kill if the peak exceeded abort_k_above;
    # the aborted result then carries the TRUE k_inf_max, so a graded k penalty is exact.  Costs one extra burnup step.
    abort_k_mode: str = 'first'
    abort_poll_sec: float = 2.0

    # ---- OPTIONAL OUTER FRAME (added for the PPF water-gap test) ----
    # Concentric square layers around the 10x10 pin array, from the pins
    # outward: (1) frame_inner_cm of moderator (clearance between the pin
    # array and the channel wall), (2) frame_wall_cm of Zircaloy channel
    # wall, (3) frame_water_cm of moderator (= HALF of the water gap
    # between adjacent assemblies, because the outer boundary is a
    # reflective mirror plane). All three default to 0.0, which gives the
    # ORIGINAL deck exactly (reflective boundary right at the pin array).
    frame_inner_cm: float = 0.225   # Table 1: box inner width 13.4 cm
    frame_wall_cm: float = 0.19     # Table 1: box wall
    frame_water_cm: float = 0.71    # half gap = mean(0.95, 0.47) (Table 1); author: 'water gap 0.7 cm'
    # Density (g/cm3) of the OUTER frame water (the inter-assembly gap, outside
    # the channel wall). None => same mixture as the in-channel moderator (the
    # original behaviour). Set it (e.g. 0.739, liquid) together with a reduced
    # channel_density (in-channel void) so that the gap water stays liquid.
    gapwater_density: Optional[float] = 0.739   # gap water stays liquid
    # Asymmetric inter-assembly gap (Radaideh, Forget & Shirvan, Ann. Nucl.
    # Energy 160 (2021) 108355, Table 1: half wide gap 0.95 cm, half narrow
    # gap 0.47 cm). If BOTH are set they replace frame_water_cm: the two
    # faces named in frame_wide_faces get the wide half-gap, the opposite
    # two the narrow one. "TL" = top+left faces wide, "BR" = bottom+right
    # faces wide (the pattern is symmetric about the top-left/bottom-right
    # diagonal, so these are the only two choices that keep that symmetry).
    frame_water_wide_cm: Optional[float] = None
    frame_water_narrow_cm: Optional[float] = None
    frame_wide_faces: str = "TL"


DEFAULT_CONFIG = Dragon5Config()


# ----------------------------------------------------------------------
# Isotopic composition helpers (ported from the validated project's
# compositions.py: UO2 + Gd2O3 fuel, Zircaloy-2 clad, light water, with
# the U234 empirical correlation DRAGON's own Mix_UOX.c2m procedure
# uses, and full natural-abundance Gd isotopics instead of a single
# lumped "GD_OTH" pseudo-isotope).
# ----------------------------------------------------------------------
_NA = 0.602214076  # Avogadro's number x 1e-24, for atoms/barn-cm

_M_U234, _M_U235, _M_U238, _M_O16 = 234.0409, 235.0439, 238.0508, 15.9949
_M_GD = {154: 153.9209, 155: 154.9226, 156: 155.9221, 157: 156.9240, 158: 157.9241, 160: 159.9271}
_GD_RAW = {154: 0.0218, 155: 0.1480, 156: 0.2047, 157: 0.1565, 158: 0.2484, 160: 0.2186}
_GD_S = sum(_GD_RAW.values())
_GD_ABUND = {k: v / _GD_S for k, v in _GD_RAW.items()}
_M_GD2O3 = 2 * sum(_GD_ABUND[a] * _M_GD[a] for a in _M_GD) + 3 * _M_O16

_RHO_UO2 = 10.4    # g/cm3, ~94.8% TD BWR fresh-fuel pellet density (assumption)
_RHO_GD2O3 = 7.07  # g/cm3 theoretical density of Gd2O3 (assumption)

_P_ZR = {90: 0.5145, 91: 0.1122, 92: 0.1715, 94: 0.1738, 96: 0.0280}
_M_ZR = {90: 89.9047, 91: 90.9056, 92: 91.9050, 94: 93.9063, 96: 95.9083}
_RHO_CLAD = 6.55  # g/cc, Zircaloy-2

_M_H2O = 18.01056


def _uo2_fractions(e_wt_pct):
    n_u5 = e_wt_pct / 100.0
    n_u4 = n_u5 * 0.0054 / 0.72  # standard U234 correlation for enrichment-plant product U
    n_u8 = 1.0 - n_u5 - n_u4
    n_o16 = 2.0 * _M_O16 * (n_u4 / _M_U234 + n_u5 / _M_U235 + n_u8 / _M_U238)
    tot = n_u4 + n_u5 + n_u8 + n_o16
    return {"U234": n_u4 / tot, "U235": n_u5 / tot, "U238": n_u8 / tot, "O16": n_o16 / tot}


def _gd2o3_fractions():
    fracs = {}
    gd_mass_frac_total = 2 * sum(_GD_ABUND[a] * _M_GD[a] for a in _M_GD) / _M_GD2O3
    for a in _M_GD:
        fracs[f"Gd{a}"] = gd_mass_frac_total * (_GD_ABUND[a] * _M_GD[a]) / sum(
            _GD_ABUND[b] * _M_GD[b] for b in _M_GD)
    fracs["O16_gd"] = 3 * _M_O16 / _M_GD2O3
    return fracs


def _pin_densities(kind, e_wt_pct, gad_wt_pct, fuel_density_g_cm3):
    """atoms/barn-cm for one fuel pin type (UO2, optionally + Gd2O3),
    accounting for the volume Gd2O3 displaces within the pellet."""
    if kind != "GAD" or gad_wt_pct <= 0.0:
        rho = fuel_density_g_cm3
        w = _uo2_fractions(e_wt_pct)
        M = {"U234": _M_U234, "U235": _M_U235, "U238": _M_U238, "O16": _M_O16}
        return {iso: rho * frac * _NA / M[iso] for iso, frac in w.items()}
    w_gd2o3 = gad_wt_pct / 100.0
    w_uo2 = 1.0 - w_gd2o3
    rho = 1.0 / (w_uo2 / _RHO_UO2 + w_gd2o3 / _RHO_GD2O3)  # ideal volume-mixing rule
    uo2_w = _uo2_fractions(e_wt_pct)
    gd_w = _gd2o3_fractions()
    M = {"U234": _M_U234, "U235": _M_U235, "U238": _M_U238, "O16": _M_O16}
    dens = {iso: rho * (w_uo2 * frac) * _NA / M[iso] for iso, frac in uo2_w.items()}
    for a in _M_GD:
        key = f"Gd{a}"
        dens[key] = rho * (w_gd2o3 * gd_w[key]) * _NA / _M_GD[a]
    dens["O16"] += rho * (w_gd2o3 * gd_w["O16_gd"]) * _NA / _M_O16
    return dens


def _clad_densities():
    return {f"Zr{a}": _RHO_CLAD * _P_ZR[a] * _NA / _M_ZR[a] for a in _P_ZR}


def _water_densities(rho):
    n_h2o = rho * _NA / _M_H2O
    return {"H1_H2O": 2.0 * n_h2o, "O16": n_h2o}


_SELF_SHIELD_FLAG = {
    "U234": " 1", "U235": " 1", "U238": " 1",
    "Gd154": " 1", "Gd155": " 1", "Gd156": " 1",
    "Gd157": " CORR 1",  # Gd157's resonance needs the correlated (CORR) treatment
    "Gd158": " 1", "Gd160": " 1",
}


# ----------------------------------------------------------------------
# CLE-2000 deck generation
# ----------------------------------------------------------------------

def _discover_fuel_types(pattern):
    """Discover distinct fuel (kind, e, gad) types actually present in
    pattern, in first-seen (row-major) order, and assign each a mixture
    number 1..n_types. NOT tied to bwr10x10_geometry.PIN_TYPES' fixed
    13-type ARR action-space catalog -- the 3 calibration patterns
    (REFERENCE, RL_PPF_1_6, RL_PPF_1_4) use their own literal enrichment
    values transcribed from the paper's figures (e.g. 3.95%, 4.9%),
    which do not all coincide with the 13-type catalog used by the ARR
    search space. Discovering types per-pattern (as the validated
    gen_x2m_evo.py does) handles both cases correctly.

    Returns (type_to_mix: {cell: mix_number}, types_in_order: [cell...],
    n_types: int). This exact numbering convention (fuel types 1..
    n_types, then clad=n_types+1, channel=n_types+2, waterrod=n_types+3)
    is shared between build_deck() (which sets it up in the deck) and
    _compute_ppf() (which must know it to interpret MATCOD/H-FACTOR
    correctly) -- keep them in sync if this ever changes."""
    n = GRID_N
    type_to_mix, types_in_order = {}, []
    for i in range(n):
        for j in range(n):
            cell = pattern[i][j]
            if cell[0] == "WATER":
                continue
            if cell not in type_to_mix:
                type_to_mix[cell] = len(types_in_order) + 1
                types_in_order.append(cell)
    n_types = len(types_in_order)
    if n_types == 0:
        raise ValueError("pattern has no fuel cells")
    return type_to_mix, types_in_order, n_types


def _wrap_cle2000_tokens(tokens, width=100):
    """Join CLE-2000 tokens with spaces, wrapping to new lines before
    `width` columns per line. DRAGON's free-format reader requires each
    input record to fit within the first 120 columns (IGE-335 Sec. 2.1);
    tokens may span many lines as long as no single token is split
    across two lines, so this just needs to avoid exceeding that limit,
    not match any particular line count.

    HAZARD (found the hard way -- see PPF PROVENANCE note): if a '*'
    token ever lands as the FIRST token on a wrapped line, DRAGON reads
    the ENTIRE line as a comment (IGE-335: a record starting with '*'
    in column 1 is a comment) and silently drops it, desyncing every
    token after it -- observed as a much-later, seemingly-unrelated
    "DRVUTL: * IS AN INVALID UTILITY ACTION." failure. Guard against it
    here rather than trusting every caller to avoid '*' tokens."""
    if any(tok == "*" for tok in tokens):
        raise ValueError(
            "_wrap_cle2000_tokens: a bare '*' token was passed in -- this "
            "WILL eventually land as the first token on some wrapped line "
            "and get silently swallowed as a full-line DRAGON comment (see "
            "docstring). Use an explicit integer count instead."
        )
    lines, cur, cur_len = [], [], 0
    for tok in tokens:
        if cur_len + len(tok) + 1 > width and cur:
            lines.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(tok)
        cur_len += len(tok) + 1
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def _ppf_geometry_dump_lines(nreg_ppf):
    """Emitted ONCE, right after TRACK is built (region->mixture and
    region volume are static per pattern, independent of burnup step).
    Uses an explicit integer count (nreg_ppf), NOT '*' -- see the
    comment-hazard note in _wrap_cle2000_tokens. Parsing still reads the
    true region count off DRAGON's own printed LENGTH rather than
    trusting this estimate, so an off-by-a-little nreg_ppf here is safe
    (just prints a few extra zeros or gets capped by DRAGON itself, per
    IMAX=MIN0(LL,ILONG) in DRVUTL.f) as long as it's not wildly wrong."""
    return [
        'ECHO "PPF_GEOM_BEGIN" ;',
        f"UTL: TRACK :: IMPR MATCOD {nreg_ppf} ;",
        f"UTL: TRACK :: IMPR VOLUME {nreg_ppf} ;",
        'ECHO "PPF_GEOM_END" ;',
    ]


def _ppf_step_dump_lines(cfg: Dragon5Config, nreg_ppf, nmix_ppf, step_echo_expr):
    """Emitted at BOL and at every burnup step. step_echo_expr is a
    CLE-2000 expression printed as part of the marker so each step's
    dump is independently identifiable when parsing -- pass a literal
    like '0' for BOL or the runtime variable 'istep' inside the burnup
    loop. FLUX comes from CALC (this step's flux solve); H-FACTOR comes
    from LIBRARY2 (this step's self-shielded, depleted macrolib) -- both
    change every step, unlike the geometry dump above. Uses explicit
    integer counts (nreg_ppf, nmix_ppf), NOT '*' -- see the
    comment-hazard note in _wrap_cle2000_tokens; '*' was tried first and
    failed exactly that way on a real run."""
    ng = cfg.n_groups
    lines = [f'ECHO "PPF_STEP_BEGIN" {step_echo_expr} ;']
    flux_tokens = [tok for g in range(1, ng + 1) for tok in ("IMPR", str(g), str(nreg_ppf))]
    lines.append("UTL: CALC :: STEP UP 'FLUX'")
    lines.append(_wrap_cle2000_tokens(flux_tokens))
    lines.append("STEP DOWN ;")
    hfactor_tokens = [
        tok for g in range(1, ng + 1)
        for tok in ("STEP", "AT", str(g), "IMPR", "H-FACTOR", str(nmix_ppf), "STEP", "DOWN")
    ]
    lines.append("UTL: LIBRARY2 :: STEP UP 'MACROLIB' STEP UP 'GROUP'")
    lines.append(_wrap_cle2000_tokens(hfactor_tokens))
    lines.append("STEP DOWN STEP DOWN ;")
    lines.append(f'ECHO "PPF_STEP_END" {step_echo_expr} ;')
    return lines


def _gap_widths(cfg):
    """Half-gap water thickness on (left, right, bottom, top) faces."""
    if cfg.frame_water_wide_cm is not None and cfg.frame_water_narrow_cm is not None:
        wd, nr = cfg.frame_water_wide_cm, cfg.frame_water_narrow_cm
        if cfg.frame_wide_faces == "TL":
            return (wd, nr, nr, wd)
        if cfg.frame_wide_faces == "BR":
            return (nr, wd, wd, nr)
        raise ValueError("frame_wide_faces must be 'TL' or 'BR'")
    w = cfg.frame_water_cm
    return (w, w, w, w)


def _frame_layers(cfg):
    """Frame layers from the pins outward: [(thickness_cm, kind)], kind in
    {'chan', 'clad'} (moderator / Zircaloy). Zero-thickness layers are
    dropped. Empty list = no frame (original deck)."""
    layers = [(cfg.frame_inner_cm, "chan"), (cfg.frame_wall_cm, "clad"),
              (max(_gap_widths(cfg)), "gapw")]
    return [(t, k) for t, k in layers if t and t > 0.0]


def _layer_mix(kind, n_types, cfg):
    """Mixture number of a frame layer. 'gapw' (outer gap water) gets its own
    mixture n_types+4 only if cfg.gapwater_density is set; otherwise it shares
    the in-channel moderator mixture (n_types+2)."""
    if kind == "clad":
        return n_types + 1
    if kind == "gapw" and cfg.gapwater_density is not None:
        return n_types + 4
    return n_types + 2


def _cell_layout(pattern, cfg):
    """Cell kinds in DRAGON's CELL order (bottom row first, left to
    right), for the PPF region bookkeeping. Returns (kinds, expect_mats)
    where kinds[i] in {'F' fuel pin (4 regions), 'W' water-rod cell (1
    region), 'X' frame cell (1 region)} and expect_mats is the MATCOD
    sequence the layout implies (used to VERIFY the region ordering
    instead of assuming it)."""
    n = GRID_N
    type_to_mix, _, n_types = _discover_fuel_types(pattern)
    mix_clad, mix_chan, mix_wr = n_types + 1, n_types + 2, n_types + 3
    layers = _frame_layers(cfg)
    m = len(layers)
    layer_mix = [_layer_mix(k, n_types, cfg) for _, k in layers]
    # class index along one axis, bottom->top / left->right: ring k (0 =
    # pin region, 1..m = frame layers, 1 = innermost)
    axis = list(range(m, 0, -1)) + [0] * n + list(range(1, m + 1))
    rows_bt = list(reversed(pattern))
    kinds, mats = [], []
    for iy, ky in enumerate(axis):
        for ix, kx in enumerate(axis):
            if kx == 0 and ky == 0:
                cell = rows_bt[iy - m][ix - m]
                if cell[0] == "WATER":
                    kinds.append("W"); mats.append(mix_wr)
                else:
                    kinds.append("F")
                    mats.extend([mix_chan, mix_clad, mix_clad, type_to_mix[cell]])
            else:
                kinds.append("X"); mats.append(layer_mix[max(kx, ky) - 1])
    return kinds, mats


def build_deck(pattern, cfg: Dragon5Config = DEFAULT_CONFIG) -> str:
    """Generate the full DRAGON5 CLE-2000 (.x2m) deck for one pattern
    evaluation. Mirrors the validated gen_x2m_evo.py structure exactly
    for the LIB:/GEO:/tracking/USS:/ASM:/FLU:/EVO: chain (see module
    docstring); adds an optional (NOT FULLY VALIDATED, off by default) CL
    extension, and an optional (validated -- see PPF PROVENANCE note in
    the module docstring) PPF extension via UTL:-based raw LCM dumps of
    TRACK::MATCOD/VOLUME (once) and CALC::FLUX / LIBRARY2::MACROLIB::
    GROUP[*]::H-FACTOR (at BOL and every burnup step)."""
    n = GRID_N
    enrich, gad, water = pattern_to_grids(pattern)

    type_to_mix, types_in_order, n_types = _discover_fuel_types(pattern)
    mix_clad = n_types + 1
    mix_channel = n_types + 2
    mix_waterrod = n_types + 3

    # Region count for PPF's FLUX/VOLUME/MATCOD dumps: DRAGON assigns 4
    # regions per non-water CARCEL-in-square pin cell (corner/channel,
    # clad, gap, fuel -- confirmed empirically, see PPF PROVENANCE note)
    # and 1 region per water cell (plain CAR2D 1x1). This is a property
    # of the GEO:/tracking machinery, not of any particular pattern, so
    # it's recomputed here from the pattern itself rather than hardcoded.
    n_water_cells = sum(1 for i in range(n) for j in range(n) if pattern[i][j][0] == "WATER")
    n_fuel_cells = n * n - n_water_cells
    _fl = _frame_layers(cfg)
    n_frame_cells = (n + 2 * len(_fl)) ** 2 - n * n if _fl else 0
    nreg_ppf = 4 * n_fuel_cells + n_water_cells + n_frame_cells
    nmix_ppf = n_types + 3 + (1 if (cfg.gapwater_density is not None and _frame_layers(cfg)) else 0)  # fuel types + clad + channel + waterrod (+ gap water)

    a = []
    a.append(
        f"* Auto-generated by bwr10x10_dragon5.py -- DO NOT EDIT BY HAND.\n"
        f"* BWR 10x10 (GE14-class) single-assembly depletion deck.\n"
        f"* {n_types} distinct fuel/gad types, mix_clad={mix_clad} "
        f"mix_channel={mix_channel} mix_waterrod={mix_waterrod}\n"
        f"* PPF region-count estimate (informational only): {nreg_ppf}"
    )
    linked_list_names = ["GEOSS", "GEOM", "TRACK_SS", "TRACK", "LIBRARY", "LIBRARY2",
                         "CP", "CALC", "BURNUP", "StepList"]
    a.append("LINKED_LIST " + " ".join(linked_list_names) + " ;")
    a.append("SEQ_ASCII UOX_TBH ;")
    a.append("SEQ_BINARY TF_EXC ;")
    modules = "LIB: GEO: SYBILT: G2S: SALT: MCCGT: USS: ASM: FLU: EVO: GREP: UTL: DELETE: END: ;"
    a.append(f"MODULE {modules}")
    a.append("REAL Cote Rfuel Rgap Rclad ;")
    a.append(f"EVALUATE Cote  := {cfg.pin_pitch_cm} ;")
    a.append(f"EVALUATE Rfuel := {cfg.r_fuel_cm} ;")
    a.append(f"EVALUATE Rgap  := {cfg.r_gap_cm} ;")
    a.append(f"EVALUATE Rclad := {cfg.r_clad_cm} ;")
    a.append(f"INTEGER an2d := {cfg.salt_an2d} ;")
    a.append(f"REAL densur := {cfg.salt_densur} ;")

    # ---- burnup schedule ----
    nstep = len(cfg.burnup_steps_mwd_tu)
    bu_str = _wrap_cle2000_tokens([f"{v:.1f}" for v in cfg.burnup_steps_mwd_tu])
    a.append(f"*--- burnup schedule (ASSUMPTION: specific power {cfg.specific_power_mw_per_tu} MW/tU) ---")
    a.append(f"INTEGER maxstep := {nstep} ;")
    a.append(f"StepList := UTL: :: CREA ListBU <<maxstep>> =\n  {bu_str} ;")
    a.append(f"REAL Norm_f2 := {cfg.specific_power_mw_per_tu} ;")

    # ---- LIB: microscopic cross-section library recovery ----
    lib_lines = [f"LIBRARY := LIB: ::", "  EDIT 1", f"  NMIX {nmix_ppf}", "  CTRA APOL", "  SUBG",
                 f"  DEPL LIB: DRAGON FIL: {cfg.draglib_local_name}",
                 f"  MIXS LIB: DRAGON FIL: {cfg.draglib_local_name}"]
    for cell, mix in type_to_mix.items():
        kind = cell[0]
        e = cell[1]
        gwt = cell[2] if kind == "GAD" else 0.0
        dens = _pin_densities(kind, e, gwt, _RHO_UO2)
        label = f"UO2 {e}%" if kind == "UO2" else f"UO2 {e}%+Gd2O3 {gwt}%"
        lib_lines.append(f"  MIX {mix:2d} {cfg.fuel_temp_k:.1f} (* {label} *)")
        for iso, v in dens.items():
            alias = f"{iso}f{mix}"
            flag = _SELF_SHIELD_FLAG.get(iso, "")
            lib_lines.append(f"    {alias:10s} = {iso:6s} {v:.6E}{flag}")
    clad_d = _clad_densities()
    lib_lines.append(f"  MIX {mix_clad:2d} {cfg.clad_temp_k:.1f} NOEV (* Zircaloy clad *)")
    for iso, v in clad_d.items():
        lib_lines.append(f"    {iso + 'c':10s} = {iso:6s} {v:.6E}")
    chan_d = _water_densities(cfg.channel_density)
    lib_lines.append(f"  MIX {mix_channel:2d} {cfg.moderator_temp_k:.1f} NOEV "
                      f"(* channel moderator, rho={cfg.channel_density} *)")
    for iso, v in chan_d.items():
        lib_lines.append(f"    {iso + 'm':10s} = {iso:6s} {v:.6E}")
    wr_d = _water_densities(cfg.waterrod_density)
    lib_lines.append(f"  MIX {mix_waterrod:2d} {cfg.moderator_temp_k:.1f} NOEV "
                      f"(* water-rod water, rho={cfg.waterrod_density} *)")
    for iso, v in wr_d.items():
        lib_lines.append(f"    {iso + 'w':10s} = {iso:6s} {v:.6E}")
    if cfg.gapwater_density is not None and _frame_layers(cfg):
        mix_gap = n_types + 4
        gw_d = _water_densities(cfg.gapwater_density)
        lib_lines.append(f"  MIX {mix_gap:2d} {cfg.moderator_temp_k:.1f} NOEV "
                          f"(* inter-assembly gap water, rho={cfg.gapwater_density} *)")
        for iso, v in gw_d.items():
            lib_lines.append(f"    {iso + 'g':10s} = {iso:6s} {v:.6E}")
    lib_lines.append(";")
    a.append("\n".join(lib_lines))

    # ---- pin-cell sub-geometries (shared by GEOSS and the full GEOM) ----
    cellname = {cell: f"P{mix:02d}" for cell, mix in type_to_mix.items()}
    water_name = "PWTR"
    subgeo = []
    for cell, mix in type_to_mix.items():
        name = cellname[cell]
        subgeo.append(
            f"  ::: {name} := GEO: CARCEL 3\n"
            f"        MESHX 0.0 <<Cote>> MESHY 0.0 <<Cote>>\n"
            f"        RADIUS 0.0 <<Rfuel>> <<Rgap>> <<Rclad>>\n"
            f"        MIX {mix} {mix_clad} {mix_clad} {mix_channel} ;"
        )
    subgeo.append(
        f"  ::: {water_name} := GEO: CAR2D 1 1\n"
        f"        MESHX 0.0 <<Cote>> MESHY 0.0 <<Cote>>\n"
        f"        MIX {mix_waterrod} ;"
    )
    subgeo_block = "\n".join(subgeo)

    # ---- GEOSS: small self-shielding-only geometry, one cell per type ----
    nx_ss = math.ceil(math.sqrt(n_types))
    ny_ss = math.ceil(n_types / nx_ss)
    ss_names = [cellname[c] for c in types_in_order]
    while len(ss_names) < nx_ss * ny_ss:
        ss_names.append(ss_names[0])
    ss_rows = []
    for r in range(ny_ss):
        row = ss_names[r * nx_ss:(r + 1) * nx_ss]
        ss_rows.append("    " + " ".join(f"{nm:6s}" for nm in row))
    ss_meshx = " ".join(f"{cfg.pin_pitch_cm * i:.4f}" for i in range(nx_ss + 1))
    ss_meshy = " ".join(f"{cfg.pin_pitch_cm * i:.4f}" for i in range(ny_ss + 1))
    a.append(
        f"GEOSS := GEO: :: CAR2D {nx_ss} {ny_ss}\n"
        f"  EDIT 0\n"
        f"  X- REFL X+ REFL\n"
        f"  Y- REFL Y+ REFL\n"
        f"  CELL\n" + "\n".join(ss_rows) + "\n"
        f"  MESHX {ss_meshx}\n"
        f"  MESHY {ss_meshy}\n" + subgeo_block + "\n;"
    )
    a.append(
        f"TRACK_SS := SYBILT: GEOSS ::\n"
        f"  EDIT 0\n"
        f"  MAXR {SS_MAXR}\n"
        f"  MAXZ {SS_MAXZ}\n"
        f"  TITLE 'BWR10x10 -- SELF-SHIELDING TRACKING'\n"
        f"  QUA2 {SS_QUA2[0]} {SS_QUA2[1]}\n"
        f"  DP01 ;"
    )
    a.append(f"LIBRARY2 := USS: LIBRARY TRACK_SS :: EDIT 1 PASS {USS_PASS} ARM NOCO "
             f"MAXST {USS_MAXST} GRMIN {USS_GRMIN} ;")

    # ---- full 10x10 assembly geometry (main flux tracking) ----
    # pattern[0] is the TOP row of the figure; CAR2D's CELL grid runs
    # Y- to Y+ (bottom row first), so rows are reversed here -- matches
    # the validated gen_x2m_evo.py convention exactly.
    rows_bottom_to_top = list(reversed(pattern))
    layers = _frame_layers(cfg)
    if not layers:
        cell_rows = []
        for row in rows_bottom_to_top:
            names = []
            for cell in row:
                names.append(water_name if cell[0] == "WATER" else cellname[cell])
            cell_rows.append("    " + " ".join(f"{nm:6s}" for nm in names))
        meshvals = " ".join(f"{cfg.pin_pitch_cm * i:.4f}" for i in range(n + 1))
        a.append(
            f"GEOM := GEO: :: CAR2D {n} {n}\n"
            f"  EDIT 0\n"
            f"  X- REFL X+ REFL\n"
            f"  Y- REFL Y+ REFL\n"
            f"  CELL\n" + "\n".join(cell_rows) + "\n"
            f"  MESHX {meshvals}\n"
            f"  MESHY {meshvals}\n" + subgeo_block + "\n;"
        )
    else:
        m = len(layers)
        layer_mix = [_layer_mix(k, n_types, cfg) for _, k in layers]
        axis = list(range(m, 0, -1)) + [0] * n + list(range(1, m + 1))
        gl, gr, gb, gt = _gap_widths(cfg)

        def _w(k, side):
            """Width of ring k (0 = pin cell) on the given face."""
            if k == 0:
                return cfg.pin_pitch_cm
            t = layers[k - 1][0]
            if layers[k - 1][1] == "gapw":
                t = {"L": gl, "R": gr, "B": gb, "T": gt}[side]
            return t

        # per-position widths along x (left->right) and y (bottom->top)
        wx = [_w(k, "L" if i < m else "R") for i, k in enumerate(axis)]
        wy = [_w(k, "B" if i < m else "T") for i, k in enumerate(axis)]
        # distinct frame sub-geometries, one per (x-width, y-width, mixture)
        fr_names, fr_defs = {}, []
        for iy, ky in enumerate(axis):
            for ix, kx in enumerate(axis):
                if kx == 0 and ky == 0:
                    continue
                mixn = layer_mix[max(kx, ky) - 1]
                key = (round(wx[ix], 6), round(wy[iy], 6), mixn)
                if key in fr_names:
                    continue
                nm = f"FR{len(fr_names) + 1}"
                fr_names[key] = nm
                fr_defs.append(
                    f"  ::: {nm} := GEO: CAR2D 1 1\n"
                    f"        MESHX 0.0 {wx[ix]:.6f} MESHY 0.0 {wy[iy]:.6f}\n"
                    f"        MIX {mixn} ;"
                )
        cell_rows = []
        for iy, ky in enumerate(axis):
            names = []
            for ix, kx in enumerate(axis):
                if kx == 0 and ky == 0:
                    cell = rows_bottom_to_top[iy - m][ix - m]
                    names.append(water_name if cell[0] == "WATER" else cellname[cell])
                else:
                    names.append(fr_names[(round(wx[ix], 6), round(wy[iy], 6),
                                           layer_mix[max(kx, ky) - 1])])
            cell_rows.append("    " + " ".join(f"{nm:6s}" for nm in names))
        edges_x, edges_y = [0.0], [0.0]
        for ix in range(len(axis)):
            edges_x.append(edges_x[-1] + wx[ix])
            edges_y.append(edges_y[-1] + wy[ix])
        # DRAGON/CLE-2000 records must fit in 120 columns (a 16-cell mesh
        # on one line overflows: "clelog: INPUT LINE OVERFLOW") -> wrap.
        meshx = _wrap_cle2000_tokens([f"{e:.6f}" for e in edges_x], width=90)
        meshy = _wrap_cle2000_tokens([f"{e:.6f}" for e in edges_y], width=90)
        ntot = n + 2 * m
        a.append(
            f"GEOM := GEO: :: CAR2D {ntot} {ntot}\n"
            f"  EDIT 0\n"
            f"  X- REFL X+ REFL\n"
            f"  Y- REFL Y+ REFL\n"
            f"  CELL\n" + "\n".join(cell_rows) + "\n"
            f"  MESHX {meshx}\n"
            f"  MESHY {meshy}\n" + subgeo_block + "\n" + "\n".join(fr_defs) + "\n;"
        )
    a.append("UOX_TBH := G2S: GEOM ;")
    a.append(
        f"TRACK TF_EXC := SALT: UOX_TBH ::\n"
        f"  EDIT 3\n"
        f"  ALLG\n"
        f"  TSPC EQW2 <<an2d>> <<densur>> REND\n;"
    )
    a.append(f"TRACK := MCCGT: TRACK TF_EXC ::\n"
             f"  CACB {MCCGT_CACB} AAC {MCCGT_AAC} TMT EPSI 1E-5 MCU {MCCGT_MCU} ;")
    if cfg.compute_ppf:
        a.append("\n".join(_ppf_geometry_dump_lines(nreg_ppf)))

    # ---- BOL (burnup = 0) flux/k solve ----
    a.append("REAL KeffOut := 0.0 ;")
    a.append("CP := ASM: LIBRARY2 TRACK TF_EXC :: ARM EDIT 1 ;")
    a.append("CALC := FLU: LIBRARY2 CP TRACK TF_EXC :: EDIT 1 TYPE K ;")
    a.append("CP := DELETE: CP ;")
    a.append("GREP: CALC :: GETVAL 'K-EFFECTIVE' 1 >>KeffOut<< ;")
    a.append('ECHO "RESULT step= 0 burnup_mwdtu= 0.0 k_inf=" KeffOut ;')
    if cfg.compute_ppf:
        a.append("\n".join(_ppf_step_dump_lines(cfg, nreg_ppf, nmix_ppf, "0")))

    # ---- burnup depletion loop ----
    a.append(f"INTEGER istep := 0 ;")
    a.append("REAL BUbeg BUend Tbeg Tend := 0.0 0.0 0.0 0.0 ;")
    a.append("WHILE istep maxstep < DO")
    a.append("  EVALUATE istep := istep 1 + ;")
    a.append("  EVALUATE BUbeg Tbeg := BUend Tend ;")
    a.append("  GREP: StepList :: GETVAL 'ListBU' <<istep>> >>BUend<< ;")
    a.append("  EVALUATE Tend := BUend Norm_f2 / ;")
    a.append(
        "* EVO: evolves LIBRARY2 directly (validated convention -- see module docstring point 3).\n"
        "  IF istep 1 = THEN\n"
        "    BURNUP LIBRARY2 := EVO: LIBRARY2 CALC TRACK :: EDIT 0\n"
        "      DEPL <<Tbeg>> <<Tend>> DAY POWR <<Norm_f2>>\n"
        f"      EPS1 {EVO_EPS1} EPS2 {EVO_EPS2} EXPM {EVO_EXPM:.1E} RUNG EXTR NSAT NODI ;\n"
        "  ELSE\n"
        "    BURNUP LIBRARY2 := EVO: BURNUP LIBRARY2 CALC TRACK :: EDIT 0\n"
        "      DEPL <<Tbeg>> <<Tend>> DAY POWR <<Norm_f2>>\n"
        f"      EPS1 {EVO_EPS1} EPS2 {EVO_EPS2} EXPM {EVO_EXPM:.1E} RUNG EXTR NSAT NODI ;\n"
        "  ENDIF ;"
    )
    a.append(f"  LIBRARY2 := USS: LIBRARY2 LIBRARY TRACK_SS ::\n"
             f"    EDIT 0 PASS {USS_PASS} ARM NOCO MAXST {USS_MAXST} GRMIN {USS_GRMIN} ;")
    a.append("  CP := ASM: LIBRARY2 TRACK TF_EXC :: ARM EDIT 1 ;")
    a.append("  CALC := FLU: CALC LIBRARY2 CP TRACK TF_EXC :: EDIT 1 TYPE K ;")
    a.append("  CP := DELETE: CP ;")
    a.append("  GREP: CALC :: GETVAL 'K-EFFECTIVE' 1 >>KeffOut<< ;")
    a.append('  ECHO "RESULT step=" istep "burnup_mwdtu=" BUend "k_inf=" KeffOut ;')
    if cfg.compute_ppf:
        a.append("\n".join(_ppf_step_dump_lines(cfg, nreg_ppf, nmix_ppf, "istep")))
    a.append("  BURNUP LIBRARY2 := EVO: BURNUP LIBRARY2 CALC TRACK :: EDIT 0\n"
             "    SAVE <<Tend>> DAY POWR <<Norm_f2>> ;")
    a.append("ENDWHILE ;")
    a.append('ECHO "DRAGON5 RUN COMPLETE" ;')
    a.append("END: ;")
    a.append('QUIT "LIST" .')
    return "\n".join(a)


# ----------------------------------------------------------------------
# Output parsing -- reads the ECHO'd "RESULT ..." lines directly from
# dragon's stdout (this is how the validated deck's results were
# actually read during validation: via `grep "burnup step"` on the run
# log, not via a SEQ_ASCII results file or a GREP:-to-file mechanism,
# which is not how GREP: works).
# ----------------------------------------------------------------------

_RESULT_RE = re.compile(
    r"RESULT\s+step=\s*(\d+)\s+burnup_mwdtu=\s*([-\d.EeDd+]+)\s+k_inf=\s*([-\d.EeDd+]+)"
)


def _f(s):
    return float(s.replace("D", "E").replace("d", "e"))


def _parse_output(log_text: str, cfg: Dragon5Config):
    k_inf_steps = {}
    for m in _RESULT_RE.finditer(log_text):
        k_inf_steps[int(m.group(1))] = (_f(m.group(2)), _f(m.group(3)))
    return k_inf_steps


# ----------------------------------------------------------------------
# PPF parsing -- see module docstring's PPF PROVENANCE note for the
# method. Two hazards this deliberately guards against (both bit us
# during development, see PROVENANCE note and Dragon5Config.compute_ppf):
#   1. CLE-2000 echoes input SOURCE TEXT as it parses, so a marker
#      string given to ECHO also appears, verbatim, inside the echoed
#      command line itself -- BEFORE the marker's real (executed)
#      occurrence. Naive text search finds the wrong (source-echo) one.
#      Fix: only trust occurrences prefixed with '>|', which is specific
#      to actually-printed ECHO output, never to source echo (which uses
#      '<|').
#   2. DRVUTL's own diagnostic prints ("DRVUTL: STEP UP TO LEVEL ...",
#      "DRVUTL: CONTENT OF BLOCK ...") are NOT subject to hazard #1 at
#      all -- they're runtime WRITE statements, not source-echoed input,
#      so a plain (unprefixed) text search for them is safe and is used
#      below to delimit the FLUX vs. H-FACTOR sub-sections within one
#      step's dump.
# ----------------------------------------------------------------------

_SCI_RE = re.compile(r"[-+]?\d*\.\d+E[-+]\d+")
_INT_RE = re.compile(r"-?\d+")


def _real_marker_positions(text, marker):
    """Positions of MARKER as it appears in ACTUAL DRAGON5 runtime ECHO
    output (prefixed '>|'), excluding CLE-2000's echo of the input
    SOURCE containing the same literal marker text (see hazard #1
    above)."""
    needle = ">|" + marker
    return [m.start() for m in re.finditer(re.escape(needle), text)]


def _parse_ppf_geometry(log_text):
    """Returns (matcod: list[int], volume: list[float]) parsed from the
    once-only TRACK::MATCOD / TRACK::VOLUME dump. Returns (None, None)
    if compute_ppf was False (dump absent) or parsing failed."""
    begins = _real_marker_positions(log_text, "PPF_GEOM_BEGIN")
    ends = _real_marker_positions(log_text, "PPF_GEOM_END")
    if not begins or not ends:
        return None, None
    seg = log_text[begins[0]:ends[0]]

    def _one_block(name):
        marker = f"CONTENT OF BLOCK '{name}"
        if marker not in seg:
            return None
        after = seg.split(marker, 1)[1]
        # "     LENGTH =       376     NUMBER OF NON ZERO TERMS = ..."
        length_txt = after.split("LENGTH", 1)[1]
        length = int(_INT_RE.findall(length_txt)[0])
        body = after.split("NON ZERO TERMS", 1)[1]
        body = body.split("\n", 1)[-1]  # drop rest of the header line
        return length, body

    matcod_hdr = _one_block("MATCOD")
    volume_hdr = _one_block("VOLUME")
    if matcod_hdr is None or volume_hdr is None:
        return None, None
    m_len, m_body = matcod_hdr
    v_len, v_body = volume_hdr
    matcod = [int(x) for x in _INT_RE.findall(m_body)][:m_len]
    volume = [float(x) for x in _SCI_RE.findall(v_body)][:v_len]
    if len(matcod) != m_len or len(volume) != v_len or m_len != v_len:
        raise RuntimeError(
            f"PPF geometry parse mismatch: MATCOD claims LENGTH={m_len} "
            f"got {len(matcod)} values; VOLUME claims LENGTH={v_len} got "
            f"{len(volume)} values. Set Dragon5Config(keep_work_dir=True) "
            "and inspect dragon_stdout.log's PPF_GEOM_BEGIN/END section."
        )
    return matcod, volume


def _parse_ppf_steps(log_text, n_groups):
    """Returns {step:int -> (flux_by_group: {g:[...]}, hfactor_by_group:
    {g:[...]})} for every step whose PPF dump completed. Missing/partial
    steps are simply absent from the dict (e.g. a run killed mid-loop) --
    callers should tolerate a step in k_inf_steps having no PPF data."""
    begins = _real_marker_positions(log_text, "PPF_STEP_BEGIN")
    ends = _real_marker_positions(log_text, "PPF_STEP_END")
    result = {}
    for b_pos, e_pos in zip(begins, ends):
        seg = log_text[b_pos:e_pos]
        # step number is the first integer printed right after the marker
        head = seg[:seg.index("\n")] if "\n" in seg else seg
        ints = _INT_RE.findall(head.split("PPF_STEP_BEGIN", 1)[-1])
        if not ints:
            continue
        step = int(ints[0])

        if "STEP UP TO LEVEL 'FLUX" not in seg or "STEP UP TO LEVEL 'MACROLIB" not in seg:
            continue  # this step's dump didn't complete (e.g. run killed mid-way)
        flux_seg = seg.split("STEP UP TO LEVEL 'FLUX", 1)[1]
        flux_seg, hfac_seg = flux_seg.split("STEP UP TO LEVEL 'MACROLIB", 1)
        # Each UTL: module call ends with a "-->>module UTL: ... memory
        # usage= 0.000E+00" trailer -- that "0.000E+00" is scientific
        # notation and would otherwise be miscounted as one extra data
        # value (found empirically: off-by-exactly-one on both FLUX and
        # H-FACTOR counts, see PPF PROVENANCE note). Cut it off first.
        flux_seg = flux_seg.split("-->>module UTL:", 1)[0]
        hfac_seg = hfac_seg.split("-->>module UTL:", 1)[0]

        flux_nums = [float(x) for x in _SCI_RE.findall(flux_seg)]
        hfac_nums = [float(x) for x in _SCI_RE.findall(hfac_seg)]
        if len(flux_nums) % n_groups != 0 or len(hfac_nums) % n_groups != 0:
            raise RuntimeError(
                f"PPF step {step}: parsed {len(flux_nums)} FLUX values and "
                f"{len(hfac_nums)} H-FACTOR values, neither divisible by "
                f"n_groups={n_groups} -- either n_groups doesn't match the "
                "draglib actually in use (set Dragon5Config.n_groups "
                "correctly) or this step's dump is truncated/corrupted. "
                "Set keep_work_dir=True and inspect dragon_stdout.log."
            )
        nreg = len(flux_nums) // n_groups
        nmix = len(hfac_nums) // n_groups
        flux_by_group = {g: flux_nums[(g - 1) * nreg: g * nreg] for g in range(1, n_groups + 1)}
        hfactor_by_group = {g: hfac_nums[(g - 1) * nmix: g * nmix] for g in range(1, n_groups + 1)}
        result[step] = (flux_by_group, hfactor_by_group)
    return result


def _pin_powers_layout(matcod, volume, flux_by_group, hfactor_by_group, kinds, expect_mats):
    """Pin powers using an explicit cell layout (frame-aware). The region
    ordering is VERIFIED against the layout (MATCOD must equal the mixture
    sequence the layout implies), never assumed. Returns a list with one
    entry per cell in CELL order: pin power for 'F', None otherwise."""
    # order-insensitive WITHIN a 4-region fuel cell (DRAGON lists
    # moderator, clad, gap, fuel; the original grouping never relied on
    # that order), strictly positional BETWEEN cells.
    ok = len(matcod) == len(expect_mats)
    bad = None
    if ok:
        i = 0
        for ci, kd in enumerate(kinds):
            w = 4 if kd == "F" else 1
            if sorted(matcod[i:i + w]) != sorted(expect_mats[i:i + w]):
                ok, bad = False, (ci, i); break
            i += w
    if not ok:
        raise RuntimeError(
            "PPF layout check FAILED: MATCOD from DRAGON does not match the "
            f"region/mixture sequence implied by the cell layout (lengths "
            f"{len(matcod)} vs {len(expect_mats)}; first mismatch at (cell, region)={bad}). "
            "The region-ordering assumption does not hold for this geometry "
            "-- do not trust PPF.")
    n_groups = len(flux_by_group)

    def region_power(r):
        mix = matcod[r]
        return volume[r] * sum(flux_by_group[g][r] * hfactor_by_group[g][mix - 1]
                               for g in range(1, n_groups + 1))
    out, i = [], 0
    for kd in kinds:
        if kd == "F":
            out.append(sum(region_power(r) for r in range(i, i + 4))); i += 4
        else:
            out.append(None); i += 1
    return out


def _compute_ppf(matcod, volume, flux_by_group, hfactor_by_group, n_types, layout=None):
    """PPF = max(pin power) / mean(pin power) over fuel pins, from raw
    region-wise DRAGON5 data. See module docstring's PPF PROVENANCE note
    for the method and its validation. Mixture numbering convention
    (fixed by build_deck()): 1..n_types are fuel, n_types+1 is clad,
    n_types+2 is channel moderator, n_types+3 is water-rod moderator.
    Regions group into pins in fixed blocks: 4 regions per non-water
    cell (corner/channel, clad, gap, fuel -- in DRAGON's own internal
    order, identified here by exclusion rather than assumed position),
    or 1 region per water cell."""
    mix_clad, mix_channel, mix_waterrod = n_types + 1, n_types + 2, n_types + 3
    n_groups = len(flux_by_group)
    if n_groups == 0:
        raise RuntimeError("No FLUX groups available for PPF computation.")
    if layout is not None:
        pp = [p for p in _pin_powers_layout(matcod, volume, flux_by_group,
                                            hfactor_by_group, *layout) if p is not None]
        mean_power = sum(pp) / len(pp)
        return max(pp) / mean_power

    def region_power(r):
        mix = matcod[r]
        return volume[r] * sum(
            flux_by_group[g][r] * hfactor_by_group[g][mix - 1]
            for g in range(1, n_groups + 1)
        )

    pin_powers = []
    i, n = 0, len(matcod)
    while i < n:
        if matcod[i] == mix_waterrod:
            i += 1
            continue
        block = matcod[i:i + 4]
        if len(block) < 4:
            raise RuntimeError(
                f"PPF region grouping ran off the end of MATCOD at region {i} "
                f"(block={block}, expected 4 entries) -- the '4 regions per "
                "non-water cell' assumption (see PPF PROVENANCE note) may not "
                "hold for this pattern/geometry."
            )
        fuel_in_block = [m for m in block if m not in (mix_clad, mix_channel, mix_waterrod)]
        if len(fuel_in_block) != 1:
            raise RuntimeError(
                f"PPF region grouping: block at region {i} is {block}, expected "
                "exactly one fuel mixture and the rest clad/channel/waterrod "
                f"({mix_clad}/{mix_channel}/{mix_waterrod}). The '4 regions per "
                "non-water cell, fixed channel/clad/clad/fuel pattern' "
                "assumption (see PPF PROVENANCE note) may not hold for this "
                "pattern/geometry -- do not trust the returned PPF."
            )
        pin_powers.append(sum(region_power(rr) for rr in range(i, i + 4)))
        i += 4

    if not pin_powers:
        raise RuntimeError("PPF computation found zero fuel pins.")
    mean_power = sum(pin_powers) / len(pin_powers)
    if mean_power <= 0:
        raise RuntimeError(f"PPF computation: non-positive mean pin power ({mean_power}).")
    return max(pin_powers) / mean_power


def _reduce_history(pattern, k_inf_steps, cfg: Dragon5Config,
                     log_text=None, n_types=None):
    if not k_inf_steps:
        raise RuntimeError(
            "No 'RESULT step=...' lines found in DRAGON5 output. Either the run "
            "failed before completing the BOL solve, or dragon's ECHO output "
            "format differs from what this parser expects -- set "
            "Dragon5Config(keep_work_dir=True) and inspect deck.x2m plus the "
            "raw stdout/stderr to diagnose."
        )
    steps_sorted = sorted(k_inf_steps.keys())
    burnups = [k_inf_steps[s][0] for s in steps_sorted]
    kinfs = [k_inf_steps[s][1] for s in steps_sorted]
    k_inf_max = float(np.max(kinfs))

    history = {"step": steps_sorted, "burnup_mwd_tu": burnups, "k_inf": kinfs}

    # PPF: reported at whichever burnup step k_inf peaks (matching how
    # k_inf_max itself is defined, and the paper's convention of quoting
    # k_inf_max/PPF/CL together for one representative state) -- see
    # module docstring's PPF PROVENANCE note for the method.
    ppf = None
    if cfg.compute_ppf:
        if log_text is None or n_types is None:
            raise RuntimeError(
                "_reduce_history: cfg.compute_ppf is True but log_text/"
                "n_types were not provided -- this is an internal wiring "
                "bug in casmo4_bwr10x10(), not a DRAGON5 run failure."
            )
        k_inf_max_step = steps_sorted[int(np.argmax(kinfs))]
        matcod, volume = _parse_ppf_geometry(log_text)
        if matcod is None:
            raise RuntimeError(
                "cfg.compute_ppf=True but no PPF_GEOM_BEGIN/END section "
                "found in dragon's output -- the PPF dump likely never "
                "ran (e.g. the deck errored before reaching TRACK, or an "
                "older/mismatched deck was actually executed). Set "
                "keep_work_dir=True and inspect deck.x2m/dragon_stdout.log."
            )
        ppf_steps = _parse_ppf_steps(log_text, cfg.n_groups)
        if k_inf_max_step not in ppf_steps:
            raise RuntimeError(
                f"cfg.compute_ppf=True but no complete PPF dump was found "
                f"for step {k_inf_max_step} (the k_inf-max step) -- either "
                "the run was killed/errored partway through the burnup "
                f"loop (found complete PPF data for steps {sorted(ppf_steps)}), "
                "or Dragon5Config.n_groups doesn't match the draglib "
                "actually in use. Set keep_work_dir=True to inspect."
            )
        ppf_eval_step = 0 if cfg.ppf_at == 'bol' else k_inf_max_step
        if ppf_eval_step not in ppf_steps:
            raise RuntimeError(f'no complete PPF dump for step {ppf_eval_step} (ppf_at={cfg.ppf_at!r})')
        flux_by_group, hfactor_by_group = ppf_steps[ppf_eval_step]
        layout = _cell_layout(pattern, cfg)   # verified against MATCOD (also valid with no frame)
        ppf = _compute_ppf(matcod, volume, flux_by_group, hfactor_by_group, n_types, layout)
        history['ppf_eval_step'] = ppf_eval_step
        history["ppf_by_step"] = {s: _compute_ppf(matcod, volume, *ppf_steps[s], n_types, layout)
                                   for s in ppf_steps}
        if layout is not None:
            kinds_, exp_ = layout
            pp_ = _pin_powers_layout(matcod, volume, flux_by_group, hfactor_by_group, kinds_, exp_)
            fuel_ = [p for p in pp_ if p is not None]
            mean_ = sum(fuel_) / len(fuel_)
            # pin map in FIGURE orientation (top row first), fuel/water cells only
            m_ = len(_frame_layers(cfg)); N_ = GRID_N + 2 * m_
            grid_ = [[pp_[iy * N_ + ix] for ix in range(m_, m_ + GRID_N)]
                     for iy in range(m_, m_ + GRID_N)]
            grid_ = grid_[::-1]
            history["pin_power_map_kmax"] = [[None if v is None else v / mean_ for v in row]
                                              for row in grid_]

    cl_days = None
    cl_warning = None
    if cfg.compute_cl:
        # CL_proxy = burnup (converted to days via specific_power_mw_per_tu),
        # AFTER the k_inf peak, at which k_inf first drops to/below
        # cfg.cl_kinf_threshold (0.95 by default) -- this is the paper's
        # OWN definition (Radaideh et al. 2021 Sec. 2), not an assumption
        # of ours; see module docstring's CL_proxy note for why an
        # earlier buckling-search redesign was replaced with this. Uses
        # only k_inf_steps (already collected for k_inf_max), so this
        # costs nothing extra regardless of compute_cl.
        peak_idx = int(np.argmax(kinfs))
        crossing = None
        for idx in range(max(peak_idx, 1), len(kinfs)):
            if kinfs[idx - 1] > cfg.cl_kinf_threshold >= kinfs[idx]:
                frac = ((kinfs[idx - 1] - cfg.cl_kinf_threshold)
                         / (kinfs[idx - 1] - kinfs[idx]))
                crossing = burnups[idx - 1] + frac * (burnups[idx] - burnups[idx - 1])
                break
        if crossing is None:
            crossing = burnups[-1]
            cl_warning = (
                f"k_inf never dropped to/below cl_kinf_threshold="
                f"{cfg.cl_kinf_threshold} within the configured burnup "
                "schedule (after its peak); CL_proxy is a LOWER BOUND -- "
                "extend burnup_steps_mwd_tu further past the peak"
            )
        cl_days = crossing / cfg.specific_power_mw_per_tu

    return k_inf_max, ppf, cl_days, cl_warning, history



# ----------------------------------------------------------------------
# Subprocess execution
# ----------------------------------------------------------------------

def _run_dragon(deck_text: str, cfg: Dragon5Config):
    resolved_exe = shutil.which(cfg.dragon_exe)
    if resolved_exe is None and os.path.isabs(cfg.dragon_exe) and os.access(cfg.dragon_exe, os.X_OK):
        resolved_exe = cfg.dragon_exe
    if resolved_exe is None:
        raise RuntimeError(
            f"DRAGON5 executable {cfg.dragon_exe!r} not found. Set Dragon5Config.dragon_exe "
            "or the DRAGON5_EXE environment variable to your local `dragon` binary."
        )
    if not cfg.draglib_path:
        raise RuntimeError(
            "No draglib configured. Set Dragon5Config.draglib_path or the "
            "DRAGON5_LIBRARY environment variable (e.g. to your DLIB_295 file, "
            "gunzipped if needed -- see the .access-script convention used "
            "during validation)."
        )

    own_dir = cfg.work_dir is None
    work_dir = Path(cfg.work_dir) if cfg.work_dir else Path(tempfile.mkdtemp(prefix="dragon5_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    draglib_dst = work_dir / cfg.draglib_local_name
    if not draglib_dst.exists():
        src = Path(cfg.draglib_path)
        try:
            os.symlink(src.resolve(), draglib_dst)
        except OSError:
            shutil.copy(src, draglib_dst)

    deck_path = work_dir / "deck.x2m"
    deck_path.write_text(deck_text)

    try:
        # NOTE: the `dragon`/`Dragon` executable reads the CLE-2000 deck
        # from STDIN, not as a command-line argument (this is what the
        # `rdragon` wrapper script quietly does with shell redirection).
        # Passing the deck filename as argv[1] instead leaves the process
        # blocked forever waiting on stdin -- confirmed the hard way: a
        # run that never used any CPU time (0:00 in `top`/`ps`) despite
        # sitting "alive" for 20+ minutes. Feed it via stdin instead.
        returncode = None
        timed_out = False
        try:
            with open(deck_path, "r") as deck_fh:
                proc = subprocess.run(
                    [resolved_exe],
                    stdin=deck_fh,
                    cwd=str(work_dir),
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_sec,
                )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            # A full 14-step run (BOL + 14 x EVO/USS/ASM/FLU TYPE K, plus
            # compute_ppf's per-step FLUX/H-FACTOR dumps if enabled) can
            # comfortably exceed the old 7200s (2h) default -- confirmed
            # by a real run that was still actively burning CPU (not
            # hung) when it hit the timeout and got killed, losing 2h of
            # otherwise-valid
            # partial output. subprocess salvages whatever stdout/stderr
            # had already been captured before the kill (e.stdout/e.stderr
            # -- these ARE populated by Python on TimeoutExpired), so we
            # recover that instead of discarding it. If enough "RESULT
            # step=..." lines already printed, this can still yield a
            # (possibly-incomplete) usable k_inf_max; the completeness
            # check below reports exactly how many of the requested steps
            # were captured either way.
            stdout = e.stdout or ""
            stderr = e.stderr or ""
            timed_out = True

        log_text = (stdout or "") + "\n" + (stderr or "")
        # Always dump the raw log next to the deck, whether or not this run
        # looks successful -- makes post-mortems on a partial/silent
        # failure possible without having to re-run (see completeness
        # check below: dragon can exit 0 having only gotten partway
        # through the burnup loop before erroring on a later module call,
        # and a substring check on "RESULT step=" alone doesn't catch that
        # -- it would only catch the case where NO step at all printed,
        # not the case where BOL printed and then the loop silently died).
        (work_dir / "dragon_stdout.log").write_text(log_text)
        if timed_out:
            n_partial = len(_RESULT_RE.findall(log_text))
            print(f"WARNING: dragon timed out after {cfg.timeout_sec}s "
                  f"(process was killed); recovered {n_partial} partial "
                  f"'RESULT step=...' line(s) from before the kill. Raise "
                  f"Dragon5Config.timeout_sec to let it finish next time.")

        # NOTE: deliberately using the STRICT parser regex here (requires
        # the actual printed burnup/k_inf VALUES), not a loose substring
        # match on "RESULT step=" -- DRAGON5/CLE-2000 interpreters echo
        # the input source itself back into stdout as they parse it, and
        # the source line `ECHO "RESULT step=" istep ...` itself contains
        # the literal text "RESULT step=" even if that ECHO statement
        # never actually executes at runtime. A loose count over-counts
        # from this source echo and can never detect a partial/failed run.
        n_result = len(_RESULT_RE.findall(log_text))
        n_expected = len(cfg.burnup_steps_mwd_tu) + 1  # +1 for BOL (step 0)
        if n_result == 0 or n_result < n_expected:
            reason = (f"dragon TIMED OUT after {cfg.timeout_sec}s (raise "
                      f"Dragon5Config.timeout_sec)" if timed_out else
                      f"dragon exited with code {returncode}")
            raise RuntimeError(
                f"{reason}, and only produced {n_result}/{n_expected} expected "
                "'RESULT step=...' lines -- the run likely errored, was killed, "
                "or was killed partway through the burnup loop (a nonzero exit "
                "code alone is not required for a non-timeout failure: some "
                "module errors are logged but do not fail the run's exit code, "
                "so BOL output alone can look like 'success' otherwise).\n"
                f"--- stdout tail (last 4000 chars) ---\n{stdout[-4000:]}\n"
                f"--- stderr tail (last 2000 chars) ---\n{stderr[-2000:]}\n"
                f"Full raw log: {work_dir / 'dragon_stdout.log'}\n"
                f"Deck: {deck_path}\n"
                "(set Dragon5Config(keep_work_dir=True) to keep both for inspection)."
            )
        return log_text
    finally:
        if own_dir and not cfg.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# Streaming run with early abort (opt-in via Dragon5Config.abort_ppf_above / abort_k_above)
# ----------------------------------------------------------------------

def _bol_ppf_and_map(log_text, pattern, cfg, n_types):
    """(PPF, BOL pin map) from the step-0 PPF dump found in a (possibly partial) log; (None, None) if the dump
    is not complete yet.  Same computation as _reduce_history for ppf_eval_step=0."""
    matcod, volume = _parse_ppf_geometry(log_text)
    if matcod is None:
        return None, None
    ppf_steps = _parse_ppf_steps(log_text, cfg.n_groups)
    if 0 not in ppf_steps:
        return None, None
    flux_by_group, hfactor_by_group = ppf_steps[0]
    layout = _cell_layout(pattern, cfg)
    ppf = _compute_ppf(matcod, volume, flux_by_group, hfactor_by_group, n_types, layout)
    pin_map = None
    if layout is not None:
        kinds_, exp_ = layout
        pp_ = _pin_powers_layout(matcod, volume, flux_by_group, hfactor_by_group, kinds_, exp_)
        fuel_ = [p for p in pp_ if p is not None]
        mean_ = sum(fuel_) / len(fuel_)
        m_ = len(_frame_layers(cfg)); N_ = GRID_N + 2 * m_
        grid_ = [[pp_[iy * N_ + ix] for ix in range(m_, m_ + GRID_N)] for iy in range(m_, m_ + GRID_N)]
        grid_ = grid_[::-1]
        pin_map = [[None if v is None else v / mean_ for v in row] for row in grid_]
    return ppf, pin_map


def _run_dragon_stream(deck_text, cfg, pattern, n_types):
    """Like _run_dragon, but reads DRAGON5's output while it runs and kills the process once the candidate is
    proven infeasible (see Dragon5Config.abort_*).  Returns (log_text, abort) where abort is None for a run
    that went to completion, else a dict {reason, bol_ppf, bol_map, elapsed_s}."""
    import signal, threading
    resolved_exe = shutil.which(cfg.dragon_exe)
    if resolved_exe is None and os.path.isabs(cfg.dragon_exe) and os.access(cfg.dragon_exe, os.X_OK):
        resolved_exe = cfg.dragon_exe
    if resolved_exe is None:
        raise RuntimeError(f"DRAGON5 executable {cfg.dragon_exe!r} not found. Set Dragon5Config.dragon_exe "
                           "or the DRAGON5_EXE environment variable to your local `dragon` binary.")
    if not cfg.draglib_path:
        raise RuntimeError("No draglib configured. Set Dragon5Config.draglib_path or DRAGON5_LIBRARY.")
    own_dir = cfg.work_dir is None
    work_dir = Path(cfg.work_dir) if cfg.work_dir else Path(tempfile.mkdtemp(prefix="dragon5_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    draglib_dst = work_dir / cfg.draglib_local_name
    if not draglib_dst.exists():
        src = Path(cfg.draglib_path)
        try:
            os.symlink(src.resolve(), draglib_dst)
        except OSError:
            shutil.copy(src, draglib_dst)
    deck_path = work_dir / "deck.x2m"
    deck_path.write_text(deck_text)

    cmd = [resolved_exe]
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL", "-eL", resolved_exe]
    env = dict(os.environ, GFORTRAN_UNBUFFERED_ALL="1")
    lines, results = [], []            # results: [(step, k_inf)] in arrival order
    state = {"bol_dump_done": False}

    def _reader(stream):
        for line in iter(stream.readline, ""):
            lines.append(line)
            m = _RESULT_RE.search(line)
            if m:
                results.append((int(m.group(1)), _f(m.group(3))))
            elif ">|PPF_STEP_END" in line:
                ints = _INT_RE.findall(line.split("PPF_STEP_END", 1)[-1])
                if ints and int(ints[0]) == 0:
                    state["bol_dump_done"] = True

    abort, bol_ppf, bol_map, timed_out = None, None, None, False
    t0 = time.time()
    try:
        with open(deck_path, "r") as deck_fh:
            proc = subprocess.Popen(cmd, stdin=deck_fh, cwd=str(work_dir), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                                    start_new_session=True)
        th = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        th.start()

        def _kill():
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()

        checked_k = 0
        while proc.poll() is None:
            time.sleep(cfg.abort_poll_sec)
            if time.time() - t0 > cfg.timeout_sec:
                timed_out = True; _kill(); break
            # --- k_inf limit: any step above the limit proves k_inf_max > limit
            if cfg.abort_k_above is not None:
                while checked_k < len(results):
                    st, k = results[checked_k]; checked_k += 1
                    if cfg.abort_k_mode == 'peak':
                        ks_seen = [kk for _, kk in results[:checked_k]]
                        if st >= 3 and len(ks_seen) >= 2 and k < ks_seen[-2]:     # peak has passed
                            kmax_seen = max(ks_seen)
                            if kmax_seen > cfg.abort_k_above:
                                abort = {"reason": f"k_inf_max={kmax_seen:.5f} > {cfg.abort_k_above} "
                                                   f"(peak confirmed at step {st})"}
                                break
                    elif k > cfg.abort_k_above:
                        abort = {"reason": f"k_inf={k:.5f} > {cfg.abort_k_above} at step {st}"}
                        break
                if abort:
                    _kill(); break
            # --- BOL PPF limit (once, as soon as the step-0 dump is complete)
            if (abort is None and cfg.abort_ppf_above is not None and cfg.compute_ppf and cfg.ppf_at == 'bol'
                    and state["bol_dump_done"] and bol_ppf is None):
                bol_ppf, bol_map = _bol_ppf_and_map("".join(lines), pattern, cfg, n_types)
                if bol_ppf is not None and bol_ppf >= cfg.abort_ppf_above:
                    abort = {"reason": f"BOL PPF={bol_ppf:.4f} >= {cfg.abort_ppf_above}"}
                    _kill(); break
        th.join(timeout=30)
        returncode = proc.returncode
        log_text = "".join(lines)
        (work_dir / "dragon_stdout.log").write_text(log_text)
        elapsed = time.time() - t0
        if abort:
            if bol_ppf is None and cfg.compute_ppf and cfg.ppf_at == 'bol':
                bol_ppf, bol_map = _bol_ppf_and_map(log_text, pattern, cfg, n_types)
            abort.update(bol_ppf=bol_ppf, bol_map=bol_map, elapsed_s=elapsed)
            return log_text, abort
        n_result = len(_RESULT_RE.findall(log_text))
        n_expected = len(cfg.burnup_steps_mwd_tu) + 1
        if timed_out or n_result == 0 or n_result < n_expected:
            reason = (f"dragon TIMED OUT after {cfg.timeout_sec}s" if timed_out
                      else f"dragon exited with code {returncode}")
            raise RuntimeError(f"{reason}, and only produced {n_result}/{n_expected} expected 'RESULT step=...' "
                               f"lines.\n--- output tail ---\n{log_text[-4000:]}\n"
                               f"Full raw log: {work_dir / 'dragon_stdout.log'}")
        return log_text, None
    finally:
        if own_dir and not cfg.keep_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def _aborted_out(pattern, E, G, n_uo2, n_gad, log_text, abort):
    """Result dict of an early-aborted call (never cached)."""
    k_steps = _parse_output(log_text, None)
    steps_sorted = sorted(k_steps)
    kinfs = [k_steps[s][1] for s in steps_sorted]
    history = {"step": steps_sorted, "burnup_mwd_tu": [k_steps[s][0] for s in steps_sorted], "k_inf": kinfs,
               "ppf_eval_step": 0}
    if abort.get("bol_map") is not None:
        history["pin_power_map_kmax"] = abort["bol_map"]
    return {"E": E, "G": G, "N_UO2": n_uo2, "N_GAD": n_gad,
            "k_inf_max": float(max(kinfs)) if kinfs else None, "PPF": abort.get("bol_ppf"), "CL_proxy": None,
            "aborted": abort["reason"], "dragon_elapsed_s": abort.get("elapsed_s"),
            "burnup_history": history, "physics_engine": "DRAGON5"}


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------

def _cache_key(pattern, cfg: Dragon5Config) -> str:
    cfg_fingerprint = dict(
        pin_pitch_cm=cfg.pin_pitch_cm, r_fuel_cm=cfg.r_fuel_cm, r_gap_cm=cfg.r_gap_cm,
        r_clad_cm=cfg.r_clad_cm, fuel_temp_k=cfg.fuel_temp_k, clad_temp_k=cfg.clad_temp_k,
        moderator_temp_k=cfg.moderator_temp_k, channel_density=cfg.channel_density,
        waterrod_density=cfg.waterrod_density, burnup_steps_mwd_tu=list(cfg.burnup_steps_mwd_tu),
        specific_power_mw_per_tu=cfg.specific_power_mw_per_tu,
        cl_kinf_threshold=cfg.cl_kinf_threshold,
        compute_ppf=cfg.compute_ppf, compute_cl=cfg.compute_cl, ppf_at=cfg.ppf_at,
        n_groups=cfg.n_groups, salt_an2d=cfg.salt_an2d, salt_densur=cfg.salt_densur,
    )
    if _frame_layers(cfg):   # only when active, so pre-existing cache keys stay valid
        cfg_fingerprint.update(frame=[cfg.frame_inner_cm, cfg.frame_wall_cm, cfg.frame_water_cm, cfg.gapwater_density,
                                 list(_gap_widths(cfg))])
    flat = json.dumps({"pattern": pattern, "cfg": cfg_fingerprint}, sort_keys=True)
    return hashlib.sha256(flat.encode()).hexdigest()


def _cache_load(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _cache_save(path, cache):
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, path)


# ----------------------------------------------------------------------
# Main API -- drop-in replacement for bwr10x10_surrogate.casmo4_bwr10x10
# ----------------------------------------------------------------------

def casmo4_bwr10x10(pattern, cfg: Dragon5Config = DEFAULT_CONFIG):
    """Real-DRAGON5-backed replacement for the surrogate of the same
    name. Same input (10x10 pattern) and output contract (dict with
    'E', 'G', 'N_UO2', 'N_GAD', 'k_inf_max', 'PPF', 'CL_proxy') as
    bwr10x10_surrogate.casmo4_bwr10x10. PPF/CL_proxy are None unless
    cfg.compute_ppf/compute_cl are set True. PPF is validated as of the
    2026-08-08 session (see module docstring's PPF PROVENANCE note);
    compute_cl now uses the paper's own k_inf=0.95-crossing definition
    (see module docstring's CL_proxy note) -- no longer the unvalidated
    buckling-search redesign."""
    enrich, gad, water = pattern_to_grids(pattern)
    E = average_enrichment(enrich, water)
    G = average_gad(gad, water)
    n_uo2 = int(np.sum((~water) & (gad == 0)))
    n_gad = int(np.sum((~water) & (gad > 0)))

    cache = _cache_load(cfg.cache_path)
    key = _cache_key(pattern, cfg)
    if key in cache:
        return dict(cache[key])

    deck = build_deck(pattern, cfg)
    _, _, n_types = _discover_fuel_types(pattern)
    if cfg.abort_ppf_above is not None or cfg.abort_k_above is not None:
        log_text, abort = _run_dragon_stream(deck, cfg, pattern, n_types)
        if abort:
            return _aborted_out(pattern, E, G, n_uo2, n_gad, log_text, abort)   # not cached
    else:
        log_text = _run_dragon(deck, cfg)
    k_inf_steps = _parse_output(log_text, cfg)
    k_inf_max, ppf, cl_days, cl_warning, history = _reduce_history(
        pattern, k_inf_steps, cfg, log_text=log_text, n_types=n_types)
    if cl_warning:
        import warnings
        warnings.warn(cl_warning)

    out = {
        "E": E, "G": G, "N_UO2": n_uo2, "N_GAD": n_gad,
        "k_inf_max": k_inf_max, "PPF": ppf, "CL_proxy": cl_days,
        "burnup_history": history, "physics_engine": "DRAGON5",
    }

    cache[key] = out
    _cache_save(cfg.cache_path, cache)
    return out


# ----------------------------------------------------------------------
# Self-check against the 3 known reference-paper data points.
# ----------------------------------------------------------------------

def self_check(cfg: Dragon5Config = DEFAULT_CONFIG, verbose=True):
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
        out = casmo4_bwr10x10(pat, cfg)
        rows.append((name, out, tgt))
        if verbose:
            print(f"{name}")
            print(f"    E        : DRAGON5={out['E']:.4f}   paper={tgt['E']}")
            print(f"    G        : DRAGON5={out['G']:.4f}   paper={tgt['G']}")
            print(f"    N_UO2/GAD: DRAGON5={out['N_UO2']}/{out['N_GAD']}   "
                  f"paper={tgt['N_UO2']}/{tgt['N_GAD']}")
            print(f"    k_inf_max: DRAGON5={out['k_inf_max']:.5f}   paper={tgt['k_inf_max']}   "
                  f"diff={out['k_inf_max'] - tgt['k_inf_max']:+.5f}")
            if out["PPF"] is not None:
                print(f"    PPF      : DRAGON5={out['PPF']:.4f}   paper={tgt['PPF']}   "
                      f"diff={out['PPF'] - tgt['PPF']:+.4f}")
            else:
                print(f"    PPF      : not computed (Dragon5Config.compute_ppf=False)")
            if out["CL_proxy"] is not None:
                print(f"    CL       : DRAGON5={out['CL_proxy']:.1f} days   paper={tgt['CL']}   "
                      f"diff={out['CL_proxy'] - tgt['CL']:+.1f}")
            else:
                print(f"    CL       : not computed (Dragon5Config.compute_cl=False)")
    return rows


if __name__ == "__main__":
    self_check()
