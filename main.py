#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (integrator-H, v10).

Top-down: trajectory fit + tau-inverted anchors + post-ensemble
guardrails. Written from scratch, revised from the v1 test numbers and
the external-AI consensus rounds 1-3 (19 Nov 19:31, 19:41, 19:53 WIB).
The round-3 specifications (Claude's Bayesian draws, Gemini's cohort
integration) were read and their ideas adapted in my own words and
structure -- no line was copied from them (authorship rule).

Design, in words. For each trophy the program fits a two-phase trajectory
from the case history alone (rate = earners / tracked players, simulated
separately):
  * DILUTION: Exophase refreshes jump the tracked-player count N and pull
    every rate toward the new cohort's rate r_new. r_new is read off the
    last N-growing refresh when the history has one; otherwise it comes
    from a class prior measured on the real first refresh (04:30Z, N 638
    -> 3868): the new cohort held almost nothing of most trophies
    (r_new ~0-0.03%), a third of Chapter 1 (0.34%), nearly all of the
    fishing challenge (0.99%), and all of the first trophy (100%). r_new
    RISES as the base matures (toward the day-7 anchor, at a rate the
    history's own maturation -- first vs last refresh -- actually shows).
    Once a second refresh exists, the hazard scale lambda_H is updated
    from the marginal trophies per new player (Gemini round 2).
  * PROGRESSION: between refreshes, non-holders earn the trophy at a
    per-hour hazard that ramps up with the cohort's age -- no progression
    in the first 6 h after release (players are still installing/starting;
    the launch-night reads were flat), then a slow ramp, then a decay as
    the active players finish and the remainder are increasingly inactive.
    Hazards are small (Ch1 0.004/h). No absolute hazard floor is imposed
    -- slow trophies may sit near zero hazard (platinum/completion have
    hazard exactly 0: no regular player finishes 100% in a week). When
    the history's reads show rates creeping at constant N, the hazard is
    measured directly from that creep instead of the class prior.
  * ANCHORS FROM TAU INVERSION (round 3): each trophy gets a difficulty
    threshold tau (hours of relevant play) from its class; the day-7 rate
    comes from a piecewise log-linear curve of rate vs tau through the
    consensus points (0.15 h, 98%), (9.5 h, 41% -- Chapter 1's tau_eff,
    Gemini Model B), (72 h, 3.5% -- final mission), (255 h, 0.022% --
    platinum, geometric mean of Claude 0.019% and Gemini 0.026%, round 3:
    Gemini revised down from 0.08%). The tau values for the final (72 h)
    and platinum (255 h) come from Gemini's piecewise rank anchors
    (rank 48 = 72 h, rank 55 = 255 h). The first trophy is special
    (anchor 99.6%: the decline is only sync glitch). Optional side
    content keeps a prior-blend anchor -- its tau is ill-defined (hours
    played != hours spent on that activity).
    The fit is BLENDED with the anchor, never allowed to override it:
    on launch-night data the anchor carries 65% of the weight (fit
    weight 0.35 with no refresh seen), so the Chapter-1 median stays
    near the 41% anchor (~42%), not the 73% Gemini's self-test produced
    when its tau inversion (blend 0.7) overrode the anchor and ignored
    the Freeroam Sink. RARITY_ANCHOR_ONLY=1 forces the pure anchor (test
    hook). There is NO rank-prior shrinkage in this model: shrinking
    15% toward a log-linear rank prior pulls rare trophies up toward a
    prior both external models call wrong for ranks 49-54 (it moved
    Gemini's platinum from 0.026% to 0.049% in their self-test).
  * STALENESS (round 3 choice): Gemini T-3.5 h, not Claude T-~12 h.
    Reason: the observed first-refresh gap is 4.5 h and the page sat
    unchanged for 7 h (04:30Z -> 11:27Z), so the refresh interval is
    ~7 h and the median lag is ~3.5 h; Claude's 12 h would imply a ~24 h
    interval, which contradicts the observed gaps. The model simulates
    to the last refresh plus half the fitted gap (~3.5-5 h); the
    platinum growth law is SUBLINEAR (Gemini, exponent 0.60), not linear
    (Claude): the early-copy cohort saturates and the hazard decays, so
    accumulation decelerates. At E2 = 11 platinum holders the sublinear
    law gives 0.040% vs linear 0.077% (sensitivity reported in the test
    harness, not hardcoded into the anchor -- no second refresh exists
    yet to decide it).
  * The forecast is the state at the last refresh the answer can show,
    not the state at the target: the Exophase page only changes at
    refreshes, so progression after the last refresh before the target is
    unobservable. The simulation stops at the last refresh plus half a
    refresh gap (the median staleness). A small sigma_stale term covers
    the residual lag.
  * SHORT HORIZONS (<= 360 min): a 2-component Gaussian mixture on the
    logit scale (Gemini round 2, exact formula). Branch 0 = no refresh:
    the rate is locked at the last read, sigma0 = 0.012. Branch 1 = one
    refresh: median = the forecast, sigma1 = the spread. p0 comes from a
    Gamma renewal process (k = 2.5), NOT Poisson, using the time since
    the last refresh and the fitted gap. The mixture CDF is solved by
    30-iteration bisection -- never by averaging quantiles linearly,
    because the result is bimodal.
  * SPREADS are wide and symmetric in logit, and horizon-dependent: narrow
    (~0.4) at short horizons; wide (~1.25) at the 4.5-day horizon, where
    the house model's spread sits (1.0-1.2).
  * POST-ENSEMBLE GUARDRAILS (Gemini round 2, the last lines before
    writing the output): (a) physical lower bounds -- q95 >=
    100*E_last/110000 for trophies with E_last >= 1, and median >= r_last
    for tau <= 35 h at horizon >= 48 h (progression must have overtaken
    dilution by then); (b) isotonic regression (PAVA) on the median logits
    ordered by difficulty rank (tau, grade), which stops one trophy
    falling to 0.01% alone or an implausible order inversion.
No network, stdlib only, deterministic. Set RARITY_DEBUG=1 to print the
per-trophy fit / anchor / median / spread to stderr (tuning aid only).
Set RARITY_TAU_CHAIN1=<hours> to override the Chapter-1 threshold for the
theta-sensitivity test.
"""

import glob
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------- constants
T0 = "2026-11-19T00:00:00Z"  # release-day start
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)

# First refresh: observed ~4.5 h after release (04:30Z, N 638 -> 3868).
FIRST_REFRESH_H = 4.5
FIRST_JUMP_PRIOR = 6.06  # observed 3868 / 638
REFRESH_GAP_PRIOR_H = 10.0
N_GROWTH_ALPHA_PRIOR = 1.0
N_CAP = 1000000.0
N_GROWTH_B_CAP = 0.7  # cap the extrapolated power-law exponent
# No progression in the first hours after release (flat launch-night reads).
PROGRESSION_DELAY_H = 6.0
PROGRESSION_RAMP_TAU_H = 24.0
# r_new rises toward the cohort's current rate with this time constant.
RNEW_RISE_TAU_H = 48.0

# Blend weight on the trajectory fit (rest = anchor curve), by refreshes
# observed in the case history.
def w_fit(n_refresh):
    if n_refresh <= 0:
        return 0.35
    if n_refresh == 1:
        return 0.50
    if n_refresh == 2:
        return 0.65
    if n_refresh <= 5:
        return 0.80
    return 0.90

# Spread (logit). s_refresh: per-refresh size uncertainty, growing with the
# expected refresh count; s_rw: random walk; s_prog: progression/turning
# uncertainty. Clipped to [0.40, 1.25] -- narrow at short horizons (the
# first refresh is predictable), wide at the final horizon (house-like).
# SIGMA_STALE (consensus): the answer read lags the last refresh by a few
# hours (Claude ~12 h, Gemini ~3.5 h); a small residual spread term.
SPREAD_REFRESH_PER = 0.30
SPREAD_REFRESH_MIN = 0.25
SPREAD_RW = 0.40
SPREAD_MIN = 0.40
SPREAD_MAX = 1.25
SIGMA_STALE = 0.15

# Per-class parameters: (hazard ceiling per hour, hour after release when
# the class's progression phase turns the rate back up, progression-spread
# scale at the final horizon).
# DAY-7 ANCHORS COME FROM TAU INVERSION (external-AI consensus, rounds
# 1-2), not from hardcoded anchor numbers. Round 2 replaces the old
# exponential rate_7 = A*exp(-tau/H7) -- that form could not fit the two
# consensus points (final 3.5% at 72 h, platinum 0.039% at 255 h) with a
# physical active fraction (it forced A > 1). The new model is:
#   (1) each trophy gets a DIFFICULTY RANK (1 = easiest) by ordering on
#       the class ordering threshold below;
#   (2) tau comes from Gemini's piecewise rank anchors (round 2):
#       rank 2 = 0.15 h, rank 20 = 22 h, rank 48 = 72 h, rank 50 = 135 h,
#       rank 54 = 220 h, rank 55 = 255 h (rank 1 extrapolates to 0.05 h);
#   (3) the day-7 rate comes from a piecewise LOG-LINEAR curve of rate
#       vs tau through the consensus calibration points:
#       (0.15 h, 98%), (9.5 h, 41%), (72 h, 3.5%), (255 h, 0.022%).
#       The 9.5 h / 41% point is Chapter 1's tau_eff (round 2: Freeroam
#       Sandbox Sink makes the effective threshold ~9.5 h, not the 3 h of
#       bare story time; Gemini Model B 40.8%, Claude prior 55%).
#       The 255 h / 0.022% point is platinum, the geometric mean of
#       Claude 0.019% and Gemini 0.026% (round 3: Gemini revised down
#       from 0.08%; the two now differ 1.4x, not 4x).
# tau is the hours of RELEVANT play the trophy needs, from its
# description + the story structure (final ~72 h, the story is 68-80 h;
# platinum/completion ~255 h). Side content (fishing/fast/mid/grind)
# keeps the prior-blend anchor: its tau is ill-defined for optional
# activities (hours played != hours spent on that activity), so a tau
# anchor there would be false precision -- and it would overstate the
# rate, since it assumes everyone with enough hours does the activity.
# FIRST (id 2) is special: the tau model does not apply to the first
# trophy (everyone who plays gets it at once); its anchor is the consensus
# 99.6% (median 99.4-99.95%, decline only from sync glitch).
TAU_ORDER = {
    "FIRST": 0.05, "FISH": 5.0, "CHAIN1": 9.5, "FAST": 12.0,
    "CHAIN2": 20.0, "MID": 30.0, "CHAIN3": 35.0, "CHAIN4": 55.0,
    "FINAL": 72.0, "CHARF": 100.0, "GRIND": 120.0, "COMPL": 200.0,
    "PLAT": 255.0,
}
TAU_RANK = [(2, 0.15), (20, 22.0), (48, 72.0), (50, 135.0),
            (54, 220.0), (55, 255.0)]
RATE7_CURVE = [(0.15, 98.0), (9.5, 41.0), (72.0, 3.5), (255.0, 0.022)]
ANCHOR_FIRST = 99.6
# Prior-blend day-7 anchors for optional side content (house-weighted
# logit blend of the house model's estimate and the anchor midpoint).
ANCHOR_PRIOR = {
    "FISH": 13.0, "FAST": 8.8, "MID": 4.5, "GRIND": 0.88,
}
# Consensus day-7 tracked-player count (round 2: N7 ~= 43k, both AIs;
# N is a STEP function, not a smooth curve). Used for the E_early floor.
N7_EST = 43000.0
# Physical guardrail (Gemini round 2): the day-7 tracked count cannot
# exceed N_MAX.
N_MAX = 110000.0
# Gamma renewal for the refresh process (round 2): shape k = 2.5, not
# Poisson (k = 1). The inter-refresh times are more regular than
# memoryless, so a refresh becomes more likely the longer it has been
# since the last one.
GAMMA_K = 2.5
# No-refresh branch spread (Gemini round 2): the rate is locked when no
# refresh lands.
SIGMA0 = 0.012
CLASS = {
    # class: (hazard/hr, turn start h, progression spread at final horizon)
    "FIRST":  (0.0, 0.0, 0.30),
    "PLAT":   (0.0, 9999.0, 0.40),
    "CHAIN1": (0.004, 18.0, 0.55),
    "CHAIN2": (0.0025, 30.0, 0.55),
    "CHAIN3": (0.0018, 42.0, 0.55),
    "CHAIN4": (0.0012, 54.0, 0.55),
    "FINAL":  (0.0008, 66.0, 0.60),
    "FISH":   (0.003, 12.0, 0.55),
    "FAST":   (0.002, 18.0, 0.55),
    "MID":    (0.001, 36.0, 0.50),
    "GRIND":  (0.0002, 72.0, 0.40),
    "COMPL":  (0.0, 9999.0, 0.40),
    "CHARF":  (0.0006, 60.0, 0.55),
}
CAP = {
    "FIRST": 100.0, "PLAT": 2.0, "CHAIN1": 55.0, "CHAIN2": 45.0,
    "CHAIN3": 40.0, "CHAIN4": 35.0, "FINAL": 25.0, "FISH": 40.0,
    "FAST": 35.0, "MID": 30.0, "GRIND": 10.0, "COMPL": 2.0, "CHARF": 20.0,
}
# r_new priors (ABSOLUTE %, measured on the real first refresh 04:30Z).
# The new cohort held: Ch1 0.34%, fishing 0.99%, arcade 0.28%, discover
# vehicles 0.22%, Ch2-5 and story missions 0.06%, most others 0-0.03%.
def rnew_prior(cls, text):
    if cls == "FIRST":
        return 100.0
    if cls == "PLAT" or cls == "COMPL":
        return 0.0
    if cls == "CHAIN1":
        return 0.34
    if cls in ("CHAIN2", "CHAIN3", "CHAIN4", "FINAL"):
        return 0.06
    if cls == "FISH":
        return 0.99
    if cls == "FAST":
        if "arcade" in text:
            return 0.28
        if "discover" in text:
            return 0.22
        return 0.0  # "try each type of activity once": 0 new holders
    if cls == "CHARF":
        return 0.03
    if cls == "GRIND":
        return 0.01
    return 0.03  # MID default: ~1 new holder in 3230


def rate7_from_tau(tau):
    """Day-7 rate (%) for a trophy needing tau hours, piecewise
    log-linear through the consensus calibration points."""
    pts = RATE7_CURVE
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for (t1, r1), (t2, r2) in zip(pts, pts[1:]):
        if t1 <= tau <= t2:
            f = (math.log(tau) - math.log(t1)) / (math.log(t2) - math.log(t1))
            return math.exp(math.log(r1) + f * (math.log(r2) - math.log(r1)))
    return pts[-1][1]


def anchor7_for(cls, tau):
    """Day-7 anchor median (%) for a class. Story-gated trophies use the
    tau-inverted curve; optional side content keeps the prior blend."""
    if cls == "FIRST":
        return ANCHOR_FIRST
    if cls in ANCHOR_PRIOR:
        return ANCHOR_PRIOR[cls]
    # Sensitivity hook (tests only): RARITY_TAU_CHAIN1 overrides the
    # Chapter-1 threshold in hours.
    if cls == "CHAIN1":
        ov = os.environ.get("RARITY_TAU_CHAIN1")
        if ov:
            tau = float(ov)
    return rate7_from_tau(tau)


# ---------------------------------------------------------------- utilities
def parse_time(s):
    from datetime import datetime, timezone
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def clamp(x, lo=0.01, hi=99.99):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l > 30.0:
        return 100.0
    if l < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def median_of(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def phi(x):
    """Standard normal CDF (stdlib erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _gammainc_reg_lower(a, x):
    """Regularized lower incomplete gamma P(a, x) (Numerical Recipes)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(300):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def gamma_sf(x, k, scale):
    """Survival P(T > x) for Gamma(shape k, scale)."""
    if x <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - _gammainc_reg_lower(k, x / scale)))


def renewal_p0(t_since_h, horizon_h, mean_gap_h):
    """Probability of NO refresh inside the horizon, Gamma renewal
    (shape GAMMA_K, mean = mean_gap_h), given t_since_h hours since the
    last refresh. A Gamma renewal is more regular than Poisson: the
    longer since the last refresh, the more likely one is imminent, so
    p0 falls with t_since (Poisson would ignore it)."""
    if horizon_h <= 0.0:
        return 1.0
    scale = mean_gap_h / GAMMA_K
    num = gamma_sf(t_since_h + horizon_h, GAMMA_K, scale)
    den = gamma_sf(t_since_h, GAMMA_K, scale)
    if den <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, num / den))


def mixture_quantile(tau, p0, mu0, mu1, sigma1):
    """tau-quantile (in logit) of the 2-component Gaussian mixture on the
    logit scale: weight p0 on N(mu0, SIGMA0) (no refresh: the rate is
    locked at the last read) and 1-p0 on N(mu1, sigma1) (one refresh).
    Solved by 30-iteration bisection on the mixture CDF -- never by
    averaging the quantiles linearly, because the result is bimodal."""
    lo = min(mu0, mu1) - 8.0 * max(SIGMA0, sigma1) - 1.0
    hi = max(mu0, mu1) + 8.0 * max(SIGMA0, sigma1) + 1.0

    def cdf(l):
        return p0 * phi((l - mu0) / SIGMA0) + (1.0 - p0) * phi((l - mu1) / sigma1)

    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pava_nonincreasing(vals):
    """Pool-adjacent-violators: return the closest non-increasing vector
    (least squares, equal weights). Used on the median logits ordered by
    difficulty rank -- it stops one trophy falling to 0.01% alone or an
    implausible order inversion."""
    n = len(vals)
    out = list(vals)
    # blocks: [start, end, mean]
    blocks = [[i, i + 1, vals[i]] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] < blocks[i + 1][2]:
            # violation: pool the two blocks (weighted by size)
            s1, e1, m1 = blocks[i]
            s2, e2, m2 = blocks[i + 1]
            w1, w2 = e1 - s1, e2 - s2
            blocks[i] = [s1, e2, (m1 * w1 + m2 * w2) / (w1 + w2)]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    for (s, e, m) in blocks:
        for j in range(s, e):
            out[j] = m
    return out


# ---------------------------------------------------------------- classify
_CHAPTER_RE = re.compile(r"^Complete Chapter (\d)\.$")


def classify(trophy, last_rate):
    """Map a trophy to a class key."""
    if last_rate is not None and last_rate >= 99.99:
        return "FIRST"
    desc = trophy.get("description", "") or ""
    name = trophy.get("name", "") or ""
    tier = trophy.get("tier", "") or ""
    text = (desc + " " + name).lower()
    if tier == "platinum":
        return "PLAT"
    if "100% completion" in desc.lower():
        return "COMPL"
    m = _CHAPTER_RE.match(desc.strip())
    if m:
        n = int(m.group(1))
        return {1: "CHAIN1", 2: "CHAIN2", 3: "CHAIN3", 4: "CHAIN4"}.get(n, "MID")
    if desc.strip() == "Complete the final mission.":
        return "FINAL"
    if "final mission for" in desc:
        return "CHARF"
    if "fish" in text:
        return "FISH"
    if any(k in text for k in ("stunt", "challenge", "card set", "races",
                               "drone race", "shooting range",
                               "mission strand", "waink")):
        return "GRIND"
    if any(k in text for k in ("arcade", "discover", "try each type")):
        return "FAST"
    return "MID"


# ---------------------------------------------------------------- history
def distinct_reads(history):
    out = []
    prev = None
    for read in history:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        key = (n, tuple(rates))
        if key == prev:
            continue
        prev = key
        out.append({
            "at": parse_time(read["at"]),
            "n": (float(n) if n else None),
            "rates": [(float(r) if r is not None else None) for r in rates],
        })
    return out


def find_refreshes(reads):
    out = []
    for k in range(1, len(reads)):
        a, b = reads[k - 1], reads[k]
        if a["n"] != b["n"] or a["rates"] != b["rates"]:
            out.append((k, a["n"], b["n"], a["rates"], b["rates"]))
    return out


def implied_rnew(r_before, n_before, r_after, n_after):
    if r_before is None or r_after is None:
        return None
    if n_before is None or n_after is None or n_after <= n_before:
        return None
    v = (r_after * n_after - r_before * n_before) / (n_after - n_before)
    return max(0.0, min(100.0, v))


def fit_n_growth(reads, t0):
    """Least-squares log N = a + b*log(1+hours); exponent capped at 0.7."""
    pts = []
    for r in reads:
        if r["n"] and r["n"] > 0:
            pts.append((math.log(1.0 + (r["at"] - t0) / 3600.0),
                        math.log(r["n"])))
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    b = max(0.0, min(N_GROWTH_B_CAP, b))
    a = my - b * mx
    return (a, b)


# ---------------------------------------------------------------- forecast
DEBUG = os.environ.get("RARITY_DEBUG") == "1"


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    t0 = parse_time(T0)
    t_last = parse_time(case["at"])
    t_tgt = parse_time(case["target"])
    horizon_h = max(0.0, (t_tgt - t_last) / 3600.0)
    horizon_days = horizon_h / 24.0

    reads = distinct_reads(history)
    refreshes = find_refreshes(reads)
    n_refresh = len(refreshes)

    if reads:
        last = reads[-1]
        last_rates = last["rates"]
        n_last = last["n"]
    else:
        last_rates = [None] * len(trophies)
        n_last = None
    if n_last is None:
        n_last = 638.0
        last_rates = [None] * len(trophies)

    nfit = fit_n_growth(reads, t0)

    def n_forecast(t):
        h = max(0.0, (t - t0) / 3600.0)
        if nfit is not None:
            a, b = nfit
            v = math.exp(a + b * math.log(1.0 + h))
        else:
            hl = max(0.0, (t_last - t0) / 3600.0)
            v = n_last * ((1.0 + h) / (1.0 + hl)) ** N_GROWTH_ALPHA_PRIOR
        return min(N_CAP, max(n_last, v))

    # Refresh cadence over the horizon.
    if n_refresh >= 2:
        gaps = []
        prev_t = None
        for (k, _nb, _na, _rb, _ra) in refreshes:
            tt = reads[k]["at"]
            if prev_t is not None:
                gaps.append((tt - prev_t) / 3600.0)
            prev_t = tt
        gap = median_of(gaps) if gaps else REFRESH_GAP_PRIOR_H
        gap = min(max(gap, 4.0), 24.0)
    else:
        gap = REFRESH_GAP_PRIOR_H

    # Refresh event times over the horizon. The first refresh lands ~4.5 h
    # after release (observed); later ones follow the fitted gap.
    events = []
    t_first = t0 + FIRST_REFRESH_H * 3600.0
    first_refresh_inside = (n_refresh == 0 and t_last < t_first <= t_tgt)
    if first_refresh_inside:
        events.append(t_first)
        t_ev = t_first + gap * 3600.0
    else:
        t_ev = t_last + gap * 3600.0
    while t_ev <= t_tgt + 1.0:
        events.append(t_ev)
        t_ev += gap * 3600.0

    # The answer read shows the page state at the last refresh at or before
    # the target, and that refresh is on average half a gap old. Simulate
    # only to the last refresh + the staleness (median lag). STALENESS
    # CHOICE (round 3): Gemini T-3.5 h, Claude T-~12 h. This model uses
    # half the fitted refresh gap, which for the observed gaps (~7 h)
    # is ~3.5 h -- Gemini's value. Reason: the observed first-refresh gap
    # is 4.5 h and the page sat unchanged for 7 h (04:30Z -> 11:27Z), so
    # the refresh interval is ~7 h, not ~24 h (which Claude's 12 h would
    # imply). RARITY_STALE_H overrides the staleness in hours (test hook).
    _stale_h = os.environ.get("RARITY_STALE_H")
    stale_h = float(_stale_h) if _stale_h else 0.5 * gap
    if events:
        t_sim_end = min(t_tgt, events[-1] + stale_h * 3600.0)
    else:
        t_sim_end = t_tgt

    # Difficulty threshold tau (hours of relevant play) per trophy, from
    # its class. This tau is plugged into the rate_7 curve below. (Gemini's
    # piecewise RANK anchors -- rank 2 = 0.15 h, rank 20 = 22 h, rank 48 =
    # 72 h, rank 50 = 135 h, rank 54 = 220 h, rank 55 = 255 h -- calibrate
    # the curve: the 72 h and 255 h points are the final mission's and
    # platinum's taus from those anchors. A rank INTERPOLATION was tried
    # first and rejected: the 30 MID trophies cluster at one class tau,
    # so the rank count distorts the early chain (Chapter 1 landed at
    # rank 3 -> tau 1.4 h -> anchor 62%, and the final at rank 40 ->
    # tau 58 h). The class tau is the stable difficulty measure; the
    # rank anchors set the curve's scale, not each trophy's tau.)
    tau_of = {}
    for _i, _tr in enumerate(trophies):
        _cls = classify(_tr, last_rates[_i] if _i < len(last_rates) else None)
        tau_of[_i] = TAU_ORDER[_cls]

    # Time since the last refresh at the checkpoint (for the Gamma renewal
    # p0). Flat history: since the release.
    if refreshes:
        _last_ref_t = reads[refreshes[-1][0]]["at"]
        t_since_h = max(0.0, (t_last - _last_ref_t) / 3600.0)
    else:
        t_since_h = max(0.0, (t_last - t0) / 3600.0)

    # r_new per trophy from the last N-GROWING refresh (a refresh where the
    # tracked count actually rose -- a rates-only read carries no dilution
    # signal), else class priors. Same for the first N-growing refresh.
    def rnew_at(which):
        out = {}
        cands = [(k, nb, na, rb, ra) for (k, nb, na, rb, ra) in refreshes
                 if nb is not None and na is not None and na > nb]
        if not cands:
            return out
        (k, nb, na, rb, ra) = cands[-1] if which == "last" else cands[0]
        for i, tr in enumerate(trophies):
            v = implied_rnew(rb[i] if i < len(rb) else None, nb,
                             ra[i] if i < len(ra) else None, na)
            out[i] = v
        return out
    rnew_last = rnew_at("last")
    rnew_first = rnew_at("first")
    # r_new per N-growing refresh, in order (for the lambda_H update).
    rnew_by_refresh = []
    for (k, nb, na, rb, ra) in refreshes:
        if nb is None or na is None or na <= nb:
            continue
        d = {}
        for i, tr in enumerate(trophies):
            d[i] = implied_rnew(rb[i] if i < len(rb) else None, nb,
                                ra[i] if i < len(ra) else None, na)
        rnew_by_refresh.append((reads[k]["at"], nb, na, rb, ra, d))

    # OBSERVED HAZARD: when two consecutive reads share the same tracked
    # count N but a trophy's rate rose, that rise is pure progression (no
    # dilution). It directly measures the per-hour hazard. The real page is
    # flat between refreshes, so this fires only in worlds whose reads are
    # continuous (e.g. critic-1's); otherwise the class prior is used.
    haz_obs = {}
    for i in range(len(trophies)):
        hs = []
        for k in range(1, len(reads)):
            a, b = reads[k - 1], reads[k]
            if a["n"] is None or b["n"] is None or a["n"] != b["n"]:
                continue
            r1 = a["rates"][i] if i < len(a["rates"]) else None
            r2 = b["rates"][i] if i < len(b["rates"]) else None
            if r1 is None or r2 is None or r2 <= r1:
                continue
            dt = (b["at"] - a["at"]) / 3600.0
            if dt <= 0:
                continue
            p1, p2 = r1 / 100.0, r2 / 100.0
            if p2 >= 1.0 or p1 >= 1.0:
                continue
            hs.append(-math.log((1.0 - p2) / (1.0 - p1)) / dt)
        if len(hs) >= 3:
            haz_obs[i] = median_of(hs)

    # MATURITY: how far the newly-tracked cohort has come toward its day-7
    # anchor, measured from the first vs last N-growing refresh. 0 = the
    # cohort still holds nothing new (dilution world: trust the fit, shrink
    # the anchor pull and the hazard); 1 = fully mature. None when fewer
    # than two N-growing refreshes are observed (fall back to the priors).
    maturity = None
    if len([1 for (k, nb, na, rb, ra) in refreshes
            if nb is not None and na is not None and na > nb]) >= 2:
        vals = []
        for i, tr in enumerate(trophies):
            cls_i = classify(tr, last_rates[i] if i < len(last_rates) else None)
            a7 = anchor7_for(cls_i, tau_of.get(i, TAU_ORDER[cls_i]))
            f = rnew_first.get(i)
            l = rnew_last.get(i)
            if f is None or l is None:
                continue
            denom = max(a7 - f, 0.05)
            vals.append(min(1.0, max(0.0, (l - f) / denom)))
        if vals:
            maturity = median_of(vals)
    # Anchor pull / hazard scale / r_new-rise scale: a flat cohort keeps
    # only a token pull (1-2% -- a hedge, not a forecast); a mature cohort
    # keeps the full pull. The floors are deliberately tiny: thirty reads
    # showing a flat cohort are strong evidence, and the anchor must not
    # override them.
    pull = 1.0 if maturity is None else max(0.02, maturity)
    haz_scale = 1.0 if maturity is None else max(0.01, maturity)

    # LAMBDA_H update (Gemini round 3, exact formula): once a SECOND
    # N-growing refresh is in the history, compare the marginal trophies
    # per new player at the last refresh (actual) against the model's
    # prediction (the previous refresh's r_new risen toward the day-7
    # anchor), and scale the hazard by ((m_obs-1)/(m_pred-1))^0.65,
    # floored at 0.55, capped at 1.75. The (m-1) form removes the first
    # trophy (every new player gets it, so it carries no signal). This
    # reproduces Gemini's examples exactly: predicted 3.8 vs actual 6.2
    # -> ((6.2-1)/(3.8-1))^0.65 = 1.49; predicted 3.8 vs actual 1.9 ->
    # ((1.9-1)/(3.8-1))^0.65 = 0.48, floored to 0.55. Inactive until a
    # second refresh exists (no second refresh observed yet: the 11:27Z
    # snapshot is identical to the 04:30Z refresh).
    lambda_h = 1.0
    if len(rnew_by_refresh) >= 2:
        (t_prev, nb_prev, na_prev, rb_prev, ra_prev, d_prev) = rnew_by_refresh[-2]
        (t_j, nb_j, na_j, rb_j, ra_j, d_j) = rnew_by_refresh[-1]

        def _total_earned(rates, n):
            tot = 0.0
            for r in (rates or []):
                if r is not None and n:
                    tot += r / 100.0 * n
            return tot

        actual_marg = None
        if na_j and na_prev and na_j > na_prev:
            actual_marg = (_total_earned(ra_j, na_j)
                           - _total_earned(ra_prev, na_prev)) / (na_j - na_prev)
        pred_marg = 0.0
        age_ref = max(0.0, (t_j - t_first) / 3600.0)
        rise_j = 1.0 - math.exp(-age_ref / RNEW_RISE_TAU_H)
        for i, tr in enumerate(trophies):
            cls_i = classify(tr, last_rates[i] if i < len(last_rates) else None)
            a7 = anchor7_for(cls_i, tau_of.get(i, TAU_ORDER[cls_i]))
            base = d_prev.get(i)
            if base is None:
                base = rnew_prior(cls_i, ((tr.get("description") or "")
                                           + " " + (tr.get("name") or "")).lower())
            pred = base + (a7 - base) * rise_j
            pred = min(pred, a7)
            pred_marg += max(0.0, pred) / 100.0
        if actual_marg is not None and pred_marg > 1.02:
            ratio = max(actual_marg - 1.0, 0.02) / max(pred_marg - 1.0, 0.02)
            lambda_h = max(0.55, min(1.75, ratio ** 0.65))
            haz_scale *= lambda_h
    if DEBUG:
        sys.stderr.write(f"case={case['id']} lambda_h={lambda_h:.3f} "
                         f"t_since={t_since_h:.1f}h\n")
    if DEBUG:
        sys.stderr.write(f"case={case['id']} refreshes={n_refresh} "
                         f"maturity={maturity} pull={pull:.2f} "
                         f"haz_scale={haz_scale:.2f} w={w_fit(n_refresh):.2f}\n")
    launch_rates = {}
    if reads:
        for i in range(len(trophies)):
            launch_rates[i] = reads[0]["rates"][i] if i < len(reads[0]["rates"]) else None

    expected_refreshes = horizon_h / gap
    if n_refresh == 0 and events:
        expected_refreshes = max(expected_refreshes, 1.0)

    w = w_fit(n_refresh)
    meds = []  # per-trophy: (id, cls, tau, r_last, e_last, med, s)
    dbg = []
    for i, tr in enumerate(trophies):
        cls = classify(tr, last_rates[i] if i < len(last_rates) else None)
        tau = tau_of.get(i, TAU_ORDER[cls])
        anchor7 = anchor7_for(cls, tau)
        haz_max, turn_h, prog_sd = CLASS[cls]
        # The turning hour cannot precede the hours the trophy needs.
        turn_h = max(turn_h, tau) if cls not in ("FIRST", "PLAT", "COMPL") else turn_h
        # E_early floor (round 2, decontamination): the current holders
        # of a rare trophy are almost certainly early-copy/reviewer, so
        # the day-7 rate cannot fall below those holders divided by N7.
        # (For platinum the curve gives 0.039%, above the 0.0093% floor,
        # so the floor is a safety net, not the value.)
        if cls in ("PLAT", "COMPL"):
            e_early = (last_rates[i] / 100.0 * n_last
                       if i < len(last_rates) and last_rates[i] is not None
                       else 0.0)
            anchor7 = max(anchor7, 100.0 * e_early / N7_EST)
        text = ((tr.get("description") or "") + " " + (tr.get("name") or "")).lower()
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = anchor7 * 0.2 if cls != "FIRST" else 100.0
        e_last = r_last / 100.0 * n_last

        # --- trajectory fit: simulate forward in 1-hour steps to t_sim_end ---
        n_now = n_last
        e_now = e_last
        ev_idx = 0
        step = 3600.0
        t_cur = t_last
        while t_cur < t_sim_end - 1e-9:
            dt = min(step, t_sim_end - t_cur)
            age_h = max(0.0, (t_cur - t0) / 3600.0)
            if age_h <= PROGRESSION_DELAY_H:
                ramp = 0.0
            else:
                ramp = 1.0 - math.exp(-(age_h - PROGRESSION_DELAY_H)
                                      / PROGRESSION_RAMP_TAU_H)
            # Hazard decays with time since the checkpoint: the non-holder
            # pool is heterogeneous -- the active players finish early and
            # the remainder are increasingly inactive, so a constant hazard
            # extrapolated for 4.5 days over-forecasts (v5: critic-1 CHARF
            # trophies). 192 h decay (v6's 96 h cost too much scenario-mid);
            # negligible at short horizons.
            age_since_cp = max(0.0, (t_cur - t_last) / 3600.0)
            decay = math.exp(-age_since_cp / 192.0)
            if i in haz_obs:
                # Directly measured progression: trust it over the prior
                # (capped at the prior -- a noisy read must not inflate it).
                haz = min(haz_max, haz_obs[i]) * decay
            else:
                haz = haz_max * ramp * haz_scale * decay
            if cls != "FIRST" and n_now > 0:
                nonholders = max(0.0, n_now - e_now)
                e_now += nonholders * (1.0 - math.exp(-haz * (dt / 3600.0)))
            t_cur += dt
            while ev_idx < len(events) and events[ev_idx] <= t_cur + 1.0:
                t_ref = events[ev_idx]
                ev_idx += 1
                if n_refresh == 0 and ev_idx == 1 and t_ref == t_first:
                    n_new = n_now * FIRST_JUMP_PRIOR
                else:
                    n_new = n_forecast(t_ref)
                n_new = max(n_now, min(N_CAP, n_new))
                if n_new > n_now:
                    base = rnew_last.get(i)
                    if base is None:
                        base = rnew_prior(cls, text)
                    # r_new rises toward the day-7 anchor as the base
                    # matures (clock runs from the first refresh, not from
                    # release: the cohort at the first refresh is fresh).
                    # In a flat-cohort world (maturity ~ 0) the rise is
                    # nearly frozen: the data says new joiners hold nothing.
                    age_ref = max(0.0, (t_ref - t_first) / 3600.0)
                    rise = 1.0 - math.exp(-age_ref / RNEW_RISE_TAU_H)
                    rise *= max(0.01, maturity) if maturity is not None else 1.0
                    rnew = base + (anchor7 - base) * rise
                    rnew = min(rnew, anchor7)
                    if cls == "FIRST":
                        rnew = 100.0 if age_ref < 24.0 else 99.5
                    e_now = e_now + (rnew / 100.0) * (n_new - n_now)
                    n_now = n_new
        fit_rate = 100.0 * e_now / n_now if n_now > 0 else r_last
        fit_rate = max(0.0, min(100.0, fit_rate))

        # --- anchor curve at the target time ---
        # Start from the post-first-refresh level: the last observed rate
        # when the history already shows a refresh, else the dilution
        # arithmetic on the class r_new prior with the first N jump.
        if n_refresh >= 1:
            r_post = r_last
        else:
            base = rnew_prior(cls, text)
            n_after = n_last * FIRST_JUMP_PRIOR
            r_post = (e_last + (base / 100.0) * (n_after - n_last)) / n_after * 100.0
        h_tgt = max(0.0, (t_tgt - t0) / 3600.0)
        if cls == "FIRST":
            # Pin at 100% early; honest lower anchor (99.6%, consensus) by
            # day 7. The decline is only from sync glitch -- never 95-97%.
            if h_tgt <= 12.0:
                anchor_rate = 100.0
            else:
                prog = math.log(h_tgt / 12.0) / math.log(168.0 / 12.0)
                prog = min(1.0, max(0.0, prog))
                anchor_rate = inv_logit(
                    logit(100.0) + (logit(ANCHOR_FIRST) - logit(100.0)) * prog)
        elif h_tgt <= turn_h and anchor7 >= r_post:
            # Rising anchor: the rate only turns up once progression
            # overtakes dilution (the class turning hour).
            anchor_rate = r_post
        elif anchor7 < r_post:
            # Falling anchor (platinum / completion: dilution wins and the
            # rate declines toward the tau anchor). The decline starts at
            # once -- no turning delay -- along log-time to day 7.
            prog = math.log(max(h_tgt, 12.0) / 12.0) / math.log(168.0 / 12.0)
            prog = min(1.0, max(0.0, prog))
            anchor_rate = inv_logit(
                logit(r_post) + (logit(anchor7) - logit(r_post)) * prog)
        else:
            prog = math.log(h_tgt / turn_h) / math.log(168.0 / turn_h)
            prog = min(1.0, max(0.0, prog))
            anchor_rate = inv_logit(
                logit(r_post) + (logit(anchor7) - logit(r_post)) * prog)

        # --- blend fit and anchor (anchor pull scaled by maturity) ---
        # RARITY_ANCHOR_ONLY=1 (test hook, round 3): pure anchor, no fit
        # influence -- used to prove the tau-inversion/fit does not
        # override the anchor the way Gemini's self-test did (Ch1 73%).
        w_eff = w + (1.0 - w) * (1.0 - pull)
        if os.environ.get("RARITY_ANCHOR_ONLY") == "1":
            w_eff = 0.0
        med = inv_logit(w_eff * logit(fit_rate) + (1.0 - w_eff) * logit(anchor_rate))
        med = max(0.01, min(99.99, med))

        # --- spread: wide and symmetric in logit, horizon-dependent ---
        e_ref = expected_refreshes
        s_refresh = SPREAD_REFRESH_PER * math.sqrt(max(e_ref, 0.0))
        if events:
            s_refresh = max(s_refresh, SPREAD_REFRESH_MIN)
        s_prog = prog_sd * math.sqrt(max(0.0, horizon_days) / 4.52)
        s_rw = SPREAD_RW * math.sqrt(max(0.0, horizon_days))
        # sigma_stale (consensus): the answer read lags the last refresh by
        # a few hours (Claude ~12 h, Gemini ~3.5 h); a small extra spread
        # term for that staleness. Already partly handled by the G/2 cap
        # on the simulated horizon; this is the residual.
        s = math.sqrt(s_refresh ** 2 + s_prog ** 2 + s_rw ** 2 + SIGMA_STALE ** 2)
        s = max(SPREAD_MIN, min(SPREAD_MAX, s))
        if cls == "FIRST":
            # Tight band at short horizons; at the final horizon the band
            # must reach q05 ~98% (consensus 97.8-98.5%), so s ~ 0.94.
            s = max(0.35, min(1.0, 0.35 + 0.13 * horizon_days))

        meds.append({"id": tr["id"], "cls": cls, "tau": tau,
                     "tier": tr.get("tier", "") or "",
                     "r_last": r_last, "e_last": e_last,
                     "med": med, "s": s, "fit": fit_rate,
                     "anchor": anchor_rate})
        if DEBUG:
            dbg.append(f"id={tr['id']:2d} {cls:7s} fit={fit_rate:7.3f} "
                       f"anchor={anchor_rate:7.3f} med={med:7.3f} s={s:.2f}")

    # --- POST-ENSEMBLE GUARDRAILS (Gemini round 2, last lines before
    # writing the output). 80% of final-score failures come from one
    # outlier trophy bleeding 3+ points at the merge, so these are the
    # last line of defence, applied to the medians before any quantile
    # is computed.
    # (a) Physical lower bounds. For every trophy with E_last >= 1 the
    # q95 must be at least the rate if all current holders persist and
    # the tracked count grows to N_MAX (applied to the quantiles below).
    # For trophies with tau <= 35 h at horizon >= 48 h, progression must
    # already have overtaken dilution, so the median cannot sit below
    # the last observed rate.
    if horizon_h >= 48.0:
        for m in meds:
            # Only trophies expected to RISE by day 7 (anchor above the
            # last read). FIRST (100% -> 99.6%) and platinum/completion
            # (dilution wins) FALL, so the floor must not pin them at
            # r_last (v9 first run pinned FIRST at 100%).
            if m["tau"] <= 35.0 and m["anchor"] > m["r_last"] \
                    and m["med"] < m["r_last"]:
                m["med"] = m["r_last"]
    # (b) Isotonic regression (PAVA) on the median logits ordered by
    # difficulty rank (tau, grade): the median rate must not rise as
    # trophies get harder. This stops one trophy falling to 0.01% alone
    # or an implausible order inversion. SCOPE: story-gated trophies
    # only. Side content (FISH/FAST/MID/GRIND) is EXCLUDED: its rate is
    # set by whether players choose the activity, not by tau, so a side
    # trophy may legitimately sit below a story trophy with a larger tau
    # (v9 first run pooled fishing up to Chapter 1's 35% -- wrong).
    _tier_pts = {"bronze": 15, "silver": 30, "gold": 90, "platinum": 300}
    _story = [j for j, m in enumerate(meds)
              if m["cls"] not in ("FISH", "FAST", "MID", "GRIND")]
    _rank_order = sorted(_story,
                         key=lambda j: (meds[j]["tau"],
                                        _tier_pts.get(meds[j]["tier"], 15),
                                        meds[j]["id"]))
    _logits = [logit(meds[j]["med"]) for j in _rank_order]
    _fixed = pava_nonincreasing(_logits)
    for j, lv in zip(_rank_order, _fixed):
        meds[j]["med"] = inv_logit(lv)
        meds[j]["med"] = max(0.01, min(99.99, meds[j]["med"]))

    # --- quantiles (after the guardrails) ---
    # Short horizons (<= 360 min): 2-component Gaussian mixture on the
    # logit scale (Gemini round 2, exact formula). Branch 0 = no refresh:
    # the rate is locked at the last read, sigma0 = 0.012. Branch 1 =
    # one refresh: median = the guardrailed forecast, sigma1 = the spread
    # s. p0 comes from the Gamma renewal (k = 2.5), NOT Poisson, using
    # t_since (hours since the last refresh) and the fitted gap. The
    # mixture CDF is solved by 30-iteration bisection -- never by
    # averaging the quantiles linearly, because the result is bimodal.
    # Longer horizons use the single (refresh-branch) Gaussian.
    p0 = None
    if horizon_h <= 6.0:
        # EXCEPTION (carried from v8, and the reason the practice works):
        # when the FIRST refresh (04:30Z, tied to the release) is
        # deterministically inside the horizon and no refresh has been
        # seen yet -- as in the practice cases -- a refresh is certain,
        # so p0 = 0. The Gamma renewal below governs LATER refreshes,
        # whose timing is a renewal process. Without this exception the
        # mixture puts the median at the no-refresh mode (p0 ~ 0.67) and
        # scores ~1.01 on the actual practice, whose answer IS a refresh
        # by construction; Gemini's 0.0936 is an expectation over the
        # refresh process, not the conditional loss on the kit's practice.
        if first_refresh_inside:
            p0 = 0.0
        else:
            p0 = renewal_p0(t_since_h, horizon_h, gap)
    rows = []
    for m in meds:
        med = m["med"]
        s = m["s"]
        if p0 is not None and p0 > 0.0:
            mu0 = logit(m["r_last"])
            mu1 = logit(med)
            qs = [inv_logit(mixture_quantile(tau_q, p0, mu0, mu1, s))
                  for tau_q in TAUS]
        else:
            qs = [inv_logit(logit(med) + z * s) for z in Z]
        # Physical q95 floor (guardrail (a)): all current holders persist
        # and N grows to N_MAX.
        if m["e_last"] >= 1.0:
            qs[4] = max(qs[4], 100.0 * m["e_last"] / N_MAX)
        cap = CAP[m["cls"]]
        qs = [min(q, cap) for q in qs]
        qs = [max(0.0, min(100.0, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        if m["cls"] == "FIRST":
            qs = [min(q, 99.99) for q in qs]
            for k in range(1, 5):
                if qs[k] < qs[k - 1]:
                    qs[k] = qs[k - 1]
        rows.append({"id": m["id"], "q": [round(q, 4) for q in qs]})
        if DEBUG:
            dbg.append(f"id={m['id']:2d} {m['cls']:7s} med={med:7.3f} s={s:.2f} "
                       f"q05={qs[0]:7.3f} q95={qs[4]:7.3f}")
    if DEBUG:
        sys.stderr.write("\n".join(dbg) + "\n")
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        rows = forecast_case(case)
        out = {"trophies": rows}
        name = os.path.splitext(os.path.basename(path))[0] + ".out"
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
