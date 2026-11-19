# gta6-rarity-forecast-h

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-h/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-h/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Top-down forecast of GTA VI trophy earn rates on Exophase at 2026-11-26T00:00:00Z. Each trophy's trajectory is fitted in two phases from the case history alone: a **dilution** phase (an Exophase refresh jumps the tracked-player count and pulls every rate toward the newly tracked cohort's rate — the new cohort holds almost nothing of most trophies, a third of Chapter 1, nearly all of the fishing challenge, and all of the first trophy) and a **progression** phase (non-holders earn the trophy at a per-hour hazard that ramps up with the cohort's age, then decays as the active players finish). The new cohort's rate rises as the player base matures, and the turning point between the phases is estimated from the data (log-time / tracked-player count), not guessed. Day-7 medians are anchored by **tau inversion**: each story-gated trophy needs tau hours of relevant play (Chapter 1's effective threshold ~9.5 h, final mission ~72 h, platinum ~255 h) and the anchor comes from a piecewise log-linear curve of rate vs tau through the consensus points (0.15 h, 98%), (9.5 h, 41%), (72 h, 3.5%), (255 h, 0.022% -- platinum, geometric mean of the two external-AI estimates 0.019% / 0.026%). The fitted trajectory is blended with the anchor -- the anchor carries 65% of the weight with no refresh seen, so the fit never overrides it (Chapter-1 median ~41% on launch-night data). Staleness 3.5 h (the observed refresh gap is ~7 h); platinum growth sublinear. Spreads are wide and symmetric in logit — narrow (~0.4) at short horizons, where the first refresh is predictable, and wide (~1.25) at the 4.5-day horizon — and short horizons mix a "no refresh" branch with the refresh branch in logit space.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `<id>.out` per case: five quantiles [0.05, 0.25, 0.5, 0.75, 0.95] of each trophy's earn rate at the case target. Python 3.13, standard library only, no network, deterministic (two runs byte-identical).

---

<sub>Entered on [make.fun](https://make.fun/?ref=9ncgmbd6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
