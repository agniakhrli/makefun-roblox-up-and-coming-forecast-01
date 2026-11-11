# makefun-roblox-up-and-coming-forecast-01

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/agniakhrli/makefun-roblox-up-and-coming-forecast-01/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/agniakhrli/makefun-roblox-up-and-coming-forecast-01/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

# Roblox Up-and-Coming forecast — make.fun entry

Forecasts Roblox's Up-and-Coming chart top 10, place by place, for the 72
hours after entries close. One static forecast, scored against every hourly
read of the window by log score.

## Method

An empirical transition kernel over the chart's history (`main.py`, Python
3.13, standard library only, no network, no model calls, no randomness):

- For every pair of past reads, record how games moved (rank now -> rank
  later, or left the chart).
- For each horizon h in a grid over the 72h window, estimate per
  current-rank bucket the distribution over future places 1..10 from those
  transitions (Gaussian kernel over the pair's time gap, recency-weighted
  with a 7-day half-life, Laplace prior).
- Average the per-horizon distributions over h (trapezoid weights) to get
  one static forecast; name the top 18 games per place, renormalised to
  0.75 per place, leaving 0.25 unnamed for games that join the chart during
  the window.
- Games that age out of the chart (published > 28 days ago, from
  `details.created` where the history carries it) contribute nothing past
  their expiry hour.

## Backtest (local, exact scorer from the rules)

The practice window's 10 scored reads were reconstructed from the public
archive (github.com/Kenny-Dhub/roblox-snapshots) — the converter reproduces
the kit's own history lines exactly, and the scorer reproduces the house
baseline's published -43.512039 on that window, so the numbers are real.

| backtest | this entry | house baseline |
|---|---|---|
| practice window (10 reads) | **-35.182** | -43.512 |
| rolling cut @ read 20 | -38.883 | -50.269 |
| rolling cut @ read 24 | -40.279 | -51.967 |
| rolling cut @ read 28 | -30.827 | -32.918 |
| rolling cut @ read 30 | -32.746 | -30.435 |
| mean | **-35.583** | -41.820 |

Typical probability per place on the practice window: 0.0297 (baseline
0.0129). Parameters were grid-searched on these five backtests; cross-chart
signals (Top Trending / Top Playing Now presence, rank velocity, playing
growth) were tested and rejected — they hurt on every cut.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same history are byte-identical. Runtime ~0.2s on the
34-read practice history, ~2s on a synthetic 500-read hourly history
(limit 300s). Output `forecast.json` is 10,411 bytes on the practice
history (limit 1,048,576).
