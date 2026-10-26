# Fantasy Baseball Projections

Projects next season's fantasy points for every MLB player using my league's head-to-head points scoring, then turns them into draft rankings and a draft-day board.

## How to run

```
pip install pandas requests numpy
python fetch_data.py     # download 2016-2026 MLB, Savant, injury and Steamer/ZiPS data into baseball.db
python project.py        # projections_2027_hitters.csv and projections_2027_pitchers.csv
python value.py          # draft_rankings_2027.csv: value over replacement for my league
python build_board.py    # draft_board.html: draft-day board (mark picks, roster, best available)
python backtest.py       # grade the model on 7 past seasons
python rank_players.py   # (optional) rank last season's actual points
```

Change point values in `scoring.json`, roster settings in `league.json`, model settings at the top of `project.py`. Every setting there notes what the backtest found.

## The model

Based on Marcel, a simple projection system that holds up well against fancier ones, plus a few tested upgrades.

1. **Baseline:** last 3 seasons, weighted 5/4/3 for hitters and 3/2/1 for pitchers.
2. **Regression:** mix in league average so small samples get pulled toward normal. For pitchers, each stat is regressed by how fast it becomes reliable: strikeout rate is trusted quickly, hits allowed slowly.
3. **Savant correction:** move halfway from actual results toward expected stats (xwOBA for hitters, xERA and xBA for pitchers).
4. **Age (skill):** +0.6% per year under 29, -0.3% per year over.
5. **Playing time:** half of last season's workload, plus a tenth of the year before, plus a baseline, then cut 1.5% per year of age over 30.
6. **Score:** projected stats times league point values.
7. **Pro blend (off until 2027 projections are out):** average with Steamer and ZiPS, rescaled to our playing-time assumptions. The monthly refresh turns this on when they release.

Built and tested but switched off because the backtest said so:
- **Team context:** rates each lineup on last season's rosters vs current 40-man rosters and adjusts R/RBI and wins. Slightly hurt hitters in 5 of 7 seasons. Retest once real opening-day rosters are saved.
- **Injury list penalty:** past playing time already reflects missed games, so penalizing IL stints double counts.
- **Surgery flag:** too few players caught to matter.

## Backtest

`backtest.py` rewinds the model to each of 2019 and 2021-2026, projects that season using only earlier data, and grades it against what happened. Graded pool: the top 200 hitters and 150 pitchers by the prior season's points, including anyone who got hurt.

| Version | Hitter rank accuracy | Pitcher rank accuracy | Hitter avg miss | Pitcher avg miss |
|---|---|---|---|---|
| Last year's points | 0.513 | 0.383 | 320 | 226 |
| First model (Oct 26) | 0.530 | 0.390 | 285 | 200 |
| Current model | 0.549 | 0.410 | 280 | 197 |

Rank accuracy is Spearman rank correlation: 1.0 is perfect order, 0 is random. Biggest single gains: age-based playing time (hitters) and stat-by-stat regression (pitchers).

## Draft value

`value.py` converts points into value over replacement (VOR) using `league.json` (10 teams, C/1B/2B/3B/SS/3 OF/2 Util, 2 SP/2 RP/4 P). It fills every starting slot league-wide, takes the best leftover player at each position as replacement level, and ranks hitters and pitchers together by how far above that line they are. Position eligibility follows Yahoo's rule: 5 starts or 10 games at a position last season.

## Data sources

- MLB Stats API (statsapi.mlb.com): stats, rosters, positions, injured list moves
- Baseball Savant expected stats leaderboards
- FanGraphs Steamer and ZiPS projections

## Roadmap

- Depth charts and closer roles for playing time
- Park factors (Coors, etc.)
- Rookies and prospects from minor league stats
- Save opening-day rosters each spring so team context can be tested fairly
- In-season updates that blend projections with current stats
