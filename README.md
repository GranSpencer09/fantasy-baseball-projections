# Fantasy Baseball Projections

Projects next season's fantasy points for every MLB player using my league's head-to-head points scoring.

## How to run

```
pip install pandas requests numpy
python fetch_data.py     # download 3 seasons of MLB + Baseball Savant data into baseball.db
python project.py        # build projections_2027_hitters.csv and projections_2027_pitchers.csv
python rank_players.py   # (optional) rank last season's actual points
```

Change point values in `scoring.json`. Change model settings at the top of `project.py`.

## The model

Based on Marcel, a simple projection system that holds up well against fancier ones.

1. **Baseline:** last 3 seasons, weighted 5/4/3 for hitters and 3/2/1 for pitchers.
2. **Regression:** mix in league average. Small samples get pulled more.
3. **Savant correction:** move halfway from actual results toward expected stats (xwOBA for hitters, xERA and xBA for pitchers).
4. **Age:** +0.6% per year under 29, -0.3% per year over.
5. **Playing time:** half of last season's workload, plus a tenth of the year before, plus a baseline.
6. **Score:** projected stats times league point values.

## Data sources

- MLB Stats API (statsapi.mlb.com)
- Baseball Savant expected stats leaderboards

## Roadmap

- Team context (offense quality for wins, R, RBI)
- Injury data and depth charts for playing time
- In-season updates that blend projections with current stats
- Draft-day board
