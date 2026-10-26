# Fantasy Baseball Projections

Projects next season's fantasy points for every MLB player using my league's head-to-head points scoring.

## How to run

```
pip install pandas requests numpy
python fetch_data.py     # download 3 seasons of MLB + Baseball Savant data into baseball.db
python project.py        # build projections_2027_hitters.csv and projections_2027_pitchers.csv
python value.py          # draft_rankings_2027.csv: value over replacement for my league
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
6. **Team context:** rate every team's lineup twice, on last season's rosters and on current 40-man rosters. If a team's lineup got better through trades or signings, its hitters' R/RBI and its pitchers' wins go up (half-strength, capped). Rerun `fetch_data.py` during the offseason to catch moves.
7. **Score:** projected stats times league point values.

Outputs include `projections_2027_team_offense.csv`, showing each team's projected runs per game and how much roster moves changed it.

## Draft value

`value.py` converts points into value over replacement (VOR) using the roster settings in `league.json` (10 teams, C/1B/2B/3B/SS/3 OF/2 Util, 2 SP/2 RP/4 P). It fills every starting slot league-wide, takes the best leftover player at each position as replacement level, and ranks hitters and pitchers together by how far above that line they are. Position eligibility follows Yahoo's rule: 5 starts or 10 games at a position last season.

## Data sources

- MLB Stats API (statsapi.mlb.com)
- Baseball Savant expected stats leaderboards

## Roadmap

- Park factors (Coors, etc.)
- Bullpen role changes for saves
- Injury data and depth charts for playing time
- In-season updates that blend projections with current stats
- Draft-day board
