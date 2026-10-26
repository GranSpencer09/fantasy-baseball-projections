"""
Fantasy baseball player rankings, v1.

What it does:
  1. Downloads a full season of MLB stats (free, from MLB's own stats API).
  2. Scores every player using the point values in scoring.json.
  3. Ranks hitters and pitchers and saves the results to CSV files.

Run it:
  python rank_players.py            # uses the 2026 season
  python rank_players.py 2025       # any other season
"""

import json
import sys

import pandas as pd
import requests

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
API = "https://statsapi.mlb.com/api/v1/stats"


def fetch_stats(group):
    """Get one season of stats for every player. group is 'hitting' or 'pitching'."""
    params = {
        "stats": "season",
        "group": group,
        "season": SEASON,
        "sportId": 1,          # 1 = MLB
        "playerPool": "ALL",   # everyone, not just qualified players
        "limit": 5000,
    }
    data = requests.get(API, params=params, timeout=30).json()

    # The API returns nested JSON. Flatten each player into one simple row.
    rows = []
    for split in data["stats"][0]["splits"]:
        row = dict(split["stat"])
        row["name"] = split["player"]["fullName"]
        row["team"] = split.get("team", {}).get("name", "")
        row["pos"] = split.get("position", {}).get("abbreviation", "")
        rows.append(row)
    return pd.DataFrame(rows)


def score(df, weights):
    """Multiply each stat by its point value and add them all up."""
    total = 0
    for stat, points in weights.items():
        total = total + df[stat] * points
    return total.round(1)


def main():
    with open("scoring.json") as f:
        rules = json.load(f)

    # ---------- Hitters ----------
    hitters = fetch_stats("hitting")
    hitters = hitters[hitters["pos"] != "P"]  # drop pitchers who batted
    # The API doesn't give singles directly, so: singles = hits minus extra-base hits
    hitters["singles"] = (
        hitters["hits"] - hitters["doubles"] - hitters["triples"] - hitters["homeRuns"]
    )
    hitters["points"] = score(hitters, rules["batting"])
    hitters["pts_per_game"] = (hitters["points"] / hitters["gamesPlayed"]).round(2)
    hitters = hitters.sort_values("points", ascending=False)
    hitters["rank"] = range(1, len(hitters) + 1)

    hit_cols = ["rank", "name", "team", "pos", "points", "pts_per_game", "gamesPlayed",
                "plateAppearances", "homeRuns", "runs", "rbi", "stolenBases",
                "baseOnBalls", "obp"]
    hitters[hit_cols].to_csv(f"hitters_{SEASON}.csv", index=False)

    # ---------- Pitchers ----------
    pitchers = fetch_stats("pitching")
    pitchers["points"] = score(pitchers, rules["pitching"])
    # Call someone a starter if at least half their games were starts
    pitchers["role"] = (pitchers["gamesStarted"] >= pitchers["gamesPitched"] / 2).map(
        {True: "SP", False: "RP"}
    )
    pitchers["pts_per_app"] = (pitchers["points"] / pitchers["gamesPitched"]).round(2)
    pitchers = pitchers.sort_values("points", ascending=False)
    pitchers["rank"] = range(1, len(pitchers) + 1)

    pit_cols = ["rank", "name", "team", "role", "points", "pts_per_app", "gamesPitched",
                "gamesStarted", "inningsPitched", "strikeOuts", "wins", "saves",
                "era", "whip"]
    pitchers[pit_cols].to_csv(f"pitchers_{SEASON}.csv", index=False)

    # ---------- Quick look in the terminal ----------
    print(f"\nTop 15 hitters, {SEASON}")
    print(hitters[["rank", "name", "pos", "points", "pts_per_game"]].head(15).to_string(index=False))
    print(f"\nTop 10 starters, {SEASON}")
    print(pitchers[pitchers.role == "SP"][["rank", "name", "points", "pts_per_app"]].head(10).to_string(index=False))
    print(f"\nTop 10 relievers, {SEASON}")
    print(pitchers[pitchers.role == "RP"][["rank", "name", "points", "saves"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
