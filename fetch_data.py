"""
Step 1: download the raw data and save it to a local database (baseball.db).

Pulls, for each season:
  - MLB season stats (hitting + pitching) from MLB's stats API
  - Baseball Savant expected stats (xBA, xSLG, xwOBA, xERA)

Run it:
  python fetch_data.py                 # default seasons 2024-2026
  python fetch_data.py 2023 2024 2025  # pick your own

Rerun any time. It replaces the old data with fresh data, which is how the
projections stay up to date.
"""

import io
import sqlite3
import sys

import pandas as pd
import requests

SEASONS = [int(s) for s in sys.argv[1:]] or [2024, 2025, 2026]
DB = "baseball.db"


def mlb_stats(group, season):
    """One row per player for one season. group = 'hitting' or 'pitching'."""
    params = dict(stats="season", group=group, season=season, sportId=1,
                  playerPool="ALL", limit=5000)
    data = requests.get("https://statsapi.mlb.com/api/v1/stats",
                        params=params, timeout=60).json()
    rows = []
    for split in data["stats"][0]["splits"]:
        row = dict(split["stat"])
        row["player_id"] = split["player"]["id"]
        row["name"] = split["player"]["fullName"]
        row["team"] = split.get("team", {}).get("name", "")
        row["team_id"] = split.get("team", {}).get("id")
        row["pos"] = split.get("position", {}).get("abbreviation", "")
        row["season"] = season
        rows.append(row)
    df = pd.DataFrame(rows)
    # Innings come as text like "180.1" (180 and 1/3). We use outs instead, so drop it.
    return df.drop(columns=["inningsPitched"], errors="ignore")


def savant_expected(player_type, season):
    """Savant's expected stats leaderboard. player_type = 'batter' or 'pitcher'."""
    url = ("https://baseballsavant.mlb.com/leaderboard/expected_statistics"
           f"?type={player_type}&year={season}&position=&team=&filterType=bip&min=1&csv=true")
    text = requests.get(url, timeout=60).content.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"year": "season"})
    for col in ["era", "xera"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def team_runs(season):
    """Runs scored and games played by every team in one season."""
    params = dict(stats="season", group="hitting", season=season, sportId=1)
    data = requests.get("https://statsapi.mlb.com/api/v1/teams/stats",
                        params=params, timeout=60).json()
    return pd.DataFrame([{"team_id": s["team"]["id"], "team": s["team"]["name"],
                          "runs": s["stat"]["runs"], "games": s["stat"]["gamesPlayed"],
                          "season": season}
                         for s in data["stats"][0]["splits"]])


def current_rosters():
    """Who is on each team's 40-man roster right now. This is what catches
    trades and free agent signings: rerun it and players move teams."""
    teams = requests.get("https://statsapi.mlb.com/api/v1/teams",
                         params={"sportId": 1}, timeout=60).json()["teams"]
    rows = []
    for t in teams:
        roster = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{t['id']}/roster",
                              params={"rosterType": "40Man"}, timeout=60).json()
        for r in roster.get("roster", []):
            rows.append({"player_id": r["person"]["id"], "team_id": t["id"],
                         "team": t["name"]})
    return pd.DataFrame(rows)


def main():
    tables = {"hitting": [], "pitching": [], "savant_batter": [], "savant_pitcher": [],
              "team_runs": []}
    for season in SEASONS:
        print(f"Pulling {season}...")
        tables["hitting"].append(mlb_stats("hitting", season))
        tables["pitching"].append(mlb_stats("pitching", season))
        tables["savant_batter"].append(savant_expected("batter", season))
        tables["savant_pitcher"].append(savant_expected("pitcher", season))
        tables["team_runs"].append(team_runs(season))
    print("Pulling current rosters...")
    tables["rosters"] = [current_rosters()]

    with sqlite3.connect(DB) as db:
        for name, frames in tables.items():
            df = pd.concat(frames, ignore_index=True)
            df.to_sql(name, db, if_exists="replace", index=False)
            print(f"  {name}: {len(df)} rows")
    print(f"Saved to {DB}")


if __name__ == "__main__":
    main()
