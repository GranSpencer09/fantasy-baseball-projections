"""
Blend our projections with professional ones (Steamer and ZiPS from FanGraphs).

Why: averaging independent projection systems almost always beats any one
of them, because each system's mistakes partly cancel out.

How:
  - fetch_pro() downloads Steamer and ZiPS stat projections (run by fetch_data.py).
  - pro_points() scores them with our league's rules and averages the systems.
  - project.py mixes them with our model's points when PRO_BLEND > 0.

Caveat: FanGraphs only serves the current projections, so the blend can't be
backtested. 50/50 is the standard choice for averaging projection systems.
Keep PRO_BLEND at 0 until the projections for the season you're drafting
for are released (Steamer usually in November, ZiPS rolls out through winter).
"""

import pandas as pd
import requests

SYSTEMS = ["steamer", "zips"]
URL = "https://www.fangraphs.com/api/projections"
BAT_COLS = {"1B": "singles", "2B": "doubles", "3B": "triples", "HR": "homeRuns", "R": "runs",
            "RBI": "rbi", "SB": "stolenBases", "BB": "baseOnBalls", "HBP": "hitByPitch", "PA": "PA"}
PIT_COLS = {"W": "wins", "SV": "saves", "IP": "IP", "SO": "strikeOuts", "H": "hits",
            "ER": "earnedRuns", "BB": "baseOnBalls", "HBP": "hitBatsmen"}


def fetch_pro():
    """Download every system's hitter and pitcher projections into one table."""
    frames = []
    for system in SYSTEMS:
        for kind, cols in [("bat", BAT_COLS), ("pit", PIT_COLS)]:
            params = dict(type=system, stats=kind, pos="all", team=0, players=0, lg="all")
            data = requests.get(URL, params=params, timeout=120).json()
            df = pd.DataFrame(data)
            df = df[df.xMLBAMID.notna()].rename(columns=cols)
            df["player_id"] = df.xMLBAMID.astype(int)
            df["system"], df["kind"] = system, kind
            frames.append(df[["player_id", "system", "kind", "PlayerName"] + list(cols.values())])
    return pd.concat(frames, ignore_index=True)


def pro_points(pro, rules):
    """Score each system's projections with our rules, then average the systems."""
    bat = pro[pro.kind == "bat"].copy()
    bat["points"] = sum(bat[s] * v for s, v in rules["batting"].items())
    pit = pro[pro.kind == "pit"].copy()
    pit["outs"] = pit.IP * 3
    pit["points"] = sum(pit[s] * v for s, v in rules["pitching"].items())
    return (bat.groupby("player_id").points.mean(),
            pit.groupby("player_id").points.mean())
