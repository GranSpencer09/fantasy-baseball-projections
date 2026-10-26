"""
Step 4: build the draft-day board.

Reads draft_rankings_2027.csv and league.json, injects them into
board_template.html, and writes draft_board.html (one self-contained page).
Rerun after value.py whenever projections change.
"""

import json

import pandas as pd

TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS", "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "FA": "FA",
}
MAX_PLAYERS = 450   # 23 rounds x 10 teams = 230 picks, plus room for deep sleepers

board = pd.read_csv("draft_rankings_2027.csv").head(MAX_PLAYERS)
players = [
    {"id": f"{r['type']}{int(r.player_id)}", "n": r["name"], "t": TEAM_ABBR.get(r.team, r.team),
     "k": r["type"], "p": r.positions, "a": int(r.age), "pts": round(r.points),
     "v": round(r.VOR, 1)}
    for _, r in board.iterrows()
]
with open("league.json") as f:
    league = json.load(f)

html = open("board_template.html").read()
html = html.replace("/*PLAYERS*/[]", json.dumps(players, ensure_ascii=False))
html = html.replace("/*LEAGUE*/{}", json.dumps(league))
open("draft_board.html", "w").write(html)
print(f"draft_board.html built with {len(players)} players")
