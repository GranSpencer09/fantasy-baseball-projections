"""
Step 3: turn projected points into draft value.

Raw points aren't the whole story. What matters is how much better a player is
than the guy you could grab off waivers at the same position. That gap is
"value over replacement" (VOR).

How it works:
  1. Figure out which positions each player qualifies for (Yahoo rule:
     5 starts or 10 games at a position last season).
  2. Simulate the league: fill every starting slot for all 10 teams with the
     best available players, scarcest positions first.
  3. Replacement level at a position = the best player left over after every
     starting slot is filled. That's who you'd stream if your starter got hurt.
  4. VOR = player's points minus replacement level at his scarcest position.
  5. Rank everyone, hitters and pitchers together, by VOR.

Reads: projections CSVs, league.json, baseball.db
Writes: draft_rankings_2027.csv
"""

import json
import sqlite3

import pandas as pd

NEXT = 2027
LAST = NEXT - 1
OF_POSITIONS = {"LF", "CF", "RF", "OF"}
SCARCITY_ORDER = ["C", "SS", "2B", "3B", "1B", "OF"]  # fill hardest-to-find spots first


def hitter_eligibility(rule):
    """Map player_id -> set of positions, using last season (or the one before
    for players who missed last season)."""
    with sqlite3.connect("baseball.db") as db:
        pos = pd.read_sql("SELECT * FROM positions", db)
    pos["position"] = pos.position.where(~pos.position.isin(OF_POSITIONS), "OF")
    pos = pos[pos.position.isin(SCARCITY_ORDER)]
    pos = pos.groupby(["player_id", "season", "position"])[["games", "starts"]].sum().reset_index()
    pos = pos[(pos.games >= rule["min_games"]) | (pos.starts >= rule["min_starts"])]
    # Use each player's most recent season with any qualifying position
    latest = pos.groupby("player_id").season.transform("max")
    pos = pos[pos.season == latest]
    return pos.groupby("player_id").position.apply(set)


def pitcher_eligibility():
    """SP if 5+ starts last season, RP if 10+ relief appearances. Some are both."""
    with sqlite3.connect("baseball.db") as db:
        p = pd.read_sql("SELECT player_id, season, gamesStarted, gamesPitched FROM pitching", db)
    latest = p.groupby("player_id").season.transform("max")
    p = p[p.season == latest].set_index("player_id")
    elig = {}
    for pid, row in p.iterrows():
        e = set()
        if row.gamesStarted >= 5:
            e.add("SP")
        if row.gamesPitched - row.gamesStarted >= 10:
            e.add("RP")
        elig[pid] = e or {"SP" if row.gamesStarted >= row.gamesPitched / 2 else "RP"}
    return pd.Series(elig)


def fill_slots(players, slot_order, flex, capacity):
    """Go down the list best-first and give each player a starting slot.
    Try his scarcest eligible position, then the flex slot (Util or P).
    Returns each player's slot (or None if he didn't make a starting lineup)."""
    left = dict(capacity)
    assigned = {}
    for pid, row in players.iterrows():
        slot = None
        for pos in slot_order:
            if pos in row.elig and left.get(pos, 0) > 0:
                slot = pos
                break
        if slot is None and left[flex] > 0:
            slot = flex
        if slot:
            left[slot] -= 1
        assigned[pid] = slot
    return pd.Series(assigned)


def replacement_levels(players, slot_order, flex):
    """Best leftover player at each position after all starting slots are full."""
    bench = players[players.slot.isna()]
    levels = {}
    for pos in slot_order:
        eligible = bench[bench.elig.apply(lambda e: pos in e)]
        levels[pos] = eligible.points.max() if len(eligible) else 0
    levels[flex] = bench.points.max()
    return levels


def add_vor(players, slot_order, flex, capacity):
    players = players.sort_values("points", ascending=False).copy()
    players["slot"] = fill_slots(players, slot_order, flex, capacity)
    levels = replacement_levels(players, slot_order, flex)
    # Compare each player to the weakest replacement among positions he can play
    players["repl_pos"] = players.elig.apply(
        lambda e: min(list(e & set(slot_order)) + [flex], key=lambda p: levels[p]))
    players["repl_pts"] = players.repl_pos.map(levels)
    players["VOR"] = (players.points - players.repl_pts).round(1)
    return players, levels


def main():
    with open("league.json") as f:
        league = json.load(f)
    n = league["teams"]

    # ---------- Hitters ----------
    h = pd.read_csv(f"projections_{NEXT}_hitters.csv").set_index("player_id")
    h["elig"] = hitter_eligibility(league["eligibility"]).reindex(h.index)
    h["elig"] = h.elig.apply(lambda e: e if isinstance(e, set) else set())  # DH-only -> Util
    hit_cap = {pos: k * n for pos, k in league["hitter_slots"].items()}
    h, hit_levels = add_vor(h, SCARCITY_ORDER, "Util", hit_cap)
    h["type"] = "H"
    h["positions"] = h.elig.apply(lambda e: "/".join(p for p in SCARCITY_ORDER if p in e) or "Util")

    # ---------- Pitchers ----------
    p = pd.read_csv(f"projections_{NEXT}_pitchers.csv").set_index("player_id")
    p["elig"] = pitcher_eligibility().reindex(p.index)
    p["elig"] = p.apply(lambda r: r.elig if isinstance(r.elig, set) else {r.role}, axis=1)
    pit_cap = {pos: k * n for pos, k in league["pitcher_slots"].items()}
    p, pit_levels = add_vor(p, ["SP", "RP"], "P", pit_cap)
    p["type"] = "P"
    p["positions"] = p.elig.apply(lambda e: "/".join(x for x in ["SP", "RP"] if x in e))

    # ---------- Combined draft board ----------
    cols = ["name", "team", "type", "positions", "age", "points", "repl_pos", "repl_pts", "VOR"]
    board = pd.concat([h[cols], p[cols]]).sort_values("VOR", ascending=False)
    board.insert(0, "rank", range(1, len(board) + 1))
    board.insert(1, "round", ((board["rank"] - 1) // n + 1))
    board.to_csv(f"draft_rankings_{NEXT}.csv", index_label="player_id")

    print("Replacement levels (points of the best player left on waivers):")
    for k, v in {**hit_levels, **pit_levels}.items():
        print(f"  {k:5} {v:7.1f}")
    pd.set_option("display.width", 200)
    print(f"\nTop 40 overall")
    print(board.head(40)[["rank", "round", "name", "positions", "points", "VOR"]].to_string(index=False))


if __name__ == "__main__":
    main()
