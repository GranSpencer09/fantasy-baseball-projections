"""
Grade the projection model on seasons we already know the answers to.

For each test season:
  1. Rewind: give the model only the seasons before it.
  2. Project the test season with several versions of the model.
  3. Compare each version's projections to what actually happened.

Test seasons: 2019 and 2021-2026. 2020 (60 games) is skipped as a test
season but still used as input, which is a realistic handicap.

Who gets graded: the players you'd actually draft, as you'd have known them
that winter. That's the top 200 hitters and top 150 pitchers by the previous
season's points. A player who got hurt and scored 0 still counts.

Grades:
  rank_corr   How well the projected order matched the real order.
              1.0 = perfect, 0 = random. This matters most for drafting.
  top_hits    Of the players a version ranked in the top 50 (hitters) or
              top 40 (pitchers), the share that really finished there.
  avg_miss    Average points off per player.

Run it:  python backtest.py
Try your own idea:  add a line to VERSIONS with the settings you want to test.
"""

import json

import pandas as pd

import project

TEST_SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
HIT_POOL, PIT_POOL = 200, 150
HIT_TOP, PIT_TOP = 50, 40

# Any setting in project.py can be overridden here by name.
VERSIONS = {
    "Last year's points":     None,
    "Current model":          {},
    "No Savant":              dict(SAVANT_TRUST=0.0),
    "Full Savant":            dict(SAVANT_TRUST=1.0),
    "With team context":      dict(TEAM_TRUST=0.5),
    "Heavier regression":     dict(HIT_REGRESS_PA=1200, PIT_REGRESS_BF=900),
    "No age playing time":    dict(AGE_PT=0.0),
    "No stat-by-stat (P)":    dict(PIT_STAT_STRENGTH=0.0),
    "Original Oct 26 model":  dict(HIT_REGRESS_PA=1200, PIT_REGRESS_BF=900, AGE_PT=0.0,
                                   PIT_STAT_STRENGTH=0.0),
}

RULES = json.load(open("scoring.json"))
DEFAULTS = {k: getattr(project, k) for k in dir(project) if k.isupper()}
_cache = {}


def actual_points(season):
    """Real fantasy points for every player in a season (hitters, pitchers)."""
    if season not in _cache:
        h = project.load("hitting")
        h = h[(h.season == season) & (h.pos != "P")].set_index("player_id")
        h["singles"] = h.hits - h.doubles - h.triples - h.homeRuns
        p = project.load("pitching")
        p = p[(p.season == season) & p.pos.isin(["P", "TWP"])].set_index("player_id")
        _cache[season] = (project.score(h, RULES["batting"]), project.score(p, RULES["pitching"]))
    return _cache[season]


def grade(proj, actual, pool, top_n):
    df = pd.DataFrame({"proj": proj.reindex(pool).fillna(0),
                       "actual": actual.reindex(pool).fillna(0)})
    real_top = set(df.actual.nlargest(top_n).index)
    proj_top = set(df.proj.nlargest(top_n).index)
    return {"rank_corr": df.proj.corr(df.actual, method="spearman"),
            "top_hits": len(real_top & proj_top) / top_n,
            "avg_miss": (df.proj - df.actual).abs().mean()}


def run_version(settings, seasons=TEST_SEASONS):
    """Grade one version of the model on every test season. Returns a table."""
    rows = []
    for season in seasons:
        act_h, act_p = actual_points(season)
        prev_h, prev_p = actual_points(season - 1)
        if settings is None:
            ph, pp = prev_h, prev_p
        else:
            for k, v in DEFAULTS.items():          # reset, then apply this version
                setattr(project, k, v)
            for k, v in settings.items():
                setattr(project, k, v)
            project.configure(last=season - 1, backtest=True)
            hitters, teams = project.project_hitters(RULES)
            pitchers = project.project_pitchers(RULES, teams.proj_R_per_G, teams.old_roster_R_per_G)
            ph, pp = hitters.points, pitchers.points
        for kind, proj, act, pool, top in [
                ("Hitters", ph, act_h, prev_h.nlargest(HIT_POOL).index, HIT_TOP),
                ("Pitchers", pp, act_p, prev_p.nlargest(PIT_POOL).index, PIT_TOP)]:
            rows.append({"season": season, "players": kind, **grade(proj, act, pool, top)})
    for k, v in DEFAULTS.items():
        setattr(project, k, v)
    return pd.DataFrame(rows)


def summarize(results):
    """Average each grade across seasons, hitters and pitchers side by side."""
    out = {}
    for label, df in results.items():
        g = df.groupby("players")[["rank_corr", "top_hits", "avg_miss"]].mean()
        out[label] = {"H rank": g.loc["Hitters", "rank_corr"], "H top": g.loc["Hitters", "top_hits"],
                      "H miss": g.loc["Hitters", "avg_miss"], "P rank": g.loc["Pitchers", "rank_corr"],
                      "P top": g.loc["Pitchers", "top_hits"], "P miss": g.loc["Pitchers", "avg_miss"]}
    return pd.DataFrame(out).T.round(3)


def main():
    results = {label: run_version(s) for label, s in VERSIONS.items()}
    long = pd.concat([df.assign(version=label) for label, df in results.items()])
    long.round(3).to_csv("backtest_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Averages over {len(TEST_SEASONS)} seasons ({', '.join(map(str, TEST_SEASONS))})")
    print(summarize(results).to_string())
    print("\nHitter rank_corr by season")
    print(long[long.players == "Hitters"].pivot(index="version", columns="season",
                                                 values="rank_corr").round(3).to_string())


if __name__ == "__main__":
    main()
