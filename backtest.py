"""
Grade the projection model on seasons we already know the answers to.

For each test season (2025 and 2026):
  1. Rewind: give the model only the 3 seasons before it.
  2. Project the test season with several versions of the model.
  3. Compare each version's projections to what actually happened.

Who gets graded: the players you'd actually draft, as you'd have known them
that winter. That's the top 200 hitters and top 150 pitchers by the previous
season's points. A player who got hurt and scored 0 still counts. Missing
on injuries is part of the job.

Grades:
  rank_corr   How well the projected order matched the real order.
              1.0 = perfect, 0 = random. This matters most for drafting.
  top_hits    Of the players a version ranked in the top 50 (hitters) or
              top 40 (pitchers), how many really finished there.
  avg_miss    Average points off per player. Our model is conservative on
              playing time on purpose, so expect this to run high.

Run it:  python backtest.py
"""

import json

import pandas as pd

import project

TEST_SEASONS = [2025, 2026]
HIT_POOL, PIT_POOL = 200, 150
HIT_TOP, PIT_TOP = 50, 40

VERSIONS = {
    "Last year's points":       None,
    "3-yr model, no Savant":    dict(savant=0.0, team=0.0),
    "+ Savant (half trust)":    dict(savant=0.5, team=0.0),
    "+ Savant (full trust)":    dict(savant=1.0, team=0.0),
    "+ Savant half + team":     dict(savant=0.5, team=0.5),
}


def actual_points(season, rules):
    """Real fantasy points for every player in a season."""
    h = project.load("hitting")
    h = h[(h.season == season) & (h.pos != "P")].set_index("player_id")
    h["singles"] = h.hits - h.doubles - h.triples - h.homeRuns
    p = project.load("pitching")
    p = p[(p.season == season) & p.pos.isin(["P", "TWP"])].set_index("player_id")
    return project.score(h, rules["batting"]), project.score(p, rules["pitching"]), h.name, p.name


def grade(proj, actual, pool, top_n):
    """Compare projections to reality for the players in the pool."""
    df = pd.DataFrame({"proj": proj.reindex(pool), "actual": actual.reindex(pool).fillna(0)})
    df["proj"] = df.proj.fillna(0)
    real_top = set(df.actual.nlargest(top_n).index)
    proj_top = set(df.proj.nlargest(top_n).index)
    return {
        "rank_corr": round(df.proj.corr(df.actual, method="spearman"), 3),
        "top_hits": f"{len(real_top & proj_top)}/{top_n}",
        "avg_miss": round((df.proj - df.actual).abs().mean(), 1),
    }


def main():
    with open("scoring.json") as f:
        rules = json.load(f)
    rows = []
    for season in TEST_SEASONS:
        act_h, act_p, _, _ = actual_points(season, rules)
        prev_h, prev_p, _, _ = actual_points(season - 1, rules)
        hit_pool = prev_h.nlargest(HIT_POOL).index
        pit_pool = prev_p.nlargest(PIT_POOL).index

        for label, settings in VERSIONS.items():
            if settings is None:
                ph, pp = prev_h, prev_p
            else:
                project.configure(last=season - 1, backtest=True, **settings)
                hitters, teams = project.project_hitters(rules)
                pitchers = project.project_pitchers(rules, teams.proj_R_per_G,
                                                    teams.old_roster_R_per_G)
                ph, pp = hitters.points, pitchers.points
            for kind, proj, actual, pool, top in [("Hitters", ph, act_h, hit_pool, HIT_TOP),
                                                  ("Pitchers", pp, act_p, pit_pool, PIT_TOP)]:
                rows.append({"season": season, "players": kind, "version": label,
                             **grade(proj, actual, pool, top)})

    res = pd.DataFrame(rows)
    res.to_csv("backtest_results.csv", index=False)
    pd.set_option("display.width", 200)
    for kind in ["Hitters", "Pitchers"]:
        print(f"\n===== {kind} =====")
        t = res[res.players == kind].pivot(index="version", columns="season",
                                           values=["rank_corr", "top_hits", "avg_miss"])
        print(t.reindex(VERSIONS.keys()).to_string())


if __name__ == "__main__":
    main()
