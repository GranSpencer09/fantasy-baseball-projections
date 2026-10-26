"""
Step 2: project next season's fantasy points for every player.

Reads baseball.db (made by fetch_data.py) and scoring.json, then writes:
  projections_hitters.csv
  projections_pitchers.csv

The model, in plain English:
  1. BASELINE   Average the last 3 seasons, counting recent years more.
  2. REGRESS    Pull everyone a bit toward league average. Small samples get
                pulled more, because a hot 150 PA could just be luck.
  3. SAVANT     If a player's quality of contact (xwOBA, xERA) says he was
                better or worse than his results, move halfway toward the
                expected number. Luck tends to even out.
  4. AGE        Young players improve a little, older players decline.
  5. PLAYING    Project plate appearances / batters faced from recent
     TIME       workload. Volume matters a lot in a points league.
  6. SCORE      Multiply projected stats by your league's point values.

This is based on "Marcel," a famously simple projection system that holds
its own against much fancier ones. Every number marked TUNABLE is a judgment
call you can change.
"""

import json
import sqlite3

import numpy as np
import pandas as pd

DB = "baseball.db"
LAST = 2026           # most recent completed season
NEXT = LAST + 1       # season we're projecting

# ---------- TUNABLE settings ----------
HIT_WEIGHTS = {LAST: 5, LAST - 1: 4, LAST - 2: 3}   # how much each season counts
PIT_WEIGHTS = {LAST: 3, LAST - 1: 2, LAST - 2: 1}
HIT_REGRESS_PA = 600      # league-average PA mixed in (weighted units). Backtest: 600 beat 1200
PIT_REGRESS_BF = 450      # league-average batters faced mixed in. Backtest: 450 beat 900
SAVANT_TRUST = 0.5        # 0 = ignore Savant, 1 = fully trust expected stats
PEAK_AGE = 29
TEAM_TRUST = 0.0          # how much a better/worse lineup moves R, RBI and wins. Backtest (7 seasons): slightly hurts, so off. Try 0.5 again once opening-day rosters are saved
# Stat-by-stat regression. Some stats tell you about a player's true skill
# after a few weeks (strikeouts, walks); others take years (doubles, hits
# allowed). Numbers are relative: 0.5 = trust it twice as fast as average,
# 2 = need twice as much data. Strength 0 = treat all stats the same.
HIT_STAT_SPEED = {"singles": 1.0, "doubles": 2.0, "triples": 2.0, "homeRuns": 0.5,
                  "baseOnBalls": 0.4, "hitByPitch": 0.6, "stolenBases": 0.5,
                  "runs": 1.0, "rbi": 1.0}
PIT_STAT_SPEED = {"outs": 1.0, "strikeOuts": 0.3, "hits": 2.0, "earnedRuns": 2.0,
                  "baseOnBalls": 0.6, "hitBatsmen": 0.8}
HIT_STAT_STRENGTH = 0.0   # Backtest: no help for hitters, so off
PIT_STAT_STRENGTH = 1.0   # Backtest: helps pitchers (trust K rate fast, hits allowed slow)
PT_LAST, PT_PREV, PT_BASE = 0.5, 0.1, 1.0   # playing time: share of last year, year before, baseline
IL_PENALTY = 0.0          # cut playing time by IL history. Backtest: hurts (past PA already counts it), so off
SURGERY_PT = 1.0          # multiplier if he ended last season hurt after surgery. Backtest: no effect, off
AGE_PT = 0.015            # playing time lost per year of age over 30. Backtest: 1.5% best
PRO_BLEND = 0.0           # share of points from Steamer/ZiPS (0.5 = 50/50). Turn on once next
                          # season's pro projections are out. See pro_blend.py.
LINEUP_SIZE = 13          # hitters per team that count toward team offense
SKILL_STATS = ["singles", "doubles", "triples", "homeRuns", "baseOnBalls",
               "hitByPitch", "stolenBases"]  # a hitter's own production, not teammates'

HIT_RATE_STATS = ["singles", "doubles", "triples", "homeRuns", "baseOnBalls",
                  "hitByPitch", "stolenBases", "runs", "rbi"]
CONTACT_STATS = ["singles", "doubles", "triples", "homeRuns", "runs", "rbi"]  # luck-sensitive
PIT_RATE_STATS = ["outs", "strikeOuts", "hits", "earnedRuns", "baseOnBalls", "hitBatsmen"]
BACKTEST = False          # True when rewinding to test the model on a past season


def configure(last=None, savant=None, team=None, backtest=None):
    """Change settings from another script (used by backtest.py).
    last=2025 means 'pretend 2025 just ended and project 2026.'"""
    global LAST, NEXT, HIT_WEIGHTS, PIT_WEIGHTS, SAVANT_TRUST, TEAM_TRUST, BACKTEST
    if last is not None:
        LAST, NEXT = last, last + 1
        HIT_WEIGHTS = {LAST: 5, LAST - 1: 4, LAST - 2: 3}
        PIT_WEIGHTS = {LAST: 3, LAST - 1: 2, LAST - 2: 1}
    if savant is not None:
        SAVANT_TRUST = savant
    if team is not None:
        TEAM_TRUST = team
    if backtest is not None:
        BACKTEST = backtest


def load(table):
    with sqlite3.connect(DB) as db:
        return pd.read_sql(f"SELECT * FROM {table}", db)


def weighted_sum(df, cols, weights):
    """Sum each stat across seasons, multiplying each season by its weight."""
    df = df[df.season.isin(weights)].copy()
    w = df.season.map(weights)
    for c in cols:
        df[c] = df[c] * w
    return df.groupby("player_id")[cols].sum()


def age_factor(age):
    """+0.6% per year under the peak age, -0.3% per year over it."""
    return np.where(age < PEAK_AGE, 1 + 0.006 * (PEAK_AGE - age),
                    1 - 0.003 * (age - PEAK_AGE))


def savant_factor(sav, weights, actual, expected, trust, low, high):
    """How much better/worse a player 'should' have been, as a multiplier.
    Example: actual wOBA .300, expected .330 -> 10% better -> factor 1.05 at trust 0.5."""
    sav = sav[sav.season.isin(weights)].copy()
    sav["w"] = sav.season.map(weights) * sav.pa
    sav["a"] = sav[actual] * sav.w
    sav["e"] = sav[expected] * sav.w
    g = sav.groupby("player_id")[["a", "e", "w"]].sum()
    ratio = (g.e / g.w) / (g.a / g.w)
    return (1 + trust * (ratio - 1)).clip(low, high), g


def score(df, weights):
    return sum(df[stat] * pts for stat, pts in weights.items()).round(1)


def injury_features(weights):
    """Two health signals per player, using only seasons up to LAST:
      il_share  Weighted share of recent seasons spent on the injured list.
      surgery   Ended last season on the IL after surgery (Tommy John, etc.)."""
    inj = load("injuries")
    inj = inj[inj.season <= LAST].copy()
    inj["date"] = pd.to_datetime(inj.date)
    inj = inj.sort_values(["player_id", "date"])
    stints = []
    for pid, g in inj.groupby("player_id"):
        start = None
        for _, r in g.iterrows():
            if r.action == "placed" and start is None:
                start = r.date
            elif r.action == "activated" and start is not None:
                stints.append((pid, start, r.date))
                start = None
        if start is not None:
            stints.append((pid, start, None))   # never activated: still hurt
    rows = []
    for pid, s, e in stints:
        yr = s.year
        open_, close = pd.Timestamp(f"{yr}-03-28"), pd.Timestamp(f"{yr}-09-30")
        end = min(e if e is not None else close, close)
        days = max((end - max(s, open_)).days, 0)
        rows.append({"player_id": pid, "season": yr, "days": days})
    days = pd.DataFrame(rows, columns=["player_id", "season", "days"])
    days = days[days.season.isin(weights)]
    days["w"] = days.season.map(weights) * days.days
    il_share = days.groupby("player_id").w.sum() / (186 * sum(weights.values()))

    # Ended last season hurt after surgery: last IL move of the season mentions surgery
    last = inj[inj.season == LAST].groupby("player_id").last()
    words = "surgery|tommy john|ucl|reconstruction"
    hurt = last[(last.action != "activated") &
                inj[inj.season == LAST].groupby("player_id").description
                .apply(lambda d: d.str.lower().str.contains(words).any())]
    return il_share, set(hurt.index)


def health_factor(index, age, weights):
    """Multiplier on projected playing time from injury history and age."""
    il_share, surgery = injury_features(weights)
    f = 1 - IL_PENALTY * il_share.reindex(index).fillna(0)
    f = f * pd.Series(np.where(index.isin(list(surgery)), SURGERY_PT, 1.0), index=index)
    f = f * (1 - AGE_PT * (age.reindex(index) - 30).clip(lower=0).fillna(0))
    return f.clip(0.2, 1.0)


def team_context():
    """Each team's runs per game last season, and each player's current team."""
    tr = load("team_runs")
    tr = tr[tr.season == LAST].set_index("team_id")
    old_rg = tr.runs / tr.games
    if BACKTEST:
        # No saved opening-day rosters from past years, so use the team each
        # player played for in the projected season as a stand-in.
        frames = [load("hitting"), load("pitching")]
        roster = pd.concat([f[f.season == NEXT][["player_id", "team_id", "team"]] for f in frames])
    else:
        roster = load("rosters")
    roster = roster.drop_duplicates("player_id").set_index("player_id")
    return old_rg, roster


def project_team_offense(hitters, team_ids, league_rg, rules):
    """Projected runs per game for each team, given which team each hitter is on.
    Steps: score each hitter's own production (no R/RBI, since those depend on
    teammates), take each team's top hitters by playing time, average their
    production, and scale league-average runs per game by how good that is.
    Runs grow faster than lineup quality (more baserunners for each hit to
    drive in), so quality is squared."""
    df = hitters.copy()
    skill_weights = {s: rules["batting"][s] for s in SKILL_STATS}
    df["skill"] = score(df, skill_weights)
    df["team_id"] = team_ids.reindex(df.index)
    df = df.dropna(subset=["team_id"])
    top = df.sort_values("PA", ascending=False).groupby("team_id").head(LINEUP_SIZE)
    g = top.groupby("team_id")[["skill", "PA"]].sum()
    quality = g.skill / g.PA
    return league_rg * (quality / quality.mean()) ** 2


def team_factor(new_rg, old_rg, low, high):
    """Better lineup than last year -> more runs/RBI/wins. Half-strength by default."""
    return (1 + TEAM_TRUST * (new_rg / old_rg - 1)).clip(low, high).fillna(1.0)


def project_hitters(rules):
    h = load("hitting")
    h = h[(h.pos != "P") & (h.season <= LAST)]   # never peek at the future
    h["singles"] = h.hits - h.doubles - h.triples - h.homeRuns
    cols = HIT_RATE_STATS + ["plateAppearances"]

    # 1. Weighted baseline
    tot = weighted_sum(h, cols, HIT_WEIGHTS)

    # 2. Regress to league average (per-PA rates from the last season)
    lg = h[h.season == LAST]
    lg_rate = lg[HIT_RATE_STATS].sum() / lg.plateAppearances.sum()
    rates = pd.DataFrame(index=tot.index)
    for s in HIT_RATE_STATS:
        reg = HIT_REGRESS_PA * HIT_STAT_SPEED[s] ** HIT_STAT_STRENGTH
        rates[s] = (tot[s] + lg_rate[s] * reg) / (tot.plateAppearances + reg)

    # 3. Savant luck correction (on contact stats only, not walks or steals)
    f, sav = savant_factor(load("savant_batter"), HIT_WEIGHTS, "woba", "est_woba",
                           SAVANT_TRUST, 0.85, 1.15)
    rates["savant_adj"] = f.reindex(rates.index).fillna(1.0)
    for s in CONTACT_STATS:
        rates[s] *= rates.savant_adj

    # Info: player identity and recent history
    latest = h.sort_values("season").groupby("player_id").last()
    pa = h.pivot_table(index="player_id", columns="season", values="plateAppearances",
                       aggfunc="sum").fillna(0)

    # 4. Age
    age_next = latest.age + (NEXT - latest.season)
    rates["age_adj"] = age_factor(age_next.reindex(rates.index))
    for s in HIT_RATE_STATS:
        rates[s] *= rates.age_adj

    # 5. Playing time: half of last year + a tenth of the year before + 200
    proj_pa = (PT_LAST * pa.get(LAST, 0) + PT_PREV * pa.get(LAST - 1, 0) + 200 * PT_BASE).clip(upper=720)
    if IL_PENALTY or SURGERY_PT != 1 or AGE_PT:
        proj_pa = proj_pa.reindex(rates.index) * health_factor(rates.index, age_next, HIT_WEIGHTS)

    out = rates[HIT_RATE_STATS].mul(proj_pa.reindex(rates.index), axis=0)
    out["PA"] = proj_pa.reindex(rates.index)

    # 6. Team context: runs and RBI depend on who's hitting around you.
    # Rate the same hitters two ways: on last season's teams and on current
    # rosters. The difference is purely roster moves (trades, signings, losses).
    actual_rg, roster = team_context()
    recent = out[(pa.get(LAST, 0) + pa.get(LAST - 1, 0)).reindex(out.index) > 0]
    last_team = h[h.season == LAST].set_index("player_id").team_id
    old_rg = project_team_offense(recent, last_team, actual_rg.mean(), rules)
    new_rg = project_team_offense(recent, roster.team_id, actual_rg.mean(), rules)
    out["team_adj"] = team_factor(roster.team_id.reindex(out.index).map(new_rg),
                                  last_team.reindex(out.index).map(old_rg), 0.9, 1.1)
    out["runs"] *= out.team_adj
    out["rbi"] *= out.team_adj

    out["points"] = score(out, rules["batting"])
    out["pts_per_PA"] = (out.points / out.PA).round(3)

    # Context columns that help you judge the projection
    lastyr = h[h.season == LAST].set_index("player_id")
    out[f"pts_{LAST}"] = score(lastyr, rules["batting"]).reindex(out.index)
    out["change"] = (out.points - out[f"pts_{LAST}"]).round(1)
    out["xwOBA_minus_wOBA"] = ((sav.e - sav.a) / sav.w).round(3).reindex(out.index)
    out["savant_adj"] = rates.savant_adj.round(3)
    out["age"] = age_next
    out["name"] = latest.name
    out["team"] = roster.team.reindex(out.index).fillna("FA")   # current team, FA if unsigned
    out["pos"] = latest.pos
    out["team_adj"] = out.team_adj.round(3)

    # Only players who played in one of the last two seasons
    out = out[(pa.get(LAST, 0) + pa.get(LAST - 1, 0)).reindex(out.index) > 0]
    out = out.sort_values("points", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    for s in HIT_RATE_STATS + ["PA"]:
        out[s] = out[s].round(0).astype(int)
    cols = ["rank", "name", "team", "pos", "age", "points", f"pts_{LAST}", "change", "pts_per_PA",
            "PA", "runs", "homeRuns", "rbi", "stolenBases", "baseOnBalls", "singles", "doubles",
            "triples", "hitByPitch", "xwOBA_minus_wOBA", "savant_adj", "team_adj"]
    team_table = pd.DataFrame({"proj_R_per_G": new_rg,
                               "old_roster_R_per_G": old_rg.reindex(new_rg.index),
                               f"actual_R_per_G_{LAST}": actual_rg.reindex(new_rg.index)}).round(2)
    team_table.index = team_table.index.astype(int)
    return out[cols], team_table


def project_pitchers(rules, new_rg, old_rg):
    p = load("pitching")
    p = p[p.pos.isin(["P", "TWP"]) & (p.season <= LAST)]  # no mop-up position players, no future
    cols = PIT_RATE_STATS + ["battersFaced"]

    # 1. Weighted baseline
    tot = weighted_sum(p, cols, PIT_WEIGHTS)

    # 2. Regress to league average per batter faced
    lg = p[p.season == LAST]
    lg_rate = lg[PIT_RATE_STATS].sum() / lg.battersFaced.sum()
    rates = pd.DataFrame(index=tot.index)
    for s in PIT_RATE_STATS:
        reg = PIT_REGRESS_BF * PIT_STAT_SPEED[s] ** PIT_STAT_STRENGTH
        rates[s] = (tot[s] + lg_rate[s] * reg) / (tot.battersFaced + reg)

    # 3. Savant: earned runs move toward xERA, hits toward expected batting avg
    sav_raw = load("savant_pitcher")
    f_er, sav = savant_factor(sav_raw, PIT_WEIGHTS, "era", "xera", SAVANT_TRUST, 0.75, 1.25)
    f_h, _ = savant_factor(sav_raw, PIT_WEIGHTS, "ba", "est_ba", SAVANT_TRUST, 0.85, 1.15)
    rates["era_adj"] = f_er.reindex(rates.index).fillna(1.0)
    rates["earnedRuns"] *= rates.era_adj
    rates["hits"] *= f_h.reindex(rates.index).fillna(1.0)

    latest = p.sort_values("season").groupby("player_id").last()
    piv = lambda col: p.pivot_table(index="player_id", columns="season", values=col,
                                    aggfunc="sum").fillna(0)
    bf, g, gs, w, sv = (piv(c) for c in
                        ["battersFaced", "gamesPitched", "gamesStarted", "wins", "saves"])

    # 4. Age: strikeouts follow the age curve, bad stats go the other way
    age_next = latest.age + (NEXT - latest.season)
    af = pd.Series(age_factor(age_next.reindex(rates.index)), index=rates.index)
    rates["strikeOuts"] *= af
    for s in ["hits", "earnedRuns", "baseOnBalls"]:
        rates[s] /= af

    # Role: starter if at least half of last season's games were starts
    role = pd.Series(np.where(latest.gamesStarted >= latest.gamesPitched / 2, "SP", "RP"),
                     index=latest.index).reindex(rates.index)

    # 5. Playing time (batters faced)
    base = np.where(role == "SP", 250, 105) * PT_BASE   # about 60 IP for starters, 25 for relievers
    cap = np.where(role == "SP", 850, 300)
    proj_bf = (PT_LAST * bf.get(LAST, 0).reindex(rates.index) + PT_PREV * bf.get(LAST - 1, 0).reindex(rates.index)
               + base).clip(upper=cap)
    if IL_PENALTY or SURGERY_PT != 1 or AGE_PT:
        proj_bf = proj_bf * health_factor(rates.index, age_next, PIT_WEIGHTS)

    out = rates[PIT_RATE_STATS].mul(proj_bf, axis=0)
    out["BF"] = proj_bf
    out["role"] = role

    # Wins: starters get wins per start, regressed toward league average
    sp_lg = lg[lg.gamesStarted >= lg.gamesPitched / 2]
    lg_w_per_gs = sp_lg.wins.sum() / sp_lg.gamesStarted.sum()
    wtd = lambda df: sum(df.get(y, 0) * k for y, k in PIT_WEIGHTS.items())
    w_rate_sp = (wtd(w) + lg_w_per_gs * 30) / (wtd(gs) + 30)
    w_rate_rp = (wtd(w) + 0.04 * 60) / (wtd(g) + 60)
    proj_gs = (proj_bf / 24).where(role == "SP", 0)       # ~24 batters per start
    proj_g = np.where(role == "SP", proj_gs, proj_bf / 4.2)  # ~4.2 batters per relief outing
    out["G"] = proj_g
    out["wins"] = np.where(role == "SP", w_rate_sp.reindex(out.index) * proj_gs,
                           w_rate_rp.reindex(out.index) * proj_g)

    # 6. Team context: better run support -> more wins
    _, roster = team_context()
    last_team = p[p.season == LAST].set_index("player_id").team_id
    out["team_adj"] = team_factor(roster.team_id.reindex(out.index).map(new_rg),
                                  last_team.reindex(out.index).map(old_rg), 0.85, 1.15)
    out["wins"] *= out.team_adj

    # Saves: mostly about last year's role, so weight it heavily
    sv_rate = (3 * sv.get(LAST, 0) + sv.get(LAST - 1, 0)) / (3 * g.get(LAST, 0) + g.get(LAST - 1, 0) + 20)
    out["saves"] = np.where(role == "RP", sv_rate.reindex(out.index) * proj_g, 0)

    out["points"] = score(out, rules["pitching"])
    out["pts_per_G"] = (out.points / out.G).round(1)
    out["IP"] = (out.outs / 3).round(0)
    out["ERA"] = (out.earnedRuns * 27 / out.outs).round(2)

    lastyr = p[p.season == LAST].set_index("player_id")
    out[f"pts_{LAST}"] = score(lastyr, rules["pitching"]).reindex(out.index)
    out["change"] = (out.points - out[f"pts_{LAST}"]).round(1)
    out["ERA_minus_xERA"] = ((sav.a - sav.e) / sav.w).round(2).reindex(out.index)
    out["age"] = age_next
    out["name"] = latest.name
    out["team"] = roster.team.reindex(out.index).fillna("FA")
    out["team_adj"] = out.team_adj.round(3)

    out = out[(bf.get(LAST, 0) + bf.get(LAST - 1, 0)).reindex(out.index) > 0]
    out = out.sort_values("points", ascending=False)
    out.insert(0, "rank", range(1, len(out) + 1))
    for s in ["G", "IP", "strikeOuts", "wins", "saves", "BF"]:
        out[s] = out[s].round(0).astype(int)
    cols = ["rank", "name", "team", "role", "age", "points", f"pts_{LAST}", "change", "pts_per_G",
            "G", "IP", "strikeOuts", "wins", "saves", "ERA", "ERA_minus_xERA", "team_adj"]
    return out[cols]


def blend_with_pros(hitters, pitchers, rules):
    """Mix our points with the Steamer/ZiPS average. Players the pros don't
    project keep our number."""
    import pro_blend
    pro_h, pro_p = pro_blend.pro_points(load("pro_projections"), rules)
    out = []
    for df, pro in [(hitters, pro_h), (pitchers, pro_p)]:
        df = df.copy()
        df["pts_model"] = df.points
        pro = pro.reindex(df.index)
        # Pros assume more playing time, so their totals run higher. Put them on our
        # scale using the top 300 players both systems project.
        both = df.pts_model.notna() & pro.notna()
        top = df[both].pts_model.nlargest(300).index
        pro = pro * (df.pts_model[top].median() / pro[top].median())
        df["pts_pro"] = pro.round(1)
        has = df.pts_pro.notna()
        df.loc[has, "points"] = ((1 - PRO_BLEND) * df.pts_model + PRO_BLEND * df.pts_pro)[has].round(1)
        df = df.sort_values("points", ascending=False)
        df["rank"] = range(1, len(df) + 1)
        out.append(df)
    return out


def main():
    with open("scoring.json") as f:
        rules = json.load(f)
    hitters, teams = project_hitters(rules)
    pitchers = project_pitchers(rules, teams.proj_R_per_G, teams.old_roster_R_per_G)
    if PRO_BLEND:
        hitters, pitchers = blend_with_pros(hitters, pitchers, rules)
    hitters.to_csv(f"projections_{NEXT}_hitters.csv", index_label="player_id")
    pitchers.to_csv(f"projections_{NEXT}_pitchers.csv", index_label="player_id")

    # Team offense table: who got better or worse
    names = load("rosters").drop_duplicates("team_id").set_index("team_id").team
    teams.insert(0, "team", names.reindex(teams.index))
    teams["roster_change"] = (teams.proj_R_per_G - teams.old_roster_R_per_G).round(2)
    teams = teams.sort_values("proj_R_per_G", ascending=False)
    teams.to_csv(f"projections_{NEXT}_team_offense.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nTop 20 hitters, {NEXT} projection")
    print(hitters.head(20)[["rank", "name", "pos", "age", "points", f"pts_{LAST}", "PA", "savant_adj"]].to_string(index=False))
    print(f"\nTop 15 SP")
    print(pitchers[pitchers.role == "SP"].head(15)[["rank", "name", "age", "points", f"pts_{LAST}", "IP", "ERA"]].to_string(index=False))
    print(f"\nTop 10 RP")
    print(pitchers[pitchers.role == "RP"].head(10)[["rank", "name", "points", f"pts_{LAST}", "saves"]].to_string(index=False))
    print(f"\nTeam offense, projected runs per game")
    print(teams.to_string(index=False))


if __name__ == "__main__":
    main()
