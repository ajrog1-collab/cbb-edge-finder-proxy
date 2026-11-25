"""
Leak-free walk-forward backtest for a simple, standard CBB model.

Ratings come only from games played BEFORE the games being predicted, so the
results show how the model would really have done. No future data is used.

Model
  Spread: margin = HCA * home_court + rating_home - rating_away
          (ridge regression on game margins, recent games weighted more, blowouts capped)
  Total:  points = league_avg + offense_team - defense_opponent + home_pts
          (ridge regression on each team's points; total = both teams' predicted points)
  Win %:  normal distribution around the edge, using the model's own recent error size

Graded against the betting lines CBBD records for each game (closing, and opening when available).
"""

import math
from datetime import datetime, timedelta
from statistics import median

import numpy as np

BREAKEVEN = 0.5238  # -110 odds


# ------------------------------------------------------------------
# Data prep
# ------------------------------------------------------------------

def _dt(raw):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def prep_games(raw_games):
    """Keep completed games with scores. Returns list of dicts sorted by date."""
    out = []
    for g in raw_games or []:
        hp = _num(g.get("homePoints", g.get("homeScore")))
        ap = _num(g.get("awayPoints", g.get("awayScore")))
        d = _dt(g.get("startDate") or g.get("date"))
        home, away = g.get("homeTeam"), g.get("awayTeam")
        if hp is None or ap is None or d is None or not home or not away:
            continue
        if hp == 0 and ap == 0:
            continue
        out.append({"id": g.get("id"), "date": d, "home": home, "away": away,
                    "hp": hp, "ap": ap, "neutral": bool(g.get("neutralSite"))})
    out.sort(key=lambda x: x["date"])
    return out


def prep_lines(raw_lines):
    """Median spread/total across books per game. Spread is from the home team's view (home -5 = -5)."""
    by_id, by_key = {}, {}
    for row in raw_lines or []:
        books = row.get("lines") or []
        sp = [_num(b.get("spread")) for b in books]
        to = [_num(b.get("overUnder")) for b in books]
        spo = [_num(b.get("spreadOpen")) for b in books]
        too = [_num(b.get("overUnderOpen")) for b in books]
        rec = {
            "spread": median([x for x in sp if x is not None]) if any(x is not None for x in sp) else None,
            "total": median([x for x in to if x is not None]) if any(x is not None for x in to) else None,
            "spreadOpen": median([x for x in spo if x is not None]) if any(x is not None for x in spo) else None,
            "totalOpen": median([x for x in too if x is not None]) if any(x is not None for x in too) else None,
        }
        gid = row.get("gameId", row.get("id"))
        if gid is not None:
            by_id[gid] = rec
        d = _dt(row.get("startDate"))
        if d and row.get("homeTeam") and row.get("awayTeam"):
            by_key[(row["homeTeam"], row["awayTeam"], d.date())] = rec
    return by_id, by_key


# ------------------------------------------------------------------
# Ridge fits
# ------------------------------------------------------------------

def _weights(dates, as_of, half_life):
    ages = np.array([(as_of - d).days for d in dates], dtype=float)
    return np.power(0.5, ages / half_life)


def _ridge(cols, vals, y, w, reg):
    """Weighted ridge regression for sparse rows (each row has a few nonzero columns).
    Builds the normal equations directly so it stays fast on a small server."""
    k = len(reg)
    A = np.diag(reg).astype(float)
    b = np.zeros(k)
    m = cols.shape[1]
    for i in range(m):
        np.add.at(b, cols[:, i], w * vals[:, i] * y)
        for j in range(m):
            np.add.at(A, (cols[:, i], cols[:, j]), w * vals[:, i] * vals[:, j])
    beta = np.linalg.solve(A, b)
    pred = np.sum(vals * beta[cols], axis=1)
    return beta, pred


def fit_spread(train, as_of, lam=2.0, half_life=45, cap=25):
    teams = sorted({g["home"] for g in train} | {g["away"] for g in train})
    idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    cols = np.array([[idx[g["home"]], idx[g["away"]], nt] for g in train])
    vals = np.array([[1.0, -1.0, 0.0 if g["neutral"] else 1.0] for g in train])
    y = np.array([max(-cap, min(cap, g["hp"] - g["ap"])) for g in train], dtype=float)
    w = _weights([g["date"] for g in train], as_of, half_life)
    reg = np.full(nt + 1, lam)
    reg[-1] = 0.01  # barely regularize home court
    beta, pred = _ridge(cols, vals, y, w, reg)
    sd = float(np.sqrt(np.average((y - pred) ** 2, weights=w)))
    return {"r": {t: beta[i] for t, i in idx.items()}, "hca": float(beta[-1]), "sd": max(sd, 8.0)}


def fit_total(train, as_of, lam=3.0, half_life=45):
    teams = sorted({g["home"] for g in train} | {g["away"] for g in train})
    idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    icol, hcol = 2 * nt, 2 * nt + 1
    cols, vals, y, dates = [], [], [], []
    for g in train:
        h, a = idx[g["home"]], idx[g["away"]]
        home = 0.0 if g["neutral"] else 1.0
        cols += [[h, nt + a, icol, hcol], [a, nt + h, icol, hcol]]
        vals += [[1.0, -1.0, 1.0, home], [1.0, -1.0, 1.0, 0.0]]
        y += [g["hp"], g["ap"]]
        dates += [g["date"], g["date"]]
    cols, vals, y = np.array(cols), np.array(vals), np.array(y, dtype=float)
    w = _weights(dates, as_of, half_life)
    reg = np.full(2 * nt + 2, lam)
    reg[-2:] = 0.01
    beta, pred = _ridge(cols, vals, y, w, reg)
    tot_resid = (y[0::2] + y[1::2]) - (pred[0::2] + pred[1::2])
    sd = float(np.std(tot_resid)) if len(tot_resid) else 13.5
    return {"off": {t: beta[i] for t, i in idx.items()},
            "dfn": {t: beta[nt + i] for t, i in idx.items()},
            "mu": float(beta[icol]), "home": float(beta[hcol]), "sd": max(sd, 9.0)}


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def predict(spread_fit, total_fit, g):
    rh, ra = spread_fit["r"].get(g["home"]), spread_fit["r"].get(g["away"])
    if rh is None or ra is None:
        return None
    margin = rh - ra + (0 if g["neutral"] else spread_fit["hca"])
    o, d = total_fit["off"], total_fit["dfn"]
    hp = total_fit["mu"] + o.get(g["home"], 0) - d.get(g["away"], 0) + (0 if g["neutral"] else total_fit["home"])
    ap = total_fit["mu"] + o.get(g["away"], 0) - d.get(g["home"], 0)
    return {"margin": margin, "total": hp + ap, "sd_spread": spread_fit["sd"], "sd_total": total_fit["sd"]}


# ------------------------------------------------------------------
# Grading + summary
# ------------------------------------------------------------------

def _grade_spread(margin_actual, line, side_home):
    cover = margin_actual + line  # home covers if > 0
    if cover == 0:
        return "push"
    return "win" if (cover > 0) == side_home else "loss"


def _grade_total(total_actual, line, side_over):
    if total_actual == line:
        return "push"
    return "win" if (total_actual > line) == side_over else "loss"


def _summ(bets):
    wl = [b for b in bets if b["result"] in ("win", "loss")]
    n = len(wl)
    if n == 0:
        return {"bets": 0}
    w = sum(1 for b in wl if b["result"] == "win")
    p = w / n
    se = math.sqrt(p * (1 - p) / n)
    units = w * (100 / 110) - (n - w)
    return {"bets": n, "winPct": round(100 * p, 1),
            "ci95": [round(100 * (p - 1.96 * se), 1), round(100 * (p + 1.96 * se), 1)],
            "units": round(units, 1), "roiPct": round(100 * units / n, 1),
            "avgModelProb": round(100 * sum(b["prob"] for b in wl) / n, 1),
            "verdict": ("Profitable and statistically clear" if p - 1.96 * se > BREAKEVEN
                        else "Above breakeven, not yet proven" if p > BREAKEVEN
                        else "Below breakeven")}


def _by_edge(bets, cuts):
    out = []
    for c in cuts:
        s = _summ([b for b in bets if abs(b["edge"]) >= c])
        s["minEdgePts"] = c
        out.append(s)
    return out


def _calibration(bets):
    bins = [(0.50, 0.53), (0.53, 0.56), (0.56, 0.60), (0.60, 0.65), (0.65, 1.01)]
    out = []
    for lo, hi in bins:
        sel = [b for b in bets if lo <= b["prob"] < hi and b["result"] in ("win", "loss")]
        if len(sel) < 20:
            continue
        w = sum(1 for b in sel if b["result"] == "win")
        out.append({"modelSays": f"{int(lo*100)}-{min(int(hi*100), 100)}%", "bets": len(sel),
                    "actualWinPct": round(100 * w / len(sel), 1)})
    return out


def run_backtest(raw_games, raw_lines, warmup_days=21, refit_days=7, lam_spread=2.0, lam_total=3.0, half_life=45,
                 progress=None):
    games = prep_games(raw_games)
    by_id, by_key = prep_lines(raw_lines)
    if len(games) < 200:
        return {"error": f"Only {len(games)} completed games found; need 200+."}

    start = games[0]["date"] + timedelta(days=warmup_days)
    end = games[-1]["date"]
    spread_bets, total_bets = [], []
    spread_open_bets, total_open_bets = [], []
    matched = 0
    errors_vs_actual = []
    sign_checks = []  # does the line agree with the model on who is favored? (catches flipped line signs)

    cur = start
    total_weeks = max(1, (end - start).days // refit_days + 1)
    week = 0
    while cur <= end:
        week += 1
        if progress:
            progress(f"Testing week {week} of {total_weeks}")
        nxt = cur + timedelta(days=refit_days)
        train = [g for g in games if g["date"] < cur]
        test = [g for g in games if cur <= g["date"] < nxt]
        if len(train) >= 100 and test:
            sf = fit_spread(train, cur, lam=lam_spread, half_life=half_life)
            tf = fit_total(train, cur, lam=lam_total, half_life=half_life)
            for g in test:
                pr = predict(sf, tf, g)
                if not pr:
                    continue
                ln = by_id.get(g["id"]) or by_key.get((g["home"], g["away"], g["date"].date()))
                if not ln:
                    continue
                matched += 1
                actual_margin = g["hp"] - g["ap"]
                actual_total = g["hp"] + g["ap"]
                errors_vs_actual.append(abs(pr["margin"] - actual_margin))
                if ln.get("spread") is not None and abs(ln["spread"]) >= 6:
                    sign_checks.append((pr["margin"] > 0) == (ln["spread"] < 0))

                for key, store in (("spread", spread_bets), ("spreadOpen", spread_open_bets)):
                    line = ln.get(key)
                    if line is None:
                        continue
                    edge = pr["margin"] + line          # >0 means take home
                    side_home = edge > 0
                    prob = _phi(abs(edge) / pr["sd_spread"])
                    store.append({"edge": edge, "prob": prob,
                                  "result": _grade_spread(actual_margin, line, side_home)})

                for key, store in (("total", total_bets), ("totalOpen", total_open_bets)):
                    line = ln.get(key)
                    if line is None:
                        continue
                    edge = pr["total"] - line           # >0 means over
                    prob = _phi(abs(edge) / pr["sd_total"])
                    store.append({"edge": edge, "prob": prob,
                                  "result": _grade_total(actual_total, line, edge > 0),
                                  "over": edge > 0})
        cur = nxt

    cuts = [0, 1, 2, 3, 4, 5, 7]
    return {
        "settings": {"warmupDays": warmup_days, "refitDays": refit_days, "lambdaSpread": lam_spread,
                     "lambdaTotal": lam_total, "halfLifeDays": half_life},
        "data": {"completedGames": len(games), "lineRows": len(raw_lines or []), "gamesWithLines": matched,
                 "lineSignAgreementPct": round(100 * sum(sign_checks) / len(sign_checks), 1) if sign_checks else None,
                 "from": games[0]["date"].date().isoformat(), "to": end.date().isoformat(),
                 "avgMarginErrorPts": round(sum(errors_vs_actual) / len(errors_vs_actual), 2) if errors_vs_actual else None},
        "breakevenPct": round(100 * BREAKEVEN, 1),
        "spreadsVsClosing": _by_edge(spread_bets, cuts),
        "spreadsVsOpening": _by_edge(spread_open_bets, cuts),
        "totalsVsClosing": _by_edge(total_bets, cuts),
        "totalsVsOpening": _by_edge(total_open_bets, cuts),
        "totalsOverVsUnder": {"overs": _summ([b for b in total_bets if b["over"]]),
                              "unders": _summ([b for b in total_bets if not b["over"]])},
        "calibrationSpreads": _calibration(spread_bets),
        "calibrationTotals": _calibration(total_bets),
        "howToRead": "Each row only counts bets where the model disagreed with the line by at least minEdgePts. "
                     "52.4% is breakeven at -110. 'Profitable and statistically clear' means even the low end of the "
                     "95% range beats breakeven. Opening lines are easier to beat than closing lines.",
    }
