"""
CBB Edge Finder proxy (v2)

Sits between the site and two data providers so API keys stay off the browser.
  - CollegeBasketballData (CBBD): ratings, games, lines, teams
  - The Odds API: current spreads and totals

Design goals:
  - No code edits needed between seasons (season is computed from the date).
  - Clear errors: a failed call says which provider failed and why.
  - Fewer quota hits: responses are cached in memory.
  - One-click check: GET /api/diagnose tests both keys and reports plain-English status.

Environment variables (set in Render):
  CBBD_API_KEY      required
  ODDS_API_KEY      required
  ALLOWED_ORIGINS   optional, comma-separated (e.g. https://thespreadsheet.com). Default "*".
"""

import os
import time
import threading
from datetime import datetime, date, timedelta

import requests
from flask import Flask, request, jsonify

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET = None

app = Flask(__name__)

CBBD_API_KEY = os.environ.get("CBBD_API_KEY", "").strip()
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

CBBD_BASE = "https://api.collegebasketballdata.com"
ODDS_BASE = "https://api.the-odds-api.com/v4"
TIMEOUT = 30

session = requests.Session()
STARTED_AT = datetime.utcnow().isoformat() + "Z"


# ============================================================
# Time / season helpers
# ============================================================

def today_et() -> date:
    return datetime.now(ET).date() if ET else date.today()


def current_season() -> int:
    """CBBD seasons are named by the ending year. Oct 2026 - Apr 2027 = 2027."""
    t = today_et()
    return t.year + 1 if t.month >= 10 else t.year


def season_window(season: int):
    """Approximate first and last game dates for a season."""
    return date(season - 1, 11, 1), date(season, 4, 15)


def parse_season(raw):
    if raw in (None, "", "current"):
        return current_season()
    try:
        s = int(raw)
    except ValueError:
        raise BadRequest(f"season must be a 4-digit year or 'current', got '{raw}'")
    if not 2002 <= s <= current_season() + 1:
        raise BadRequest(f"season {s} is out of range")
    return s


def parse_date(raw, field):
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError):
        raise BadRequest(f"{field} must be YYYY-MM-DD, got '{raw}'")


# ============================================================
# Errors
# ============================================================

class BadRequest(Exception):
    pass


class UpstreamError(Exception):
    def __init__(self, provider, status, message):
        super().__init__(message)
        self.provider = provider
        self.status = status
        self.message = message


def explain(provider, status, body_text):
    key_name = "CBBD_API_KEY" if provider == "CBBD" else "ODDS_API_KEY"
    if status in (401, 403):
        return f"{provider} rejected the API key (invalid, expired, or tier does not allow this endpoint). Check {key_name} in Render."
    if status == 429:
        return f"{provider} quota or rate limit reached. Wait, or upgrade the plan."
    if status == 404:
        return f"{provider} endpoint not found. The provider may have changed its API."
    if status >= 500:
        return f"{provider} is having problems (HTTP {status}). Try again later."
    return f"{provider} returned HTTP {status}: {body_text[:200]}"


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.errorhandler(UpstreamError)
def handle_upstream(e):
    # Pass auth and quota codes through so the browser shows a meaningful status.
    code = e.status if e.status in (401, 403, 429) else 502
    return jsonify({"error": e.message, "provider": e.provider, "upstreamStatus": e.status}), code


# ============================================================
# Cache + usage
# ============================================================

_cache = {}
_cache_lock = threading.Lock()
usage = {"cbbdCalls": 0, "oddsCalls": 0, "cacheHits": 0,
         "oddsRemaining": None, "oddsUsed": None, "oddsCheckedAt": None}


def cached(key, ttl, fetch):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            usage["cacheHits"] += 1
            return hit[1]
    data = fetch()
    with _cache_lock:
        _cache[key] = (now + ttl, data)
    return data


# ============================================================
# Provider calls
# ============================================================

def cbbd_get(path, params=None):
    if not CBBD_API_KEY:
        raise UpstreamError("CBBD", 0, "CBBD_API_KEY is not set in Render.")
    try:
        r = session.get(f"{CBBD_BASE}{path}", params=params or {}, timeout=TIMEOUT,
                        headers={"Authorization": f"Bearer {CBBD_API_KEY}", "Accept": "application/json"})
    except requests.RequestException as e:
        raise UpstreamError("CBBD", 0, f"Could not reach CBBD: {e.__class__.__name__}")
    usage["cbbdCalls"] += 1
    if r.status_code != 200:
        raise UpstreamError("CBBD", r.status_code, explain("CBBD", r.status_code, r.text))
    try:
        return r.json()
    except ValueError:
        raise UpstreamError("CBBD", r.status_code, "CBBD returned a non-JSON response.")


def odds_get(path, params=None):
    if not ODDS_API_KEY:
        raise UpstreamError("Odds API", 0, "ODDS_API_KEY is not set in Render.")
    p = dict(params or {})
    p["apiKey"] = ODDS_API_KEY
    try:
        r = session.get(f"{ODDS_BASE}{path}", params=p, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise UpstreamError("Odds API", 0, f"Could not reach The Odds API: {e.__class__.__name__}")
    usage["oddsCalls"] += 1
    # Real quota numbers come from the provider's headers, not a local counter.
    if "x-requests-remaining" in r.headers:
        usage["oddsRemaining"] = r.headers.get("x-requests-remaining")
        usage["oddsUsed"] = r.headers.get("x-requests-used")
        usage["oddsCheckedAt"] = datetime.utcnow().isoformat() + "Z"
    if r.status_code != 200:
        raise UpstreamError("Odds API", r.status_code, explain("Odds API", r.status_code, r.text))
    return r.json()


def month_chunks(start: date, end: date):
    """Split a date range into calendar-month pieces (keeps each CBBD response small)."""
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield cur, min(end, nxt - timedelta(days=1))
        cur = nxt


def fetch_ranged(path, season, start, end, ttl_label):
    """Fetch a date-ranged CBBD endpoint month by month, caching finished months for a day."""
    today = today_et()
    out, seen = [], set()
    for a, b in month_chunks(start, end):
        ttl = 86400 if b < today - timedelta(days=2) else 900
        key = f"{ttl_label}:{season}:{a}:{b}"
        chunk = cached(key, ttl, lambda a=a, b=b: cbbd_get(path, {
            "season": season,
            "startDateRange": a.isoformat(),
            "endDateRange": b.isoformat(),
        }))
        for row in chunk or []:
            rid = row.get("id") if isinstance(row, dict) else None
            if rid is not None:
                if rid in seen:
                    continue
                seen.add(rid)
            out.append(row)
    return out


def default_range(season):
    s_start, s_end = season_window(season)
    return s_start, min(today_et(), s_end)


# ============================================================
# CORS (no extra dependency)
# ============================================================

@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin")
    if "*" in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, ngrok-skip-browser-warning"
    resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return resp


# ============================================================
# Routes
# ============================================================

@app.route("/")
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "CBB Edge Finder Proxy v2",
        "season": current_season(),
        "today": today_et().isoformat(),
        "keys": {"cbbd": bool(CBBD_API_KEY), "odds": bool(ODDS_API_KEY)},
        "startedAt": STARTED_AT,
    })


@app.route("/api/usage")
def get_usage():
    return jsonify({**usage, "note": "Call counts reset when the server restarts. oddsRemaining comes from The Odds API itself."})


@app.route("/api/ratings")
def get_ratings():
    """Adjusted efficiency ratings. If the requested season has none yet (preseason), falls back one season."""
    season = parse_season(request.args.get("season"))
    data = cached(f"ratings:{season}", 6 * 3600, lambda: cbbd_get("/ratings/adjusted", {"season": season}))
    used = season
    if not data and request.args.get("fallback", "1") == "1":
        used = season - 1
        data = cached(f"ratings:{used}", 6 * 3600, lambda: cbbd_get("/ratings/adjusted", {"season": used}))
    resp = jsonify(data)
    resp.headers["X-Season"] = str(used)
    resp.headers["Access-Control-Expose-Headers"] = "X-Season"
    return resp


@app.route("/api/games")
def get_games():
    """Games for a season. With no dates, returns the whole season to date (fixes stale 'pending' picks)."""
    season = parse_season(request.args.get("season"))
    d_start, d_end = default_range(season)
    start = parse_date(request.args["startDate"], "startDate") if request.args.get("startDate") else d_start
    end = parse_date(request.args["endDate"], "endDate") if request.args.get("endDate") else d_end
    if start > end:
        raise BadRequest("startDate is after endDate")
    return jsonify(fetch_ranged("/games", season, start, end, "games"))


@app.route("/api/lines")
def get_lines():
    """Historical/closing betting lines from CBBD (for grading, CLV, and backtests)."""
    season = parse_season(request.args.get("season"))
    d_start, d_end = default_range(season)
    start = parse_date(request.args["startDate"], "startDate") if request.args.get("startDate") else d_start
    end = parse_date(request.args["endDate"], "endDate") if request.args.get("endDate") else d_end
    if start > end:
        raise BadRequest("startDate is after endDate")
    return jsonify(fetch_ranged("/lines", season, start, end, "lines"))


@app.route("/api/teams")
def get_teams():
    season = parse_season(request.args.get("season"))
    return jsonify(cached(f"teams:{season}", 86400, lambda: cbbd_get("/teams", {"season": season})))


@app.route("/api/odds")
def get_odds():
    """Current NCAAB spreads and totals. Cached 5 minutes. Costs 2 Odds API credits per uncached call."""
    data = cached("odds", 300, lambda: odds_get("/sports/basketball_ncaab/odds/", {
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
    }))
    return jsonify(data)


# ============================================================
# Polymarket (read-only market data; free, no key)
# ============================================================

GAMMA_BASE = "https://gamma-api.polymarket.com"
POLY_TRADE_BASE = os.environ.get("POLYMARKET_TRADE_URL", "https://polymarket.us")


def gamma_get(path, params=None):
    try:
        r = session.get(f"{GAMMA_BASE}{path}", params=params or {}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise UpstreamError("Polymarket", 0, f"Could not reach Polymarket: {e.__class__.__name__}")
    usage["polymarketCalls"] = usage.get("polymarketCalls", 0) + 1
    if r.status_code != 200:
        raise UpstreamError("Polymarket", r.status_code, explain("Polymarket", r.status_code, r.text))
    return r.json()


def _jsonish(v):
    """Gamma returns some arrays as JSON strings, e.g. '["Yes","No"]'."""
    if isinstance(v, str):
        try:
            import json
            return json.loads(v)
        except ValueError:
            return [v]
    return v or []


def _american(p):
    if not p or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def ncaab_sport_entry():
    def fetch():
        sports = gamma_get("/sports")
        return next((s for s in sports if s.get("sport") in ("ncaab", "cbb")), {}), sports
    return cached("poly:sports", 86400, fetch)


def ncaab_tag_ids():
    """Find tags unique to NCAAB (drops generic tags shared with other sports)."""
    def fetch():
        _, sports = ncaab_sport_entry()
        by_sport = {s.get("sport"): {t for t in str(s.get("tags", "")).split(",") if t} for s in sports}
        mine = by_sport.get("ncaab") or by_sport.get("cbb") or set()
        others = set().union(*[v for k, v in by_sport.items() if k not in ("ncaab", "cbb")]) if by_sport else set()
        specific = mine - others
        return sorted(specific or mine)
    return cached("poly:ncaab_tags", 86400, fetch)


def market_type(m):
    t = m.get("sportsMarketType") or (m.get("sports") or {}).get("sportsMarketType")
    if t:
        return t
    q = f"{m.get('question') or ''} {m.get('slug') or ''}".lower()
    if "spread" in q:
        return "spreads"
    if "o/u" in q or "total" in q or "over/under" in q:
        return "totals"
    names = [str(n).lower() for n in _jsonish(m.get("outcomes"))]
    if len(names) == 2 and set(names) != {"yes", "no"}:
        return "moneyline"
    return "other"


def normalize_market(m):
    names = _jsonish(m.get("outcomes"))
    prices = _jsonish(m.get("outcomePrices"))
    outcomes = []
    for i, name in enumerate(names):
        try:
            p = float(prices[i])
        except (IndexError, TypeError, ValueError):
            p = None
        outcomes.append({"name": name, "price": p, "americanOdds": _american(p),
                         "breakevenPct": round(p * 100, 1) if p else None})
    return {
        "type": market_type(m),   # moneyline | spreads | totals | ...
        "line": m.get("line"),
        "question": m.get("question"),
        "slug": m.get("slug"),
        "outcomes": outcomes,
        "liquidity": m.get("liquidityNum", m.get("liquidity")),
        "volume": m.get("volumeNum", m.get("volume")),
        "acceptingOrders": m.get("acceptingOrders"),
    }


def _event_start(e):
    return e.get("startTime") or e.get("gameStartTime") or e.get("startDate") or e.get("endDate")


def fetch_poly_events():
    """Open NCAAB events, found by NCAAB tag(s) and, if Polymarket lists one, the NCAAB series."""
    entry, _ = ncaab_sport_entry()
    queries = [{"tag_id": t} for t in ncaab_tag_ids()]
    if entry.get("series"):
        queries.append({"series_id": entry["series"]})
    events = []
    for q in queries:
        cursor = None
        for _ in range(20):  # page cap
            params = {**q, "closed": "false", "limit": 100}
            if cursor:
                params["after_cursor"] = cursor
            page = gamma_get("/events/keyset", params)
            batch = page.get("events", []) if isinstance(page, dict) else page
            events.extend(batch)
            cursor = page.get("next_cursor") if isinstance(page, dict) else None
            if not cursor or not batch:
                break
    return events


def fetch_poly_ncaab():
    events = fetch_poly_events()
    # Some listings omit nested markets: fetch full detail for games in the next 4 days (capped).
    soon = datetime.utcnow() + timedelta(days=4)
    detail_budget = 60
    for i, e in enumerate(events):
        if e.get("markets") or detail_budget <= 0 or not e.get("slug"):
            continue
        st = _event_start(e)
        try:
            if st and datetime.fromisoformat(str(st).replace("Z", "+00:00")).replace(tzinfo=None) > soon:
                continue
        except ValueError:
            pass
        detail_budget -= 1
        try:
            events[i] = cached(f"poly:event:{e['slug']}", 300, lambda s=e["slug"]: gamma_get(f"/events/slug/{s}"))
        except UpstreamError:
            pass
    out, seen = [], set()
    for e in events:
        if e.get("id") in seen:
            continue
        seen.add(e.get("id"))
        teams = e.get("teams") or []
        home = next((t.get("name") for t in teams if t.get("ordering") == "home"), None)
        away = next((t.get("name") for t in teams if t.get("ordering") == "away"), None)
        markets = [normalize_market(m) for m in (e.get("markets") or [])]
        markets = [m for m in markets if m["type"] in ("moneyline", "spreads", "totals")]
        if not markets:
            continue
        out.append({
            "eventSlug": e.get("slug"),
            "title": e.get("title"),
            "homeTeam": home,
            "awayTeam": away,
            "startTime": _event_start(e),
            "markets": markets,
            "dataUrl": f"https://polymarket.com/event/{e.get('slug')}",
            "tradeUrl": POLY_TRADE_BASE,
        })
    return out


@app.route("/api/polymarket/debug")
def polymarket_debug():
    """Shows what Polymarket returned, to troubleshoot matching. Not used by the site."""
    entry, _ = ncaab_sport_entry()
    events = fetch_poly_events()
    sample = events[0] if events else {}
    m0 = (sample.get("markets") or [{}])[0] if sample else {}
    return jsonify({
        "sportEntry": entry,
        "tags": ncaab_tag_ids(),
        "eventsFound": len(events),
        "eventsWithMarkets": sum(1 for e in events if e.get("markets")),
        "games": len(cached("poly:ncaab", 300, fetch_poly_ncaab)),
        "sampleEventKeys": sorted(sample.keys()) if sample else [],
        "sampleEventTitle": sample.get("title"),
        "sampleMarketKeys": sorted(m0.keys()) if m0 else [],
        "sampleMarket": {k: m0.get(k) for k in ("question", "sportsMarketType", "line", "outcomes", "outcomePrices")} if m0 else {},
    })


@app.route("/api/polymarket")
def get_polymarket():
    """Open NCAAB game markets from Polymarket: moneyline, spreads, totals with prices. Cached 5 minutes."""
    return jsonify(cached("poly:ncaab", 300, fetch_poly_ncaab))


@app.route("/api/diagnose")
def diagnose():
    """Tests each connection without spending Odds API credits. Open this in a browser when refresh fails."""
    season = current_season()
    report = {"season": season, "today": today_et().isoformat(), "checks": []}

    def check(name, fn):
        try:
            detail = fn()
            report["checks"].append({"name": name, "ok": True, "detail": detail})
        except UpstreamError as e:
            report["checks"].append({"name": name, "ok": False, "detail": e.message, "upstreamStatus": e.status})
        except Exception as e:  # pragma: no cover
            report["checks"].append({"name": name, "ok": False, "detail": f"{e.__class__.__name__}: {e}"})

    def ratings_check():
        rows = cbbd_get("/ratings/adjusted", {"season": season})
        if not rows:
            prev = cbbd_get("/ratings/adjusted", {"season": season - 1})
            return f"No {season} ratings published yet; {len(prev)} teams available for {season - 1} (site will fall back)."
        return f"{len(rows)} teams for {season}."

    def games_check():
        t = today_et()
        rows = cbbd_get("/games", {"season": season,
                                    "startDateRange": (t - timedelta(days=3)).isoformat(),
                                    "endDateRange": t.isoformat()})
        return f"{len(rows)} games in the last 3 days."

    def odds_check():
        # /sports does not count against the Odds API quota.
        sports = odds_get("/sports", {})
        active = any(s.get("key") == "basketball_ncaab" and s.get("active") for s in sports)
        rem = usage.get("oddsRemaining")
        return f"Key OK. NCAAB market {'active' if active else 'not active right now'}. Credits remaining: {rem if rem is not None else 'unknown'}."

    def poly_check():
        tags = ncaab_tag_ids()
        games = cached("poly:ncaab", 300, fetch_poly_ncaab)
        return f"Free data OK. NCAAB tags {tags}. {len(games)} open games with markets."

    check("CBBD ratings", ratings_check)
    check("CBBD games", games_check)
    check("Odds API", odds_check)
    check("Polymarket", poly_check)
    report["allOk"] = all(c["ok"] for c in report["checks"])
    return jsonify(report), (200 if report["allOk"] else 503)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"CBB Edge Finder Proxy v2 on port {port} | season {current_season()} | "
          f"CBBD key {'set' if CBBD_API_KEY else 'MISSING'} | Odds key {'set' if ODDS_API_KEY else 'MISSING'}")
    app.run(host="0.0.0.0", port=port)
