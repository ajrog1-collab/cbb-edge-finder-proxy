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
import re
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


@app.errorhandler(Exception)
def handle_unexpected(e):
    """Any unexpected crash returns a readable JSON message instead of a blank 'Internal Server Error'."""
    import traceback
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    tb = traceback.extract_tb(e.__traceback__)
    where = [f"{f.filename.split('/')[-1]}:{f.lineno} in {f.name}" for f in tb[-3:]]
    app.logger.exception("Unhandled error")
    return jsonify({"error": f"{e.__class__.__name__}: {e}", "where": where}), 500


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
    """Current NCAAB moneylines, spreads and totals. Cached 5 minutes. Costs 3 Odds API credits per uncached call."""
    data = cached("odds", 300, lambda: odds_get("/sports/basketball_ncaab/odds/", {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }))
    return jsonify(data)


# ============================================================
# Polymarket (read-only market data; free, no key)
# ============================================================

GAMMA_BASE = "https://gamma-api.polymarket.com"
# Polymarket US game pages look like https://polymarket.us/sports/cbb/cbb-marq-vir-2026-11-25
POLY_TRADE_BASE = os.environ.get("POLYMARKET_TRADE_URL", "https://polymarket.us/sports/cbb").rstrip("/")
GAME_SLUG_RE = re.compile(r"^(cbb-[a-z0-9]+-[a-z0-9]+-\d{4}-\d{2}-\d{2})")
US_GATEWAY = "https://gateway.polymarket.us/v1/events/slug/"


def us_slug_exists(slug):
    """True/False from Polymarket US's public gateway; None if it could not be checked. Cached."""
    key = f"usslug:{slug}"
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    try:
        r = session.get(US_GATEWAY + slug, timeout=10, headers={"User-Agent": "cbb-edge-finder-proxy"})
    except requests.RequestException:
        return None
    usage["polymarketUsCalls"] = usage.get("polymarketUsCalls", 0) + 1
    if r.status_code == 200:
        found, ttl = True, 12 * 3600
    elif r.status_code == 404:
        found, ttl = False, 3 * 3600
    else:
        return None   # rate limited or down: don't cache, try again later
    with _cache_lock:
        _cache[key] = (now + ttl, found)
    return found


def us_trade_url(base_slug):
    """Polymarket US sometimes lists a game with the teams in the other order (cbb-smu-utahst vs
    cbb-utahst-smu) or under the next/previous date. Check the real slug on Polymarket US."""
    m = re.match(r"^cbb-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$", base_slug or "")
    if not m:
        return POLY_TRADE_BASE
    a, b, d = m.groups()
    day = date.fromisoformat(d)
    dates = [day, day - timedelta(days=1), day + timedelta(days=1)]
    candidates = []
    for dd in dates:
        candidates += [f"cbb-{a}-{b}-{dd.isoformat()}", f"cbb-{b}-{a}-{dd.isoformat()}"]
    unknown = False
    for c in candidates:
        found = us_slug_exists(c)
        if found:
            return f"{POLY_TRADE_BASE}/{c}"
        if found is None:
            unknown = True
            break
    # Gateway unreachable: best guess is the international slug. Confirmed missing: the CBB page.
    return f"{POLY_TRADE_BASE}/{base_slug}" if unknown else POLY_TRADE_BASE


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


CBB_SPORT_KEYS = ("cbb", "ncaab", "ncaam", "mbb", "cbk")


def _is_cbb(s):
    key = str(s.get("sport", "")).lower()
    name = str(s.get("name", "")).lower()
    if "women" in name or key.startswith("w"):
        return False
    return key in CBB_SPORT_KEYS or "college basketball" in name or "ncaa basketball" in name


def cbb_sport_entries():
    """All men's college basketball entries. 'ncaab' is March Madness futures; regular-season games live under 'cbb'."""
    def fetch():
        sports = gamma_get("/sports")
        return [s for s in sports if _is_cbb(s)], sports
    return cached("poly:sports", 86400, fetch)


def ncaab_sport_entry():
    entries, sports = cbb_sport_entries()
    return (entries[0] if entries else {}), sports


def ncaab_tag_ids():
    """Find tags unique to NCAAB (drops generic tags shared with other sports)."""
    def fetch():
        entries, sports = cbb_sport_entries()
        tagset = lambda e: {t for t in str(e.get("tags", "")).split(",") if t}
        mine = set().union(*[tagset(e) for e in entries]) if entries else set()
        others = set().union(*[tagset(e) for e in sports if not _is_cbb(e)]) if sports else set()
        specific = mine - others
        return sorted(specific or mine)
    return cached("poly:ncaab_tags", 86400, fetch)


def _fnum(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower().replace("state", "st"))


def _name_score(a, b):
    """How well two team names match (0 = no match). Handles 'Virginia' vs 'Virginia Cavaliers'."""
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return 0
    if x == y:
        return 100
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    return 50 + len(short) if long_.startswith(short) and len(short) >= 4 else 0


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
    # Cost to buy each side: first outcome = best ask; second = 1 - best bid (order book is shared)
    ask0, bid0 = _fnum(m.get("bestAsk")), _fnum(m.get("bestBid"))
    asks = [ask0, (1 - bid0) if bid0 is not None else None]
    outcomes = []
    for i, name in enumerate(names):
        try:
            p = float(prices[i])
        except (IndexError, TypeError, ValueError):
            p = None
        ask = asks[i] if i < 2 and asks[i] is not None and 0 < asks[i] < 1 else None
        outcomes.append({"name": name, "price": p, "ask": ask, "americanOdds": _american(p),
                         "breakevenPct": round(p * 100, 1) if p else None})
    # Spread line for each side, from a question like "Spread: Virginia (-2.5)"
    if market_type(m) == "spreads":
        mq = re.search(r"(?:spread:\s*)?(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)", str(m.get("question") or ""), re.I)
        if mq and len(outcomes) == 2:
            team, val = mq.group(1).strip(), float(mq.group(2))
            first_is_team = _name_score(outcomes[0]["name"], team) >= _name_score(outcomes[1]["name"], team)
            outcomes[0]["line"] = val if first_is_team else -val
            outcomes[1]["line"] = -val if first_is_team else val
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
    entries, _ = cbb_sport_entries()
    queries = [{"tag_id": t} for t in ncaab_tag_ids()]
    for e in entries:
        if e.get("series"):
            queries.append({"series_id": e["series"]})
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
    games, seen = {}, set()
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
        slug = e.get("slug") or ""
        m = GAME_SLUG_RE.match(slug)
        base = m.group(1) if m else slug
        g = games.get(base)
        if g:
            g["markets"].extend(markets)
            g["homeTeam"] = g["homeTeam"] or home
            g["awayTeam"] = g["awayTeam"] or away
            if slug == base:
                g["title"] = e.get("title") or g["title"]
            continue
        games[base] = {
            "eventSlug": base,
            "title": e.get("title"),
            "homeTeam": home,
            "awayTeam": away,
            "startTime": _event_start(e),
            "markets": markets,
            "dataUrl": f"https://polymarket.com/event/{base}",
            "tradeUrl": None,   # filled in below from Polymarket US's own slug
        }
    for gm in games.values():
        gm["tradeUrl"] = us_trade_url(gm["eventSlug"]) if GAME_SLUG_RE.match(gm["eventSlug"] or "") else POLY_TRADE_BASE
    return list(games.values())


# ============================================================
# Value finder: Polymarket price vs sportsbook consensus
# ============================================================

def _implied(american):
    a = _fnum(american)
    if a is None or a == 0:
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def _consensus(game, market, pick_name, line=None):
    """Average no-vig probability for one side across books offering the same line. None if < 2 books."""
    probs = []
    for b in game.get("bookmakers") or []:
        mk = next((m for m in b.get("markets") or [] if m.get("key") == market), None)
        if not mk or len(mk.get("outcomes") or []) != 2:
            continue
        o = mk["outcomes"]
        i = next((k for k in (0, 1) if _norm(o[k].get("name")) == _norm(pick_name)), None)
        if i is None:
            continue
        if line is not None and _fnum(o[i].get("point")) != line:
            continue
        p1, p2 = _implied(o[i].get("price")), _implied(o[1 - i].get("price"))
        if p1 and p2:
            probs.append(p1 / (p1 + p2))
    return (sum(probs) / len(probs), len(probs)) if len(probs) >= 2 else (None, len(probs))


def _match_poly_event(game, events):
    t = _dt_any(game.get("commence_time"))
    best, best_score = None, 0
    for ev in events:
        names = [n for n in (ev.get("homeTeam"), ev.get("awayTeam")) if n]
        if len(names) < 2 and ev.get("title"):
            names = re.split(r"\s+(?:vs\.?|@|at)\s+", ev["title"], flags=re.I)
        if len(names) != 2:
            continue
        h, a = game.get("home_team"), game.get("away_team")
        score = max(min(_name_score(names[0], h), _name_score(names[1], a)),
                    min(_name_score(names[0], a), _name_score(names[1], h)))
        if score <= 0:
            continue
        st = _dt_any(ev.get("startTime"))
        if t and st and abs((st - t).total_seconds()) > 36 * 3600:
            continue
        if score > best_score:
            best, best_score = ev, score
    return best


def _dt_any(raw):
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def find_value(odds_games, poly_events, min_edge, stats=None):
    rows, compared = [], 0
    st = stats if stats is not None else {}
    bump = lambda k: st.__setitem__(k, st.get(k, 0) + 1)
    now = datetime.utcnow()
    for g in odds_games or []:
        start = _dt_any(g.get("commence_time"))
        bump("games")
        if start and start <= now:
            bump("games_started")
            continue  # started; prices are live, skip
        ev = _match_poly_event(g, poly_events)
        if not ev:
            bump("games_no_polymarket_match")
            continue
        bump("games_matched")
        home, away = g.get("home_team"), g.get("away_team")

        def team_for(name):
            sh, sa = _name_score(name, home), _name_score(name, away)
            return (home if sh >= sa else away) if max(sh, sa) > 0 else None

        for m in ev.get("markets") or []:
            for o in m.get("outcomes") or []:
                # Only real buy prices: a last-trade price can be hours old on a thin market
                bump("sides_seen")
                price = o.get("ask")
                if price is None or not (0.02 < price < 0.98):
                    bump("skip_no_buy_price")
                    continue
                if m["type"] == "moneyline":
                    team = team_for(o.get("name"))
                    if not team:
                        bump("skip_team_not_matched")
                        continue
                    fair, n = _consensus(g, "h2h", team)
                    label, side, line, kind = f"{team} to win", team, None, "moneyline"
                elif m["type"] == "spreads":
                    team, line = team_for(o.get("name")), o.get("line")
                    if not team or line is None:
                        bump("skip_team_or_line_not_matched")
                        continue
                    fair, n = _consensus(g, "spreads", team, line)
                    label, side, kind = f"{team} {line:+g}", team, "spread"
                elif m["type"] == "totals":
                    nm = str(o.get("name", "")).lower()
                    line = _fnum(m.get("line"))
                    if line is None:
                        nums = re.findall(r"\d+(?:\.\d+)?", str(m.get("question") or ""))
                        line = float(nums[-1]) if nums else None
                    if nm not in ("over", "under") or line is None:
                        continue
                    fair, n = _consensus(g, "totals", nm.capitalize(), line)
                    label, side, kind = f"{nm.capitalize()} {line:g}", nm, "total"
                else:
                    continue
                if fair is None:
                    bump(f"skip_under_2_books_at_line_{kind}")
                    continue
                edge = fair - price
                # A gap this big almost always means mismatched lines or a stale market, not a real discount
                if abs(edge) > 0.15:
                    bump("skip_gap_over_15c")
                    st.setdefault("examples_big_gap", [])
                    if len(st["examples_big_gap"]) < 6:
                        st["examples_big_gap"].append({"bet": label, "game": f"{away} @ {home}", "price": price, "worth": round(fair, 3), "polyLine": line, "question": m.get("question")})
                    continue
                compared += 1
                c = edge * 100
                bucket = "edge_below_-3c" if c < -3 else "edge_-3c_to_0" if c < 0 else "edge_0_to_1c" if c < 1 else "edge_1_to_2c" if c < 2 else "edge_2_to_3c" if c < 3 else "edge_3c_plus"
                bump(bucket)
                st.setdefault("_all", []).append((edge, label, f"{away} @ {home}", price, fair, n))
                if edge < min_edge:
                    continue
                rows.append({
                    "id": f"{g.get('id')}_{kind}_{_norm(side)}_{line if line is not None else 'ml'}",
                    "gameId": g.get("id"), "homeTeam": home, "awayTeam": away,
                    "startTime": g.get("commence_time"), "type": kind, "side": side, "line": line,
                    "label": label, "polyPrice": round(price, 3),
                    "priceType": "ask" if o.get("ask") is not None else "last",
                    "fairProb": round(fair, 3), "edge": round(edge, 3), "books": n,
                    "tradeUrl": ev.get("tradeUrl"),
                })
    rows.sort(key=lambda r: -r["edge"])
    return rows, compared


@app.route("/api/value/debug")
def value_debug():
    """Why value bets do or don't appear: counts at each step, edge distribution, closest calls."""
    odds = cached("odds", 300, lambda: odds_get("/sports/basketball_ncaab/odds/", {
        "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}))
    poly = cached("poly:ncaab", 300, fetch_poly_ncaab)
    stats = {}
    find_value(odds, poly, -1, stats)
    allc = sorted(stats.pop("_all", []), key=lambda r: -r[0])
    stats["closest_to_value"] = [{"bet": b, "game": g, "price": round(p, 3), "worth": round(f, 3),
                                  "edgeCents": round(e * 100, 2), "books": n} for e, b, g, p, f, n in allc[:12]]
    stats["polymarket_games_listed"] = len(poly)
    return jsonify(stats)


@app.route("/api/value")
def get_value():
    """Bets where Polymarket's price is below the sportsbook consensus (no-vig) probability.
    Edge = books' fair probability - Polymarket price. Uses only books posting the same line."""
    try:
        min_edge = float(request.args.get("minEdge", "0.02"))
    except ValueError:
        raise BadRequest("minEdge must be a number like 0.03")
    odds = cached("odds", 300, lambda: odds_get("/sports/basketball_ncaab/odds/", {
        "regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"}))
    poly = cached("poly:ncaab", 300, fetch_poly_ncaab)
    rows, compared = find_value(odds, poly, min_edge)
    return jsonify({"minEdge": min_edge, "compared": compared, "opportunities": rows,
                    "note": "Prices from Polymarket's public data. Confirm the price on Polymarket US before trading."})


@app.route("/api/polymarket/debug")
def polymarket_debug():
    """Shows what Polymarket returned, to troubleshoot matching. Not used by the site."""
    entries, sports = cbb_sport_entries()
    events = fetch_poly_events()
    # Prefer a game-looking event (title with "vs") as the sample
    sample = next((e for e in events if " vs" in str(e.get("title", "")).lower()), events[0] if events else {})
    m0 = (sample.get("markets") or [{}])[0] if sample else {}
    return jsonify({
        "cbbSportEntries": [{k: e.get(k) for k in ("sport", "name", "tags", "series")} for e in entries],
        "basketballLikeSports": [{k: e.get(k) for k in ("sport", "name", "series")} for e in sports
                                 if any(w in (str(e.get("sport", "")) + " " + str(e.get("name", ""))).lower()
                                        for w in ("bask", "cb", "ncaa", "college"))],
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


# Backtests take a few minutes on a small server, so they run in the background.
_backtests = {}
_backtests_lock = threading.Lock()


def _run_backtest_job(key, season, kw):
    from model import run_backtest

    def progress(msg):
        with _backtests_lock:
            _backtests[key]["step"] = msg

    try:
        s_start, s_end = season_window(season)
        end = min(today_et(), s_end)
        progress("Downloading games")
        games = fetch_ranged("/games", season, s_start, end, "games")
        progress(f"Downloading betting lines ({len(games)} games found)")
        lines = fetch_ranged("/lines", season, s_start, end, "lines")
        progress(f"Running model ({len(lines)} line records)")
        result = run_backtest(games, lines, progress=progress, **kw)
        result["season"] = season
        with _backtests_lock:
            _backtests[key].update(status="done", result=result, finishedAt=datetime.utcnow().isoformat() + "Z")
    except UpstreamError as e:
        with _backtests_lock:
            _backtests[key].update(status="error", error=e.message)
    except Exception as e:  # pragma: no cover
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        with _backtests_lock:
            _backtests[key].update(status="error", error=f"{e.__class__.__name__}: {e}",
                                   where=[f"{f.filename.split('/')[-1]}:{f.lineno} in {f.name}" for f in tb[-3:]])


@app.route("/api/backtest")
def backtest():
    """Leak-free walk-forward test of the rebuilt model on a finished season (default: last season).
    First visit starts the test; refresh the page to check progress and see results."""
    season = parse_season(request.args.get("season") or str(current_season() - 1))
    kw = {}
    for arg, name in (("lambdaSpread", "lam_spread"), ("lambdaTotal", "lam_total"), ("halfLife", "half_life")):
        if request.args.get(arg):
            try:
                kw[name] = float(request.args[arg])
            except ValueError:
                raise BadRequest(f"{arg} must be a number")
    key = f"{season}:{sorted(kw.items())}"
    restart = request.args.get("rerun") == "1"

    with _backtests_lock:
        job = _backtests.get(key)
        if job is None or restart or job["status"] == "error":
            job = {"status": "running", "step": "Starting", "season": season,
                   "startedAt": time.time()}
            _backtests[key] = job
            threading.Thread(target=_run_backtest_job, args=(key, season, kw), daemon=True).start()
        snapshot = dict(job)

    if snapshot["status"] == "done":
        return jsonify(snapshot["result"])
    elapsed = int(time.time() - snapshot["startedAt"])
    if snapshot["status"] == "error":
        return jsonify({"status": "error", "error": snapshot.get("error"), "where": snapshot.get("where"),
                        "note": "Refresh to try again."}), 500
    return jsonify({"status": "running", "step": snapshot["step"], "elapsedSeconds": elapsed,
                    "note": "Still working. Refresh this page in about 30 seconds."})


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
