# CBB Edge Finder Proxy v2

Keeps API keys off the browser and serves CollegeBasketballData (CBBD) and The Odds API data to The Spread Sheet.

## Each season
Nothing to change. The season is computed from the date (Oct–Apr = the ending year, e.g. 2026-27 = 2027).

## When refresh fails
Open `https://<your-proxy>.onrender.com/api/diagnose` in a browser. Each check says what is wrong:

| Message | Fix |
|---|---|
| rejected the API key | Get a new key (CBBD: collegebasketballdata.com/key) and update the env var in Render |
| quota or rate limit | Wait, or upgrade the provider plan |
| is not set in Render | Add the env var in Render |
| endpoint not found | Provider changed its API; update the path in `app.py` |

## Endpoints
| Path | Purpose | Cache |
|---|---|---|
| `/health` | Status, current season, whether keys are set | none |
| `/api/diagnose` | Tests every connection (spends no Odds credits) | none |
| `/api/ratings?season=` | Adjusted efficiency ratings; falls back one season if none published yet | 6 h |
| `/api/games?season=&startDate=&endDate=` | Games; no dates = whole season to date | 15 min current month, 24 h past months |
| `/api/lines?season=&startDate=&endDate=` | CBBD historical lines (grading, CLV, backtests) | same as games |
| `/api/teams?season=` | Team list | 24 h |
| `/api/odds` | Current spreads and totals (2 Odds credits per uncached call) | 5 min |
| `/api/usage` | Call counts since restart + real Odds credits remaining | none |

`season` accepts a year or `current`; omitted = current.

## Environment variables
- `CBBD_API_KEY` (required)
- `ODDS_API_KEY` (required)
- `ALLOWED_ORIGINS` (optional, comma-separated site URLs; default allows any origin)

## Render start command
`gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`

## Polymarket
`/api/polymarket` returns open NCAAB game markets (moneyline, spreads, totals) with each side's price,
American-odds equivalent, and breakeven %. Data comes from Polymarket's free public API (no key).
If the site shows no Polymarket prices, open `/api/polymarket/debug` and send the output for troubleshooting.
`tradeUrl` points to Polymarket US (the regulated US exchange). Override with env var `POLYMARKET_TRADE_URL`.

## Value finder
`/api/value?minEdge=0.03` compares each Polymarket price (the ask, i.e. cost to buy) with the sportsbooks'
consensus no-vig probability for the same bet at the same line (2+ books required). Edge = fair probability − price.
Odds now include moneylines, so each uncached `/api/odds` call costs 3 Odds API credits (cached 5 minutes).

## Backtest
`/api/backtest` runs a leak-free walk-forward test of a ridge-ratings model on last season (runs in the background;
refresh to see progress). 2025-26 result: 50.3% vs closing spreads, 50.2% on totals, so model picks are shown as unproven "leans".
