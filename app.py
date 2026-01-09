from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from datetime import datetime, date

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================
# CONFIGURATION - Set these in Render dashboard
# ============================================
CBBD_API_KEY = os.environ.get('CBBD_API_KEY', '')
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')

# API Base URLs
CBBD_API_BASE = "https://api.collegebasketballdata.com"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Rate limiting (optional - to protect your API quotas)
DAILY_LIMIT = 100
usage_data = {"daily": {}, "monthly": {}}

def get_today_key():
    return date.today().isoformat()

def get_month_key():
    return date.today().strftime("%Y-%m")

def get_usage_stats():
    today = get_today_key()
    month = get_month_key()
    today_count = usage_data["daily"].get(today, 0)
    month_count = usage_data["monthly"].get(month, 0)
    return {
        "today": today_count,
        "todayRemaining": max(0, DAILY_LIMIT - today_count),
        "month": month_count,
        "dailyLimit": DAILY_LIMIT,
    }

def increment_usage():
    today = get_today_key()
    month = get_month_key()
    usage_data["daily"][today] = usage_data["daily"].get(today, 0) + 1
    usage_data["monthly"][month] = usage_data["monthly"].get(month, 0) + 1
    return get_usage_stats()

# ============================================
# ROUTES
# ============================================

@app.route('/')
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "CBB Edge Finder Proxy",
        "usage": get_usage_stats()
    })

@app.route('/api/usage')
def usage():
    """Get API usage stats"""
    return jsonify(get_usage_stats())

@app.route('/api/odds')
def get_odds():
    """Proxy for The Odds API - Get NCAAB betting lines"""
    if not ODDS_API_KEY:
        return jsonify({"error": "ODDS_API_KEY not configured"}), 500
    
    try:
        url = f"{ODDS_API_BASE}/sports/basketball_ncaab/odds/"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'spreads,totals',
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Odds fetched successfully")
            return jsonify(response.json())
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Odds API error: {response.status_code}")
            return jsonify(response.json()), response.status_code
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Odds error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ratings')
def get_ratings():
    """Proxy for CollegeBasketballData - Get team efficiency ratings"""
    if not CBBD_API_KEY:
        return jsonify({"error": "CBBD_API_KEY not configured"}), 500
    
    season = request.args.get('season', '2026')
    
    try:
        url = f"{CBBD_API_BASE}/ratings/adjusted"
        params = {'season': season}
        headers = {
            'Authorization': f'Bearer {CBBD_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            stats = increment_usage()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Ratings fetched ({stats['today']}/{DAILY_LIMIT} today)")
            return jsonify(response.json())
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Ratings API error: {response.status_code}")
            return jsonify({"error": f"API returned {response.status_code}"}), response.status_code
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Ratings error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/games')
def get_games():
    """Proxy for CollegeBasketballData - Get games"""
    if not CBBD_API_KEY:
        return jsonify({"error": "CBBD_API_KEY not configured"}), 500
    
    season = request.args.get('season', '2026')
    
    try:
        url = f"{CBBD_API_BASE}/games"
        params = {'season': season}
        headers = {
            'Authorization': f'Bearer {CBBD_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            stats = increment_usage()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Games fetched ({stats['today']}/{DAILY_LIMIT} today)")
            return jsonify(response.json())
        else:
            return jsonify({"error": f"API returned {response.status_code}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams')
def get_teams():
    """Proxy for CollegeBasketballData - Get teams list"""
    if not CBBD_API_KEY:
        return jsonify({"error": "CBBD_API_KEY not configured"}), 500
    
    try:
        url = f"{CBBD_API_BASE}/teams"
        headers = {
            'Authorization': f'Bearer {CBBD_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            stats = increment_usage()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Teams fetched ({stats['today']}/{DAILY_LIMIT} today)")
            return jsonify(response.json())
        else:
            return jsonify({"error": f"API returned {response.status_code}"}), response.status_code
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🏀 CBB Edge Finder Proxy starting on port {port}")
    print(f"   CBBD API Key: {'✓ Set' if CBBD_API_KEY else '✗ Missing'}")
    print(f"   Odds API Key: {'✓ Set' if ODDS_API_KEY else '✗ Missing'}")
    app.run(host='0.0.0.0', port=port)
