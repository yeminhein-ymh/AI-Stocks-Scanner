from __future__ import annotations

DEFAULT_UNIVERSE = [
    "NVDA", "MU", "SNDK", "META", "GOOGL", "AMZN", "NBIS", "PLTR",
    "AAPL", "MSFT", "AVGO", "AMD", "TSM", "NFLX", "CRWD", "NOW",
    "JPM", "V", "LLY", "COST", "WMT", "XOM", "CAT", "GE",
]

BENCHMARKS = ["SPY", "QQQ", "SOXX", "^VIX"]

SECTOR_ETFS = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Communication Services": "XLC", "Industrials": "XLI", "Energy": "XLE",
    "Basic Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}

SCORE_WEIGHTS = {
    "trend": 0.22, "momentum": 0.15, "volume": 0.10,
    "relative_strength": 0.14, "risk_reward": 0.14,
    "fundamental": 0.15, "macro": 0.10,
}

DATA_PROVIDERS = {
    "price": "Yahoo Finance via yfinance (delayed/end-of-day; provider terms apply)",
    "fundamental": "Yahoo Finance company metadata (availability varies)",
    "dark_pool": None,
    "options_flow": None,
    "dealer_gamma": None,
    "analyst_consensus": None,
    "news_sentiment": None,
}

