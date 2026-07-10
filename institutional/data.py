from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


class DataUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    prices: pd.DataFrame
    metadata: dict[str, Any]
    fetched_at: datetime


def _yf():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataUnavailable("yfinance is not installed. Run: pip install -r requirements.txt") from exc
    return yf


def normalize_tickers(raw: str | list[str], limit: int = 75) -> list[str]:
    values = raw.replace("\n", ",").split(",") if isinstance(raw, str) else raw
    cleaned: list[str] = []
    for value in values:
        ticker = str(value).strip().upper()
        if ticker and ticker not in cleaned and all(c.isalnum() or c in ".-^" for c in ticker):
            cleaned.append(ticker)
    return cleaned[:limit]


def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    yf = _yf()
    try:
        frame = yf.download(ticker, period=period, interval=interval, auto_adjust=True,
                            progress=False, threads=False, timeout=15)
    except Exception:
        frame = pd.DataFrame()
    if frame is None or frame.empty:
        frame = _fetch_yahoo_chart(ticker, period, interval)
    if frame is None or frame.empty:
        raise DataUnavailable(f"No {interval} price history returned for {ticker}.")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in frame]
    if missing:
        raise DataUnavailable(f"{ticker} data is missing: {', '.join(missing)}")
    return frame[required].dropna(subset=["Close"]).sort_index()


def _fetch_yahoo_chart(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Use Yahoo's chart JSON when yfinance's higher-level client is unavailable."""
    try:
        import requests
    except ImportError:
        return pd.DataFrame()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    try:
        response = requests.get(url, params={"range": period, "interval": interval, "events": "div,splits"},
                                headers={"User-Agent": "Mozilla/5.0 AxiomResearch/1.0"}, timeout=15)
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame({
            "Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"),
            "Close": quote.get("close"), "Volume": quote.get("volume"),
        }, index=pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None))
        adjusted = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
        if adjusted and len(adjusted) == len(frame):
            ratio = pd.Series(adjusted, index=frame.index) / frame["Close"].replace(0, pd.NA)
            for column in ("Open", "High", "Low", "Close"):
                frame[column] = frame[column] * ratio
        return frame
    except Exception:
        return pd.DataFrame()


def fetch_metadata(ticker: str) -> dict[str, Any]:
    yf = _yf()
    try:
        return dict(yf.Ticker(ticker).info or {})
    except Exception:
        return {}


def fetch_snapshot(ticker: str, include_metadata: bool = True) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        prices=fetch_prices(ticker),
        metadata=fetch_metadata(ticker) if include_metadata else {},
        fetched_at=datetime.now(timezone.utc),
    )
