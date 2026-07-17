from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd


TRADIER_MARKET_URL = "https://api.tradier.com/v1/markets"
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class OptionsDataUnavailable(RuntimeError):
    """Raised when an options source cannot provide a usable snapshot."""


@dataclass
class OptionsSnapshot:
    ticker: str
    spot: float
    as_of: str
    contracts: pd.DataFrame
    expirations: list[str]
    provider: str = "Tradier"


def _number(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _request_json(path: str, token: str, params: dict[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.get(
        f"{TRADIER_MARKET_URL}/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params=params,
        timeout=20,
    )
    if response.status_code == 401:
        raise OptionsDataUnavailable("Tradier rejected the API token.")
    if response.status_code == 429:
        raise OptionsDataUnavailable("Tradier rate limit reached; try again shortly.")
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise OptionsDataUnavailable(f"Tradier request failed with HTTP {response.status_code}.") from exc
    if not isinstance(payload, dict):
        raise OptionsDataUnavailable("Tradier returned an unexpected response.")
    return payload


def _alpaca_request_json(url: str, key_id: str, secret_key: str,
                         params: dict[str, Any]) -> dict[str, Any]:
    import requests

    response = requests.get(
        url,
        headers={
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        },
        params=params,
        timeout=25,
    )
    if response.status_code == 401:
        raise OptionsDataUnavailable("Alpaca rejected the API credentials.")
    if response.status_code == 403:
        raise OptionsDataUnavailable(
            "Alpaca denied this options-data request. Check the selected OPRA/indicative feed and account permissions."
        )
    if response.status_code == 429:
        raise OptionsDataUnavailable("Alpaca rate limit reached; try again shortly.")
    try:
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise OptionsDataUnavailable(f"Alpaca request failed with HTTP {response.status_code}.") from exc
    if not isinstance(payload, dict):
        raise OptionsDataUnavailable("Alpaca returned an unexpected response.")
    return payload


def _alpaca_base(value: str, default: str) -> str:
    base = (value or default).strip().rstrip("/")
    return base[:-3] if base.endswith("/v2") else base


def normalize_alpaca_chain(contracts: list[dict[str, Any]], snapshots: dict[str, Any],
                           spot: float) -> pd.DataFrame:
    """Normalize Alpaca chain snapshots and contract metadata without inferring trade intent."""
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        symbol = str(contract.get("symbol") or "")
        snapshot = snapshots.get(symbol) if isinstance(snapshots, dict) else None
        if not isinstance(snapshot, dict):
            continue
        quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or {}
        trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
        daily = snapshot.get("dailyBar") or snapshot.get("daily_bar") or {}
        greeks = snapshot.get("greeks") or {}
        expiration = str(contract.get("expiration_date") or "")[:10]
        try:
            expiry_date = date.fromisoformat(expiration)
        except ValueError:
            continue
        bid = _number(quote.get("bp", quote.get("bid_price")))
        ask = _number(quote.get("ap", quote.get("ask_price")))
        last = _number(trade.get("p", trade.get("price")))
        mid = (bid + ask) / 2 if np.isfinite(bid) and np.isfinite(ask) and ask >= bid else last
        strike = _number(contract.get("strike_price"))
        option_type = str(contract.get("type") or "").lower()
        side = "Call" if option_type.startswith("c") else "Put" if option_type.startswith("p") else "Unknown"
        iv = _number(snapshot.get("impliedVolatility", snapshot.get("implied_volatility")))
        volume = max(_number(daily.get("v", daily.get("volume")), 0.0), 0.0)
        open_interest = max(_number(contract.get("open_interest"), 0.0), 0.0)
        spread_pct = ((ask - bid) / mid * 100) if np.isfinite(mid) and mid > 0 and np.isfinite(bid) and np.isfinite(ask) else np.nan
        rows.append({
            "Contract": symbol,
            "Type": side,
            "Expiration": expiration,
            "DTE": max((expiry_date - date.today()).days, 0),
            "Strike": strike,
            "Bid": bid,
            "Ask": ask,
            "Mid": mid,
            "Last": last,
            "Volume": volume,
            "Open Interest": open_interest,
            "Spread %": spread_pct,
            "IV %": iv * 100 if np.isfinite(iv) and iv <= 5 else iv,
            "Delta": _number(greeks.get("delta")),
            "Gamma": _number(greeks.get("gamma")),
            "Theta": _number(greeks.get("theta")),
            "Vega": _number(greeks.get("vega")),
            "Rho": _number(greeks.get("rho")),
            "Moneyness %": (strike / spot - 1) * 100 if spot > 0 and np.isfinite(strike) else np.nan,
            "Quoted Activity $": mid * volume * 100 if np.isfinite(mid) else np.nan,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise OptionsDataUnavailable("Alpaca returned no usable option-chain snapshots for the selected ticker.")
    return frame.sort_values(["Expiration", "Strike", "Type"]).reset_index(drop=True)


def fetch_alpaca_snapshot(ticker: str, key_id: str, secret_key: str, spot: float,
                          feed: str = "indicative", trading_base_url: str = ALPACA_PAPER_URL,
                          data_base_url: str = ALPACA_DATA_URL,
                          max_expirations: int = 6) -> OptionsSnapshot:
    if not key_id.strip() or not secret_key.strip():
        raise OptionsDataUnavailable("Alpaca credentials are not configured in Streamlit Secrets.")
    feed = feed.strip().lower()
    if feed not in {"indicative", "opra"}:
        raise OptionsDataUnavailable("ALPACA_OPTIONS_FEED must be 'indicative' or 'opra'.")
    trading_base = _alpaca_base(trading_base_url, ALPACA_PAPER_URL)
    data_base = _alpaca_base(data_base_url, ALPACA_DATA_URL)
    start = date.today()
    end = start + timedelta(days=120)
    strike_low, strike_high = max(0.01, spot * 0.65), spot * 1.35

    contract_params: dict[str, Any] = {
        "underlying_symbols": ticker,
        "status": "active",
        "expiration_date_gte": start.isoformat(),
        "expiration_date_lte": end.isoformat(),
        "strike_price_gte": round(strike_low, 2),
        "strike_price_lte": round(strike_high, 2),
        "limit": 10000,
    }
    contracts: list[dict[str, Any]] = []
    for _ in range(3):
        payload = _alpaca_request_json(
            f"{trading_base}/v2/options/contracts", key_id, secret_key, contract_params
        )
        contracts.extend(item for item in _items(payload.get("option_contracts")) if isinstance(item, dict))
        page_token = payload.get("page_token") or payload.get("next_page_token")
        if not page_token:
            break
        contract_params["page_token"] = page_token
    expirations = sorted({str(item.get("expiration_date"))[:10] for item in contracts if item.get("expiration_date")})[:max_expirations]
    if not expirations:
        raise OptionsDataUnavailable(f"Alpaca returned no current option contracts for {ticker}.")
    selected_contracts = [item for item in contracts if str(item.get("expiration_date"))[:10] in expirations]

    chain_params: dict[str, Any] = {
        "feed": feed,
        "expiration_date_gte": expirations[0],
        "expiration_date_lte": expirations[-1],
        "strike_price_gte": round(strike_low, 2),
        "strike_price_lte": round(strike_high, 2),
        "limit": 1000,
    }
    snapshots: dict[str, Any] = {}
    for _ in range(8):
        payload = _alpaca_request_json(
            f"{data_base}/v1beta1/options/snapshots/{ticker}", key_id, secret_key, chain_params
        )
        values = payload.get("snapshots") or {}
        if isinstance(values, dict):
            snapshots.update(values)
        page_token = payload.get("next_page_token") or payload.get("page_token")
        if not page_token:
            break
        chain_params["page_token"] = page_token
    frame = normalize_alpaca_chain(selected_contracts, snapshots, spot)
    return OptionsSnapshot(
        ticker=ticker,
        spot=spot,
        as_of=datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        contracts=frame,
        expirations=expirations,
        provider=f"Alpaca ({feed})",
    )


def _expiration_dates(payload: dict[str, Any]) -> list[str]:
    expirations = payload.get("expirations") or {}
    values = expirations.get("date") if isinstance(expirations, dict) else expirations
    result = sorted({str(value)[:10] for value in _items(values) if value})
    return [value for value in result if value >= date.today().isoformat()]


def normalize_tradier_chain(rows: list[dict[str, Any]], spot: float) -> pd.DataFrame:
    """Normalize Tradier contracts without inferring buyer/seller direction."""
    normalized: list[dict[str, Any]] = []
    today = date.today()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        greeks = raw.get("greeks") if isinstance(raw.get("greeks"), dict) else {}
        expiration = str(raw.get("expiration_date") or raw.get("expiration") or "")[:10]
        try:
            expiry_date = date.fromisoformat(expiration)
        except ValueError:
            continue
        bid, ask = _number(raw.get("bid")), _number(raw.get("ask"))
        if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
            mid = (bid + ask) / 2
        else:
            mid = _number(raw.get("last"))
        strike = _number(raw.get("strike"))
        volume = max(_number(raw.get("volume"), 0.0), 0.0)
        open_interest = max(_number(raw.get("open_interest"), 0.0), 0.0)
        option_type = str(raw.get("option_type") or raw.get("type") or "").lower()
        side = "Call" if option_type.startswith("c") else "Put" if option_type.startswith("p") else "Unknown"
        iv = _number(greeks.get("mid_iv"))
        if not np.isfinite(iv):
            iv = _number(greeks.get("smv_vol"))
        if not np.isfinite(iv):
            iv = _number(raw.get("implied_volatility"))
        spread_pct = ((ask - bid) / mid * 100) if np.isfinite(mid) and mid > 0 and np.isfinite(bid) and np.isfinite(ask) else np.nan
        normalized.append({
            "Contract": str(raw.get("symbol") or ""),
            "Type": side,
            "Expiration": expiration,
            "DTE": max((expiry_date - today).days, 0),
            "Strike": strike,
            "Bid": bid,
            "Ask": ask,
            "Mid": mid,
            "Last": _number(raw.get("last")),
            "Volume": volume,
            "Open Interest": open_interest,
            "Spread %": spread_pct,
            "IV %": iv * 100 if np.isfinite(iv) and iv <= 5 else iv,
            "Delta": _number(greeks.get("delta")),
            "Gamma": _number(greeks.get("gamma")),
            "Theta": _number(greeks.get("theta")),
            "Vega": _number(greeks.get("vega")),
            "Rho": _number(greeks.get("rho")),
            "Moneyness %": (strike / spot - 1) * 100 if spot > 0 and np.isfinite(strike) else np.nan,
            "Quoted Activity $": mid * volume * 100 if np.isfinite(mid) else np.nan,
        })
    frame = pd.DataFrame(normalized)
    if frame.empty:
        raise OptionsDataUnavailable("The selected expirations returned no option contracts.")
    return frame.sort_values(["Expiration", "Strike", "Type"]).reset_index(drop=True)


def fetch_tradier_snapshot(ticker: str, token: str, spot: float, max_expirations: int = 6) -> OptionsSnapshot:
    if not token.strip():
        raise OptionsDataUnavailable("TRADIER_TOKEN is not configured in Streamlit Secrets.")
    expiration_payload = _request_json(
        "options/expirations", token, {"symbol": ticker, "includeAllRoots": "true", "strikes": "false"}
    )
    expirations = _expiration_dates(expiration_payload)[:max_expirations]
    if not expirations:
        raise OptionsDataUnavailable(f"No current option expirations were returned for {ticker}.")
    rows: list[dict[str, Any]] = []
    for expiration in expirations:
        payload = _request_json(
            "options/chains", token,
            {"symbol": ticker, "expiration": expiration, "greeks": "true"},
        )
        options = payload.get("options") or {}
        values = options.get("option") if isinstance(options, dict) else options
        rows.extend(item for item in _items(values) if isinstance(item, dict))
    frame = normalize_tradier_chain(rows, spot)
    return OptionsSnapshot(
        ticker=ticker,
        spot=spot,
        as_of=datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"),
        contracts=frame,
        expirations=expirations,
    )


def options_summary(snapshot: OptionsSnapshot) -> dict[str, float]:
    frame = snapshot.contracts
    calls, puts = frame[frame["Type"] == "Call"], frame[frame["Type"] == "Put"]
    call_volume, put_volume = calls["Volume"].sum(), puts["Volume"].sum()
    call_oi, put_oi = calls["Open Interest"].sum(), puts["Open Interest"].sum()
    valid_spreads = frame.loc[(frame["Mid"] > 0) & frame["Spread %"].notna(), "Spread %"]
    return {
        "Contracts": float(len(frame)),
        "Expirations": float(frame["Expiration"].nunique()),
        "Put/Call Volume": float(put_volume / call_volume) if call_volume else np.nan,
        "Put/Call OI": float(put_oi / call_oi) if call_oi else np.nan,
        "Median Spread %": float(valid_spreads.median()) if not valid_spreads.empty else np.nan,
        "Call Volume Share %": float(call_volume / (call_volume + put_volume) * 100) if call_volume + put_volume else np.nan,
    }


def expected_move_table(snapshot: OptionsSnapshot) -> pd.DataFrame:
    frame = snapshot.contracts
    pivot = frame.pivot_table(index=["Expiration", "DTE", "Strike"], columns="Type", values="Mid", aggfunc="first").reset_index()
    if not {"Call", "Put"}.issubset(pivot.columns):
        return pd.DataFrame()
    pivot = pivot.dropna(subset=["Call", "Put"])
    rows: list[dict[str, Any]] = []
    for expiration, group in pivot.groupby("Expiration"):
        closest = group.loc[(group["Strike"] - snapshot.spot).abs().idxmin()]
        move = float(closest["Call"] + closest["Put"])
        rows.append({
            "Expiration": expiration,
            "DTE": int(closest["DTE"]),
            "ATM Strike": float(closest["Strike"]),
            "ATM Straddle": move,
            "Expected Move %": move / snapshot.spot * 100 if snapshot.spot else np.nan,
            "Expected Low": max(snapshot.spot - move, 0),
            "Expected High": snapshot.spot + move,
        })
    return pd.DataFrame(rows).sort_values("DTE").reset_index(drop=True)


def income_candidates(snapshot: OptionsSnapshot, strategy: str, min_open_interest: int = 100,
                      max_spread_pct: float = 20.0) -> pd.DataFrame:
    """Screen conservative quoted income candidates; this is not an execution recommendation."""
    frame = snapshot.contracts.copy()
    if strategy == "Cash-Secured Put":
        frame = frame[(frame["Type"] == "Put") & (frame["Strike"] < snapshot.spot)]
    elif strategy == "Covered Call":
        frame = frame[(frame["Type"] == "Call") & (frame["Strike"] > snapshot.spot)]
    else:
        raise ValueError("Unknown income strategy")
    frame = frame[
        frame["DTE"].between(7, 45)
        & (frame["Bid"] > 0)
        & (frame["Open Interest"] >= min_open_interest)
        & (frame["Spread %"] <= max_spread_pct)
    ].copy()
    if frame.empty:
        return frame
    premium = frame["Bid"] * 100
    if strategy == "Cash-Secured Put":
        capital = frame["Strike"] * 100 - premium
        frame["Breakeven"] = frame["Strike"] - frame["Bid"]
        frame["Maximum Profit"] = premium
    else:
        capital = snapshot.spot * 100
        frame["Breakeven"] = snapshot.spot - frame["Bid"]
        frame["Maximum Profit"] = (frame["Strike"] - snapshot.spot + frame["Bid"]) * 100
    frame["Premium $"] = premium
    frame["Capital $"] = capital
    frame["Period Return %"] = premium / capital * 100
    frame["Annualized Yield %"] = frame["Period Return %"] * 365 / frame["DTE"]
    frame["Delta OTM Proxy %"] = (1 - frame["Delta"].abs()).clip(0, 1) * 100
    frame["Liquidity Score"] = (
        np.clip(np.log1p(frame["Open Interest"]) / np.log(10001), 0, 1) * 60
        + np.clip(1 - frame["Spread %"] / max_spread_pct, 0, 1) * 40
    )
    columns = [
        "Contract", "Expiration", "DTE", "Strike", "Bid", "Ask", "Spread %", "Volume",
        "Open Interest", "IV %", "Delta", "Premium $", "Capital $", "Breakeven",
        "Period Return %", "Annualized Yield %", "Delta OTM Proxy %", "Liquidity Score",
    ]
    return frame[columns].sort_values(["Liquidity Score", "Annualized Yield %"], ascending=False).head(20)
