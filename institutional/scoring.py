from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .config import SCORE_WEIGHTS


def _clip(value: float, low: float = 0, high: float = 100) -> float:
    return float(np.clip(value, low, high))


def _safe(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    prev = frame["Close"].shift(1)
    tr = pd.concat([(frame["High"] - frame["Low"]),
                    (frame["High"] - prev).abs(), (frame["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def _adx(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    up = frame["High"].diff()
    down = -frame["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(frame, window).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    return (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).ewm(
        alpha=1 / window, adjust=False).mean()


@dataclass
class ScoreComponent:
    score: float
    reasons: list[str] = field(default_factory=list)
    coverage: float = 1.0


@dataclass
class Analysis:
    ticker: str
    as_of: str
    price: float
    overall_score: float
    bull_probability: float
    bear_probability: float
    sideways_probability: float
    confidence: float
    risk_score: float
    expected_return_20d: float
    expected_drawdown: float
    stop_loss: float
    trailing_stop: float
    target_1: float
    target_2: float
    target_3: float
    reward_risk: float
    kelly_fraction: float
    recommended_position_size: float
    holding_period: str
    classification: str
    trend_stage: str
    exit_condition: str
    metrics: dict[str, Any]
    components: dict[str, ScoreComponent]
    probabilities: dict[str, dict[str, float]]
    unavailable: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["Close"]
    for window in (9, 20, 50, 60, 100, 200):
        df[f"EMA{window}"] = close.ewm(span=window, adjust=False).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["RSI14"] = _rsi(close)
    df["ATR14"] = _atr(df)
    df["ADX14"] = _adx(df)
    df["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RVOL"] = df["Volume"] / df["VOL20"].replace(0, np.nan)
    df["RET1"] = close.pct_change()
    df["RET20"] = close.pct_change(20)
    df["RET60"] = close.pct_change(60)
    df["HV20"] = df["RET1"].rolling(20).std() * sqrt(252)
    df["HIGH20"] = df["High"].rolling(20).max().shift(1)
    df["LOW20"] = df["Low"].rolling(20).min().shift(1)
    return df


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _forward_table(mu_daily: float, sigma_daily: float, confidence: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label, days in (("3 trading days", 3), ("2 weeks", 10), ("1 month", 20), ("3 months", 60)):
        mu, sigma = mu_daily * days, max(sigma_daily * sqrt(days), 0.001)
        bull = _normal_cdf(mu / sigma) * 100
        result[label] = {
            "bull": round(bull, 1), "bear": round(100 - bull, 1),
            "expected_return": round(mu * 100, 2),
            "confidence": round(confidence * min(1, sqrt(20 / days)), 1),
        }
    return result


def _fundamental_component(meta: dict[str, Any]) -> ScoreComponent:
    votes: list[float] = []
    reasons: list[str] = []
    checks = [
        ("revenueGrowth", 0.10, "Revenue growth above 10%", "Revenue growth below 10%"),
        ("earningsGrowth", 0.10, "Earnings growth above 10%", "Earnings growth below 10%"),
        ("profitMargins", 0.10, "Net margin above 10%", "Net margin below 10%"),
        ("returnOnEquity", 0.15, "ROE above 15%", "ROE below 15%"),
    ]
    for key, threshold, good, weak in checks:
        value = _safe(meta.get(key))
        if np.isfinite(value):
            votes.append(75 if value >= threshold else 40)
            reasons.append(good if value >= threshold else weak)
    debt = _safe(meta.get("debtToEquity"))
    if np.isfinite(debt):
        votes.append(70 if debt < 100 else 35)
        reasons.append("Debt-to-equity below 100%" if debt < 100 else "Elevated debt-to-equity")
    return ScoreComponent(round(float(np.mean(votes)), 1) if votes else 50.0,
                          reasons or ["Fundamental fields unavailable; neutral prior used"],
                          min(1, len(votes) / 5))


def analyze(ticker: str, frame: pd.DataFrame, benchmark: pd.DataFrame | None = None,
            metadata: dict[str, Any] | None = None, max_position_pct: float = 20.0,
            risk_per_trade_pct: float = 1.0) -> Analysis:
    if len(frame) < 80:
        raise ValueError(f"{ticker} needs at least 80 daily observations; received {len(frame)}")
    df = prepare_features(frame)
    row = df.iloc[-1]
    price = float(row["Close"])
    reasons: dict[str, ScoreComponent] = {}

    trend_votes = []
    trend_reasons = []
    for fast, slow in (("Close", "EMA20"), ("EMA20", "EMA60"), ("EMA60", "EMA200")):
        if np.isfinite(_safe(row.get(slow))):
            positive = _safe(row[fast]) > _safe(row[slow])
            trend_votes.append(85 if positive else 25)
            trend_reasons.append(f"{fast} above {slow}" if positive else f"{fast} below {slow}")
    adx = _safe(row["ADX14"], 20)
    trend_votes.append(_clip(40 + adx * 1.2))
    trend_reasons.append(f"ADX {adx:.1f} indicates {'strong' if adx >= 25 else 'developing'} trend strength")
    hh = row["High"] >= df["High"].iloc[-21:-1].max()
    trend_votes.append(85 if hh else 50)
    trend_reasons.append("New 20-day high" if hh else "No fresh 20-day breakout")
    reasons["trend"] = ScoreComponent(round(float(np.mean(trend_votes)), 1), trend_reasons)

    rsi = _safe(row["RSI14"], 50)
    macd_positive = _safe(row["MACD"]) > _safe(row["MACD_SIGNAL"])
    ret20 = _safe(row["RET20"], 0)
    momentum_score = _clip(50 + (rsi - 50) * 0.8 + (15 if macd_positive else -10) + np.clip(ret20 * 100, -15, 15))
    reasons["momentum"] = ScoreComponent(round(momentum_score, 1), [
        f"RSI(14) is {rsi:.1f}", f"MACD is {'above' if macd_positive else 'below'} its signal line",
        f"20-day return is {ret20 * 100:+.1f}%",
    ])

    rvol = _safe(row["RVOL"], 1)
    up_volume = df.loc[df["RET1"] > 0, "Volume"].tail(20).mean()
    down_volume = df.loc[df["RET1"] < 0, "Volume"].tail(20).mean()
    pressure = _safe(up_volume / down_volume, 1)
    volume_score = _clip(45 + (rvol - 1) * 25 + (pressure - 1) * 20)
    reasons["volume"] = ScoreComponent(round(volume_score, 1), [
        f"Relative volume is {rvol:.2f}x", f"Up/down volume pressure ratio is {pressure:.2f}x",
    ])

    bench_ret = 0.0
    coverage = 0.5
    if benchmark is not None and not benchmark.empty:
        bench_ret = _safe(benchmark["Close"].pct_change(60).iloc[-1], 0)
        coverage = 1.0
    relative = _safe(row["RET60"], 0) - bench_ret
    rs_score = _clip(50 + relative * 180)
    reasons["relative_strength"] = ScoreComponent(round(rs_score, 1), [
        f"60-day relative return versus SPY is {relative * 100:+.1f}%",
    ], coverage)

    atr = _safe(row["ATR14"], price * 0.03)
    atr_pct = atr / price
    stop = price - 2 * atr
    target1, target2, target3 = price + 2 * atr, price + 4 * atr, price + 6 * atr
    reward_risk = (target2 - price) / max(price - stop, 0.01)
    risk_reward_score = _clip(65 + (reward_risk - 2) * 10 - max(0, atr_pct - 0.04) * 500)
    reasons["risk_reward"] = ScoreComponent(round(risk_reward_score, 1), [
        f"ATR is {atr_pct * 100:.1f}% of price", f"Target 2 offers {reward_risk:.1f}:1 reward/risk",
    ])

    reasons["fundamental"] = _fundamental_component(metadata or {})
    macro_score = 50.0
    macro_reasons = ["Macro prior is neutral; connect rates, dollar, commodities, and sector feeds for a full regime vote"]
    if benchmark is not None and len(benchmark) >= 200:
        b = benchmark["Close"]
        macro_score = 72.0 if b.iloc[-1] > b.ewm(span=200, adjust=False).mean().iloc[-1] else 32.0
        macro_reasons = [f"SPY is {'above' if macro_score > 50 else 'below'} its 200-day trend"]
    reasons["macro"] = ScoreComponent(macro_score, macro_reasons, 0.6)

    weighted = sum(reasons[key].score * weight for key, weight in SCORE_WEIGHTS.items())
    data_coverage = sum(reasons[key].coverage * weight for key, weight in SCORE_WEIGHTS.items())
    returns = df["RET1"].dropna().tail(252)
    sigma_daily = _safe(returns.std(), 0.02)
    historical_mu = _safe(returns.mean(), 0)
    score_mu = (weighted - 50) / 50 * sigma_daily * 0.22
    mu_daily = 0.55 * historical_mu + 0.45 * score_mu
    sample_conf = min(1, sqrt(len(returns) / 252))
    agreement = 1 - min(1, np.std([c.score for c in reasons.values()]) / 35)
    confidence = _clip(100 * data_coverage * sample_conf * (0.55 + 0.45 * agreement), 10, 92)
    probs = _forward_table(mu_daily, sigma_daily, confidence)
    bull = probs["1 month"]["bull"]
    sideways = _clip(35 - abs(bull - 50) * 0.7, 8, 35)
    directional = 100 - sideways
    bull = directional * bull / 100
    bear = 100 - sideways - bull
    expected_return = mu_daily * 20 * 100
    expected_drawdown = -1.65 * sigma_daily * sqrt(20) * 100
    win_rate = bull / 100
    payoff = max(reward_risk, 0.1)
    kelly = _clip(win_rate - (1 - win_rate) / payoff, 0, 0.25)
    risk_position = risk_per_trade_pct / max((price - stop) / price * 100, 0.1) * 100
    position = min(max_position_pct, risk_position, kelly * 100 if kelly > 0 else 0)

    ema50, ema200 = _safe(row["EMA50"]), _safe(row["EMA200"])
    if price > ema50 > ema200 and _safe(row["EMA200"] - df["EMA200"].iloc[-20], 0) > 0:
        stage = "Stage 2 — Advancing"
    elif price < ema50 < ema200:
        stage = "Stage 4 — Declining"
    elif price >= ema200:
        stage = "Stage 1/3 — Base or Distribution"
    else:
        stage = "Stage 1 — Basing"
    if weighted >= 78 and confidence >= 60:
        classification = "Strong Buy"
    elif weighted >= 65:
        classification = "Buy / Swing Trade"
    elif weighted >= 55:
        classification = "Watchlist"
    elif weighted < 35:
        classification = "Avoid / Short Candidate"
    else:
        classification = "Neutral"

    metrics = {
        "RSI14": rsi, "ADX14": adx, "ATR14": atr, "ATR %": atr_pct * 100,
        "Relative Volume": rvol, "20D Return %": ret20 * 100,
        "60D Relative Strength %": relative * 100, "Annualized Volatility %": sigma_daily * sqrt(252) * 100,
        "Beta": _safe(metadata.get("beta") if metadata else None),
        "Market Cap": _safe(metadata.get("marketCap") if metadata else None),
        "Forward PE": _safe(metadata.get("forwardPE") if metadata else None),
    }
    unavailable = ["Dark pool activity", "Options flow", "Dealer gamma exposure", "Live news sentiment"]
    return Analysis(
        ticker=ticker, as_of=str(df.index[-1].date()), price=round(price, 2), overall_score=round(weighted, 1),
        bull_probability=round(bull, 1), bear_probability=round(bear, 1), sideways_probability=round(sideways, 1),
        confidence=round(confidence, 1), risk_score=round(_clip(atr_pct * 1000 + (100 - risk_reward_score) * .5), 1),
        expected_return_20d=round(expected_return, 2), expected_drawdown=round(expected_drawdown, 2),
        stop_loss=round(stop, 2), trailing_stop=round(price - 2.5 * atr, 2), target_1=round(target1, 2),
        target_2=round(target2, 2), target_3=round(target3, 2), reward_risk=round(reward_risk, 2),
        kelly_fraction=round(kelly * 100, 2), recommended_position_size=round(position, 2),
        holding_period="2–8 weeks" if stage.startswith("Stage 2") else "3–10 trading days",
        classification=classification, trend_stage=stage,
        exit_condition=f"Daily close below ${stop:.2f}, a bearish structure break, or score below 45",
        metrics=metrics, components=reasons, probabilities=probs, unavailable=unavailable,
    )


def price_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_features(frame)

