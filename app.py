from __future__ import annotations

import io
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from institutional.config import DATA_PROVIDERS, DEFAULT_UNIVERSE
from institutional.data import DataUnavailable, fetch_metadata, fetch_prices, normalize_tickers
from institutional.scoring import Analysis, analyze, price_indicators
from institutional.signals import SignalStore, performance_metrics


st.set_page_config(page_title="Axiom AI Research", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

TICKER_LIMIT = 30
SHORT_TERM_WEIGHTS = {
    "Technical Analysis": 25.0,
    "Volume & Price Action": 15.0,
    "Options Flow": 15.0,
    "News & Sentiment": 15.0,
    "Fundamental Momentum": 10.0,
    "Sector Rotation": 8.0,
    "Macro Economy": 7.0,
    "AI / Statistical Model": 5.0,
}
TICKER_STORAGE = components.declare_component(
    "ticker_storage",
    path=str(Path(__file__).parent / "ticker_storage"),
)

CSS = """
<style>
:root { --ink:#243247; --muted:#62748a; --panel:#ffffff; --line:#d7e0ea; --teal:#087f72; --soft:#f4f7fb; }
.stApp, [data-testid="stAppViewContainer"] { background:linear-gradient(180deg,#fbfcfe 0%,#f3f6fa 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(251,252,254,.94); }
[data-testid="stSidebar"] { background:#edf3f8; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#334155; }
[data-testid="stMetric"] { min-width:0; height:100%; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:11px 12px; box-shadow:0 3px 12px rgba(30,55,80,.05); }
[data-testid="stMetricLabel"] { color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricLabel"] p { font-size:.68rem !important; line-height:1.2 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] * { color:#1f2f44 !important; font-size:1.35rem !important; line-height:1.18 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; overflow-wrap:break-word !important; word-break:normal !important; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { font-size:.70rem !important; line-height:1.2 !important; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label { color:var(--ink); }
.block-container { padding-top:1.5rem; max-width:1600px; }
.eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; }
.hero { color:#18273b; font-size:1.72rem; line-height:1.2; font-weight:700; margin:.15rem 0; }
.subtle { color:var(--muted); font-size:.86rem; }
.pill { display:inline-block; padding:.25rem .65rem; border-radius:999px; background:#e3f4ef; color:#07695f; border:1px solid #a9d9cf; font-size:.78rem; }
.reason { padding:.55rem .7rem; margin:.3rem 0; border-left:3px solid var(--teal); background:#eef7f5; color:#31465b; border-radius:0 7px 7px 0; }
.warningbox { background:#fff8e8; border:1px solid #ead5a0; padding:.7rem; border-radius:8px; color:#70561d; }
.stTabs [data-baseweb="tab-list"] { gap:1.2rem; border-bottom:1px solid var(--line); }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, textarea { background:#ffffff !important; color:#25364b !important; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def cached_prices(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    return fetch_prices(ticker, period, interval)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_metadata(ticker: str) -> dict[str, Any]:
    return fetch_metadata(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def cached_analysis(ticker: str, max_position: float, risk_per_trade: float) -> tuple[Analysis, pd.DataFrame]:
    prices = cached_prices(ticker)
    try:
        benchmark = cached_prices("SPY")
    except DataUnavailable:
        benchmark = None
    result = analyze(ticker, prices, benchmark, cached_metadata(ticker), max_position, risk_per_trade)
    return result, prices


def fmt(value: Any, kind: str = "number") -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return "N/A"
        if kind == "pct":
            return f"{number:.1f}%"
        if kind == "price":
            return f"${number:,.2f}"
        if kind == "money":
            return f"${number:,.0f}"
        return f"{number:,.1f}"
    except (TypeError, ValueError):
        return "N/A"


def excel_bytes(frame: pd.DataFrame) -> bytes | None:
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="Axiom Research", index=False)
        return buffer.getvalue()
    except ImportError:
        return None


def header(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">Axiom Intelligence Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero">{title}</div><div class="subtle">{subtitle}</div>', unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str]], columns_per_row: int) -> None:
    """Render readable metric cards in bounded-width rows."""
    for start in range(0, len(items), columns_per_row):
        row = st.columns(columns_per_row)
        for column, (label, value) in zip(row, items[start:start + columns_per_row]):
            column.metric(label, value)


def freshness(as_of: str) -> None:
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    st.caption(f"Price data through {as_of} · App refreshed {now} · Delayed/end-of-day source")


def saved_ticker_universe() -> str:
    """Load the user's last universe from the current page URL."""
    if "tickers" not in st.query_params:
        return ", ".join(normalize_tickers(DEFAULT_UNIVERSE, limit=TICKER_LIMIT))
    saved = st.query_params.get("tickers", "")
    if isinstance(saved, list):
        saved = ",".join(saved)
    return ", ".join(normalize_tickers(str(saved), limit=TICKER_LIMIT))


def persist_ticker_universe() -> None:
    """Replace the saved universe with the user's latest edited list."""
    raw = st.session_state.get("ticker_universe", "")
    tickers = normalize_tickers(raw, limit=TICKER_LIMIT)
    st.session_state["ticker_universe"] = ", ".join(tickers)
    st.query_params["tickers"] = ",".join(tickers)


def restore_browser_ticker_universe() -> None:
    """Restore the latest browser-local list once, including a deliberately empty list."""
    if "ticker_universe" not in st.session_state:
        st.session_state["ticker_universe"] = saved_ticker_universe()

    storage_loaded = st.session_state.get("ticker_storage_loaded", False)
    storage_state = TICKER_STORAGE(
        value=st.session_state["ticker_universe"],
        write=storage_loaded,
        default=None,
        key="ticker_storage_bridge",
    )
    if storage_loaded or not isinstance(storage_state, dict):
        return

    if storage_state.get("found"):
        stored = ", ".join(
            normalize_tickers(str(storage_state.get("value", "")), limit=TICKER_LIMIT)
        )
        st.session_state["ticker_universe"] = stored
        st.query_params["tickers"] = ",".join(normalize_tickers(stored, limit=TICKER_LIMIT))

    # When no browser value exists, the URL/default value is written on the next
    # run. When a value exists, it has already replaced the temporary fallback.
    st.session_state["ticker_storage_loaded"] = True
    st.rerun()


def probability_chart(a: Analysis) -> go.Figure:
    horizons = list(a.probabilities)
    fig = go.Figure()
    fig.add_bar(name="Bull", x=horizons, y=[a.probabilities[h]["bull"] for h in horizons], marker_color="#20c997")
    fig.add_bar(name="Bear", x=horizons, y=[a.probabilities[h]["bear"] for h in horizons], marker_color="#ff6b6b")
    fig.update_layout(barmode="group", height=310, margin=dict(l=10, r=10, t=25, b=10), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      yaxis_title="Modeled probability (%)", legend_orientation="h")
    return fig


def price_chart(prices: pd.DataFrame, title: str) -> go.Figure:
    df = price_indicators(prices)
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                   name="Price", increasing_line_color="#2dd4bf", decreasing_line_color="#fb7185"))
    for column, color in (("EMA20", "#60a5fa"), ("EMA50", "#f59e0b"), ("EMA200", "#a78bfa")):
        fig.add_scatter(x=df.index, y=df[column], name=column, line=dict(width=1.2, color=color))
    fig.update_layout(title=title, height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=45, b=10),
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", legend_orientation="h")
    return fig


def factor_profile_chart(a: Analysis) -> go.Figure:
    labels = [name.replace("_", " ").title() for name in a.components]
    scores = [component.score for component in a.components.values()]
    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h", text=[f"{score:.1f}" for score in scores],
        textposition="outside", cliponaxis=False, marker=dict(color="#159487", line=dict(color="#087f72", width=1)),
        hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dot", line_color="#64748b", annotation_text="Neutral 50")
    fig.update_layout(
        title="Factor score profile · weighted inputs", height=345,
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        margin=dict(l=15, r=45, t=50, b=35), xaxis=dict(range=[0, 105], title="Score (0–100)", gridcolor="#e2e8f0"),
        yaxis=dict(autorange="reversed"), showlegend=False,
    )
    return fig


def component_reference_chart(name: str, a: Analysis, features: pd.DataFrame,
                              benchmark_features: pd.DataFrame | None) -> tuple[go.Figure, str]:
    component = a.components[name]
    recent = features.tail(180)
    if name == "trend":
        fig = go.Figure()
        for column, color, width in (("Close", "#172b4d", 2.4), ("EMA20", "#159487", 1.7),
                                     ("EMA60", "#d99a20", 1.5), ("EMA200", "#8b5fbf", 1.5)):
            fig.add_scatter(x=recent.index, y=recent[column], name=column, line=dict(color=color, width=width))
        fig.update_layout(title="Trend reference · price versus EMA20, EMA60 and EMA200", yaxis_title="Adjusted price")
        caption = "Last 180 daily observations. Alignment and slope show whether short-, medium-, and long-term trends agree."
    elif name == "momentum":
        momentum = features.tail(120)
        fig = go.Figure(go.Scatter(x=momentum.index, y=momentum["RSI14"], name="RSI(14)",
                                   line=dict(color="#159487", width=2.2)))
        fig.add_hline(y=70, line_dash="dot", line_color="#b65c5c", annotation_text="Overbought 70")
        fig.add_hline(y=50, line_dash="dot", line_color="#64748b", annotation_text="Neutral 50")
        fig.add_hline(y=30, line_dash="dot", line_color="#4e79a7", annotation_text="Oversold 30")
        fig.update_layout(title="Momentum reference · RSI(14) over 120 sessions", yaxis=dict(title="RSI", range=[0, 100]))
        caption = "RSI shows the speed and persistence of recent price moves; the score also considers MACD and 20-day return."
    elif name == "volume":
        volume = features.tail(60)
        colors = ["#159487" if value >= 1 else "#b8c5d1" for value in volume["RVOL"].fillna(0)]
        fig = go.Figure(go.Bar(x=volume.index, y=volume["RVOL"], marker_color=colors, name="Relative volume"))
        fig.add_hline(y=1, line_dash="dot", line_color="#64748b", annotation_text="20-day average")
        fig.update_layout(title="Volume reference · daily volume relative to its 20-day average", yaxis_title="Relative volume (×)")
        caption = "Bars above 1.0× indicate above-average participation; stronger volume gives price moves more evidential weight."
    elif name == "relative_strength" and benchmark_features is not None:
        joined = pd.concat([features["Close"].rename(a.ticker), benchmark_features["Close"].rename("SPY")], axis=1).dropna().tail(126)
        indexed = joined / joined.iloc[0] * 100
        fig = go.Figure()
        fig.add_scatter(x=indexed.index, y=indexed[a.ticker], name=a.ticker, line=dict(color="#159487", width=2.3))
        fig.add_scatter(x=indexed.index, y=indexed["SPY"], name="SPY", line=dict(color="#64748b", width=1.8, dash="dash"))
        fig.update_layout(title=f"Relative-strength reference · {a.ticker} versus SPY (indexed to 100)", yaxis_title="Indexed performance")
        caption = "Both series start at 100 over the latest 126 sessions; a widening positive gap indicates benchmark outperformance."
    elif name == "risk_reward":
        levels = [a.stop_loss, a.price, a.target_1, a.target_2, a.target_3]
        labels = ["Stop", "Current", "Target 1", "Target 2", "Target 3"]
        colors = ["#b65c5c", "#172b4d", "#159487", "#159487", "#159487"]
        fig = go.Figure()
        fig.add_scatter(x=[a.stop_loss, a.target_3], y=["Trade plan", "Trade plan"], mode="lines",
                        line=dict(color="#94a3b8", width=4), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=levels, y=["Trade plan"] * len(levels), mode="markers+text", text=labels,
                        textposition="top center", marker=dict(size=15, color=colors, line=dict(color="#ffffff", width=2)),
                        customdata=labels, hovertemplate="%{customdata}: $%{x:,.2f}<extra></extra>", showlegend=False)
        fig.update_layout(title="Risk/reward reference · ATR-based stop and target ladder", xaxis_title="Price", yaxis_title="")
        caption = f"Current reward/risk is {a.reward_risk:.2f}:1 using Target 2 versus the ATR-defined stop; levels are estimates, not guarantees."
    elif name == "macro" and benchmark_features is not None:
        macro = benchmark_features.tail(252)
        fig = go.Figure()
        fig.add_scatter(x=macro.index, y=macro["Close"], name="SPY", line=dict(color="#172b4d", width=2.2))
        fig.add_scatter(x=macro.index, y=macro["EMA200"], name="SPY EMA200", line=dict(color="#d99a20", width=1.8))
        fig.update_layout(title="Macro reference · SPY versus its 200-day trend", yaxis_title="Adjusted price")
        caption = "The current macro vote uses the broad-market long-term trend; rates, dollar, commodities, and sector feeds remain optional enrichment."
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=component.score, number={"suffix": " / 100"},
            title={"text": f"{name.replace('_', ' ').title()} score · {component.coverage:.0%} data coverage"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#159487"},
                   "steps": [{"range": [0, 40], "color": "#f1f5f9"}, {"range": [40, 70], "color": "#e2e8f0"},
                             {"range": [70, 100], "color": "#d8eee9"}]},
        ))
        caption = "A neutral prior is used when source fields are unavailable; the displayed coverage prevents missing data from looking like strong evidence."
    fig.update_layout(
        height=430, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        margin=dict(l=45, r=25, t=55, b=45), legend=dict(orientation="h", y=1.02, x=0),
        font=dict(color="#334155", size=12), xaxis_gridcolor="#e2e8f0", yaxis_gridcolor="#e2e8f0",
    )
    return fig, caption


def short_term_probability_table(a: Analysis, prices: pd.DataFrame, ensemble_score: float,
                                 effective_coverage: float) -> pd.DataFrame:
    """Create source-aware distribution forecasts without claiming certainty."""
    features = price_indicators(prices)
    returns = features["RET1"].replace([np.inf, -np.inf], np.nan).dropna().tail(252)
    sigma_daily = max(float(returns.std()) if len(returns) > 1 else 0.02, 0.004)
    historical_mu = float(returns.tail(60).mean()) if not returns.empty else 0.0
    score_mu = (ensemble_score - 50) / 50 * sigma_daily * 0.18
    mu_daily = 0.60 * historical_mu + 0.40 * score_mu
    base_confidence = a.confidence * (0.50 + 0.50 * effective_coverage)

    normal_cdf = lambda value: 0.5 * (1 + erf(value / sqrt(2)))
    rows: list[dict[str, Any]] = []
    for label, days, decay in (("1 Trading Day", 1, 1.00), ("3 Trading Days", 3, 0.96),
                               ("1 Week", 5, 0.92), ("2 Weeks", 10, 0.85)):
        expected = mu_daily * days
        sigma = max(sigma_daily * sqrt(days), 0.001)
        neutral_band = 0.28 * sigma
        bear = normal_cdf((-neutral_band - expected) / sigma)
        bull = 1 - normal_cdf((neutral_band - expected) / sigma)
        neutral = max(0.0, 1 - bull - bear)
        z80 = 1.2816
        expected_high = a.price * (1 + expected + z80 * sigma)
        expected_low = max(0.0, a.price * (1 + expected - z80 * sigma))

        def probability_above(threshold: float) -> float:
            return (1 - normal_cdf((threshold - expected) / sigma)) * 100

        def probability_below(threshold: float) -> float:
            return normal_cdf((threshold - expected) / sigma) * 100

        rows.append({
            "Timeframe": label,
            "Days": days,
            "Bull %": round(bull * 100, 1),
            "Neutral %": round(neutral * 100, 1),
            "Bear %": round(bear * 100, 1),
            "Expected Return %": round(expected * 100, 2),
            "Expected Move %": round(sigma * 100, 2),
            "Expected High": round(expected_high, 2),
            "Expected Low": round(expected_low, 2),
            "80% Range": f"${expected_low:,.2f} – ${expected_high:,.2f}",
            "Max Drawdown est. %": round(-1.65 * sigma * 100, 2),
            "P(+5%)": round(probability_above(0.05), 1),
            "P(-5%)": round(probability_below(-0.05), 1),
            "P(+10%)": round(probability_above(0.10), 1),
            "P(-10%)": round(probability_below(-0.10), 1),
            "Confidence %": round(float(np.clip(base_confidence * decay, 10, 92)), 1),
        })
    return pd.DataFrame(rows)


def short_term_probability_chart(forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = {"Bull %": "#159487", "Neutral %": "#b8c5d1", "Bear %": "#d99a20"}
    for column in ("Bull %", "Neutral %", "Bear %"):
        fig.add_bar(
            y=forecast["Timeframe"], x=forecast[column], name=column.replace(" %", ""),
            orientation="h", marker=dict(color=palette[column], line=dict(color="#ffffff", width=1)),
            text=forecast[column].map(lambda value: f"{value:.1f}%"), textposition="inside",
            hovertemplate=f"%{{y}} · {column.replace(' %', '')}: %{{x:.1f}}%<extra></extra>",
        )
    fig.update_layout(
        title="Directional probability by forecast horizon", barmode="stack", height=330,
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        xaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="#e2e8f0"),
        yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=25, r=20, t=65, b=40), font=dict(color="#334155", size=12),
    )
    return fig


def prediction_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": " / 100"},
        title={"text": "Source-aware ensemble score"},
        gauge={
            "axis": {"range": [0, 100]}, "bar": {"color": "#159487"},
            "steps": [
                {"range": [0, 40], "color": "#f6e8df"},
                {"range": [40, 60], "color": "#eef2f6"},
                {"range": [60, 100], "color": "#d8eee9"},
            ],
            "threshold": {"line": {"color": "#172b4d", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(height=300, margin=dict(l=35, r=35, t=60, b=25),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155"))
    return fig


def short_term_risk_table(a: Analysis, prices: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    features = price_indicators(prices)
    recent = features.tail(60)
    atr_pct = float(a.metrics.get("ATR %", np.nan))
    gaps = (recent["Open"] / recent["Close"].shift(1) - 1).abs().dropna()
    gap95 = float(gaps.quantile(0.95) * 100) if not gaps.empty else float("nan")
    dollar_volume = float((recent["Close"] * recent["Volume"]).tail(20).mean())

    volatility_level = "High" if atr_pct >= 4 else "Medium" if atr_pct >= 2 else "Low"
    gap_level = "High" if gap95 >= 4 else "Medium" if gap95 >= 2 else "Low"
    liquidity_level = "High" if dollar_volume < 10_000_000 else "Medium" if dollar_volume < 50_000_000 else "Low"
    market_level = "High" if a.components["macro"].score < 40 else "Medium" if a.components["macro"].score < 55 else "Low"

    earnings_value = metadata.get("earningsTimestamp") or metadata.get("earningsTimestampStart")
    earnings_level, earnings_detail = "Unscored", "Earnings date unavailable; verify independently."
    try:
        earnings_date = datetime.fromtimestamp(float(earnings_value), tz=timezone.utc)
        days_to_earnings = (earnings_date.date() - datetime.now(timezone.utc).date()).days
        if 0 <= days_to_earnings <= 14:
            earnings_level = "High"
        elif 14 < days_to_earnings <= 45:
            earnings_level = "Medium"
        elif days_to_earnings > 45:
            earnings_level = "Low"
        earnings_detail = f"Reported earnings date: {earnings_date.date().isoformat()} ({days_to_earnings} days)."
    except (TypeError, ValueError, OSError):
        pass

    return pd.DataFrame([
        {"Risk": "Event risk", "Level": "Unscored", "Evidence": "Live news and event feed is not connected."},
        {"Risk": "Earnings risk", "Level": earnings_level, "Evidence": earnings_detail},
        {"Risk": "Gap risk", "Level": gap_level, "Evidence": f"60-session 95th percentile absolute overnight gap: {gap95:.2f}%."},
        {"Risk": "Liquidity risk", "Level": liquidity_level, "Evidence": f"Average 20-day dollar volume: ${dollar_volume:,.0f}."},
        {"Risk": "Volatility risk", "Level": volatility_level, "Evidence": f"ATR(14) is {atr_pct:.2f}% of price."},
        {"Risk": "Market risk", "Level": market_level, "Evidence": a.components["macro"].reasons[0]},
    ])


def thesis(a: Analysis, meta: dict[str, Any]) -> str:
    md_price = lambda value: fmt(value, "price").replace("$", r"\$")
    bull = "; ".join(a.components["trend"].reasons[:2] + a.components["relative_strength"].reasons[:1])
    bear = "; ".join(a.components["risk_reward"].reasons[:1] + ([a.components["momentum"].reasons[0]] if a.components["momentum"].score < 50 else []))
    sector = meta.get("sector", "sector unavailable")
    return f"""### Executive summary
{a.ticker} is classified **{a.classification}** with an AI score of **{a.overall_score:.1f}/100** and **{a.confidence:.1f}% model confidence**. The 20-day distribution implies {a.bull_probability:.1f}% bullish, {a.bear_probability:.1f}% bearish and {a.sideways_probability:.1f}% sideways probability; these are estimates, not guarantees.

### Regime and multi-timeframe trend
The security is in **{a.trend_stage}**. The trend vote is {a.components['trend'].score:.1f}/100 because {bull}.

### Technical and institutional evidence
Momentum scores {a.components['momentum'].score:.1f}, volume {a.components['volume'].score:.1f}, and relative strength {a.components['relative_strength'].score:.1f}. Institutional-flow conclusions are withheld because dark-pool, options-flow and dealer-positioning feeds are not connected.

### Fundamental, macro and sector context
Fundamentals score {a.components['fundamental'].score:.1f} at {a.components['fundamental'].coverage:.0%} field coverage. The company maps to **{sector}**. Macro scores {a.components['macro'].score:.1f}; the current vote uses the SPY long-term trend and is intentionally incomplete.

### Bull case vs. bear case
**Bull case:** {bull}.  
**Bear case:** {bear or 'A reversal in trend breadth or a break of the ATR-defined invalidation level would negate the setup.'}

### Trade plan and risk
Entry reference: {md_price(a.price)} · Stop: {md_price(a.stop_loss)} · Targets: {md_price(a.target_1)}, {md_price(a.target_2)}, {md_price(a.target_3)}. Suggested size is **{a.recommended_position_size:.2f}% of portfolio**, capped by configured portfolio and per-trade risk. {a.exit_condition.replace('$', r'\$')}.
"""


def export_pdf(a: Analysis, meta: dict[str, Any]) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    y = 750
    pdf.setTitle(f"{a.ticker} Axiom Research Note")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(45, y, f"Axiom AI Research — {a.ticker}")
    y -= 30
    pdf.setFont("Helvetica", 9)
    lines = thesis(a, meta).replace("###", "").replace("**", "").splitlines()
    for line in lines:
        words, current = line.split(), ""
        for word in words:
            if pdf.stringWidth(current + " " + word, "Helvetica", 9) > 520:
                pdf.drawString(45, y, current)
                y -= 13
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            pdf.drawString(45, y, current)
            y -= 13
        if y < 45:
            pdf.showPage(); pdf.setFont("Helvetica", 9); y = 750
    pdf.save()
    return buffer.getvalue()


def scanner_page(tickers: list[str], max_position: float, risk_per_trade: float) -> None:
    header("Multi-Stock AI Scanner", "Explainable ranking across trend, momentum, volume, relative strength, risk/reward, fundamentals, and macro regime.")
    st.markdown('<span class="pill">Weighted voting · probability-based · source-aware</span>', unsafe_allow_html=True)
    analyses: list[Analysis] = []
    failures: list[str] = []
    progress = st.progress(0, text="Loading market evidence…")
    for i, ticker in enumerate(tickers):
        try:
            result, _ = cached_analysis(ticker, max_position, risk_per_trade)
            analyses.append(result)
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
        progress.progress((i + 1) / max(len(tickers), 1), text=f"Analyzed {i + 1}/{len(tickers)}")
    progress.empty()
    if not analyses:
        st.error("No source-backed analyses could be produced. " + (failures[0] if failures else "Add at least one ticker."))
        return
    analyses.sort(key=lambda x: (x.overall_score, x.confidence), reverse=True)
    top = analyses[0]
    freshness(top.as_of)
    cols = st.columns(6)
    cols[0].metric("Universe size", len(analyses))
    cols[1].metric("Top-ranked ticker", top.ticker, top.classification)
    cols[2].metric("AI score", fmt(top.overall_score), "of 100")
    cols[3].metric("Bull probability", fmt(top.bull_probability, "pct"))
    cols[4].metric("AI confidence", fmt(top.confidence, "pct"))
    cols[5].metric("Position size", fmt(top.recommended_position_size, "pct"))

    records = []
    for rank, a in enumerate(analyses, 1):
        records.append({"Rank": rank, "Ticker": a.ticker, "Classification": a.classification,
                        "AI Score": a.overall_score, "Bull %": a.bull_probability, "Bear %": a.bear_probability,
                        "Confidence %": a.confidence, "Risk Score": a.risk_score, "Expected 20D %": a.expected_return_20d,
                        "Drawdown est. %": a.expected_drawdown, "R/R": a.reward_risk,
                        "Trend": a.components["trend"].score, "Momentum": a.components["momentum"].score,
                        "Volume": a.components["volume"].score, "RS": a.components["relative_strength"].score,
                        "Fundamental": a.components["fundamental"].score, "Position %": a.recommended_position_size,
                        "Stage": a.trend_stage, "Stop": a.stop_loss, "Current Price": a.price,
                        "Target-1": a.target_1, "Target-2": a.target_2})
    table = pd.DataFrame(records)
    st.subheader("Institutional ranking")
    st.dataframe(table, column_config={
        "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.1f"),
        "Bull %": st.column_config.ProgressColumn("Bull %", min_value=0, max_value=100, format="%.1f%%"),
        "Confidence %": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100, format="%.1f%%"),
        "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
        "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
        "Target-1": st.column_config.NumberColumn("Target-1", format="$%.2f"),
        "Target-2": st.column_config.NumberColumn("Target-2", format="$%.2f"),
    }, use_container_width=True, hide_index=True, height=min(680, 42 + 35 * len(table)))
    export_cols = st.columns([1, 1, 5])
    export_cols[0].download_button("Export CSV", table.to_csv(index=False).encode(), "axiom_scanner.csv", "text/csv")
    workbook = excel_bytes(table)
    if workbook:
        export_cols[1].download_button("Export Excel", workbook, "axiom_scanner.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if failures:
        with st.expander(f"Data exceptions ({len(failures)})"):
            st.write("\n".join(failures))

    selected = st.selectbox("Explain a ranked security", [a.ticker for a in analyses], key="scanner_security")
    a = next(item for item in analyses if item.ticker == selected)
    st.subheader(f"Why {a.ticker} scores {a.overall_score:.1f}")
    score_cols = st.columns(len(a.components))
    for col, (name, component) in zip(score_cols, a.components.items()):
        col.metric(name.replace("_", " ").title(), fmt(component.score), f"{component.coverage:.0%} coverage")
    profile_col, probability_col = st.columns([1.2, 1])
    with profile_col:
        st.plotly_chart(factor_profile_chart(a), use_container_width=True, key="scanner_factor_profile")
        st.caption("Scores are comparable 0–100 weighted votes. The dotted 50 line is neutral; weights and data coverage also affect the overall score.")
    with probability_col:
        probability_fig = probability_chart(a)
        probability_fig.update_layout(title="Modeled probability · direction by horizon", height=345)
        st.plotly_chart(probability_fig, use_container_width=True, key="scanner_probability")
        st.markdown(f"**Risk plan:** stop {fmt(a.stop_loss, 'price')} · trail {fmt(a.trailing_stop, 'price')} · targets {fmt(a.target_1, 'price')} / {fmt(a.target_2, 'price')} / {fmt(a.target_3, 'price')}")

    st.subheader("Factor evidence and reference chart")
    factor_names = list(a.components)
    selected_factor = st.selectbox(
        "Detailed reference graph",
        factor_names,
        format_func=lambda name: name.replace("_", " ").title(),
        key="scanner_reference_factor",
    )
    selected_component = a.components[selected_factor]
    st.caption(f"Current {selected_factor.replace('_', ' ')} score: {selected_component.score:.1f} / 100")
    for reason in selected_component.reasons:
        st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)

    _, selected_prices = cached_analysis(selected, max_position, risk_per_trade)
    selected_features = price_indicators(selected_prices)
    try:
        benchmark_features = price_indicators(cached_prices("SPY"))
    except Exception:
        benchmark_features = None
    try:
        reference_fig, reference_caption = component_reference_chart(
            selected_factor, a, selected_features, benchmark_features
        )
        # Keep identities bounded to the seven factors while avoiding reuse across
        # incompatible Plotly trace types (line, bar, scatter and gauge).
        st.plotly_chart(
            reference_fig,
            use_container_width=True,
            key=f"scanner_reference_chart_{selected_factor}",
        )
        st.caption(reference_caption)
    except (KeyError, IndexError, TypeError, ValueError):
        st.warning(
            f"The {selected_factor.replace('_', ' ')} reference graph is temporarily unavailable "
            "for this ticker because its source history is incomplete. The factor score and evidence remain available."
        )

    with st.expander("View evidence for all factors"):
        for name, component in a.components.items():
            st.markdown(f"**{name.replace('_', ' ').title()} — {component.score:.1f}**")
            for reason in component.reasons:
                st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)


def matrix_page(tickers: list[str], max_position: float, risk_per_trade: float) -> None:
    header("Top Prediction Matrix", "Cross-sectional opportunity map built from comparable, reconciled score definitions.")
    rows, failures = [], []
    for ticker in tickers:
        try:
            a, _ = cached_analysis(ticker, max_position, risk_per_trade)
            meta = cached_metadata(ticker)
            rows.append({"Ticker": ticker, "Company": meta.get("shortName", ticker), "Sector": meta.get("sector", "N/A"),
                         "Trend Stage": a.trend_stage, "Overall": a.overall_score, "Expected Return %": a.expected_return_20d,
                         "Expected Risk %": abs(a.expected_drawdown), "Bull %": a.bull_probability, "Bear %": a.bear_probability,
                         "Confidence %": a.confidence, "Entry Quality": a.components["risk_reward"].score,
                         "R/R": a.reward_risk, "Volume": a.components["volume"].score,
                         "Momentum": a.components["momentum"].score, "RS vs SPY": a.components["relative_strength"].score,
                         "Fundamental": a.components["fundamental"].score, "Macro": a.components["macro"].score,
                         "P(+5%) 20D": max(0, min(100, a.probabilities["1 month"]["bull"] - 12)),
                         "P(+10%) 60D": max(0, min(100, a.probabilities["3 months"]["bull"] - 18)),
                         "Classification": a.classification})
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
    if not rows:
        st.error("No matrix data available. " + (failures[0] if failures else "")); return
    matrix = pd.DataFrame(rows)
    mode = st.selectbox("Rank by", ["Overall", "Bull %", "Expected Return %", "Confidence %", "Momentum", "RS vs SPY", "Entry Quality", "Fundamental"])
    matrix = matrix.sort_values(mode, ascending=False).reset_index(drop=True)
    matrix.insert(0, "Rank", range(1, len(matrix) + 1))
    st.subheader("Prediction matrix")
    st.dataframe(matrix, column_config={
        "Overall": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f"),
        "Bull %": st.column_config.ProgressColumn("Bull %", min_value=0, max_value=100, format="%.1f%%"),
        "Confidence %": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100, format="%.1f%%"),
        "Momentum": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.1f"),
        "RS vs SPY": st.column_config.ProgressColumn("RS vs SPY", min_value=0, max_value=100, format="%.1f"),
    }, use_container_width=True, hide_index=True, height=min(650, 42 + 35 * len(matrix)))
    workbook = excel_bytes(matrix)
    if workbook:
        st.download_button("Export matrix (Excel)", workbook, "axiom_prediction_matrix.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Opportunity map")
    st.caption("Upper-left is preferable: higher modeled return with lower expected drawdown. Bubble size represents model confidence; color represents overall score.")
    label_positions = ["top center", "bottom center", "middle right", "middle left",
                       "top right", "top left", "bottom right", "bottom left"]
    text_positions = [label_positions[i % len(label_positions)] for i in range(len(matrix))]
    marker_sizes = np.clip(matrix["Confidence %"] * 0.45, 18, 34)
    fig = go.Figure(go.Scatter(
        x=matrix["Expected Risk %"], y=matrix["Expected Return %"], mode="markers+text",
        text=matrix.Ticker, textposition=text_positions, cliponaxis=False,
        customdata=np.stack([matrix["Overall"], matrix["Confidence %"], matrix["Bull %"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>Expected drawdown: %{x:.1f}%<br>Expected 20D return: %{y:.1f}%"
                       "<br>Overall score: %{customdata[0]:.1f}<br>Confidence: %{customdata[1]:.1f}%"
                       "<br>Bull probability: %{customdata[2]:.1f}%<extra></extra>"),
        marker=dict(size=marker_sizes, color=matrix["Overall"], colorscale="RdYlGn", cmin=0, cmax=100,
                    opacity=.88, line=dict(color="#ffffff", width=2), showscale=True,
                    colorbar=dict(title="Overall<br>score", thickness=16, len=.72)),
        textfont=dict(size=13, color="#334155"),
    ))
    fig.add_vline(x=float(matrix["Expected Risk %"].median()), line_dash="dot", line_color="#94a3b8", opacity=.65)
    fig.add_hline(y=0, line_width=1.5, line_color="#64748b", opacity=.65)
    fig.add_annotation(xref="paper", yref="paper", x=.01, y=.98, text="Higher return · Lower risk",
                       showarrow=False, font=dict(size=12, color="#087f72"), bgcolor="#e3f4ef", borderpad=6)
    fig.update_layout(
        height=610, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        margin=dict(l=70, r=80, t=30, b=65), hovermode="closest",
        xaxis=dict(title="Expected drawdown magnitude (%) — lower is better", rangemode="tozero",
                   gridcolor="#e2e8f0", zeroline=False),
        yaxis=dict(title="Expected 20-day return (%) — higher is better", gridcolor="#e2e8f0", zeroline=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="warningbox">Institutional buying, options flow, dark pools, live news, and analyst targets are omitted until licensed sources are connected.</div>', unsafe_allow_html=True)


def terminal_page(tickers: list[str], max_position: float, risk_per_trade: float) -> None:
    header("Technical & Investment Terminal", "One evidence chain from price structure to portfolio-aware action.")
    ticker = st.selectbox("Security", tickers, key="terminal_ticker")
    try:
        a, prices = cached_analysis(ticker, max_position, risk_per_trade)
        meta = cached_metadata(ticker)
    except Exception as exc:
        st.error(str(exc)); return
    freshness(a.as_of)
    primary_metrics = st.columns(4)
    for col, label, value in zip(primary_metrics,
        ["Current price", "Overall AI score", "Bull probability", "AI confidence"],
        [fmt(a.price, "price"), f"{a.overall_score:.1f} / 100", fmt(a.bull_probability, "pct"),
         fmt(a.confidence, "pct")]):
        col.metric(label, value)
    secondary_metrics = st.columns(3)
    for col, label, value in zip(secondary_metrics,
        ["Risk score", "Reward / risk", "Recommended position"],
        [f"{a.risk_score:.1f} / 100", f"{a.reward_risk:.1f} : 1", fmt(a.recommended_position_size, "pct")]):
        col.metric(label, value)
    tabs = st.tabs(["Investment Thesis", "Trend & Structure", "Momentum & Volume", "Fundamentals & Macro", "Risk Plan", "Data Coverage"])
    with tabs[0]:
        st.markdown(thesis(a, meta))
        pdf = export_pdf(a, meta)
        if pdf:
            st.download_button("Download research note (PDF)", pdf, f"{ticker}_axiom_research.pdf", "application/pdf")
    with tabs[1]:
        st.plotly_chart(price_chart(prices, f"{ticker} daily structure"), use_container_width=True)
        st.write(f"**Weinstein stage:** {a.trend_stage}")
        for reason in a.components["trend"].reasons:
            st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)
    with tabs[2]:
        ind = price_indicators(prices).tail(120)
        fig = go.Figure()
        fig.add_scatter(x=ind.index, y=ind.RSI14, name="RSI(14)", line_color="#33d6c5")
        fig.add_hline(y=70, line_dash="dot", line_color="#fb7185"); fig.add_hline(y=30, line_dash="dot", line_color="#60a5fa")
        fig.update_layout(height=300, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI(14)", fmt(a.metrics["RSI14"])); c2.metric("ADX(14)", fmt(a.metrics["ADX14"]))
        c3.metric("Relative volume", f"{a.metrics['Relative Volume']:.2f}x"); c4.metric("Volatility", fmt(a.metrics["Annualized Volatility %"], "pct"))
    with tabs[3]:
        st.write(f"**Company:** {meta.get('longName', ticker)} · **Sector:** {meta.get('sector', 'N/A')} · **Industry:** {meta.get('industry', 'N/A')}")
        for name in ("fundamental", "macro", "relative_strength"):
            component = a.components[name]
            st.subheader(f"{name.replace('_', ' ').title()} — {component.score:.1f}")
            st.caption(f"Field coverage: {component.coverage:.0%}")
            for reason in component.reasons:
                st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)
    with tabs[4]:
        st.plotly_chart(probability_chart(a), use_container_width=True)
        risk = pd.DataFrame({"Level": ["Reference", "Stop", "Trailing stop", "Target 1", "Target 2", "Target 3"],
                             "Price": [a.price, a.stop_loss, a.trailing_stop, a.target_1, a.target_2, a.target_3]})
        st.dataframe(risk, use_container_width=True, hide_index=True)
        st.info(f"Kelly: {a.kelly_fraction:.2f}% · Final suggested position: {a.recommended_position_size:.2f}% · Expected 20D drawdown: {a.expected_drawdown:.2f}%")
    with tabs[5]:
        source_rows = [{"Dataset": key.replace("_", " ").title(), "Status": "Connected" if value else "Not connected", "Source": value or "Requires licensed/API source"}
                       for key, value in DATA_PROVIDERS.items()]
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
        st.warning("Unavailable datasets are never replaced with synthetic institutional-flow claims.")


def short_term_prediction_page(tickers: list[str], max_position: float, risk_per_trade: float) -> None:
    header(
        "AI Short-Term Trend Prediction Center",
        "Probability distributions for 1 day, 3 days, 1 week, and 2 weeks—supported by connected evidence, never certainty.",
    )
    ticker = st.selectbox("Security", tickers, key="short_term_ticker")
    try:
        a, prices = cached_analysis(ticker, max_position, risk_per_trade)
        metadata = cached_metadata(ticker)
    except Exception as exc:
        st.error(f"A source-backed short-term forecast could not be produced: {exc}")
        return
    freshness(a.as_of)

    with st.expander("Configure weighted ensemble", expanded=False):
        st.caption(
            "Weights are normalized automatically. Modules without connected evidence cast a neutral vote and reduce effective coverage."
        )
        weight_cols = st.columns(4)
        weights: dict[str, float] = {}
        for index, (module, default) in enumerate(SHORT_TERM_WEIGHTS.items()):
            weights[module] = weight_cols[index % 4].number_input(
                module, min_value=0.0, max_value=100.0, value=default, step=1.0,
                key=f"short_weight_{module.lower().replace(' ', '_').replace('/', '_')}",
            )
        total_weight = sum(weights.values())
        if total_weight <= 0:
            st.error("At least one ensemble module must have a positive weight.")
            return
        st.caption(f"Configured total: {total_weight:.0f}; normalized to 100% for calculation.")

    technical_score = float(np.mean([
        a.components["trend"].score,
        a.components["momentum"].score,
        a.components["relative_strength"].score,
    ]))
    technical_coverage = float(np.mean([
        a.components["trend"].coverage,
        a.components["momentum"].coverage,
        a.components["relative_strength"].coverage,
    ]))
    module_specs = {
        "Technical Analysis": (technical_score, technical_coverage, "Connected price/technical evidence"),
        "Volume & Price Action": (a.components["volume"].score, a.components["volume"].coverage, "Connected price/volume evidence"),
        "Options Flow": (50.0, 0.0, "Not connected—no options-flow vote"),
        "News & Sentiment": (50.0, 0.0, "Not connected—no live news/social vote"),
        "Fundamental Momentum": (a.components["fundamental"].score, a.components["fundamental"].coverage, "Metadata coverage varies"),
        "Sector Rotation": (50.0, 0.0, "Sector-relative feed not connected"),
        "Macro Economy": (a.components["macro"].score, a.components["macro"].coverage, "SPY trend proxy only"),
        "AI / Statistical Model": (a.overall_score, 0.35, "Statistical proxy—not a trained ML model ensemble"),
    }
    module_rows: list[dict[str, Any]] = []
    ensemble_score = 0.0
    effective_coverage = 0.0
    for module, raw_weight in weights.items():
        score, coverage, status = module_specs[module]
        normalized_weight = raw_weight / total_weight
        effective_vote = 50 + (score - 50) * coverage
        contribution = effective_vote * normalized_weight
        ensemble_score += contribution
        effective_coverage += coverage * normalized_weight
        module_rows.append({
            "Module": module,
            "Configured Weight %": round(normalized_weight * 100, 1),
            "Raw Score": round(score, 1),
            "Coverage %": round(coverage * 100, 1),
            "Coverage-adjusted Vote": round(effective_vote, 1),
            "Weighted Contribution": round(contribution, 2),
            "Status": status,
        })
    module_table = pd.DataFrame(module_rows)
    forecast = short_term_probability_table(a, prices, ensemble_score, effective_coverage)
    one_day = forecast.iloc[0]
    two_weeks = forecast.iloc[-1]

    if ensemble_score >= 65:
        bias = "Strong Bullish"
    elif ensemble_score >= 55:
        bias = "Bullish"
    elif ensemble_score <= 35:
        bias = "Bearish"
    elif ensemble_score <= 45:
        bias = "Cautious / Bearish"
    else:
        bias = "Neutral"

    metric_grid([
        ("Current price", fmt(a.price, "price")),
        ("Overall bias", bias),
        ("1-day bull", fmt(one_day["Bull %"], "pct")),
        ("1-day confidence", fmt(one_day["Confidence %"], "pct")),
        ("2-week expected return", f"{two_weeks['Expected Return %']:+.2f}%"),
        ("Connected evidence", fmt(effective_coverage * 100, "pct")),
    ], 3)

    gauge_col, probability_col = st.columns([0.8, 1.4])
    with gauge_col:
        st.plotly_chart(prediction_gauge(ensemble_score), use_container_width=True, key="short_term_gauge")
    with probability_col:
        st.plotly_chart(short_term_probability_chart(forecast), use_container_width=True, key="short_term_probabilities")
    st.caption(
        "Bull, neutral, and bear probabilities are a volatility-based statistical distribution conditioned on connected factor evidence. "
        "They are estimates, not guarantees."
    )

    forecast_tab, evidence_tab, risk_tab, report_tab = st.tabs([
        "Probability Forecast", "Evidence & Module Audit", "Risk Dashboard", "AI Report & Learning",
    ])
    with forecast_tab:
        st.subheader("Probability matrix")
        st.dataframe(forecast, column_config={
            "Bull %": st.column_config.ProgressColumn("Bull %", min_value=0, max_value=100, format="%.1f%%"),
            "Neutral %": st.column_config.ProgressColumn("Neutral %", min_value=0, max_value=100, format="%.1f%%"),
            "Bear %": st.column_config.ProgressColumn("Bear %", min_value=0, max_value=100, format="%.1f%%"),
            "Confidence %": st.column_config.ProgressColumn("Confidence %", min_value=0, max_value=100, format="%.1f%%"),
            "Expected High": st.column_config.NumberColumn("Expected High", format="$%.2f"),
            "Expected Low": st.column_config.NumberColumn("Expected Low", format="$%.2f"),
        }, use_container_width=True, hide_index=True)
        st.subheader("Trade plan")
        metric_grid([
            ("Stop", fmt(a.stop_loss, "price")),
            ("Current Price", fmt(a.price, "price")),
            ("Target-1", fmt(a.target_1, "price")),
            ("Target-2", fmt(a.target_2, "price")),
            ("Target-3", fmt(a.target_3, "price")),
            ("Trailing Stop", fmt(a.trailing_stop, "price")),
        ], 3)
        st.caption(
            f"Reward/risk: {a.reward_risk:.2f}:1 · Kelly estimate: {a.kelly_fraction:.2f}% · "
            f"Suggested position after configured caps: {a.recommended_position_size:.2f}%"
        )

    with evidence_tab:
        metric_grid([
            ("Technical score", f"{technical_score:.1f} / 100"),
            ("Volume score", f"{a.components['volume'].score:.1f} / 100"),
            ("News sentiment", "Not connected"),
            ("Institutional activity", "Not connected"),
        ], 2)
        st.subheader("Weighted ensemble audit")
        st.dataframe(module_table, column_config={
            "Coverage %": st.column_config.ProgressColumn("Coverage %", min_value=0, max_value=100, format="%.1f%%"),
            "Coverage-adjusted Vote": st.column_config.ProgressColumn(
                "Coverage-adjusted Vote", min_value=0, max_value=100, format="%.1f"
            ),
        }, use_container_width=True, hide_index=True)
        st.subheader("AI explanation")
        for name in ("trend", "momentum", "volume", "relative_strength", "fundamental", "macro"):
            component = a.components[name]
            st.markdown(f"**{name.replace('_', ' ').title()} — {component.score:.1f}/100**")
            for reason in component.reasons:
                st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)
        st.warning(
            "Options flow, dark pools, dealer gamma, social sentiment, live news, sector rotation, and trained ML models are not connected. "
            "Their absence lowers coverage rather than generating synthetic evidence."
        )

    with risk_tab:
        risk_frame = short_term_risk_table(a, prices, metadata)
        st.subheader("Risk dashboard")
        st.dataframe(risk_frame, use_container_width=True, hide_index=True)
        st.subheader("Price and volatility context")
        st.plotly_chart(price_chart(prices.tail(252), f"{ticker} · latest 252 sessions"),
                        use_container_width=True, key="short_term_price_context")
        st.error(
            "Expected range and drawdown are model estimates, not worst-case limits. Earnings, news, halts, and overnight gaps can exceed them."
        )

    with report_tab:
        strongest = sorted(a.components.items(), key=lambda item: item[1].score, reverse=True)[:3]
        weakest = sorted(a.components.items(), key=lambda item: item[1].score)[:3]
        with st.expander("Executive summary", expanded=True):
            st.markdown(
                f"**{ticker}** has a source-aware short-term ensemble score of **{ensemble_score:.1f}/100** "
                f"with an overall bias of **{bias}**. The 1-day distribution is "
                f"**{one_day['Bull %']:.1f}% bull / {one_day['Neutral %']:.1f}% neutral / {one_day['Bear %']:.1f}% bear**. "
                f"Only **{effective_coverage:.0%}** of configured weighted evidence is currently connected."
            )
        with st.expander("Bull case and bear case"):
            st.markdown("**Bull case**")
            for name, component in strongest:
                st.markdown(f"- {name.replace('_', ' ').title()}: {component.reasons[0]}")
            st.markdown("**Bear case / constraints**")
            for name, component in weakest:
                st.markdown(f"- {name.replace('_', ' ').title()}: {component.reasons[0]}")
        with st.expander("Catalysts, institutional activity, and news"):
            st.markdown(
                "- Earnings timing is shown in the Risk Dashboard when the metadata source provides it.\n"
                "- Federal Reserve, CPI/PPI, employment, GDP, geopolitical, analyst-change, and product-launch feeds are not connected.\n"
                "- Dark-pool, options whale, insider, ETF-flow, and dealer-positioning feeds are not connected.\n"
                "- Verify all catalysts independently before trading."
            )
        with st.expander("Continuous learning and prediction storage"):
            st.markdown(
                "Predictions can be stored for later outcome labeling in **Accuracy Lab**. Automated horizon closing, calibration analysis, "
                "rolling back-tests, feature reweighting, and model retraining require a persistent scheduled pipeline and are not claimed here."
            )
            if st.button("Store all four prediction horizons", key="store_short_term_predictions"):
                store = SignalStore()
                inserted = 0
                signal_date = datetime.now(timezone.utc).date().isoformat()
                for _, row in forecast.iterrows():
                    direction = max(("Bullish", row["Bull %"]), ("Sideways", row["Neutral %"]),
                                    ("Bearish", row["Bear %"]), key=lambda item: item[1])[0]
                    target = a.price * (1 + float(row["Expected Return %"]) / 100)
                    if store.record({
                        "signal_date": signal_date, "ticker": ticker, "strategy": "Short-Term Ensemble",
                        "prediction": direction, "entry": a.price, "target": target, "stop": a.stop_loss,
                        "confidence": float(row["Confidence %"]), "horizon_days": int(row["Days"]),
                    }):
                        inserted += 1
                st.success(f"Stored {inserted} new horizon predictions. Existing same-day horizons were not duplicated.")

        feature_status = pd.DataFrame([
            {"Evidence family": "Price/technical", "Connected": "Yes", "Coverage": "EMA 9/20/50/60/100/200, SMA200, MACD, RSI, ADX, ATR, RVOL, returns, volatility, 20-day highs/lows"},
            {"Evidence family": "Advanced technical", "Connected": "Partial", "Coverage": "Stochastic RSI, VWAP/anchored VWAP, SuperTrend, Ichimoku, Bollinger/Keltner/Donchian, volume profile, BOS/CHOCH, Elliott, Wyckoff, Darvas, pivots, Fibonacci not yet implemented"},
            {"Evidence family": "Fundamentals", "Connected": "Partial", "Coverage": "Yahoo company metadata; availability varies"},
            {"Evidence family": "Macro", "Connected": "Partial", "Coverage": "SPY 200-day trend proxy only"},
            {"Evidence family": "News/social", "Connected": "No", "Coverage": "No trusted live feed configured"},
            {"Evidence family": "Institutional/options", "Connected": "No", "Coverage": "No licensed flow, dark-pool, gamma, insider, ETF, or 13F pipeline"},
            {"Evidence family": "Machine learning", "Connected": "No", "Coverage": "Statistical proxy only; no trained XGBoost/LSTM/Transformer ensemble"},
        ])
        st.subheader("Source and feature coverage")
        st.dataframe(feature_status, use_container_width=True, hide_index=True)


def accuracy_page() -> None:
    header("Signal Accuracy & Feedback Loop", "Immutable predictions become labeled outcomes; performance is evaluated by strategy and regime, not headline accuracy alone.")
    store = SignalStore()
    frame = store.frame()
    metrics = performance_metrics(frame)
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, fmt(value, "pct" if name == "Win Rate" else "number"))
    if frame.empty:
        st.info("No signals recorded yet. Record an analysis from the form below; outcomes can be entered after the horizon closes.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        closed = frame.dropna(subset=["actual_result"])
        if not closed.empty:
            by_strategy = closed.groupby("strategy").agg(Signals=("id", "count"), Win_Rate=("actual_result", lambda x: (x > 0).mean() * 100),
                                                          Expectancy=("actual_result", "mean"), Avg_Winner=("actual_result", lambda x: x[x > 0].mean()),
                                                          Avg_Loser=("actual_result", lambda x: x[x <= 0].mean())).reset_index()
            st.subheader("Strategy attribution")
            st.dataframe(by_strategy, use_container_width=True, hide_index=True)
    with st.expander("Record a new model prediction"):
        with st.form("record_signal"):
            c1, c2, c3 = st.columns(3)
            ticker = c1.text_input("Ticker", "NVDA").upper(); strategy = c2.selectbox("Strategy", ["AI Ensemble", "Breakout", "Pullback", "Trend", "Mean Reversion"])
            horizon = c3.number_input("Horizon (trading days)", 1, 252, 20)
            c4, c5, c6, c7 = st.columns(4)
            entry = c4.number_input("Entry", min_value=0.01, value=100.0); target = c5.number_input("Target", min_value=0.01, value=110.0)
            stop = c6.number_input("Stop", min_value=0.01, value=95.0); confidence = c7.slider("Confidence", 0, 100, 60)
            prediction = st.selectbox("Prediction", ["Bullish", "Bearish", "Sideways"])
            submitted = st.form_submit_button("Record immutable signal")
            if submitted:
                inserted = store.record({"signal_date": datetime.now(timezone.utc).date().isoformat(), "ticker": ticker,
                                         "strategy": strategy, "prediction": prediction, "entry": entry, "target": target,
                                         "stop": stop, "confidence": confidence, "horizon_days": horizon})
                st.success("Signal recorded." if inserted else "That signal already exists for this date and horizon.")
                st.rerun()
    st.caption("Retraining is deliberately not automatic in this MVP: a scheduled, walk-forward validated training pipeline should promote models only after out-of-sample and transaction-cost checks.")


def reference_page() -> None:
    header(
        "Stock Trading Reference Guide",
        "Plain-language definitions for the scanner, technical analysis, risk management, and responsible trade planning.",
    )
    st.info(
        "Start with trend, then confirm momentum and volume, define the invalidation level, and size the position from risk. "
        "No single indicator should be used as a standalone buy or sell signal."
    )

    st.subheader("Beginner learning path")
    learning_cols = st.columns(4)
    learning_cols[0].markdown("**1 · Context**\n\nIdentify the market and stock trend.")
    learning_cols[1].markdown("**2 · Confirmation**\n\nCheck momentum, volume, and relative strength.")
    learning_cols[2].markdown("**3 · Risk plan**\n\nSet entry, stop, targets, and position size.")
    learning_cols[3].markdown("**4 · Review**\n\nRecord the result and improve the process.")

    trend_tab, indicator_tab, scanner_tab, risk_tab, fundamental_tab = st.tabs([
        "Trend stages", "Indicators", "Scanner scores", "Risk & execution", "Fundamentals & data",
    ])

    with trend_tab:
        st.subheader("Weinstein-style trend stages")
        stage_rows = [
            {"Stage": "Stage 1 — Basing", "Plain meaning": "Sideways stabilization after weakness; a new uptrend is not confirmed.",
             "Exact scanner rule": "Price is below EMA200, but Price < EMA50 < EMA200 is not fully aligned."},
            {"Stage": "Stage 2 — Advancing", "Plain meaning": "Established long-term uptrend; normally the most constructive stage for long positions.",
             "Exact scanner rule": "Price > EMA50 > EMA200 and EMA200 is higher than it was 20 sessions ago."},
            {"Stage": "Stage 3 — Distribution", "Plain meaning": "The advance is losing strength and may be forming a top.",
             "Exact scanner rule": "Reported as Stage 1/3 when price is at or above EMA200 but full Stage 2 alignment is absent."},
            {"Stage": "Stage 4 — Declining", "Plain meaning": "Established downtrend; long positions face elevated trend risk.",
             "Exact scanner rule": "Price < EMA50 < EMA200."},
        ]
        st.dataframe(pd.DataFrame(stage_rows), use_container_width=True, hide_index=True)
        st.warning(
            "Stage 1/3 is intentionally ambiguous: moving averages alone cannot reliably distinguish accumulation from distribution. "
            "Use price structure, volume, momentum, and relative strength for confirmation."
        )
        with st.expander("Price structure vocabulary"):
            st.markdown(
                "- **Uptrend:** a sequence of higher highs and higher lows.\n"
                "- **Downtrend:** a sequence of lower highs and lower lows.\n"
                "- **Support:** an area where buying previously absorbed selling; it can fail.\n"
                "- **Resistance:** an area where selling previously absorbed buying; it can break.\n"
                "- **Breakout:** price moves beyond a defined range or prior high, preferably with confirmation.\n"
                "- **Pullback:** a temporary move against the prevailing trend.\n"
                "- **Consolidation/base:** price trades in a range while supply and demand rebalance."
            )

    with indicator_tab:
        indicator_rows = [
            {"Indicator": "EMA20 / EMA50 / EMA60 / EMA200", "What it measures": "Smoothed price trend over short to long horizons.",
             "How to read": "Price and faster averages above slower rising averages are constructive; crossings can lag."},
            {"Indicator": "RSI(14)", "What it measures": "Momentum on a 0–100 scale.",
             "How to read": "Above 70 is conventionally overbought, near 50 neutral, below 30 oversold; extremes do not guarantee reversal."},
            {"Indicator": "MACD", "What it measures": "Difference between 12- and 26-period EMAs, compared with a 9-period signal line.",
             "How to read": "MACD above its signal supports positive momentum; it is a lagging confirmation tool."},
            {"Indicator": "ADX(14)", "What it measures": "Trend strength, not trend direction.",
             "How to read": "Below 20 often weak/ranging, 20–25 developing, above 25 stronger; direction comes from price structure."},
            {"Indicator": "ATR(14)", "What it measures": "Average daily trading range and volatility.",
             "How to read": "Higher ATR means wider normal movement; the scanner uses ATR to estimate stops and targets."},
            {"Indicator": "Relative volume (RVOL)", "What it measures": "Current volume divided by its 20-day average.",
             "How to read": "1.0× is average; above 1.0× shows greater participation, but direction still matters."},
            {"Indicator": "Relative strength vs SPY", "What it measures": "The stock's 60-day return minus SPY's 60-day return.",
             "How to read": "Positive means outperformance; this is not the same calculation as RSI."},
            {"Indicator": "Historical volatility", "What it measures": "Annualized variability of recent daily returns.",
             "How to read": "Higher volatility implies a wider range of possible outcomes and usually requires smaller sizing."},
        ]
        st.dataframe(pd.DataFrame(indicator_rows), use_container_width=True, hide_index=True)
        st.caption("Indicators summarize historical data. They do not know future news, earnings surprises, gaps, or liquidity shocks.")

    with scanner_tab:
        score_rows = [
            {"Term": "AI score (0–100)", "Definition": "Weighted combination of trend, momentum, volume, relative strength, risk/reward, fundamental, and macro factor scores."},
            {"Term": "Factor score", "Definition": "A standardized 0–100 vote for one evidence category; 50 is broadly neutral."},
            {"Term": "Data coverage", "Definition": "Share of expected source fields available. Missing fields reduce confidence and may use a neutral prior rather than fabricated evidence."},
            {"Term": "Bull / Bear / Sideways probability", "Definition": "Model-estimated directional distribution for a stated horizon—not certainty, odds from an exchange, or a guarantee."},
            {"Term": "Confidence", "Definition": "Reliability adjustment based on coverage, sample size, and agreement among factor scores."},
            {"Term": "Expected 20D return", "Definition": "Model estimate of the average return over about 20 trading sessions; realized returns can differ materially."},
            {"Term": "Expected drawdown", "Definition": "Volatility-based adverse-move estimate over the horizon, not a worst-case loss."},
            {"Term": "Risk score", "Definition": "Composite emphasizing ATR volatility and the quality of the estimated reward/risk setup; higher means more risk."},
        ]
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
        st.subheader("Scanner classification thresholds")
        st.markdown(
            "- **Strong Buy:** AI score at least 78 **and** confidence at least 60%.\n"
            "- **Buy / Swing Trade:** AI score at least 65.\n"
            "- **Watchlist:** AI score at least 55.\n"
            "- **Neutral:** AI score from 35 up to, but not including, 55.\n"
            "- **Avoid / Short Candidate:** AI score below 35."
        )
        st.warning("A classification is research shorthand, not personalized advice or an instruction to place an order.")

    with risk_tab:
        risk_rows = [
            {"Term": "Entry/reference price", "Definition": "Price used to construct the plan; actual execution may differ because of spread, slippage, or gaps."},
            {"Term": "Stop-loss", "Definition": "Predefined invalidation level intended to limit loss. A stop order can fill below its trigger during a gap."},
            {"Term": "Trailing stop", "Definition": "A stop that follows favorable price movement; setting it too tightly can cause premature exits."},
            {"Term": "Target", "Definition": "Potential profit-taking level, not a forecast that price must reach it."},
            {"Term": "Reward/risk (R/R)", "Definition": "Potential reward divided by planned risk. Example: $10 upside / $5 downside = 2:1."},
            {"Term": "Risk per trade", "Definition": "Maximum intended portfolio loss if the stop is reached; commonly kept small to survive losing streaks."},
            {"Term": "Position size", "Definition": "Shares × price. Risk-based shares ≈ allowed dollar risk / distance from entry to stop."},
            {"Term": "Kelly fraction", "Definition": "Mathematical sizing estimate using win probability and payoff. It is highly sensitive to estimation error, so the app caps it."},
            {"Term": "Liquidity", "Definition": "Ability to trade without materially moving price. Low liquidity usually means wider spreads and greater slippage."},
        ]
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
        with st.expander("Common order types"):
            st.markdown(
                "- **Market order:** prioritizes execution, not price.\n"
                "- **Limit order:** sets the worst acceptable price, but may not fill.\n"
                "- **Stop order:** activates after a trigger and can experience slippage.\n"
                "- **Stop-limit order:** controls price after triggering, but may remain unfilled."
            )
        st.error("Never size a trade from potential profit alone. Define the invalidation point and affordable loss first.")

    with fundamental_tab:
        fundamental_rows = [
            {"Term": "Market capitalization", "Definition": "Share price × shares outstanding; a measure of company equity value, not revenue."},
            {"Term": "Forward P/E", "Definition": "Price divided by forecast earnings per share. Comparisons are most useful among similar businesses."},
            {"Term": "Revenue growth", "Definition": "Change in sales; growth quality depends on durability, margins, and cash generation."},
            {"Term": "Earnings growth", "Definition": "Change in profit; can be affected by accounting items and share buybacks."},
            {"Term": "Profit margin", "Definition": "Profit as a percentage of revenue."},
            {"Term": "Return on equity (ROE)", "Definition": "Net income relative to shareholder equity; leverage can inflate it."},
            {"Term": "Debt-to-equity", "Definition": "Debt relative to shareholder equity; acceptable levels vary by industry."},
            {"Term": "Beta", "Definition": "Historical sensitivity to broad-market movements. Beta above 1 has historically moved more than the benchmark."},
            {"Term": "Macro regime", "Definition": "Broad environment including market trend, rates, currency, commodities, and sector conditions."},
            {"Term": "Delayed/end-of-day data", "Definition": "Not a live quote. Prices may be unsuitable for immediate execution decisions."},
        ]
        st.dataframe(pd.DataFrame(fundamental_rows), use_container_width=True, hide_index=True)
        st.subheader("Important data limitations")
        st.markdown(
            "- A displayed **N/A** means the source field is unavailable.\n"
            "- A neutral score caused by missing data is not evidence that the company is average.\n"
            "- Dark-pool activity, options flow, dealer gamma, and live news sentiment are not currently connected.\n"
            "- Backtests and historical relationships can fail when market structure or regime changes."
        )

    st.divider()
    st.caption(
        "Educational reference only. Trading can result in substantial loss. Verify prices, liquidity, corporate events, taxes, "
        "transaction costs, and personal suitability independently before making a decision."
    )


with st.sidebar:
    st.markdown("## GPT AI Stocks Scanner")
    st.caption("Probabilistic Equity Intelligence")
    page = st.radio(
        "Workspace",
        ["AI Scanner", "Prediction Matrix", "Technical Terminal", "Accuracy Lab",
         "Short-Term Prediction", "Reference Guide"],
    )
    st.divider()
    restore_browser_ticker_universe()
    raw = st.text_area(
        "Universe (comma-separated)",
        key="ticker_universe",
        height=120,
        on_change=persist_ticker_universe,
    )
    tickers = normalize_tickers(raw, limit=TICKER_LIMIT)
    st.caption(f"{len(tickers)} saved tickers · limit {TICKER_LIMIT} · additions and deletions save automatically")
    st.subheader("Portfolio constraints")
    risk_per_trade = st.slider("Risk per trade (%)", .25, 3.0, 1.0, .25)
    max_position = st.slider("Maximum position (%)", 2.0, 30.0, 15.0, 1.0)
    if st.button("Refresh market data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.caption("Research decision support only. Not personalized investment advice. Validate liquidity, taxes, costs, and suitability independently.")

if not tickers and page not in ("Accuracy Lab", "Reference Guide"):
    st.error("Enter at least one valid ticker in the sidebar.")
elif page == "AI Scanner":
    scanner_page(tickers, max_position, risk_per_trade)
elif page == "Prediction Matrix":
    matrix_page(tickers, max_position, risk_per_trade)
elif page == "Technical Terminal":
    terminal_page(tickers, max_position, risk_per_trade)
elif page == "Accuracy Lab":
    accuracy_page()
elif page == "Short-Term Prediction":
    short_term_prediction_page(tickers, max_position, risk_per_trade)
else:
    reference_page()
