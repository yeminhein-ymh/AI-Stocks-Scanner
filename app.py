from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from institutional.config import DATA_PROVIDERS, DEFAULT_UNIVERSE
from institutional.data import DataUnavailable, fetch_metadata, fetch_prices, normalize_tickers
from institutional.scoring import Analysis, analyze, price_indicators
from institutional.signals import SignalStore, performance_metrics


st.set_page_config(page_title="Axiom AI Research", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root { --ink:#e8edf6; --muted:#8b98aa; --panel:#111923; --line:#263241; --cyan:#33d6c5; --amber:#ffbe55; }
.stApp { background: radial-gradient(circle at 80% 0%, #162535 0, #091018 34%, #070c12 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#091019; border-right:1px solid var(--line); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#111b26,#0c141d); border:1px solid var(--line); border-radius:10px; padding:14px; }
[data-testid="stMetricLabel"] { color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; }
.block-container { padding-top:1.5rem; max-width:1600px; }
.eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; }
.hero { font-size:2rem; font-weight:700; margin:.15rem 0; }
.subtle { color:var(--muted); font-size:.86rem; }
.pill { display:inline-block; padding:.25rem .65rem; border-radius:999px; background:#15342f; color:#7ff2d5; border:1px solid #245848; font-size:.78rem; }
.reason { padding:.45rem .6rem; margin:.25rem 0; border-left:2px solid var(--cyan); background:#0d1720; color:#c8d1dd; }
.warningbox { background:#2c2112; border:1px solid #65491c; padding:.7rem; border-radius:8px; color:#ffd184; }
.stTabs [data-baseweb="tab-list"] { gap:1.2rem; border-bottom:1px solid var(--line); }
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


def freshness(as_of: str) -> None:
    now = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    st.caption(f"Price data through {as_of} · App refreshed {now} · Delayed/end-of-day source")


def probability_chart(a: Analysis) -> go.Figure:
    horizons = list(a.probabilities)
    fig = go.Figure()
    fig.add_bar(name="Bull", x=horizons, y=[a.probabilities[h]["bull"] for h in horizons], marker_color="#20c997")
    fig.add_bar(name="Bear", x=horizons, y=[a.probabilities[h]["bear"] for h in horizons], marker_color="#ff6b6b")
    fig.update_layout(barmode="group", height=310, margin=dict(l=10, r=10, t=25, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="Modeled probability (%)", legend_orientation="h")
    return fig


def price_chart(prices: pd.DataFrame, title: str) -> go.Figure:
    df = price_indicators(prices)
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
                                   name="Price", increasing_line_color="#2dd4bf", decreasing_line_color="#fb7185"))
    for column, color in (("EMA20", "#60a5fa"), ("EMA50", "#f59e0b"), ("EMA200", "#a78bfa")):
        fig.add_scatter(x=df.index, y=df[column], name=column, line=dict(width=1.2, color=color))
    fig.update_layout(title=title, height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_orientation="h")
    return fig


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
    cols[0].metric("Universe", len(analyses))
    cols[1].metric("Top idea", top.ticker, top.classification)
    cols[2].metric("AI score", fmt(top.overall_score), "of 100")
    cols[3].metric("Bull probability", fmt(top.bull_probability, "pct"))
    cols[4].metric("Confidence", fmt(top.confidence, "pct"))
    cols[5].metric("Suggested size", fmt(top.recommended_position_size, "pct"))

    records = []
    for rank, a in enumerate(analyses, 1):
        records.append({"Rank": rank, "Ticker": a.ticker, "Classification": a.classification,
                        "AI Score": a.overall_score, "Bull %": a.bull_probability, "Bear %": a.bear_probability,
                        "Confidence %": a.confidence, "Risk Score": a.risk_score, "Expected 20D %": a.expected_return_20d,
                        "Drawdown est. %": a.expected_drawdown, "R/R": a.reward_risk,
                        "Trend": a.components["trend"].score, "Momentum": a.components["momentum"].score,
                        "Volume": a.components["volume"].score, "RS": a.components["relative_strength"].score,
                        "Fundamental": a.components["fundamental"].score, "Position %": a.recommended_position_size,
                        "Stage": a.trend_stage, "Stop": a.stop_loss, "Target 2": a.target_2})
    table = pd.DataFrame(records)
    st.subheader("Institutional ranking")
    st.dataframe(table.style.background_gradient(subset=["AI Score", "Bull %", "Confidence %"], cmap="RdYlGn"),
                 use_container_width=True, hide_index=True, height=min(680, 42 + 35 * len(table)))
    export_cols = st.columns([1, 1, 5])
    export_cols[0].download_button("Export CSV", table.to_csv(index=False).encode(), "axiom_scanner.csv", "text/csv")
    workbook = excel_bytes(table)
    if workbook:
        export_cols[1].download_button("Export Excel", workbook, "axiom_scanner.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if failures:
        with st.expander(f"Data exceptions ({len(failures)})"):
            st.write("\n".join(failures))

    selected = st.selectbox("Explain a ranked security", [a.ticker for a in analyses])
    a = next(item for item in analyses if item.ticker == selected)
    st.subheader(f"Why {a.ticker} scores {a.overall_score:.1f}")
    score_cols = st.columns(len(a.components))
    for col, (name, component) in zip(score_cols, a.components.items()):
        col.metric(name.replace("_", " ").title(), fmt(component.score), f"{component.coverage:.0%} coverage")
    left, right = st.columns([1.1, .9])
    with left:
        for name, component in a.components.items():
            with st.expander(f"{name.replace('_', ' ').title()} — {component.score:.1f}"):
                for reason in component.reasons:
                    st.markdown(f'<div class="reason">{reason}</div>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(probability_chart(a), use_container_width=True)
        st.markdown(f"**Risk plan:** stop {fmt(a.stop_loss, 'price')} · trail {fmt(a.trailing_stop, 'price')} · targets {fmt(a.target_1, 'price')} / {fmt(a.target_2, 'price')} / {fmt(a.target_3, 'price')}")


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
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.dataframe(matrix.style.background_gradient(subset=["Overall", "Bull %", "Confidence %", "Momentum", "RS vs SPY"], cmap="RdYlGn"),
                     use_container_width=True, hide_index=True, height=650)
    with c2:
        fig = go.Figure(go.Scatter(x=matrix["Expected Risk %"], y=matrix["Expected Return %"], mode="markers+text",
                                   text=matrix.Ticker, textposition="top center", marker=dict(size=matrix["Confidence %"] / 4,
                                   color=matrix["Overall"], colorscale="RdYlGn", showscale=True, colorbar_title="Score")))
        fig.update_layout(title="Opportunity map", xaxis_title="Expected drawdown magnitude (%)", yaxis_title="Expected 20D return (%)",
                          height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Bubble size = model confidence. Returns and drawdowns are distribution estimates, not forecasts with certainty.")
        st.markdown('<div class="warningbox">Institutional buying, options flow, dark pools, live news, and analyst targets are omitted until licensed sources are connected.</div>', unsafe_allow_html=True)
        workbook = excel_bytes(matrix)
        if workbook:
            st.download_button("Export matrix (Excel)", workbook, "axiom_prediction_matrix.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def terminal_page(tickers: list[str], max_position: float, risk_per_trade: float) -> None:
    header("Technical & Investment Terminal", "One evidence chain from price structure to portfolio-aware action.")
    ticker = st.selectbox("Security", tickers, key="terminal_ticker")
    try:
        a, prices = cached_analysis(ticker, max_position, risk_per_trade)
        meta = cached_metadata(ticker)
    except Exception as exc:
        st.error(str(exc)); return
    freshness(a.as_of)
    cols = st.columns(7)
    for col, label, value in zip(cols,
        ["Price", "AI score", "Bull", "Confidence", "Risk", "R/R", "Position"],
        [fmt(a.price, "price"), fmt(a.overall_score), fmt(a.bull_probability, "pct"), fmt(a.confidence, "pct"),
         fmt(a.risk_score), f"{a.reward_risk:.1f}:1", fmt(a.recommended_position_size, "pct")]):
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
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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


with st.sidebar:
    st.markdown("## ◈ AXIOM")
    st.caption("Probabilistic Equity Intelligence")
    page = st.radio("Workspace", ["AI Scanner", "Prediction Matrix", "Technical Terminal", "Accuracy Lab"])
    st.divider()
    raw = st.text_area("Universe (comma-separated)", ", ".join(DEFAULT_UNIVERSE[:10]), height=120)
    tickers = normalize_tickers(raw)
    st.caption(f"{len(tickers)} valid tickers · scanner limit 75")
    st.subheader("Portfolio constraints")
    risk_per_trade = st.slider("Risk per trade (%)", .25, 3.0, 1.0, .25)
    max_position = st.slider("Maximum position (%)", 2.0, 30.0, 15.0, 1.0)
    if st.button("Refresh market data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.divider()
    st.caption("Research decision support only. Not personalized investment advice. Validate liquidity, taxes, costs, and suitability independently.")

if not tickers and page != "Accuracy Lab":
    st.error("Enter at least one valid ticker in the sidebar.")
elif page == "AI Scanner":
    scanner_page(tickers, max_position, risk_per_trade)
elif page == "Prediction Matrix":
    matrix_page(tickers, max_position, risk_per_trade)
elif page == "Technical Terminal":
    terminal_page(tickers, max_position, risk_per_trade)
else:
    accuracy_page()
