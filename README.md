# Axiom AI Research

A source-aware, explainable Streamlit equity research terminal. It combines independent trend, momentum, volume, relative-strength, risk/reward, fundamental, and macro votes instead of issuing single-indicator calls.

## What is implemented

- Multi-stock scanner with ranked composite scores, three-way probabilities, confidence, risk levels, ATR stops/targets, Kelly-capped sizing, and component-level explanations.
- Prediction matrix with sortable opportunity ranking and a risk/return/confidence map.
- Technical and investment terminal with price structure, indicators, professional thesis, probability table, data coverage, and PDF export.
- SQLite-backed signal ledger and strategy attribution for a future walk-forward model feedback loop.
- Explicit source coverage: unavailable premium datasets are shown as unavailable and are never synthesized.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app needs outbound network access for Yahoo Finance. Quotes are delayed/end-of-day and are subject to the upstream provider's coverage and terms.

## Metric and modeling contract

- `Overall Score` is a 0–100 weighted vote: trend 22%, momentum 15%, volume 10%, relative strength 14%, risk/reward 14%, fundamentals 15%, macro 10%.
- Probabilities use a normal return approximation whose drift blends recent observed returns with the cross-sectional composite. They are scenario estimates, not calibrated guarantees.
- Confidence is reduced by limited history, missing fields, low component agreement, and incomplete source coverage; it is capped at 92%.
- Position size is the minimum of portfolio maximum, ATR-stop risk budget, and quarter-Kelly. A zero edge produces a zero suggested position.
- Historical observations and forward-looking estimates are labeled separately throughout the UI.

## Production extensions

Connect licensed feeds for dark pools, options flow, open-interest change, dealer gamma, short interest, live news, analyst estimates, economic calendars, and institutional ownership. Model retraining should run outside the Streamlit process using purged walk-forward validation, point-in-time features, transaction costs, drift monitoring, and champion/challenger promotion.

### Optional Tradier option chain

The **Smart Money Options** page can use Tradier option-chain quotes, volume, open interest, IV, and Greeks. Add the token to Streamlit Secrets; never commit it to GitHub:

```toml
TRADIER_TOKEN = "your-private-token"
```

An option chain does not identify trade aggressor, opening/closing status, multi-leg relationships, or institutional ownership. Those conclusions remain withheld until a licensed trade-level OPRA/NBBO source is connected.

## Tests

```powershell
python -m pytest -q
```

This software is research decision support, not personalized investment advice.
