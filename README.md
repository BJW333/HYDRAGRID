# HYDRAGRID

**Regime-gated adaptive grid trading bot for crypto.**

HYDRAGRID profits from sideways/oscillating markets. It divides a price range
into levels, places limit buys below the current price and limit sells above,
and captures the spread each time price bounces through a level. A Hidden Markov
Model regime classifier gates the strategy: the grid runs in SIDEWAYS regimes
and **goes flat (liquidates to cash) on BEAR**, which is where an ungated grid
bleeds.

Part of a three-repo system:
- **HYDRAGRID** (this repo) — the sideways/range sleeve
- [HYDRAFLOW](https://github.com/BJW333/hydraflow_project) — the bull-trend sleeve
- [HYDRA_regime_classifier](https://github.com/BJW333/HYDRA_regime_classifier) — the HMM brain both sleeves consult

---

## Why the regime gate matters

A plain grid is short-gamma: it makes steady income while price oscillates and
bleeds while price trends down through its levels. The HMM gate is what turns
that liability into a controlled strategy.

In a 5-year backtest across SOL/ETH/LINK/AVAX ($1,100 capital), adding HMM
gating to the grid **turned a multi-year loss into a gain** by sitting out the
bear legs instead of catching every falling knife. The validated configuration
(grid + liquidate-to-cash on BEAR) is the product; merging the grid into a
shared capital pool with the trend sleeve was tested and *hurt* returns, so the
two run as independent bots.

> Per-coin results vary widely — LINK and ETH carried the portfolio while SOL
> dragged in backtest. SOL is on a watch list in live paper trading. Numbers are
> from historical backtests, not a promise of future performance.

---

## How it works

1. Divide an adaptive price range into N levels (default 10).
2. Place limit BUY orders at levels below the current price.
3. When a buy fills, place a SELL one level above. A completed buy→sell is one
   cycle = captured profit.
4. The range adapts to volatility via ATR (wider in volatile regimes, tighter in
   calm ones) and rebalances when price nears an edge.
5. Each cycle, the HMM classifier is consulted per coin. **BEAR → the grid is
   liquidated and the coin sits in cash** until the regime leaves BEAR.

---

## Layout

| path | role |
|------|------|
| `live_trader.py` | live runner (Alpaca paper), regime-gated multi-coin loop |
| `src/grid_strategy.py` | grid setup, order placement, fill handling, cycles |
| `src/alpaca_adapter.py` | Alpaca order/market-data adapter (via ccxt-style API) |
| `src/data_fetcher.py` | live + historical OHLCV fetch |
| `src/risk_manager.py`, `src/order_manager.py` | sizing, risk, order bookkeeping |
| `config/settings.py` | capital, coins, allocations, env-var credentials |
| `backtest.py`, `backtest_with_regime.py` | backtest engines (with/without gate) |
| `HYDRAGRID_docs/` | extended design + performance notes |

## Run

```bash
pip install -r requirements.txt

# credentials come from environment (never hardcoded)
export ALPACA_API_KEY="your-key"
export ALPACA_API_SECRET="your-secret"

# fetch historical data (CSVs are gitignored; regenerate locally)
python download_data_5yr.py

python live_trader.py --test     # one cycle, prints status, places no live orders unless armed
python live_trader.py            # live paper loop
python backtest_with_regime.py   # validate the regime-gated strategy
```

Requires the [HYDRA_regime_classifier](https://github.com/BJW333/HYDRA_regime_classifier)
repo as a sibling folder (the runner imports its trained HMM models).

## Status

Live in **paper trading** on Alpaca (isolated paper account, $1,100). This is
research/educational code, not financial advice. Trade real capital at your own
risk and only after your own validation.
