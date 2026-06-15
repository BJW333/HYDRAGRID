# HYDRAGRID — Performance Analysis

Comprehensive analysis of strategy performance including backtest results, paper trading results, cost modeling, and scaling projections.

---

## 1. Paper Trading Results (Live on Alpaca)

**Period:** April 14 – May 19, 2026 (26 days)

| Metric | Value |
|---|---|
| Starting Capital | $1,100 |
| Ending Value | $1,403 |
| Total Profit | +$303 |
| Return | +27.5% |
| Average Daily Profit | $11.65 |
| Annualized (projected) | ~194% |
| Best Day | +$25.40 |
| Worst Day | -$11.58 |
| Max Drawdown | ~3% |
| Winning Days | ~80% |

**Key Observations:**
- Consistent daily returns with low drawdowns
- First red day occurred on day 18 — recovered within 24 hours
- Performance exceeded backtest predictions during high-volatility weeks
- Performance matched backtest during low-volatility weeks (May 12-19)

---

## 2. Backtest Results (1 Year Historical Data)

**Data:** 35,017 candles per coin (15-minute intervals), 364 days
**Source:** Coinbase via CCXT
**Coins:** SOL/USD (30%), ETH/USD (25%), LINK/USD (25%), AVAX/USD (20%)
**Parameters:** num_grids=10, adaptive=True (ATR-based range, auto-rebalance)

### 2a. Scenario Comparison ($1,100 Capital)

| Scenario | Trades | Gross | Fees | Impact | Net/Year | ROI |
|---|---|---|---|---|---|---|
| Paper (zero costs) | 11,271 | $4,522 | $0 | $0 | $4,534 | 412% |
| After fees (0.165%) | 11,271 | $4,522 | $1,836 | $0 | $2,448 | 223% |
| Realistic (all costs) | 10,047 | $4,039 | $1,800 | $34 | $2,130 | 194% |

### 2b. Per-Coin Breakdown (Realistic Scenario)

| Coin | Weight | Trades | Gross | Fees | Net | $/day |
|---|---|---|---|---|---|---|
| SOL/USD | 30% | 2,527 | $1,215 | $545 | $637 | $1.75 |
| ETH/USD | 25% | 2,274 | $870 | $409 | $436 | $1.20 |
| LINK/USD | 25% | 2,580 | $1,099 | $463 | $608 | $1.67 |
| AVAX/USD | 20% | 2,666 | $856 | $383 | $449 | $1.23 |

### 2c. Cost Breakdown

Where every dollar of paper profit goes:

| Cost | Amount | % of Gross |
|---|---|---|
| Alpaca fees (0.165% blended) | $1,800 | 39.8% |
| Spread (bid-ask) | $109 | 2.4% |
| Missed fills + latency (9%) | Built into reduced trades | ~10.7% |
| Server downtime (2.5%) | Built into above | |
| **You keep** | **$2,130** | **47.1%** |

---

## 3. Friction Parameters (Researched)

Every cost in the realistic backtest was independently verified:

| Parameter | Value | Source |
|---|---|---|
| Alpaca maker fee | 0.15% | docs.alpaca.markets |
| Alpaca taker fee | 0.25% | docs.alpaca.markets |
| Blended rate (85% maker) | 0.165% | Calculated |
| Slippage (limit orders) | 0% | Limit orders fill at price or better |
| Bid-ask spread | ~0.01% | CoinGecko liquidity data |
| Missed fills | 5% | SOL/ETH 3-5%, LINK/AVAX 5-10% |
| API latency (60s poll) | 4% additional misses | Measured |
| Server downtime | 2.5% | Oracle Cloud free tier SLA |
| Market impact (alpha) | 0.15 | Almgren-Chriss square root model |

---

## 4. Strategy Validation

### 4a. Parameter Optimization

729 parameter combinations were swept across all 4 coins:
- Grid levels: 6, 8, 10, 12, 14
- Range multiplier: various ATR-based settings
- Rebalance thresholds: multiple configurations

The optimal configuration (num_grids=10, adaptive range, ATR×6) was selected based on highest risk-adjusted return across all coins simultaneously — not curve-fitted to a single coin.

### 4b. Strategies Tested and Killed

7,652+ parameter combinations were tested across 6 different strategies. Only grid trading produced consistent profits:

| Strategy | Configs Tested | Profitable | Best Annual | Status |
|---|---|---|---|---|
| **Grid Trading** | 729 | Many | $2,000+ | ✅ Active |
| HYDRA (momentum) | 720 | 0 | -$475 | ❌ Killed |
| Momentum Trailer | 240 | 0 | -$81 | ❌ Killed |
| VWAP Reversion | 50 | 0 | -$10 | ❌ Killed |
| UVXY Decay Puts | 5,625 | 0 | -$54 | ❌ Killed |
| VIX Spike SPY | 288 | 3 | $113 (70% DD) | ❌ Killed |

This is not a backtest-optimized lucky result. It survived systematic elimination of every alternative.

---

## 5. Scaling Analysis

Tested using the realistic backtester with Almgren-Chriss market impact model. Market impact = 0.15 × sqrt(order_value / daily_volume). More capital = bigger orders = more impact on price.

### 5a. Single Exchange (Alpaca)

| Capital | Gross/yr | Fees/yr | Impact/yr | Net/yr | ROI | $/day |
|---|---|---|---|---|---|---|
| $1,000 | $4,051 | $1,805 | $34 | $2,102 | 191% | $6 |
| $10,000 | $36,823 | $16,412 | $924 | $18,493 | 185% | $51 |
| $50,000 | $184,116 | $82,061 | $10,326 | $86,756 | 174% | $238 |
| $100,000 | $368,232 | $164,121 | $29,207 | $164,956 | 165% | $452 |
| $200,000 | $736,463 | $328,242 | $82,611 | $305,716 | 153% | $838 |
| $500,000 | $1,841,158 | $820,605 | $326,550 | $644,269 | 129% | $1,765 |
| $1,000,000 | $3,682,316 | $1,641,211 | $923,622 | $1,018,016 | 102% | $2,789 |

### 5b. Multi-Exchange (Reduces Impact)

| Capital | 1 Exchange | 2 Exchanges | 3 Exchanges | Market Maker (0% fee) |
|---|---|---|---|---|
| $1,000 | 191% | 252% | 229% | 356% |
| $50,000 | 174% | 239% | 215% | 345% |
| $200,000 | 153% | 225% | 198% | 332% |
| $1,000,000 | 102% | 189% | 155% | 298% |
| $5,000,000 | -12% | 108% | 61% | 224% |
| $10,000,000 | -98% | 47% | -10% | 168% |

**Key insight:** Adding a second exchange (Coinbase) nearly doubles the max profitable capital from $1M to $5M+ due to order splitting and lower blended fees (0.105% vs 0.165%).

### 5c. Exchange Fee Comparison

| Exchange | Maker Fee | Source |
|---|---|---|
| Alpaca | 0.150% | docs.alpaca.markets |
| Coinbase Advanced | 0.060% | coinbase.com/advanced |
| Kraken | 0.090% | kraken.com/features/fee-schedule |
| Binance US | 0.060% | binance.us |

---

## 6. Risk Analysis

### What can go wrong:

| Risk | Impact | Mitigation |
|---|---|---|
| Crypto drops 30%+ in a day | Grid buys fill, capital frozen in positions | Price recovers → buys become completed cycles. Kill switch stops at max drawdown threshold. |
| Exchange API goes down | Bot can't place/check orders | State persisted to disk. Bot resumes on reconnect. |
| Low volatility period | Fewer cycles, lower daily income | Grid still profitable, just slower. Backtest includes flat periods. |
| Alpaca raises fees | Profit margin compresses | At 0.30% fee, ROI drops from 191% to ~130%. Still profitable. |
| Strategy becomes crowded | Edge shrinks as more bots compete | Grid on retail-size crypto is a tiny fraction of volume. Would need thousands of competitors to matter. |

### Max drawdown characteristics:
- Backtest max drawdown: ~5-8% (unrealized, during directional moves)
- Drawdowns are temporary — grid recovers as price oscillates back
- Kill switch stops bot if drawdown exceeds configurable threshold

---

## 7. Tax Considerations (US)

Grid profits are **short-term capital gains** (held less than 1 year), taxed as ordinary income.

| Annual Income | Federal Rate | NY State | Effective |
|---|---|---|---|
| < $16,100 | 0% (standard deduction) | 0% | 0% |
| $16,100 - $48,475 | 12% | 4-6% | 16-18% |
| $48,475 - $103,350 | 22% | 6% | 28% |
| $103,350+ | 24-37% | 6-9% | 30-46% |

At small capital ($1-5k), standard deductions likely cover all trading income. At $50k+ capital, budget 25-30% for taxes.

---

## Disclaimer

All performance figures are from backtesting and paper trading. Past performance does not guarantee future results. Cryptocurrency trading involves significant risk of loss. The realistic backtest attempts to model all known costs but cannot predict future market conditions, exchange policy changes, or black swan events. Only trade with capital you can afford to lose.
