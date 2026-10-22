# Experiment Log

Paper-trading contest, 13-27 October 2026. Goal: maximum percentage gain on a
$100,000 Alpaca paper account using a CrewAI crew with a local MCP server.
Each entry records what was changed, why, and what was observed.

## 15 Oct 2026 - From manual dry runs to an autonomous agent

**Starting point.** The crew (Market Data, News Sentiment, Technical Analyst,
Risk Manager and Trader agents) produced correct plans in dry-run mode, but no
orders were placed: trading was disabled in `.env` and every run was started by hand.

**Observed problem 1 - exits blocked by protective stops.** Every buy is an OTO
order with a 6% protective stop. The stop leg reserves the shares, so any other
sell of that stock was rejected by the broker.
*Change:* a sell first cancels that stock's own open orders, waits until they are
gone, submits the sell, and re-attaches the same stop to any shares left.

**Observed problem 2 - no autonomy.** *Change:* `run_session.py` starts the MCP
server, waits for the market open, runs the full crew in `--auto` mode every
15 minutes until 10 minutes before the close, and restarts the server if it stops.

**First live run (23:30 IST).** RAMS ranked XYZ first (75.2). The crew sold FCX
(stop-loss hit) and half of V (take-profit). The XYZ buy was rejected because
orders were sent as extended-hours orders, which the OTO buy does not allow.
*Change:* orders are sent for regular market hours.

**Observed problem 3 - single point of failure in the AI.** If the one Groq model
reached its free daily token limit, the run ended before any order, including
stop-loss exits. *Change:* a fallback chain of four free models; a model at its
daily limit is skipped at once; if every model is unavailable the run continues
with the rule-based RAMS plan (neutral news), so exits and strong buys still execute.

**Observed problem 4 - exits that depend on the AI.** The Trader agent could leave
a stop-loss sell out of its final list. *Change:* stop-loss and breakeven exits are
always executed; the agents decide the remaining orders.

**Observed problem 5 - profit taken repeatedly.** "Take half profit" fired on every
run while the price stayed high, selling a winner down to zero within about an
hour. *Change:* half is taken once per position, after which the rest is sold only
if the price falls back to the entry price (breakeven stop).

**Observed problem 6 - duplicate buys.** An unfilled buy order could be placed again
on the next run. *Change:* stocks with an open buy order are skipped.

**Observed problem 7 - idle cash.** Position caps of 12% (8% for high-beta stocks),
and a high-beta list that did not contain this portfolio's stocks, left most of the
account in cash. *Change:* caps of 20% per stock and 15% for the high-beta stocks
MU, XYZ, HOOD and SMCI, still with the 6% protective stop on every buy (at most
about 1.2% of equity at risk per position).


**Observed problem 8 - AI veto without evidence (23:55 IST).** The Risk Manager
rejected the XYZ buy (RAMS 70.3, RSI 69.9, news 0.0) on the grounds that a RAMS
score above the buy level meant the stock was overbought, which misreads RAMS
(a strength score where higher is better). The Trader followed it and no buy was
placed. *Change:* the Risk Manager now receives each buy's RSI and news score with
explicit definitions (overbought = RSI 80 or more; very bad news = news score
-0.25 or lower). A buy that passed every rule is skipped only with such evidence;
otherwise it is kept and the overruled veto is logged.

**Observed problem 9 - cash left idle.** Only stocks above the buy level (70 in a
bear regime) were bought, so about 94% of the account stayed in cash. *Change:*
careful allocation. 20% of equity always stays in cash; the other 80% is spread
across healthy stocks only (RAMS 50 or more, trend pillar 25 or more, news score
above -0.25, no SELL signal), weighted by RAMS, at most 27% per stock and 15% per
high-beta stock, never adding to a losing position. With few healthy stocks the
caps bind and the remainder stays in cash rather than going into falling stocks.

**Run interval (16 Oct).** The session interval was changed from 15 to 45 minutes
(about 8 runs per evening): fewer, more deliberate decisions for a careful investor,
less churn from short-term noise, and well inside the free Groq daily token quota.

**AI availability.** The agents use a chain of free models: Groq gpt-oss-120b, then
gpt-oss-20b, qwen and llama-3.3-70b (each with its own daily quota), then OpenRouter
and the college Ollama server when configured. A model that reaches its daily limit
is skipped until the next day. If no model is available, the run continues with the
rule-based plan, so exits and allocation still happen.

**Records.** Every run appends a line to `crew_ai/outputs/session_log.txt`
(equity, change since the $100,000 start, % invested, cash, positions). At the end
of each session `crew_ai/outputs/daily_summary_<date>.md` lists the start and end
equity, every order of the day and the final positions, and the records are
committed to GitHub. Agent reports for every run are in `crew_ai/outputs/run_*`.

## 22 Oct 2026 - Backtest-driven fixes (less churn, daily signals, protected positions)

**Evidence.** `backtest_rams.py` (3 years of daily bars, 5 bps cost, news neutral):
current rules +24.4% with 305 trades; wider buy/sell gap + confirmation + minimum hold
+76.2% with 128 trades (Sharpe 0.62 -> 1.45). RAMS itself did not predict the next 5 days
(correlation -0.04), so the gain comes from trading less, not from a better signal.
A regime gate (buy only at RAMS 70 in a bear market) lowered the return and was not adopted.

**Changes.** (1) RAMS now uses daily bars (the rules were tested on daily data).
(2) Buy at RAMS 60+ (was 50), sell below 30 (was 40). (3) A sell signal must show on 2 different
days and the position must be held 3 days; stop-losses are never delayed. (4) 2-day cooldown before
re-buying a stock just sold. (5) "Take half profit" is off by default (TAKE_HALF_PROFIT) and its
flag is now keyed by stock, so an average-price change cannot reset it. (6) `ensure_protective_stops`
re-attaches a GTC stop to every position at the start of each run (DAY stops had expired).
(7) Stop-loss exits use a 1.5% limit buffer so they fill. (8) Every SELL is always executed; the
Risk Manager reviews BUYs only and receives code-computed exposure numbers. (9) AI temperature 0
and news scores smoothed with the previous run. (10) 3 runs a day (`--every 150`).
