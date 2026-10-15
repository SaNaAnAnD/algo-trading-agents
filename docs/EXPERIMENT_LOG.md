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

Run-by-run results: `crew_ai/outputs/session_log.txt` and `crew_ai/outputs/run_*`.
