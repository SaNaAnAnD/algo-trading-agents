# Person 2: CrewAI trading crew using the Person 1 MCP server.
# Run:  python clients/trading_crew.py          (asks y/N before placing orders)
#       python clients/trading_crew.py --auto   (places orders without asking)

import os
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

SERVER_URL = "http://localhost:8001/mcp"
PROJECT = Path(__file__).resolve().parent.parent
STOCKS_FILE = PROJECT / "clients" / "my_stocks.txt"
TARGET_INVESTED = 0.95   # keep up to 95% of equity invested
STOP_ATR = 2.5           # sell all if price falls 2.5 x ATR below entry
TAKE_PROFIT_ATR = 3.0    # sell half if price rises 3 x ATR above entry

llm = LLM(
    model=os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)


# ---------------- helpers ----------------

async def _call(tool, args):
    async with Client(SERVER_URL) as c:
        return (await c.call_tool(tool, args)).data


def mcp(tool, args=None):
    """Call one tool on the MCP server."""
    return asyncio.run(_call(tool, args or {}))


def load_symbols():
    if not STOCKS_FILE.exists():
        sys.exit(f"Create {STOCKS_FILE} with one ticker per line.")
    syms = [l.strip().upper() for l in STOCKS_FILE.read_text().splitlines()]
    return [s for s in syms if s and not s.startswith("#")]


def parse_json(text, default):
    """Take the first JSON object or list out of an AI answer."""
    m = re.search(r"(\{.*\}|\[.*\])", text or "", re.S)
    if not m:
        return default
    try:
        return json.loads(m.group(1))
    except Exception:
        return default


def save(folder, name, stamp, text):
    path = folder / f"{name}{stamp}.md"
    path.write_text(text)
    print("   saved", path.relative_to(PROJECT))


def rams_table(result):
    rows = ["| Symbol | RAMS | Trend | Momentum | Volume | Breakout | RSI | Signal |",
            "|---|---|---|---|---|---|---|---|"]
    for s in result["stocks"]:
        if "error" in s:
            rows.append(f"| {s['symbol']} | error: {s['error']} | | | | | | |")
        else:
            rows.append(f"| {s['symbol']} | {s['rams']} | {s['trend']} | {s['momentum']} | "
                        f"{s['volume']} | {s['breakout']} | {s['rsi']} | {s['signal']} |")
    return "\n".join(rows)


def build_plan(final, positions, account):
    """Turn RAMS signals into a list of proposed orders."""
    equity = float(account["equity"])
    held = {p["symbol"]: p for p in positions["positions"]}
    invested = sum(float(p["market_value"]) for p in held.values())
    info = {s["symbol"]: s for s in final["stocks"] if "error" not in s}
    plan = []

    # 1) exits for stocks we already hold
    for sym, p in held.items():
        s = info.get(sym)
        if not s:
            continue
        qty = int(float(p["qty"]))
        entry, price, atr = float(p["avg_entry_price"]), s["price"], s["atr"]
        if qty <= 0:
            continue
        if price <= entry - STOP_ATR * atr:
            plan.append(dict(symbol=sym, side="sell", qty=qty, price=price,
                             rams=s["rams"], reason="stop-loss hit"))
        elif s["signal"] == "SELL":
            plan.append(dict(symbol=sym, side="sell", qty=qty, price=price,
                             rams=s["rams"], reason=f"RAMS {s['rams']} below sell level"))
        elif price >= entry + TAKE_PROFIT_ATR * atr and qty >= 2:
            plan.append(dict(symbol=sym, side="sell", qty=qty // 2, price=price,
                             rams=s["rams"], reason="take half profit"))

    # 2) new buys and adds, highest score first
    budget = TARGET_INVESTED * equity - invested
    for s in sorted(info.values(), key=lambda x: x["rams"], reverse=True):
        sym, price = s["symbol"], s["price"]
        if s["signal"] == "BUY":
            qty, reason = s["suggested_qty"], f"RAMS {s['rams']} above buy level"
        elif s["signal"] == "ADD" and float(held[sym]["unrealized_pl"]) > 0:
            have = float(held[sym]["market_value"])
            qty = int(max(0, s["suggested_qty"] * price - have) // price)
            reason = f"RAMS {s['rams']} strong and position in profit"
        else:
            continue
        qty = min(qty, int(max(0, budget) // price))
        if qty <= 0:
            continue
        plan.append(dict(symbol=sym, side="buy", qty=qty, price=price,
                         rams=s["rams"], reason=reason))
        budget -= qty * price
    return plan


# ---------------- main run ----------------

def main():
    auto = "--auto" in sys.argv
    now = datetime.now()
    stamp = now.strftime("%d%m%Y-%H:%M")
    folder = PROJECT / "outputs" / f"run_{now.strftime('%d%m%Y-%H%M')}"
    folder.mkdir(parents=True, exist_ok=True)

    symbols = load_symbols()
    print(f"\n[1/5] Fetching data for {len(symbols)} stocks from the MCP server...")
    account = mcp("get_account_info")
    positions = mcp("get_all_positions")
    technical = mcp("compute_rams", {"symbols": symbols})
    headlines = {}
    for s in symbols:
        items = mcp("get_news", {"symbol": s, "limit": 3}).get("headlines", [])
        headlines[s] = [h["headline"] for h in items]
    news_text = "\n".join(f"{s}: " + (" | ".join(h) if h else "no recent news")
                          for s, h in headlines.items())
    pos_text = (json.dumps(positions["positions"], indent=1)
                if positions["count"] else "No open positions.")

    # ----- agents -----
    common = dict(llm=llm, allow_delegation=False, verbose=False)
    market_agent = Agent(role="Market Data Agent",
        goal="Summarise the market regime, the account and the analysts' key findings.",
        backstory="You monitor the US market and the paper-trading account for a student trading desk.",
        **common)
    news_agent = Agent(role="News Sentiment Analyst",
        goal="Score how positive or negative recent news is for each stock.",
        backstory="You read financial headlines and judge their likely short-term price impact.",
        **common)
    tech_agent = Agent(role="Technical Analyst",
        goal="Explain what the RAMS indicator scores say about each stock.",
        backstory="You are an expert in trend, momentum, volume and breakout indicators.",
        **common)
    risk_agent = Agent(role="Risk Manager",
        goal="Approve or reject each proposed order to protect the portfolio.",
        backstory="You check concentration, sector overlap, overbought stocks and bad news.",
        **common)
    trader_agent = Agent(role="Portfolio Manager and Trader",
        goal="Give the final list of orders to execute.",
        backstory="You make the final decisions for an aggressive 16-day paper-trading contest.",
        **common)

    # ----- crew 1: News + Technical in PARALLEL, then Market summary -----
    news_task = Task(
        description=("Score the news sentiment for each stock from -1 (very negative) to +1 "
                     "(very positive). Use 0 when there is no news or it is neutral.\n\n"
                     "Headlines:\n" + news_text +
                     "\n\nReply with ONLY a JSON object, for example {\"AAPL\": 0.4, \"TSLA\": -0.2}. "
                     "Include every one of these tickers: " + ", ".join(symbols)),
        expected_output="A JSON object mapping each ticker to a number between -1 and 1.",
        agent=news_agent, async_execution=True)
    tech_task = Task(
        description=(f"Market regime: {technical['regime']} "
                     f"(buy threshold {technical['buy_threshold']}).\n"
                     "RAMS scores (0-100) with pillar scores:\n" + rams_table(technical) +
                     "\n\nWrite a short analysis with bullet points: the strongest stocks, the "
                     "weakest stocks, and any warnings (e.g. RSI above 70, weak volume)."),
        expected_output="A short markdown technical analysis.",
        agent=tech_agent, async_execution=True)
    market_task = Task(
        description=(f"Account: {json.dumps(account)}\nOpen positions: {pos_text}\n"
                     f"Market regime from SPY: {technical['regime']}.\n"
                     "Using the news and technical reports, write a short market summary."),
        expected_output="A short markdown market summary.",
        agent=market_agent, context=[news_task, tech_task])

    print("[2/5] News and Technical agents working in parallel...")
    Crew(agents=[news_agent, tech_agent, market_agent],
         tasks=[news_task, tech_task, market_task],
         process=Process.sequential, max_rpm=20, verbose=False).kickoff()

    news_scores = parse_json(news_task.output.raw, {})
    news_scores = {str(k).upper(): float(v) for k, v in news_scores.items()
                   if isinstance(v, (int, float))}
    save(folder, "newsSentimentAgent", stamp,
         "# News Sentiment Agent\n\n## Headlines\n" + news_text +
         "\n\n## Scores\n```json\n" + json.dumps(news_scores, indent=2) + "\n```\n")
    save(folder, "technicalAnalystAgent", stamp,
         "# Technical Analyst Agent\n\n" + rams_table(technical) + "\n\n" + tech_task.output.raw)
    save(folder, "marketDataAgent", stamp, "# Market Data Agent\n\n" + market_task.output.raw)

    # ----- final scores with news included -----
    print("[3/5] Final RAMS scores (technical + news)...")
    final = mcp("compute_rams", {"symbols": symbols, "news_scores": news_scores})
    print(rams_table(final))
    plan = build_plan(final, positions, account)

    if not plan:
        print("\n[4/5] No trades suggested this run.")
        save(folder, "traderAgent", stamp,
             "# Trader Agent\n\nNo trades this run - no stock met the buy/sell rules.\n\n"
             + rams_table(final))
        return

    plan_text = "\n".join(
        f"{i+1}. {o['side'].upper()} {o['qty']} {o['symbol']} @ ~{o['price']} "
        f"(RAMS {o['rams']}) - {o['reason']}" for i, o in enumerate(plan))

    # ----- crew 2: Risk Manager then Trader -----
    risk_task = Task(
        description=(f"Account: {json.dumps(account)}\nOpen positions: {pos_text}\n"
                     f"Regime: {final['regime']}\nNews scores: {json.dumps(news_scores)}\n\n"
                     f"Proposed orders:\n{plan_text}\n\n"
                     "Review each order. Reject only with a clear risk reason (too much in one "
                     "sector, very bad news, extremely overbought). Give a short reason for each."),
        expected_output="A markdown list: each order APPROVED or REJECTED with a reason.",
        agent=risk_agent)
    trade_task = Task(
        description=("Using the Risk Manager's review, choose the final orders. Reply with ONLY a "
                     "JSON list of the order numbers to execute, for example [1, 2, 4]. "
                     "Never add new orders."),
        expected_output="A JSON list of order numbers.",
        agent=trader_agent, context=[risk_task])

    print("\n[4/5] Risk Manager and Trader reviewing the plan...")
    Crew(agents=[risk_agent, trader_agent], tasks=[risk_task, trade_task],
         process=Process.sequential, max_rpm=20, verbose=False).kickoff()

    chosen = parse_json(trade_task.output.raw, [])
    chosen = sorted({i for i in chosen if isinstance(i, int) and 1 <= i <= len(plan)})
    final_orders = [plan[i - 1] for i in chosen]
    save(folder, "riskManagerAgent", stamp,
         "# Risk Manager Agent\n\n## Proposed orders\n" + plan_text +
         "\n\n## Review\n" + risk_task.output.raw)
    save(folder, "traderAgent", stamp,
         f"# Trader Agent\n\nChosen order numbers: {chosen}\n\n" + trade_task.output.raw)

    # ----- place orders -----
    print("\n[5/5] Final orders:")
    if not final_orders:
        print("   None - all orders were rejected.")
        return
    for o in final_orders:
        print(f"   {o['side'].upper()} {o['qty']} {o['symbol']} @ ~{o['price']} - {o['reason']}")
    if not auto and input("\nPlace these orders? [y/N]: ").strip().lower() != "y":
        print("Cancelled - no orders placed.")
        save(folder, "executionLog", stamp, "# Execution\n\nCancelled by user.\n")
        return

    results = []
    for o in final_orders:
        limit = round(o["price"] * (1.003 if o["side"] == "buy" else 0.997), 2)
        try:
            if o["side"] == "sell":
                mcp("cancel_all_orders", {"symbol": o["symbol"]})
            r = mcp("place_limit_order", {"symbol": o["symbol"], "side": o["side"],
                                          "qty": int(o["qty"]), "limit_price": limit,
                                          "extended_hours": True})
            results.append(f"- {o['side'].upper()} {o['qty']} {o['symbol']} "
                           f"limit {limit}: {r.get('status')}")
        except Exception as e:
            results.append(f"- {o['side'].upper()} {o['qty']} {o['symbol']}: FAILED - {e}")
    print("\n".join(results))
    save(folder, "executionLog", stamp, "# Executed orders\n\n" + "\n".join(results) + "\n")


if __name__ == "__main__":
    main()