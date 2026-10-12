"""CrewAI paper-trading crew.

Run with ``python clients/trading_crew.py`` for per-order confirmation,
``python clients/trading_crew.py --dry-run`` to print and log without trading,
or ``python clients/trading_crew.py --auto`` to place orders without prompts.

Settings from .env: MCP_URL (http://localhost:8001/mcp),
LLM_MODEL (groq/openai/gpt-oss-120b), TARGET_INVESTED (0.95, 0 to 1),
STOP_ATR (2.5, above 0), TAKE_PROFIT_ATR (3.0, above 0),
LIMIT_BUFFER (0.003), MAX_RPM (20), NEWS_HEADLINES (3),
EXTENDED_HOURS (true), and LLM_RETRIES (3).
"""

import os
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import asyncio
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from crewai import Agent, Task, Crew, Process, LLM

load_dotenv()

PROJECT = Path(__file__).resolve().parent.parent
STOCKS_FILE = PROJECT / "clients" / "my_stocks.txt"


def read_setting(name, default, parser, valid=lambda value: True):
    """Read one environment setting, warning and falling back if invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = parser(raw)
        if not valid(value):
            raise ValueError("value is outside the allowed range")
        return value
    except (TypeError, ValueError):
        print(f"Warning: invalid {name}={raw!r}; using default {default!r}.")
        return default


def parse_bool(value):
    """Parse an explicit true/false environment value."""
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes", "y"):
        return True
    if lowered in ("false", "0", "no", "n"):
        return False
    raise ValueError("expected true or false")


def parse_finite_float(value):
    """Parse a finite numeric setting."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("expected a finite number")
    return parsed


SETTINGS = {
    "MCP_URL": read_setting("MCP_URL", "http://localhost:8001/mcp", str),
    "LLM_MODEL": read_setting("LLM_MODEL", "groq/openai/gpt-oss-120b", str),
    "TARGET_INVESTED": read_setting("TARGET_INVESTED", 0.95, parse_finite_float,
                                    lambda value: 0 <= value <= 1),
    "STOP_ATR": read_setting("STOP_ATR", 2.5, parse_finite_float, lambda value: value > 0),
    "TAKE_PROFIT_ATR": read_setting("TAKE_PROFIT_ATR", 3.0, parse_finite_float,
                                    lambda value: value > 0),
    "LIMIT_BUFFER": read_setting("LIMIT_BUFFER", 0.003, parse_finite_float),
    "MAX_RPM": read_setting("MAX_RPM", 20, int, lambda value: value > 0),
    "NEWS_HEADLINES": read_setting("NEWS_HEADLINES", 3, int, lambda value: value >= 0),
    "EXTENDED_HOURS": read_setting("EXTENDED_HOURS", True, parse_bool),
    "LLM_RETRIES": read_setting("LLM_RETRIES", 3, int, lambda value: value >= 0),
}

llm = LLM(
    model=SETTINGS["LLM_MODEL"],
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)


# ---------------- helpers ----------------

async def _call(tool, args):
    async with Client(SETTINGS["MCP_URL"]) as c:
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


def rams_table(result, news_scores=None):
    """Format the RAMS rows, optionally adding each stock's news score."""
    columns = ["Symbol", "RAMS", "Trend", "Momentum", "Volume", "Breakout", "RSI", "Signal"]
    if news_scores is not None:
        columns.append("News")
    rows = ["| " + " | ".join(columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|"]
    for s in result["stocks"]:
        if "error" in s:
            values = [s.get("symbol", ""), f"error: {s['error']}"] + [""] * (len(columns) - 2)
        else:
            values = [s.get("symbol", ""), s.get("rams", ""), s.get("trend", ""),
                      s.get("momentum", ""), s.get("volume", ""), s.get("breakout", ""),
                      s.get("rsi", ""), s.get("signal", "")]
            if news_scores is not None:
                values.append(news_scores.get(str(s.get("symbol", "")).upper(), 0))
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def number(value, default=0.0):
    """Convert MCP numeric strings safely for planning and display."""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def build_plan(final, positions, account, settings):
    """Build orders from supplied data only; this function makes no network calls."""
    held = {str(position["symbol"]).upper(): position
            for position in positions.get("positions", [])}
    info = {str(stock.get("symbol", "")).upper(): stock
            for stock in final.get("stocks", [])
            if "error" not in stock and stock.get("symbol")}
    equity = number(account.get("equity"))
    invested = sum(number(position.get("market_value")) for position in held.values())
    stop_sells, signal_sells, profit_sells = [], [], []
    planned_sell_value = 0.0
    planned_symbols = set()

    # Decide exits first so their value can safely fund later buys.
    for symbol, position in held.items():
        stock = info.get(symbol)
        if not stock:
            continue
        qty = max(0, int(number(position.get("qty"))))
        price = number(stock.get("price"))
        atr = number(stock.get("atr"))
        entry = number(position.get("avg_entry_price"))
        if qty <= 0 or price <= 0:
            continue
        order = {
            "symbol": symbol, "side": "sell", "price": price,
            "rams": number(stock.get("rams")),
            "stop_loss": stock.get("stop_loss"),
        }
        if price <= entry - settings["STOP_ATR"] * atr:
            order.update(qty=qty, reason="stop-loss hit")
            stop_sells.append(order)
        elif stock.get("signal") == "SELL":
            order.update(qty=qty, reason=f"RAMS {stock.get('rams')} below sell level")
            signal_sells.append(order)
        elif price >= entry + settings["TAKE_PROFIT_ATR"] * atr and qty >= 2:
            order.update(qty=qty // 2, reason="take half profit")
            profit_sells.append(order)

    sells = stop_sells + signal_sells + profit_sells
    for order in sells:
        planned_symbols.add(order["symbol"])
        planned_sell_value += order["qty"] * order["price"]

    # Cap buys by both target allocation and cash available after planned sells.
    target_budget = settings["TARGET_INVESTED"] * equity - (invested - planned_sell_value)
    cash_budget = number(account.get("cash")) + planned_sell_value
    budget = max(0.0, min(target_budget, cash_budget))
    buys = []
    for stock in sorted(info.values(), key=lambda item: number(item.get("rams")), reverse=True):
        symbol = str(stock.get("symbol", "")).upper()
        price = number(stock.get("price"))
        if symbol in planned_symbols or price <= 0:
            continue
        signal = stock.get("signal")
        if signal == "BUY":
            qty = max(0, int(number(stock.get("suggested_qty"))))
            reason = f"RAMS {stock.get('rams')} above buy level"
        elif signal == "ADD" and symbol in held and number(held[symbol].get("unrealized_pl")) > 0:
            current_value = number(held[symbol].get("market_value"))
            target_value = number(stock.get("suggested_qty")) * price
            qty = max(0, int((target_value - current_value) // price))
            reason = f"RAMS {stock.get('rams')} strong and position in profit"
        else:
            continue
        qty = min(qty, int(budget // price))
        if qty <= 0:
            continue
        buys.append({
            "symbol": symbol, "side": "buy", "qty": qty, "price": price,
            "rams": number(stock.get("rams")), "stop_loss": stock.get("stop_loss"),
            "reason": reason,
        })
        planned_symbols.add(symbol)
        budget -= qty * price
    return sells + buys


def parse_json_answer(text):
    """Parse JSON from a plain response or a fenced/prose-wrapped response."""
    try:
        return json.loads(text), True
    except (TypeError, json.JSONDecodeError):
        match = re.search(r"(\{.*\}|\[.*\])", text or "", re.S)
        if match:
            try:
                return json.loads(match.group(1)), True
            except json.JSONDecodeError:
                pass
    return None, False


def run_crew(crew):
    """Run a Crew, retrying rate-limit failures with the requested backoff."""
    retries = SETTINGS["LLM_RETRIES"]
    for attempt in range(retries + 1):
        try:
            crew.kickoff()
            return True
        except Exception as error:
            message = str(error)
            lowered = message.lower()
            if "model_not_found" in lowered or "does not exist" in lowered:
                print("Model not found - check LLM_MODEL in .env against console.groq.com/docs/models")
                return False
            if ("429" in message or "rate" in lowered) and attempt < retries:
                delay = 30 * (attempt + 1)
                print(f"LLM rate limit; retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            print(f"LLM error: {message}")
            return False
    return False


def limit_price(order):
    """Calculate a buffered limit price; orders are always limit orders."""
    direction = 1 if order["side"] == "buy" else -1
    return round(order["price"] * (1 + direction * SETTINGS["LIMIT_BUFFER"]), 2)


def order_line(order, index, total):
    """Format one proposed order for confirmation or dry-run output."""
    stop = order.get("stop_loss")
    stop_text = f"{number(stop):.2f}" if stop not in (None, "") else "n/a"
    return (f"[{index}/{total}] {order['side'].upper()} {order['qty']} {order['symbol']} "
            f"@ limit {limit_price(order):.2f} (RAMS {order['rams']}, stop-loss {stop_text}) "
            f"- {order['reason']}")


def confirm_order():
    """Require an explicit y or n, repeating for every other response."""
    while True:
        sys.stdout.flush()
        answer = input("Place this order? [y/n]: ")
        answer = "".join(answer.split()).lower()
        if answer in ("y", "n"):
            return answer == "y"
        print("Please enter y or n.")


def settings_report(timestamp, flags, symbols, regime, buy_threshold):
    """Return the non-secret run configuration record."""
    lines = [f"# Settings Used\n\nTimestamp: {timestamp}",
             f"Flags: {', '.join(flags) if flags else 'none'}",
             f"Model: {SETTINGS['LLM_MODEL']}",
             f"Stocks: {', '.join(symbols) if symbols else 'none'}",
             f"Regime: {regime}", f"Buy threshold: {buy_threshold}", "", "## Settings"]
    lines.extend(f"{key}: {value}" for key, value in SETTINGS.items())
    return "\n".join(lines) + "\n"


# ---------------- main run ----------------

def main():
    auto = "--auto" in sys.argv
    dry_run = "--dry-run" in sys.argv
    flags = [flag for flag, enabled in (("--auto", auto), ("--dry-run", dry_run)) if enabled]
    now = datetime.now()
    timestamp = now.strftime("%d%m%Y-%H-%M")
    folder = PROJECT / "outputs" / f"run_{now.strftime('%d%m%Y-%H-%M')}"
    folder.mkdir(parents=True, exist_ok=True)
    symbols, regime, buy_threshold = [], "unavailable", "unavailable"
    execution_entries = []
    run_status = "Run ended before orders were selected."
    try:
        symbols = load_symbols()
        print(f"\n[1/5] Fetching data for {len(symbols)} stocks from the MCP server...")
        try:
            account = mcp("get_account_info")
            positions = mcp("get_all_positions")
            technical = mcp("compute_rams", {"symbols": symbols})
        except Exception as error:
            print("Start the server first: python servers/alpaca_trading_mcpServer.py")
            print(f"MCP error: {error}")
            run_status = f"MCP unavailable: {error}"
            return
        if isinstance(technical, dict) and technical.get("error"):
            print(technical["error"])
            run_status = f"compute_rams error: {technical['error']}"
            return
        regime = technical.get("regime", "unavailable")
        buy_threshold = technical.get("buy_threshold", "unavailable")

        positions_list = positions.get("positions", []) if isinstance(positions, dict) else []
        watched = set(symbols)
        for position in positions_list:
            symbol = str(position.get("symbol", "")).upper()
            if symbol and symbol not in watched:
                print(f"Note: leaving position {symbol} alone; it is not in my_stocks.txt.")
        headlines = {}
        for symbol in symbols:
            try:
                news = mcp("get_news", {"symbol": symbol,
                                        "limit": SETTINGS["NEWS_HEADLINES"]})
                items = news.get("headlines", []) if isinstance(news, dict) and not news.get("error") else []
                headlines[symbol] = [item.get("headline", "") for item in items
                                     if isinstance(item, dict) and item.get("headline")]
            except Exception:
                headlines[symbol] = []
        news_text = "\n".join(f"{symbol}: " + (" | ".join(items) if items else "no recent news")
                              for symbol, items in headlines.items())
        pos_text = json.dumps(positions_list, indent=1) if positions_list else "No open positions."

        # These five agents keep their existing responsibilities and handoffs.
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

        # News and Technical work in parallel, followed by the Market summary.
        news_task = Task(
            description=("Score the news sentiment for each stock from -1 (very negative) to +1 "
                         "(very positive). Use 0 when there is no news or it is neutral.\n\n"
                         "Headlines:\n" + news_text +
                         "\n\nReply with ONLY a JSON object, for example {\"AAPL\": 0.4, \"TSLA\": -0.2}. "
                         "Include every one of these tickers: " + ", ".join(symbols)),
            expected_output="A JSON object mapping each ticker to a number between -1 and 1.",
            agent=news_agent, async_execution=True)
        tech_task = Task(
            description=(f"Market regime: {regime} (buy threshold {buy_threshold}).\n"
                         "RAMS scores (0-100) with pillar scores:\n" + rams_table(technical) +
                         "\n\nWrite a short analysis with bullet points: the strongest stocks, the "
                         "weakest stocks, and any warnings (e.g. RSI above 70, weak volume)."),
            expected_output="A short markdown technical analysis.",
            agent=tech_agent, async_execution=True)
        market_task = Task(
            description=(f"Account: {json.dumps(account)}\nOpen positions: {pos_text}\n"
                         f"Market regime from SPY: {regime}.\n"
                         "Using the news and technical reports, write a short market summary."),
            expected_output="A short markdown market summary.",
            agent=market_agent, context=[news_task, tech_task])

        print("[2/5] News and Technical agents working in parallel...")
        first_crew = Crew(agents=[news_agent, tech_agent, market_agent],
                          tasks=[news_task, tech_task, market_task],
                          process=Process.sequential, max_rpm=SETTINGS["MAX_RPM"],
                          verbose=False, tracing=False)
        if not run_crew(first_crew):
            run_status = "First CrewAI run failed."
            return

        news_data, news_valid = parse_json_answer(news_task.output.raw)
        if not news_valid or not isinstance(news_data, dict):
            print("Warning: News agent response was not valid JSON; using neutral scores.")
            news_scores = {symbol: 0 for symbol in symbols}
        else:
            news_scores = {symbol: 0 for symbol in symbols}
            for key, value in news_data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    news_scores[str(key).upper()] = float(value)
        save(folder, "newsSentimentAgent", timestamp,
             "# News Sentiment Agent\n\n## Headlines\n" + news_text +
             "\n\n## Scores\n```json\n" + json.dumps(news_scores, indent=2) + "\n```\n")
        save(folder, "technicalAnalystAgent", timestamp,
             "# Technical Analyst Agent\n\n" + rams_table(technical) +
             "\n\n" + str(tech_task.output.raw))
        save(folder, "marketDataAgent", timestamp,
             "# Market Data Agent\n\n" + str(market_task.output.raw))

        print("[3/5] Final RAMS scores (technical + news)...")
        try:
            final = mcp("compute_rams", {"symbols": symbols, "news_scores": news_scores})
        except Exception as error:
            print("Start the server first: python servers/alpaca_trading_mcpServer.py")
            print(f"MCP error: {error}")
            run_status = f"MCP unavailable: {error}"
            return
        if isinstance(final, dict) and final.get("error"):
            print(final["error"])
            run_status = f"compute_rams error: {final['error']}"
            return
        regime = final.get("regime", regime)
        buy_threshold = final.get("buy_threshold", buy_threshold)
        print(rams_table(final, news_scores))
        plan = build_plan(final, positions, account, SETTINGS)
        if not plan:
            print("\n[4/5] No trades suggested this run.")
            run_status = "No trades proposed."
            save(folder, "traderAgent", timestamp,
                  "# Trader Agent\n\nNo trades this run - no stock met the buy/sell rules.\n\n"
                  "## Final RAMS table (including news)\n\n"
                  + rams_table(final, news_scores))
            execution_entries.append("No trades proposed.")
            return

        plan_text = "\n".join(
            f"{index}. {order['side'].upper()} {order['qty']} {order['symbol']} "
            f"@ limit {limit_price(order):.2f} (RAMS {order['rams']}) - {order['reason']}"
            for index, order in enumerate(plan, 1))

        # The Trader can select only numbered Python proposals.
        risk_task = Task(
            description=(f"Account: {json.dumps(account)}\nOpen positions: {pos_text}\n"
                         f"Regime: {regime}\nNews scores: {json.dumps(news_scores)}\n\n"
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
        second_crew = Crew(agents=[risk_agent, trader_agent], tasks=[risk_task, trade_task],
                           process=Process.sequential, max_rpm=SETTINGS["MAX_RPM"],
                           verbose=False, tracing=False)
        if not run_crew(second_crew):
            run_status = "Risk Manager and Trader CrewAI run failed."
            return

        chosen_data, chosen_valid = parse_json_answer(trade_task.output.raw)
        chosen_valid = (chosen_valid and isinstance(chosen_data, list)
                        and all(isinstance(item, int) and not isinstance(item, bool)
                                and 1 <= item <= len(plan) for item in chosen_data))
        if not chosen_valid:
            print("Warning: Trader response was not a valid list of order numbers; proposing no orders.")
            chosen = []
        else:
            chosen = sorted(set(chosen_data))
        final_orders = [plan[index - 1] for index in chosen]
        save(folder, "riskManagerAgent", timestamp,
             "# Risk Manager Agent\n\n## Proposed orders\n" + plan_text +
             "\n\n## Review\n" + str(risk_task.output.raw))
        save(folder, "traderAgent", timestamp,
             f"# Trader Agent\n\n## Final RAMS table (including news)\n\n"
             f"{rams_table(final, news_scores)}\n\n"
             f"Chosen order numbers: {chosen}\n\n" + str(trade_task.output.raw))

        print("\n[5/5] Final orders:")
        if not final_orders:
            print("   None - no orders selected by the Trader.")
            run_status = "No orders selected by the Trader."
            execution_entries.append(run_status)
            return
        if dry_run:
            for index, order in enumerate(final_orders, 1):
                print(order_line(order, index, len(final_orders)))
                execution_entries.append(order_line(order, index, len(final_orders)) + " - DRY RUN")
            print("DRY RUN - no orders placed")
            run_status = "Dry run; no orders placed."
            return

        for index, order in enumerate(final_orders, 1):
            line = order_line(order, index, len(final_orders))
            print(line)
            if not auto and not confirm_order():
                print("Order declined; moving to the next order.")
                execution_entries.append(line + " - DECLINED by user")
                continue
            try:
                if order["side"] == "sell":
                    mcp("cancel_all_orders", {"symbol": order["symbol"]})
                result = mcp("place_limit_order", {
                    "symbol": order["symbol"], "side": order["side"],
                    "qty": int(order["qty"]), "limit_price": limit_price(order),
                    "extended_hours": SETTINGS["EXTENDED_HOURS"],
                })
                status = result.get("status", "submitted") if isinstance(result, dict) else str(result)
                record = line + f" - {status}"
                print(f"   {status}")
                execution_entries.append(record)
            except Exception as error:
                record = line + f" - ERROR: {error}"
                print(f"   Order failed: {error}")
                execution_entries.append(record)
        run_status = "Order processing completed."
    except Exception as error:
        print(f"Run error: {error}")
        run_status = f"Run error: {error}"
        execution_entries.append(run_status)
    finally:
        save(folder, "settings_used", "",
             settings_report(now.isoformat(timespec="seconds"), flags, symbols,
                             regime, buy_threshold))
        if not execution_entries:
            execution_entries.append(run_status)
        save(folder, "executionLog", "",
             "# Execution Log\n\n" + "\n".join(f"- {entry}" for entry in execution_entries) + "\n")


if __name__ == "__main__":
    main()