"""CrewAI paper-trading crew.

Run with ``python clients/trading_crew.py`` for per-order confirmation,
``python clients/trading_crew.py --dry-run`` to print and log without trading,
or ``python clients/trading_crew.py --auto`` to place orders without prompts.

Settings from .env: MCP_URL (http://localhost:8001/mcp),
LLM_MODEL (groq/openai/gpt-oss-120b), TARGET_INVESTED (0.95, 0 to 1),
STOP_ATR (2.5, above 0), TAKE_PROFIT_ATR (3.0, above 0),
LIMIT_BUFFER (0.003), MAX_RPM (20), NEWS_HEADLINES (3),
EXTENDED_HOURS (false), CREW_VERBOSE (true), LLM_RETRIES (1) and LLM_BACKUP_MODELS
(comma list, tried in order when the main model fails or hits its daily limit).
Extra free fallbacks, used when set: OPENROUTER_API_KEY (+ OPENROUTER_MODEL) and
OLLAMA_URL + OLLAMA_MODEL (e.g. the college Ollama server). If every model is
unavailable the run continues with the rule-based plan, so trading never stops.
"""

import os
import sys
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
for _stream in (sys.stdout, sys.stderr):   # Windows consoles: print emojis safely
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import asyncio
import json
import math
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from crewai import Agent, Task, Crew, Process, LLM
from trading_safety import protective_stop, safe_buy_quantity

load_dotenv()

PROJECT = Path(__file__).resolve().parent.parent
STOCKS_FILE = PROJECT / "clients" / "my_stocks.txt"
# Each team member's records go to their own folder (TRADER_NAME in .env), so
# everyone can pull and push the same project without overwriting each other.
TRADER = re.sub(r"[^a-z0-9_-]", "", os.getenv("TRADER_NAME", "").strip().lower())
OUTPUTS = PROJECT / "outputs" / TRADER if TRADER else PROJECT / "outputs"


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
    # "careful": spread TARGET_INVESTED over healthy stocks only (rest stays cash);
    # "signal": the original rule (buy only on a BUY / ADD signal).
    "ALLOCATION": read_setting("ALLOCATION", "careful", str,
                               lambda value: value in ("careful", "signal")),
    # Backtest: buying at RAMS 60+ and selling below 30 beat 50 / 40 by a wide margin.
    "HEALTHY_MIN_RAMS": read_setting("HEALTHY_MIN_RAMS", 60.0, parse_finite_float),
    # Anti-churn rules (calendar days). Stop-losses are never delayed by these.
    "MIN_HOLD_DAYS": read_setting("MIN_HOLD_DAYS", 3, int, lambda value: value >= 0),
    "CONFIRM_BUY_DAYS": read_setting("CONFIRM_BUY_DAYS", 1, int, lambda value: value >= 1),
    "CONFIRM_SELL_DAYS": read_setting("CONFIRM_SELL_DAYS", 2, int, lambda value: value >= 1),
    "COOLDOWN_DAYS": read_setting("COOLDOWN_DAYS", 2, int, lambda value: value >= 0),
    "TAKE_HALF_PROFIT": read_setting("TAKE_HALF_PROFIT", False, parse_bool),
    "HEALTHY_MIN_TREND": read_setting("HEALTHY_MIN_TREND", 25.0, parse_finite_float),
    "MIN_TRADE_FRACTION": read_setting("MIN_TRADE_FRACTION", 0.02, parse_finite_float,
                                       lambda value: 0 <= value < 1),
    "STOP_ATR": read_setting("STOP_ATR", 2.5, parse_finite_float, lambda value: value > 0),
    "TAKE_PROFIT_ATR": read_setting("TAKE_PROFIT_ATR", 3.0, parse_finite_float,
                                    lambda value: value > 0),
    "LIMIT_BUFFER": read_setting("LIMIT_BUFFER", 0.003, parse_finite_float),
    "MAX_RPM": read_setting("MAX_RPM", 20, int, lambda value: value > 0),
    "NEWS_HEADLINES": read_setting("NEWS_HEADLINES", 3, int, lambda value: value >= 0),
    "EXTENDED_HOURS": read_setting("EXTENDED_HOURS", False, parse_bool),
    "LLM_RETRIES": read_setting("LLM_RETRIES", 1, int, lambda value: value >= 0),
    # Show each agent's live reasoning panels in the terminal (CrewAI verbose mode).
    "CREW_VERBOSE": read_setting("CREW_VERBOSE", True, parse_bool),
    "LLM_BACKUP_MODELS": read_setting(
        "LLM_BACKUP_MODELS",
        "groq/openai/gpt-oss-20b,groq/qwen/qwen3.8-27b,groq/llama-3.3-70b-versatile", str),
    "MAX_TRADE_RISK_FRACTION": read_setting(
        "MAX_TRADE_RISK_FRACTION", 0.015, parse_finite_float,
        lambda value: 0 < value <= 0.05),
    "MAX_POSITION_FRACTION": read_setting(
        "MAX_POSITION_FRACTION", 0.12, parse_finite_float,
        lambda value: 0 < value <= 1),
    "MAX_HIGH_BETA_POSITION_FRACTION": read_setting(
        "MAX_HIGH_BETA_POSITION_FRACTION", 0.08, parse_finite_float,
        lambda value: 0 < value <= 1),
    "PROTECTIVE_STOP_FRACTION": read_setting(
        "PROTECTIVE_STOP_FRACTION", 0.06, parse_finite_float,
        lambda value: 0 < value < 1),
}
HIGH_BETA_SYMBOLS = {
    symbol.strip().upper()
    for symbol in os.getenv("HIGH_BETA_SYMBOLS", "MU,XYZ,HOOD,SMCI").split(",")
    if symbol.strip()
}

def make_llm(model):
    if model.startswith("openrouter/"):
        return LLM(model=model, api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.0)
    if model.startswith("ollama/"):
        return LLM(model=model, base_url=os.getenv("OLLAMA_URL"), temperature=0.0, timeout=300)
    return LLM(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0.0)


# Main model first, then the backups. Each free Groq model has its own daily
# token quota, so a model that hits its limit is skipped for the rest of the run.
MODELS = [SETTINGS["LLM_MODEL"]] + [
    m.strip() for m in SETTINGS["LLM_BACKUP_MODELS"].split(",")
    if m.strip() and m.strip() != SETTINGS["LLM_MODEL"]]
if os.getenv("OPENROUTER_API_KEY"):
    MODELS.append("openrouter/" + os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"))
if os.getenv("OLLAMA_URL") and os.getenv("OLLAMA_MODEL"):
    MODELS.append("ollama/" + os.getenv("OLLAMA_MODEL"))

# Models that used up their free daily quota are skipped until the next UTC day.
LIMITS_FILE = OUTPUTS / "llm_daily_limits.json"


def _today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def load_daily_limits():
    try:
        data = json.loads(LIMITS_FILE.read_text(encoding="utf-8"))
        return set(data.get(_today(), []))
    except (OSError, ValueError, AttributeError):
        return set()


def mark_daily_limit(model):
    try:
        LIMITS_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIMITS_FILE.write_text(json.dumps({_today(): sorted(load_daily_limits() | {model})}), encoding="utf-8")
    except OSError:
        pass


DEAD_MODELS = load_daily_limits()
llm = make_llm(MODELS[0])


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
    syms = [l.strip().upper() for l in STOCKS_FILE.read_text(encoding="utf-8-sig").splitlines()]
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
    path.write_text(text, encoding="utf-8")
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


def capped_split(scores, total, caps):
    """Split `total` in proportion to `scores`, no name above its cap.

    Money a capped name cannot take is shared among the others.
    """
    alloc, left, remaining = {}, dict(scores), total
    while left:
        weight = sum(left.values())
        if weight <= 0:
            break
        over = [k for k, v in left.items() if remaining * v / weight >= caps[k]]
        if not over:
            for k, v in left.items():
                alloc[k] = remaining * v / weight
            break
        for k in over:
            alloc[k] = caps[k]
            remaining -= caps[k]
            del left[k]
    return alloc


def is_healthy(stock, settings):
    """True when a stock passes the buy rules (RAMS, trend, news, no SELL signal)."""
    news = number(stock.get("news_points"), 50.0) / 50.0 - 1.0
    return (number(stock.get("rams")) >= settings.get("HEALTHY_MIN_RAMS", 50.0)
            and number(stock.get("trend")) >= settings.get("HEALTHY_MIN_TREND", 25.0)
            and news > -0.25 and stock.get("signal") != "SELL")


def healthy_targets(info, held, equity, settings, skip):
    """Target value per healthy stock (careful mode).

    Healthy = RAMS at least HEALTHY_MIN_RAMS, trend pillar at least
    HEALTHY_MIN_TREND (not in a downtrend), news not bad, no SELL signal.
    TARGET_INVESTED of equity is spread over them by RAMS, each capped at
    MAX_POSITION_FRACTION (high-beta: MAX_HIGH_BETA_POSITION_FRACTION).
    With few healthy stocks the caps bind and the rest stays in cash.
    """
    scores, caps = {}, {}
    for symbol, stock in info.items():
        if symbol in skip:
            continue
        if is_healthy(stock, settings):
            scores[symbol] = number(stock.get("rams"))
            frac = (settings.get("MAX_HIGH_BETA_POSITION_FRACTION", 0.08) if symbol in HIGH_BETA_SYMBOLS
                    else settings.get("MAX_POSITION_FRACTION", 0.12))
            caps[symbol] = frac * equity
    return capped_split(scores, settings["TARGET_INVESTED"] * equity, caps)


STATE_FILE = OUTPUTS / "strategy_state.json"

# Used only to give the Risk Manager real exposure numbers instead of guesses.
SECTORS = {"MU": "Semiconductors", "AVGO": "Semiconductors", "SMCI": "Tech hardware",
           "AMZN": "Consumer / cloud", "GOOGL": "Communication / tech",
           "XYZ": "Payments / fintech", "V": "Payments / fintech", "HOOD": "Fintech / brokerage",
           "FCX": "Materials", "GE": "Industrials"}


def _days_between(older, newer):
    """Whole days from one YYYY-MM-DD date to another (huge number if unreadable)."""
    try:
        return (date.fromisoformat(newer) - date.fromisoformat(older)).days
    except (TypeError, ValueError):
        return 10 ** 6


def load_state():
    """Memory that survives between runs: entry dates, signal streaks, sell dates, news."""
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    except OSError:
        pass


def update_signal_days(state, final, settings, today):
    """Record each day a stock passed the buy rules / showed a SELL signal.

    A day counts once however many runs happen in it; one failed check resets the streak.
    """
    for kind in ("buy_days", "sell_days"):
        state.setdefault(kind, {})
    for stock in final.get("stocks", []):
        if "error" in stock or not stock.get("symbol"):
            continue
        symbol = str(stock["symbol"]).upper()
        checks = (("buy_days", is_healthy(stock, settings)),
                  ("sell_days", stock.get("signal") == "SELL"))
        for kind, passed in checks:
            days = set(state[kind].get(symbol, []))
            state[kind][symbol] = sorted(days | {today})[-5:] if passed else []


def signal_confirmed(state, kind, symbol, need):
    """True when the signal has held on at least `need` different days."""
    return len(set(state.get(kind, {}).get(symbol, []))) >= need


def hold_ok(state, symbol, today, settings):
    """True when a position has been held for at least MIN_HOLD_DAYS (unknown = old)."""
    entry = state.get("entry", {}).get(symbol)
    if not entry:
        return True
    return _days_between(entry, today) >= settings.get("MIN_HOLD_DAYS", 0)


def sync_entries(state, positions_list, pending_buys, settings, today):
    """Keep an entry date for every held stock. Stocks first seen are treated as old."""
    entry = state.setdefault("entry", {})
    held = {str(p.get("symbol", "")).upper() for p in positions_list}
    old = (date.fromisoformat(today) - timedelta(days=settings.get("MIN_HOLD_DAYS", 0))).isoformat()
    for symbol in held:
        entry.setdefault(symbol, old)
    for symbol in list(entry):
        if symbol not in held and symbol not in pending_buys:
            del entry[symbol]


def exposure_report(positions_list, plan, account):
    """Real exposure numbers (after the proposed orders) for the Risk Manager."""
    equity = number(account.get("equity"))
    if equity <= 0:
        return "Exposure facts unavailable."
    value = {str(p.get("symbol", "")).upper(): number(p.get("market_value")) for p in positions_list}
    for order in plan:
        amount = order["qty"] * order["price"]
        value[order["symbol"]] = value.get(order["symbol"], 0.0) + (
            amount if order["side"] == "buy" else -amount)
    value = {symbol: max(0.0, amount) for symbol, amount in value.items()}
    invested = sum(value.values())
    sectors = {}
    for symbol, amount in value.items():
        sector = SECTORS.get(symbol, "Other")
        sectors[sector] = sectors.get(sector, 0.0) + amount
    high_beta = sum(amount for symbol, amount in value.items() if symbol in HIGH_BETA_SYMBOLS)
    top = sorted(value.items(), key=lambda kv: -kv[1])[:5]
    return "\n".join([
        f"After these orders (computed by code, do not recompute): invested "
        f"{100 * invested / equity:.0f}% of equity, cash {100 * (1 - invested / equity):.0f}%, "
        f"high-beta stocks {100 * high_beta / equity:.0f}% (the contest wants at least 30%).",
        "Largest positions: " + ", ".join(f"{s} {100 * v / equity:.0f}%" for s, v in top if v > 0),
        "By sector: " + ", ".join(f"{k} {100 * v / equity:.0f}%" for k, v in
                                  sorted(sectors.items(), key=lambda kv: -kv[1]) if v > 0)])


def build_plan(final, positions, account, settings, profit_taken=None, pending_buys=(),
               state=None, today=None):
    """Build orders from supplied data only; this function makes no network calls.

    profit_taken: {symbol: entry price} of positions that already sold half,
    so the half is taken once and the rest keeps a breakeven stop.
    pending_buys: symbols with an unfilled buy order (never buy twice).
    state/today: run-to-run memory (see load_state). With state=None the anti-churn
    rules (minimum hold, confirmation, cooldown) are off.
    """
    profit_taken = profit_taken or {}
    use_state = state is not None
    state = state or {}
    today = today or _today()
    cooling = set()
    if use_state:
        cooling = {symbol for symbol, day in state.get("last_sold", {}).items()
                   if _days_between(day, today) < settings.get("COOLDOWN_DAYS", 0)}
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
        took_half = symbol in profit_taken   # keyed by symbol: an average-price change must not reset it
        signal_ok = (not use_state or (
            hold_ok(state, symbol, today, settings)
            and signal_confirmed(state, "sell_days", symbol, settings.get("CONFIRM_SELL_DAYS", 1))))
        if price <= entry - settings["STOP_ATR"] * atr:
            order.update(qty=qty, reason="stop-loss hit")
            stop_sells.append(order)
        elif took_half and price <= entry:
            order.update(qty=qty, reason="breakeven stop after taking profit")
            stop_sells.append(order)
        elif stock.get("signal") == "SELL" and signal_ok:
            order.update(qty=qty, reason=f"RAMS {stock.get('rams')} below sell level")
            signal_sells.append(order)
        elif (settings.get("TAKE_HALF_PROFIT", True) and not took_half
              and price >= entry + settings["TAKE_PROFIT_ATR"] * atr and qty >= 2):
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
    if settings.get("ALLOCATION") == "careful":
        targets = healthy_targets(info, held, equity, settings,
                                  skip=planned_symbols | set(pending_buys) | cooling)
        for symbol, target in sorted(targets.items(), key=lambda kv: -kv[1]):
            stock, price = info[symbol], number(info[symbol].get("price"))
            current = number(held[symbol].get("market_value")) if symbol in held else 0.0
            if use_state and not signal_confirmed(state, "buy_days", symbol,
                                                  settings.get("CONFIRM_BUY_DAYS", 1)):
                continue   # the buy signal has not held long enough yet
            if symbol in held and number(held[symbol].get("unrealized_pl")) < 0:
                continue   # careful: never add to a losing position
            if price <= 0 or target - current < settings.get("MIN_TRADE_FRACTION", 0.02) * equity:
                continue
            try:
                stop_price = protective_stop(price, settings.get("PROTECTIVE_STOP_FRACTION", 0.06))
            except (TypeError, ValueError):
                continue
            safe_qty = safe_buy_quantity(
                price=price, stop_price=stop_price, equity=equity,
                current_position_value=current, high_beta=symbol in HIGH_BETA_SYMBOLS,
                max_risk_fraction=settings.get("MAX_TRADE_RISK_FRACTION", 0.015),
                max_position_fraction=settings.get("MAX_POSITION_FRACTION", 0.12),
                high_beta_fraction=settings.get("MAX_HIGH_BETA_POSITION_FRACTION", 0.08))
            qty = min(int((target - current) // price), safe_qty, int(budget // price))
            if qty <= 0:
                continue
            buys.append({
                "symbol": symbol, "side": "buy", "qty": qty, "price": price,
                "rams": number(stock.get("rams")), "stop_loss": stock.get("stop_loss"),
                "rsi": number(stock.get("rsi"), 50.0),
                "reason": (f"healthy stock (RAMS {stock.get('rams')}, trend {stock.get('trend')}): "
                           f"target {100 * target / equity:.0f}% of equity"),
            })
            planned_symbols.add(symbol)
            budget -= qty * price
        return sells + buys
    for stock in sorted(info.values(), key=lambda item: number(item.get("rams")), reverse=True):
        symbol = str(stock.get("symbol", "")).upper()
        price = number(stock.get("price"))
        if symbol in planned_symbols or symbol in pending_buys or symbol in cooling or price <= 0:
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
        # Apply deterministic risk and position limits before proposing a buy.
        try:
            stop_price = protective_stop(
                price, settings.get("PROTECTIVE_STOP_FRACTION", 0.06)
            )
        except (TypeError, ValueError):
            continue

        current_value = (
            number(held[symbol].get("market_value")) if symbol in held else 0.0
        )
        safe_qty = safe_buy_quantity(
            price=price,
            stop_price=stop_price,
            equity=equity,
            current_position_value=current_value,
            high_beta=symbol in HIGH_BETA_SYMBOLS,
            max_risk_fraction=settings.get("MAX_TRADE_RISK_FRACTION", 0.015),
            max_position_fraction=settings.get("MAX_POSITION_FRACTION", 0.12),
            high_beta_fraction=settings.get("MAX_HIGH_BETA_POSITION_FRACTION", 0.08),
        )
        qty = min(qty, safe_qty, int(budget // price))
        if qty <= 0:
            continue
        buys.append({
            "symbol": symbol, "side": "buy", "qty": qty, "price": price,
            "rams": number(stock.get("rams")), "stop_loss": stock.get("stop_loss"),
            "rsi": number(stock.get("rsi"), 50.0), "reason": reason,
        })
        planned_symbols.add(symbol)
        budget -= qty * price
    return sells + buys


PROFIT_FILE = OUTPUTS / "profit_taken.json"


def load_profit_taken(positions_list):
    """{symbol: entry} of positions that already took half profit (still held)."""
    try:
        data = json.loads(PROFIT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    held = {str(p.get("symbol", "")).upper() for p in positions_list}
    return {sym: entry for sym, entry in data.items() if sym in held}


def save_profit_taken(data):
    try:
        PROFIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROFIT_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def task_text(task):
    """A task's answer, or '' when the AI did not run."""
    output = getattr(task, "output", None)
    return str(getattr(output, "raw", "") or "") if output is not None else ""


VETO_NEWS = -0.25   # a buy may be vetoed for bad news only at or below this news score
VETO_RSI = 80.0     # ... or for being extremely overbought only at or above this RSI


def buy_veto_evidence(order, news_scores):
    """Hard evidence that justifies skipping a rule-approved buy, or ''."""
    news = number(news_scores.get(order["symbol"], 0))
    if news <= VETO_NEWS:
        return f"bad news ({news:+.2f})"
    if number(order.get("rsi"), 50.0) >= VETO_RSI:
        return f"extremely overbought (RSI {number(order.get('rsi')):.1f})"
    return ""


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


def model_chain_text():
    usable = [m for m in MODELS if m not in DEAD_MODELS]
    return " -> ".join(usable + ["rule-based plan"])


def run_crew(crew):
    """Run a Crew, falling back through the backup models.

    A short per-minute rate limit is retried after a pause; a daily limit,
    an unknown model or any other error moves straight to the next model.
    Returns False only when every model failed (the caller then continues
    with the rule-based plan, so stop-losses still execute).
    """
    for model in MODELS:
        if model in DEAD_MODELS:
            continue
        model_llm = make_llm(model)
        for agent in crew.agents:
            agent.llm = model_llm
        for attempt in range(SETTINGS["LLM_RETRIES"] + 1):
            try:
                crew.kickoff()
                if model != MODELS[0]:
                    print(f"   (used backup model {model})")
                return True
            except Exception as error:
                message = str(error)
                lowered = message.lower()
                quota = "per day" in lowered or "tpd" in lowered or "rpd" in lowered
                daily = (quota or "model_not_found" in lowered or "does not exist" in lowered
                         or "decommissioned" in lowered or "invalid api key" in lowered)
                short_limit = "429" in message or "rate" in lowered
                if short_limit and not daily and attempt < SETTINGS["LLM_RETRIES"]:
                    print(f"LLM rate limit on {model}; retrying in 20 seconds...")
                    time.sleep(20)
                    continue
                print(f"LLM {model} unavailable ({message[:120]}); trying the next model.")
                DEAD_MODELS.add(model)
                if quota:
                    mark_daily_limit(model)
                break
    print("All AI models unavailable this run; using the rule-based plan.")
    return False


def limit_price(order):
    """Calculate a buffered limit price; orders are always limit orders."""
    direction = 1 if order["side"] == "buy" else -1
    buffer = SETTINGS["LIMIT_BUFFER"]
    if order["side"] == "sell" and "stop" in str(order.get("reason", "")):
        buffer = max(buffer, 0.015)   # a stop-loss exit must fill even in a fast drop
    return round(order["price"] * (1 + direction * buffer), 2)


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
    dry_run_setting = os.getenv("DRY_RUN", "true").strip().lower()
    trading_enabled = os.getenv("TRADING_ENABLED", "false").strip().lower() == "true"

    # Fail closed: explicit CLI dry-run or environment safety settings block orders.
    dry_run = (
        "--dry-run" in sys.argv
        or dry_run_setting != "false"
        or not trading_enabled
    )
    if not trading_enabled:
        print("SAFETY: TRADING_ENABLED is not true; orders are disabled.")
    elif dry_run:
        print("SAFETY: DRY_RUN is enabled; no orders will be placed.")
    flags = [flag for flag, enabled in (("--auto", auto), ("--dry-run", dry_run)) if enabled]
    now = datetime.now()
    timestamp = now.strftime("%d%m%Y-%H-%M")
    folder = OUTPUTS / f"run_{now.strftime('%d%m%Y-%H-%M')}"
    folder.mkdir(parents=True, exist_ok=True)
    symbols, regime, buy_threshold = [], "unavailable", "unavailable"
    execution_entries = []
    run_status = "Run ended before orders were selected."
    state, today = {}, _today()
    try:
        symbols = load_symbols()
        print(f"AI models today: {model_chain_text()}")
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
        try:
            open_orders = mcp("get_open_orders").get("orders", [])
        except Exception:
            open_orders = []
        pending_buys = {str(o.get("symbol", "")).upper() for o in open_orders
                        if "buy" in str(o.get("side", "")).lower()}
        profit_taken = load_profit_taken(positions_list)
        state = load_state()
        sync_entries(state, positions_list, pending_buys, SETTINGS, today)
        if not dry_run:
            # Every share must have a live (GTC) protective stop; DAY stops expire overnight.
            try:
                stops = mcp("ensure_protective_stops")
                for line in (stops.get("actions", []) if isinstance(stops, dict) else []):
                    print(f"   Stop check: {line}")
                    execution_entries.append(f"Stop check: {line}")
            except Exception as error:
                print(f"   Stop check skipped: {error}")
        held_before = {str(p.get("symbol", "")).upper(): number(p.get("qty")) for p in positions_list}
        ai_ok = True
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
        common = dict(llm=llm, allow_delegation=False, verbose=SETTINGS["CREW_VERBOSE"])
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
            backstory=("You make the final decisions for an investor in a short paper-trading "
                       f"contest judged on % gain: about {100 * SETTINGS['TARGET_INVESTED']:.0f}% "
                       "invested in healthy stocks. Exits come from fixed rules and always run."),
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
                          verbose=SETTINGS["CREW_VERBOSE"], tracing=False)
        if not run_crew(first_crew):
            ai_ok = False
            print("   AI analysts unavailable: neutral news scores, rules continue.")
            execution_entries.append("AI analysts unavailable; neutral news, rule-based plan.")

        news_data, news_valid = parse_json_answer(task_text(news_task))
        if not news_valid or not isinstance(news_data, dict):
            print("Warning: News agent response was not valid JSON; using neutral scores.")
            news_scores = {symbol: 0 for symbol in symbols}
        else:
            news_scores = {symbol: 0 for symbol in symbols}
            for key, value in news_data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    news_scores[str(key).upper()] = float(value)
            # Smooth with the previous run so one odd AI answer cannot swing a score.
            previous = state.get("news", {}) if isinstance(state.get("news"), dict) else {}
            for key in list(news_scores):
                if key in previous and isinstance(previous[key], (int, float)):
                    news_scores[key] = round(0.5 * float(previous[key]) + 0.5 * news_scores[key], 2)
            state["news"] = dict(news_scores)
        save(folder, "newsSentimentAgent", timestamp,
             "# News Sentiment Agent\n\n## Headlines\n" + news_text +
             "\n\n## Scores\n```json\n" + json.dumps(news_scores, indent=2) + "\n```\n")
        save(folder, "technicalAnalystAgent", timestamp,
             "# Technical Analyst Agent\n\n" + rams_table(technical) +
             "\n\n" + (task_text(tech_task) or "(AI unavailable this run)"))
        save(folder, "marketDataAgent", timestamp,
             "# Market Data Agent\n\n" + (task_text(market_task) or "(AI unavailable this run)"))

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
        update_signal_days(state, final, SETTINGS, today)
        save_state(state)
        plan = build_plan(final, positions, account, SETTINGS,
                          profit_taken=profit_taken, pending_buys=pending_buys,
                          state=state, today=today)
        if pending_buys:
            print(f"   Waiting for earlier buy order(s) to fill: {', '.join(sorted(pending_buys))}")
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
            f"@ limit {limit_price(order):.2f} (RAMS {order['rams']}"
            + (f", RSI {number(order.get('rsi')):.1f}, news {number(news_scores.get(order['symbol'], 0)):+.2f}"
               if order["side"] == "buy" else "")
            + f") - {order['reason']}"
            for index, order in enumerate(plan, 1))

        # The Trader can select only numbered Python proposals.
        risk_task = Task(
            description=(f"Account: {json.dumps(account)}\nOpen positions: {pos_text}\n"
                         f"Regime: {regime}\nNews scores: {json.dumps(news_scores)}\n\n"
                         f"Proposed orders:\n{plan_text}\n\n"
                         f"Exposure facts:\n{exposure_report(positions_list, plan, account)}\n\n"
                         "How to read the numbers: RAMS is a 0-100 STRENGTH score (higher is better; "
                         f"buys go only to healthy stocks: RAMS {SETTINGS['HEALTHY_MIN_RAMS']:.0f}+, "
                         "not in a downtrend, no bad news). Overbought is "
                         "judged by RSI only: extremely overbought means RSI 80 or more. Very bad "
                         f"news means a news score of {VETO_NEWS} or lower. Sizes and the 6% "
                         "protective stop were already set by the risk rules.\n"
                         "SELL orders reduce risk and come from fixed rules: always APPROVE every "
                         "SELL, never reject one. Review BUY orders only.\n"
                         f"The investor wants about {100 * SETTINGS['TARGET_INVESTED']:.0f}% invested "
                         "and cash earns nothing. APPROVE each BUY unless one of these applies: "
                         "very bad news, RSI 80 or more, or the exposure facts above show more than "
                         "40% of equity in one sector. Use the exposure facts exactly as given; "
                         "do not make up numbers. Give a short reason for each."),
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
                           verbose=SETTINGS["CREW_VERBOSE"], tracing=False)
        if not (ai_ok and run_crew(second_crew)):
            ai_ok = False
        chosen_data, chosen_valid = parse_json_answer(task_text(trade_task))
        chosen_valid = (ai_ok and chosen_valid and isinstance(chosen_data, list)
                        and all(isinstance(item, int) and not isinstance(item, bool)
                                and 1 <= item <= len(plan) for item in chosen_data))
        if not chosen_valid:
            # No usable AI decision: the Python plan already passed every risk limit.
            print("   No valid Trader answer; executing the rule-based plan.")
            chosen = list(range(1, len(plan) + 1))
        else:
            chosen = sorted(set(chosen_data))
        # A rule-approved buy is skipped only with hard evidence (bad news or RSI 80+).
        if chosen_valid:
            for i, order in enumerate(plan, 1):
                if order["side"] == "buy" and i not in chosen:
                    evidence = buy_veto_evidence(order, news_scores)
                    if evidence:
                        print(f"   Trader skipped {order['symbol']}: {evidence}.")
                    else:
                        print(f"   Trader skipped {order['symbol']} without evidence "
                              f"(news {number(news_scores.get(order['symbol'], 0)):+.2f}, "
                              f"RSI {number(order.get('rsi')):.1f}); keeping the buy.")
                        execution_entries.append(f"{order['symbol']} buy kept: AI veto had no evidence")
                        chosen.append(i)
        # Every SELL comes from fixed rules (stop, weak signal, profit target) and is
        # never skipped, whatever the AI said: it only lowers risk.
        chosen = sorted(set(chosen) | {i for i, order in enumerate(plan, 1)
                                       if order["side"] == "sell"})
        final_orders = [plan[index - 1] for index in chosen]
        save(folder, "riskManagerAgent", timestamp,
             "# Risk Manager Agent\n\n## Proposed orders\n" + plan_text +
             "\n\n## Review\n" + (task_text(risk_task) or "(AI unavailable: rule-based plan)"))
        save(folder, "traderAgent", timestamp,
             f"# Trader Agent\n\n## Final RAMS table (including news)\n\n"
             f"{rams_table(final, news_scores)}\n\n"
             f"Chosen order numbers: {chosen}\n\n" + task_text(trade_task))

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
                result = mcp("place_limit_order", {
                    "symbol": order["symbol"], "side": order["side"],
                    "qty": int(order["qty"]), "limit_price": limit_price(order),
                    "extended_hours": SETTINGS["EXTENDED_HOURS"],
                })
                status = result.get("status", "submitted") if isinstance(result, dict) else str(result)
                if order["side"] == "sell" and order["reason"] == "take half profit":
                    entry = next((number(p.get("avg_entry_price")) for p in positions_list
                                  if str(p.get("symbol", "")).upper() == order["symbol"]), 0)
                    profit_taken[order["symbol"]] = entry
                    save_profit_taken(profit_taken)
                if order["side"] == "sell" and order["qty"] >= held_before.get(order["symbol"], 0):
                    state.setdefault("last_sold", {})[order["symbol"]] = today   # starts the cooldown
                if order["side"] == "buy" and order["symbol"] not in held_before:
                    state.setdefault("entry", {})[order["symbol"]] = today       # starts the minimum hold
                save_state(state)
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