# Alpaca PAPER trading MCP server
# Professor's original tools + Person 1 additions:
#   get_bars_multi, compute_rams, get_news, place_limit_order, get_all_positions

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from fastmcp import FastMCP
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config

load_dotenv()

mcp = FastMCP("alpaca-paper-trading")

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

_trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
_data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
_news_client = NewsClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


# ---------------- helper functions ----------------

def _order_to_dict(order) -> dict:
    return {
        "order_id": str(order.id),
        "symbol": order.symbol,
        "qty": str(order.qty),
        "side": str(order.side),
        "status": str(order.status),
        "submitted_at": str(order.submitted_at),
    }


def _fetch_bars(symbols):
    """ONE Alpaca call for all symbols + SPY. Returns {symbol: DataFrame}."""
    syms = sorted({s.upper().strip() for s in symbols} | {"SPY"})
    req = StockBarsRequest(
        symbol_or_symbols=syms,
        timeframe=TimeFrame.Hour,
        start=datetime.now(timezone.utc) - timedelta(days=config.HISTORY_DAYS),
        feed=DataFeed.IEX,  # free data feed for paper accounts
    )
    df = _data_client.get_stock_bars(req).df
    out = {}
    available = set(df.index.get_level_values(0))
    for s in syms:
        if s in available:
            out[s] = df.xs(s, level=0).sort_index()
    return out


def _col(frame, prefix):
    """Find a pandas_ta column by its name start (names differ by version)."""
    for c in frame.columns:
        if str(c).startswith(prefix):
            return frame[c]
    raise KeyError(f"column starting with {prefix} not found")


def _last(series, n=1):
    """Value n bars back from the end, as a normal float."""
    try:
        return float(series.iloc[-n])
    except Exception:
        return float("nan")


def _pts(condition, points):
    return points if condition else 0


def _score_symbol(d, spy_close):
    """Calculate all indicators and the 4 technical pillar scores for one stock."""
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    v = d["volume"].astype(float)
    price = _last(c)

    # --- indicators (pandas_ta) ---
    ema20, ema50, ema200 = ta.ema(c, 20), ta.ema(c, 50), ta.ema(c, 200)
    adx = ta.adx(h, l, c, 14)
    rsi = ta.rsi(c, 14)
    macd = ta.macd(c)
    roc10 = ta.roc(c, 10)
    tsi = ta.tsi(c)
    cmf = ta.cmf(h, l, c, v, length=20)
    obv = ta.obv(c, v)
    obv_ema = ta.ema(obv, 20)
    mfi = ta.mfi(h, l, c, v, 14)
    pvo = ta.pvo(v)
    atr = ta.atr(h, l, c, 14)
    bb = ta.bbands(c, 20, 2)
    dc = ta.donchian(h, l, 20, 20)
    z = ta.zscore(c, 20)

    # --- Trend pillar (0-100) ---
    trend = 0
    trend += _pts(price > _last(ema20), 25)
    trend += _pts(_last(ema20) > _last(ema50), 25)
    trend += _pts(_last(ema50) > _last(ema200), 25)
    trend += _pts(_last(_col(adx, "ADX_")) > 25 and
                  _last(_col(adx, "DMP_")) > _last(_col(adx, "DMN_")), 25)

    # --- Momentum pillar (0-100) ---
    r = _last(rsi)
    hist = _col(macd, "MACDh_")
    rel_strength = _last(c.pct_change(20)) - _last(spy_close.pct_change(20))
    mom = 0
    mom += _pts(50 <= r <= 70, 20)
    mom += _pts(_last(hist) > 0 and _last(hist) > _last(hist, 2), 20)
    mom += _pts(_last(roc10) > 0, 20)
    mom += _pts(_last(_col(tsi, "TSI_")) > _last(_col(tsi, "TSIs_")), 20)
    mom += _pts(rel_strength > 0, 20)

    # --- Volume pillar (0-100) ---
    vol = 0
    vol += _pts(_last(cmf) > 0, 25)
    vol += _pts(_last(obv) > _last(obv_ema), 25)
    vol += _pts(50 <= _last(mfi) <= 80, 25)
    vol += _pts(_last(_col(pvo, "PVO_")) > 0, 25)

    # --- Breakout pillar (0-100) ---
    bbw = _col(bb, "BBB_")
    bbp = _last(_col(bb, "BBP_"))
    dcl, dcu = _last(_col(dc, "DCL_")), _last(_col(dc, "DCU_"))
    dc_pos = (price - dcl) / (dcu - dcl) if dcu > dcl else 0
    brk = 0
    brk += _pts(_last(bbw) > _last(bbw, 5), 33)
    brk += _pts(dc_pos >= 0.8, 33)
    brk += _pts(0.5 <= bbp <= 1, 34)

    # technical score out of 100 (the 4 pillars only, news added later)
    tech_weight = config.W_TREND + config.W_MOMENTUM + config.W_VOLUME + config.W_BREAKOUT
    technical = (config.W_TREND * trend + config.W_MOMENTUM * mom +
                 config.W_VOLUME * vol + config.W_BREAKOUT * brk) / tech_weight

    zval = _last(z)
    penalty = config.ZSCORE_PENALTY_POINTS if zval > config.ZSCORE_PENALTY_LEVEL else 0

    return {
        "price": round(price, 2),
        "atr": round(_last(atr), 4),
        "trend": trend, "momentum": mom, "volume": vol, "breakout": brk,
        "technical_score": round(technical, 1),
        "rsi": round(r, 1),
        "zscore": round(zval, 2),
        "penalty": penalty,
    }


# ---------------- NEW TOOLS (Person 1) ----------------

@mcp.tool()
def get_bars_multi(symbols: list[str]) -> dict:
    """Fetch 1-hour bars for ALL symbols (plus SPY) in ONE Alpaca call.
    Returns a short summary per symbol.

    Args:
        symbols: List of tickers, e.g. ["AAPL", "NVDA", "TSLA"].
    """
    bars = _fetch_bars(symbols)
    out = {}
    for s, d in bars.items():
        out[s] = {
            "bars": len(d),
            "last_time": str(d.index[-1]),
            "last_close": round(float(d["close"].iloc[-1]), 2),
            "change_20_bars_pct": round(_last(d["close"].pct_change(20)) * 100, 2),
        }
    return out


@mcp.tool()
def compute_rams(symbols: list[str], news_scores: dict[str, float] | None = None) -> dict:
    """Calculate the RAMS score (0-100), signal and suggested quantity for each stock.

    Args:
        symbols: List of tickers, e.g. ["AAPL", "NVDA"].
        news_scores: Optional news sentiment per ticker from -1 (bad) to +1 (good),
                     e.g. {"AAPL": 0.4}. Missing tickers count as neutral (0).
    """
    bars = _fetch_bars(symbols)
    if "SPY" not in bars:
        return {"error": "Could not download SPY data."}

    # market regime from SPY
    spy = bars["SPY"]["close"].astype(float)
    bull = _last(spy) > _last(ta.ema(spy, 50))
    buy_th = config.BUY_THRESHOLD_BULL if bull else config.BUY_THRESHOLD_BEAR

    equity = float(_trading_client.get_account().equity)
    held = {p.symbol for p in _trading_client.get_all_positions()}
    news_scores = news_scores or {}

    stocks = []
    for s in sorted({x.upper().strip() for x in symbols}):
        if s not in bars or len(bars[s]) < 60:
            stocks.append({"symbol": s, "error": "not enough data"})
            continue
        try:
            r = _score_symbol(bars[s], spy)
        except Exception as e:
            stocks.append({"symbol": s, "error": str(e)})
            continue

        # news pillar: -1..+1  ->  0..100
        news = max(-1.0, min(1.0, float(news_scores.get(s, 0.0))))
        news_pts = (news + 1) * 50
        rams = (1 - config.W_NEWS) * r["technical_score"] + config.W_NEWS * news_pts
        rams = round(max(0, min(100, rams - r["penalty"])), 1)

        # signal
        if s in held and rams < config.SELL_THRESHOLD:
            signal = "SELL"
        elif s in held and rams >= config.ADD_THRESHOLD:
            signal = "ADD"
        elif s not in held and rams >= buy_th:
            signal = "BUY"
        elif s in held:
            signal = "HOLD"
        else:
            signal = "WAIT"

        # position size (whole shares)
        if s in config.HIGH_BETA:
            cap = config.MAX_WEIGHT_HIGH_BETA_BULL if bull else config.MAX_WEIGHT_HIGH_BETA_BEAR
        else:
            cap = config.MAX_WEIGHT
        atr, price = r["atr"], r["price"]
        qty_by_risk = (equity * config.RISK_PER_TRADE) / (config.ATR_STOP_MULT * atr) if atr > 0 else 0
        qty_by_cap = (equity * cap) / price if price > 0 else 0
        qty = int(max(0, min(qty_by_risk, qty_by_cap)))

        stocks.append({
            "symbol": s,
            "rams": rams,
            "signal": signal,
            "held": s in held,
            "high_beta": s in config.HIGH_BETA,
            "suggested_qty": qty,
            "stop_loss": round(price - config.ATR_STOP_MULT * atr, 2),
            "news_points": round(news_pts, 1),
            **r,
        })

    stocks.sort(key=lambda x: x.get("rams", -1), reverse=True)
    return {
        "regime": "BULL" if bull else "BEAR",
        "buy_threshold": buy_th,
        "equity": round(equity, 2),
        "stocks": stocks,
    }


@mcp.tool()
def get_news(symbol: str, limit: int = 5) -> dict:
    """Get the latest news headlines for one stock from Alpaca.

    Args:
        symbol: Ticker, e.g. "AAPL".
        limit: How many headlines (default 5).
    """
    try:
        result = _news_client.get_news(NewsRequest(symbols=symbol.upper(), limit=limit))
        items = result.data.get("news", [])
        return {
            "symbol": symbol.upper(),
            "headlines": [
                {"headline": n.headline, "summary": n.summary, "time": str(n.created_at)}
                for n in items
            ],
        }
    except Exception as e:
        return {"symbol": symbol.upper(), "headlines": [], "error": str(e)}


@mcp.tool()
def place_limit_order(symbol: str, side: str, qty: int, limit_price: float,
                      extended_hours: bool = False) -> dict:
    """Place a PAPER-trading LIMIT order (use this outside regular market hours).

    Args:
        symbol: Ticker, e.g. "AAPL".
        side: "buy" or "sell".
        qty: Whole number of shares.
        limit_price: Highest price to pay (buy) or lowest to accept (sell).
        extended_hours: True to allow pre-market / after-hours trading.
    """
    req = LimitOrderRequest(
        symbol=symbol.upper(),
        qty=int(qty),
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
        extended_hours=extended_hours,
    )
    return _order_to_dict(_trading_client.submit_order(req))


@mcp.tool()
def get_all_positions() -> dict:
    """List every stock currently held in the paper account."""
    ps = _trading_client.get_all_positions()
    return {
        "count": len(ps),
        "positions": [
            {
                "symbol": p.symbol,
                "qty": str(p.qty),
                "avg_entry_price": str(p.avg_entry_price),
                "current_price": str(p.current_price),
                "market_value": str(p.market_value),
                "unrealized_pl": str(p.unrealized_pl),
                "unrealized_plpc": str(p.unrealized_plpc),
            }
            for p in ps
        ],
    }


# ---------------- PROFESSOR'S ORIGINAL TOOLS (unchanged) ----------------

@mcp.tool()
def buy_stock(symbol: str, qty: float) -> dict:
    """Place a PAPER-trading market BUY order on Alpaca."""
    order_request = MarketOrderRequest(
        symbol=symbol.upper(), qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
    )
    return _order_to_dict(_trading_client.submit_order(order_request))


@mcp.tool()
def sell_stock(symbol: str, qty: float) -> dict:
    """Place a PAPER-trading market SELL order on Alpaca."""
    order_request = MarketOrderRequest(
        symbol=symbol.upper(), qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
    )
    return _order_to_dict(_trading_client.submit_order(order_request))


@mcp.tool()
def get_position(symbol: str) -> dict:
    """Get the current open position size for a symbol, if any."""
    try:
        pos = _trading_client.get_open_position(symbol.upper())
        return {
            "symbol": pos.symbol,
            "qty": str(pos.qty),
            "market_value": str(pos.market_value),
            "unrealized_pl": str(pos.unrealized_pl),
        }
    except Exception:
        return {"symbol": symbol.upper(), "qty": "0", "note": "No open position for this symbol."}


@mcp.tool()
def get_open_orders(symbol: str = "") -> dict:
    """List currently open (unfilled) orders, optionally filtered by symbol."""
    orders = _trading_client.get_orders()
    if symbol:
        orders = [o for o in orders if o.symbol.upper() == symbol.upper()]
    return {"count": len(orders), "orders": [_order_to_dict(o) for o in orders]}


@mcp.tool()
def cancel_all_orders(symbol: str = "") -> dict:
    """Cancel all open orders, optionally restricted to one symbol."""
    if symbol:
        matching = [o for o in _trading_client.get_orders() if o.symbol.upper() == symbol.upper()]
        for o in matching:
            _trading_client.cancel_order_by_id(o.id)
        return {"cancelled_count": len(matching), "symbol": symbol.upper()}
    _trading_client.cancel_orders()
    return {"cancelled": "all open orders"}


@mcp.tool()
def get_account_info() -> dict:
    """Get current paper-trading account cash, buying power and portfolio value."""
    acct = _trading_client.get_account()
    return {
        "cash": str(acct.cash),
        "buying_power": str(acct.buying_power),
        "portfolio_value": str(acct.portfolio_value),
        "equity": str(acct.equity),
    }


if __name__ == "__main__":
    # streamable-http so the crew AND the web UI can share one server
    mcp.run(transport="streamable-http", host="localhost", port=8001)