"""Backtest the RAMS rules on DAILY bars: current rules vs proposed changes.

Run from the project root (needs your .env with ALPACA keys):
    python backtest_rams.py                 # uses my_stocks.txt, 3 years of data
    python backtest_rams.py --years 2       # shorter history
    python backtest_rams.py --synthetic     # offline self-test with fake prices

Prints: (1) does a high RAMS actually predict higher forward returns?
        (2) return / drawdown / trades for each rule set vs buy-and-hold and SPY.
Writes backtest_results.csv next to the script.
Signals use day t close; trades happen at day t+1 open (no look-ahead).
News is neutral (0) because historical headlines are not available.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

ROOT = Path(__file__).resolve().parent
W_TREND, W_MOM, W_VOL, W_BRK = 0.30, 0.30, 0.15, 0.10
W_NEWS = 0.15
COST = 0.0005          # 5 bps slippage per trade
START_CASH = 100_000.0
HIGH_BETA = {"MU", "XYZ", "HOOD", "SMCI"}


def col(frame, prefix):
    for c in frame.columns:
        if str(c).startswith(prefix):
            return frame[c]
    raise KeyError(prefix)


# ---------------------------------------------------------------- data
def load_symbols():
    for p in (ROOT / "crew_ai" / "clients" / "my_stocks.txt",
              ROOT / "clients" / "my_stocks.txt"):
        if p.exists():
            return [s.strip().upper() for s in p.read_text(encoding="utf-8-sig").splitlines()
                    if s.strip() and not s.startswith("#")]
    sys.exit("my_stocks.txt not found - create it first (see README step 4).")


def download(symbols, years):
    from dotenv import load_dotenv
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    load_dotenv(ROOT / ".env")
    client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    req = StockBarsRequest(symbol_or_symbols=sorted(set(symbols) | {"SPY"}), timeframe=TimeFrame.Day,
                           start=datetime.now(timezone.utc) - timedelta(days=int(365 * years) + 30),
                           feed=DataFeed.IEX)
    df = client.get_stock_bars(req).df
    out = {}
    for s in df.index.get_level_values(0).unique():
        d = df.xs(s, level=0).sort_index()
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        out[s] = d[["open", "high", "low", "close", "volume"]].astype(float)
    return out


def synthetic(symbols, years):
    rng = np.random.default_rng(7)
    n = int(252 * years)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for s in sorted(set(symbols) | {"SPY"}):
        r = rng.normal(0.0004, 0.02 if s != "SPY" else 0.01, n)
        close = 100 * np.exp(np.cumsum(r))
        op = close * (1 + rng.normal(0, 0.003, n))
        hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.006, n)))
        lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.006, n)))
        out[s] = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close,
                               "volume": rng.integers(1e6, 5e6, n).astype(float)}, index=idx)
    return out


# ------------------------------------------------------- RAMS (daily, vectorised copy of the server code)
def rams_frame(d, spy_close):
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
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

    trend = (25 * (c > ema20) + 25 * (ema20 > ema50) + 25 * (ema50 > ema200)
             + 25 * ((col(adx, "ADX_") > 25) & (col(adx, "DMP_") > col(adx, "DMN_"))))
    hist = col(macd, "MACDh_")
    rel = c.pct_change(20) - spy_close.reindex(c.index).pct_change(20)
    mom = (20 * ((rsi >= 50) & (rsi <= 70)) + 20 * ((hist > 0) & (hist > hist.shift(1)))
           + 20 * (roc10 > 0) + 20 * (col(tsi, "TSI_") > col(tsi, "TSIs_")) + 20 * (rel > 0))
    vol = (25 * (cmf > 0) + 25 * (obv > obv_ema) + 25 * ((mfi >= 50) & (mfi <= 80))
           + 25 * (col(pvo, "PVO_") > 0))
    bbw, bbp = col(bb, "BBB_"), col(bb, "BBP_")
    dcl, dcu = col(dc, "DCL_"), col(dc, "DCU_")
    dc_pos = ((c - dcl) / (dcu - dcl)).where(dcu > dcl, 0)
    brk = 33 * (bbw > bbw.shift(5)) + 33 * (dc_pos >= 0.8) + 34 * ((bbp >= 0.5) & (bbp <= 1))
    tech = (W_TREND * trend + W_MOM * mom + W_VOL * vol + W_BRK * brk) / (W_TREND + W_MOM + W_VOL + W_BRK)
    penalty = 10 * (z > 2.5)
    rams = ((1 - W_NEWS) * tech + W_NEWS * 50 - penalty).clip(0, 100)   # neutral news = 50 pts
    out = pd.DataFrame({"rams": rams, "trend": trend, "atr": atr, "rsi": rsi}, index=c.index)
    return out.astype(float)


# ------------------------------------------------------- Strategy engine
RULES = {
    # name: entry mode, exit rams, confirm days, min hold days, exit style
    "A. Current rules": dict(gate=False, buy_min=50, trend_min=25, sell_below=40, confirm=1, hold=0,
                             stop="atr_fixed", half_profit=True),
    "B. + hysteresis/confirm/hold": dict(gate=False, buy_min=60, trend_min=25, sell_below=30, confirm=2,
                                         hold=3, stop="atr_fixed", half_profit=True),
    "C. + regime gate": dict(gate=True, buy_min=60, trend_min=25, sell_below=30, confirm=2, hold=3,
                             stop="atr_fixed", half_profit=True),
    "D. Full proposal": dict(gate=True, buy_min=60, trend_min=25, sell_below=30, confirm=2, hold=3,
                             stop="trail", half_profit=False),
}


def simulate(data, symbols, rules, rams, bull, target_invested=0.80, cap=0.20, cap_hb=0.15):
    idx = data["SPY"].index
    cash, pos, log = START_CASH, {}, []
    equity, trades = [], []
    streak_buy = {s: 0 for s in symbols}
    streak_sell = {s: 0 for s in symbols}
    pending = []   # orders decided at close of t, executed at open of t+1
    for i, day in enumerate(idx):
        # 1) execute yesterday's decisions at today's open
        for side, s, qty in pending:
            if s not in data or day not in data[s].index:
                continue
            px = float(data[s].loc[day, "open"])
            if side == "sell" and s in pos:
                q = min(qty, pos[s]["qty"])
                cash += q * px * (1 - COST)
                if q >= pos[s]["qty"]:
                    trades.append((s, pos[s]["entry"], px))
                    del pos[s]
                else:
                    pos[s]["qty"] -= q
                    pos[s]["half"] = True
            elif side == "buy":
                q = int(min(qty, cash // (px * (1 + COST))))
                if q > 0:
                    cash -= q * px * (1 + COST)
                    if s in pos:
                        tot = pos[s]["qty"] + q
                        pos[s]["entry"] = (pos[s]["entry"] * pos[s]["qty"] + px * q) / tot
                        pos[s]["qty"] = tot
                    else:
                        pos[s] = dict(qty=q, entry=px, day=i, high=px, half=False)
        pending = []
        # 2) mark to market at close
        eq = cash + sum(p["qty"] * float(data[s].loc[day, "close"]) for s, p in pos.items()
                        if day in data[s].index)
        equity.append(eq)
        # 3) decide for tomorrow
        sells, cands = [], {}
        for s in symbols:
            if s not in rams or day not in rams[s].index:
                continue
            r = rams[s].loc[day]
            if np.isnan(r["rams"]):
                continue
            px = float(data[s].loc[day, "close"])
            if s in pos:
                p = pos[s]
                p["high"] = max(p["high"], px)
                held_days = i - p["day"]
                streak_sell[s] = streak_sell[s] + 1 if r["rams"] < rules["sell_below"] else 0
                hit = False
                if rules["stop"] == "atr_fixed":
                    hit = px <= p["entry"] - 2.5 * r["atr"]
                else:
                    hit = px <= p["high"] - 3.0 * r["atr"]
                if hit:
                    sells.append((s, p["qty"]))
                elif held_days >= rules["hold"] and streak_sell[s] >= rules["confirm"]:
                    sells.append((s, p["qty"]))
                elif (rules["half_profit"] and not p["half"] and px >= p["entry"] + 3.0 * r["atr"]
                      and p["qty"] >= 2):
                    sells.append((s, p["qty"] // 2))
            else:
                ok_level = rules["buy_min"]
                if rules["gate"]:
                    ok_level = max(ok_level, 62 if bull.loc[day] else 70)
                ok = r["rams"] >= ok_level and r["trend"] >= rules["trend_min"]
                streak_buy[s] = streak_buy[s] + 1 if ok else 0
                if streak_buy[s] >= rules["confirm"]:
                    cands[s] = r["rams"]
        for s, q in sells:
            pending.append(("sell", s, q))
        if cands:
            sold_value = sum(q * float(data[s].loc[day, "close"]) for s, q in sells)
            invested = sum(p["qty"] * float(data[s].loc[day, "close"]) for s, p in pos.items()) - sold_value
            budget = max(0.0, min(target_invested * eq - invested, cash + sold_value))
            tot = sum(cands.values())
            for s, sc in sorted(cands.items(), key=lambda kv: -kv[1]):
                px = float(data[s].loc[day, "close"])
                limit = (cap_hb if s in HIGH_BETA else cap) * eq
                want = min(budget * sc / tot, limit)
                q = int(want // px)
                if q > 0:
                    pending.append(("buy", s, q))
    return pd.Series(equity, index=idx), trades


def stats(eq, trades):
    ret = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    d = eq.pct_change().dropna()
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0
    wins = sum(1 for _, e, x in trades if x > e)
    return dict(total_return_pct=round(100 * ret, 1), max_drawdown_pct=round(100 * dd, 1),
                sharpe=round(sharpe, 2), closed_trades=len(trades),
                win_rate_pct=round(100 * wins / len(trades), 0) if trades else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    symbols = ["MU", "XYZ", "HOOD", "SMCI", "AVGO", "FCX", "AMZN", "GOOGL", "GE", "V"] if a.synthetic \
        else load_symbols()
    data = synthetic(symbols, a.years) if a.synthetic else download(symbols, a.years)
    symbols = [s for s in symbols if s in data and len(data[s]) > 260]
    missing = [s for s in load_symbols() if s not in symbols] if not a.synthetic else []
    if missing:
        print("Skipped (not enough history):", ", ".join(missing))
    spy = data["SPY"]["close"]
    bull = spy > ta.ema(spy, 50)
    rams = {s: rams_frame(data[s], spy) for s in symbols}

    # ---- warm-up: first 220 bars are needed for EMA200
    start = data["SPY"].index[220]
    for s in list(data):
        data[s] = data[s].loc[data[s].index >= data["SPY"].index[0]]
    print(f"\nStocks: {', '.join(symbols)}\nTest window: {start.date()} -> {spy.index[-1].date()} "
          f"({len(spy) - 220} trading days)\n")

    # ---- (1) does RAMS predict forward returns?
    rows = []
    for s in symbols:
        f = pd.DataFrame({"rams": rams[s]["rams"],
                          "fwd5": data[s]["close"].shift(-5) / data[s]["close"] - 1,
                          "fwd10": data[s]["close"].shift(-10) / data[s]["close"] - 1}).loc[start:].dropna()
        rows.append(f)
    allf = pd.concat(rows)
    allf["bucket"] = pd.cut(allf["rams"], [0, 30, 40, 50, 60, 70, 100],
                            labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"])
    t = allf.groupby("bucket", observed=True).agg(days=("rams", "size"),
                                                  avg_fwd_5d_pct=("fwd5", lambda x: 100 * x.mean()),
                                                  avg_fwd_10d_pct=("fwd10", lambda x: 100 * x.mean()),
                                                  hit_rate_5d_pct=("fwd5", lambda x: 100 * (x > 0).mean())).round(2)
    print("(1) FORWARD RETURN BY RAMS BUCKET (all your stocks, daily bars)")
    print(t.to_string())
    corr = allf["rams"].corr(allf["fwd5"])
    print(f"Correlation RAMS vs next-5-day return: {corr:+.3f}  "
          "(near 0 = RAMS does not predict; >0.05 = some signal)\n")

    # ---- (2) strategy comparison
    results = {}
    for name, rules in RULES.items():
        eq, tr = simulate(data, symbols, rules, rams, bull)
        eq = eq.loc[start:]
        results[name] = stats(eq / eq.iloc[0] * START_CASH, tr)
    ew = pd.concat([data[s]["close"].loc[start:] / data[s]["close"].loc[start:].iloc[0] for s in symbols],
                   axis=1).mean(axis=1) * START_CASH
    results["Benchmark: equal-weight buy & hold"] = stats(ew, [])
    sp = spy.loc[start:] / spy.loc[start:].iloc[0] * START_CASH
    results["Benchmark: SPY"] = stats(sp, [])
    res = pd.DataFrame(results).T
    print("(2) STRATEGY COMPARISON (start $100,000, 5 bps cost per trade, news neutral)")
    print(res.to_string())
    res.to_csv(ROOT / "backtest_results.csv")
    print("\nSaved backtest_results.csv")


if __name__ == "__main__":
    main()
