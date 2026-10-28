import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fastmcp import Client
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

load_dotenv()
st.cache_data.clear()

ROOT = Path(__file__).resolve().parent
STOCKS_FILE = ROOT / "crew_ai" / "clients" / "my_stocks.txt"
OUTPUTS_DIR = ROOT / "crew_ai" / "outputs"
TEAM_DIR = ROOT / "data" / "team"
MCP_URL = "http://localhost:8001/mcp"

DEFAULT_STOCKS = [
    "NVDA",
    "APP",
    "UBER",
    "ANET",
    "AAPL",
    "META",
    "JNJ",
    "HD",
    "GS",
    "NEE",
]
INDICATOR_ORDER = ["Trend", "Momentum", "Volume", "Breakout", "News"]
MEMBERS = ["sana", "divisha", "vani"]
TEAM_STOCK_SECTORS = TEAM_DIR / "stock_sectors.csv"
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
MEMBER_COLORS = {"sana": "#5DA5DA", "divisha": "#F17CB0", "vani": "#60BD68"}
SPY_COLOR = "#9aa0a6"

st.set_page_config(page_title="Team Trading Dashboard", page_icon="📊", layout="wide")

if "plotly_dark" not in px.colors.named_colorscales():
    pass

plotly_template = "plotly_dark"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_numeric_string(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        if cleaned in ("", "-"):
            return None
        return cleaned
    return value


def coerce_numeric_series(series):
    return pd.to_numeric(series.map(clean_numeric_string), errors="coerce")


@st.cache_data(show_spinner=False)
def get_latest_run_folder() -> Path | None:
    if not OUTPUTS_DIR.exists():
        return None
    folders = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not folders:
        return None
    return max(folders, key=lambda p: p.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_stock_sectors() -> pd.DataFrame:
    if not TEAM_STOCK_SECTORS.exists():
        return pd.DataFrame(columns=["member", "symbol", "sector", "sector_etf", "high_beta"])
    df = pd.read_csv(TEAM_STOCK_SECTORS)
    if "member" not in df.columns:
        return pd.DataFrame(columns=["member", "symbol", "sector", "sector_etf", "high_beta"])
    df["member"] = df["member"].astype(str).str.lower()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["high_beta"] = df["high_beta"].astype(str).str.lower().eq("yes")
    return df


@st.cache_data(show_spinner=False)
def read_member_file(member: str, file_name: str) -> pd.DataFrame | None:
    path = TEAM_DIR / f"{member}_{file_name}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def parse_stock_list(raw: str) -> list[str]:
    parsed = []
    for item in raw.splitlines():
        symbol = item.strip().upper()
        if not symbol or symbol.startswith("#"):
            continue
        parsed.append(symbol)
    return parsed


def write_stock_file(stock_text: str) -> None:
    STOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOCKS_FILE.write_text("\n".join(parse_stock_list(stock_text)) + "\n", encoding="utf-8")


def is_port_open(host: str = "127.0.0.1", port: int = 8001, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def launch_mcp_server_if_needed(api_key: str, secret_key: str, selected_indicators: dict[str, bool]) -> bool:
    if is_port_open():
        return True

    server_env = os.environ.copy()
    server_env["ALPACA_API_KEY"] = api_key
    server_env["ALPACA_SECRET_KEY"] = secret_key
    server_env["TRADING_ENABLED"] = "true"
    server_env["RAMS_SELECTION"] = json.dumps(selected_indicators)

    root_str = str(ROOT)
    launcher = """
import importlib.util
import json
import os
import runpy
import sys

root = __import__('pathlib').Path(r""" + root_str.replace("\\", "\\\\") + """)
config_path = root / 'servers' / 'config.py'
spec = importlib.util.spec_from_file_location('runtime_config', config_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
selection = {str(k).lower(): bool(v) for k, v in json.loads(os.environ.get('RAMS_SELECTION', '{}')).items()}
weights = {'trend': 0.30, 'momentum': 0.30, 'volume': 0.15, 'breakout': 0.10, 'news': 0.15}
active = {name for name, enabled in selection.items() if enabled}
if not active:
    active = set(weights)
scale = sum(weights[name] for name in active) or 1.0
for name in weights:
    value = weights[name] / scale if name in active else 0.0
    setattr(module, f'W_{name.upper()}', value)
sys.modules['config'] = module
sys.path.insert(0, str(root / 'servers'))
runpy.run_path(str(root / 'servers' / 'alpaca_trading_mcpServer.py'), run_name='__main__')
"""

    subprocess.Popen(
        [sys.executable, "-c", launcher],
        cwd=str(ROOT),
        env=server_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    for _ in range(60):
        if is_port_open():
            return True
        time.sleep(0.5)
    return False


async def call_mcp_tool(tool_name: str, payload: dict | None = None):
    payload = payload or {}
    async with Client(MCP_URL) as client:
        response = await client.call_tool(tool_name, payload)
        return response.data


def fetch_portfolio() -> dict:
    try:
        account = asyncio.run(call_mcp_tool("get_account_info"))
        positions = asyncio.run(call_mcp_tool("get_all_positions"))
        open_orders = asyncio.run(call_mcp_tool("get_open_orders", {"symbol": ""}))
        return {"account": account, "positions": positions, "open_orders": open_orders}
    except Exception as exc:
        return {"error": str(exc)}


def run_crew_script(run_mode: str, selected_indicators: dict[str, bool], alpaca_key: str, alpaca_secret: str, groq_key: str) -> tuple[int, str, str]:
    cmd = [sys.executable, str(ROOT / "crew_ai" / "clients" / "trading_crew.py")]
    if run_mode == "dry_run":
        cmd.append("--dry-run")
    elif run_mode == "live":
        cmd.append("--auto")
    else:
        raise ValueError(f"Unsupported run mode: {run_mode}")

    env = os.environ.copy()
    env["ALPACA_API_KEY"] = alpaca_key
    env["ALPACA_SECRET_KEY"] = alpaca_secret
    env["GROQ_API_KEY"] = groq_key
    env["MCP_URL"] = MCP_URL
    env["RAMS_SELECTION"] = json.dumps(selected_indicators)
    env["TRADING_ENABLED"] = "true"

    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    return completed.returncode, completed.stdout, completed.stderr


@st.cache_data(show_spinner=False)
def load_team_equity_data() -> dict[str, pd.DataFrame]:
    out = {}
    for member in MEMBERS:
        df = read_member_file(member, "equity")
        if df is None or df.empty:
            out[member] = pd.DataFrame(columns=["date", "equity"])
            continue
        temp = df.copy()
        temp.columns = [str(c).strip().lower().replace(" ", "_") for c in temp.columns]
        if "account_value" in temp.columns and "equity" not in temp.columns:
            temp = temp.rename(columns={"account_value": "equity"})
        if "portfolio_value" in temp.columns and "equity" not in temp.columns:
            temp = temp.rename(columns={"portfolio_value": "equity"})
        if "date" not in temp.columns:
            out[member] = pd.DataFrame(columns=["date", "equity"])
            continue
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        if "equity" in temp.columns:
            temp["equity"] = coerce_numeric_series(temp["equity"])
        out[member] = temp.dropna(subset=["date", "equity"]).sort_values("date")
        out[member]["date"] = out[member]["date"].dt.date
    return out


@st.cache_data(show_spinner=False)
def load_team_trade_data() -> dict[str, pd.DataFrame]:
    out = {}
    for member in MEMBERS:
        df = read_member_file(member, "trades")
        if df is None or df.empty:
            out[member] = pd.DataFrame(columns=["date", "symbol", "side", "qty", "price", "dollar_value"])
            continue
        temp = df.copy()
        temp.columns = [str(c).strip().lower().replace(" ", "_") for c in temp.columns]
        for src, target in {"filled_price": "price", "avg_fill_price": "price", "trade_price": "price", "order_price": "price"}.items():
            if src in temp.columns and target in temp.columns:
                temp[target] = temp[target].fillna(temp[src])
        if "date" not in temp.columns or "symbol" not in temp.columns:
            out[member] = pd.DataFrame(columns=["date", "symbol", "side", "qty", "price", "dollar_value"])
            continue
        temp["member"] = member
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp["qty"] = coerce_numeric_series(temp.get("qty", pd.Series(index=temp.index, dtype=float)))
        temp["price"] = coerce_numeric_series(temp.get("price", pd.Series(index=temp.index, dtype=float)))
        if "side" in temp.columns:
            temp["side"] = temp["side"].astype(str).str.split(".").str[-1].str.lower()
        temp["dollar_value"] = temp["qty"].abs() * temp["price"]
        out[member] = temp.dropna(subset=["date", "symbol"]).sort_values("date")
    return out


@st.cache_data(show_spinner=False)
def load_team_position_data() -> dict[str, pd.DataFrame]:
    out = {}
    latest_prices = fetch_latest_trade_prices([symbol for symbol in load_stock_sectors()["symbol"].dropna().astype(str).str.upper().unique()])
    for member in MEMBERS:
        df = read_member_file(member, "positions")
        fallback_path = TEAM_DIR / f"{member}_last_positions.csv"
        if (df is None or df.empty) and fallback_path.exists():
            try:
                fallback = pd.read_csv(fallback_path)
            except Exception:
                fallback = None
            if fallback is not None and not fallback.empty:
                df = fallback
        if df is None or df.empty:
            out[member] = pd.DataFrame(columns=["symbol", "qty", "market_value", "unrealized_pl", "unrealized_plpc", "avg_entry_price", "current_price"])
            continue
        temp = df.copy()
        temp.columns = [str(c).strip().lower().replace(" ", "_") for c in temp.columns]
        if "symbol" in temp.columns:
            temp["symbol"] = temp["symbol"].astype(str).str.upper()
        for col in ["qty", "market_value", "unrealized_pl", "unrealized_plpc", "avg_entry_price", "current_price"]:
            if col in temp.columns:
                temp[col] = coerce_numeric_series(temp[col])

        if "current_price" in temp.columns:
            mask = temp["current_price"].isna() & temp["symbol"].isin(latest_prices.keys())
            temp.loc[mask, "current_price"] = temp.loc[mask, "symbol"].map(latest_prices)
        if "market_value" in temp.columns:
            qty = temp["qty"] if "qty" in temp.columns else pd.Series(0.0, index=temp.index)
            current_price = temp["current_price"] if "current_price" in temp.columns else pd.Series(pd.NA, index=temp.index)
            mask = temp["market_value"].isna() & qty.notna() & current_price.notna()
            temp.loc[mask, "market_value"] = qty[mask] * current_price[mask]
        if "unrealized_pl" in temp.columns:
            qty = temp["qty"] if "qty" in temp.columns else pd.Series(0.0, index=temp.index)
            current_price = temp["current_price"] if "current_price" in temp.columns else pd.Series(pd.NA, index=temp.index)
            avg_entry_price = temp["avg_entry_price"] if "avg_entry_price" in temp.columns else pd.Series(pd.NA, index=temp.index)
            mask = temp["unrealized_pl"].isna() & qty.notna() & current_price.notna() & avg_entry_price.notna()
            temp.loc[mask, "unrealized_pl"] = (current_price[mask] - avg_entry_price[mask]) * qty[mask]
        if "unrealized_plpc" in temp.columns:
            avg_entry_price = temp["avg_entry_price"] if "avg_entry_price" in temp.columns else pd.Series(pd.NA, index=temp.index)
            unrealized_pl = temp["unrealized_pl"] if "unrealized_pl" in temp.columns else pd.Series(pd.NA, index=temp.index)
            mask = temp["unrealized_plpc"].isna() & avg_entry_price.notna() & avg_entry_price.ne(0) & unrealized_pl.notna()
            temp.loc[mask, "unrealized_plpc"] = unrealized_pl[mask] / (avg_entry_price[mask] * (temp.loc[mask, "qty"] if "qty" in temp.columns else pd.Series(1.0, index=temp.index)[mask]))
        out[member] = temp
    return out


@st.cache_data(show_spinner=False)
def fetch_daily_prices(symbols: list[str]) -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        return pd.DataFrame(columns=["symbol", "date", "close"])

    symbols = sorted({s.upper() for s in symbols if s and s.strip()})
    if not symbols:
        return pd.DataFrame(columns=["symbol", "date", "close"]) 

    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start="2026-10-09",
        feed=DataFeed.IEX,
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "date", "close"])

    df = bars.reset_index()
    df.columns = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["symbol", "date", "close"]].dropna(subset=["close"])


@st.cache_data(show_spinner=False)
def fetch_latest_trade_prices(symbols: list[str]) -> dict[str, float]:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    unique_symbols = sorted({s.upper() for s in symbols if s and str(s).strip()})
    if not unique_symbols or not api_key or not api_secret:
        return {}

    try:
        from alpaca.data.requests import StockLatestTradeRequest
        client = StockHistoricalDataClient(api_key, api_secret)
        request = StockLatestTradeRequest(symbol_or_symbols=unique_symbols)
        latest = client.get_stock_latest_trade(request)
    except Exception:
        return {}

    prices = {}
    for symbol, trade in latest.items():
        value = getattr(trade, "price", None)
        if value is not None:
            try:
                prices[str(symbol).upper()] = float(value)
            except (TypeError, ValueError):
                pass
    return prices


def to_numeric_series(series):
    return pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)


def compute_cumulative_return(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    values = to_numeric_series(series)
    if values.dropna().empty:
        return pd.Series(index=series.index, dtype=float)
    base = values.dropna().iloc[0]
    if pd.isna(base) or base == 0:
        return pd.Series(index=series.index, dtype=float)
    return ((values / base) - 1) * 100


def member_last_date(member: str, file_name: str = "equity") -> str | None:
    path = TEAM_DIR / f"{member}_{file_name}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "date" not in df.columns:
        return None
    last_date = pd.to_datetime(df["date"], errors="coerce").dropna().max()
    if pd.isna(last_date):
        return None
    return pd.Timestamp(last_date).strftime("%Y-%m-%d")


def member_data_source_label(member: str) -> str:
    if member == "vani":
        return "Live from Alpaca"
    if member == "sana":
        return "From Sana's Alpaca exports"
    if member == "divisha":
        return "From Divisha's trading logs"
    member_dir = OUTPUTS_DIR / member
    latest_date = None
    if member_dir.exists():
        files = [p for p in member_dir.rglob("*") if p.is_file()]
        if files:
            latest_date = max(p.stat().st_mtime for p in files)
    if latest_date is not None:
        return f"From team logs, last updated {datetime.fromtimestamp(latest_date, tz=timezone.utc).strftime('%Y-%m-%d')}"
    return "From team logs"


def build_member_summary(member: str):
    equity_df = load_team_equity_data().get(member)
    trades_df = load_team_trade_data().get(member)
    positions_df = load_team_position_data().get(member)
    if equity_df is None or equity_df.empty:
        return {
            "member": member,
            "account_value": "Data not uploaded yet",
            "pnl": "Data not uploaded yet",
            "return_pct": "Data not uploaded yet",
            "trade_count": "Not uploaded",
            "source": member_data_source_label(member),
            "last_date": None,
            "status": "No data yet",
        }

    equity = equity_df["equity"].dropna()
    if equity.empty:
        return {"member": member, "account_value": "Data not uploaded yet", "pnl": "Data not uploaded yet", "return_pct": "Data not uploaded yet", "trade_count": "Not uploaded", "source": member_data_source_label(member), "last_date": None, "status": "No data yet"}

    latest = float(equity.iloc[-1])
    pnl = latest - 100000.0
    pct = ((latest / 100000.0) - 1) * 100
    if trades_df is None or trades_df.empty:
        trade_count = "Not uploaded"
    else:
        trade_count = int(len(trades_df))
    last_date = pd.to_datetime(equity_df["date"].max()).strftime("%Y-%m-%d") if "date" in equity_df.columns and not equity_df.empty else None
    return {
        "member": member,
        "account_value": f"${latest:,.2f}",
        "pnl": f"${pnl:,.2f}",
        "return_pct": f"{pct:.2f}%",
        "trade_count": trade_count,
        "source": member_data_source_label(member),
        "last_date": last_date,
        "status": "OK",
        "portfolio_status": "All in cash" if (positions_df is None or positions_df.empty) else "Portfolio loaded",
    }


def compute_stock_activity_data() -> dict:
    trades_by_member = load_team_trade_data()
    out = {"trades_per_day": pd.DataFrame(columns=["date", "member", "trades"]), "heatmap": pd.DataFrame(columns=["member", "symbol", "trades"]), "buys_vs_sells": pd.DataFrame(columns=["member", "side", "count"]), "dollar_amount": pd.DataFrame(columns=["member", "dollar_amount"]), "distinct_stocks": pd.DataFrame(columns=["member", "stock_count"]), "avg_holding_days": pd.DataFrame(columns=["member", "avg_holding_days"])}
    frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        daily = df.groupby(pd.to_datetime(df["date"]).dt.date).size().reset_index()
        daily.columns = ["date", "trades"]
        daily["member"] = member
        frames.append(daily[["date", "member", "trades"]])

    if frames:
        out["trades_per_day"] = pd.concat(frames, ignore_index=True)

    heatmap_frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        grouped = df.groupby(["symbol"], as_index=False).size().rename(columns={"size": "trades"})
        grouped["member"] = member
        heatmap_frames.append(grouped[["member", "symbol", "trades"]])
    if heatmap_frames:
        out["heatmap"] = pd.concat(heatmap_frames, ignore_index=True)

    buy_sell_frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        grouped = df.groupby(["member", "side"], as_index=False).size().rename(columns={"size": "count"})
        buy_sell_frames.append(grouped)
    if buy_sell_frames:
        out["buys_vs_sells"] = pd.concat(buy_sell_frames, ignore_index=True)

    dollar_frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        total = df["dollar_value"].sum()
        dollar_frames.append(pd.DataFrame({"member": [member], "dollar_amount": [float(total)]}))
    if dollar_frames:
        out["dollar_amount"] = pd.concat(dollar_frames, ignore_index=True)

    stock_count_frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        stock_count_frames.append(pd.DataFrame({"member": [member], "stock_count": [df["symbol"].nunique()]}))
    if stock_count_frames:
        out["distinct_stocks"] = pd.concat(stock_count_frames, ignore_index=True)

    avg_holding_frames = []
    for member, df in trades_by_member.items():
        if df.empty:
            continue
        trades = df.sort_values("date").copy()
        buys = trades[trades["side"].str.lower() == "buy"].copy()
        sells = trades[trades["side"].str.lower() == "sell"].copy()
        durations = []
        for symbol in sorted(set(buys["symbol"]).intersection(sells["symbol"])):
            buy_dates = buys.loc[buys["symbol"] == symbol, "date"].tolist()
            sell_dates = sells.loc[sells["symbol"] == symbol, "date"].tolist()
            for bdate in buy_dates:
                if sell_dates:
                    next_sell = min(sell_dates, key=lambda x: abs((x - bdate).days))
                    durations.append((next_sell - bdate).total_seconds() / 86400)
        avg_holding_frames.append(pd.DataFrame({"member": [member], "avg_holding_days": [float(sum(durations) / len(durations)) if durations else 0]}))
    if avg_holding_frames:
        out["avg_holding_days"] = pd.concat(avg_holding_frames, ignore_index=True)
    return out


def render_team_overview():
    summaries = {member: build_member_summary(member) for member in MEMBERS}
    cards = st.columns(len(MEMBERS) + 1)
    for i, member in enumerate(MEMBERS):
        summary = summaries[member]
        with cards[i]:
            st.markdown(f"### {member.title()}")
            if summary["status"] == "No data yet":
                st.caption(summary["source"])
                st.info("Data not uploaded yet")
            else:
                st.caption(summary["source"])
                if summary.get("last_date"):
                    st.caption(f"Last date: {summary['last_date']}")
                st.metric("Account value", summary["account_value"])
                st.metric("P/L ($)", summary["pnl"])
                st.metric("Return %", summary["return_pct"])
                st.metric("Trades", summary["trade_count"])
    with cards[-1]:
        st.markdown("### Group")
        ok_summaries = [summary for summary in summaries.values() if summary["status"] == "OK"]
        if ok_summaries:
            values = [float(summary["account_value"].replace('$', '').replace(',', '')) for summary in ok_summaries]
            returns = [float(summary["return_pct"].replace('%', '')) for summary in ok_summaries]
            st.metric("Combined account value", f"${sum(values):,.2f}")
            st.metric("Group avg return", f"{(sum(returns) / len(returns)):.2f}%")
            st.metric("Members with data", len(ok_summaries))
            if len(ok_summaries) < 2:
                st.caption("Group average appears when at least two members have uploaded data.")
        else:
            st.info("Data not uploaded yet")

    spy_prices = fetch_daily_prices(["SPY"]).copy()
    if not spy_prices.empty:
        spy_prices = spy_prices[spy_prices["symbol"] == "SPY"].sort_values("date")
        if not spy_prices.empty:
            spy_prices["pct_return"] = compute_cumulative_return(spy_prices["close"])
            spy_prices["date"] = pd.to_datetime(spy_prices["date"])
    else:
        spy_prices = pd.DataFrame(columns=["date", "close", "pct_return"])

    fig = go.Figure()
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        temp = equity_df.copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.dropna(subset=["date", "equity"]).sort_values("date")
        if temp.empty:
            continue
        temp["pct_return"] = ((temp["equity"] / 100000.0) - 1) * 100
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["pct_return"], mode="lines", name=member.title(), line=dict(color=MEMBER_COLORS.get(member, "#7F8C8D"), width=3)))
    if not spy_prices.empty and not spy_prices["pct_return"].dropna().empty:
        fig.add_trace(go.Scatter(x=spy_prices["date"], y=spy_prices["pct_return"], mode="lines", name="SPY", line=dict(color=SPY_COLOR, width=2, dash="dash")))
    fig.update_layout(title="Return % over time by member vs SPY", template=plotly_template, xaxis_title="Date", yaxis_title="Return %", legend_title="Member", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each member's return is measured against the $100,000 starting capital; the benchmark SPY is shaded grey.")

    member_returns = []
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        latest = equity_df["equity"].dropna().iloc[-1] if not equity_df.empty else None
        if latest is None or pd.isna(latest):
            continue
        member_returns.append({"member": member, "return_pct": ((latest / 100000.0) - 1) * 100})
    if member_returns:
        spy_base = spy_prices["close"].dropna().iloc[0] if not spy_prices.empty and not spy_prices["close"].dropna().empty else None
        spy_last = spy_prices["close"].dropna().iloc[-1] if not spy_prices.empty and not spy_prices["close"].dropna().empty else None
        spy_return = ((spy_last / spy_base) - 1) * 100 if spy_base and spy_last else 0
        group_avg = sum(item["return_pct"] for item in member_returns) / len(member_returns) if member_returns else 0
        return_df = pd.DataFrame(member_returns)
        if len(member_returns) >= 2:
            return_df = pd.concat([
                return_df,
                pd.DataFrame([{"member": "Group avg", "return_pct": group_avg}, {"member": "SPY", "return_pct": spy_return}]),
            ], ignore_index=True)
        else:
            return_df = pd.concat([
                return_df,
                pd.DataFrame([{"member": "SPY", "return_pct": spy_return}]),
            ], ignore_index=True)
        bar_fig = px.bar(return_df, x="member", y="return_pct", color="member", title="Final return by member vs SPY", labels={"member": "Member", "return_pct": "Return %"}, template=plotly_template, color_discrete_map={**MEMBER_COLORS, "Group avg": "#D0D0D0", "SPY": SPY_COLOR})
        bar_fig.update_layout(xaxis_title="Member", yaxis_title="Return %")
        st.plotly_chart(bar_fig, width="stretch")
        st.caption("Each member's return is measured to their own last available date; Divisha is shown through 21 Oct and all returns are benchmarked against $100,000.")
    else:
        st.info("Data not uploaded yet")

    data_rows = [summary for summary in summaries.values() if summary["status"] == "OK"]
    if data_rows:
        best = max(data_rows, key=lambda x: float(x["return_pct"].replace('%', '')))
        worst = min(data_rows, key=lambda x: float(x["return_pct"].replace('%', '')))
        avg_return = sum(float(x["return_pct"].replace('%', '')) for x in data_rows) / len(data_rows)
        st.markdown("### Insights")
        st.info(
            f"{best['member'].title()} had the strongest return at {best['return_pct']}, while {worst['member'].title()} was lowest at {worst['return_pct']}. "
            f"The average return across members with data is {avg_return:.2f}%. "
            f"The combined member portfolio stands at ${sum(float(summary['account_value'].replace('$','').replace(',','')) for summary in data_rows):,.2f}."
        )
    else:
        st.markdown("### Insights")
        st.info("No team member data has been uploaded yet.")


def render_stock_activity():
    activity = compute_stock_activity_data()
    if activity["trades_per_day"].empty:
        st.info("No data yet")
        return

    trade_days = activity["trades_per_day"].copy()
    trade_days["date"] = pd.to_datetime(trade_days["date"])
    fig1 = px.line(trade_days, x="date", y="trades", color="member", title="Trades per day by member", labels={"date": "Date", "trades": "Trades", "member": "Member"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
    fig1.update_layout(xaxis_title="Date", yaxis_title="Trades")
    st.plotly_chart(fig1, width="stretch")
    st.caption("Daily trade count per member, based on the filled orders recorded in each member's trade log.")

    heatmap_df = activity["heatmap"].copy()
    if not heatmap_df.empty:
        heat = heatmap_df.pivot_table(index="member", columns="symbol", values="trades", aggfunc="sum").fillna(0)
        fig2 = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Trades by member and stock", labels={"x": "Stock", "y": "Member", "color": "Trades"}, template=plotly_template)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Heatmap of filled trade counts by member and stock. Darker cells mean more trades.")

    buy_sell = activity["buys_vs_sells"].copy()
    if not buy_sell.empty:
        fig3 = px.bar(buy_sell, x="member", y="count", color="side", barmode="group", title="Buys vs sells per member", labels={"member": "Member", "count": "Orders", "side": "Side"}, template=plotly_template)
        fig3.update_layout(xaxis_title="Member", yaxis_title="Orders")
        st.plotly_chart(fig3, width="stretch")
        st.caption("Buy and sell counts from each member's trade log.")

    dollar_amt = activity["dollar_amount"].copy()
    if not dollar_amt.empty:
        fig4 = px.bar(dollar_amt, x="member", y="dollar_amount", color="member", title="Dollar amount traded per member", labels={"member": "Member", "dollar_amount": "$ Traded"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
        fig4.update_layout(xaxis_title="Member", yaxis_title="$ Traded")
        st.plotly_chart(fig4, width="stretch")
        st.caption("Total notional value of each member's filled trades.")

    stock_counts = activity["distinct_stocks"].copy()
    if not stock_counts.empty:
        fig5 = px.bar(stock_counts, x="member", y="stock_count", color="member", title="Different stocks traded by member", labels={"member": "Member", "stock_count": "Distinct stocks"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
        fig5.update_layout(xaxis_title="Member", yaxis_title="Distinct stocks")
        st.plotly_chart(fig5, width="stretch")
        st.caption("Number of unique symbols each member traded.")

    avg_hold = activity["avg_holding_days"].copy()
    if not avg_hold.empty:
        fig6 = px.bar(avg_hold, x="member", y="avg_holding_days", color="member", title="Average holding time per member", labels={"member": "Member", "avg_holding_days": "Days"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
        fig6.update_layout(xaxis_title="Member", yaxis_title="Days held")
        st.plotly_chart(fig6, width="stretch")
        st.caption("Average time between a buy and the matched sell for each member.")


@st.cache_data(show_spinner=False)
def get_sector_etf_prices() -> pd.DataFrame:
    sector_map = load_stock_sectors()
    symbols = sorted(set(sector_map["sector_etf"].tolist()) | {"SPY"})
    return fetch_daily_prices(symbols)


def compute_stock_vs_sector_comparison():
    sector_map = load_stock_sectors()
    prices = fetch_daily_prices(sorted(set(sector_map["symbol"].tolist()) | set(sector_map["sector_etf"].tolist()) | {"SPY"}))
    if prices.empty:
        return pd.DataFrame(columns=["member", "symbol", "return_pct", "stock_return", "sector_return", "beat_stock", "beat_sector", "sector", "sector_etf"])

    trades = load_team_trade_data()
    rows = []
    for member, trade_df in trades.items():
        if trade_df.empty:
            continue
        for symbol in sorted(trade_df["symbol"].dropna().unique()):
            symbol_data = trade_df[trade_df["symbol"] == symbol].sort_values("date")
            if symbol_data.empty:
                continue
            symbol_prices = prices[prices["symbol"] == symbol].sort_values("date")
            if symbol_prices.empty:
                continue
            start = symbol_data["date"].min().date()
            end = symbol_data["date"].max().date()
            window = symbol_prices[(symbol_prices["date"] >= start) & (symbol_prices["date"] <= end)]
            if window.empty:
                continue
            stock_return = ((window["close"].iloc[-1] / window["close"].iloc[0]) - 1) * 100 if window["close"].iloc[0] else 0
            sector_meta = sector_map[sector_map["symbol"] == symbol]
            sector_etf = sector_meta["sector_etf"].iloc[0] if not sector_meta.empty else None
            if sector_etf is None:
                continue
            etf_prices = prices[prices["symbol"] == sector_etf].sort_values("date")
            if etf_prices.empty:
                continue
            etf_window = etf_prices[(etf_prices["date"] >= start) & (etf_prices["date"] <= end)]
            if etf_window.empty:
                continue
            sector_return = ((etf_window["close"].iloc[-1] / etf_window["close"].iloc[0]) - 1) * 100 if etf_window["close"].iloc[0] else 0
            ours = float(symbol_data["dollar_value"].sum())
            rows.append({
                "member": member,
                "symbol": symbol,
                "return_pct": ours,
                "stock_return": stock_return,
                "sector_return": sector_return,
                "beat_stock": stock_return < ours,
                "beat_sector": sector_return < ours,
                "sector": sector_meta["sector"].iloc[0] if not sector_meta.empty else "Unknown",
                "sector_etf": sector_etf,
            })
    return pd.DataFrame(rows)


def render_vs_industry():
    comparison = compute_stock_vs_sector_comparison()
    if comparison.empty:
        st.info("No data yet")
        return
    fig = go.Figure()
    for metric in ["stock_return", "sector_return"]:
        fig.add_trace(go.Bar(name=metric.replace('_', ' ').title(), x=comparison["symbol"], y=comparison[metric], marker_color="#7f8c8d" if metric == "sector_return" else "#3a86ff"))
    fig.update_layout(title="Stock vs sector performance", barmode="group", template=plotly_template)
    st.plotly_chart(fig, width="stretch")

    sector_summary = comparison.groupby("sector").agg(stock_return=("stock_return", "mean"), sector_return=("sector_return", "mean")).reset_index()
    fig2 = px.bar(sector_summary, x="sector", y=["stock_return", "sector_return"], title="Sector level return summary", template=plotly_template, barmode="group")
    st.plotly_chart(fig2, width="stretch")

    table_df = comparison[["member", "symbol", "sector", "sector_etf", "stock_return", "sector_return", "beat_stock", "beat_sector"]].copy()
    table_df["beat_stock"] = table_df["beat_stock"].map({True: "Yes", False: "No"})
    table_df["beat_sector"] = table_df["beat_sector"].map({True: "Yes", False: "No"})
    st.dataframe(table_df, width="stretch")


def render_risk():
    risk_df = []
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        temp = equity_df.copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp = temp.dropna(subset=["date"]).sort_values("date")
        temp["equity"] = coerce_numeric_series(temp["equity"])
        temp = temp.dropna(subset=["equity"])
        if temp.empty:
            continue
        temp["pct_change"] = temp["equity"].pct_change().fillna(0)
        temp["drawdown"] = (temp["equity"] / temp["equity"].cummax() - 1) * 100
        temp["volatility"] = temp["pct_change"].rolling(20).std() * (252 ** 0.5)
        risk_df.append({
            "member": member,
            "drawdown": float(temp["drawdown"].min()) if not temp["drawdown"].empty else 0.0,
            "volatility": float(temp["volatility"].dropna().iloc[-1]) if not temp["volatility"].dropna().empty else 0.0,
        })

    if risk_df:
        drawdown_df = pd.DataFrame(risk_df)
        fig1 = px.bar(drawdown_df, x="member", y="drawdown", color="member", title="Max drawdown by member", labels={"member": "Member", "drawdown": "Drawdown (%)"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
        fig1.update_layout(xaxis_title="Member", yaxis_title="Drawdown (%)")
        st.plotly_chart(fig1, width="stretch")
        st.caption("Largest decline from a prior peak in each member's account equity series.")

        fig2 = px.bar(drawdown_df, x="member", y="volatility", color="member", title="Daily return volatility by member vs SPY", labels={"member": "Member", "volatility": "Volatility"}, template=plotly_template, color_discrete_map=MEMBER_COLORS)
        fig2.update_layout(xaxis_title="Member", yaxis_title="Annualized volatility")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Annualized volatility based on daily returns in the member's equity history.")

    positions = load_team_position_data()
    high_beta_summary = []
    sector_map = load_stock_sectors()
    for member, df in positions.items():
        if df.empty:
            continue
        numeric_cols = [c for c in ["market_value", "unrealized_pl", "qty", "avg_entry_price", "current_price", "unrealized_plpc"] if c in df.columns]
        for col in numeric_cols:
            df[col] = coerce_numeric_series(df[col])
        total_value = float(df["market_value"].sum()) if "market_value" in df.columns else 0.0
        beta_symbols = sector_map.loc[(sector_map["member"] == member) & (sector_map["high_beta"] == True), "symbol"].tolist()
        high_beta_value = float(df.loc[df["symbol"].isin(beta_symbols), "market_value"].sum()) if not df.empty and beta_symbols else 0.0
        high_beta_summary.append({
            "member": member,
            "high_beta": float(high_beta_value) if high_beta_value is not None else 0.0,
            "core": float(total_value) - float(high_beta_value if high_beta_value is not None else 0.0),
        })
    if high_beta_summary:
        high_beta_df = pd.DataFrame(high_beta_summary)
        fig3 = px.bar(high_beta_df, x="member", y=["high_beta", "core"], barmode="group", title="High-beta vs core share per member", labels={"value": "$", "member": "Member"}, template=plotly_template, color_discrete_map={"high_beta": "#DAA520", "core": "#7570B3"})
        fig3.update_layout(xaxis_title="Member", yaxis_title="$ in portfolio")
        st.plotly_chart(fig3, width="stretch")
        st.caption("High-beta positions are those flagged as such in the team sector map; the rest are treated as core holdings.")


def render_portfolios():
    positions = load_team_position_data()
    for member in MEMBERS:
        df = positions.get(member)
        equity_exists = load_team_equity_data().get(member) is not None and not load_team_equity_data().get(member).empty
        raw_positions = read_member_file(member, "positions")
        fallback_path = TEAM_DIR / f"{member}_last_positions.csv"
        fallback_used = (raw_positions is None or raw_positions.empty) and fallback_path.exists()
        if df is None or df.empty:
            st.subheader(member.title())
            if equity_exists:
                st.info("All in cash")
            else:
                st.info("No data yet")
            continue
        st.subheader(member.title())
        sector_map = load_stock_sectors()
        merged = df.merge(sector_map[["symbol", "sector"]], on="symbol", how="left")
        if fallback_used:
            try:
                last_df = pd.read_csv(fallback_path)
                if not last_df.empty and "as_of_date" in last_df.columns:
                    as_of = pd.to_datetime(last_df["as_of_date"].dropna().iloc[0]).to_pydatetime()
                    next_day = (as_of + pd.Timedelta(days=1)).strftime("%d %b")
                    caption = f"Last positions held on {as_of.strftime('%d %b %Y')} - {member.title()} has been all in cash since {next_day}"
                    st.caption(caption)
            except Exception:
                pass
        if "unrealized_pl" in merged.columns:
            fig = px.bar(merged, x="symbol", y="unrealized_pl", color="symbol", title=f"{member.title()} current position P&L", template=plotly_template, color_discrete_map={symbol: MEMBER_COLORS.get(member, "#7F8C8D") for symbol in merged["symbol"].unique()})
            st.plotly_chart(fig, width="stretch")
            if fallback_used:
                st.caption(f"Last positions held on {as_of.strftime('%d %b %Y')} - {member.title()} has been all in cash since {next_day}")
        sectors = merged.groupby("sector", as_index=False)["market_value"].sum()
        if not sectors.empty:
            pie = px.pie(sectors, names="sector", values="market_value", title=f"{member.title()} sector allocation", template=plotly_template)
            st.plotly_chart(pie, width="stretch")
            if fallback_used:
                st.caption(f"Last positions held on {as_of.strftime('%d %b %Y')} - {member.title()} has been all in cash since {next_day}")
        st.dataframe(merged[["symbol", "qty", "market_value", "unrealized_pl", "sector"]], width="stretch")


def render_run_the_crew():
    st.sidebar.header("Run settings")
    stock_text = st.text_area("Stock list", value="\n".join(DEFAULT_STOCKS), height=220)
    selected_indicators = {}
    st.sidebar.write("RAMS indicators")
    for label in INDICATOR_ORDER:
        selected_indicators[label] = st.checkbox(label, value=True, key=f"ram_{label.lower()}")
    st.sidebar.write("API keys (stored only in memory for this run)")
    alpaca_key = st.sidebar.text_input("Alpaca API key", type="password", key="alpaca_key")
    alpaca_secret = st.sidebar.text_input("Alpaca secret key", type="password", key="alpaca_secret")
    groq_key = st.sidebar.text_input("Groq API key", type="password", key="groq_key")

    st.sidebar.divider()
    if st.sidebar.button("Test run", width="stretch"):
        write_stock_file(stock_text)
        if not alpaca_key or not alpaca_secret or not groq_key:
            st.sidebar.warning("Enter all three API keys before running the crew.")
        else:
            with st.spinner("Running dry-run crew..."):
                launched = launch_mcp_server_if_needed(alpaca_key, alpaca_secret, selected_indicators)
                if not launched:
                    st.error("The MCP server did not start on localhost:8001. Check that the server can run and try again.")
                else:
                    rc, stdout, stderr = run_crew_script("dry_run", selected_indicators, alpaca_key, alpaca_secret, groq_key)
                    st.session_state["last_run_output"] = {"mode": "Dry run", "stdout": stdout, "stderr": stderr, "returncode": rc}
                    if rc == 0:
                        st.success("Dry run completed.")
                    else:
                        st.error(f"Dry run failed with exit code {rc}.")

    st.sidebar.write("")
    confirm_live = st.sidebar.checkbox("I confirm", key="confirm_live")
    if st.sidebar.button("Run and place orders", type="primary", width="stretch"):
        write_stock_file(stock_text)
        if not confirm_live:
            st.sidebar.warning("You must check 'I confirm' before placing live orders.")
            st.stop()
        if not alpaca_key or not alpaca_secret or not groq_key:
            st.sidebar.warning("Enter all three API keys before running the live crew.")
            st.stop()
        try:
            portfolio = fetch_portfolio()
            if portfolio.get("error"):
                raise RuntimeError(portfolio["error"])
            open_orders = portfolio.get("open_orders", {})
            if isinstance(open_orders, dict) and open_orders.get("count", 0) > 0:
                st.sidebar.warning("There are already open orders on the account. Clear them or confirm there are no duplicates before placing new live orders.")
                st.stop()
        except Exception as exc:
            st.sidebar.warning(f"Could not check existing orders before live trading: {exc}")

        with st.spinner("Running live crew with auto execution..."):
            launched = launch_mcp_server_if_needed(alpaca_key, alpaca_secret, selected_indicators)
            if not launched:
                st.error("The MCP server did not start on localhost:8001. Live trade execution is blocked.")
            else:
                rc, stdout, stderr = run_crew_script("live", selected_indicators, alpaca_key, alpaca_secret, groq_key)
                st.session_state["last_run_output"] = {"mode": "Live trade run", "stdout": stdout, "stderr": stderr, "returncode": rc}
                if rc == 0:
                    st.success("Live run completed, and the crew was allowed to place orders.")
                else:
                    st.error(f"Live run failed with exit code {rc}.")

    st.header("Crew output")
    last_run_output = st.session_state.get("last_run_output")
    if last_run_output:
        st.caption(last_run_output["mode"])
        st.code(last_run_output["stdout"] or last_run_output["stderr"] or "No output captured.")
        if last_run_output["stderr"]:
            st.code(last_run_output["stderr"], language="text")
    else:
        st.info("No crew run has been executed yet.")

    st.header("Newest run reports")
    latest_folder = get_latest_run_folder()
    if latest_folder is None:
        st.info("No run folder has been generated yet in crew_ai/outputs.")
    else:
        st.caption(f"Folder: {latest_folder.name}")
        files = sorted(latest_folder.glob("*.md"), key=lambda p: p.name)
        if not files:
            st.info(f"No markdown reports were generated in {latest_folder.name}.")
        else:
            for report in files:
                with st.expander(report.name, expanded=False):
                    st.markdown(report.read_text(encoding="utf-8"))


def main():
    if not APP_PASSWORD:
        st.error("APP_PASSWORD is not set in the project .env file. Add it, then reload the page.")
        st.stop()

    entered = st.text_input("Enter password", type="password", key="app_password_entry")
    if entered.strip() != APP_PASSWORD:
        st.warning("Enter the correct password to access the dashboard.")
        st.stop()

    st.title("Team Trading Dashboard")
    st.markdown("<style>body { background: #0e1117; } .stApp { background: #0e1117; } .stTabs [role='tablist'] button { background: #111827; color: white; }</style>", unsafe_allow_html=True)

    tabs = st.tabs(["Team overview", "Stock activity", "Vs industry", "Risk", "Portfolios", "Run the crew"])
    with tabs[0]:
        render_team_overview()
    with tabs[1]:
        render_stock_activity()
    with tabs[2]:
        render_vs_industry()
    with tabs[3]:
        render_risk()
    with tabs[4]:
        render_portfolios()
    with tabs[5]:
        render_run_the_crew()


if __name__ == "__main__":
    main()
