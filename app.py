import asyncio
import json
import os
import socket
import subprocess
import sys
import time
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
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

st.set_page_config(page_title="Team Trading Dashboard", page_icon="📊", layout="wide")

if "plotly_dark" not in px.colors.named_colorscales():
    pass

plotly_template = "plotly_dark"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        if df is not None and not df.empty:
            temp = df.copy()
            if "date" in temp.columns:
                temp["date"] = pd.to_datetime(temp["date"]).dt.date
            if "equity" in temp.columns:
                temp["equity"] = pd.to_numeric(temp["equity"], errors="coerce")
            out[member] = temp.dropna(subset=["equity"]).sort_values("date")
        else:
            out[member] = pd.DataFrame(columns=["date", "equity"])
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
        temp["member"] = member
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
        temp["qty"] = pd.to_numeric(temp["qty"], errors="coerce")
        temp["price"] = pd.to_numeric(temp["price"], errors="coerce")
        temp["dollar_value"] = temp["qty"].abs() * temp["price"]
        out[member] = temp.dropna(subset=["date", "symbol"]).sort_values("date")
    return out


@st.cache_data(show_spinner=False)
def load_team_position_data() -> dict[str, pd.DataFrame]:
    out = {}
    for member in MEMBERS:
        df = read_member_file(member, "positions")
        if df is None or df.empty:
            out[member] = pd.DataFrame(columns=["symbol", "qty", "market_value", "unrealized_pl", "unrealized_plpc", "avg_entry_price", "current_price"])
            continue
        temp = df.copy()
        temp["symbol"] = temp["symbol"].astype(str).str.upper()
        temp["qty"] = pd.to_numeric(temp["qty"], errors="coerce")
        temp["market_value"] = pd.to_numeric(temp["market_value"], errors="coerce")
        temp["unrealized_pl"] = pd.to_numeric(temp["unrealized_pl"], errors="coerce")
        temp["unrealized_plpc"] = pd.to_numeric(temp["unrealized_plpc"], errors="coerce")
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
    df.columns = ["symbol", "date", "open", "high", "low", "close", "volume"]
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["symbol", "date", "close"]]


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


def build_member_summary(member: str):
    equity_df = load_team_equity_data().get(member)
    trades_df = load_team_trade_data().get(member)
    if equity_df is None or equity_df.empty:
        return {
            "member": member,
            "account_value": "No data yet",
            "return_pct": "No data yet",
            "trade_count": 0,
            "trading_days": 0,
            "status": "No data yet",
        }

    equity = equity_df["equity"].dropna()
    if equity.empty:
        return {"member": member, "account_value": "No data yet", "return_pct": "No data yet", "trade_count": 0, "trading_days": 0, "status": "No data yet"}

    start = equity.iloc[0]
    latest = equity.iloc[-1]
    pct = ((latest / start) - 1) * 100 if start else 0
    trade_count = int(len(trades_df)) if trades_df is not None else 0
    trading_days = int(equity_df["date"].nunique()) if "date" in equity_df.columns else 0
    return {
        "member": member,
        "account_value": f"${latest:,.2f}",
        "return_pct": f"{pct:.2f}%",
        "trade_count": trade_count,
        "trading_days": trading_days,
        "status": "OK",
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
                st.info("No data yet")
            else:
                st.metric("Account value", summary["account_value"])
                st.metric("Return", summary["return_pct"])
                st.metric("Trades", summary["trade_count"])
                st.metric("Trading days", summary["trading_days"])
    with cards[-1]:
        st.markdown("### Group")
        values = [float(summary["account_value"].replace('$', '').replace(',', '')) for summary in summaries.values() if summary["status"] == "OK"]
        returns = []
        for summary in summaries.values():
            if summary["status"] == "OK":
                returns.append(float(summary["return_pct"].replace('%', '')))
        st.metric("Member account value", f"${sum(values):,.2f}")
        st.metric("Group avg return", f"{(sum(returns) / len(returns)) if returns else 0:.2f}%")
        st.metric("Members with data", len(values))

    price_df = fetch_daily_prices(["SPY"] + list(set([x for _, row in load_stock_sectors().iterrows() for x in [row["symbol"], row["sector_etf"]]])))
    spy_prices = price_df[price_df["symbol"] == "SPY"].sort_values("date") if not price_df.empty else pd.DataFrame(columns=["date", "close"])
    if not spy_prices.empty:
        spy_prices["pct_return"] = compute_cumulative_return(spy_prices["close"])
        spy_prices["date"] = pd.to_datetime(spy_prices["date"])
    fig = go.Figure()
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        temp = equity_df.copy()
        temp["date"] = pd.to_datetime(temp["date"])
        temp = temp.sort_values("date")
        temp["pct_return"] = compute_cumulative_return(temp["equity"])
        fig.add_trace(go.Scatter(x=temp["date"], y=temp["pct_return"], mode="lines", name=member.title()))
    if not spy_prices.empty:
        fig.add_trace(go.Scatter(x=spy_prices["date"], y=spy_prices["pct_return"], mode="lines", line=dict(dash="dash"), name="SPY"))
    fig.update_layout(title="Member return vs SPY", template=plotly_template, xaxis_title="Date", yaxis_title="Cumulative return (%)", legend_title="Member")
    st.plotly_chart(fig, use_container_width=True)

    member_returns = []
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        start = equity_df["equity"].dropna().iloc[0]
        latest = equity_df["equity"].dropna().iloc[-1]
        member_returns.append({"member": member, "return_pct": ((latest / start) - 1) * 100 if start else 0})
    if member_returns:
        spy_base = spy_prices["close"].dropna().iloc[0] if not spy_prices.empty else None
        spy_last = spy_prices["close"].dropna().iloc[-1] if not spy_prices.empty else None
        spy_return = ((spy_last / spy_base) - 1) * 100 if spy_base and spy_last else 0
        group_avg = sum(item["return_pct"] for item in member_returns) / len(member_returns)
        return_df = pd.DataFrame(member_returns)
        return_df.loc[len(return_df)] = {"member": "Group avg", "return_pct": group_avg}
        return_df.loc[len(return_df)] = {"member": "SPY", "return_pct": spy_return}
        bar_fig = px.bar(return_df, x="member", y="return_pct", color="member", title="Total return by member vs group avg and SPY", template=plotly_template)
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("No data yet")


def render_stock_activity():
    activity = compute_stock_activity_data()
    if activity["trades_per_day"].empty:
        st.info("No data yet")
        return

    trade_days = activity["trades_per_day"].copy()
    trade_days["date"] = pd.to_datetime(trade_days["date"])
    fig1 = px.line(trade_days, x="date", y="trades", color="member", title="Trades per day by member", template=plotly_template)
    st.plotly_chart(fig1, use_container_width=True)

    heatmap_df = activity["heatmap"].copy()
    if not heatmap_df.empty:
        heat = heatmap_df.pivot_table(index="member", columns="symbol", values="trades", aggfunc="sum").fillna(0)
        fig2 = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title="Trades by member and stock", template=plotly_template)
        st.plotly_chart(fig2, use_container_width=True)

    buy_sell = activity["buys_vs_sells"].copy()
    if not buy_sell.empty:
        fig3 = px.bar(buy_sell, x="member", y="count", color="side", barmode="group", title="Buys vs sells per member", template=plotly_template)
        st.plotly_chart(fig3, use_container_width=True)

    dollar_amt = activity["dollar_amount"].copy()
    if not dollar_amt.empty:
        fig4 = px.bar(dollar_amt, x="member", y="dollar_amount", color="member", title="Dollar amount traded per member", template=plotly_template)
        st.plotly_chart(fig4, use_container_width=True)

    stock_counts = activity["distinct_stocks"].copy()
    if not stock_counts.empty:
        fig5 = px.bar(stock_counts, x="member", y="stock_count", color="member", title="Different stocks traded by member", template=plotly_template)
        st.plotly_chart(fig5, use_container_width=True)

    avg_hold = activity["avg_holding_days"].copy()
    if not avg_hold.empty:
        fig6 = px.bar(avg_hold, x="member", y="avg_holding_days", color="member", title="Average holding time per member", template=plotly_template)
        st.plotly_chart(fig6, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)

    sector_summary = comparison.groupby("sector").agg(stock_return=("stock_return", "mean"), sector_return=("sector_return", "mean")).reset_index()
    fig2 = px.bar(sector_summary, x="sector", y=["stock_return", "sector_return"], title="Sector level return summary", template=plotly_template, barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

    table_df = comparison[["member", "symbol", "sector", "sector_etf", "stock_return", "sector_return", "beat_stock", "beat_sector"]].copy()
    table_df["beat_stock"] = table_df["beat_stock"].map({True: "Yes", False: "No"})
    table_df["beat_sector"] = table_df["beat_sector"].map({True: "Yes", False: "No"})
    st.dataframe(table_df, use_container_width=True)


def render_risk():
    risk_df = []
    for member in MEMBERS:
        equity_df = load_team_equity_data().get(member)
        if equity_df is None or equity_df.empty:
            continue
        temp = equity_df.copy()
        temp["date"] = pd.to_datetime(temp["date"])
        temp = temp.sort_values("date")
        temp["pct_change"] = temp["equity"].pct_change().fillna(0)
        temp["drawdown"] = (temp["equity"] / temp["equity"].cummax() - 1) * 100
        temp["volatility"] = temp["pct_change"].rolling(20).std() * (252 ** 0.5)
        risk_df.append({"member": member, "drawdown": float(temp["drawdown"].min()), "volatility": float(temp["volatility"].dropna().iloc[-1]) if not temp["volatility"].dropna().empty else 0.0})

    if risk_df:
        drawdown_df = pd.DataFrame(risk_df)
        fig1 = px.bar(drawdown_df, x="member", y="drawdown", color="member", title="Max drawdown by member", template=plotly_template)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(drawdown_df, x="member", y="volatility", color="member", title="Daily return volatility by member vs SPY", template=plotly_template)
        st.plotly_chart(fig2, use_container_width=True)

    positions = load_team_position_data()
    high_beta_summary = []
    for member, df in positions.items():
        if df.empty:
            continue
        total_value = df["market_value"].sum() if "market_value" in df.columns else 0
        high_beta_value = df[df["symbol"].isin(load_stock_sectors()[load_stock_sectors()["member"] == member]["symbol"].tolist())].sum() if not df.empty else 0
        high_beta_summary.append({"member": member, "high_beta": high_beta_value, "core": total_value - high_beta_value})
    if high_beta_summary:
        high_beta_df = pd.DataFrame(high_beta_summary)
        fig3 = px.bar(high_beta_df, x="member", y=["high_beta", "core"], barmode="group", title="High-beta vs core share per member", template=plotly_template)
        st.plotly_chart(fig3, use_container_width=True)


def render_portfolios():
    positions = load_team_position_data()
    for member in MEMBERS:
        df = positions.get(member)
        if df is None or df.empty:
            st.subheader(member.title())
            st.info("No data yet")
            continue
        st.subheader(member.title())
        sector_map = load_stock_sectors()
        merged = df.merge(sector_map[["symbol", "sector"]], on="symbol", how="left")
        if "unrealized_pl" in merged.columns:
            fig = px.bar(merged, x="symbol", y="unrealized_pl", color="symbol", title=f"{member.title()} current position P&L", template=plotly_template)
            st.plotly_chart(fig, use_container_width=True)
        sectors = merged.groupby("sector", as_index=False)["market_value"].sum()
        if not sectors.empty:
            pie = px.pie(sectors, names="sector", values="market_value", title=f"{member.title()} sector allocation", template=plotly_template)
            st.plotly_chart(pie, use_container_width=True)
        st.dataframe(merged[["symbol", "qty", "market_value", "unrealized_pl", "sector"]], use_container_width=True)


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
    if st.sidebar.button("Test run", use_container_width=True):
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
    if st.sidebar.button("Run and place orders", type="primary", use_container_width=True):
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
    if entered != APP_PASSWORD:
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
