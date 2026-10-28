import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
from fastmcp import Client

ROOT = Path(__file__).resolve().parent
STOCKS_FILE = ROOT / "crew_ai" / "clients" / "my_stocks.txt"
OUTPUTS_DIR = ROOT / "crew_ai" / "outputs"
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
INDICATOR_TO_KEY = {
    "Trend": "trend",
    "Momentum": "momentum",
    "Volume": "volume",
    "Breakout": "breakout",
    "News": "news",
}


@st.cache_data(show_spinner=False)
def get_latest_run_folder() -> Path | None:
    if not OUTPUTS_DIR.exists():
        return None
    folders = [p for p in OUTPUTS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not folders:
        return None
    return max(folders, key=lambda p: p.stat().st_mtime)


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
from pathlib import Path

root = Path(r""" + root_str.replace("\\", "\\\\") + """)
config_path = root / "servers" / "config.py"

spec = importlib.util.spec_from_file_location("runtime_config", config_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

selection = {str(name).lower(): bool(enabled) for name, enabled in json.loads(os.environ.get("RAMS_SELECTION", "{}")).items()}
weights = {
    "trend": 0.30,
    "momentum": 0.30,
    "volume": 0.15,
    "breakout": 0.10,
    "news": 0.15,
}
active = {name for name, enabled in selection.items() if enabled}
if not active:
    active = set(weights)
scale = sum(weights[name] for name in active) or 1.0
for name in weights:
    value = weights[name] / scale if name in active else 0.0
    setattr(module, f"W_{name.upper()}", value)

sys.modules["config"] = module
sys.path.insert(0, str(root / "servers"))
runpy.run_path(str(root / "servers" / "alpaca_trading_mcpServer.py"), run_name="__main__")
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
        return {
            "account": account,
            "positions": positions,
            "open_orders": open_orders,
        }
    except Exception as exc:  # pragma: no cover - user-facing UI handling
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

    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return completed.returncode, completed.stdout, completed.stderr


st.set_page_config(page_title="Trading Crew Dashboard", page_icon="📈", layout="wide")
st.title("Trading Crew Dashboard")

with st.sidebar:
    st.header("Run settings")
    stock_text = st.text_area(
        "Stock list",
        value="\n".join(DEFAULT_STOCKS),
        height=220,
        help="Edit the list before running the crew. The file is updated for the current run.",
    )

    selected_indicators = {}
    st.write("RAMS indicators")
    for label in INDICATOR_ORDER:
        selected_indicators[label] = st.checkbox(label, value=True, key=f"ram_{label.lower()}")

    st.write("API keys (stored only in memory for this run)")
    alpaca_key = st.text_input("Alpaca API key", type="password", key="alpaca_key")
    alpaca_secret = st.text_input("Alpaca secret key", type="password", key="alpaca_secret")
    groq_key = st.text_input("Groq API key", type="password", key="groq_key")

    st.divider()
    if st.button("Test run", use_container_width=True):
        write_stock_file(stock_text)
        if not alpaca_key or not alpaca_secret or not groq_key:
            st.warning("Enter all three API keys before running the crew.")
        else:
            with st.spinner("Running dry-run crew..."):
                launched = launch_mcp_server_if_needed(alpaca_key, alpaca_secret, selected_indicators)
                if not launched:
                    st.error("The MCP server did not start on localhost:8001. Check that the server can run and try again.")
                else:
                    rc, stdout, stderr = run_crew_script("dry_run", selected_indicators, alpaca_key, alpaca_secret, groq_key)
                    st.session_state["last_run_output"] = {
                        "mode": "Dry run",
                        "returncode": rc,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                    if rc == 0:
                        st.success("Dry run completed.")
                    else:
                        st.error(f"Dry run failed with exit code {rc}.")

    st.write("")
    confirm_live = st.checkbox("I confirm", key="confirm_live")
    if st.button("Run and place orders", type="primary", use_container_width=True):
        write_stock_file(stock_text)
        if not confirm_live:
            st.warning("You must check 'I confirm' before placing live orders.")
            st.stop()
        if not alpaca_key or not alpaca_secret or not groq_key:
            st.warning("Enter all three API keys before running the live crew.")
            st.stop()
        try:
            portfolio = fetch_portfolio()
            if portfolio.get("error"):
                raise RuntimeError(portfolio["error"])
            open_orders = portfolio.get("open_orders", {})
            if isinstance(open_orders, dict) and open_orders.get("count", 0) > 0:
                st.warning("There are already open orders on the account. Clear them or confirm there are no duplicates before placing new live orders.")
                st.stop()
        except Exception as exc:
            st.warning(f"Could not check existing orders before live trading: {exc}")

        with st.spinner("Running live crew with auto execution..."):
            launched = launch_mcp_server_if_needed(alpaca_key, alpaca_secret, selected_indicators)
            if not launched:
                st.error("The MCP server did not start on localhost:8001. Live trade execution is blocked.")
            else:
                rc, stdout, stderr = run_crew_script("live", selected_indicators, alpaca_key, alpaca_secret, groq_key)
                st.session_state["last_run_output"] = {
                    "mode": "Live trade run",
                    "returncode": rc,
                    "stdout": stdout,
                    "stderr": stderr,
                }
                if rc == 0:
                    st.success("Live run completed, and the crew was allowed to place orders.")
                else:
                    st.error(f"Live run failed with exit code {rc}.")

st.header("Portfolio dashboard")
portfolio = fetch_portfolio()
if portfolio.get("error"):
    st.error(f"Unable to read portfolio data from the MCP server: {portfolio['error']}")
else:
    account = portfolio.get("account") or {}
    positions = portfolio.get("positions") or {}

    account_value = account.get("equity") or account.get("portfolio_value") or "n/a"
    available_cash = account.get("cash") or account.get("available_cash") or "n/a"

    col1, col2, col3 = st.columns(3)
    col1.metric("Account value", str(account_value))
    col2.metric("Available cash", str(available_cash))
    col3.metric("Open positions", str(positions.get("count", 0)))

    position_rows = positions.get("positions", [])
    if position_rows:
        st.subheader("Current positions")
        st.dataframe(position_rows, use_container_width=True)
    else:
        st.info("No positions are currently open.")

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
