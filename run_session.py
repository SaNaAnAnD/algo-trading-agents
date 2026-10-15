"""
Autonomous trading session (one command for the whole evening):

    conda activate trading-agents
    cd ~/algo-trading-agents
    python run_session.py

- Starts the Alpaca MCP server in the background if it is not already running
  (its log goes to crew_ai/outputs/server_log.txt).
- Waits for the US market open (times shown in IST), first run 10 minutes
  after the open, then runs the full crew in --auto mode every 15 minutes:
  market data, news and technical agents, RAMS scores, Risk Manager and
  Trader, then the orders - with no questions asked.
- No new run in the last 10 minutes; stops by itself after the close.
- One line per run is added to crew_ai/outputs/session_log.txt.
- Ctrl + C stops it at any time.
Options: --every 20 (minutes between runs), --now (run once immediately, then continue).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "crew_ai" / "outputs"   # same folder as the crew's run_* reports
IST = timezone(timedelta(hours=5, minutes=30))
load_dotenv(PROJECT / ".env")


def ist(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%a %d %b %I:%M %p IST")


def log(text: str) -> None:
    OUT.mkdir(exist_ok=True)
    line = f"{datetime.now(IST):%d-%m-%Y %H:%M} IST  {text}"
    print(">> " + line)
    with (OUT / "session_log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def server_up() -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("localhost", 8001)) == 0


def start_server():
    if server_up():
        print("MCP server already running on port 8001.")
        return None
    OUT.mkdir(exist_ok=True)
    logf = open(OUT / "server_log.txt", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(PROJECT / "servers" / "alpaca_trading_mcpServer.py")],
                            cwd=PROJECT, stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    for _ in range(60):
        if server_up():
            print("MCP server started (log: crew_ai/outputs/server_log.txt).")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    print("!! The MCP server did not start. Look at crew_ai/outputs/server_log.txt")
    sys.exit(1)


def clock(tc: TradingClient):
    for _ in range(5):
        try:
            return tc.get_clock()
        except Exception as e:
            print(f"  market clock unavailable ({e}); retrying in 1 min")
            time.sleep(60)
    return None


def wait(secs: float, label: str) -> None:
    if secs > 0:
        print(f"  {label} (in {int(secs // 60)} min). Leave this window open.")
        time.sleep(secs)


def run_crew(n: int) -> None:
    env = dict(os.environ, CREWAI_TRACING_ENABLED="false", PYTHONUNBUFFERED="1")
    print(f"\n{'#' * 70}\n# Run {n} | {ist(datetime.now(timezone.utc))}\n{'#' * 70}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(PROJECT / "crew_ai" / "clients" / "trading_crew.py"), "--auto"],
                           cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, timeout=12 * 60)
        log(f"run {n}: finished (exit {r.returncode}) in {int(time.time() - t0)} s - details in the newest crew_ai/outputs/run_* folder")
    except subprocess.TimeoutExpired:
        log(f"run {n}: took longer than 12 min, stopped; next run continues")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=15, help="minutes between runs (default 15)")
    ap.add_argument("--now", action="store_true", help="first run immediately if the market is open")
    a = ap.parse_args()
    if os.getenv("TRADING_ENABLED", "").lower() != "true" or os.getenv("DRY_RUN", "").lower() != "false":
        print("NOTE: .env has TRADING_ENABLED/DRY_RUN set so that no orders are placed (dry run).")
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    every = timedelta(minutes=max(10, a.every))
    print("=" * 70)
    print(f" Autonomous trading session | a run every {every.seconds // 60} min | Ctrl + C to stop")
    print("=" * 70)
    server = start_server()
    runs, opened = 0, False
    try:
        while True:
            c = clock(tc)
            if c is None:
                print("Market clock unavailable for 5 minutes. Check the internet and start again.")
                break
            now, close = c.timestamp, c.next_close
            if not c.is_open:
                if opened:
                    print(f"\nMarket closed ({ist(now)}). {runs} run(s) today.")
                    break
                if c.next_open - now > timedelta(hours=14):
                    print(f"No US trading session in the next 14 hours. Next open: {ist(c.next_open)}.")
                    break
                target = min(c.next_open + timedelta(minutes=10), now + timedelta(minutes=30))
                wait((target - now).total_seconds(), f"Market not open yet; waiting until {ist(target)}")
                continue
            opened = True
            if runs == 0 and not a.now:
                first = now.replace(hour=9, minute=40, second=0, microsecond=0)   # New York time
                if now < first:
                    wait((first - now).total_seconds(), f"Market just opened; first run at {ist(first)}")
                    continue
            if close - now < timedelta(minutes=10):
                wait((close - now).total_seconds() + 60, "Close is near; no new run. Stopping after the close")
                continue
            runs += 1
            run_crew(runs)
            c2 = clock(tc)
            now2 = c2.timestamp if c2 else now
            nxt = now + every
            if nxt >= close - timedelta(minutes=10):
                wait((close - now2).total_seconds() + 60, "That was the last run today. Stopping after the close")
            else:
                wait((nxt - now2).total_seconds(), f"Run {runs} done. Next run at {ist(nxt)}")
    except KeyboardInterrupt:
        print("\nStopped by you (Ctrl + C).")
    log(f"session ended after {runs} run(s)")
    if server is not None:
        server.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
