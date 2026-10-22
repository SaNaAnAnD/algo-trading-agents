"""
Autonomous trading session (one command for the whole evening):

    conda activate trading-agents
    cd ~/algo-trading-agents
    python run_session.py

- Starts the Alpaca MCP server in the background if it is not already running
  (its log goes to crew_ai/outputs/<name>/server_log.txt).
- Waits for the US market open (times shown in IST), first run 10 minutes
  after the open, then runs the full crew in --auto mode every 45 minutes:
  market data, news and technical agents, RAMS scores, Risk Manager and
  Trader, then the orders - with no questions asked.
- No new run in the last 10 minutes; stops by itself after the close.
- One line per run is added to crew_ai/outputs/<name>/session_log.txt: equity, % change
  since the $100,000 start, % invested, cash and the positions held.
- At the end it writes crew_ai/outputs/<name>/daily_summary_<date>.md (start and end
  equity, every order of the day, final positions) and saves everything to
  GitHub (git add, commit, push). --no-save skips the GitHub step.
- Ctrl + C stops it at any time.
Options: --every 20 (minutes between runs), --now (run once immediately, then continue).
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

for _stream in (sys.stdout, sys.stderr):   # Windows consoles: print emojis safely
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# UTF-8 for the server and crew child processes too (Windows defaults to cp1252)
CHILD_ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

PROJECT = Path(__file__).resolve().parent
IST = timezone(timedelta(hours=5, minutes=30))
load_dotenv(PROJECT / ".env")
# same folder as the crew's run_* reports: crew_ai/outputs/<TRADER_NAME>/
_TRADER = re.sub(r"[^a-z0-9_-]", "", os.getenv("TRADER_NAME", "").strip().lower())
OUT = PROJECT / "crew_ai" / "outputs" / _TRADER if _TRADER else PROJECT / "crew_ai" / "outputs"


def ist(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%a %d %b %I:%M %p IST")


def log(text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(IST):%d-%m-%Y %H:%M} IST  {text}"
    print(">> " + line)
    with (OUT / "session_log.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


START_EQUITY = 100_000.0   # contest starting capital
EVERY_MIN = 45             # minutes between runs (set from --every)


def snapshot(tc):
    """Account numbers for the log, or None if Alpaca is unreachable."""
    try:
        acct = tc.get_account()
        equity, cash = float(acct.equity), float(acct.cash)
        positions = tc.get_all_positions()
    except Exception:
        return None
    invested = sum(float(p.market_value) for p in positions)
    return {"equity": equity, "cash": cash, "invested": invested,
            "positions": [(p.symbol, p.qty, float(p.market_value), float(p.unrealized_pl))
                          for p in positions]}


def snap_text(snap):
    if not snap:
        return "account unavailable"
    held = ", ".join(f"{s} {q}" for s, q, _, _ in snap["positions"]) or "none"
    return (f"equity ${snap['equity']:,.2f} ({100 * (snap['equity'] / START_EQUITY - 1):+.2f}% vs start) | "
            f"invested {100 * snap['invested'] / snap['equity']:.0f}% | cash ${snap['cash']:,.0f} | "
            f"holding: {held}")


def daily_summary(tc, first, last, runs, started_utc):
    """crew_ai/outputs/<name>/daily_summary_<date>.md for the report."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    day = started_utc   # the US trading day (the session ends after midnight IST)
    lines = [f"# Trading session {day:%d %b %Y} (US market day)", "",
             f"Runs: {runs} (one every {EVERY_MIN} minutes, full CrewAI crew in auto mode)", ""]
    if first and last:
        change = last["equity"] - first["equity"]
        lines += ["| | Start of session | End of session |", "|---|---|---|",
                  f"| Equity | ${first['equity']:,.2f} | ${last['equity']:,.2f} |",
                  f"| Invested | {100 * first['invested'] / first['equity']:.0f}% | "
                  f"{100 * last['invested'] / last['equity']:.0f}% |",
                  f"| Cash | ${first['cash']:,.0f} | ${last['cash']:,.0f} |", "",
                  f"Session change: ${change:+,.2f} ({100 * change / first['equity']:+.2f}%). "
                  f"Since the $100,000 start: {100 * (last['equity'] / START_EQUITY - 1):+.2f}%.", ""]
    try:
        orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=started_utc, limit=500))
        lines += ["## Orders", "", "| Time (IST) | Side | Qty | Stock | Type | Status | Fill price |",
                  "|---|---|---|---|---|---|---|"]
        for o in sorted(orders, key=lambda o: o.submitted_at):
            price = f"${float(o.filled_avg_price):,.2f}" if o.filled_avg_price else ""
            lines.append(f"| {o.submitted_at.astimezone(IST):%H:%M} | {str(o.side).split('.')[-1]} | {o.qty} | "
                         f"{o.symbol} | {str(o.order_type).split('.')[-1]} | {str(o.status).split('.')[-1]} | {price} |")
        lines.append("")
    except Exception as e:
        lines += [f"(orders unavailable: {e})", ""]
    if last:
        lines += ["## Positions at the end", "", "| Stock | Qty | Value | Unrealised P/L |", "|---|---|---|---|"]
        lines += [f"| {s} | {q} | ${v:,.2f} | ${pl:+,.2f} |" for s, q, v, pl in last["positions"]]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"daily_summary_{day:%Y-%m-%d}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Daily summary written: {path.relative_to(PROJECT)}")


def save_to_github(text):
    """git add / commit / push of the day's records (never the .env file)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=120)
    try:
        git("add", "-A")
        staged = git("diff", "--cached", "--name-only").stdout.split()
        if any(Path(f).name == ".env" for f in staged):
            git("reset", "-q", "HEAD", "--", ".env")
        git("commit", "-q", "-m", text)
        r = git("push")
        print("Saved to GitHub." if r.returncode == 0 else
              f"GitHub push failed ({(r.stderr or '').strip()[:150]}). Run  git push  later.")
    except subprocess.TimeoutExpired:
        print("GitHub push timed out (login needed?). Run  git push  when you are at the PC.")


def server_up() -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("localhost", 8001)) == 0


def start_server():
    if server_up():
        print("MCP server already running on port 8001.")
        return None
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "server_log.txt", "a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(PROJECT / "servers" / "alpaca_trading_mcpServer.py")],
                            cwd=PROJECT, stdout=logf, stderr=subprocess.STDOUT, env=CHILD_ENV,
                            stdin=subprocess.DEVNULL)
    for _ in range(60):
        if server_up():
            print(f"MCP server started (log: {(OUT / 'server_log.txt').relative_to(PROJECT)}).")
            return proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    print("!! The MCP server did not start. Look at crew_ai/outputs/<name>/server_log.txt")
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
    env = dict(CHILD_ENV, CREWAI_TRACING_ENABLED="false", PYTHONUNBUFFERED="1")
    print(f"\n{'#' * 70}\n# Run {n} | {ist(datetime.now(timezone.utc))}\n{'#' * 70}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, str(PROJECT / "crew_ai" / "clients" / "trading_crew.py"), "--auto"],
                           cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, timeout=12 * 60)
        log(f"run {n}: finished (exit {r.returncode}) in {int(time.time() - t0)} s - details in the newest run_* folder")
    except subprocess.TimeoutExpired:
        log(f"run {n}: took longer than 12 min, stopped; next run continues")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=45, help="minutes between runs (default 45)")
    ap.add_argument("--now", action="store_true", help="first run immediately if the market is open")
    ap.add_argument("--no-save", action="store_true", help="do not commit and push to GitHub at the end")
    a = ap.parse_args()
    if os.getenv("TRADING_ENABLED", "").lower() != "true" or os.getenv("DRY_RUN", "").lower() != "false":
        print("NOTE: .env has TRADING_ENABLED/DRY_RUN set so that no orders are placed (dry run).")
    tc = TradingClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True)
    global EVERY_MIN
    EVERY_MIN = max(10, a.every)
    every = timedelta(minutes=EVERY_MIN)
    print("=" * 70)
    print(f" Autonomous trading session | a run every {every.seconds // 60} min | Ctrl + C to stop")
    print("=" * 70)
    server = start_server()
    runs, opened = 0, False
    first_snap = last_snap = None
    started_utc = datetime.now(timezone.utc)
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
            if not server_up():   # the MCP server stopped: start it again
                log("MCP server was not running; restarting it")
                server = start_server() or server
            if first_snap is None:
                first_snap = snapshot(tc)
                log("start: " + snap_text(first_snap))
            runs += 1
            run_crew(runs)
            last_snap = snapshot(tc) or last_snap
            log(f"after run {runs}: " + snap_text(last_snap))
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
    if runs:
        last_snap = snapshot(tc) or last_snap
        daily_summary(tc, first_snap, last_snap, runs, started_utc)
        if not a.no_save:
            eq = f", equity ${last_snap['equity']:,.0f}" if last_snap else ""
            save_to_github(f"Trading session {started_utc:%d %b %Y}: {runs} runs{eq}")
    print("Session finished. You can close VS Code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
