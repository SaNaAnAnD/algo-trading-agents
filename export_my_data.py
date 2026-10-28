#!/usr/bin/env python3
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
TEAM_DIR = ROOT / "data" / "team"


def parse_args():
    parser = argparse.ArgumentParser(description="Export one team member's Alpaca trading data to data/team.")
    parser.add_argument("--member", required=True, help="Team member name, e.g. sana")
    return parser.parse_args()


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def export_member(member: str):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env before exporting.")

    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    start_date = datetime(2026, 10, 9, tzinfo=timezone.utc)
    end_date = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=1)

    try:
        history = client.get_portfolio_history(start=start_date, end=end_date, timeframe="1D")
        equity_rows = []
        for idx, value in enumerate(history.equity):
            equity_rows.append({"date": history.timestamp[idx].strftime("%Y-%m-%d"), "equity": float(value)})
        equity_df = pd.DataFrame(equity_rows)
        save_csv(equity_df, TEAM_DIR / f"{member}_equity.csv")
    except Exception:
        save_csv(pd.DataFrame(columns=["date", "equity"]), TEAM_DIR / f"{member}_equity.csv")

    try:
        orders = client.get_orders(status="closed")
        rows = []
        for order in orders:
            if getattr(order, "filled_at", None) is None:
                continue
            rows.append({
                "date": getattr(order.filled_at, "strftime", lambda fmt: str(order.filled_at))("%Y-%m-%d"),
                "symbol": str(order.symbol),
                "side": str(order.side).lower(),
                "qty": float(getattr(order, "qty", 0) or 0),
                "price": float(getattr(order, "limit_price", getattr(order, "avg_fill_price", 0)) or 0),
            })
        trades_df = pd.DataFrame(rows)
        save_csv(trades_df, TEAM_DIR / f"{member}_trades.csv")
    except Exception:
        save_csv(pd.DataFrame(columns=["date", "symbol", "side", "qty", "price"]), TEAM_DIR / f"{member}_trades.csv")

    try:
        positions = client.get_all_positions()
        rows = []
        for position in positions:
            rows.append({
                "symbol": str(position.symbol),
                "qty": float(getattr(position, "qty", 0) or 0),
                "avg_entry_price": float(getattr(position, "avg_entry_price", 0) or 0),
                "current_price": float(getattr(position, "current_price", 0) or 0),
                "market_value": float(getattr(position, "market_value", 0) or 0),
                "unrealized_pl": float(getattr(position, "unrealized_pl", 0) or 0),
                "unrealized_plpc": float(getattr(position, "unrealized_plpc", 0) or 0),
            })
        positions_df = pd.DataFrame(rows)
        save_csv(positions_df, TEAM_DIR / f"{member}_positions.csv")
    except Exception:
        save_csv(pd.DataFrame(columns=["symbol", "qty", "avg_entry_price", "current_price", "market_value", "unrealized_pl", "unrealized_plpc"]), TEAM_DIR / f"{member}_positions.csv")

    print(f"Exported data for {member} to {TEAM_DIR}")
    print(f"Account value: {account.equity}")


if __name__ == "__main__":
    args = parse_args()
    export_member(args.member.strip().lower())
