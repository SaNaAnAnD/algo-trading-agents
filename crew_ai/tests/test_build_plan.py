"""Network-free tests for the paper-trading plan builder."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))
from trading_crew import build_plan


SETTINGS = {
    "TARGET_INVESTED": 0.95,
    "STOP_ATR": 2.5,
    "TAKE_PROFIT_ATR": 3.0,
}


def stock(symbol, signal, price=100, atr=5, qty=10, rams=80, **extra):
    result = {
        "symbol": symbol,
        "signal": signal,
        "price": price,
        "atr": atr,
        "suggested_qty": qty,
        "rams": rams,
        "stop_loss": price - 2 * atr,
    }
    result.update(extra)
    return result


def position(symbol, qty=5, entry=100, value=500, unrealized=10):
    return {
        "symbol": symbol,
        "qty": str(qty),
        "avg_entry_price": str(entry),
        "market_value": str(value),
        "unrealized_pl": str(unrealized),
    }


def account(equity=1000, cash=1000):
    return {"equity": str(equity), "cash": str(cash)}


def test_take_profit_wins_over_add():
    final = {"stocks": [stock("HELD", "ADD", price=120, atr=10, qty=10)]}
    positions = {"positions": [position("HELD", qty=4, entry=80, value=480)]}
    plan = build_plan(final, positions, account(), SETTINGS)
    assert len(plan) == 1
    assert plan[0]["side"] == "sell" and plan[0]["qty"] == 2


def test_stop_loss_sell():
    final = {"stocks": [stock("STOP", "BUY", price=70, atr=10, qty=4)]}
    positions = {"positions": [position("STOP", qty=3, entry=100, value=210)]}
    plan = build_plan(final, positions, account(), SETTINGS)
    assert len(plan) == 1
    assert plan[0]["side"] == "sell" and plan[0]["qty"] == 3
    assert plan[0]["reason"] == "stop-loss hit"


def test_buy_within_budget_without_positions():
    final = {"stocks": [stock("BUY", "BUY", price=100, qty=20)]}
    plan = build_plan(final, {"positions": []}, account(), SETTINGS)
    assert len(plan) == 1
    assert plan[0]["side"] == "buy"
    assert plan[0]["qty"] == 1  # 12% of $1,000 allows one $100 share
    assert plan[0]["qty"] * plan[0]["price"] <= SETTINGS["TARGET_INVESTED"] * 1000


def test_planned_sells_increase_buy_budget():
    final = {
        "stocks": [
            stock("EXIT", "HOLD", price=80, atr=1),
            stock("BUY", "BUY", price=100, qty=10),
        ]
    }
    positions = {"positions": [position("EXIT", qty=5, entry=100, value=500)]}
    constrained = {**SETTINGS, "TARGET_INVESTED": 0.5}
    plan = build_plan(final, positions, account(cash=0), constrained)
    buys = [order for order in plan if order["side"] == "buy"]
    assert len(buys) == 1
    assert buys[0]["qty"] == 1  # Position cap limits the $100 purchase
    assert sum(order["qty"] * order["price"] for order in plan
               if order["side"] == "sell") == 400


def test_stock_error_is_ignored():
    final = {"stocks": [{"symbol": "BAD", "error": "data unavailable"}]}
    plan = build_plan(final, {"positions": []}, account(), SETTINGS)
    assert plan == []


def test_symbol_never_appears_twice():
    final = {
        "stocks": [
            stock("HELD", "BUY", price=120, atr=10, qty=10),
            stock("OTHER", "BUY", price=100, qty=3),
        ]
    }
    positions = {"positions": [position("HELD", qty=4, entry=80, value=480)]}
    plan = build_plan(final, positions, account(), SETTINGS)
    symbols = [order["symbol"] for order in plan]
    assert len(symbols) == len(set(symbols))
    assert [order for order in plan if order["symbol"] == "HELD"][0]["side"] == "sell"


TESTS = [
    test_take_profit_wins_over_add,
    test_stop_loss_sell,
    test_buy_within_budget_without_positions,
    test_planned_sells_increase_buy_budget,
    test_stock_error_is_ignored,
    test_symbol_never_appears_twice,
]


def test_profit_taken_only_once_then_breakeven():
    final = {"stocks": [stock("WIN", "HOLD", price=130, atr=5, qty=0)]}
    positions = {"positions": [position("WIN", qty=4, entry=100, value=520)]}
    first = build_plan(final, positions, account(), SETTINGS)
    assert first[0]["reason"] == "take half profit" and first[0]["qty"] == 2
    again = build_plan(final, positions, account(), SETTINGS, profit_taken={"WIN": 100})
    assert again == []
    final = {"stocks": [stock("WIN", "HOLD", price=99, atr=5, qty=0)]}
    back = build_plan(final, positions, account(), SETTINGS, profit_taken={"WIN": 100})
    assert back[0]["reason"] == "breakeven stop after taking profit" and back[0]["qty"] == 4


def test_pending_buy_not_bought_twice():
    final = {"stocks": [stock("NEW", "BUY", price=10, atr=1, qty=5)]}
    plan = build_plan(final, {"positions": []}, account(equity=100000, cash=100000),
                      SETTINGS, pending_buys={"NEW"})
    assert plan == []


TESTS += [test_profit_taken_only_once_then_breakeven, test_pending_buy_not_bought_twice]


if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    if failures:
        raise SystemExit(1)