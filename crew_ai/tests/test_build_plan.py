"""Network-free tests for the paper-trading plan builder."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))
from trading_crew import build_plan, buy_veto_evidence


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



def test_buy_veto_needs_evidence():
    order = {"symbol": "XYZ", "side": "buy", "rsi": 69.9}
    assert buy_veto_evidence(order, {"XYZ": 0.0}) == ""
    assert "bad news" in buy_veto_evidence(order, {"XYZ": -0.5})
    assert "overbought" in buy_veto_evidence(dict(order, rsi=83), {"XYZ": 0.2})



CAREFUL = {**SETTINGS, "ALLOCATION": "careful", "TARGET_INVESTED": 0.80,
           "MAX_POSITION_FRACTION": 0.27, "MAX_HIGH_BETA_POSITION_FRACTION": 0.15,
           "MAX_TRADE_RISK_FRACTION": 0.017}


def test_careful_invests_only_healthy_and_keeps_cash():
    final = {"stocks": [
        stock("V", "HOLD", price=400, atr=2, rams=75.6, trend=100, news_points=65),
        stock("XYZ", "BUY", price=80, atr=1, rams=70.3, trend=50, news_points=50),
        stock("GE", "WAIT", price=200, atr=2, rams=55.1, trend=25, news_points=70),
        stock("AMZN", "WAIT", price=200, atr=2, rams=32.3, trend=25, news_points=65),
        stock("MU", "WAIT", price=100, atr=2, rams=60, trend=0, news_points=50),
        stock("BADNEWS", "WAIT", price=100, atr=2, rams=65, trend=50, news_points=20),
    ]}
    plan = build_plan(final, {"positions": []}, account(equity=100000, cash=100000), CAREFUL)
    bought = {o["symbol"]: o["qty"] * o["price"] for o in plan if o["side"] == "buy"}
    assert set(bought) == {"V", "XYZ", "GE"}           # weak, falling and bad-news names skipped
    assert bought["XYZ"] <= 15000                       # high-beta cap 15%
    assert all(v <= 27000 for v in bought.values())    # 27% cap
    assert sum(bought.values()) <= 80000                # at least 20% stays in cash


def test_careful_does_not_add_to_a_loser():
    final = {"stocks": [stock("V", "HOLD", price=380, atr=10, rams=75, trend=100, news_points=50)]}
    positions = {"positions": [position("V", qty=15, entry=390, value=5700, unrealized=-150)]}
    plan = build_plan(final, positions, account(equity=100000, cash=94000), CAREFUL)
    assert plan == []


TESTS += [test_profit_taken_only_once_then_breakeven, test_pending_buy_not_bought_twice,
          test_buy_veto_needs_evidence, test_careful_invests_only_healthy_and_keeps_cash,
          test_careful_does_not_add_to_a_loser]


from trading_crew import (update_signal_days, exposure_report, limit_price, sync_entries,
                          load_profit_taken, signal_confirmed)

CHURN = {**CAREFUL, "HEALTHY_MIN_RAMS": 60, "MIN_HOLD_DAYS": 3, "CONFIRM_BUY_DAYS": 1,
         "CONFIRM_SELL_DAYS": 2, "COOLDOWN_DAYS": 2, "TAKE_HALF_PROFIT": False}


def _weak_held(rams=25):
    final = {"stocks": [stock("V", "SELL", price=400, atr=2, rams=rams, trend=25, news_points=50)]}
    positions = {"positions": [position("V", qty=10, entry=400, value=4000, unrealized=0)]}
    return final, positions


def test_weak_signal_needs_two_days_before_selling():
    final, positions = _weak_held()
    state = {"sell_days": {"V": ["2026-10-20"]}, "entry": {"V": "2026-10-01"}}
    assert build_plan(final, positions, account(100000, 90000), CHURN, state=state, today="2026-10-20") == []
    state["sell_days"]["V"] = ["2026-10-19", "2026-10-20"]
    plan = build_plan(final, positions, account(100000, 90000), CHURN, state=state, today="2026-10-20")
    assert [o["side"] for o in plan] == ["sell"]


def test_minimum_hold_blocks_signal_sell_but_not_stop_loss():
    final, positions = _weak_held()
    state = {"sell_days": {"V": ["2026-10-19", "2026-10-20"]}, "entry": {"V": "2026-10-19"}}
    assert build_plan(final, positions, account(100000, 90000), CHURN, state=state, today="2026-10-20") == []
    crash = {"stocks": [stock("V", "SELL", price=300, atr=2, rams=25, trend=25, news_points=50)]}
    plan = build_plan(crash, positions, account(100000, 90000), CHURN, state=state, today="2026-10-20")
    assert plan and plan[0]["reason"] == "stop-loss hit"


def test_cooldown_blocks_rebuying_a_stock_just_sold():
    final = {"stocks": [stock("V", "WAIT", price=400, atr=2, rams=80, trend=100, news_points=60)]}
    state = {"buy_days": {"V": ["2026-10-20"]}, "last_sold": {"V": "2026-10-19"}}
    assert build_plan(final, {"positions": []}, account(100000, 100000), CHURN,
                      state=state, today="2026-10-20") == []
    state["last_sold"]["V"] = "2026-10-10"
    plan = build_plan(final, {"positions": []}, account(100000, 100000), CHURN,
                      state=state, today="2026-10-20")
    assert [o["side"] for o in plan] == ["buy"]


def test_buy_needs_rams_60():
    final = {"stocks": [stock("GE", "WAIT", price=200, atr=2, rams=55, trend=50, news_points=60)]}
    state = {"buy_days": {}}
    update_signal_days(state, final, CHURN, "2026-10-20")
    assert build_plan(final, {"positions": []}, account(100000, 100000), CHURN,
                      state=state, today="2026-10-20") == []


def test_signal_days_count_once_per_day_and_reset():
    final = {"stocks": [stock("V", "SELL", price=400, atr=2, rams=25, trend=25, news_points=50)]}
    state = {}
    update_signal_days(state, final, CHURN, "2026-10-20")
    update_signal_days(state, final, CHURN, "2026-10-20")      # second run, same day
    assert state["sell_days"]["V"] == ["2026-10-20"]
    update_signal_days(state, final, CHURN, "2026-10-21")
    assert signal_confirmed(state, "sell_days", "V", 2)
    ok = {"stocks": [stock("V", "HOLD", price=400, atr=2, rams=75, trend=100, news_points=50)]}
    update_signal_days(state, ok, CHURN, "2026-10-22")         # one good day resets the streak
    assert state["sell_days"]["V"] == []


def test_half_profit_flag_survives_a_changed_average_price():
    held = [{"symbol": "AMZN", "avg_entry_price": "259.35"}]
    import trading_crew
    trading_crew.PROFIT_FILE = Path(__file__).parent / "_tmp_profit.json"
    trading_crew.PROFIT_FILE.write_text('{"AMZN": 256.0, "GONE": 10.0}')
    try:
        assert load_profit_taken(held) == {"AMZN": 256.0}
    finally:
        trading_crew.PROFIT_FILE.unlink()


def test_stop_exit_uses_wide_limit_buffer():
    stop = {"symbol": "X", "side": "sell", "price": 100.0, "reason": "stop-loss hit"}
    normal = {"symbol": "X", "side": "sell", "price": 100.0, "reason": "RAMS 20 below sell level"}
    assert limit_price(stop) == 98.5 and limit_price(normal) == 99.7


def test_exposure_report_uses_real_numbers():
    positions = [{"symbol": "V", "market_value": "27000"}]
    plan = [{"symbol": "MU", "side": "buy", "qty": 10, "price": 900.0}]
    text = exposure_report(positions, plan, {"equity": "100000"})
    assert "invested 36%" in text and "Payments / fintech 27%" in text and "high-beta stocks 9%" in text


def test_new_position_gets_entry_date_old_one_is_treated_as_old():
    state = {}
    sync_entries(state, [{"symbol": "V"}], set(), CHURN, "2026-10-20")
    assert state["entry"]["V"] == "2026-10-17"      # 3 days earlier: free to sell on a real signal


TESTS += [test_weak_signal_needs_two_days_before_selling,
          test_minimum_hold_blocks_signal_sell_but_not_stop_loss,
          test_cooldown_blocks_rebuying_a_stock_just_sold, test_buy_needs_rams_60,
          test_signal_days_count_once_per_day_and_reset,
          test_half_profit_flag_survives_a_changed_average_price,
          test_stop_exit_uses_wide_limit_buffer, test_exposure_report_uses_real_numbers,
          test_new_position_gets_entry_date_old_one_is_treated_as_old]


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