"""Network-free tests for deterministic trading safety checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients"))
from trading_safety import (
    protective_stop,
    safe_buy_quantity,
    validate_sell_quantity,
)


def test_six_percent_protective_stop():
    assert protective_stop(100) == 94


def test_normal_position_cap():
    qty = safe_buy_quantity(100, 90, 10000)
    assert qty == 12


def test_high_beta_position_cap():
    qty = safe_buy_quantity(100, 90, 10000, high_beta=True)
    assert qty == 8


def test_risk_cap_limits_quantity():
    qty = safe_buy_quantity(100, 50, 10000)
    assert qty == 3


def test_invalid_prices_rejected():
    assert safe_buy_quantity(100, 100, 10000) == 0
    assert safe_buy_quantity(-1, 90, 10000) == 0


def test_sell_cannot_exceed_position():
    assert validate_sell_quantity(5, 5)
    assert not validate_sell_quantity(6, 5)
    assert not validate_sell_quantity(0, 5)


if __name__ == "__main__":
    tests = [value for name, value in globals().copy().items()
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} safety tests passed")
