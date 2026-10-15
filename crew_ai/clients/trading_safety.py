"""Deterministic, network-free trading safety checks."""
import math


def protective_stop(entry_price, stop_fraction=0.06):
    """Return a stop price 6% below the entry price."""
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("Entry price must be positive and finite")
    if not 0 < stop_fraction < 1:
        raise ValueError("Stop fraction must be between 0 and 1")
    return round(entry_price * (1 - stop_fraction), 2)


def safe_buy_quantity(
    price,
    stop_price,
    equity,
    current_position_value=0,
    high_beta=False,
    max_risk_fraction=0.015,
    max_position_fraction=0.12,
    high_beta_fraction=0.08,
):
    """Cap shares by stop-loss risk and maximum position exposure."""
    values = (price, stop_price, equity, current_position_value)
    if not all(math.isfinite(v) for v in values):
        return 0
    if price <= 0 or stop_price <= 0 or equity <= 0:
        return 0
    if stop_price >= price or current_position_value < 0:
        return 0

    risk_budget = equity * max_risk_fraction
    risk_per_share = price - stop_price
    risk_qty = math.floor(risk_budget / risk_per_share)

    cap_fraction = high_beta_fraction if high_beta else max_position_fraction
    exposure_remaining = max(
        0, equity * cap_fraction - current_position_value
    )
    exposure_qty = math.floor(exposure_remaining / price)

    return max(0, min(risk_qty, exposure_qty))


def validate_sell_quantity(quantity, held_quantity):
    """Prevent selling more shares than the current position holds."""
    if quantity <= 0 or held_quantity < 0:
        return False
    return quantity <= held_quantity
