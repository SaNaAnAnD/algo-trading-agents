# config.py - all settings for the RAMS indicator in one place

# How many calendar days of 1-hour bars to download
HISTORY_DAYS = 480   # daily bars: the 200-day average needs about 300 trading days

# Pillar weights (must add up to 1.0)
W_TREND = 0.30
W_MOMENTUM = 0.30
W_VOLUME = 0.15
W_BREAKOUT = 0.10
W_NEWS = 0.15

# Buy/sell thresholds
BUY_THRESHOLD_BULL = 62   # buy level when SPY is above its EMA50
BUY_THRESHOLD_BEAR = 70   # stricter buy level when SPY is below its EMA50
ADD_THRESHOLD = 80        # add more to a stock you already hold
SELL_THRESHOLD = 30       # sell below this score (backtest: 30 beat 40; stops buying/selling at 50/40 churned)

# Risk settings
ATR_STOP_MULT = 2.5       # stop-loss = price - 2.5 x ATR
RISK_PER_TRADE = 0.015    # risk 1.5% of equity per trade
MAX_WEIGHT = 0.20         # max 20% of account in one normal stock (contest: max % gain)
MAX_WEIGHT_HIGH_BETA_BULL = 0.15
MAX_WEIGHT_HIGH_BETA_BEAR = 0.15
ZSCORE_PENALTY_LEVEL = 2.5
ZSCORE_PENALTY_POINTS = 10

# High-beta stocks in this portfolio (same list as HIGH_BETA_SYMBOLS in .env)
HIGH_BETA = ["MU", "XYZ", "HOOD", "SMCI"]