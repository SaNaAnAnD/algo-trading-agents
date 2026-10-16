# Risk Manager Agent

## Proposed orders
1. SELL 354 FCX @ limit 69.61 (RAMS 36.4) - RAMS 36.4 below sell level
2. BUY 42 GE @ limit 316.39 (RAMS 68.9, RSI 60.5, news +0.60) - healthy stock (RAMS 68.9, trend 50): target 27% of equity

## Review
- **Order 1 – SELL 354 FCX @ 69.61** – **APPROVED**  
  *Reason:* No violation of the reject criteria (news score for FCX is +0.3, not a very bad news signal; RSI is not relevant for a sell; sector exposure will actually decrease, staying well below the 40 % limit).

- **Order 2 – BUY 42 GE @ 316.39** – **REJECTED**  
  *Reason:* Executing the trade would leave cash at roughly $6,900, which is far below the required 20 % of equity cash buffer (≈ $19,900). All other criteria (RAMS 68.9, RSI 60.5, news +0.60, sector exposure < 40 %) are satisfied, but the cash‑reserve rule is breached.