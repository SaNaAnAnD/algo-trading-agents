# algo-trading-agents
Agentic paper trading on Alpaca (FORE project): a CrewAI crew of five agents
(Market Data, News Sentiment, Technical Analyst, Risk Manager, Trader) with a
local MCP server, running autonomously every 45 minutes during US market hours.
Design and results: `docs/EXPERIMENT_LOG.md`. Records: `crew_ai/outputs/<name>/`.

## Set up (once per person)
1. `git clone https://github.com/SaNaAnAnD/algo-trading-agents.git` then `cd algo-trading-agents`
2. Python environment (pick one), then `pip install -r requirements.txt`
   - conda: `conda create -n trading-agents python=3.12 -y` and `conda activate trading-agents`
   - venv: `python3 -m venv .venv` and `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. Keys: `cp .env.example .env` (Windows: `copy .env.example .env`), open `.env`, fill in your
   Alpaca paper keys, your Groq key, `TRADER_NAME` (your first name) and `HIGH_BETA_SYMBOLS`.
4. Stocks: `cp crew_ai/clients/my_stocks.example.txt crew_ai/clients/my_stocks.txt` and put
   your own tickers in it, one per line.
5. Open the folder in VS Code (`code .`), `Ctrl+Shift+P` > **Python: Select Interpreter** >
   choose your environment.
6. Check: `python servers/test_order_safety.py`, then in `crew_ai/clients`:
   `python ../tests/test_trading_safety.py` and `python ../tests/test_build_plan.py`.
7. Practice run (no orders): Run and Debug (`Ctrl+Shift+D`) > **4. Practice run**.

## Every trading evening (from 7:00 PM IST)
`git pull`, then Run and Debug > **1. Trading SESSION** > green play button. It waits for
the open, runs every 45 minutes, stops after the close (1:30 AM IST), writes
`daily_summary_<date>.md` and pushes your records to GitHub.
