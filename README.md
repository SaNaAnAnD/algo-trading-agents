# algo-trading-agents
Agentic paper trading on Alpaca (FORE project): a CrewAI crew of five agents
(Market Data, News Sentiment, Technical Analyst, Risk Manager, Trader) with a
local MCP server, running autonomously every 45 minutes during US market hours.
Design and results: `docs/EXPERIMENT_LOG.md`. Records: `crew_ai/outputs/<name>/`.

## Set up on Windows (once per person, about 20 minutes)
Have ready: your Alpaca **paper** API key and secret (app.alpaca.markets > Paper > API Keys >
Generate), a free Groq API key (console.groq.com > API Keys), your own stock list.

1. Press the Windows key, type `PowerShell`, open it. Paste with right-click, Enter after each line:
   ```
   winget install --id Microsoft.VisualStudioCode -e --source winget
   winget install --id Git.Git -e --source winget
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   Close PowerShell and open it again.
2. Download the project and install the libraries (5-10 minutes):
   ```
   cd $HOME
   git clone https://github.com/SaNaAnAnD/algo-trading-agents.git
   cd algo-trading-agents
   uv venv --python 3.12
   uv pip install -r requirements.txt
   ```
3. Keys: `copy .env.example .env` then `notepad .env`. Fill in after each `=` (no spaces, no
   quotes): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `GROQ_API_KEY`, `TRADER_NAME` (your first
   name, e.g. `sana`), `HIGH_BETA_SYMBOLS` (e.g. `TSLA,AMD`). Save and close.
4. Stocks: `copy crew_ai\clients\my_stocks.example.txt crew_ai\clients\my_stocks.txt` then
   `notepad crew_ai\clients\my_stocks.txt`: your own tickers, one per line. Save and close.
5. Git identity: `git config user.name "Your Name"` and `git config user.email "your-github-email"`.
6. `code .` In VS Code: **Yes, I trust the authors**; install the Python extension if asked;
   `Ctrl+Shift+P` > **Python: Select Interpreter** > the one with **.venv**.
7. Tests (VS Code: Terminal > New Terminal):
   ```
   uv run python servers\test_order_safety.py
   cd crew_ai\clients
   uv run python ..\tests\test_trading_safety.py
   uv run python ..\tests\test_build_plan.py
   cd ..\..
   ```
8. Practice run (no orders): `Ctrl+Shift+D` > **4. Practice run (no orders)** > green play button.
9. Windows Settings > System > Power: **Sleep: Never** when plugged in. Keep the laptop plugged in.

## Set up on Linux (conda)
1. `git clone https://github.com/SaNaAnAnD/algo-trading-agents.git` then `cd algo-trading-agents`
2. `conda create -n trading-agents python=3.12 -y`, `conda activate trading-agents`,
   `pip install -r requirements.txt`
3. `cp .env.example .env` and fill it in (same fields as above); `cp crew_ai/clients/my_stocks.example.txt
   crew_ai/clients/my_stocks.txt` and put your tickers.
4. `code .`, **Python: Select Interpreter** > trading-agents, run the tests, then **4. Practice run**.

## Every trading evening (from 7:00 PM IST)
Open the project in VS Code, in its terminal run `git pull`, then Run and Debug
(`Ctrl+Shift+D`) > **1. Trading SESSION** > green play button. Keep the computer awake and plugged in. It waits for
the open, runs every 45 minutes, stops after the close (1:30 AM IST), writes
`daily_summary_<date>.md` and pushes your records to GitHub.
