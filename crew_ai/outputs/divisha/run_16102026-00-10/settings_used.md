# Settings Used

Timestamp: 2026-10-16T00:10:55
Flags: --auto
Model: groq/openai/gpt-oss-120b
Stocks: MU, XYZ, HOOD, SMCI, AVGO, FCX, AMZN, GOOGL, GE, V
Regime: BEAR
Buy threshold: 70

## Settings
MCP_URL: http://localhost:8001/mcp
LLM_MODEL: groq/openai/gpt-oss-120b
TARGET_INVESTED: 0.95
STOP_ATR: 2.5
TAKE_PROFIT_ATR: 3.0
LIMIT_BUFFER: 0.003
MAX_RPM: 20
NEWS_HEADLINES: 3
EXTENDED_HOURS: False
LLM_RETRIES: 1
CREW_VERBOSE: True
LLM_BACKUP_MODELS: groq/openai/gpt-oss-20b,groq/qwen/qwen3.8-27b,groq/llama-3.3-70b-versatile
MAX_TRADE_RISK_FRACTION: 0.015
MAX_POSITION_FRACTION: 0.2
MAX_HIGH_BETA_POSITION_FRACTION: 0.15
PROTECTIVE_STOP_FRACTION: 0.06
