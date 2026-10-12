import asyncio
from fastmcp import Client

STOCKS = ["AAPL", "NVDA", "TSLA", "JPM"]

async def main():
    async with Client("http://localhost:8001/mcp") as c:
        tools = await c.list_tools()
        print("TOOLS:", [t.name for t in tools], "\n")
        print("ACCOUNT:", (await c.call_tool("get_account_info", {})).data, "\n")
        print("BARS:", (await c.call_tool("get_bars_multi", {"symbols": STOCKS})).data, "\n")
        print("NEWS:", (await c.call_tool("get_news", {"symbol": "AAPL", "limit": 3})).data, "\n")
        rams = (await c.call_tool("compute_rams", {"symbols": STOCKS})).data
        print("REGIME:", rams["regime"], "| BUY THRESHOLD:", rams["buy_threshold"])
        for s in rams["stocks"]:
            print(s.get("symbol"), "| RAMS:", s.get("rams"), "| SIGNAL:", s.get("signal"),
                  "| QTY:", s.get("suggested_qty"), "| ERROR:", s.get("error", "-"))
        print("\nPOSITIONS:", (await c.call_tool("get_all_positions", {})).data)

asyncio.run(main())
