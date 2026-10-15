"""Offline tests for server-side order validation."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Prevent Alpaca credentials or network access during module import.
with patch("alpaca.trading.client.TradingClient", return_value=MagicMock()), \
     patch("alpaca.data.historical.StockHistoricalDataClient", return_value=MagicMock()), \
     patch("alpaca.data.historical.news.NewsClient", return_value=MagicMock()):
    import alpaca_trading_mcpServer as server


class ServerOrderSafetyTests(unittest.TestCase):
    def setUp(self):
        # Enable order paths only inside these mocked, offline tests.
        self.trading_enabled_patch = patch.dict(
            "os.environ", {"TRADING_ENABLED": "true"}
        )
        self.trading_enabled_patch.start()
        self.addCleanup(self.trading_enabled_patch.stop)

        self.positions = []
        self.account = SimpleNamespace(equity="10000")
        self.position_patch = patch.object(
            server._trading_client, "get_all_positions",
            side_effect=lambda: self.positions,
        )
        self.account_patch = patch.object(
            server._trading_client, "get_account",
            side_effect=lambda: self.account,
        )
        self.position_patch.start()
        self.account_patch.start()
        self.addCleanup(self.position_patch.stop)
        self.addCleanup(self.account_patch.stop)

    def test_invalid_side_rejected(self):
        with self.assertRaisesRegex(ValueError, "side"):
            server._validate_order_request("AAPL", "hold", 1, 100)

    def test_zero_quantity_rejected(self):
        with self.assertRaisesRegex(ValueError, "Quantity"):
            server._validate_order_request("AAPL", "buy", 0, 100)

    def test_fractional_quantity_rejected(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            server._validate_order_request("AAPL", "buy", 1.5, 100)

    def test_buy_without_price_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference price"):
            server._validate_order_request("AAPL", "buy", 1)

    def test_buy_over_position_cap_rejected(self):
        # $10,000 equity; normal cap is 12% = $1,200.
        with self.assertRaisesRegex(ValueError, "at most"):
            server._validate_order_request("AAPL", "buy", 13, 100)

    def test_safe_buy_allowed(self):
        symbol, qty = server._validate_order_request("AAPL", "buy", 10, 100)
        self.assertEqual((symbol, qty), ("AAPL", 10))

    def test_sell_above_holding_rejected(self):
        self.positions = [
            SimpleNamespace(symbol="AAPL", qty="5", market_value="500")
        ]
        with self.assertRaisesRegex(ValueError, "only 5"):
            server._validate_order_request("AAPL", "sell", 6)

    def test_sell_within_holding_allowed(self):
        self.positions = [
            SimpleNamespace(symbol="AAPL", qty="5", market_value="500")
        ]
        self.assertEqual(
            server._validate_order_request("AAPL", "sell", 5),
            ("AAPL", 5),
        )



    def test_sell_blocked_when_symbol_has_open_order(self):
        self.positions = [
            SimpleNamespace(symbol="AAPL", qty="5", market_value="500")
        ]
        open_order = SimpleNamespace(symbol="AAPL", id="protective-stop-1")

        with patch.object(
            server._trading_client, "get_orders", return_value=[open_order]
        ), patch.object(server._trading_client, "submit_order") as submit:
            with self.assertRaisesRegex(ValueError, "open order"):
                server.sell_stock("AAPL", 1)
            submit.assert_not_called()

    def test_kill_switch_blocks_limit_order(self):
        with patch.dict("os.environ", {"TRADING_ENABLED": "false"}), \
             patch.object(server._trading_client, "submit_order") as submit:
            with self.assertRaisesRegex(ValueError, "Trading is disabled"):
                server.place_limit_order("AAPL", "buy", 1, 100)
            submit.assert_not_called()

    def test_sell_allowed_when_no_open_orders(self):
        self.positions = [
            SimpleNamespace(symbol="AAPL", qty="5", market_value="500")
        ]
        mock_order = MagicMock()
        mock_order.id = "mock-sell"
        mock_order.symbol = "AAPL"
        mock_order.side = "sell"
        mock_order.qty = "1"
        mock_order.status = "accepted"

        with patch.object(
            server._trading_client, "get_orders", return_value=[]
        ), patch.object(
            server._trading_client, "submit_order", return_value=mock_order
        ) as submit:
            server.sell_stock("AAPL", 1)
            submit.assert_called_once()

if __name__ == "__main__":
    unittest.main(verbosity=2)
