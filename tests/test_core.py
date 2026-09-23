import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import indicators as ta
from src.backtest import _mtf_enabled
from src.config import IndicatorSettings
from src.data import price_change_pct_24h
from src.patterns import PatternState
from src.validation import confidence_report
from src.quality import _barrier_label, _laplace_probability
from src.execution import estimate_market_fill, execution_guard
from src.scorecard import build_scorecard
from src.order_flow import analyze_liquidity
from src.signals import evaluate


class CoreTests(unittest.TestCase):
    def test_price_change_uses_candle_at_or_before_24h_target(self):
        now = pd.Timestamp("2026-01-03 12:00:00", tz="UTC")
        index = pd.DatetimeIndex([
            "2026-01-01 12:00:00+00:00",
            "2026-01-02 12:00:00+00:00",
            "2026-01-03 12:00:00+00:00",
        ])
        df = pd.DataFrame({"close": [90.0, 100.0, 110.0]}, index=index)
        self.assertAlmostEqual(price_change_pct_24h(df, now), 10.0)

    def test_mtf_auto_uses_signal_timeframe(self):
        self.assertTrue(_mtf_enabled("4h", None))
        self.assertFalse(_mtf_enabled("1d", None))
        self.assertTrue(_mtf_enabled("4h", True))
        self.assertFalse(_mtf_enabled("4h", False))

    def test_choppiness_has_warmup_then_values(self):
        n = 80
        index = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = pd.Series(np.linspace(100, 140, n), index=index)
        df = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }, index=index)
        chop = ta.choppiness(df, 14)
        self.assertTrue(chop.iloc[:13].isna().all())
        self.assertTrue(np.isfinite(chop.iloc[-1]))


    def test_confidence_report_returns_horizon_metrics(self):
        n = 260
        index = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(4)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))), index=index)
        df = pd.DataFrame({
            "open": close * 0.999, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1000.0,
        }, index=index)
        report = confidence_report(df, IndicatorSettings(), timeframe="4h",
                                   use_mtf=False, horizons=(1, 3))
        self.assertIn("1", report["horizons"])
        self.assertIn("3", report["horizons"])
        self.assertIn("win_rate_pct", report["horizons"]["1"])

    def test_quality_barrier_is_conservative_when_both_levels_hit(self):
        index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 110.0, 100.0],
            "low": [100.0, 90.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": 1000.0,
        }, index=index)
        self.assertEqual(_barrier_label(df, 0, 1, 108.0, 92.0, 1), 0)
        self.assertEqual(_barrier_label(df, 0, 1, 108.0, 92.0, 2), 0)

    def test_quality_probability_is_laplace_smoothed(self):
        self.assertAlmostEqual(_laplace_probability(9, 10), 10 / 12)
        self.assertAlmostEqual(_laplace_probability(0, 0), 0.5)

    def test_execution_fill_requires_available_depth(self):
        fill = estimate_market_fill([(100.0, 2.0), (101.0, 3.0)], 4.0)
        self.assertTrue(fill["filled"])
        self.assertAlmostEqual(fill["average_price"], 100.5)
        self.assertFalse(estimate_market_fill([(100.0, 2.0)], 3.0)["filled"])

    def test_execution_guard_blocks_stale_signal(self):
        now = pd.Timestamp("2026-01-03 12:05:00", tz="UTC")
        result = execution_guard(
            "BTC/USDT", "LONG", 100.0, 104.0, 98.0,
            {"action": "TRADE", "probability": 0.7},
            pd.Timestamp("2026-01-03 10:00:00", tz="UTC"), "1h",
            book={"source": "test", "bids": [(99.9, 100.0)],
                  "asks": [(100.1, 100.0)]},
            account_equity=1000.0, max_delay_minutes=10)
        self.assertEqual(result["action"], "WAIT")
        self.assertIn("fresh execution window", " ".join(result["reasons"]))

    def test_execution_guard_accepts_clean_mock_book(self):
        now = pd.Timestamp.now(tz="UTC")
        timestamp = now - pd.Timedelta(hours=1) - pd.Timedelta(minutes=5)
        result = execution_guard(
            "BTC/USDT", "LONG", 100.0, 104.0, 98.0,
            {"action": "TRADE", "probability": 0.7}, timestamp, "1h",
            book={"source": "test", "bids": [(99.99, 100.0)],
                  "asks": [(100.01, 100.0)]},
            account_equity=1000.0, max_delay_minutes=10)
        self.assertEqual(result["action"], "EXECUTION READY")
        self.assertGreater(result["quantity"], 0)
        self.assertLess(result["spread_bps"], 12.0)

    def test_scorecard_excludes_open_records_and_calculates_drawdown(self):
        trades = [
            {"id": "1", "status": "CLOSED", "coin": "BTC/USDT",
             "timeframe": "1h", "direction": "LONG", "entry": 100,
             "stop": 98, "pnl_pct": 2.0, "quality_grade": "A",
             "regime": "TRENDING", "closed_at": "2026-01-01"},
            {"id": "2", "status": "CLOSED", "coin": "BTC/USDT",
             "timeframe": "1h", "direction": "LONG", "entry": 100,
             "stop": 98, "pnl_pct": -1.0, "quality_grade": "B",
             "regime": "CHOPPY", "closed_at": "2026-01-02"},
            {"id": "3", "status": "OPEN", "coin": "ETH/USDT",
             "timeframe": "1h", "direction": "LONG", "entry": 100,
             "stop": 98, "pnl_pct": None},
        ]
        result = build_scorecard(trades)
        self.assertEqual(result["summary"]["trades"], 2)
        self.assertEqual(result["summary"]["wins"], 1)
        self.assertEqual(result["summary"]["losses"], 1)
        self.assertEqual(len(result["by_coin"]), 1)
        self.assertLess(result["summary"]["max_drawdown_pct"], 0.0)

    def test_liquidity_scan_requires_persistence_and_absorption(self):
        book = {
            "source": "test",
            "bids": [(100.0, 100.0), (99.9, 10.0), (99.8, 9.0), (99.7, 11.0)],
            "asks": [(100.1, 10.0), (100.2, 9.0), (100.3, 11.0), (100.4, 10.0)],
        }
        first = analyze_liquidity(
            book,
            trades=[{"price": 100.0, "quantity": 20.0, "buyer_maker": True}],
            now=1000.0,
        )
        self.assertEqual(first["action"], "WATCH")
        self.assertEqual(first["wall"]["side"], "BID")
        self.assertEqual(first["wall_age_seconds"], 0.0)

        second = analyze_liquidity(
            book,
            trades=[{"price": 100.0, "quantity": 20.0, "buyer_maker": True}],
            previous=first["tracking"],
            now=1030.0,
        )
        self.assertEqual(second["action"], "CONFIRMED BID WALL")
        self.assertGreaterEqual(second["wall_age_seconds"], 30.0)
        self.assertGreater(second["flow"]["near_wall_sell_notional"], 0.0)

    def test_liquidity_scan_flags_disappeared_wall(self):
        previous = {
            "side": "BID", "price": 100.0, "notional": 10000.0,
            "tick_size": 0.1, "first_seen": 1000.0,
        }
        book = {
            "source": "test",
            "bids": [(99.9, 10.0), (99.8, 9.0)],
            "asks": [(100.1, 10.0), (100.2, 9.0)],
        }
        result = analyze_liquidity(book, previous=previous, now=1030.0)
        self.assertEqual(result["action"], "NO WALL")
        self.assertEqual(result["cancel_risk"], "HIGH")

    def test_enabled_bearish_pattern_vetoes_buy(self):
        n = 300
        index = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        close = pd.Series(np.linspace(100, 150, n), index=index)
        df = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
        }, index=index)
        cfg = IndicatorSettings()
        ind = ta.compute_all(df, cfg)
        bullish_votes = {
            "rsi": 1, "macd": 1, "bb": 1, "stoch": 0, "ema": 1,
            "supertrend": 1, "adx": 0, "atr": 0, "obv": 0, "mfi": 0,
            "cci": 0, "roc": 0,
        }
        with patch("src.signals.row_votes", return_value=bullish_votes):
            sig = evaluate(
                df, ind, cfg, symbol="TEST/USDT", timeframe="4h",
                patterns=PatternState(bearish=True, names=["Shooting star"]),
                use_patterns=True,
            )
        self.assertEqual(sig.final, "NEUTRAL")
        self.assertTrue(sig.pattern_vetoed)


if __name__ == "__main__":
    unittest.main()
