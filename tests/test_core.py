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
