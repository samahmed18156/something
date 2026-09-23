#!/usr/bin/env python3
"""
Crypto Signal System — CLI runner.

Fetches closed candles from the Binance public API (no keys), computes the
12 core indicators plus live derivatives sentiment, applies the regime
(Choppiness) and multi-timeframe filters, and prints the per-indicator votes
plus the final consensus signal for each coin.

Usage:
    python main.py                          # all default coins, 4h candles
    python main.py BTC/USDT ETH/USDT        # specific coins
    python main.py --timeframe 1h           # different timeframe
    python main.py --no-mtf                 # disable the higher-TF filter
    python main.py --no-deriv               # disable funding/OI sentiment
    python main.py --watch 300              # live mode: refresh every 5 min
    python main.py --json                   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CANDLE_LIMIT, DEFAULT_COINS, TIMEFRAMES
from src.signals import full_signal


def fmt_price(x: float) -> str:
    """Adaptive price formatting so PEPE (0.0000048) doesn't display as 0.00."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:,.2f}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def analyze(symbol: str, timeframe: str, use_mtf: bool | None, use_deriv: bool):
    return full_signal(symbol, timeframe, CANDLE_LIMIT,
                       use_mtf=use_mtf, use_deriv=use_deriv)


def print_report(sig) -> None:
    print("═" * 92)
    print(f" {sig.symbol}  ·  {sig.timeframe}  ·  candle {sig.timestamp:%Y-%m-%d %H:%M} UTC"
          f"  ·  close {fmt_price(sig.close)}")
    mtf_label = {1: "UP", -1: "DOWN", 0: "n/a"}[sig.mtf_trend]
    blocked = (f"  ⚠️  SIGNAL SUPPRESSED — fights the {sig.mtf_timeframe} trend"
               if sig.mtf_blocked else "")
    print(f" REGIME   {sig.regime} (CHOP {sig.chop:.1f})  ·  "
          f"MTF {sig.mtf_timeframe or '—'} trend: {mtf_label}{blocked}")
    p = sig.patterns
    if p and p.names:
        pat_note = ", ".join(p.names)
        if p.bearish:
            pat_note += "  ⛔ LONG ENTRY VETOED (bearish pattern on last closed candle)"
        elif p.bullish:
            pat_note += "  ✅ long entry confirmed (bullish pattern)"
        else:
            pat_note += "  (indecision flag)"
    else:
        pat_note = "none on last closed candle"
    print(f" PATTERN  {pat_note}")
    print("═" * 92)
    print(f" {'INDICATOR':<22} {'LATEST':<46} {'VOTE':<7} {'WT':>5}  REASON")
    print("─" * 92)
    for v in sig.votes:
        vote = {1: "🟢 +1", 0: "⚪  0", -1: "🔴 -1"}[v.vote]
        print(f" {v.name:<22} {v.value:<46} {vote:<7} {v.weight:>5.2f}  {v.reason}")
    print("─" * 92)
    print(f" SCORE {sig.score:+.3f}   |   🟢 {sig.bullish}   🔴 {sig.bearish}   ⚪ {sig.neutral}"
          f"   |   agreement {sig.agreement}%")
    print(f" SIGNAL:  {sig.final}")
    print(f" LONG   → entry {fmt_price(sig.close)}  stop {fmt_price(sig.stop_long)}"
          f" ({(sig.stop_long / sig.close - 1) * 100:+.2f}%)  target {fmt_price(sig.target_long)}"
          f" ({(sig.target_long / sig.close - 1) * 100:+.2f}%)")
    print(f" SHORT  → entry {fmt_price(sig.close)}  stop {fmt_price(sig.stop_short)}"
          f" ({(sig.stop_short / sig.close - 1) * 100:+.2f}%)  target {fmt_price(sig.target_short)}"
          f" ({(sig.target_short / sig.close - 1) * 100:+.2f}%)")
    print()


def sig_to_dict(sig) -> dict:
    return {
        "symbol": sig.symbol,
        "timeframe": sig.timeframe,
        "timestamp": sig.timestamp.isoformat(),
        "close": sig.close,
        "score": sig.score,
        "final": sig.final,
        "bullish": sig.bullish,
        "bearish": sig.bearish,
        "neutral": sig.neutral,
        "agreement_pct": sig.agreement,
        "regime": sig.regime,
        "chop": sig.chop,
        "mtf_timeframe": sig.mtf_timeframe,
        "mtf_trend": sig.mtf_trend,
        "mtf_blocked": sig.mtf_blocked,
        "derivatives": sig.deriv,
        "patterns": (vars(sig.patterns) if sig.patterns else None),
        "stop_long": sig.stop_long,
        "target_long": sig.target_long,
        "stop_short": sig.stop_short,
        "target_short": sig.target_short,
        "atr": sig.atr,
        "atr_pct": sig.atr_pct,
        "votes": [vars(v) for v in sig.votes],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="10+ indicator crypto consensus signal system with "
                    "regime & multi-timeframe filters")
    ap.add_argument("coins", nargs="*", default=DEFAULT_COINS,
                    help="ccxt-format symbols, e.g. BTC/USDT")
    ap.add_argument("--timeframe", default="4h", choices=TIMEFRAMES)
    ap.add_argument("--mtf", action="store_true",
                    help="force the higher-TF filter ON "
                         "(default: per-timeframe auto — on for 15m/1h/4h, off for 1d)")
    ap.add_argument("--no-mtf", action="store_true",
                    help="force the higher-TF filter OFF")
    ap.add_argument("--no-deriv", action="store_true",
                    help="disable the funding/OI derivatives sentiment")
    ap.add_argument("--watch", type=int, default=0, metavar="SECONDS",
                    help="live mode: re-run every N seconds (Ctrl+C to stop)")
    ap.add_argument("--json", action="store_true",
                    help="print JSON instead of the table")
    args = ap.parse_args()

    use_mtf: bool | None = (False if args.no_mtf
                            else True if args.mtf
                            else None)

    try:
        while True:
            if args.watch and sys.stdout.isatty():
                os.system("cls" if os.name == "nt" else "clear")

            if args.json:
                out = []
                for s in args.coins:
                    try:
                        out.append(sig_to_dict(
                            analyze(s, args.timeframe, use_mtf, not args.no_deriv)))
                    except Exception as e:  # keep going if one coin fails
                        out.append({"symbol": s, "error": str(e)})
                print(json.dumps(out, indent=2, default=str))
            else:
                for s in args.coins:
                    try:
                        print_report(analyze(s, args.timeframe, use_mtf,
                                             not args.no_deriv))
                    except Exception as e:
                        print(f" {s}: ERROR {e}\n")

            if not args.watch:
                break
            print(f" Next refresh in {args.watch}s (Ctrl+C to stop)…", flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
