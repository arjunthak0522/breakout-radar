from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import scanner

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
DELAY = float(os.getenv("BACKTEST_DELAY", "0.18"))
START = pd.Timestamp(os.getenv("BACKTEST_START", "2023-01-01"), tz="UTC")
HOLD_DAYS = [5, 10, 20, 40]
MIN_SETUP = 55
COST_BPS = float(os.getenv("BACKTEST_COST_BPS", "10"))


def fetch_daily(symbol: str) -> pd.DataFrame:
    df, _ = scanner.fetch_chart(symbol, "5y", "1d")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def atr_at(df: pd.DataFrame, i: int) -> float:
    return scanner.atr14(df.iloc[: i + 1].copy())


def rs_at(st: pd.DataFrame, qqq_by_date: pd.Series, i: int) -> float:
    if i < 63:
        return np.nan
    d0, d1 = st.timestamp.iloc[i - 63], st.timestamp.iloc[i]
    if d0 not in qqq_by_date.index or d1 not in qqq_by_date.index:
        return np.nan
    q0, q1 = qqq_by_date.loc[d0], qqq_by_date.loc[d1]
    return float(((st.close.iloc[i] / st.close.iloc[i - 63] - 1) - (q1 / q0 - 1)) * 100)


def feature_at(symbol: str, df: pd.DataFrame, i: int, qqq_by_date: pd.Series):
    # Everything below uses information available through close of day i only.
    hist = df.iloc[: i + 1].copy()
    if len(hist) < 65:
        return None
    cur = float(hist.close.iloc[-1])
    ma20 = float(hist.close.tail(20).mean())
    ma50 = float(hist.close.tail(50).mean())
    trend = (6 if cur > ma20 else 0) + (7 if cur > ma50 else 0) + (5 if ma20 > ma50 else 0) + (4 if scanner.lin_slope(hist.close.tail(20)) > 0 else 0) + (3 if scanner.lin_slope(hist.close.tail(50)) > 0 else 0)
    trigger = float(hist.high.iloc[-51:-1].max())
    p = (cur / trigger - 1) * 100
    vcp = scanner.vcp_proxy(hist, cur)
    avg10 = float(hist.volume.tail(10).mean())
    prev40 = float(hist.volume.iloc[-50:-10].mean())
    dry = avg10 / prev40 if prev40 > 0 else np.nan
    avg20 = float(hist.volume.iloc[-21:-1].mean())
    rvol = float(hist.volume.iloc[-1] / avg20) if avg20 > 0 else np.nan
    rs = rs_at(hist, qqq_by_date, len(hist) - 1)
    atr = atr_at(hist, len(hist) - 1)
    low10 = float(hist.low.tail(10).min())
    valid = [x for x in [low10, ma20, ma50] if x < trigger and x >= .8 * trigger and trigger - x >= .75 * atr]
    support = max(valid) if valid else low10
    stop = support - .25 * atr
    d60 = hist.tail(60)
    depth = float(d60.high.max() - d60.low.min())
    target = max(trigger + 1.25 * atr, trigger + .5 * depth)
    brr = (target - trigger) / (trigger - stop) if trigger > stop else np.nan
    srr = (target - support) / (support - stop) if support > stop else np.nan
    upside = (target / trigger - 1) * 100
    score = trend
    score += 18 if -5 <= p < -3 else 12 if -3 <= p < -.5 else 9 if -.5 <= p <= 1 else 15 if 1 < p <= 5 else 6 if -10 <= p < -5 else 0
    score += 16 if vcp >= .7 else 12 if vcp >= .55 else 7 if vcp >= .4 else 2
    score += 18 if np.isfinite(rs) and rs >= 10 else 10 if np.isfinite(rs) and rs >= 5 else 5 if np.isfinite(rs) and rs >= 0 else -8 if np.isfinite(rs) else 0
    score += 8 if np.isfinite(dry) and dry <= .8 else 5 if np.isfinite(dry) and dry <= 1 else 1 if np.isfinite(dry) else 0
    score += 8 if np.isfinite(rvol) and rvol >= 1.5 else 5 if np.isfinite(rvol) and rvol >= 1.15 else 2 if np.isfinite(rvol) else 0
    score += 8 if upside >= 15 else 5 if upside >= 10 else 2 if upside >= 5 else 0
    score = int(np.clip(round(score), 0, 100))
    return {"setup": score, "support": support, "trigger": trigger, "stop": stop, "target": target, "support_rr": srr, "breakout_rr": brr, "rs": rs, "vcp": vcp, "target_upside": upside, "close": cur}


def next_open_entry(df: pd.DataFrame, i: int, f: dict):
    # Signal is generated at close i. Entry is earliest possible next session open.
    j = i + 1
    if j >= len(df):
        return None
    op = float(df.open.iloc[j]) if np.isfinite(df.open.iloc[j]) else float(df.close.iloc[j])
    ps = (f["close"] / f["support"] - 1) * 100
    if f["setup"] < MIN_SETUP:
        return None
    if f["close"] >= f["trigger"] and (f["close"] / f["trigger"] - 1) * 100 <= 5:
        return "BUY BREAKOUT", j, op
    if f["close"] >= f["support"] and 0 <= ps <= 1:
        return "BUY SUPPORT", j, op
    return None


def qqq_forward(qqq_by_date: pd.Series, entry_date, exit_date):
    try:
        return float(qqq_by_date.loc[exit_date] / qqq_by_date.loc[entry_date] - 1)
    except Exception:
        return np.nan


def main():
    universe = scanner.load_universe()
    qqq = fetch_daily("QQQ")
    qqq_by_date = qqq.set_index("timestamp").close
    trades = []
    errors = {}
    for n, sym in enumerate(universe, 1):
        try:
            df = fetch_daily(sym)
            for i in range(64, len(df) - max(HOLD_DAYS) - 1):
                if df.timestamp.iloc[i] < START:
                    continue
                f = feature_at(sym, df, i, qqq_by_date)
                if not f:
                    continue
                ent = next_open_entry(df, i, f)
                if not ent:
                    continue
                action, j, entry = ent
                row = {"Stock": sym, "Signal Date": df.timestamp.iloc[i], "Entry Date": df.timestamp.iloc[j], "Action": action, "Setup": f["setup"], "Entry": entry, "Support": f["support"], "Trigger": f["trigger"], "Stop": f["stop"], "Target": f["target"], "RS vs QQQ": f["rs"], "VCP": f["vcp"], "Target Upside %": f["target_upside"], "Support R/R": f["support_rr"], "Breakout R/R": f["breakout_rr"]}
                for h in HOLD_DAYS:
                    k = j + h
                    exit_px = float(df.close.iloc[k])
                    gross = exit_px / entry - 1
                    net = gross - 2 * COST_BPS / 10000
                    qret = qqq_forward(qqq_by_date, df.timestamp.iloc[j], df.timestamp.iloc[k])
                    row[f"Return {h}d"] = net
                    row[f"QQQ {h}d"] = qret
                    row[f"Alpha {h}d"] = net - qret if np.isfinite(qret) else np.nan
                    row[f"Win {h}d"] = net > 0
                trades.append(row)
        except Exception as e:
            errors[sym] = str(e)
        print(f"{n}/{len(universe)} {sym} trades={len(trades)}")
        time.sleep(DELAY)
    t = pd.DataFrame(trades)
    t.to_csv(OUT / "backtest_trades.csv", index=False)
    summaries = []
    for action in ["ALL", "BUY SUPPORT", "BUY BREAKOUT"]:
        d = t if action == "ALL" else t[t.Action == action]
        if d.empty:
            continue
        for h in HOLD_DAYS:
            summaries.append({"Action": action, "Horizon": h, "Trades": len(d), "Win Rate %": 100 * d[f"Win {h}d"].mean(), "Avg Return %": 100 * d[f"Return {h}d"].mean(), "Median Return %": 100 * d[f"Return {h}d"].median(), "Avg Alpha vs QQQ %": 100 * d[f"Alpha {h}d"].mean(), "Median Alpha vs QQQ %": 100 * d[f"Alpha {h}d"].median()})
    s = pd.DataFrame(summaries)
    s.to_csv(OUT / "backtest_summary.csv", index=False)
    manifest = {"finished_utc": datetime.now(timezone.utc).isoformat(), "universe_size": len(universe), "start": str(START.date()), "hold_days": HOLD_DAYS, "cost_bps_each_side": COST_BPS, "trades": len(t), "symbols_with_errors": errors, "lookahead_control": "signals use data through close t; entries occur at next session open t+1"}
    (OUT / "backtest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(s.to_string(index=False))
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
