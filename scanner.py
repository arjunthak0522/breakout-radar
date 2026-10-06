from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&offset=0&download=true"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "universe.txt"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.18"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}
session = requests.Session()
session.headers.update(HEADERS)


def load_universe() -> List[str]:
    symbols = [x.strip().upper() for x in UNIVERSE_PATH.read_text().splitlines() if x.strip()]
    if len(symbols) != 318 or len(set(symbols)) != 318:
        raise RuntimeError(f"Expected exactly 318 unique symbols, got {len(symbols)}")
    return symbols


def num(v, default=np.nan):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return default


def get_json(url: str, params: Optional[dict] = None, retries: int = 4) -> dict:
    last = None
    for a in range(retries):
        try:
            r = session.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last = e
        time.sleep(min(8, 1.2 * (2 ** a)))
    raise RuntimeError(f"Request failed: {url}: {last}")


def fetch_nasdaq_snapshot() -> Dict[str, dict]:
    p = get_json(NASDAQ_URL)
    rows = ((p.get("data") or {}).get("rows") or (((p.get("data") or {}).get("table") or {}).get("rows")) or [])
    return {str(r.get("symbol") or "").upper().strip(): r for r in rows if r.get("symbol")}


def stage1_rows(symbols, snap):
    out = []
    for s in symbols:
        r = snap.get(s, {})
        price = num(r.get("lastsale"))
        vol = num(r.get("volume"))
        cap = num(r.get("marketCap"))
        pct = num(r.get("pctchange"), 0)
        dv = price * vol if np.isfinite(price) and np.isfinite(vol) else np.nan
        att = (
            1.5 * abs(pct) + 2 * math.log10(dv) + .35 * math.log10(cap) + .5 * max(0, pct)
            if np.isfinite(dv) and dv > 0 and np.isfinite(cap) and cap > 0
            else -1e9
        )
        out.append(
            {
                "symbol": s,
                "price": price,
                "daily_pct": pct,
                "volume": vol,
                "dollar_volume": dv,
                "market_cap": cap,
                "attention": att,
                "snapshot_found": bool(r),
            }
        )
    return pd.DataFrame(out).sort_values(["attention", "symbol"], ascending=[False, True]).reset_index(drop=True)


def fetch_chart(symbol, range_="6mo", interval="1d"):
    p = get_json(
        YAHOO_CHART.format(symbol=symbol),
        {"range": range_, "interval": interval, "includePrePost": "false", "events": "div,splits"},
    )
    res = ((p.get("chart") or {}).get("result") or [None])[0]
    if not res:
        raise RuntimeError(f"No Yahoo result for {symbol}")
    meta = res.get("meta") or {}
    ts = res.get("timestamp") or []
    q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if not ts:
        return pd.DataFrame(), meta
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, unit="s", utc=True),
            "open": q.get("open", []),
            "high": q.get("high", []),
            "low": q.get("low", []),
            "close": q.get("close", []),
            "volume": q.get("volume", []),
        }
    )
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["high", "low", "close"]).reset_index(drop=True), meta


def lin_slope(s):
    y = s.dropna().to_numpy(float)
    return float(np.polyfit(np.arange(len(y)), y, 1)[0] / np.mean(y)) if len(y) >= 5 and np.mean(y) else 0


def atr14(df):
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return float(tr.tail(14).mean())


def rs_period(st, q, period):
    if len(st) < period + 1 or len(q) < period + 1:
        return np.nan
    return float(
        ((st.close.iloc[-1] / st.close.iloc[-period - 1] - 1)
         - (q.close.iloc[-1] / q.close.iloc[-period - 1] - 1))
        * 100
    )


def rs63(st, q):
    return rs_period(st, q, 63)


def vcp_proxy(df, current):
    def rg(d):
        return float((d.high.max() - d.low.min()) / current)

    r60, r30, r15 = rg(df.tail(60)), rg(df.tail(30)), rg(df.tail(15))
    d40 = df.tail(40)
    lo, hi = float(d40.low.min()), float(d40.high.max())
    pos = np.clip((current - lo) / (hi - lo) if hi > lo else .5, 0, 1)
    contraction = max(0, 1 - r15 / r60) if r60 else 0
    return float(
        np.clip(
            (.25 if r30 < r60 else 0)
            + (.25 if r15 < r30 else 0)
            + .25 * np.clip(contraction, 0, 1)
            + .25 * pos,
            0,
            1,
        )
    )


def continuation_features(df, current, trigger, ma10, ma20, ma50):
    """Detect a power move followed by a high-level shelf."""
    n = len(df)
    best = None
    for consolidation_days in range(4, 13):
        thrust_end = n - 1 - consolidation_days
        if thrust_end < 6:
            continue
        for thrust_span in range(5, 11):
            thrust_start = thrust_end - thrust_span
            if thrust_start < 0:
                continue
            start_close = float(df.close.iloc[thrust_start])
            end_close = float(df.close.iloc[thrust_end])
            if start_close <= 0:
                continue
            thrust_pct = (end_close / start_close - 1) * 100
            if best is None or thrust_pct > best["thrust_pct"]:
                best = {
                    "thrust_pct": thrust_pct,
                    "consolidation_days": consolidation_days,
                    "thrust_start": thrust_start,
                    "thrust_end": thrust_end,
                    "thrust_end_close": end_close,
                }

    if best is None:
        return {
            "Active Support": np.nan,
            "Thrust %": np.nan,
            "Consolidation Days": np.nan,
            "Shelf Drawdown %": np.nan,
            "Continuation Structure": False,
        }

    thrust_end = best["thrust_end"]
    consolidation = df.iloc[thrust_end + 1 :].copy()
    anchored = df.iloc[thrust_end:].copy()
    vol = anchored.volume.fillna(0).to_numpy(float)
    typical = ((anchored.high + anchored.low + anchored.close) / 3).to_numpy(float)
    avwap = float(np.average(typical, weights=vol)) if vol.sum() > 0 else float(anchored.close.mean())

    shelf_window = df.tail(min(5, max(2, best["consolidation_days"])))
    shelf = float(shelf_window.low.median())

    active_candidates = [
        x
        for x in [ma10, avwap, shelf]
        if np.isfinite(x) and x <= current and x > 0 and (current / x - 1) * 100 <= 8
    ]
    active_support = max(active_candidates) if active_candidates else np.nan

    thrust_end_close = best["thrust_end_close"]
    shelf_drawdown = (
        (float(consolidation.low.min()) / thrust_end_close - 1) * 100
        if not consolidation.empty and thrust_end_close > 0
        else np.nan
    )
    retention = current / thrust_end_close if thrust_end_close > 0 else 0
    active_distance = (
        (current / active_support - 1) * 100 if np.isfinite(active_support) and active_support > 0 else np.nan
    )
    room_to_trigger = (trigger / current - 1) * 100 if current > 0 else np.nan

    structure = bool(
        best["thrust_pct"] >= 15
        and 4 <= best["consolidation_days"] <= 12
        and retention >= .92
        and np.isfinite(shelf_drawdown)
        and shelf_drawdown >= -12
        and current > ma20
        and current > ma50
        and ma20 > ma50
        and lin_slope(df.close.tail(20)) > 0
        and np.isfinite(active_distance)
        and 0 <= active_distance <= 2.5
        and np.isfinite(room_to_trigger)
        and .5 <= room_to_trigger <= 8
    )

    return {
        "Active Support": active_support,
        "Thrust %": best["thrust_pct"],
        "Consolidation Days": best["consolidation_days"],
        "Shelf Drawdown %": shelf_drawdown,
        "Continuation Structure": structure,
    }


def feature_row(symbol, df, qqq):
    if len(df) < 30:
        raise RuntimeError(f"{symbol}: only {len(df)} daily rows; need at least 30")

    cur = float(df.close.iloc[-1])
    ma10 = float(df.close.tail(10).mean())
    ma20 = float(df.close.tail(20).mean())
    ma50 = float(df.close.tail(50).mean())
    trend = (
        (6 if cur > ma20 else 0)
        + (7 if cur > ma50 else 0)
        + (5 if ma20 > ma50 else 0)
        + (4 if lin_slope(df.close.tail(20)) > 0 else 0)
        + (3 if lin_slope(df.close.tail(50)) > 0 else 0)
    )

    trigger = float(df.high.iloc[-51:-1].max())
    p = (cur / trigger - 1) * 100
    vcp = vcp_proxy(df, cur)
    avg10 = float(df.volume.tail(10).mean())
    prev40 = float(df.volume.iloc[-50:-10].mean())
    dry = avg10 / prev40 if prev40 > 0 else np.nan
    avg20 = float(df.volume.iloc[-21:-1].mean())
    rvol = float(df.volume.iloc[-1] / avg20) if avg20 > 0 else np.nan
    rs = rs63(df, qqq)
    rs10 = rs_period(df, qqq, 10)
    rs_accel = rs10 - rs if np.isfinite(rs10) and np.isfinite(rs) else rs10
    atr = atr14(df)

    low10 = float(df.low.tail(10).min())
    valid = [x for x in [low10, ma20, ma50] if x < trigger and x >= .8 * trigger and trigger - x >= .75 * atr]
    support = max(valid) if valid else low10
    stop = support - .25 * atr

    d60 = df.tail(60)
    depth = float(d60.high.max() - d60.low.min())
    target = max(trigger + 1.25 * atr, trigger + .5 * depth)
    stretch = max(trigger + 3 * atr, trigger + depth)
    brr = (target - trigger) / (trigger - stop) if trigger > stop else np.nan
    srr = (target - support) / (support - stop) if support > stop else np.nan
    upside = (target / trigger - 1) * 100

    continuation = continuation_features(df, cur, trigger, ma10, ma20, ma50)

    score = trend
    score += 18 if -5 <= p < -3 else 12 if -3 <= p < -.5 else 9 if -.5 <= p <= 1 else 15 if 1 < p <= 5 else 6 if -10 <= p < -5 else 0
    score += 16 if vcp >= .7 else 12 if vcp >= .55 else 7 if vcp >= .4 else 2
    score += 18 if np.isfinite(rs) and rs >= 10 else 10 if np.isfinite(rs) and rs >= 5 else 5 if np.isfinite(rs) and rs >= 0 else -8 if np.isfinite(rs) else 0
    if np.isfinite(rs10) and np.isfinite(rs_accel):
        score += 8 if rs10 >= 8 and rs_accel >= 8 else 5 if rs10 >= 4 and rs_accel >= 4 else 2 if rs10 >= 0 and rs_accel >= 0 else 0
    score += 8 if np.isfinite(dry) and dry <= .8 else 5 if np.isfinite(dry) and dry <= 1 else 1 if np.isfinite(dry) else 0
    score += 8 if np.isfinite(rvol) and rvol >= 1.5 else 5 if np.isfinite(rvol) and rvol >= 1.15 else 2 if np.isfinite(rvol) else 0
    score += 8 if upside >= 15 else 5 if upside >= 10 else 2 if upside >= 5 else 0
    score = int(np.clip(round(score), 0, 100))

    rr = 95 if 1.5 <= brr <= 2 else 90 if 1.25 <= brr < 1.5 else 80 if brr > 2 else 55 if brr >= 1 else 35 if brr >= .75 else 50
    room = 90 if upside >= 15 else 70 if upside >= 10 else 45 if upside >= 5 else 60
    loc = 85 if -5 <= p < -3 else 90 if 1 < p <= 5 else 55 if -3 <= p < -.5 else 50 if -.5 <= p <= 1 else 45
    tq = round(.5 * rr + .3 * room + .2 * loc, 1)

    return {
        "Stock": symbol,
        "Daily Close": cur,
        "Support": support,
        "Active Support": continuation["Active Support"],
        "Support R/R Raw": srr,
        "Setup": score,
        "Trade Quality": tq,
        "Trigger": trigger,
        "% From Trigger Daily": p,
        "Breakout R/R": brr,
        "Initial Target": target,
        "Stretch Target": stretch,
        "Stop": stop,
        "ATR14": atr,
        "MA10": ma10,
        "MA20": ma20,
        "MA50": ma50,
        "RS vs QQQ": rs,
        "RS10 vs QQQ": rs10,
        "RS Acceleration": rs_accel,
        "VCP": vcp,
        "Volume Dry-Up": dry,
        "RVOL": rvol,
        "Target Upside %": upside,
        "Thrust %": continuation["Thrust %"],
        "Consolidation Days": continuation["Consolidation Days"],
        "Shelf Drawdown %": continuation["Shelf Drawdown %"],
        "Continuation Structure": continuation["Continuation Structure"],
    }


def live_price(symbol):
    df, meta = fetch_chart(symbol, "1d", "1m")
    if not df.empty and df.close.notna().any():
        return float(df.close.dropna().iloc[-1])
    p = num(meta.get("regularMarketPrice"))
    if np.isfinite(p):
        return float(p)
    raise RuntimeError(f"No live price for {symbol}")


def action_for(live, support, active_support, trigger, setup, trade_quality, continuation_structure, rs10, rs_accel):
    if setup < 55:
        return "WAIT"

    ps = (live / support - 1) * 100
    pt = (live / trigger - 1) * 100
    pas = (live / active_support - 1) * 100 if np.isfinite(active_support) and active_support > 0 else np.nan

    if pt > 5:
        return "EXTENDED"
    if live >= trigger:
        return "BUY BREAKOUT"
    if (
        continuation_structure
        and trade_quality >= 80
        and np.isfinite(rs10)
        and rs10 > 0
        and np.isfinite(rs_accel)
        and rs_accel >= 3
        and np.isfinite(pas)
        and 0 <= pas <= 2.5
    ):
        return "BUY CONTINUATION"
    if live < support:
        return "WAIT"
    if 0 <= ps <= 1:
        return "BUY SUPPORT"
    if 1 < ps <= 2.5:
        return "WAIT FOR SUPPORT"
    return "WAIT"


def fmt_rr(x):
    return "" if not np.isfinite(x) else "5R+" if x > 5 else f"{x:.2f}R"


def main():
    started = datetime.now(timezone.utc)
    universe = load_universe()
    snap = fetch_nasdaq_snapshot()
    stage1 = stage1_rows(universe, snap)
    stage1.to_csv(OUTPUT_DIR / "stage1_all_318.csv", index=False)
    qqq, _ = fetch_chart("QQQ", "6mo", "1d")

    results = []
    errors = {}
    for sym in universe:
        try:
            df, _ = fetch_chart(sym, "6mo", "1d")
            results.append(feature_row(sym, df, qqq))
        except Exception as e:
            errors[sym] = str(e)
        time.sleep(REQUEST_DELAY)

    detailed = pd.DataFrame(results)
    if detailed.empty:
        raise RuntimeError("No detailed results produced")
    if len(detailed) != len(universe):
        missing = sorted(set(universe) - set(detailed.Stock))
        raise RuntimeError(f"Full-universe validation failed: evaluated {len(detailed)}/{len(universe)}; missing={missing}; errors={errors}")

    detailed = detailed.sort_values(["Setup", "Trade Quality"], ascending=[False, False]).reset_index(drop=True)
    detailed.insert(0, "Rank", np.arange(1, len(detailed) + 1))

    lives = {}
    for sym in detailed.Stock:
        try:
            lives[sym] = live_price(sym)
        except Exception as e:
            errors[f"{sym}:live"] = str(e)
            lives[sym] = np.nan
        time.sleep(REQUEST_DELAY)

    detailed["Live"] = detailed.Stock.map(lives)
    detailed["% From Trigger"] = (detailed.Live / detailed.Trigger - 1) * 100
    detailed["% Above Support"] = (detailed.Live / detailed.Support - 1) * 100
    detailed["% Above Active Support"] = (detailed.Live / detailed["Active Support"] - 1) * 100
    detailed["Action"] = [
        action_for(l, s, a, t, sc, tq, cs, r10, ra) if np.isfinite(l) else "WAIT"
        for l, s, a, t, sc, tq, cs, r10, ra in zip(
            detailed.Live,
            detailed.Support,
            detailed["Active Support"],
            detailed.Trigger,
            detailed.Setup,
            detailed["Trade Quality"],
            detailed["Continuation Structure"],
            detailed["RS10 vs QQQ"],
            detailed["RS Acceleration"],
        )
    ]
    detailed["Support R/R"] = detailed["Support R/R Raw"].map(fmt_rr)

    detailed.to_csv(OUTPUT_DIR / "full_318_evaluated.csv", index=False)
    detailed.to_csv(OUTPUT_DIR / "stage2_detailed.csv", index=False)
    cols = [
        "Rank", "Stock", "Live", "Support", "Active Support", "Support R/R", "Setup", "Trade Quality",
        "Trigger", "% From Trigger", "Breakout R/R", "RS10 vs QQQ", "RS Acceleration", "Thrust %",
        "Consolidation Days", "Action"
    ]
    leaderboard = detailed[cols].copy()
    leaderboard.to_csv(OUTPUT_DIR / "latest.csv", index=False)
    actionable = leaderboard[leaderboard.Action != "WAIT"].copy()
    actionable.to_csv(OUTPUT_DIR / "actionable.csv", index=False)

    manifest = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "stage1_snapshot_found": int(stage1.snapshot_found.sum()),
        "historical_evaluation_attempted": len(universe),
        "historical_evaluation_completed": len(detailed),
        "live_evaluation_attempted": len(detailed),
        "leaderboard_rows": len(leaderboard),
        "actionable_rows": len(actionable),
        "historical_source": "Yahoo Finance chart API 6mo/1d",
        "live_source": "Yahoo Finance chart API 1d/1m",
        "stage1_source": "Nasdaq stock screener API",
        "entry_states": ["BUY SUPPORT", "BUY CONTINUATION", "BUY BREAKOUT"],
        "errors": errors,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUTPUT_DIR / "latest.json").write_text(json.dumps({"manifest": manifest, "leaderboard": leaderboard.replace({np.nan: None}).to_dict("records"), "actionable": actionable.replace({np.nan: None}).to_dict("records")}, indent=2))
    print(json.dumps(manifest, indent=2))
    print(actionable.to_string(index=False))


if __name__ == "__main__":
    main()
