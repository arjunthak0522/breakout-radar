from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&offset=0&download=true"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
ROOT = Path(__file__).resolve().parent
UNIVERSE_PATH = ROOT / "universe.txt"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

STAGE2_COUNT = int(os.getenv("STAGE2_COUNT", "139"))
LEADERBOARD_COUNT = int(os.getenv("LEADERBOARD_COUNT", "40"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.18"))

FORCE_STAGE2 = {
    "NBIS","BE","GLW","COHR","META","LITE","CIEN","AAOI","CRDO","VRT","ANET",
    "WDC","INTC","MRVL","ORCL","SHOP","CRWV","TEM","PLTR","NVDA","AMD","MU","HPE","DELL"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nasdaq.com/",
}

session = requests.Session()
session.headers.update(HEADERS)


def load_universe() -> List[str]:
    symbols = [x.strip().upper() for x in UNIVERSE_PATH.read_text().splitlines() if x.strip()]
    if len(symbols) != 318 or len(set(symbols)) != 318:
        raise RuntimeError(f"Expected exactly 318 unique symbols, got {len(symbols)} / {len(set(symbols))} unique")
    return symbols


def num(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
    if s in {"", "N/A", "NA", "None", "--"}:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def get_json(url: str, params: Optional[dict] = None, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            last = exc
        time.sleep(min(8, 1.2 * (2 ** attempt)))
    raise RuntimeError(f"Request failed: {url}: {last}")


def fetch_nasdaq_snapshot() -> Dict[str, dict]:
    payload = get_json(NASDAQ_URL)
    rows = (((payload or {}).get("data") or {}).get("rows")
            or ((((payload or {}).get("data") or {}).get("table") or {}).get("rows"))
            or [])
    out = {}
    for row in rows:
        sym = str(row.get("symbol") or "").upper().strip()
        if sym:
            out[sym] = row
    return out


def stage1_rows(symbols: List[str], snap: Dict[str, dict]) -> pd.DataFrame:
    records = []
    for symbol in symbols:
        row = snap.get(symbol, {})
        price = num(row.get("lastsale"))
        volume = num(row.get("volume"))
        market_cap = num(row.get("marketCap"))
        pct = num(row.get("pctchange"), 0.0)
        dollar_volume = price * volume if np.isfinite(price) and np.isfinite(volume) else np.nan
        if all(np.isfinite(x) and x > 0 for x in [dollar_volume, market_cap]):
            attention = (
                1.5 * abs(pct)
                + 2.0 * math.log10(dollar_volume)
                + 0.35 * math.log10(market_cap)
                + 0.5 * max(0.0, pct)
            )
        else:
            attention = -1e9
        records.append({
            "symbol": symbol,
            "price": price,
            "daily_pct": pct,
            "volume": volume,
            "dollar_volume": dollar_volume,
            "market_cap": market_cap,
            "attention": attention,
            "snapshot_found": bool(row),
        })
    return pd.DataFrame(records).sort_values(["attention", "symbol"], ascending=[False, True]).reset_index(drop=True)


def select_stage2(stage1: pd.DataFrame) -> List[str]:
    ordered = stage1["symbol"].tolist()
    selected = ordered[:STAGE2_COUNT]
    for sym in FORCE_STAGE2:
        if sym in ordered and sym not in selected:
            selected.append(sym)
    # Keep total near the configured count while guaranteeing force-inclusions.
    if len(selected) > STAGE2_COUNT:
        forced = [s for s in selected if s in FORCE_STAGE2]
        nonforced = [s for s in selected if s not in FORCE_STAGE2]
        selected = nonforced[: max(0, STAGE2_COUNT - len(forced))] + forced
    return list(dict.fromkeys(selected))


def fetch_chart(symbol: str, range_: str = "6mo", interval: str = "1d") -> tuple[pd.DataFrame, dict]:
    url = YAHOO_CHART.format(symbol=symbol)
    payload = get_json(url, {"range": range_, "interval": interval, "includePrePost": "false", "events": "div,splits"})
    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise RuntimeError(f"No Yahoo result for {symbol}")
    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = (((result.get("indicators") or {}).get("quote")) or [{}])[0]
    if not timestamps:
        return pd.DataFrame(), meta
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    })
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high","low","close"]).reset_index(drop=True)
    return df, meta


def lin_slope(values: pd.Series) -> float:
    y = values.dropna().to_numpy(dtype=float)
    if len(y) < 5:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    denom = np.nanmean(y)
    return float(slope / denom) if denom else 0.0


def atr14(df: pd.DataFrame) -> float:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return float(tr.tail(14).mean())


def rs63(stock: pd.DataFrame, qqq: pd.DataFrame) -> float:
    if len(stock) < 64 or len(qqq) < 64:
        return np.nan
    sr = stock["close"].iloc[-1] / stock["close"].iloc[-64] - 1
    qr = qqq["close"].iloc[-1] / qqq["close"].iloc[-64] - 1
    return float((sr - qr) * 100)


def vcp_proxy(df: pd.DataFrame, current: float) -> tuple[float, float, float, float, float]:
    d60, d30, d15 = df.tail(60), df.tail(30), df.tail(15)
    def rangex(d):
        return float((d["high"].max() - d["low"].min()) / current) if current else np.nan
    r60, r30, r15 = rangex(d60), rangex(d30), rangex(d15)
    d40 = df.tail(40)
    lo40, hi40 = float(d40["low"].min()), float(d40["high"].max())
    pos40 = (current - lo40) / (hi40 - lo40) if hi40 > lo40 else 0.5
    pos40 = float(np.clip(pos40, 0, 1))
    contraction = max(0.0, 1.0 - (r15 / r60)) if r60 and np.isfinite(r60) else 0.0
    proxy = (
        (0.25 if r30 < r60 else 0.0)
        + (0.25 if r15 < r30 else 0.0)
        + 0.25 * float(np.clip(contraction, 0, 1))
        + 0.25 * pos40
    )
    return float(np.clip(proxy, 0, 1)), r60, r30, r15, pos40


def feature_row(symbol: str, df: pd.DataFrame, qqq: pd.DataFrame) -> dict:
    if len(df) < 65:
        raise RuntimeError(f"{symbol}: only {len(df)} daily rows")
    current = float(df["close"].iloc[-1])
    ma20 = float(df["close"].tail(20).mean())
    ma50 = float(df["close"].tail(50).mean())
    slope20 = lin_slope(df["close"].tail(20))
    slope50 = lin_slope(df["close"].tail(50))

    trend = 0
    trend += 6 if current > ma20 else 0
    trend += 7 if current > ma50 else 0
    trend += 5 if ma20 > ma50 else 0
    trend += 4 if slope20 > 0 else 0
    trend += 3 if slope50 > 0 else 0

    trigger = float(df["high"].iloc[-51:-1].max())
    pct_from_trigger = (current / trigger - 1) * 100

    vcp, r60, r30, r15, pos40 = vcp_proxy(df, current)

    avg10 = float(df["volume"].tail(10).mean())
    prev40 = float(df["volume"].iloc[-50:-10].mean()) if len(df) >= 50 else float(df["volume"].iloc[:-10].mean())
    dry = avg10 / prev40 if prev40 > 0 else np.nan

    avg20_prev = float(df["volume"].iloc[-21:-1].mean())
    rvol = float(df["volume"].iloc[-1] / avg20_prev) if avg20_prev > 0 else np.nan
    rs = rs63(df, qqq)
    atr = atr14(df)

    low10 = float(df["low"].tail(10).min())
    support_candidates = [low10, ma20, ma50]
    valid = [
        x for x in support_candidates
        if x < trigger and x >= 0.80 * trigger and (trigger - x) >= 0.75 * atr
    ]
    support = max(valid) if valid else low10
    stop = support - 0.25 * atr

    d60 = df.tail(60)
    base_depth = float(d60["high"].max() - d60["low"].min())
    initial_target = max(trigger + 1.25 * atr, trigger + 0.5 * base_depth)
    stretch_target = max(trigger + 3 * atr, trigger + base_depth)

    breakout_risk = trigger - stop
    breakout_reward = initial_target - trigger
    breakout_rr = breakout_reward / breakout_risk if breakout_risk > 0 else np.nan

    support_risk = support - stop
    support_reward = initial_target - support
    support_rr = support_reward / support_risk if support_risk > 0 else np.nan

    target_upside = (initial_target / trigger - 1) * 100

    score = trend
    p = pct_from_trigger
    if -5 <= p < -3: score += 18
    elif -3 <= p < -0.5: score += 12
    elif -0.5 <= p <= 1: score += 9
    elif 1 < p <= 5: score += 15
    elif -10 <= p < -5: score += 6

    if vcp >= 0.70: score += 16
    elif vcp >= 0.55: score += 12
    elif vcp >= 0.40: score += 7
    else: score += 2

    if np.isfinite(rs):
        if rs >= 10: score += 18
        elif rs >= 5: score += 10
        elif rs >= 0: score += 5
        else: score -= 8

    if np.isfinite(dry):
        if dry <= 0.8: score += 8
        elif dry <= 1.0: score += 5
        else: score += 1

    if np.isfinite(rvol):
        if rvol >= 1.5: score += 8
        elif rvol >= 1.15: score += 5
        else: score += 2

    if target_upside >= 15: score += 8
    elif target_upside >= 10: score += 5
    elif target_upside >= 5: score += 2

    score = int(np.clip(round(score), 0, 100))

    if 1.5 <= breakout_rr <= 2.0: rr_component = 95
    elif 1.25 <= breakout_rr < 1.5: rr_component = 90
    elif breakout_rr > 2.0: rr_component = 80
    elif breakout_rr >= 1.0: rr_component = 55
    elif breakout_rr >= 0.75: rr_component = 35
    else: rr_component = 50

    if target_upside >= 15: room_component = 90
    elif target_upside >= 10: room_component = 70
    elif target_upside >= 5: room_component = 45
    else: room_component = 60

    if -5 <= p < -3: location_component = 85
    elif 1 < p <= 5: location_component = 90
    elif -3 <= p < -0.5: location_component = 55
    elif -0.5 <= p <= 1: location_component = 50
    else: location_component = 45

    trade_quality = round(0.50 * rr_component + 0.30 * room_component + 0.20 * location_component, 1)

    return {
        "Stock": symbol,
        "Daily Close": current,
        "Support": support,
        "Support R/R Raw": support_rr,
        "Setup": score,
        "Trade Quality": trade_quality,
        "Trigger": trigger,
        "% From Trigger Daily": pct_from_trigger,
        "Breakout R/R": breakout_rr,
        "Initial Target": initial_target,
        "Stretch Target": stretch_target,
        "Stop": stop,
        "ATR14": atr,
        "MA20": ma20,
        "MA50": ma50,
        "RS vs QQQ": rs,
        "VCP": vcp,
        "Volume Dry-Up": dry,
        "RVOL": rvol,
        "Target Upside %": target_upside,
    }


def live_price(symbol: str) -> float:
    df, meta = fetch_chart(symbol, "1d", "1m")
    if not df.empty and df["close"].notna().any():
        return float(df["close"].dropna().iloc[-1])
    p = num(meta.get("regularMarketPrice"))
    if np.isfinite(p):
        return float(p)
    raise RuntimeError(f"No live price for {symbol}")


def action_for(live: float, support: float, trigger: float, setup: float) -> str:
    if setup < 55:
        return "WAIT"
    pct_support = (live / support - 1) * 100
    pct_trigger = (live / trigger - 1) * 100
    if pct_trigger > 5:
        return "EXTENDED"
    if live >= trigger:
        return "BUY BREAKOUT"
    if live < support:
        return "WAIT"
    if 0 <= pct_support <= 1.0:
        return "BUY SUPPORT"
    if 1.0 < pct_support <= 2.5:
        return "BUY NEAR SUPPORT"
    return "WAIT"


def fmt_support_rr(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return "5R+" if x > 5 else f"{x:.2f}R"


def main():
    started = datetime.now(timezone.utc)
    universe = load_universe()
    snap = fetch_nasdaq_snapshot()
    stage1 = stage1_rows(universe, snap)
    stage1.to_csv(OUTPUT_DIR / "stage1_all_318.csv", index=False)

    selected = select_stage2(stage1)
    qqq, _ = fetch_chart("QQQ", "6mo", "1d")

    results, errors = [], {}
    for i, sym in enumerate(selected, 1):
        try:
            df, _ = fetch_chart(sym, "6mo", "1d")
            results.append(feature_row(sym, df, qqq))
        except Exception as exc:
            errors[sym] = str(exc)
        time.sleep(REQUEST_DELAY)

    detailed = pd.DataFrame(results)
    if detailed.empty:
        raise RuntimeError("No Stage 2 results produced")
    detailed = detailed.sort_values(["Setup", "Trade Quality"], ascending=[False, False]).reset_index(drop=True)
    detailed.insert(0, "Rank", np.arange(1, len(detailed) + 1))
    detailed.to_csv(OUTPUT_DIR / "stage2_detailed.csv", index=False)

    top = detailed.head(LEADERBOARD_COUNT).copy()
    lives = {}
    for sym in top["Stock"]:
        try:
            lives[sym] = live_price(sym)
        except Exception as exc:
            errors[f"{sym}:live"] = str(exc)
            lives[sym] = np.nan
        time.sleep(REQUEST_DELAY)

    top["Live"] = top["Stock"].map(lives)
    top["% From Trigger"] = (top["Live"] / top["Trigger"] - 1) * 100
    top["% Above Support"] = (top["Live"] / top["Support"] - 1) * 100
    top["Action"] = [
        action_for(l, s, t, sc) if np.isfinite(l) else "WAIT"
        for l, s, t, sc in zip(top["Live"], top["Support"], top["Trigger"], top["Setup"])
    ]
    top["Support R/R"] = top["Support R/R Raw"].map(fmt_support_rr)

    columns = [
        "Rank","Stock","Live","Support","Support R/R","Setup","Trade Quality",
        "Trigger","% From Trigger","Breakout R/R","Action"
    ]
    leaderboard = top[columns].copy()
    for c in ["Live","Support","Trigger"]:
        leaderboard[c] = leaderboard[c].round(2)
    leaderboard["% From Trigger"] = leaderboard["% From Trigger"].round(2)
    leaderboard["Breakout R/R"] = leaderboard["Breakout R/R"].round(2)
    leaderboard.to_csv(OUTPUT_DIR / "latest.csv", index=False)

    manifest = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "stage1_snapshot_found": int(stage1["snapshot_found"].sum()),
        "stage2_selected": len(selected),
        "stage2_completed": len(detailed),
        "leaderboard_rows": len(leaderboard),
        "historical_source": "Yahoo Finance chart API 6mo/1d",
        "live_source": "Yahoo Finance chart API 1d/1m",
        "stage1_source": "Nasdaq stock screener API",
        "errors": errors,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    payload = {
        "manifest": manifest,
        "leaderboard": leaderboard.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "latest.json").write_text(json.dumps(payload, indent=2))

    print(json.dumps(manifest, indent=2))
    print()
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
