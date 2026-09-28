# Breakout Radar

Standalone stock breakout/support radar. This repository is intentionally separate from Swing Intelligence.

## Fixed universe

- `universe.txt` contains the exact 318-symbol reference universe.
- Ordinary radar runs use this fixed universe.
- Updating the universe is a separate maintenance task.

## Data sources

- Stage 1 current snapshot: Nasdaq stock screener API
- Historical daily OHLCV: Yahoo Finance chart API, 6mo / 1d
- Live/intraday price refresh: Yahoo Finance chart API, 1d / 1m
- Relative strength benchmark: QQQ

## Execution

The scanner:

1. Screens all 318 names in Stage 1.
2. Ranks by the documented attention formula.
3. Sends the strongest 139 candidates to Stage 2 while force-including important tech-adjacent names.
4. Calculates moving-average structure, trigger distance, VCP proxy, volume dry-up, RVOL, RS vs QQQ, ATR, structural support, stop, targets, support R/R, breakout R/R, Setup Score and Trade Quality.
5. Refreshes live 1-minute Yahoo prices for the top leaderboard names.
6. Produces the action labels: BUY SUPPORT, BUY NEAR SUPPORT, BUY BREAKOUT, WAIT or EXTENDED.

## Outputs

- `output/stage1_all_318.csv`
- `output/stage2_detailed.csv`
- `output/latest.csv`
- `output/latest.json`
- `output/manifest.json`

Support R/R is displayed as `5R+` when the raw value exceeds 5.

## Action-label implementation note

The source specification says support/breakout entries require acceptable setup quality but does not assign a numeric cutoff. The executable implementation uses Setup Score >= 55 as the minimum quality threshold. EXTENDED is defined as more than 5% above trigger. These two implementation constants are explicit here so they can be revised without ambiguity.

## Run

The GitHub Actions workflow runs on changes to scanner inputs and can also be started manually from the Actions tab.
