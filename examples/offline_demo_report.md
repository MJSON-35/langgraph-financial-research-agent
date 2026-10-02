# Multi-Agent Stock Picker Demo Report

Scenario: `offline_portfolio_demo`

## Macro Summary

- Status: completed
- Summary: neutral macro overlay from structured growth, inflation, policy, and volatility signals.
- Preferred sectors: Technology, Semiconductors, Consumer
- Risk sectors: Energy
- Key macro risks: policy uncertainty

## Top 3 Picks

1. **FFF** (Future Chips) | score=0.52
   - Rationale: weighted quant 0.42 + weighted equity 0.15 + macro adjustment 0.02 - risk penalty 0.03 - missing-data penalty 0.04.
2. **AAA** (Alpha Systems) | score=0.51
   - Rationale: weighted quant 0.41 + weighted equity 0.15 + macro adjustment 0.02 - risk penalty 0.03 - missing-data penalty 0.04.
3. **CCC** (Core Retail) | score=0.48
   - Rationale: weighted quant 0.39 + weighted equity 0.14 + macro adjustment 0.02 - risk penalty 0.03 - missing-data penalty 0.04.

## Rejected Stocks

- **EEE** (Ever Health) | score=0.43 | reason=moderate risk penalty; missing fundamental evidence; insufficient qualitative evidence
- **BBB** (Beta Holdings) | score=0.41 | reason=moderate risk penalty; missing fundamental evidence; insufficient qualitative evidence
- **DDD** (Delta Energy) | score=0.38 | reason=moderate risk penalty; poor macro sector fit; missing fundamental evidence

## Notes

This checked in example was produced by `data/demo_run.json` using fictional stocks and no external service. Runtime reports under `outputs/` are intentionally ignored.
