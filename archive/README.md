# Archive

Superseded scripts, kept for reference only. They are not part of the D2 pipeline
(`build_model_inputs.py` + `lp_model.py`, see the main README, section "Model").

| File | What it was | Replaced by |
|---|---|---|
| `rebalancing_model.py` | first model draft (06.10.2026): multi-period min-cost flow on quartier zones, solved with PuLP | `lp_model.py` |
| `coverage_curves.py` | share of a quartier's demand with an available bike within walking distance (06.10.2026) | not used in D2 |

They import from the repo root (`analyze_snapshots`, `build_demand`) and read
`model_input/`, so to run one, copy it back to the repo root and rebuild its
input with `build_demand.py` first. Their old outputs are in
`archive/old_model_input/` (local only, ignored by git).
