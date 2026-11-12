# Rhine-Corridor Modal-Shift Model

Disruption-aware, cost- and carbon-efficient modal-shift optimiser for
road → rail → inland-waterway freight on the Rotterdam/Antwerp–Duisburg–Basel
axis of the Rhine. Companion program for the M.Sc. thesis proposal
"Decarbonising Rhine-Corridor Freight Transport".

## Live dashboard
Deployed with Streamlit Community Cloud. Main file: `dashboard.py`.

## Run the interactive dashboard
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
Sidebar controls: cargo, Kaub water level (κ_water), rail state + single-track
fraction (κ_DB), monthly demand, carbon price λ, capacity headroom.
Tabs: Scenario optimiser (RQ1) · Disruptions (RQ2) · Forecasting (RQ3) ·
Data & validation.

## Reproduce all results (CSV outputs → ./outputs)
```bash
python run_all.py
```
Runs the data assembly, optimiser verification (greedy == PuLP/CBC),
scenario sweep, break-even thresholds, Kaub aggregation robustness, and the
demand-forecasting decision test.

## Files
- `params.py` — all model parameters (costs, emissions, κ tiers, λ) with provenance
- `data_loader.py` — Destatis + Kaub → monthly panel
- `optimiser.py` — the LP (closed-form greedy + PuLP/CBC cross-check)
- `scenarios.py` — scenario grid, break-even, decision map
- `forecasting.py` — four demand models + decision test (RQ3)
- `dashboard.py` — Streamlit dashboard
- `run_all.py` — full pipeline
- `data/` — Destatis 46321/46131/46231 (International receipt) + PEGELONLINE Kaub

## Data sources
Costs: Panteia (2023). Emissions: EcoTransIT World / ISO 14083.
κ_water: CCNR (2023) "Act now!" Fig. 18. Demand: Destatis GENESIS 46321/46131/46231.
Water levels: WSV/PEGELONLINE Kaub gauge. Carbon prices: UBA (2019) & nEHS/BEHG.
