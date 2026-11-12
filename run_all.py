"""
run_all.py — Run the full pipeline end to end and print a summary.

  1. Build the monthly panel from the raw data.
  2. Verify the optimiser (greedy == PuLP/CBC) across the scenario grid.
  3. Run the scenario sweep, clean-capacity and break-even tables.
  4. Run demand forecasting and the decision test (RQ3).

All CSV outputs land in ./outputs. This is the script to run after dropping in
updated data; the dashboard (streamlit run dashboard.py) reads the same code.
"""
import os
import pandas as pd

import params as P
from data_loader import build_panel, observed_peaks
from optimiser import cross_check
from scenarios import (run_grid, clean_capacity_table, break_even_table,
                       decision_map, representative_demand, kaub_stat_robustness)
from forecasting import run_forecasts, decision_test

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def main():
    print("=" * 72)
    print("RHINE-CORRIDOR MODAL-SHIFT MODEL — full pipeline")
    print("=" * 72)

    # 1. data
    df = build_panel(save=True)
    peaks = {m: v["tonnes"] for m, v in observed_peaks(df).items()}
    demand, cc = representative_demand(df)
    print(f"\n[1] Data: {len(df)} months {df['date'].min():%Y-%m}→{df['date'].max():%Y-%m}; "
          f"barge peak {peaks['barge']/1e6:.2f} Mt, rail peak {peaks['rail']/1e6:.2f} Mt; "
          f"representative demand {demand/1e6:.2f} Mt.")

    # 2. verification: greedy == LP over the whole grid
    nchk = 0
    for cargo in P.CARGOES:
        for _, kw in P.KWATER_TIERS:
            for kdb in P.KDB_SCENARIOS.values():
                for lam in P.LAMBDAS.values():
                    cross_check(cargo, kw, kdb, demand, lam, peaks)
                    nchk += 1
    print(f"[2] Verification: greedy == PuLP/CBC in all {nchk} grid cells. ✓")

    # 3. scenarios
    grid = run_grid(peaks, demand, save=True)
    clean_capacity_table(peaks, save=True)
    bet = break_even_table(peaks, demand, save=True)
    decision_map(peaks, demand, save=True)
    lam_indep = grid.groupby(
        ["water_tier", "rail_scenario", "headroom", "cargo"])["road_share"].nunique().max()
    print(f"[3] Scenarios: {len(grid)} rows. Carbon price changes the split: "
          f"{'NO' if lam_indep == 1 else 'YES'} (λ is non-tipping on this corridor).")
    print("    Break-even (base-case 15% headroom):")
    row = bet[bet["headroom"] == P.HEADROOM_BASE].iloc[0]
    print(f"      water: {row['water_result']}")
    print(f"      rail : {row['rail_result']}")
    ksr = kaub_stat_robustness(peaks, save=True)
    stable = ksr["water_break_even"].nunique() == 1
    print(f"    Kaub aggregation robustness (min/p10/median): water break-even "
          f"{'STABLE' if stable else 'CHANGES'} across statistics.")

    # 4. forecasting + decision test
    preds, metrics = run_forecasts(df)
    preds.to_csv(os.path.join(OUT, "forecasts.csv"), index=False)
    dt, summ = decision_test(df, preds, peaks)
    dt.to_csv(os.path.join(OUT, "decision_test.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "decision_test_summary.csv"), index=False)
    best = min(metrics, key=lambda m: metrics[m]["MAE"])
    print(f"[4] Forecasting ({len(preds)} test months): best = {best} "
          f"(MAE {metrics[best]['MAE']/1e6:.2f} Mt, MAPE {metrics[best]['MAPE']:.1f}%).")
    print("    RQ3 decision test (mean |road-share Δ| vs observed demand):")
    for _, r in summ.iterrows():
        print(f"      {r['model']:15} {r['mean_abs_road_share_diff_pp']:.2f} pp "
              f"(max {r['max_abs_road_share_diff_pp']:.1f} pp)")

    print("\nAll outputs written to ./outputs/:")
    for f in sorted(os.listdir(OUT)):
        print("   -", f)
    print("\nDashboard:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
