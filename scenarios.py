"""
scenarios.py — Scenario sweep, break-even thresholds and decision map.

Produces (in ./outputs):
  scenario_grid.csv   full factorial: cargo × κ_water tier × κ_DB scenario
                      × λ × headroom  →  split, cost, CO₂, road spill
  break_even.csv      per cargo & headroom: the Kaub level and κ_DB at which
                      road spillover begins, for a representative demand
  clean_capacity.csv  rail/barge/clean capacity at every κ tier (the tables
                      in §6.3)
  decision_map.csv    road-share surface over κ_water × κ_DB (for the dashboard)
"""
import os
import numpy as np
import pandas as pd
import params as P
from optimiser import optimise, capacities
from data_loader import build_panel, observed_peaks

OUT = os.path.join(os.path.dirname(__file__), "outputs")


def representative_demand(df):
    """Median monthly demand over complete-case months (all three modes present)."""
    cc = df.dropna(subset=["barge_t", "rail_t", "road_t"])
    return float(cc["demand_t"].median()), cc


# ---------------------------------------------------------------------------
# 1. Full factorial scenario grid
# ---------------------------------------------------------------------------
def run_grid(peaks, demand, save=True):
    rows = []
    for cargo in P.CARGOES:
        for head in P.HEADROOM_SWEEP:
            for wname, (lo, kw) in [(f"Kaub≥{int(lo)}", (lo, kw))
                                    for lo, kw in P.KWATER_TIERS]:
                for sname, kdb in P.KDB_SCENARIOS.items():
                    for lname, lam in P.LAMBDAS.items():
                        r = optimise(cargo, kw, kdb, demand, lam, peaks, head)
                        rows.append(dict(
                            cargo=cargo, headroom=head,
                            water_tier=wname, kappa_water=kw,
                            rail_scenario=sname, kappa_db=kdb,
                            lam_name=lname, lam=lam,
                            rail_share=r.shares["rail"], barge_share=r.shares["barge"],
                            road_share=r.shares["road"],
                            cost_eur=r.cost_eur, co2_t=r.co2_t,
                            objective=r.objective, road_spill_t=r.road_spill))
    g = pd.DataFrame(rows)
    if save:
        g.to_csv(os.path.join(OUT, "scenario_grid.csv"), index=False)
    return g


# ---------------------------------------------------------------------------
# 2. Clean-capacity table (§6.3)
# ---------------------------------------------------------------------------
def clean_capacity_table(peaks, headroom=P.HEADROOM_BASE, save=True):
    cap_barge0 = peaks["barge"] * (1 + headroom)
    cap_rail0 = peaks["rail"] * (1 + headroom)
    rows = []
    # water sweep at full rail
    for lo, kw in P.KWATER_TIERS:
        rows.append(dict(dimension="water", label=f"Kaub≥{int(lo)}cm",
                         kappa=kw, barge_cap_Mt=kw * cap_barge0 / 1e6,
                         rail_cap_Mt=cap_rail0 / 1e6,
                         clean_cap_Mt=(kw * cap_barge0 + cap_rail0) / 1e6))
    # rail sweep at full water
    for sname, kdb in P.KDB_SCENARIOS.items():
        rows.append(dict(dimension="rail", label=sname,
                         kappa=kdb, barge_cap_Mt=cap_barge0 / 1e6,
                         rail_cap_Mt=kdb * cap_rail0 / 1e6,
                         clean_cap_Mt=(cap_barge0 + kdb * cap_rail0) / 1e6))
    t = pd.DataFrame(rows)
    if save:
        t.to_csv(os.path.join(OUT, "clean_capacity.csv"), index=False)
    return t


# ---------------------------------------------------------------------------
# 3. Break-even thresholds (solve capacity balance for the parameter)
# ---------------------------------------------------------------------------
def _kaub_for_kappa(kw_star):
    """Lowest Kaub band whose κ_water ≥ kw_star (what water level is needed)."""
    # tiers are high→low; find the smallest level giving at least kw_star
    for lo, kw in sorted(P.KWATER_TIERS, key=lambda t: t[0]):
        if kw >= kw_star:
            return lo, kw
    return None, None


def break_even_table(peaks, demand, save=True):
    """For each cargo & headroom, the κ (and Kaub band) where road spillover starts.

    Road appears when κ_water·Cap_barge + κ_DB·Cap_rail < D.
      • water threshold (rail normal): κ_water* = (D − Cap_rail)/Cap_barge
      • rail  threshold (water normal): κ_DB*   = (D − Cap_barge)/Cap_rail
    The split is independent of cargo/λ here (capacity-driven), but headroom
    changes the base capacities, so we report per headroom.
    """
    rows = []
    for head in P.HEADROOM_SWEEP:
        cap_barge0 = peaks["barge"] * (1 + head)
        cap_rail0 = peaks["rail"] * (1 + head)
        # water threshold with rail normal (κ_DB = 1)
        kw_star = (demand - cap_rail0) / cap_barge0
        if kw_star > 1:
            water_label = "road unavoidable (even full water)"
        elif kw_star <= 0:
            water_label = "always road-free (barge not needed)"
        else:
            kaub_lo, _ = _kaub_for_kappa(min(kw_star, 1.0))
            water_label = f"road-free only above Kaub {int(kaub_lo)} cm"
        # rail threshold with water normal (κ_water = 1)
        kdb_star = (demand - cap_barge0) / cap_rail0
        if kdb_star > 1:
            rail_label = "road unavoidable (even full rail)"
        elif kdb_star <= 0:
            rail_label = "no — barge alone covers demand"
        else:
            rail_label = f"yes — road appears below κ_DB {round(kdb_star, 2)}"
        rows.append(dict(
            headroom=head, demand_Mt=round(demand / 1e6, 2),
            clean_cap_full_Mt=round((cap_barge0 + cap_rail0) / 1e6, 2),
            kappa_water_threshold=round(kw_star, 3),
            water_result=water_label,
            kappa_db_threshold=round(kdb_star, 3),
            rail_result=rail_label))
    t = pd.DataFrame(rows)
    if save:
        t.to_csv(os.path.join(OUT, "break_even.csv"), index=False)
    return t


# ---------------------------------------------------------------------------
# 4. Decision map (road share over κ_water × κ_DB)
# ---------------------------------------------------------------------------
def decision_map(peaks, demand, cargo="Dry bulk", lam=P.LAMBDA_BASE,
                 headroom=P.HEADROOM_BASE, n=21, save=True):
    kws = np.linspace(0.15, 1.0, n)
    kdbs = np.linspace(0.0, 1.0, n)
    rows = []
    for kw in kws:
        for kdb in kdbs:
            r = optimise(cargo, kw, kdb, demand, lam, peaks, headroom)
            rows.append(dict(kappa_water=kw, kappa_db=kdb,
                             road_share=r.shares["road"]))
    m = pd.DataFrame(rows)
    if save:
        m.to_csv(os.path.join(OUT, "decision_map.csv"), index=False)
    return m


def kaub_stat_robustness(peaks, save=True):
    """Robustness of κ_water to the monthly aggregation statistic.

    Rebuilds the panel under the monthly minimum (base case), 10th percentile
    and median, and reports (a) how the months distribute across κ_water tiers
    and (b) the water break-even at representative demand. If the break-even
    conclusion is stable across statistics, the choice of minimum is safe.
    """
    from data_loader import build_panel, KAUB_STAT_COL
    rows = []
    for stat in KAUB_STAT_COL:
        d = build_panel(save=False, kaub_stat=stat)
        demand, _ = representative_demand(d)
        bet = break_even_table(peaks, demand, save=False)
        base = bet[bet["headroom"] == P.HEADROOM_BASE].iloc[0]
        tier_counts = d["kappa_water"].value_counts().sort_index().to_dict()
        rows.append(dict(
            statistic=stat,
            months_kw_0_15=int(tier_counts.get(0.15, 0)),
            months_kw_0_25=int(tier_counts.get(0.25, 0)),
            months_kw_0_50=int(tier_counts.get(0.50, 0)),
            months_kw_1_00=int(tier_counts.get(1.00, 0)),
            water_break_even=base["water_result"]))
    t = pd.DataFrame(rows)
    if save:
        t.to_csv(os.path.join(OUT, "kaub_stat_robustness.csv"), index=False)
    return t


if __name__ == "__main__":
    df = build_panel(save=False)
    peaks = {m: v["tonnes"] for m, v in observed_peaks(df).items()}
    demand, cc = representative_demand(df)
    print(f"Representative demand (median of {len(cc)} complete months): "
          f"{demand/1e6:.2f} Mt/month\n")

    grid = run_grid(peaks, demand)
    print(f"Scenario grid: {len(grid)} rows "
          f"({len(P.CARGOES)} cargo × {len(P.HEADROOM_SWEEP)} headroom × "
          f"{len(P.KWATER_TIERS)} water × {len(P.KDB_SCENARIOS)} rail × "
          f"{len(P.LAMBDAS)} λ)")
    nz = (grid["road_share"] > 1e-6).mean()
    print(f"  scenarios with any road spillover: {nz*100:.0f}%")
    # confirm λ doesn't change the split
    piv = grid.groupby(["water_tier", "rail_scenario", "headroom", "cargo"])["road_share"].nunique()
    print(f"  distinct road-shares across the 4 λ values (should be 1): "
          f"max={piv.max()}  → λ {'does NOT' if piv.max()==1 else 'DOES'} change the split\n")

    cct = clean_capacity_table(peaks)
    print("Clean-capacity table (headroom 15%):")
    with pd.option_context("display.width", 160):
        print(cct.round(2).to_string(index=False))

    bet = break_even_table(peaks, demand)
    print(f"\nBreak-even thresholds at representative demand {demand/1e6:.2f} Mt:")
    with pd.option_context("display.width", 160):
        print(bet.to_string(index=False))

    decision_map(peaks, demand)
    print("\n✓ scenario_grid.csv, clean_capacity.csv, break_even.csv, decision_map.csv written.")
