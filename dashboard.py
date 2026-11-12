"""
dashboard.py — Interactive Streamlit dashboard for the Rhine-corridor
modal-shift model. Covers every component of the thesis proposal:

  • Scenario optimiser   — cargo, Kaub level, rail state + single-track fraction,
                           demand, carbon price λ, capacity headroom → optimal
                           split, cost, CO₂, savings vs all-road baseline (RQ1),
                           decision map (current scenario marked), break-even.
  • Disruptions (RQ2)    — clean-capacity table, water & rail sweeps, break-even
                           across headroom, λ non-tipping demonstration.
  • Forecasting (RQ3)    — four models, accuracy metrics, decision test
                           (does a better forecast change the decision?).
  • Data & validation    — assembled panel, low-water events, Kaub aggregation
                           robustness, source provenance.

Run with:   streamlit run dashboard.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import params as P
from optimiser import optimise, capacities
from data_loader import build_panel, observed_peaks
from scenarios import (break_even_table, decision_map, representative_demand,
                       clean_capacity_table, kaub_stat_robustness)
from forecasting import run_forecasts, decision_test

st.set_page_config(page_title="Rhine Modal-Shift Optimiser", layout="wide")

RAIL = ["#2a7fb8", "#5bb07f", "#c05a5a"]   # rail, barge, road


# --------------------------------------------------------------------------- #
# Cached heavy computations
# --------------------------------------------------------------------------- #
@st.cache_data
def load():
    df = build_panel(save=False)
    peaks = {m: v["tonnes"] for m, v in observed_peaks(df).items()}
    demand, cc = representative_demand(df)
    return df, peaks, demand, len(cc)


@st.cache_data
def forecasting_results():
    df, peaks, _, _ = load()
    preds, metrics = run_forecasts(df)
    dt, summ = decision_test(df, preds, peaks)
    return preds, metrics, dt, summ


def all_road(cargo, demand):
    """All-road baseline cost (€) and CO₂ (t) for RQ1 comparison."""
    cost = P.COST[cargo]["road"] * P.DIST["road"] * demand
    co2 = P.EMIS["road"] * P.DIST["road"] * demand / 1e6
    return cost, co2


df, peaks, med_demand, n_complete = load()

st.title("Decarbonising Rhine-Corridor Freight — Modal-Shift Optimiser")
st.caption(f"Disruption-aware cost- and carbon-efficient allocation across rail, "
           f"inland waterway and road · Rotterdam–Basel corridor · "
           f"{n_complete} complete months {df['date'].min():%Y-%m}–{df['date'].max():%Y-%m}")

# --------------------------------------------------------------------------- #
# Sidebar — scenario controls (every lever from the proposal)
# --------------------------------------------------------------------------- #
s = st.sidebar
s.header("Scenario controls")

cargo = s.selectbox("Cargo type", P.CARGOES)

s.markdown("**Rhine low water**")
kaub = s.slider("Kaub water level (cm)", 0, 400, 150, 1)
kw = P.kappa_water(kaub)
s.caption(f"→ κ_water = **{kw:.2f}** · remaining barge capacity")

s.markdown("**Rail works**")
rail_state = s.radio("Rail state", ["Normal (two-track)", "Single-track", "Full closure"],
                     index=0)
if rail_state == "Normal (two-track)":
    kdb = 1.00
elif rail_state == "Full closure":
    kdb = 0.00
else:
    kdb = s.slider("Single-track remaining fraction κ_DB", 0.25, 0.70, P.KDB_REF, 0.05,
                   help="The single-track freight-capacity fraction is an assumption "
                        "(reference 0.35); examined as a sweep 0.25–0.70.")
s.caption(f"→ κ_DB = **{kdb:.2f}** · remaining rail capacity")

s.markdown("**Demand & economics**")
demand_mt = s.slider("Monthly demand (Mt)", 5.0, 22.0, round(med_demand / 1e6, 1), 0.1)
demand = demand_mt * 1e6
s.caption(f"Representative (median) demand = {med_demand/1e6:.2f} Mt")

lam_name = s.selectbox("Carbon price λ", list(P.LAMBDAS.keys()),
                       index=list(P.LAMBDAS).index("Model B central (€180)"))
lam = P.LAMBDAS[lam_name]

headroom = s.select_slider("Capacity headroom", P.HEADROOM_SWEEP, value=P.HEADROOM_BASE,
                           format_func=lambda h: f"{int(h*100)}%")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_opt, tab_dis, tab_fc, tab_data = st.tabs(
    ["🎯 Scenario optimiser", "⚠️ Disruptions (RQ2)",
     "🔮 Forecasting (RQ3)", "📊 Data & validation"])

# =========================================================================== #
# TAB 1 — Scenario optimiser
# =========================================================================== #
with tab_opt:
    r = optimise(cargo, kw, kdb, demand, lam, peaks, headroom)
    base_cost, base_co2 = all_road(cargo, demand)
    cost_saving = (base_cost - r.cost_eur) / base_cost * 100 if base_cost else 0
    co2_saving = (base_co2 - r.co2_t) / base_co2 * 100 if base_co2 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rail share", f"{r.shares['rail']*100:.1f}%")
    c2.metric("Barge share", f"{r.shares['barge']*100:.1f}%")
    c3.metric("Road share", f"{r.shares['road']*100:.1f}%",
              delta=f"{r.road_spill/1e6:.2f} Mt spill" if r.road_spill > 1 else "no spill",
              delta_color="inverse")
    cap = capacities(kw, kdb, peaks, headroom)
    clean = cap["barge"] + cap["rail"]
    c4.metric("Clean capacity", f"{clean/1e6:.1f} Mt",
              delta="covers demand" if clean >= demand else "below demand",
              delta_color="normal" if clean >= demand else "inverse")

    st.markdown(f"#### Saving vs an all-road baseline  (RQ1 · {cargo}, this scenario)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Cost", f"€{r.cost_eur/1e6:.0f} M", delta=f"−{cost_saving:.1f}% vs all-road")
    d2.metric("CO₂", f"{r.co2_t/1e3:.0f} kt", delta=f"−{co2_saving:.1f}% vs all-road")
    d3.metric("All-road cost", f"€{base_cost/1e6:.0f} M")
    d4.metric("All-road CO₂", f"{base_co2/1e3:.0f} kt")

    left, right = st.columns(2)
    with left:
        st.subheader("Optimal modal split")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        modes = ["rail", "barge", "road"]
        vals = [r.alloc[m] / 1e6 for m in modes]
        ax.bar(modes, vals, color=RAIL)
        ax.set_ylabel("Mt / month")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig)

    with right:
        st.subheader("Decision map — road share (%)")
        dm = decision_map(peaks, demand, cargo, lam, headroom, n=25, save=False)
        piv = dm.pivot(index="kappa_db", columns="kappa_water", values="road_share") * 100
        fig2, ax2 = plt.subplots(figsize=(5, 3.4))
        im = ax2.imshow(piv.values, origin="lower", aspect="auto", cmap="YlOrRd",
                        extent=[0.15, 1.0, 0.0, 1.0], vmin=0, vmax=100)
        ax2.scatter([kw], [kdb], c="black", s=80, marker="x", linewidths=2)
        ax2.set_xlabel("κ_water (barge)")
        ax2.set_ylabel("κ_DB (rail)")
        fig2.colorbar(im, ax=ax2, label="road %")
        st.pyplot(fig2)
        st.caption("× marks the current scenario.")

    st.subheader("Break-even thresholds at this demand")
    bet = break_even_table(peaks, demand, save=False)
    st.dataframe(bet, use_container_width=True, hide_index=True)

# =========================================================================== #
# TAB 2 — Disruptions (RQ2)
# =========================================================================== #
with tab_dis:
    st.subheader("Clean capacity by disruption tier (this headroom)")
    cct = clean_capacity_table(peaks, headroom, save=False)
    st.dataframe(cct.round(2), use_container_width=True, hide_index=True)

    colw, colr = st.columns(2)
    # water sweep: road share vs Kaub level
    with colw:
        st.subheader("Water sweep — road share vs Kaub level")
        levels = np.arange(0, 201, 5)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for cg in P.CARGOES:
            rs = [optimise(cg, P.kappa_water(l), 1.0, demand, lam, peaks, headroom)
                  .shares["road"] * 100 for l in levels]
            ax.plot(levels, rs, label=cg)
        for b in (44, 72, 134):
            ax.axvline(b, color="grey", ls=":", lw=0.8)
        ax.axvline(kaub, color="black", ls="--", lw=1.2, label="current")
        ax.set_xlabel("Kaub level (cm)  ·  rail normal")
        ax.set_ylabel("road share %")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    # rail sweep: road share vs κ_DB
    with colr:
        st.subheader("Rail sweep — road share vs κ_DB")
        kdbs = np.linspace(0, 1, 41)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for cg in P.CARGOES:
            rs = [optimise(cg, 1.0, k, demand, lam, peaks, headroom)
                  .shares["road"] * 100 for k in kdbs]
            ax.plot(kdbs, rs, label=cg)
        ax.axvline(kdb, color="black", ls="--", lw=1.2, label="current")
        ax.set_xlabel("κ_DB  ·  water normal")
        ax.set_ylabel("road share %")
        ax.legend(fontsize=7)
        st.pyplot(fig)

    st.subheader("Break-even across headroom (0 / 15 / 30%)")
    st.dataframe(break_even_table(peaks, demand, save=False),
                 use_container_width=True, hide_index=True)

    # λ non-tipping demonstration
    st.subheader("Does the carbon price change the split?")
    rows = []
    for ln, lv in P.LAMBDAS.items():
        rr = optimise(cargo, kw, kdb, demand, lv, peaks, headroom)
        rows.append(dict(**{"λ": ln},
                         rail=f"{rr.shares['rail']*100:.1f}%",
                         barge=f"{rr.shares['barge']*100:.1f}%",
                         road=f"{rr.shares['road']*100:.1f}%",
                         cost=f"€{rr.cost_eur/1e6:.0f}M", CO2=f"{rr.co2_t/1e3:.0f}kt"))
    lam_df = pd.DataFrame(rows)
    nunique = lam_df[["rail", "barge", "road"]].drop_duplicates().shape[0]
    st.dataframe(lam_df, use_container_width=True, hide_index=True)
    st.info(f"The split is **{'identical' if nunique==1 else 'different'}** across "
            f"€0–640/t → carbon price is **{'non-tipping' if nunique==1 else 'tipping'}** "
            "on this corridor: rail and barge are already cheaper than road on line-haul "
            "cost alone, so λ values the climate benefit rather than triggering the shift.")

# =========================================================================== #
# TAB 3 — Forecasting (RQ3)
# =========================================================================== #
with tab_fc:
    st.subheader("Demand forecasting — four models")
    with st.spinner("Training forecasts…"):
        preds, metrics, dt, summ = forecasting_results()

    md = pd.DataFrame(metrics).T
    md["MAE (Mt)"] = md["MAE"] / 1e6
    md = md[["MAE (Mt)", "RMSE", "MAPE"]].round({"MAE (Mt)": 2, "RMSE": 0, "MAPE": 2})
    best = min(metrics, key=lambda m: metrics[m]["MAE"])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Accuracy (held-out test months, time-ordered split)**")
        st.dataframe(md, use_container_width=True)
        st.caption(f"Best by MAE: **{best.replace('_',' ')}** "
                   f"({metrics[best]['MAE']/1e6:.2f} Mt). "
                   f"Test window {preds['date'].min():%Y-%m}–{preds['date'].max():%Y-%m} "
                   f"({len(preds)} months). Features are strictly lagged — no look-ahead.")
    with c2:
        st.markdown("**Forecast vs observed demand**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(pd.to_datetime(preds["date"]), preds["observed"] / 1e6, "k-", lw=1.5, label="observed")
        ax.plot(pd.to_datetime(preds["date"]), preds[best] / 1e6, "--", color="#2a7fb8",
                lw=1.3, label=best.replace("_", " "))
        ax.set_ylabel("Mt / month"); ax.legend(fontsize=8)
        st.pyplot(fig)

    st.subheader("RQ3 — does a better forecast change the DECISION?")
    show = summ.copy()
    show.columns = ["model", "mean |demand err| %", "mean |road Δ| pp",
                    "max |road Δ| pp", "mean |cost Δ| %"]
    st.dataframe(show.round(2), use_container_width=True, hide_index=True)
    st.info("Across all models the modal decision moves only a few percentage points, "
            "even where demand error differs. **The optimiser's decision is robust to the "
            "forecast model** — the ML component is supporting analysis, not a driver.")

# =========================================================================== #
# TAB 4 — Data & validation
# =========================================================================== #
with tab_data:
    cc = df.dropna(subset=["barge_t", "rail_t", "road_t"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Complete months", f"{len(cc)}",
              f"{cc['date'].min():%Y-%m}–{cc['date'].max():%Y-%m}")
    c2.metric("Barge peak", f"{peaks['barge']/1e6:.2f} Mt")
    c3.metric("Rail peak", f"{peaks['rail']/1e6:.2f} Mt")

    st.subheader("Monthly modal volumes & Kaub level")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for col, cl, lab in [("barge_t", RAIL[1], "barge"), ("rail_t", RAIL[0], "rail"),
                         ("road_t", RAIL[2], "road")]:
        ax.plot(df["date"], df[col] / 1e6, color=cl, lw=0.9, label=lab)
    ax.set_ylabel("Mt / month"); ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(df["date"], df["kaub_w_min_cm"], color="grey", lw=0.6, alpha=0.6)
    ax2.set_ylabel("Kaub min (cm)", color="grey")
    st.pyplot(fig)

    st.subheader("Low-water validation events")
    events = {"2018 crisis": "2018-10", "2022 event": "2022-08", "2026 record": "2026-07"}
    rows = []
    for name, ym in events.items():
        row = df[df["date"].dt.strftime("%Y-%m") == ym]
        if not row.empty:
            rr = row.iloc[0]
            rows.append(dict(event=name, month=ym, kaub_min_cm=rr["kaub_w_min_cm"],
                             kappa_water=rr["kappa_water"],
                             barge_share=f"{rr['barge_t']/rr['demand_t']*100:.1f}%"
                             if pd.notna(rr["barge_t"]) else "n/a"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Kaub aggregation robustness (min / p10 / median)")
    st.dataframe(kaub_stat_robustness(peaks, save=False),
                 use_container_width=True, hide_index=True)
    st.caption("The water break-even conclusion is stable across the three statistics, "
               "so the conservative monthly-minimum choice does not drive the result.")

    with st.expander("Assembled monthly panel (tail)"):
        st.dataframe(df.tail(18), use_container_width=True, hide_index=True)

st.divider()
st.caption("Sources — costs: Panteia (2023) · emissions: EcoTransIT World / ISO 14083 · "
           "κ_water: CCNR (2023) Act now! Fig. 18 · demand: Destatis 46321/46131/46231 · "
           "water: WSV/PEGELONLINE Kaub · carbon: UBA (2019) & nEHS/BEHG. "
           "15% headroom is the base case (swept 0/15/30%); the single-track κ_DB band is "
           "a scenario assumption (reference 0.35).")
