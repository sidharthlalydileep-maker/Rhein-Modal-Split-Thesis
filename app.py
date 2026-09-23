import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pulp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

st.set_page_config(page_title="Rhine Forecast and Modal Shift", layout="wide")

df = pd.read_csv("corridor_demand_monthly.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["y"] = df.total_demand_tonnes / 1e6
df["month"] = df.date.dt.month
df["quarter"] = df.date.dt.quarter
df["t"] = np.arange(len(df))
df["lag1"] = df.y.shift(1)
df["lag12"] = df.y.shift(12)
df["roll3"] = df.y.shift(1).rolling(3).mean()
df = df.dropna().reset_index(drop=True)

features = ["month", "quarter", "t", "lag1", "lag12", "roll3", "kaub_w_min_cm"]
split = int(len(df) * 0.8)

X_train = df[features].iloc[:split]
X_test = df[features].iloc[split:]
y_train = df.y.iloc[:split]
y_test = df.y.iloc[split:]

with st.spinner("Training models..."):
    pred_naive = df.lag12.iloc[split:].values

    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=0
    ).fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

def metrics(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred)
    }

results = [
    metrics("Naive", y_test, pred_naive),
    metrics("Linear Regression", y_test, pred_lr),
    metrics("Random Forest", y_test, pred_rf),
    metrics("XGBoost", y_test, pred_xgb)
]

results_df = pd.DataFrame(results)

E = {"road": 81.7, "rail": 12.8, "barge": 32.1}
C = {"road": 0.08, "rail": 0.035, "barge": 0.025}
DIST = {"road": 745.0, "rail": 765.0, "barge": 846.0}
LAM = 180.0
HEADROOM = 1.15

cost_pt = {m: C[m] * DIST[m] for m in E}
co2_pt = {m: E[m] * DIST[m] / 1000 for m in E}

Cap0 = {
    "barge": HEADROOM * df.barge_tonnes.max(),
    "rail": HEADROOM * df.rail_tonnes.max(),
    "road": 1e12
}

def kappa_water(cm):
    if cm >= 200:
        return 1.0
    if cm >= 150:
        return 0.85
    if cm >= 80:
        return 0.55
    if cm >= 40:
        return 0.30
    if cm >= 30:
        return 0.15
    return 0.05

def optimise(D, kw, kdb, Cap0):
    cap = {
        "barge": kw * Cap0["barge"],
        "rail": kdb * Cap0["rail"],
        "road": 1e12
    }
    obj = {m: cost_pt[m] + LAM * co2_pt[m] / 1000 for m in E}

    p = pulp.LpProblem("x", pulp.LpMinimize)
    x = {m: pulp.LpVariable(m, 0, cap[m]) for m in E}

    p += pulp.lpSum(obj[m] * x[m] for m in E)
    p += pulp.lpSum(x[m] for m in E) == D

    p.solve(pulp.PULP_CBC_CMD(msg=0))

    al = {m: x[m].value() for m in E}
    total_cost = sum(cost_pt[m] * al[m] for m in E)
    total_co2 = sum(co2_pt[m] * al[m] for m in E) / 1000

    return al, total_cost, total_co2

def forecast_months_ahead(model, X_last, months):
    X_future = X_last.copy()
    preds = []
    for i in range(months):
        X_future["t"] += 1
        X_future["month"] = (X_future["month"] % 12) + 1
        X_future["quarter"] = ((X_future["month"] - 1) // 3) + 1
        pred = model.predict(pd.DataFrame([X_future]))[0]
        preds.append(pred)
    return preds

X_last = df[features].iloc[-1]

st.title("Rhine Corridor Forecast and Modal Shift Optimiser")

tab_comp, tab_naive, tab_lr, tab_rf, tab_xgb, tab_opt, tab_heat, tab_sim, tab_db = st.tabs(
    ["Model comparison", "Naive", "Linear Regression", "Random Forest", "XGBoost",
     "Optimiser", "Scenario heatmap", "Multi-scenario simulation", "DB disruption source"]
)

with tab_comp:
    st.subheader("Model comparison table")
    st.dataframe(results_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    metrics_list = ["MAE", "RMSE", "MAPE", "R2"]
    colors = ["#4e79a7", "#f1a340", "#e15759", "#59a14f"]

    for i, m in enumerate(metrics_list):
        r = i // 2
        c = i % 2
        ax[r, c].bar(results_df["Model"], results_df[m], color=colors)
        ax[r, c].set_title(m)
        ax[r, c].tick_params(axis="x", rotation=30)
        ax[r, c].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df.date.iloc[split:], y_test, label="Actual", linewidth=3, color="black")
    ax2.plot(df.date.iloc[split:], pred_naive, label="Naive", linestyle="--", color="red")
    ax2.plot(df.date.iloc[split:], pred_lr, label="Linear Regression", linestyle="--", color="blue")
    ax2.plot(df.date.iloc[split:], pred_rf, label="Random Forest", linestyle="--", color="orange")
    ax2.plot(df.date.iloc[split:], pred_xgb, label="XGBoost", linestyle="--", color="green")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

def model_tab(name, preds, color):
    st.subheader(name)
    m = metrics(name, y_test, preds)
    st.dataframe(pd.DataFrame([m]).round(4))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.date.iloc[split:], y_test, label="Actual", linewidth=3, color="black")
    ax.plot(df.date.iloc[split:], preds, label=name, linestyle="--", color=color)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with tab_naive:
    model_tab("Naive", pred_naive, "#e15759")

with tab_lr:
    model_tab("Linear Regression", pred_lr, "#4e79a7")

with tab_rf:
    model_tab("Random Forest", pred_rf, "#f1a340")

with tab_xgb:
    model_tab("XGBoost", pred_xgb, "#59a14f")

with tab_opt:
    st.subheader("Scenario-based modal-shift optimisation")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("### Scenario inputs")
        kaub_level = st.slider("Kaub water level (cm)", 0, 350, 120)
        rail_disruption = st.checkbox("DB Generalsanierung")
        kdb_input = st.slider("Rail capacity coefficient", 0.1, 1.0, 0.35, step=0.05)

        st.markdown("### Time selection")
        time_mode = st.radio(
            "Choose time period:",
            ["Single month", "Quarter", "Full year"],
            horizontal=True
        )

        st.markdown("### Demand selection")
        demand_source = st.radio(
            "Choose demand source:",
            ["Linear Regression forecast", "Naive forecast (last year)", "Manual input"],
            horizontal=True
        )

        if time_mode == "Single month":
            sel_year = st.number_input("Year", 2026, 2035, 2026)
            sel_month = st.selectbox("Month", list(range(1, 13)))
            months_ahead = (sel_year - df.date.iloc[-1].year) * 12 + (sel_month - df.date.iloc[-1].month)

            if demand_source == "Linear Regression forecast":
                forecast_vals = forecast_months_ahead(lr, X_last, max(months_ahead, 1))
                demand_mt = forecast_vals[-1]
                st.success("LR forecast for {}/{}: {:.2f} Mt".format(sel_month, sel_year, demand_mt))

            elif demand_source == "Naive forecast (last year)":
                mask = (df.date.dt.year == sel_year - 1) & (df.date.dt.month == sel_month)
                if mask.any():
                    demand_mt = df.loc[mask, "y"].iloc[0]
                    st.info("Naive forecast (same month last year): {:.2f} Mt".format(demand_mt))
                else:
                    demand_mt = float(pred_naive[-1])
                    st.warning("No last-year data, using naive fallback: {:.2f} Mt".format(demand_mt))

            else:
                demand_mt = st.number_input(
                    "Enter monthly demand (Mt):",
                    min_value=5.0,
                    max_value=25.0,
                    value=13.0,
                    step=0.1
                )
                st.warning("Manual demand for {}/{}: {:.2f} Mt".format(sel_month, sel_year, demand_mt))

        elif time_mode == "Quarter":
            sel_year = st.number_input("Year", 2026, 2035, 2026)
            sel_quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
            q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}

            if demand_source == "Linear Regression forecast":
                vals = []
                for m in q_map[sel_quarter]:
                    months_ahead = (sel_year - df.date.iloc[-1].year) * 12 + (m - df.date.iloc[-1].month)
                    f = forecast_months_ahead(lr, X_last, max(months_ahead, 1))[-1]
                    vals.append(f)
                demand_mt = np.mean(vals)
                st.success("LR forecast for {} {}: {:.2f} Mt (avg)".format(sel_quarter, sel_year, demand_mt))

            elif demand_source == "Naive forecast (last year)":
                vals = []
                for m in q_map[sel_quarter]:
                    mask = (df.date.dt.year == sel_year - 1) & (df.date.dt.month == m)
                    if mask.any():
                        vals.append(df.loc[mask, "y"].iloc[0])
                if vals:
                    demand_mt = np.mean(vals)
                    st.info("Naive forecast for {} {}: {:.2f} Mt (avg)".format(sel_quarter, sel_year, demand_mt))
                else:
                    demand_mt = float(pred_naive[-1])
                    st.warning("No last-year quarter data, using naive fallback: {:.2f} Mt".format(demand_mt))

            else:
                demand_mt = st.number_input(
                    "Enter avg monthly demand for quarter (Mt):",
                    min_value=5.0,
                    max_value=25.0,
                    value=13.0,
                    step=0.1
                )
                st.warning("Manual avg monthly demand for {} {}: {:.2f} Mt".format(sel_quarter, sel_year, demand_mt))

        else:
            sel_year = st.number_input("Year", 2026, 2035, 2026)

            if demand_source == "Linear Regression forecast":
                vals = []
                for m in range(1, 13):
                    months_ahead = (sel_year - df.date.iloc[-1].year) * 12 + (m - df.date.iloc[-1].month)
                    f = forecast_months_ahead(lr, X_last, max(months_ahead, 1))[-1]
                    vals.append(f)
                demand_mt = sum(vals)
                st.success("LR forecast for full year {}: {:.2f} Mt".format(sel_year, demand_mt))

            elif demand_source == "Naive forecast (last year)":
                mask = (df.date.dt.year == sel_year - 1)
                if mask.any():
                    demand_mt = df.loc[mask, "y"].sum()
                    st.info("Naive forecast for full year {}: {:.2f} Mt".format(sel_year, demand_mt))
                else:
                    demand_mt = float(pred_naive[-1]) * 12
                    st.warning("No last-year data, using naive fallback: {:.2f} Mt".format(demand_mt))

            else:
                demand_mt = st.number_input(
                    "Enter total annual demand (Mt):",
                    min_value=50.0,
                    max_value=300.0,
                    value=150.0,
                    step=1.0
                )
                st.warning("Manual annual demand for {}: {:.2f} Mt".format(sel_year, demand_mt))

        st.write("Demand used: {:.2f} Mt".format(demand_mt))

    with col_left:
        kw = kappa_water(kaub_level)
        kdb = kdb_input if rail_disruption else 1.0

        with st.spinner("Running optimiser..."):
            al, cost, co2 = optimise(demand_mt * 1e6, kw, kdb, Cap0)

        shares = {m: 100 * al[m] / (demand_mt * 1e6) for m in E}

        st.write("Barge share: {:.1f}%".format(shares["barge"]))
        st.write("Rail share: {:.1f}%".format(shares["rail"]))
        st.write("Road share: {:.1f}%".format(shares["road"]))

        st.write("Total cost: {:.2f} million EUR".format(cost / 1e6))
        st.write("Total CO2: {:.2f} kt".format(co2))

with tab_heat:
    st.subheader("Scenario heatmap")

    demand_mt_heat = st.slider("Demand for heatmap (Mt)", 5.0, 25.0, 13.0, 0.5)

    kaub_values = [40, 80, 120, 160, 200]
    kdb_values = [0.2, 0.35, 0.5, 0.75, 1.0]

    co2_matrix = np.zeros((len(kaub_values), len(kdb_values)))

    progress = st.progress(0)
    total = len(kaub_values) * len(kdb_values)
    done = 0

    for i, kv in enumerate(kaub_values):
        for j, kdbv in enumerate(kdb_values):
            kw = kappa_water(kv)
            al_h, cost_h, co2_h = optimise(demand_mt_heat * 1e6, kw, kdbv, Cap0)
            co2_matrix[i, j] = co2_h
            done += 1
            progress.progress(done / total)

    fig_h, ax_h = plt.subplots(figsize=(8, 4))
    sns.heatmap(co2_matrix, annot=True, fmt=".1f",
                xticklabels=[str(x) for x in kdb_values],
                yticklabels=[str(x) for x in kaub_values],
                cmap="magma", ax=ax_h)
    ax_h.set_xlabel("Rail capacity coefficient")
    ax_h.set_ylabel("Kaub water level (cm)")
    ax_h.set_title("CO2 (kt)")
    st.pyplot(fig_h)

with tab_sim:
    st.subheader("Multi-scenario simulation")

    n_scen = st.slider("Number of random scenarios", 5, 50, 15, 1)
    demand_base = st.slider("Base demand (Mt)", 5.0, 25.0, 13.0, 0.5)

    st.write("Randomly varying Kaub level and DB coefficient.")

    scenarios = []
    progress2 = st.progress(0)

    for k in range(n_scen):
        kv = np