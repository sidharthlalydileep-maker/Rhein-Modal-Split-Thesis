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

# ---------------------------------------------------------
# BASIC PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Rhine Forecast and Modal Shift", layout="wide")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("corridor_demand_monthly.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
df["y"] = df.total_demand_tonnes / 1e6  # million tonnes
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

# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
with st.spinner("Training models..."):
    # Naive (lag-12)
    pred_naive = df.lag12.iloc[split:].values

    # Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    # XGBoost
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
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred)
    }

results = [
    metrics("Naive", y_test, pred_naive),
    metrics("Linear Regression", y_test, pred_lr),
    metrics("Random Forest", y_test, pred_rf),
    metrics("XGBoost", y_test, pred_xgb)
]
results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# MODAL SHIFT PARAMETERS
# ---------------------------------------------------------
E = {"road": 81.7, "rail": 12.8, "barge": 32.1}      # g/tkm
C = {"road": 0.08, "rail": 0.035, "barge": 0.025}    # EUR/tkm
DIST = {"road": 745.0, "rail": 765.0, "barge": 846.0}
LAM = 180.0
HEADROOM = 1.15

cost_pt = {m: C[m] * DIST[m] for m in E}             # EUR/t
co2_pt = {m: E[m] * DIST[m] / 1000 for m in E}       # kg/t -> kg/tkm*km/1000

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

    p = pulp.LpProblem("modal_shift", pulp.LpMinimize)
    x = {m: pulp.LpVariable(m, 0, cap[m]) for m in E}

    p += pulp.lpSum(obj[m] * x[m] for m in E)
    p += pulp.lpSum(x[m] for m in E) == D

    p.solve(pulp.PULP_CBC_CMD(msg=0))

    al = {m: x[m].value() for m in E}
    total_cost = sum(cost_pt[m] * al[m] for m in E)
    total_co2 = sum(co2_pt[m] * al[m] for m in E) / 1000  # kt

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

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("Rhine Corridor Forecast and Modal Shift Optimiser")

tab_comp, tab_opt = st.tabs(["Model comparison", "Optimiser"])

# ---------------------------------------------------------
# MODEL COMPARISON TAB
# ---------------------------------------------------------
with tab_comp:
    st.subheader("Model comparison table")
    st.dataframe(results_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    metrics_list = ["MAE", "RMSE", "MAPE (%)", "R2"]
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

# ---------------------------------------------------------
# OPTIMISER TAB
# ---------------------------------------------------------
with tab_opt:
    st.subheader("Time period selection")

    time_mode = st.radio(
        "Select time period:",
        ["Single month", "Quarter", "Full year"],
        horizontal=True
    )

    st.subheader("Scenario inputs")
    kaub_level = st.slider("Kaub water level (cm)", 0, 350, 120)
    rail_disruption = st.checkbox("Rail disruption (DB Generalsanierung)")
    kdb_input = st.slider("Rail capacity coefficient", 0.1, 1.0, 0.35, step=0.05)

    # -----------------------------------------------------
    # TIME + DEMAND LOGIC
    # -----------------------------------------------------
    if time_mode == "Single month":
        sel_year = st.number_input("Year", 2026, 2035, 2026)
        sel_month = st.selectbox("Month", list(range(1, 13)))
        months_ahead = (sel_year - df.date.iloc[-1].year) * 12 + (sel_month - df.date.iloc[-1].month)
    elif time_mode == "Quarter":
        sel_year = st.number_input("Year", 2026, 2035, 2026)
        sel_quarter = st.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
        q_map = {"Q1": [1, 2, 3], "Q2": [4, 5, 6], "Q3": [7, 8, 9], "Q4": [10, 11, 12]}
        months_ahead = 1  # we will average over quarter
    else:
        sel_year = st.number_input("Year", 2026, 2035, 2026)
        months_ahead = 1  # we will sum over year

    # -----------------------------------------------------
    # DEMAND OPTIONS (ALL MODELS)
    # -----------------------------------------------------
    st.subheader("Demand options (Naive, LR, RF, XGBoost)")

    # LR forecast (for selected horizon)
    ma = max(months_ahead, 1)
    d_lr = forecast_months_ahead(lr, X_last, ma)[-1]

    # Naive forecast (last value)
    d_naive = float(pred_naive[-1])

    # RF forecast (one-step ahead proxy)
    d_rf = rf.predict(pd.DataFrame([X_last]))[0]

    # XGB forecast (one-step ahead proxy)
    d_xgb = xgb.predict(pd.DataFrame([X_last]))[0]

    df_demand = pd.DataFrame({
        "Model": ["Linear Regression", "Naive (lag-12)", "Random Forest", "XGBoost"],
        "Forecast (Mt)": [d_lr, d_naive, d_rf, d_xgb]
    })

    st.dataframe(df_demand.round(2), use_container_width=True)

    # -----------------------------------------------------
    # CHOOSE DEMAND SOURCE
    # -----------------------------------------------------
    st.subheader("Select demand source for optimisation")

    demand_source = st.radio(
        "Demand source:",
        ["Linear Regression", "Naive", "Random Forest", "XGBoost", "Manual input"],
        horizontal=True
    )

    if demand_source == "Linear Regression":
        demand_mt = d_lr
    elif demand_source == "Naive":
        demand_mt = d_naive
    elif demand_source == "Random Forest":
        demand_mt = d_rf
    elif demand_source == "XGBoost":
        demand_mt = d_xgb
    else:
        demand_mt = st.number_input(
            "Enter demand (Mt):",
            min_value=5.0,
            max_value=300.0,
            value=13.0,
            step=0.1
        )

    st.write("Demand used for optimisation: {:.2f} Mt (source: {})".format(demand_mt, demand_source))

    # -----------------------------------------------------
    # RUN OPTIMISER
    # -----------------------------------------------------
    kw = kappa_water(kaub_level)
    kdb = kdb_input if rail_disruption else 1.0

    with st.spinner("Running optimiser..."):
        al, cost, co2 = optimise(demand_mt * 1e6, kw, kdb, Cap0)

    shares = {m: 100 * al[m] / (demand_mt * 1e6) for m in E}

    # -----------------------------------------------------
    # COST AND CO2 NOTIFICATIONS
    # -----------------------------------------------------
    st.subheader("Cost and CO2 summary")

    st.info("Total cost: {:.2f} million EUR".format(cost / 1e6))
    st.info("Total CO2 emissions: {:.2f} kt".format(co2))

    baseline_al, baseline_cost, baseline_co2 = optimise(
        demand_mt * 1e6,
        kw=1.0,
        kdb=1.0,
        Cap0=Cap0
    )

    co2_diff = co2 - baseline_co2

    if co2_diff > 0:
        st.warning("CO2 increased by {:.2f} kt compared to baseline.".format(co2_diff))
    else:
        st.success("CO2 decreased by {:.2f} kt compared to baseline.".format(abs(co2_diff)))

    # -----------------------------------------------------
    # SUMMARY OF MODAL SPLIT
    # -----------------------------------------------------
    st.subheader("Modal split summary")

    st.write("Barge: {:.2f} Mt ({:.1f}%)".format(al["barge"] / 1e6, shares["barge"]))
    st.write("Rail: {:.2f} Mt ({:.1f}%)".format(al["rail"] / 1e6, shares["rail"]))
    st.write("Road: {:.2f} Mt ({:.1f}%)".format(al["road"] / 1e6, shares["road"]))

    # -----------------------------------------------------
    # PROFESSIONAL GRAPHS (PERCENTAGE, TONNES, PIE)
    # -----------------------------------------------------
    st.subheader("Modal split – percentage")

    fig_perc, ax_perc = plt.subplots(figsize=(8, 3))
    modes = ["barge", "rail", "road"]
    colors = ["#4e79a7", "#f1a340", "#e15759"]
    values = [shares[m] for m in modes]

    ax_perc.barh(modes, values, color=colors)
    for i, v in enumerate(values):
        ax_perc.text(v + 1, i, "{:.1f}%".format(v), va="center")
    ax_perc.set_xlim(0, 100)
    ax_perc.set_xlabel("Percentage")
    st.pyplot(fig_perc)

    st.subheader("Modal split – tonnes")

    fig_tonnes, ax_tonnes = plt.subplots(figsize=(8, 3))
    tonnes = [al[m] / 1e6 for m in modes]

    ax_tonnes.bar(modes, tonnes, color=colors)
    for i, v in enumerate(tonnes):
        ax_tonnes.text(i, v + 0.1, "{:.2f} Mt".format(v), ha="center")
    ax_tonnes.set_ylabel("Million tonnes")
    st.pyplot(fig_tonnes)

    st.subheader("Modal split – pie chart")

    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    ax_pie.pie(values, labels=modes, autopct="%1.1f%%", colors=colors)
    st.pyplot(fig_pie)
