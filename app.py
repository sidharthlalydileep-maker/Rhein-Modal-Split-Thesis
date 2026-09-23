import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

# ----------------- DATA -----------------
df = pd.read_csv("corridor_demand_monthly.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['y'] = df.total_demand_tonnes / 1e6
df['month'] = df.date.dt.month
df['quarter'] = df.date.dt.quarter
df['t'] = np.arange(len(df))
df['lag1'] = df.y.shift(1)
df['lag12'] = df.y.shift(12)
df['roll3'] = df.y.shift(1).rolling(3).mean()
df = df.dropna().reset_index(drop=True)

features = ['month','quarter','t','lag1','lag12','roll3','kaub_w_min_cm']
split = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df.y.iloc[:split], df.y.iloc[split:]

# ----------------- MODELS -----------------
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
        "RMSE": mean_squared_error(y_true, y_pred)**0.5,
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred)*100,
        "R²": r2_score(y_true, y_pred)
    }

results = [
    metrics("Naive (lag-12)", y_test, pred_naive),
    metrics("Linear Regression", y_test, pred_lr),
    metrics("Random Forest", y_test, pred_rf),
    metrics("XGBoost", y_test, pred_xgb)
]
results_df = pd.DataFrame(results)

# ----------------- OPTIMISER -----------------
E    = {'road':81.7, 'rail':12.8, 'barge':32.1}
C    = {'road':0.08, 'rail':0.035, 'barge':0.025}
DIST = {'road':745.0,'rail':765.0,'barge':846.0}
LAM  = 180.0
HEADROOM = 1.15

cost_pt = {m: C[m]*DIST[m]      for m in E}
co2_pt  = {m: E[m]*DIST[m]/1000 for m in E}

Cap0 = {
    'barge': HEADROOM * df.barge_tonnes.max(),
    'rail':  HEADROOM * df.rail_tonnes.max(),
    'road':  1e12
}

def kappa_water(cm):
    if cm >= 200: return 1.00
    elif cm >= 150: return 0.85
    elif cm >= 80:  return 0.55
    elif cm >= 40:  return 0.30
    elif cm >= 30:  return 0.15
    else:           return 0.05

def optimise(D, kw, kdb, Cap0):
    cap={'barge':kw*Cap0['barge'],'rail':kdb*Cap0['rail'],'road':1e12}
    obj={m:cost_pt[m]+LAM*co2_pt[m]/1000 for m in E}
    p=pulp.LpProblem('x',pulp.LpMinimize)
    x={m:pulp.LpVariable(m,0,cap[m]) for m in E}
    p+=pulp.lpSum(obj[m]*x[m] for m in x)
    p+=pulp.lpSum(x[m] for m in x)==D
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    al={m:x[m].value() for m in x}
    return al, sum(cost_pt[m]*al[m] for m in x), sum(co2_pt[m]*al[m] for m in x)/1000

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Rhine Forecast & Modal Shift", layout="wide")
st.title("🚢 Rhine Corridor — Forecast Comparison & Modal-Shift Optimiser")

tab_comp, tab_naive, tab_lr, tab_rf, tab_xgb, tab_opt = st.tabs(
    ["📊 Model comparison", "Naive", "Linear Regression", "Random Forest", "XGBoost", "⚙️ Optimiser"]
)

# ---------- COMPARISON TAB ----------
with tab_comp:
    st.subheader("Model comparison table")
    st.dataframe(results_df.round(4), use_container_width=True)

    st.subheader("Metric comparison charts")
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    metrics_list = ["MAE", "RMSE", "MAPE (%)", "R²"]
    colors = ['#888','#4e79a7','#f1a340','#e15759']

    for i, m in enumerate(metrics_list):
        r = i // 2
        c = i % 2
        ax[r, c].bar(results_df["Model"], results_df[m], color=colors)
        ax[r, c].set_title(m)
        ax[r, c].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Actual vs predicted — all models")
    fig2, ax2 = plt.subplots(figsize=(12,4))
    ax2.plot(df.date.iloc[split:], y_test, label="Actual", linewidth=3, color='black')
    ax2.plot(df.date.iloc[split:], pred_naive, label="Naive", linestyle='--', color='red')
    ax2.plot(df.date.iloc[split:], pred_lr, label="Linear Regression", linestyle='--', color='blue')
    ax2.plot(df.date.iloc[split:], pred_rf, label="Random Forest", linestyle='--', color='orange')
    ax2.plot(df.date.iloc[split:], pred_xgb, label="XGBoost", linestyle='--', color='green')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# ---------- PER-MODEL TABS ----------
def model_tab(name, preds):
    st.subheader(f"{name} — performance")
    m = metrics(name, y_test, preds)
    st.write(pd.DataFrame([m]).round(4))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.date.iloc[split:], y_test, label="Actual", linewidth=3, color='black')
    ax.plot(df.date.iloc[split:], preds, label=name, linestyle='--', color='blue')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab_naive:
    model_tab("Naive (lag-12)", pred_naive)

with tab_lr:
    model_tab("Linear Regression", pred_lr)

with tab_rf:
    model_tab("Random Forest", pred_rf)

with tab_xgb:
    model_tab("XGBoost", pred_xgb)

# ---------- OPTIMISER TAB ----------
with tab_opt:
    st.subheader("Scenario-based modal-shift optimisation (Rotterdam–Basel corridor)")

    col_left, col_right = st.columns([2,1])

    with col_right:
        st.markdown("**Scenario inputs**")
        kaub_level = st.slider("Kaub water level (cm)", 0, 350, 120)
        rail_disruption = st.checkbox("DB Generalsanierung (Rotterdam–Basel rail works)")
        kdb_input = st.slider("Rail capacity coefficient (DB disruption)", 0.1, 1.0, 0.35, step=0.05)

        demand_source = st.radio(
            "Demand source",
            ["Use Linear Regression forecast", "Use Naive forecast", "Manual input"]
        )

        if demand_source == "Use Linear Regression forecast":
            demand_mt = float(pred_lr[-1])
        elif demand_source == "Use Naive forecast":
            demand_mt = float(pred_naive[-1])
        else:
            demand_mt = st.number_input("Monthly demand (million tonnes)", 5.0, 25.0, 13.0)

        st.markdown(f"**Chosen demand:** {demand_mt:.2f} Mt")

    with col_left:
        kw = kappa_water(kaub_level)
        kdb = kdb_input if rail_disruption else 1.0

        al, cost, co2 = optimise(demand_mt * 1e6, kw, kdb, Cap0)

        st.markdown(f"**Kaub level:** {kaub_level} cm → barge capacity factor = {kw:.2f}")
        st.markdown(f"**Rail capacity factor (Rotterdam–Basel, DB):** {kdb:.2f}")
        st.markdown(f"**Demand used:** {demand_mt:.2f} Mt")

        shares = {m: 100 * al[m] / (demand_mt * 1e6) for m in E}

        st.markdown("### Modal allocation (tonnes)")
        st.write(pd.DataFrame.from_dict(al, orient='index', columns=['tonnes']).round(0))

        st.markdown("### Modal shares (%)")
        st.write(pd.DataFrame.from_dict(shares, orient='index', columns=['share %']).round(1))

        st.markdown("### Cost and CO₂")
        st.write(f"**Total cost:** €{cost:,.0f}")
        st.write(f"**Total CO₂:** {co2:.1f} kt")

        fig3, ax3 = plt.subplots(figsize=(6,3))
        modes = list(E.keys())
        vals = [al[m] for m in modes]
        ax3.bar(modes, vals, color=['#4e79a7','#f1a340','#e15759'])
        ax3.set_title("Modal allocation (tonnes)")
        st.pyplot(fig3)
