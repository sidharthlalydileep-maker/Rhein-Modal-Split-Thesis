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

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(page_title="Rhine Forecast & Modal Shift", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f7f9fc 0%, #e8f0ff 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f3864 0%, #2a4f8a 100%);
}
section[data-testid="stSidebar"] * {
    color: #e8edf5 !important;
}
h1, h2, h3, h4 {
    color: #1f3864;
    font-weight: 700;
}
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(31,56,100,0.15);
    text-align: center;
    border-top: 4px solid #4e79a7;
}
.metric-label {
    color: #6b7a90;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.metric-value {
    color: #1f3864;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 4px;
}
.metric-sub {
    color: #8a97a8;
    font-size: 0.78rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("corridor_demand_monthly.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
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

# ============================================================
# TRAIN MODELS
# ============================================================
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

# ============================================================
# OPTIMISER SETUP
# ============================================================
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

# ============================================================
# FORECAST HORIZON (MONTH / QUARTER / YEAR)
# ============================================================
def forecast_months_ahead(model, X_last, months):
    X_future = X_last.copy()
    preds = []
    for i in range(months):
        X_future["t"] += 1
        X_future["month"] = ((X_future["month"] % 12) + 1)
        X_future["quarter"] = ((X_future["month"] - 1)//3) + 1
        pred = model.predict(pd.DataFrame([X_future]))[0]
        preds.append(pred)
    return preds

X_last = df[features].iloc[-1]
