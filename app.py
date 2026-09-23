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
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Rhine Forecast & Modal Shift", layout="wide")

# Dark mode toggle
mode = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if mode == "Light":
    st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #f7f9fc 0%, #e8f0ff 100%); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1f3864 0%, #2a4f8a 100%); }
    section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
    h1, h2, h3, h4 { color: #1f3864; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(31,56,100,0.15);
        text-align: center;
        border-top: 4px solid #4e79a7;
    }
    .metric-label { color: #6b7a90; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #1f3864; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .metric-sub { color: #8a97a8; font-size: 0.78rem; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body { background: radial-gradient(circle at top, #1b1f3b 0%, #05060a 60%); color: #e8edf5; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #05060a 0%, #1b1f3b 100%); }
    section[data-testid="stSidebar"] * { color: #e8edf5 !important; }
    h1, h2, h3, h4 { color: #e8edf5; font-weight: 700; }
    .metric-card {
        background: linear-gradient(135deg, #141726 0%, #252a3f 100%);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.6);
        text-align: center;
        border-top: 4px solid #ff4b4b;
    }
    .metric-label { color: #b0b7c9; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .metric-sub { color: #9aa3b8; font-size: 0.78rem; }
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
# UI TABS
# ============================================================
st.title("🚢 Rhine Corridor — Forecast Comparison & Modal-Shift Optimiser")

tab_comp, tab_naive, tab_lr, tab_rf, tab_xgb, tab_opt, tab_heat, tab_sim = st.tabs(
    ["📊 Model comparison", "Naive", "Linear Regression", "Random Forest", "XGBoost",
     "⚙️ Optimiser", "🔥 Scenario heatmap", "🧪 Multi-scenario simulation"]
)

# ---------- COMPARISON TAB ----------
with tab_comp:
    st.subheader("Model comparison table")
    st.dataframe(results_df.round(4), use_container_width=True)

    st.subheader("Metric comparison charts")
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    metrics_list = ["MAE", "RMSE", "MAPE (%)", "R²"]
    colors = ['#4e79a7','#f1a340','#e15759','#59a14f']

    for i, m in enumerate(metrics_list):
        r = i // 2
        c = i % 2
        ax[r, c].bar(results_df["Model"], results_df[m], color=colors)
        ax[r, c].set_title(m, fontsize=12, fontweight='bold')
        ax[r, c].tick_params(axis='x', rotation=30)
        ax[r, c].grid(alpha=0.3)

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
def model_tab(name, preds, color):
    st.subheader(f"{name} — performance")
    m = metrics(name, y_test, preds)
    st.dataframe(pd.DataFrame([m]).round(4))
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.date.iloc[split:], y_test, label="Actual", linewidth=3, color='black')
    ax.plot(df.date.iloc[split:], preds, label=name, linestyle='--', color=color, linewidth=2)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with tab_naive:
    model_tab("Naive (lag-12)", pred_naive, "#e15759")

with tab_lr:
    model_tab("Linear Regression", pred_lr, "#4e79a7")

with tab_rf:
    model_tab("Random Forest", pred_rf, "#f1a340")

with tab_xgb:
    model_tab("XGBoost", pred_xgb, "#59a14f")

# ---------- OPTIMISER TAB ----------
with tab_opt:
    st.subheader("Scenario-based modal-shift optimisation (Rotterdam–Basel corridor)")

    col_left, col_right = st.columns([2,1])

    with col_right:
        st.markdown("### Scenario inputs")
        kaub_level = st.slider("Kaub water level (cm)", 0, 350, 120)
        rail_disruption = st.checkbox("DB Generalsanierung (Rotterdam–Basel rail works)")
        kdb_input = st.slider("Rail capacity coefficient (DB disruption)", 0.1, 1.0, 0.35, step=0.05)

        st.markdown("### Demand selection")
        demand_source = st.radio(
            "Choose demand source:",
            ["Linear Regression forecast", "Naive forecast", "Manual input"],
            horizontal=True
        )

        if demand_source == "Linear Regression forecast":
            demand_mt = float(pred_lr[-1])
            st.success(f"Using Linear Regression forecast: **{demand_mt:.2f} Mt**")
        elif demand_source == "Naive forecast":
            demand_mt = float(pred_naive[-1])
            st.info(f"Using Naive forecast: **{demand_mt:.2f} Mt**")
        else:
            demand_mt = st.number_input(
                "Enter monthly demand (in million tonnes):",
                min_value=5.0,
                max_value=25.0,
                value=13.0,
                step=0.1
            )
            st.warning(f"Using manual demand: **{demand_mt:.2f} Mt**")

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Demand used</div>
            <div class="metric-value">{demand_mt:.2f} Mt</div>
            <div class="metric-sub">Source: {demand_source}</div>
        </div>
        """, unsafe_allow_html=True)

        max_cap_mt = (Cap0['barge'] + Cap0['rail']) / 1e6
        demand_pct = 100 * demand_mt / max_cap_mt

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Demand vs corridor capacity</div>
            <div class="metric-value">{demand_pct:.1f}%</div>
            <div class="metric-sub">Max corridor capacity: {max_cap_mt:.1f} Mt</div>
        </div>
        """, unsafe_allow_html=True)

    with col_left:
        kw = kappa_water(kaub_level)
        kdb = kdb_input if rail_disruption else 1.0

        with st.spinner("Running optimiser..."):
            al, cost, co2 = optimise(demand_mt * 1e6, kw, kdb, Cap0)

        shares = {m: 100 * al[m] / (demand_mt * 1e6) for m in E}

        st.info(f"""
        ### Scenario summary
        - **Kaub water level:** {kaub_level} cm → barge capacity = {kw*100:.0f}%
        - **Rail capacity (DB disruption):** {kdb*100:.0f}%
        - **Demand used:** {demand_mt:.2f} Mt
        """)

        # Metric cards
        st.markdown("### Key metrics")
        c1, c2, c3 = st.columns(3)

        c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Barge share</div>
            <div class="metric-value">{shares['barge']:.1f}%</div>
            <div class="metric-sub">{al['barge']/1e6:.2f} Mt</div>
        </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rail share</div>
            <div class="metric-value">{shares['rail']:.1f}%</div>
            <div class="metric-sub">{al['rail']/1e6:.2f} Mt</div>
        </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Road share</div>
            <div class="metric-value">{shares['road']:.1f}%</div>
            <div class="metric-sub">{al['road']/1e6:.2f} Mt</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        c4, c5 = st.columns(2)
        c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total cost</div>
            <div class="metric-value">{cost/1e6:.1f} M€</div>
            <div class="metric-sub">Includes CO₂ penalty</div>
        </div>
        """, unsafe_allow_html=True)

        c5.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total CO₂</div>
            <div class="metric-value">{co2:.1f} kt</div>
            <div class="metric-sub">All modes combined</div>
        </div>
        """, unsafe_allow_html=True)

        # CO₂ savings vs baseline (no disruption, high water)
        baseline_al, baseline_cost, baseline_co2 = optimise(
            demand_mt * 1e6,
            kw=1.0,
            kdb=1.0,
            Cap0=Cap0
        )
        co2_savings = baseline_co2 - co2

        st.markdown("---")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CO₂ difference vs baseline</div>
            <div class="metric-value">{co2_savings:.1f} kt</div>
            <div class="metric-sub">Positive = more emissions than baseline</div>
        </div>
        """, unsafe_allow_html=True)

        # Percentage bar
        st.markdown("### Modal split (percentage bar)")
        fig, ax = plt.subplots(figsize=(8,1.4))
        left = 0
        colors = {'barge':'#4e79a7', 'rail':'#f1a340', 'road':'#e15759'}

        for m in ['barge','rail','road']:
            ax.barh(0, shares[m], left=left, color=colors[m])
            ax.text(left + shares[m]/2, 0, f"{m.capitalize()} {shares[m]:.1f}%", 
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            left += shares[m]

        ax.set_xlim(0,100)
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("### Detailed allocation (tonnes)")
        st.dataframe(
            pd.DataFrame({
                "Mode": ["Barge","Rail","Road"],
                "Tonnes": [al['barge'], al['rail'], al['road']],
                "Share (%)": [shares['barge'], shares['rail'], shares['road']]
            }).round(2),
            use_container_width=True
        )

# ---------- SCENARIO HEATMAP TAB ----------
with tab_heat:
    st.subheader("Scenario comparison heatmap (CO₂ vs Kaub & DB coefficient)")

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
            progress.progress(done/total)

    fig_h, ax_h = plt.subplots(figsize=(8,4))
    sns.heatmap(co2_matrix, annot=True, fmt=".1f",
                xticklabels=[f"{x:.2f}" for x in kdb_values],
                yticklabels=[str(x) for x in kaub_values],
                cmap="magma", ax=ax_h)
    ax_h.set_xlabel("Rail capacity coefficient (DB)")
    ax_h.set_ylabel("Kaub water level (cm)")
    ax_h.set_title(f"CO₂ (kt) for demand {demand_mt_heat:.1f} Mt")
    st.pyplot(fig_h)

# ---------- MULTI-SCENARIO SIM TAB ----------
with tab_sim:
    st.subheader("Multi-scenario simulation")

    n_scen = st.slider("Number of random scenarios", 5, 50, 15, 1)
    demand_base = st.slider("Base demand (Mt)", 5.0, 25.0, 13.0, 0.5)

    st.write("Randomly varying Kaub level and DB coefficient around realistic ranges.")

    scenarios = []
    progress2 = st.progress(0)

    for k in range(n_scen):
        kv = np.random.uniform(30, 220)
        kdbv = np.random.uniform(0.2, 1.0)
        demand_s = np.random.uniform(demand_base*0.8, demand_base*1.2)

        kw = kappa_water(kv)
        al_s, cost_s, co2_s = optimise(demand_s * 1e6, kw, kdbv, Cap0)

        scenarios.append({
            "Kaub_cm": kv,
            "Rail_coeff": kdbv,
            "Demand_Mt": demand_s,
            "Cost_M€": cost_s/1e6,
            "CO2_kt": co2_s,
            "Road_share_%": 100 * al_s['road']/(demand_s*1e6)
        })
        progress2.progress((k+1)/n_scen)

    scen_df = pd.DataFrame(scenarios)
    st.dataframe(scen_df.round(2), use_container_width=True)

    st.subheader("CO₂ vs road share")
    fig_s, ax_s = plt.subplots(figsize=(7,4))
    ax_s.scatter(scen_df["Road_share_%"], scen_df["CO2_kt"], c=scen_df["Kaub_cm"], cmap="viridis")
    ax_s.set_xlabel("Road share (%)")
    ax_s.set_ylabel("CO₂ (kt)")
    ax_s.set_title("Random scenarios: CO₂ vs road share (colour = Kaub level)")
    st.pyplot(fig_s)
