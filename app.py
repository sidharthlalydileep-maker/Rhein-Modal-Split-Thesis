# ================================================================
#  Rhine Corridor — Modal-Shift Dashboard (cloud version)
#  Interactive optimiser + analysis results.
#  Deploy on Streamlit Community Cloud.
# ================================================================
import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE, r2_score

# ---------------- PARAMETERS ----------------
E    = {'road':81.7, 'rail':12.8, 'barge':32.1}
C    = {'road':0.08, 'rail':0.035, 'barge':0.025}
DIST = {'road':745.0,'rail':765.0,'barge':846.0}
LAM  = 180.0; KDB = 0.35; HEADROOM = 1.15
cost_pt = {m: C[m]*DIST[m]      for m in E}
co2_pt  = {m: E[m]*DIST[m]/1000 for m in E}
COL = {'road':'#e15759', 'rail':'#f1a340', 'barge':'#4e79a7'}

def kappa_water(cm):
    if   cm >= 134: return 1.00
    elif cm >=  72: return 0.50
    elif cm >=  44: return 0.25
    else:           return 0.15

@st.cache_data
def load_data():
    df = pd.read_csv('corridor_demand_monthly.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return df

def optimise(D, kw, kdb, Cap0):
    cap={'barge':kw*Cap0['barge'],'rail':kdb*Cap0['rail'],'road':1e12}
    obj={m:cost_pt[m]+LAM*co2_pt[m]/1000 for m in E}
    p=pulp.LpProblem('x',pulp.LpMinimize); x={m:pulp.LpVariable(m,0,cap[m]) for m in E}
    p+=pulp.lpSum(obj[m]*x[m] for m in x); p+=pulp.lpSum(x[m] for m in x)==D
    p.solve(pulp.PULP_CBC_CMD(msg=0)); al={m:x[m].value() for m in x}
    return al, sum(cost_pt[m]*al[m] for m in x), sum(co2_pt[m]*al[m] for m in x)/1000

@st.cache_data
def run_forecast(df):
    d=df.copy(); d['y']=d.total_demand_tonnes/1e6
    d['month']=d.date.dt.month; d['t']=np.arange(len(d))
    d['lag1']=d.y.shift(1); d['lag12']=d.y.shift(12); d['roll3']=d.y.shift(1).rolling(3).mean()
    d=d.dropna().reset_index(drop=True)
    F=['month','t','lag1','lag12','roll3','kaub_w_min_cm']; sp=int(len(d)*0.8)
    Xtr,Xte,ytr,yte=d[F].iloc[:sp],d[F].iloc[sp:],d.y.iloc[:sp],d.y.iloc[sp:]
    out={}
    out['Naive']=MAPE(yte,d.lag12.iloc[sp:])*100
    out['Linear']=MAPE(yte,LinearRegression().fit(Xtr,ytr).predict(Xte))*100
    out['RandomForest']=MAPE(yte,RandomForestRegressor(n_estimators=200,random_state=0).fit(Xtr,ytr).predict(Xte))*100
    return out

# ============================== PAGE ==============================
st.set_page_config(page_title="Rhine Modal-Shift Tool", page_icon="🚢", layout="wide")
st.markdown("""<style>
.main{background:#f7f9fc;} .block-container{padding-top:2rem;}
h1{color:#1f3864;font-weight:700;}
.metric-card{background:white;border-radius:14px;padding:16px 18px;box-shadow:0 2px 10px rgba(31,56,100,0.08);text-align:center;border-top:4px solid #4e79a7;}
.metric-label{color:#6b7a90;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;}
.metric-value{color:#1f3864;font-size:1.6rem;font-weight:700;margin-top:4px;}
.metric-sub{color:#8a97a8;font-size:0.78rem;}
section[data-testid="stSidebar"]{background:#1f3864;} section[data-testid="stSidebar"] *{color:#e8edf5 !important;}
</style>""", unsafe_allow_html=True)

st.title("🚢 Rhine Corridor — Modal-Shift Decision Tool")
st.markdown("<p style='color:#6b7a90;margin-top:-10px;'>Cost- and carbon-optimal road / rail / barge allocation under Rhine low-water and rail-renovation disruptions</p>", unsafe_allow_html=True)

# load data + capacities
try:
    df = load_data()
    Cap0 = {'barge':HEADROOM*df.barge_tonnes.max(), 'rail':HEADROOM*df.rail_tonnes.max(), 'road':1e12}
    data_ok = True
except Exception:
    Cap0 = {'barge':11.5e6, 'rail':6.8e6, 'road':1e12}
    data_ok = False

tab1, tab2 = st.tabs(["🎛️ Interactive tool", "📊 Analysis results"])

# ---------- TAB 1: interactive ----------
with tab1:
    st.sidebar.markdown("## ⚙️ Scenario inputs")
    kaub = st.sidebar.slider("Kaub water level (cm)", 0, 350, 300, step=1)
    rail_works = st.sidebar.checkbox("🚧 Rail renovation (DB Generalsanierung)")
    demand_mt = st.sidebar.number_input("Monthly demand (million tonnes)", 5.0, 25.0, 13.2, step=0.5)

    kw = kappa_water(kaub); kdb = 0.35 if rail_works else 1.0
    al, cost, co2 = optimise(demand_mt*1e6, kw, kdb, Cap0)
    D = demand_mt*1e6; shares = {m:100*al[m]/D for m in E}

    def card(col,label,value,sub=""):
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>""",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    card(c1,"Barge capacity",f"{kw*100:.0f}%","of normal")
    card(c2,"Road share",f"{shares['road']:.0f}%",f"{al['road']/1e6:.2f} Mt")
    card(c3,"Total cost",f"€{cost/1e6:.0f}M","per month")
    card(c4,"Total CO₂",f"{co2/1000:.0f} kt","per month")

    st.markdown("<br>",unsafe_allow_html=True)
    left,right=st.columns([3,2])
    with left:
        st.markdown("#### Optimal modal split")
        fig,ax=plt.subplots(figsize=(7,2.6)); fig.patch.set_facecolor('#f7f9fc'); ax.set_facecolor('#f7f9fc')
        lx=0
        for m in ['barge','rail','road']:
            w=shares[m]
            if w>0.3:
                ax.barh(0,w,left=lx,color=COL[m],edgecolor='white',height=0.5)
                ax.text(lx+w/2,0,f"{m}\n{al[m]/1e6:.1f} Mt ({w:.0f}%)",ha='center',va='center',color='white',fontsize=9,fontweight='bold')
            lx+=w
        ax.set_xlim(0,100); ax.set_ylim(-0.5,0.5); ax.axis('off'); st.pyplot(fig)
        for m in ['barge','rail','road']:
            st.markdown(f"<span style='color:{COL[m]};font-size:1.3rem;'>●</span> **{m.capitalize()}** — {al[m]/1e6:.2f} Mt ({shares[m]:.0f}%)",unsafe_allow_html=True)
    with right:
        st.markdown("#### Conditions")
        st.markdown(f"""| | |
|---|---|
| **Kaub level** | {kaub} cm |
| **Barge capacity** | {kw*100:.0f}% |
| **Rail renovation** | {'Yes — 35%' if rail_works else 'No — 100%'} |
| **Demand** | {demand_mt:.1f} Mt/month |""")
        st.markdown("<br>",unsafe_allow_html=True)
        if shares['road']>50: st.error("⚠️ **Over half the freight is on road** — decarbonisation largely reversed.")
        elif shares['road']>10: st.warning("🟠 Some freight has shifted to road due to disruption.")
        else: st.success("✅ Freight stays on low-carbon modes (barge / rail).")

# ---------- TAB 2: analysis ----------
with tab2:
    if not data_ok:
        st.warning("Upload corridor_demand_monthly.csv to the repository to enable the analysis results.")
    else:
        st.markdown("#### Forecast model comparison (test-set error, lower = better)")
        fc = run_forecast(df)
        fig,ax=plt.subplots(figsize=(6,3))
        ax.bar(list(fc.keys()), list(fc.values()), color=['#aaa','#4e79a7','#f1a340'])
        ax.set_ylabel('MAPE %')
        for i,(k,v) in enumerate(fc.items()): ax.text(i,v+0.1,f'{v:.1f}',ha='center',fontsize=9)
        st.pyplot(fig)
        st.caption("Linear regression is the most accurate (it captures the demand trend; tree models cannot extrapolate it).")

        st.markdown("#### Optimal modal split across the four disruption scenarios")
        SCEN={'Normal':(1.0,1.0),'Low water':(0.25,1.0),'Rail disruption':(1.0,KDB),'Combined':(0.25,KDB)}
        Dm=df.total_demand_tonnes.mean()
        table=[]
        for name,(kw2,kdb2) in SCEN.items():
            al2,cost2,co2_2=optimise(Dm,kw2,kdb2,Cap0)
            table.append({'Scenario':name,
                'Road':f"{al2['road']/1e6:.1f} Mt ({100*al2['road']/Dm:.0f}%)",
                'Rail':f"{al2['rail']/1e6:.1f} Mt ({100*al2['rail']/Dm:.0f}%)",
                'Barge':f"{al2['barge']/1e6:.1f} Mt ({100*al2['barge']/Dm:.0f}%)",
                'Cost':f"€{cost2/1e6:.0f}M",'CO₂':f"{co2_2/1000:.0f} kt"})
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption("Combined low water + rail works forces most freight onto road, roughly doubling CO₂ — the decarbonisation benefit is reversed.")

st.markdown("<br><hr><center><small style='color:#8a97a8;'>M.Sc. thesis — Decarbonising Rhine-corridor freight under disruption · optimisation model (PuLP/CBC)</small></center>",unsafe_allow_html=True)
