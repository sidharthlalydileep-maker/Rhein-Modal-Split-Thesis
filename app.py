# ================================================================
#  Rhine Corridor — Modal-Shift Dashboard (CLOUD, two tabs)
#  Tab 1: Model comparison (4 ML models, all metrics)
#  Tab 2: Optimiser (parameters left, results right)
#  kappa_water = CCNR "Act now!" tiers.  Deploy on Streamlit Cloud.
# ================================================================
import streamlit as st
import pulp, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE, mean_absolute_percentage_error as MAPE, r2_score, mean_squared_error

# ---------------- DEFAULT PARAMETERS ----------------
E_def={'road':81.7,'rail':12.8,'barge':32.1}      # gCO2e/tkm (EcoTransIT)
C_def={'road':0.08,'rail':0.035,'barge':0.025}    # EUR/tkm   (IRU/EU/EBU)
DIST ={'road':745.0,'rail':765.0,'barge':846.0}   # km per mode
COL  ={'road':'#e15759','rail':'#f1a340','barge':'#4e79a7'}

def kappa_water(cm):    # CCNR "Act now!" (Ed.3.0, 2023), Fig. 18
    if   cm>=134: return 1.00
    elif cm>= 72: return 0.50
    elif cm>= 44: return 0.25
    else:         return 0.15

@st.cache_data
def load_data():
    return pd.read_csv('corridor_demand_monthly.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)

@st.cache_data
def run_models(_df):
    d=_df.copy(); d['y']=d.total_demand_tonnes/1e6
    d['month']=d.date.dt.month; d['quarter']=d.date.dt.quarter; d['t']=np.arange(len(d))
    # leakage-free: PAST total demand + calendar + water (NOT same-month components)
    d['lag1']=d.y.shift(1); d['lag12']=d.y.shift(12); d['roll3']=d.y.shift(1).rolling(3).mean()
    d=d.dropna().reset_index(drop=True)
    F=['month','quarter','t','lag1','lag12','roll3','kaub_w_min_cm']; sp=int(len(d)*0.8)
    Xtr,Xte,ytr,yte=d[F].iloc[:sp],d[F].iloc[sp:],d.y.iloc[:sp],d.y.iloc[sp:]
    preds={'Naive(-12)':d.lag12.iloc[sp:].values,
           'Linear':LinearRegression().fit(Xtr,ytr).predict(Xte),
           'RandomForest':RandomForestRegressor(n_estimators=200,random_state=0).fit(Xtr,ytr).predict(Xte),
           'XGBoost':XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=3,random_state=0).fit(Xtr,ytr).predict(Xte)}
    rows=[]
    for n,p in preds.items():
        rows.append({'Model':n,'MAE':MAE(yte,p),'RMSE':mean_squared_error(yte,p)**.5,'MAPE':MAPE(yte,p)*100,'R2':r2_score(yte,p)})
    m=pd.DataFrame(rows)
    dates=d.date.iloc[sp:].reset_index(drop=True); actual=yte.reset_index(drop=True)
    return m, preds, dates, actual

def optimise(D,kw,kdb,E,C,LAM,bb,br):
    cost_pt={m:C[m]*DIST[m] for m in E}; co2_pt={m:E[m]*DIST[m]/1000 for m in E}
    cap={'barge':kw*bb,'rail':kdb*br,'road':1e12}; obj={m:cost_pt[m]+LAM*co2_pt[m]/1000 for m in E}
    p=pulp.LpProblem('x',pulp.LpMinimize); x={m:pulp.LpVariable(m,0,cap[m]) for m in E}
    p+=pulp.lpSum(obj[m]*x[m] for m in x); p+=pulp.lpSum(x[m] for m in x)==D
    p.solve(pulp.PULP_CBC_CMD(msg=0)); al={m:x[m].value() for m in x}
    return al,sum(cost_pt[m]*al[m] for m in x),sum(co2_pt[m]*al[m] for m in x)/1000

# ============================== PAGE ==============================
st.set_page_config(page_title="Rhine Modal-Shift Tool", page_icon="🚢", layout="wide")
st.markdown("""<style>
.main{background:#f7f9fc;} .block-container{padding-top:2rem;} h1{color:#1f3864;font-weight:700;}
.metric-card{background:white;border-radius:14px;padding:16px 18px;box-shadow:0 2px 10px rgba(31,56,100,0.08);text-align:center;border-top:4px solid #4e79a7;}
.metric-label{color:#6b7a90;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.5px;}
.metric-value{color:#1f3864;font-size:1.6rem;font-weight:700;margin-top:4px;}
.metric-sub{color:#8a97a8;font-size:0.78rem;}
section[data-testid="stSidebar"]{background:#1f3864;} section[data-testid="stSidebar"] *{color:#e8edf5 !important;}
</style>""", unsafe_allow_html=True)

st.title("🚢 Rhine Corridor — Modal-Shift Decision Tool")
df = load_data()

tab1, tab2 = st.tabs(["📊 Model comparison", "🎛️ Optimiser"])

# ---------------- TAB 1: MODEL COMPARISON ----------------
with tab1:
    st.markdown("### Forecast model comparison")
    metrics, preds, dates, actual = run_models(df)
    best = metrics[metrics.Model!='Naive(-12)'].sort_values('MAPE').iloc[0].Model
    st.markdown(f"Four demand-forecasting models are compared on a time-ordered test set. **Best model: {best}.**")
    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**All four metrics (lower better, except R²)**")
        fig,ax=plt.subplots(1,4,figsize=(11,2.8)); cs=['#aaa','#4e79a7','#f1a340','#e15759']
        for i,(c,t) in enumerate([('MAE','MAE'),('RMSE','RMSE'),('MAPE','MAPE %'),('R2','R²')]):
            ax[i].bar(metrics.Model,metrics[c],color=cs); ax[i].set_title(t,fontsize=9,fontweight='bold')
            ax[i].tick_params(axis='x',rotation=45,labelsize=6)
            if c=='R2': ax[i].axhline(0,color='k',lw=0.5)
        plt.tight_layout(); st.pyplot(fig)
    with col2:
        st.markdown("**Forecast vs actual (test period)**")
        fig2,ax2=plt.subplots(figsize=(6,3))
        ax2.plot(dates,actual,'o-',color='#1f3864',label='Actual',lw=2,ms=3)
        ax2.plot(dates,preds['Linear'],'s--',color='#4e79a7',label='Linear',lw=1.5)
        ax2.plot(dates,preds['Naive(-12)'],'^:',color='#e15759',label='Naive',lw=1.5)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.tick_params(axis='x',rotation=30,labelsize=7)
        ax2.set_ylabel('Mt/month'); plt.tight_layout(); st.pyplot(fig2)
    st.caption(f"{best} captures the demand trend; tree models cannot extrapolate it (worse than naive). Features are leakage-free (past demand + calendar + water, not same-month components).")

# ---------------- TAB 2: OPTIMISER ----------------
with tab2:
    sb=st.sidebar
    sb.markdown("## ⚙️ Scenario")
    kaub=sb.slider("Kaub water level (cm)",0,350,300,1)
    rail_works=sb.checkbox("🚧 Rail renovation (DB)")
    kdb=sb.slider("Rail capacity κ_DB",0.0,1.0,0.35,0.05) if rail_works else 1.0
    demand_mt=sb.number_input("Monthly demand (Mt)",5.0,25.0,float(round(df.total_demand_tonnes.mean()/1e6,1)),0.5)
    sb.markdown("## 💶 Economics")
    LAM=sb.slider("Carbon price λ (€/t CO₂)",0,640,180,10)
    sb.markdown("## 🏗️ Base capacities (Mt)")
    bb=sb.number_input("Barge",5.0,20.0,round(1.15*df.barge_tonnes.max()/1e6,1),0.5)*1e6
    br=sb.number_input("Rail",3.0,15.0,round(1.15*df.rail_tonnes.max()/1e6,1),0.5)*1e6
    with sb.expander("🔧 Cost & emission factors"):
        E={m:st.number_input(f"CO₂ {m}",0.0,200.0,E_def[m],1.0) for m in E_def}
        C={m:st.number_input(f"Cost {m}",0.0,0.5,C_def[m],0.005,format="%.3f") for m in C_def}
    sb.markdown("---"); sb.markdown("<small>κ_water: CCNR 'Act now!' tiers.</small>",unsafe_allow_html=True)

    kw=kappa_water(kaub)
    al,cost,co2=optimise(demand_mt*1e6,kw,kdb,E,C,LAM,bb,br)
    D=demand_mt*1e6; shares={m:100*al[m]/D for m in E}
    def card(col,l,v,s=""):
        col.markdown(f"""<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div><div class="metric-sub">{s}</div></div>""",unsafe_allow_html=True)
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
| **Rail capacity** | {kdb*100:.0f}% |
| **Carbon price** | €{LAM}/t |
| **Demand** | {demand_mt:.1f} Mt |""")
        st.markdown("<br>",unsafe_allow_html=True)
        if shares['road']>50: st.error("⚠️ **Over half on road** — decarbonisation largely reversed.")
        elif shares['road']>10: st.warning("🟠 Some freight shifted to road.")
        else: st.success("✅ Freight stays on low-carbon modes.")

st.markdown("<br><hr><center><small style='color:#8a97a8;'>M.Sc. thesis — Rhine-corridor freight under disruption · optimisation (PuLP/CBC) · κ_water: CCNR 'Act now!'</small></center>",unsafe_allow_html=True)
