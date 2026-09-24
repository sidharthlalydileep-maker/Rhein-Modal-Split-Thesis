# ================================================================
#  Rhine Corridor — Modal-Shift Dashboard (with CARGO SELECTOR)
#  Costs: Panteia/KiM (2021), by cargo type.  kappa_water: CCNR "Act now!"
# ================================================================
import streamlit as st
import pulp
import matplotlib.pyplot as plt

# ---- emission factors (EcoTransIT, ISO 14083) gCO2e/tkm ----
E = {'road':81.7, 'rail':12.8, 'barge':32.1}
DIST = {'road':745.0,'rail':765.0,'barge':846.0}
COL = {'road':'#e15759', 'rail':'#f1a340', 'barge':'#4e79a7'}

# ---- COST per tkm by CARGO TYPE (Panteia/KiM 2021) ----
CARGO_COST = {
    'Dry bulk':    {'road':0.246, 'rail':0.014, 'barge':0.018},
    'Liquid bulk': {'road':0.138, 'rail':0.018, 'barge':0.029},
    'Break bulk':  {'road':0.203, 'rail':0.047, 'barge':0.022},
    'Container':   {'road':0.125, 'rail':0.045, 'barge':0.025},
}

def kappa_water(cm):   # CCNR "Act now!" tiers
    if   cm >= 134: return 1.00
    elif cm >=  72: return 0.50
    elif cm >=  44: return 0.25
    else:           return 0.15

def optimise(D, kw, kdb, C, LAM, bb, br):
    cost_pt={m:C[m]*DIST[m] for m in E}; co2_pt={m:E[m]*DIST[m]/1000 for m in E}
    cap={'barge':kw*bb,'rail':kdb*br,'road':1e12}
    obj={m:cost_pt[m]+LAM*co2_pt[m]/1000 for m in E}
    p=pulp.LpProblem('x',pulp.LpMinimize); x={m:pulp.LpVariable(m,0,cap[m]) for m in E}
    p+=pulp.lpSum(obj[m]*x[m] for m in x); p+=pulp.lpSum(x[m] for m in x)==D
    p.solve(pulp.PULP_CBC_CMD(msg=0)); al={m:x[m].value() for m in x}
    return al, sum(cost_pt[m]*al[m] for m in x), sum(co2_pt[m]*al[m] for m in x)/1000

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
st.markdown("<p style='color:#6b7a90;margin-top:-10px;'>Cost- and carbon-optimal road / rail / barge allocation, Rotterdam–Basel, under disruption</p>", unsafe_allow_html=True)

sb = st.sidebar
sb.markdown("## 📦 Cargo type")
cargo = sb.selectbox("Select cargo", list(CARGO_COST.keys()))
C = CARGO_COST[cargo]
sb.caption(f"Cost/tkm — barge €{C['barge']}, rail €{C['rail']}, road €{C['road']} (Panteia/KiM 2021)")

sb.markdown("## ⚙️ Scenario")
kaub = sb.slider("Kaub water level (cm)", 0, 350, 300, 1)
rail_works = sb.checkbox("🚧 Rail renovation (DB)")
kdb = sb.slider("Rail capacity κ_DB", 0.0, 1.0, 0.35, 0.05) if rail_works else 1.0
demand_mt = sb.number_input("Monthly demand (Mt)", 5.0, 25.0, 13.2, 0.5)
sb.markdown("## 💶 Economics")
LAM = sb.slider("Carbon price λ (€/t CO₂)", 0, 640, 180, 10)
sb.markdown("## 🏗️ Base capacities (Mt)")
bb = sb.number_input("Barge", 5.0, 20.0, 11.5, 0.5)*1e6
br = sb.number_input("Rail", 3.0, 15.0, 6.8, 0.5)*1e6
sb.markdown("---")
sb.markdown("<small>Costs: Panteia/KiM (2021) by cargo type. Emissions: EcoTransIT (ISO 14083). Carbon price: UBA. κ_water: CCNR 'Act now!'. LP: PuLP/CBC.</small>", unsafe_allow_html=True)

kw = kappa_water(kaub)
al, cost, co2 = optimise(demand_mt*1e6, kw, kdb, C, LAM, bb, br)
D = demand_mt*1e6; shares = {m:100*al[m]/D for m in E}

def card(col,l,v,s=""):
    col.markdown(f"""<div class="metric-card"><div class="metric-label">{l}</div><div class="metric-value">{v}</div><div class="metric-sub">{s}</div></div>""",unsafe_allow_html=True)
st.markdown(f"#### Results — {cargo}")
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
| **Cargo** | {cargo} |
| **Kaub level** | {kaub} cm |
| **Barge capacity** | {kw*100:.0f}% |
| **Rail capacity** | {kdb*100:.0f}% |
| **Carbon price** | €{LAM}/t |
| **Demand** | {demand_mt:.1f} Mt |""")
    st.markdown("<br>",unsafe_allow_html=True)
    if shares['road']>50: st.error("⚠️ **Over half on road** — decarbonisation largely reversed.")
    elif shares['road']>10: st.warning("🟠 Some freight shifted to road.")
    else: st.success("✅ Freight stays on low-carbon modes.")

st.markdown("<br><hr><center><small style='color:#8a97a8;'>M.Sc. thesis — Rhine-corridor freight under disruption · optimisation (PuLP/CBC) · costs: Panteia/KiM 2021 · κ_water: CCNR 'Act now!'</small></center>",unsafe_allow_html=True)