"""
params.py — All model parameters for the Rhine-corridor modal-shift optimiser.

Every value here is the finalised figure from the thesis proposal
("Methodology and Implementation"). Provenance tags follow Riad's point 13:
  (1) directly published   (2) calculated from published data   (3) assumption
See the proposal's Table 4 (provenance) for sources.
"""

# ---------------------------------------------------------------------------
# A. COST RATES  cₘ  (€ per tonne-kilometre), by cargo type   [① Published]
#    Source: Panteia (2023) "Cost figures for freight transport".
#      Road dry  = Table 4.3 (truck+trailer);  road liquid = Table 4.4 (tractor+trailer)
#      Rail      = Table 6.2 (charter);         barge = Table 3.5 (large ships)
# ---------------------------------------------------------------------------
COST = {
    "Dry bulk":    {"road": 0.246, "rail": 0.014, "barge": 0.018},
    "Liquid bulk": {"road": 0.138, "rail": 0.018, "barge": 0.029},
}
CARGOES = list(COST.keys())
MODES = ["rail", "barge", "road"]   # listed cheap→dear for reference

# ---------------------------------------------------------------------------
# B. EMISSION FACTORS  eₘ  (g CO₂e per tonne-kilometre), per mode  [① Published]
#    Source: EcoTransIT World (2023), per ISO 14083. Common to both cargoes.
# ---------------------------------------------------------------------------
EMIS = {"road": 81.7, "rail": 12.8, "barge": 32.1}

# ---------------------------------------------------------------------------
# C. CORRIDOR DISTANCES  dₘ  (km)                                 [② Calculated]
#    Rotterdam–Basel per mode (route planning; Rhine fairway km for barge).
# ---------------------------------------------------------------------------
DIST = {"road": 745.0, "rail": 765.0, "barge": 846.0}

# ---------------------------------------------------------------------------
# D. CAPACITY  — base capacity = observed monthly peak × (1 + headroom)
#    Headroom 15% is the base case (③ assumption); swept 0 / 15 / 30 %.
#    Observed peaks are computed from the Destatis series by the loader, but
#    the proposal's reference values are recorded here for cross-checking.
# ---------------------------------------------------------------------------
HEADROOM_BASE = 0.15
HEADROOM_SWEEP = [0.00, 0.15, 0.30]

# Reference observed peaks (Mt/month) — loader recomputes from data.
PEAK_REF = {"barge": 9.985e6, "rail": 5.928e6}

# ---------------------------------------------------------------------------
# E. DISRUPTION FACTORS
#    κ_water — remaining barge transport capacity by Kaub water level  [① Published]
#      CCNR (2023) "Act Now!" Fig. 18 step function.
# ---------------------------------------------------------------------------
# (lower_cm_inclusive, kappa) tiers, evaluated high→low.
KWATER_TIERS = [
    (134.0, 1.00),   # Kaub ≥ 134 cm  → full
    (72.0,  0.50),   # 72 ≤ Kaub < 134
    (44.0,  0.25),   # 44 ≤ Kaub < 72
    (0.0,   0.15),   # Kaub < 44 cm
]

def kappa_water(kaub_cm: float) -> float:
    """Remaining barge capacity fraction for a given Kaub level (cm)."""
    for lo, k in KWATER_TIERS:
        if kaub_cm >= lo:
            return k
    return KWATER_TIERS[-1][1]

#    κ_DB — remaining rail capacity during works (scenario analysis, Table 2)
#      Normal/closure (0.0, 1.0) published; single-track band 0.25–0.70 is a
#      ③ assumption examined as a sweep; 0.35 is the reference scenario.
KDB_SCENARIOS = {
    "Normal (reference)":        1.00,
    "Mild restriction":          0.70,
    "Higher":                    0.60,
    "Optimistic (½ two-track)":  0.50,
    "Mid":                       0.40,
    "Reference single-track":    0.35,
    "Severe single-track":       0.25,
    "Full closure":              0.00,
}
KDB_REF = 0.35

# ---------------------------------------------------------------------------
# F. CARBON PRICE  λ  (€ per tonne CO₂), shadow price
#    0 = Model A (BAU).  55 = actual nEHS/BEHG today (§10 Abs. 2 BEHG).
#    180 = UBA central (1% time pref).  640 = UBA upper (0%).  [UBA 2019, Table 1 p.8]
# ---------------------------------------------------------------------------
LAMBDAS = {
    "Model A (λ=0)":        0.0,
    "Actual today (€55)":  55.0,
    "Model B central (€180)": 180.0,
    "Upper (€640)":       640.0,
}
LAMBDA_BASE = 180.0
