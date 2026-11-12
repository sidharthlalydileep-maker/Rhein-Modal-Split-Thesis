"""
optimiser.py — The single-commodity modal-allocation LP for one corridor month.

Decision variables  x_rail, x_barge, x_road  (tonnes).
Minimise  Z = Σ cₘ·dₘ·xₘ  +  λ · Σ (eₘ·dₘ / 1e6) · xₘ
s.t.      Σ xₘ = D                         (all demand served)
          x_barge ≤ κ_water · Cap⁰_barge   (low-water bound)
          x_rail  ≤ κ_DB   · Cap⁰_rail     (rail-works bound)
          x_road  ≥ 0, uncapped            (fallback)
          xₘ ≥ 0

Because the structure is a single demand equality with independent upper
bounds, the optimum has a closed form: fill modes in increasing generalised
per-tonne cost gₘ = cₘ·dₘ + λ·eₘ·dₘ/1e6. `optimise()` does the greedy fill;
`optimise_lp()` solves the same LP with PuLP/CBC; `cross_check()` asserts they
agree (the verification step in the proposal, §3.6).
"""
from dataclasses import dataclass, field
import pulp
import params as P


@dataclass
class Result:
    alloc: dict               # tonnes per mode
    demand: float
    cost_eur: float           # money cost only (€)
    co2_t: float              # emissions (t CO₂e)
    objective: float          # generalised objective (€), = cost + λ·co2
    shares: dict = field(default_factory=dict)   # fraction per mode
    road_spill: float = 0.0   # tonnes forced onto road


def gen_cost(cargo, lam):
    """Generalised per-tonne cost gₘ (€/t) for each mode."""
    c = P.COST[cargo]
    return {m: c[m] * P.DIST[m] + lam * P.EMIS[m] * P.DIST[m] / 1e6 for m in P.MODES}


def capacities(kappa_water, kappa_db, peaks, headroom):
    """Upper bounds (tonnes) per mode for this scenario. Road = None (uncapped)."""
    return {
        "barge": kappa_water * peaks["barge"] * (1 + headroom),
        "rail":  kappa_db * peaks["rail"] * (1 + headroom),
        "road":  None,
    }


def _summarise(alloc, cargo, lam, demand):
    c, e, d = P.COST[cargo], P.EMIS, P.DIST
    cost = sum(c[m] * d[m] * alloc[m] for m in P.MODES)
    co2 = sum(e[m] * d[m] * alloc[m] / 1e6 for m in P.MODES)
    shares = {m: (alloc[m] / demand if demand else 0.0) for m in P.MODES}
    return Result(alloc=alloc, demand=demand, cost_eur=cost, co2_t=co2,
                  objective=cost + lam * co2, shares=shares,
                  road_spill=alloc["road"])


def optimise(cargo, kappa_water, kappa_db, demand, lam, peaks, headroom=P.HEADROOM_BASE):
    """Closed-form greedy cheapest-fill allocation."""
    g = gen_cost(cargo, lam)
    cap = capacities(kappa_water, kappa_db, peaks, headroom)
    alloc = {m: 0.0 for m in P.MODES}
    remaining = demand
    for m in sorted(P.MODES, key=lambda x: g[x]):          # cheapest first
        if remaining <= 0:
            break
        limit = remaining if cap[m] is None else min(remaining, cap[m])
        alloc[m] = max(0.0, limit)
        remaining -= alloc[m]
    # any residual (should only happen if road were capped) → road
    if remaining > 1e-6:
        alloc["road"] += remaining
    return _summarise(alloc, cargo, lam, demand)


def optimise_lp(cargo, kappa_water, kappa_db, demand, lam, peaks,
                headroom=P.HEADROOM_BASE):
    """Same LP solved explicitly with PuLP/CBC (verification backend)."""
    g = gen_cost(cargo, lam)
    cap = capacities(kappa_water, kappa_db, peaks, headroom)
    prob = pulp.LpProblem("modal_allocation", pulp.LpMinimize)
    x = {m: pulp.LpVariable(f"x_{m}", lowBound=0) for m in P.MODES}
    prob += pulp.lpSum(g[m] * x[m] for m in P.MODES)       # objective
    prob += pulp.lpSum(x[m] for m in P.MODES) == demand    # demand equality
    for m in P.MODES:
        if cap[m] is not None:
            prob += x[m] <= cap[m]
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    alloc = {m: float(x[m].value() or 0.0) for m in P.MODES}
    return _summarise(alloc, cargo, lam, demand)


def cross_check(cargo, kappa_water, kappa_db, demand, lam, peaks,
                headroom=P.HEADROOM_BASE, tol=1.0):
    """Assert greedy and LP allocations match (tol in tonnes). Returns greedy Result."""
    gr = optimise(cargo, kappa_water, kappa_db, demand, lam, peaks, headroom)
    lp = optimise_lp(cargo, kappa_water, kappa_db, demand, lam, peaks, headroom)
    for m in P.MODES:
        if abs(gr.alloc[m] - lp.alloc[m]) > tol:
            raise AssertionError(
                f"greedy≠LP for {m}: {gr.alloc[m]:.1f} vs {lp.alloc[m]:.1f} "
                f"(cargo={cargo}, kw={kappa_water}, kdb={kappa_db}, λ={lam})")
    return gr


if __name__ == "__main__":
    from data_loader import build_panel, observed_peaks
    df = build_panel(save=False)
    peaks = {m: v["tonnes"] for m, v in observed_peaks(df).items()}
    D = 13.0e6   # illustrative 13 Mt month
    print("Demonstration (D = 13.0 Mt, dry bulk, λ = €180):\n")
    for name, (kw, kdb) in {
        "Normal (full water, full rail)": (1.00, 1.00),
        "Low water (Kaub ~60cm)":         (0.25, 1.00),
        "Rail works (κ_DB=0.35)":         (1.00, 0.35),
        "Both (low water + rail works)":  (0.25, 0.35),
    }.items():
        r = cross_check("Dry bulk", kw, kdb, D, 180.0, peaks)
        sh = {m: f"{r.shares[m]*100:4.1f}%" for m in P.MODES}
        print(f"  {name:32} rail {sh['rail']}  barge {sh['barge']}  road {sh['road']}"
              f"   cost €{r.cost_eur/1e6:6.1f}M  CO₂ {r.co2_t/1e3:6.1f} kt")
    print("\n✓ greedy and PuLP/CBC allocations agree in every scenario above.")
