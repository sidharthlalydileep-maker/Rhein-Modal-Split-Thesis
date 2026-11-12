"""
data_loader.py — Load and clean the Destatis modal series and the Kaub gauge
record, and join them into a single monthly panel.

Inputs (in ./data):
  barge_46321.csv  Destatis 46321-0002  International receipt, inland waterway (t)
  rail_46131.csv   Destatis 46131-0004  International receipt, rail (t)
  road_46231.csv   Destatis 46231-0002  International receipt, road (1000 t!)
  kaub.csv         PEGELONLINE Kaub gauge, 15-min water level (cm)

Output:
  monthly_panel.csv  one row per month: barge/rail/road tonnes, demand,
                     kaub_w_min_cm, kappa_water
"""
import os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def _find(fname):
    """Locate a data file whether it sits in ./data, next to this script,
    or in the current working directory. Makes the loader robust to how the
    files were uploaded (e.g. GitHub web upload placing them at the repo root)."""
    for cand in (os.path.join(DATA, fname),
                 os.path.join(HERE, fname),
                 os.path.join(os.getcwd(), fname),
                 fname):
        if os.path.exists(cand):
            return cand
    # fall back to the expected ./data path so the error message is clear
    return os.path.join(DATA, fname)


MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _read_destatis(fname, value_scale=1.0):
    """Read a Destatis 'year;month;value;flag;...' export.

    The first data column after year/month is 'International receipt –
    Goods carried' in tonnes (road: 1000 t, hence value_scale).
    Returns a DataFrame [date, tonnes].
    """
    path = _find(fname)
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(";")
            if len(parts) < 3:
                continue
            yr, mo, val = parts[0].strip().lstrip("﻿"), parts[1].strip(), parts[2].strip()
            if not yr.isdigit() or mo not in MONTHS:
                continue
            val = val.replace(",", "").replace(" ", "")
            if val in ("", "-", ".", "..."):
                continue
            try:
                tonnes = float(val) * value_scale
            except ValueError:
                continue
            rows.append((pd.Timestamp(int(yr), MONTHS[mo], 1), tonnes))
    df = pd.DataFrame(rows, columns=["date", "tonnes"]).sort_values("date")
    return df.reset_index(drop=True)


def load_modal_series():
    barge = _read_destatis("barge_46321.csv", 1.0).rename(columns={"tonnes": "barge_t"})
    rail = _read_destatis("rail_46131.csv", 1.0).rename(columns={"tonnes": "rail_t"})
    road = _read_destatis("road_46231.csv", 1000.0).rename(columns={"tonnes": "road_t"})  # 1000 t → t
    df = barge.merge(rail, on="date", how="outer").merge(road, on="date", how="outer")
    return df.sort_values("date").reset_index(drop=True)


def load_kaub_monthly():
    """Aggregate the 15-min Kaub record to the monthly MINIMUM level (cm).

    The monthly minimum is the governing low-water statistic for navigation.
    """
    path = _find("kaub.csv")
    k = pd.read_csv(path, sep=";", names=["timestamp", "value"], header=0)
    k["value"] = pd.to_numeric(k["value"], errors="coerce")
    k = k.dropna(subset=["value"])
    k["ts"] = pd.to_datetime(k["timestamp"], errors="coerce")
    k = k.dropna(subset=["ts"])
    k["date"] = k["ts"].dt.to_period("M").dt.to_timestamp()
    g = k.groupby("date")["value"].agg(
        kaub_w_min_cm="min",
        kaub_w_p10_cm=lambda s: s.quantile(0.10),
        kaub_w_median_cm="median",
        kaub_w_mean_cm="mean",
        kaub_n_readings="count",
    ).reset_index()
    for c in ["kaub_w_p10_cm", "kaub_w_median_cm", "kaub_w_mean_cm"]:
        g[c] = g[c].round(1)
    return g


# which monthly statistic drives κ_water (base case = the conservative minimum)
KAUB_STAT_COL = {"min": "kaub_w_min_cm", "p10": "kaub_w_p10_cm",
                 "median": "kaub_w_median_cm"}


def build_panel(save=True, kaub_stat="min"):
    """Build the monthly panel. kaub_stat selects the water statistic driving
    κ_water: 'min' (base case, conservative), 'p10', or 'median'."""
    from params import kappa_water
    modal = load_modal_series()
    kaub = load_kaub_monthly()
    df = modal.merge(kaub, on="date", how="left")
    # corridor demand proxy = sum of the three modal volumes (t)
    df["demand_t"] = df[["barge_t", "rail_t", "road_t"]].sum(axis=1, min_count=1)
    col = KAUB_STAT_COL[kaub_stat]
    df["kappa_water"] = df[col].apply(
        lambda c: kappa_water(c) if pd.notna(c) else np.nan)
    df = df.sort_values("date").reset_index(drop=True)
    if save:
        out = os.path.join(os.path.dirname(__file__), "outputs", "monthly_panel.csv")
        df.to_csv(out, index=False)
    return df


def observed_peaks(df=None):
    """Observed monthly peak volumes (t) for barge and rail, with the month."""
    if df is None:
        df = build_panel(save=False)
    peaks = {}
    for mode, col in [("barge", "barge_t"), ("rail", "rail_t")]:
        s = df.dropna(subset=[col])
        i = s[col].idxmax()
        peaks[mode] = {"tonnes": float(s.loc[i, col]),
                       "month": s.loc[i, "date"].strftime("%b %Y")}
    return peaks


if __name__ == "__main__":
    df = build_panel()
    pk = observed_peaks(df)
    print(f"Panel: {len(df)} months, {df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}")
    print("\nObserved peaks (base-capacity anchors):")
    for m, v in pk.items():
        print(f"  {m:5}: {v['tonnes']/1e6:6.3f} Mt  ({v['month']})")
    print("\nKaub monthly-min range: "
          f"{df['kaub_w_min_cm'].min():.0f} – {df['kaub_w_min_cm'].max():.0f} cm")
    print("\nLast 6 months:")
    cols = ["date", "barge_t", "rail_t", "road_t", "demand_t", "kaub_w_min_cm", "kappa_water"]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[cols].tail(6).to_string(index=False))
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
