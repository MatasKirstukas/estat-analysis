"""
Transport vs. Gender-Equality analysis for selected EU countries (NUTS2 regions).

Research question (see README.md):
    Do transport indicators relate to gender-equality indicators in selected EU
    countries (by NUTS2 regions)?

Geography:
    Sweden (gender-equality index 82, highest) vs. Romania (57.5, lowest).

Design:
    * Unit of analysis  : NUTS2 region (8 Swedish + 8 Romanian = 16 regions).
    * Reference period   : mean over the WINDOW years (default 2018-2022) to
                           stabilise estimates and maximise coverage.
    * Dependent (gender-equality) indicators are built from the sex breakdown of
      the labour-market / education datasets.
    * Independent (transport) indicators come from the five transport datasets;
      GDP per capita is used as an optional economic control.

Outputs (written to ./outputs):
    region_indicators.csv, correlations.csv, regressions.csv,
    country_summary.csv  + PNG figures.
"""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

WINDOW = list(range(2018, 2023))  # 2018..2022 inclusive

SE_REGIONS = ["SE11", "SE12", "SE21", "SE22", "SE23", "SE31", "SE32", "SE33"]
RO_REGIONS = ["RO11", "RO12", "RO21", "RO22", "RO31", "RO32", "RO41", "RO42"]
REGIONS = SE_REGIONS + RO_REGIONS

FILES = {
    "gdp": "estat_nama_10r_2gdp.tsv",
    "rapa": "estat_tran_r_rapa.tsv",
    "edat": "2025-08-09 estat_edat_lfse_16_education.tsv.xlsx",
    "emp": "2025-08-09 estat_lfst_r_lfe2_employment.tsv.xlsx",
    "unemp": "2025-08-09 estat_lfst_r_lfu3rt_employment.tsv.xlsx",
    "acci": "2025-08-09 estat_tran_r_acci.tsv.xlsx",
    "avpa": "2025-08-09 estat_tran_r_avpa_nm.tsv.xlsx",
    "net": "2025-08-09 estat_tran_r_net.tsv.xlsx",
    "vehst": "2025-08-09 estat_tran_r_vehst.tsv.xlsx",
}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def to_num(cell):
    """Parse a Eurostat cell -> float, ignoring flags (e, p, b, @C) and ':'=missing."""
    if cell is None:
        return np.nan
    s = str(cell).strip()
    if s in ("", ":", "-"):
        return np.nan
    s = s.replace(" ", "")          # remove thousands spaces
    m = _NUM.search(s)
    return float(m.group()) if m else np.nan


def load(key: str) -> pd.DataFrame:
    """Load any dataset into a tidy frame with explicit dimension columns + year cols."""
    path = os.path.join(HERE, FILES[key])
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name=0, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        geo = [c for c in df.columns if "geo" in c.lower()][0]
        df = df.rename(columns={geo: "geo"})
    else:
        raw = pd.read_csv(path, sep="\t", dtype=str)
        first = raw.columns[0]
        dim_names = first.split("\\")[0].split(",")
        packed = raw[first].astype(str).str.split(",", expand=True)
        packed.columns = dim_names
        years = raw.drop(columns=[first])
        years.columns = [str(c).strip() for c in years.columns]
        df = pd.concat([packed, years], axis=1).rename(columns={"geo": "geo"})
        if "geo" not in df.columns:                      # geo is last packed dim
            df = df.rename(columns={dim_names[-1]: "geo"})
    df["geo"] = df["geo"].astype(str).str.strip()
    return df


def window_mean(df: pd.DataFrame, filt: dict[str, str], regions=REGIONS) -> pd.Series:
    """Filter by dimension values, restrict to `regions`, return mean over WINDOW years."""
    sub = df.copy()
    for k, v in filt.items():
        sub = sub[sub[k].astype(str).str.strip() == v]
    sub = sub[sub["geo"].isin(regions)]
    year_cols = [c for c in sub.columns if re.match(r"^\d{4}$", str(c).strip())]
    keep = [c for c in year_cols if int(str(c).strip()) in WINDOW]
    if not keep:
        return pd.Series(np.nan, index=regions)
    vals = sub.set_index("geo")[keep].apply(lambda col: col.map(to_num))
    return vals.mean(axis=1, skipna=True).reindex(regions)


# --------------------------------------------------------------------------- #
#  Build the region-level indicator table
# --------------------------------------------------------------------------- #
def build_table() -> pd.DataFrame:
    gdp = load("gdp")
    emp = load("emp")
    unemp = load("unemp")
    edat = load("edat")
    acci = load("acci")
    avpa = load("avpa")
    net = load("net")
    vehst = load("vehst")

    t = pd.DataFrame(index=REGIONS)
    t.index.name = "region"
    t["country"] = ["Sweden" if r.startswith("SE") else "Romania" for r in REGIONS]

    # ---- economic control + derived population ----
    gdp_phab = window_mean(gdp, {"unit": "EUR_HAB"})
    gdp_total = window_mean(gdp, {"unit": "MIO_EUR"})           # million EUR
    population = (gdp_total * 1e6) / gdp_phab                    # persons
    t["gdp_per_capita_eur"] = gdp_phab
    t["population"] = population

    # ---- DEPENDENT: gender-equality indicators ----
    # Employment (counts, THS_PER, age 20-64)
    emp_f = window_mean(emp, {"sex": "F", "age": "Y20-64", "unit": "THS_PER"})
    emp_m = window_mean(emp, {"sex": "M", "age": "Y20-64", "unit": "THS_PER"})
    t["emp_female_share"] = emp_f / (emp_f + emp_m)             # 0.5 == parity
    t["emp_f_to_m_ratio"] = emp_f / emp_m

    # Unemployment rate (%, TOTAL education, age 20-64)
    un_f = window_mean(unemp, {"sex": "F", "age": "Y20-64", "unit": "PC", "isced11": "TOTAL"})
    un_m = window_mean(unemp, {"sex": "M", "age": "Y20-64", "unit": "PC", "isced11": "TOTAL"})
    t["unemp_rate_f"] = un_f
    t["unemp_rate_m"] = un_m
    t["unemp_gap_pp"] = un_f - un_m                            # +ve => women worse

    # Early leavers from education/training (%, age 18-24)
    el_f = window_mean(edat, {"sex": "F", "age": "Y18-24", "unit": "PC"})
    el_m = window_mean(edat, {"sex": "M", "age": "Y18-24", "unit": "PC"})
    t["early_leavers_f"] = el_f
    t["early_leavers_m"] = el_m
    t["early_leavers_gap_pp"] = el_m - el_f                    # +ve => boys leave more

    # ---- INDEPENDENT: transport indicators ----
    # Networks (km) -> normalise per million inhabitants (provision intensity)
    mway_km = window_mean(net, {"tra_infr": "MWAY", "unit": "KM"})
    rail_km = window_mean(net, {"tra_infr": "RL", "unit": "KM"})
    t["motorway_km_per_Mhab"] = mway_km / (population / 1e6)
    t["rail_km_per_Mhab"] = rail_km / (population / 1e6)

    # Vehicle stock: passenger cars (absolute NR) -> per 1000 inhabitants
    car_nr = window_mean(vehst, {"vehicle": "CAR", "unit": "NR"})
    t["cars_per_1000hab"] = car_nr / population * 1000

    # Road accidents -> already per-capita (per million inhabitants)
    t["road_deaths_per_Mhab"] = window_mean(acci, {"victim": "KIL", "unit": "P_MHAB"})
    t["road_injuries_per_Mhab"] = window_mean(acci, {"victim": "INJ", "unit": "P_MHAB"})

    # Air passengers carried (thousand) -> per inhabitant
    air = window_mean(avpa, {"tra_meas": "PAS_CRD", "unit": "THS_PAS"})
    t["air_pax_per_capita"] = (air * 1000) / population

    return t


if __name__ == "__main__":
    table = build_table()
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("Region-level indicator table (window mean %d-%d):" % (WINDOW[0], WINDOW[-1]))
    print(table.round(3).to_string())
    print("\nNon-null counts per column:")
    print(table.notna().sum().to_string())
    table.to_csv(os.path.join(OUT, "region_indicators.csv"))
    print(f"\nSaved -> {os.path.join(OUT, 'region_indicators.csv')}")
