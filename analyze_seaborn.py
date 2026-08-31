"""
Population-enhanced, seaborn-based analysis for the transport vs. gender-equality
study (Sweden vs. Romania, NUTS2 regions).

This script closes the gaps flagged in RESEARCH_REPORT.md by folding the two new
Eurostat population datasets into the pipeline:

    estat_demo_r_pjangroup.tsv  - population on 1 Jan by 5-year age group, sex, NUTS2
    estat_demo_r_d2jan.tsv      - population on 1 Jan by single year of age, sex, NUTS2
                                  (identical TOTAL/sex figures; kept for provenance)

What the population data unlocks vs. the original head-count-only analysis:
  * Real regional population by sex -> accurate per-capita transport denominators
    (replaces the GDP-derived MIO_EUR / EUR_HAB estimate).
  * Working-age (20-64) population by sex -> sex-specific EMPLOYMENT RATES and the
    employment-rate gender gap. This is the #1 limitation named in the report
    ("employment equality uses head-counts ... no regional employment-rate by sex")
    and its suggested next step ("use sex-specific employment rates").
  * Demographic context: age-sex structure (population pyramid), female share of
    the working-age population, old-age dependency.

Outputs (./outputs):
  region_indicators_pop.csv          - enriched 16-region indicator table
  country_summary_pop.csv            - SE vs RO means + Mann-Whitney p (new indicators)
  fig_population_pyramid.png
  fig_employment_rate_by_sex.png
  fig_country_comparison_sns.png
  fig_correlation_heatmap_sns.png
  fig_scatter_key_sns.png
  fig_emp_rate_gap_vs_road_safety.png
  fig_demographic_baseline.png

Run (venv):  python analyze_seaborn.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# Reuse the existing loading / windowing primitives and the base indicator table.
import analysis
from analysis import load, window_mean, build_table, WINDOW, REGIONS, OUT
from analyze_stats import add_composite

warnings.filterwarnings("ignore")

# Register the population files with analysis.load so we get the same TSV parsing.
analysis.FILES.setdefault("pjan", "estat_demo_r_pjangroup.tsv")
analysis.FILES.setdefault("d2jan", "estat_demo_r_d2jan.tsv")

sns.set_theme(style="whitegrid", context="talk")
COUNTRY_PALETTE = {"Sweden": "#1f77b4", "Romania": "#d62728"}
SEX_PALETTE = {"Male": "#4c72b0", "Female": "#c44e52"}

# 5-year bands that sum to the working-age population (20-64).
WA_BANDS = ["Y20-24", "Y25-29", "Y30-34", "Y35-39",
            "Y40-44", "Y45-49", "Y50-54", "Y55-59", "Y60-64"]
# Non-overlapping bands covering the whole age range (for the pyramid).
PYR_BANDS = ["Y_LT5", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29",
             "Y30-34", "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59",
             "Y60-64", "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"]
ELDERLY_BANDS = ["Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"]
PYR_LABELS = {b: b.replace("Y_LT5", "0-4").replace("Y_GE85", "85+")
              .replace("Y", "").replace("-", "-") for b in PYR_BANDS}


def _sum_bands(df: pd.DataFrame, sex: str, bands: list[str],
               regions=REGIONS) -> pd.Series:
    """Window-mean population summed across `bands`; NaN if any band is missing."""
    mat = pd.DataFrame(
        {b: window_mean(df, {"sex": sex, "age": b, "unit": "NR"}, regions) for b in bands}
    )
    return mat.sum(axis=1, min_count=len(bands))


# --------------------------------------------------------------------------- #
#  Build the enriched indicator table
# --------------------------------------------------------------------------- #
def build_enriched_table() -> pd.DataFrame:
    t = add_composite(build_table())
    t = t.rename(columns={"population": "population_gdp_derived"})

    pj = load("pjan")

    # ---- real population by sex ----
    pop_f = window_mean(pj, {"sex": "F", "age": "TOTAL", "unit": "NR"})
    pop_m = window_mean(pj, {"sex": "M", "age": "TOTAL", "unit": "NR"})
    pop_t = window_mean(pj, {"sex": "T", "age": "TOTAL", "unit": "NR"})
    t["pop_total"] = pop_t
    t["pop_female_share"] = pop_f / (pop_f + pop_m)

    # ---- working-age (20-64) population by sex ----
    wa_f = _sum_bands(pj, "F", WA_BANDS)
    wa_m = _sum_bands(pj, "M", WA_BANDS)
    wa_t = _sum_bands(pj, "T", WA_BANDS)
    t["wa_pop_20_64"] = wa_t
    t["wa_female_share"] = wa_f / (wa_f + wa_m)

    # ---- old-age dependency (65+ / 20-64), a demographic control ----
    eld_t = _sum_bands(pj, "T", ELDERLY_BANDS)
    t["old_age_dependency"] = eld_t / wa_t * 100

    # ---- NEW gender indicators: sex-specific EMPLOYMENT RATES (20-64) ----
    emp = load("emp")
    emp_f = window_mean(emp, {"sex": "F", "age": "Y20-64", "unit": "THS_PER"})
    emp_m = window_mean(emp, {"sex": "M", "age": "Y20-64", "unit": "THS_PER"})
    t["emp_rate_f"] = emp_f * 1000 / wa_f * 100
    t["emp_rate_m"] = emp_m * 1000 / wa_m * 100
    t["emp_rate_gap_pp"] = t["emp_rate_m"] - t["emp_rate_f"]   # +ve => men favoured

    # ---- refine per-capita transport denominators with REAL population ----
    net = load("net")
    vehst = load("vehst")
    avpa = load("avpa")
    mway_km = window_mean(net, {"tra_infr": "MWAY", "unit": "KM"})
    rail_km = window_mean(net, {"tra_infr": "RL", "unit": "KM"})
    car_nr = window_mean(vehst, {"vehicle": "CAR", "unit": "NR"})
    air = window_mean(avpa, {"tra_meas": "PAS_CRD", "unit": "THS_PAS"})
    t["motorway_km_per_Mhab"] = mway_km / (pop_t / 1e6)
    t["rail_km_per_Mhab"] = rail_km / (pop_t / 1e6)
    t["cars_per_1000hab"] = car_nr / pop_t * 1000
    t["air_pax_per_capita"] = air * 1000 / pop_t
    # road_deaths/injuries stay as Eurostat P_MHAB (already per-capita).
    return t


# Indicator label maps -------------------------------------------------------
GENDER_EXT = {
    "emp_rate_f": "Female employment rate (20-64, %)",
    "emp_rate_gap_pp": "Employment-rate gap M-F (pp)",
    "emp_female_share": "Female share of employment",
    "unemp_gap_pp": "Unemployment gap F-M (pp)",
    "early_leavers_gap_pp": "Early-leavers gap M-F (pp)",
    "gender_gap_magnitude": "Gender-gap magnitude (composite)",
}
TRANSPORT = {
    "motorway_km_per_Mhab": "Motorway km / M inhab.",
    "rail_km_per_Mhab": "Railway km / M inhab.",
    "cars_per_1000hab": "Cars / 1000 inhab.",
    "road_deaths_per_Mhab": "Road deaths / M inhab.",
    "road_injuries_per_Mhab": "Road injuries / M inhab.",
    "air_pax_per_capita": "Air passengers / capita",
}
CONTROL = {"gdp_per_capita_eur": "GDP per capita (EUR)"}


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_population_pyramid(pj: pd.DataFrame) -> str:
    """Age-sex structure of Sweden vs Romania as % of national population."""
    rows = []
    for country, geo in (("Sweden", "SE"), ("Romania", "RO")):
        tot = 0.0
        vals = {}
        for sex in ("F", "M"):
            for b in PYR_BANDS:
                v = window_mean(pj, {"sex": sex, "age": b, "unit": "NR"}, regions=[geo]).get(geo, np.nan)
                vals[(sex, b)] = v
                tot += 0 if np.isnan(v) else v
        for sex in ("F", "M"):
            for b in PYR_BANDS:
                pct = 100 * vals[(sex, b)] / tot
                rows.append({
                    "country": country,
                    "sex": "Female" if sex == "F" else "Male",
                    "age": PYR_LABELS[b],
                    "pct": -pct if sex == "M" else pct,   # males extend left
                })
    d = pd.DataFrame(rows)
    order = [PYR_LABELS[b] for b in reversed(PYR_BANDS)]   # oldest at top

    g = sns.catplot(
        data=d, kind="bar", y="age", x="pct", hue="sex", col="country",
        order=order, hue_order=["Male", "Female"], palette=SEX_PALETTE,
        dodge=False, height=7, aspect=0.75, orient="h", errorbar=None,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Share of national population (%)", "Age group")
    for ax in g.axes.flat:
        ax.axvline(0, color="0.3", lw=0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.0f}"))
    g.figure.suptitle("Population age-sex structure, 2018-2022 mean "
                      "(Male \u25c4 | \u25ba Female)", y=1.03)
    p = os.path.join(OUT, "fig_population_pyramid.png")
    g.figure.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    return p


def fig_employment_rate_by_sex(t: pd.DataFrame) -> str:
    """Employment rate (20-64) by sex, Sweden vs Romania (regional spread)."""
    long = t.melt(id_vars="country", value_vars=["emp_rate_f", "emp_rate_m"],
                  var_name="sex", value_name="emp_rate")
    long["sex"] = long["sex"].map({"emp_rate_f": "Female", "emp_rate_m": "Male"})

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.barplot(data=long, x="country", y="emp_rate", hue="sex",
                hue_order=["Male", "Female"], palette=SEX_PALETTE,
                errorbar="sd", capsize=0.12, err_kws={"linewidth": 1.4}, ax=ax)
    sns.stripplot(data=long, x="country", y="emp_rate", hue="sex",
                  hue_order=["Male", "Female"], palette=SEX_PALETTE,
                  dodge=True, jitter=0.12, size=6, edgecolor="0.25",
                  linewidth=0.6, alpha=0.85, ax=ax, legend=False)
    for c in ("Sweden", "Romania"):
        gap = t.loc[t.country == c, "emp_rate_gap_pp"].mean()
        y = t.loc[t.country == c, ["emp_rate_f", "emp_rate_m"]].mean().max()
        ax.text(["Sweden", "Romania"].index(c), y + 3,
                f"gap {gap:+.1f} pp", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Employment rate 20-64 (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 95)
    ax.set_title("Employment rate by sex — regional mean \u00b1 SD\n"
                 "(the gender gap the head-count share could not show)", fontsize=14)
    ax.legend(title="", loc="lower right")
    p = os.path.join(OUT, "fig_employment_rate_by_sex.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_country_comparison(t: pd.DataFrame) -> str:
    """SE vs RO means across the key gender + transport indicators."""
    cols = ["emp_rate_f", "emp_rate_gap_pp", "emp_female_share",
            "unemp_gap_pp", "early_leavers_gap_pp",
            "motorway_km_per_Mhab", "rail_km_per_Mhab", "cars_per_1000hab",
            "road_deaths_per_Mhab", "air_pax_per_capita"]
    labels = {**GENDER_EXT, **TRANSPORT}
    long = t.melt(id_vars="country", value_vars=cols,
                  var_name="indicator", value_name="value")
    long["indicator"] = long["indicator"].map(labels)
    order = [labels[c] for c in cols]

    g = sns.catplot(
        data=long, kind="bar", x="country", y="value", col="indicator",
        col_order=order, col_wrap=5, hue="country", palette=COUNTRY_PALETTE,
        order=["Sweden", "Romania"], errorbar="sd", capsize=0.15,
        height=3.2, aspect=0.9, sharey=False, legend=False,
    )
    g.set_titles("{col_name}", size=11)
    g.set_axis_labels("", "")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=10)
    g.figure.suptitle("Sweden vs Romania — NUTS2 regional means \u00b1 SD (2018-2022)",
                      y=1.02, fontsize=15)
    p = os.path.join(OUT, "fig_country_comparison_sns.png")
    g.figure.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    return p


def fig_correlation_heatmap(t: pd.DataFrame) -> pd.DataFrame:
    """Spearman heatmap: transport/GDP predictors x gender indicators."""
    preds = list(TRANSPORT) + list(CONTROL)
    genders = list(GENDER_EXT)
    corr = t[preds + genders].corr(method="spearman").loc[preds, genders]

    disp = corr.rename(index={**TRANSPORT, **CONTROL}, columns=GENDER_EXT)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    sns.heatmap(disp, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman r"}, annot_kws={"size": 11}, ax=ax)
    ax.set_title("Transport vs gender-equality indicators (pooled, n=16)\n"
                 "including population-based employment rates", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_correlation_heatmap_sns.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return corr


def fig_scatter_key(t: pd.DataFrame, corr: pd.DataFrame) -> str:
    """Four strongest predictor relationships with female employment rate."""
    target = "emp_rate_f"
    ranked = corr[target].abs().sort_values(ascending=False)
    picks = list(ranked.index[:4])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, x in zip(axes.flat, picks):
        for c in ("Sweden", "Romania"):
            sub = t[t.country == c]
            sns.regplot(data=sub, x=x, y=target, ax=ax, ci=None,
                        color=COUNTRY_PALETTE[c], label=c,
                        scatter_kws={"s": 60, "alpha": 0.85, "edgecolor": "0.3"},
                        line_kws={"linewidth": 1.6})
        rs = corr.loc[x, target]
        ax.set_title(f"{TRANSPORT.get(x, CONTROL.get(x, x))}   (r$_s$={rs:+.2f})",
                     fontsize=12)
        ax.set_xlabel(TRANSPORT.get(x, CONTROL.get(x, x)), fontsize=11)
        ax.set_ylabel(GENDER_EXT[target], fontsize=11)
        ax.legend(fontsize=10)
    fig.suptitle("Strongest transport relationships with the female employment rate "
                 "(NUTS2, 2018-2022)", fontsize=15, y=1.01)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_scatter_key_sns.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_emp_rate_gap_vs_safety(t: pd.DataFrame) -> str:
    """The robust, development-independent signal expressed with the NEW gap."""
    x, y = "road_deaths_per_Mhab", "emp_rate_gap_pp"
    rs, ps = stats.spearmanr(t[x], t[y])
    g = sns.lmplot(data=t, x=x, y=y, hue="country", palette=COUNTRY_PALETTE,
                   ci=None, height=6.5, aspect=1.3, legend=False,
                   scatter_kws={"s": 70, "alpha": 0.85, "edgecolor": "0.3"})
    ax = g.axes.flat[0]
    sns.regplot(data=t, x=x, y=y, ax=ax, scatter=False, ci=None,
                color="0.25", line_kws={"linestyle": "--", "linewidth": 1.6},
                truncate=False)
    ax.set_xlabel(TRANSPORT[x])
    ax.set_ylabel(GENDER_EXT[y])
    ax.set_title(f"Safer roads track a smaller employment-rate gender gap\n"
                 f"pooled Spearman r={rs:+.2f} (p={ps:.3f}); dashed = pooled trend",
                 fontsize=13)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COUNTRY_PALETTE[c],
                      markersize=10, label=c) for c in ("Sweden", "Romania")]
    ax.legend(handles=handles, title="", loc="upper left")
    p = os.path.join(OUT, "fig_emp_rate_gap_vs_road_safety.png")
    g.figure.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    return p


def fig_demographic_baseline(t: pd.DataFrame) -> str:
    """Employment share vs the demographic (working-age) female share."""
    fig, ax = plt.subplots(figsize=(8.5, 8))
    lo, hi = 0.40, 0.52
    ax.plot([lo, hi], [lo, hi], color="0.5", linestyle="--", linewidth=1.4,
            label="parity with working-age share")
    for c in ("Sweden", "Romania"):
        sub = t[t.country == c]
        ax.scatter(sub["wa_female_share"], sub["emp_female_share"],
                   s=80, alpha=0.85, edgecolor="0.3",
                   color=COUNTRY_PALETTE[c], label=c)
    ax.set_xlim(0.455, 0.515)
    ax.set_ylim(lo, 0.50)
    ax.set_xlabel("Female share of working-age (20-64) population")
    ax.set_ylabel("Female share of employment")
    ax.set_title("Women are employed below their demographic weight\n"
                 "(points below the dashed line = employment gender gap)", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_demographic_baseline.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def country_summary(t: pd.DataFrame) -> pd.DataFrame:
    cols = (["pop_total", "pop_female_share", "wa_female_share", "old_age_dependency",
             "emp_rate_f", "emp_rate_m", "emp_rate_gap_pp"]
            + list(GENDER_EXT) + list(TRANSPORT) + list(CONTROL))
    seen, ordered = set(), []
    for c in cols:                                   # de-duplicate, keep order
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    rows = []
    for col in ordered:
        se = t.loc[t.country == "Sweden", col].dropna()
        ro = t.loc[t.country == "Romania", col].dropna()
        p = (stats.mannwhitneyu(se, ro, alternative="two-sided")[1]
             if len(se) >= 2 and len(ro) >= 2 else np.nan)
        rows.append({"indicator": col, "Sweden_mean": se.mean(),
                     "Romania_mean": ro.mean(),
                     "difference_SE_minus_RO": se.mean() - ro.mean(),
                     "mannwhitney_p": p})
    return pd.DataFrame(rows)


def main():
    t = build_enriched_table()
    pj = load("pjan")

    t.to_csv(os.path.join(OUT, "region_indicators_pop.csv"))
    cs = country_summary(t)
    cs.to_csv(os.path.join(OUT, "country_summary_pop.csv"), index=False)

    p1 = fig_population_pyramid(pj)
    p2 = fig_employment_rate_by_sex(t)
    p3 = fig_country_comparison(t)
    corr = fig_correlation_heatmap(t)
    p5 = fig_scatter_key(t, corr)
    p6 = fig_emp_rate_gap_vs_safety(t)
    p7 = fig_demographic_baseline(t)

    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("=" * 78)
    print("SWEDEN vs ROMANIA — new population-based indicators (NUTS2 means, 2018-2022)")
    print("=" * 78)
    show = cs[cs.indicator.isin(
        ["pop_total", "pop_female_share", "wa_female_share", "old_age_dependency",
         "emp_rate_f", "emp_rate_m", "emp_rate_gap_pp"])]
    print(show.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("SANITY CHECK — real vs GDP-derived population (persons)")
    print("=" * 78)
    chk = t[["country", "pop_total", "population_gdp_derived"]].copy()
    chk["pct_diff"] = (chk.pop_total / chk.population_gdp_derived - 1) * 100
    print(chk.round(1).to_string())

    print("\n" + "=" * 78)
    print("POOLED SPEARMAN — predictors vs the NEW employment-rate indicators")
    print("=" * 78)
    print(corr[["emp_rate_f", "emp_rate_gap_pp"]].round(2).to_string())

    print("\nFigures saved:")
    for p in (p1, p2, p3,
              os.path.join(OUT, "fig_correlation_heatmap_sns.png"), p5, p6, p7):
        print("  ", p)


if __name__ == "__main__":
    main()
