"""
Statistical analysis + figures for the transport vs. gender-equality study.

Run AFTER (or alongside) analysis.py; it imports build_table().

Produces in ./outputs:
    country_summary.csv      - SE vs RO means + Mann-Whitney U test
    correlations.csv         - pooled & within-country Spearman/Pearson r (+p)
    regressions.csv          - OLS: gender indicator ~ transport (+ GDP control)
    fig_country_comparison.png
    fig_correlation_heatmap.png
    fig_scatter_key.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import build_table, OUT, WINDOW

# Indicator groups -----------------------------------------------------------
GENDER = {
    "emp_female_share": "Female share of employment (20-64)",
    "unemp_gap_pp": "Unemployment gap F-M (pp)",
    "early_leavers_gap_pp": "Early-leavers gap M-F (pp)",
}
TRANSPORT = {
    "motorway_km_per_Mhab": "Motorway km / M inhab.",
    "rail_km_per_Mhab": "Railway km / M inhab.",
    "cars_per_1000hab": "Passenger cars / 1000 inhab.",
    "road_deaths_per_Mhab": "Road deaths / M inhab.",
    "road_injuries_per_Mhab": "Road injuries / M inhab.",
    "air_pax_per_capita": "Air passengers / capita",
}
CONTROL = {"gdp_per_capita_eur": "GDP per capita (EUR)"}


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def add_composite(t: pd.DataFrame) -> pd.DataFrame:
    """Composite gender-gap magnitude: higher = LESS gender-equal."""
    emp_parity_dist = (t["emp_female_share"] - 0.5).abs()
    comp = pd.concat(
        [
            zscore(emp_parity_dist),
            zscore(t["unemp_gap_pp"].abs()),
            zscore(t["early_leavers_gap_pp"].abs()),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    t["gender_gap_magnitude"] = comp
    return t


# --------------------------------------------------------------------------- #
def country_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(GENDER) + ["gender_gap_magnitude"] + list(TRANSPORT) + list(CONTROL)
    for col in cols:
        se = t.loc[t.country == "Sweden", col].dropna()
        ro = t.loc[t.country == "Romania", col].dropna()
        # Mann-Whitney U (small N, non-parametric)
        if len(se) >= 2 and len(ro) >= 2:
            u, p = stats.mannwhitneyu(se, ro, alternative="two-sided")
        else:
            p = np.nan
        rows.append(
            {
                "indicator": col,
                "Sweden_mean": se.mean(),
                "Romania_mean": ro.mean(),
                "difference_SE_minus_RO": se.mean() - ro.mean(),
                "mannwhitney_p": p,
            }
        )
    return pd.DataFrame(rows)


def correlations(t: pd.DataFrame) -> pd.DataFrame:
    """Pooled and within-country (country-demeaned) Spearman & Pearson."""
    preds = list(TRANSPORT) + list(CONTROL)
    targets = list(GENDER) + ["gender_gap_magnitude"]
    # country-demeaned copy for within-country association
    dm = t.copy()
    for col in preds + targets:
        dm[col] = t.groupby("country")[col].transform(lambda s: s - s.mean())

    rows = []
    for y in targets:
        for x in preds:
            sub = t[[x, y]].dropna()
            sub_dm = dm[[x, y]].dropna()
            if len(sub) >= 4:
                pr, pp = stats.pearsonr(sub[x], sub[y])
                sr, sp = stats.spearmanr(sub[x], sub[y])
            else:
                pr = pp = sr = sp = np.nan
            if len(sub_dm) >= 4:
                wr, wp = stats.pearsonr(sub_dm[x], sub_dm[y])
            else:
                wr = wp = np.nan
            rows.append(
                {
                    "gender_indicator": y,
                    "transport_indicator": x,
                    "n": len(sub),
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_r": sr,
                    "spearman_p": sp,
                    "within_country_pearson_r": wr,
                    "within_country_pearson_p": wp,
                }
            )
    return pd.DataFrame(rows)


def regressions(t: pd.DataFrame) -> pd.DataFrame:
    """OLS: each gender indicator ~ transport indicator, with/without GDP control."""
    t = t.copy()
    t["log_gdp"] = np.log(t["gdp_per_capita_eur"])
    rows = []
    targets = list(GENDER) + ["gender_gap_magnitude"]
    for y in targets:
        for x in list(TRANSPORT):
            d = t[[y, x, "log_gdp"]].dropna()
            if len(d) < 6:
                continue
            # standardise predictors so coefficients are comparable
            d = d.assign(_x=zscore(d[x]), _g=zscore(d["log_gdp"]))
            for label, formula in (
                ("simple", f"Q('{y}') ~ _x"),
                ("with_gdp", f"Q('{y}') ~ _x + _g"),
            ):
                m = smf.ols(formula, data=d).fit()
                rows.append(
                    {
                        "gender_indicator": y,
                        "transport_indicator": x,
                        "model": label,
                        "n": int(m.nobs),
                        "beta_transport_std": m.params.get("_x", np.nan),
                        "p_transport": m.pvalues.get("_x", np.nan),
                        "beta_gdp_std": m.params.get("_g", np.nan),
                        "p_gdp": m.pvalues.get("_g", np.nan),
                        "r_squared": m.rsquared,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Figures
# --------------------------------------------------------------------------- #
def fig_country_comparison(t: pd.DataFrame):
    cols = list(GENDER) + list(TRANSPORT)
    labels = {**GENDER, **TRANSPORT}
    means = t.groupby("country")[cols].mean()
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.flat, cols):
        vals = means[col]
        ax.bar(vals.index, vals.values, color=["#1f77b4", "#d62728"])
        ax.set_title(labels[col], fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Sweden vs Romania — mean by NUTS2 region (2018-2022)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "fig_country_comparison.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_heatmap(corr: pd.DataFrame):
    targets = list(GENDER) + ["gender_gap_magnitude"]
    preds = list(TRANSPORT) + list(CONTROL)
    mat = corr.pivot(index="transport_indicator", columns="gender_indicator",
                     values="spearman_r").reindex(index=preds, columns=targets)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels(preds, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman r")
    ax.set_title("Transport vs gender-equality indicators (pooled, n=16)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_correlation_heatmap.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def fig_scatter_key(t: pd.DataFrame, corr: pd.DataFrame):
    """Scatter the 4 strongest pooled relationships, coloured by country."""
    ranked = (corr.assign(absr=corr["spearman_r"].abs())
                  .dropna(subset=["absr"]).sort_values("absr", ascending=False))
    seen, picks = set(), []
    for _, r in ranked.iterrows():
        key = (r["gender_indicator"], r["transport_indicator"])
        if r["transport_indicator"] in seen:
            continue
        seen.add(r["transport_indicator"])
        picks.append(r)
        if len(picks) == 4:
            break
    colors = {"Sweden": "#1f77b4", "Romania": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, r in zip(axes.flat, picks):
        x, y = r["transport_indicator"], r["gender_indicator"]
        for c in ("Sweden", "Romania"):
            sub = t[t.country == c]
            ax.scatter(sub[x], sub[y], label=c, color=colors[c], s=55, alpha=0.85)
        ax.set_xlabel(TRANSPORT.get(x, x), fontsize=9)
        ax.set_ylabel(GENDER.get(y, y), fontsize=9)
        ax.set_title(f"r_s={r['spearman_r']:.2f} (p={r['spearman_p']:.3f})", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Strongest transport–gender relationships (NUTS2, 2018-2022)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "fig_scatter_key.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main():
    t = add_composite(build_table())

    cs = country_summary(t)
    corr = correlations(t)
    reg = regressions(t)

    cs.to_csv(os.path.join(OUT, "country_summary.csv"), index=False)
    corr.to_csv(os.path.join(OUT, "correlations.csv"), index=False)
    reg.to_csv(os.path.join(OUT, "regressions.csv"), index=False)
    t.to_csv(os.path.join(OUT, "region_indicators.csv"))

    p1 = fig_country_comparison(t)
    p2 = fig_heatmap(corr)
    p3 = fig_scatter_key(t, corr)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("=" * 78)
    print("COUNTRY SUMMARY (Sweden vs Romania, NUTS2 means, 2018-2022)")
    print("=" * 78)
    print(cs.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("STRONGEST POOLED CORRELATIONS (|Spearman r|, n=16)")
    print("=" * 78)
    top = (corr.assign(absr=corr.spearman_r.abs())
               .sort_values("absr", ascending=False).head(12))
    print(top[["gender_indicator", "transport_indicator", "n", "spearman_r",
               "spearman_p", "within_country_pearson_r"]].round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("SELECTED OLS REGRESSIONS (standardised predictors)")
    print("=" * 78)
    sig = reg[(reg.model == "with_gdp")].copy()
    sig = sig.sort_values("p_transport").head(12)
    print(sig[["gender_indicator", "transport_indicator", "n",
               "beta_transport_std", "p_transport", "beta_gdp_std",
               "p_gdp", "r_squared"]].round(3).to_string(index=False))

    print("\nFigures saved:")
    for p in (p1, p2, p3):
        print("  ", p)


if __name__ == "__main__":
    main()
