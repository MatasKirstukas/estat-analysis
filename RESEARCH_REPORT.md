# Transport and Gender Equality across EU Regions — Sweden vs Romania

**Research question.** *Do transport indicators relate to gender-equality
indicators in selected EU countries, measured at the NUTS2 regional level?*

**Geography.** Following the EIGE Gender Equality Index 2024, the country with the
highest score (**Sweden, 82**) is compared with the lowest (**Romania, 57.5**).
The unit of analysis is the **NUTS2 region**: 8 Swedish + 8 Romanian regions
(**16 regions** in total).

**Reference period.** All indicators are averaged over **2018–2022** (mean of the
available years per region) to reduce year-to-year noise and to maximise data
coverage.

> Reproduce everything with:
> `python analysis.py` (builds the indicator table) and
> `python analyze_stats.py` (statistics + figures).
> Inputs are the Eurostat files listed in `README.md`; results are written to
> `outputs/`.

---

## 1. Variables and how they were built

### Dependent variables — gender equality (from the sex breakdown)
| Indicator | Definition | Source file |
|---|---|---|
| `emp_female_share` | Female ÷ (female+male) employed, age 20–64. **0.5 = parity** | `estat_lfst_r_lfe2_employment` |
| `unemp_gap_pp` | Female − male unemployment rate (pp), age 20–64, all education | `estat_lfst_r_lfu3rt_employment` |
| `early_leavers_gap_pp` | Male − female early school-leavers (pp), age 18–24 | `estat_edat_lfse_16_education` |
| `gender_gap_magnitude` | Composite z-score of the **absolute** size of the three gaps (higher = **less** equal) | derived |

### Independent variables — transport
| Indicator | Definition | Source file |
|---|---|---|
| `motorway_km_per_Mhab` | Motorway km per million inhabitants | `estat_tran_r_net` |
| `rail_km_per_Mhab` | Railway-line km per million inhabitants | `estat_tran_r_net` |
| `cars_per_1000hab` | Passenger cars per 1 000 inhabitants | `estat_tran_r_vehst` |
| `road_deaths_per_Mhab` | Persons killed in road accidents per million | `estat_tran_r_acci` |
| `road_injuries_per_Mhab` | Persons injured per million | `estat_tran_r_acci` |
| `air_pax_per_capita` | Air passengers carried per inhabitant | `estat_tran_r_avpa_nm` |

### Optional control
| Indicator | Definition | Source file |
|---|---|---|
| `gdp_per_capita_eur` | GDP per inhabitant (EUR) | `estat_nama_10r_2gdp` |

**Technical notes.**
* Regional **population** was derived from the GDP file
  (`MIO_EUR ÷ EUR_HAB`) and used to convert absolute car-stock and
  air-passenger counts into per-capita rates (the per-capita vehicle unit
  `P_THAB` is empty for SE/RO).
* **Rail passengers (`estat_tran_r_rapa`)** is an origin–destination matrix.
  For Sweden it is only populated around 2005 and for Romania only from 2010,
  so it is **not comparable** across the two countries and was excluded from the
  statistical models. Rail provision is instead captured by **railway network
  length**. (The flows remain available in the raw file for descriptive use.)
* Eurostat flag letters (`e`, `p`, `b`, `:` , `@C`) are parsed out; `:`/`@C`
  are treated as missing.

---

## 2. Sweden vs Romania — descriptive comparison

NUTS2-region means, 2018–2022 (Mann–Whitney U test between the two groups):

| Indicator | Sweden | Romania | SE − RO | p |
|---|---:|---:|---:|---:|
| Female share of employment | 0.471 | 0.423 | +0.047 | **0.010** |
| Unemployment gap F−M (pp) | −0.02 | −0.83 | +0.81 | 0.328 |
| Early-leavers gap M−F (pp) | 3.02 | −0.08 | +3.10 | 0.073 |
| **Gender-gap magnitude (↓=equal)** | −0.40 | +0.37 | −0.77 | **0.007** |
| Motorway km / M inhab. | 173 | 50 | +124 | **0.028** |
| Railway km / M inhab. | 1 703 | 578 | +1 125 | **0.050** |
| Cars / 1 000 inhab. | 501 | 381 | +120 | **0.015** |
| Road deaths / M inhab. | 28.5 | 90.8 | −62.3 | **0.000** |
| Road injuries / M inhab. | 1 662 | 1 804 | −142 | 0.234 |
| Air passengers / capita | 2.45 | 1.00 | +1.45 | **0.038** |
| GDP per capita (EUR) | 45 960 | 12 488 | +33 473 | **0.000** |

**Reading.** Swedish regions are simultaneously **more gender-equal** (higher
female employment share, smaller composite gap) **and** more transport-rich
(more motorway/rail length, more cars, more air travel) and far **safer** on the
roads (one-third of Romania's road-death rate). Romania's roads are markedly more
lethal despite a similar injury rate, indicating more severe crashes.

See `outputs/fig_country_comparison.png`.

---

## 3. Do transport indicators relate to gender equality?

### 3.1 Pooled correlations (16 regions)
Strongest associations (Spearman *r*; `outputs/correlations.csv`,
`outputs/fig_correlation_heatmap.png`):

| Gender indicator | Transport indicator | *r*ₛ | p | within-country *r* |
|---|---|---:|---:|---:|
| Female employment share | Air passengers / capita | **+0.86** | 0.000 | +0.62 |
| Female employment share | Road deaths / M inhab. | **−0.84** | 0.000 | **−0.87** |
| Gender-gap magnitude | GDP per capita | −0.79 | 0.000 | −0.31 |
| Female employment share | Cars / 1 000 inhab. | +0.69 | 0.003 | +0.61 |
| Gender-gap magnitude | Air passengers / capita | −0.70 | 0.005 | −0.28 |
| Gender-gap magnitude | Road deaths / M inhab. | +0.69 | 0.003 | +0.32 |

Directionally, **more/safer transport goes with greater gender equality**:
regions with more air connectivity, more cars and—above all—**fewer road
deaths** have a higher female employment share and a smaller overall gender gap.

### 3.2 Controlling for economic development (OLS, standardised predictors)
The two countries differ enormously in GDP per capita, and GDP is itself the
strongest correlate of the gender gap. When `log(GDP per capita)` is added as a
control (`outputs/regressions.csv`):

* The associations of **cars, rail, motorways and air travel** with the female
  employment share **lose significance** (transport *p* > 0.25); GDP absorbs the
  effect. These transport measures are essentially **markers of prosperity**.
* **Road safety is the exception.** Female employment share vs road-death rate
  keeps a negative transport coefficient that stays borderline significant
  with GDP in the model (*p* ≈ 0.05), and its **within-country** correlation is
  the strongest in the whole study (*r* = −0.87). This is the most
  transport-specific signal: independent of national wealth, **regions with
  safer roads tend to be more gender-equal**.

See `outputs/fig_scatter_key.png` for the strongest relationships, coloured by
country.

---

## 4. Answer to the research objective

**Yes — transport indicators are statistically related to gender-equality
indicators across the Swedish and Romanian NUTS2 regions, but the relationship is
largely an expression of overall economic development.**

1. At the country level Sweden is both more gender-equal and more
   transport-developed/safer than Romania, so almost every transport measure
   correlates with every gender measure in the pooled sample.
2. Once GDP per capita is controlled, the infrastructure/ownership/air-travel
   links fade — they are proxies for prosperity rather than independent drivers.
3. The one robust, development-independent association is **road safety**:
   lower road-death rates accompany higher female employment and smaller gender
   gaps, and this holds **within** each country.

### Limitations
* Only **16 regions** → low statistical power; results are exploratory and
  correlational, **not causal**.
* Two countries cannot be separated from all other national differences;
  pooled correlations conflate "transport" with "being Sweden vs Romania".
* Employment equality uses **head-counts** (no regional employment-*rate* by sex
  in the supplied files), so it partly reflects population structure.
* Rail passenger flows (`tran_r_rapa`) were too sparse/misaligned to use.

### Suggested next iteration
Add more countries spanning the EIGE index (to raise N and break the
Sweden-vs-Romania confound), use sex-specific employment **rates**, and add a
time dimension (panel/fixed-effects) to move beyond a single cross-section.

---

### Output files (`outputs/`)
| File | Contents |
|---|---|
| `region_indicators.csv` | The 16-region × indicator table |
| `country_summary.csv` | SE vs RO means + Mann–Whitney p |
| `correlations.csv` | Pooled & within-country Spearman/Pearson (+p) |
| `regressions.csv` | OLS gender ~ transport (+GDP) |
| `fig_country_comparison.png` | SE vs RO bar charts |
| `fig_correlation_heatmap.png` | Spearman heatmap |
| `fig_scatter_key.png` | Strongest relationships, by country |
