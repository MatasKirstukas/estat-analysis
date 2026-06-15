# 1. Research Objective

- Do transport indicators relate to gender equality indicators in selected EU countries (by NUTS 2 regions)?

## 2. Research Geography

- Based on the Gender Equality Index: <https://eige.europa.eu/gender-equality-index/2024/compare-countries/index/graph>
- For the first iteration, countries with the highest and lowest scores are compared:
	- Sweden: 82
	- Romania: 57.5

## 3. Variables

- **Dependent Variables (Outcomes):**
	- Employment by sex, age and NUTS 2 region (1 000) (`lfst_r_lfe2emp`)
	- Unemployment rates by sex, age, educational attainment level and NUTS 2 region (%) (`lfst_r_lfu3rt`)
	- Early leavers from education and training by sex and NUTS 2 region (`edat_lfse_16`)

- **Independent Variables (Predictors):**
	- Road, rail and navigable inland waterways networks by NUTS 2 region (`tran_r_net`)
	- Stock of vehicles by category and NUTS 2 region (`tran_r_vehst`)
	- Victims in road accidents by NUTS 2 region (`tran_r_acci`)
	- Air transport of passengers by NUTS 2 region (`tran_r_avpa_nm`)
	- Railway transport - national and international railway passengers transport by embarking/disembarking NUTS 2 region (`tran_r_rapa`)

- **Optional Independent Variables (choose one):**
	- Gross domestic product (GDP) at current market prices by NUTS 2 region (`nama_10r_2gdp`)
	- Gross domestic product (GDP) and Gross value added (GVA) in volume by NUTS 2 region (`nama_10r_2gvagr`)

