# Data Documentation

This directory contains all machine-readable input data used in the analysis of
China’s solar PV module supply–demand alignment. All datasets are compiled from
publicly available sources and provided in CSV format to ensure transparency
and reproducibility.

Units throughout are **gigawatts (GW)** unless otherwise noted.

---

## Directory Structure

```text
data/
├── raw/
│   ├── china_pv_module_production.csv
│   ├── china_pv_module_exports.csv
│   ├── china_pv_module_installation.csv
│   └── china_pv_installation_quarterly_cumulative.csv
└── README.md

1. PV Module Production (China)

File: raw/china_pv_module_production.csv

Description

Annual production of crystalline silicon solar PV modules in China.

Columns
	•	year
Calendar year.
	•	production_gw
Total PV module production in China.

Source
	•	China Photovoltaic Industry Association (CPIA)
	•	Compiled from publicly released CPIA statistics and securities research
summaries (e.g., Eastmoney).

⸻

2. PV Module Exports (China)

File: raw/china_pv_module_exports.csv

Description

Annual export volumes of solar PV modules from China.

Columns
	•	year
Calendar year.
	•	export_gw
Total PV module exports from China.
	•	source_url
Direct link to the original public report or announcement.
	•	source_note
Institution providing or compiling the data.

Source
	•	China Photovoltaic Industry Association (CPIA)
	•	China Chamber of Commerce for Import and Export of Machinery and Electronic
Products (CCCME)
	•	Public securities research reports (Eastmoney)

⸻

3. Domestic PV Installations (China, Annual)

File: raw/china_pv_module_installation.csv

Description

Annual newly installed grid-connected solar PV capacity in mainland China.

Columns
	•	year
Calendar year.
	•	installation_gw
Newly installed PV capacity in China.
	•	source_url
Link to the original public data release (when available).
	•	source_note
Data provider or compilation note.

Source
	•	National Energy Administration (NEA)

⸻

4. Domestic PV Installations (China, Quarterly Cumulative)

File: raw/china_pv_installation_quarterly_cumulative.csv

Description

Quarterly cumulative PV installation data used to infer intra-year seasonal
patterns of project grid connection.

Columns
	•	year
Calendar year.
	•	Q1_cum_gw
Cumulative installations from January to March.
	•	H1_cum_gw
Cumulative installations from January to June.
	•	Q3_cum_gw
Cumulative installations from January to September.
	•	Year_cum_gw
Total annual installations (January to December).

Source
	•	National Energy Administration (NEA)

⸻

Notes on Data Use
	•	All data are annual totals unless otherwise specified.
	•	Quarterly cumulative installation data are used to derive approximate monthly
installation profiles under the assumption of uniform distribution within
each quarter.
	•	These datasets are intended for applied research and policy analysis rather
than real-time market tracking.

⸻

Disclaimer

While every effort has been made to ensure consistency across sources, minor
differences may exist between compilations published by different institutions.
Users are encouraged to refer to the original source documents linked in each
dataset for official figures.


