# PV Module Supply–Demand Alignment with Lead-Time Adjustment (China)

This repository provides a transparent, reproducible analysis of the apparent
mismatch between China’s solar PV module production and installation statistics.
By explicitly accounting for project procurement and construction lead times,
the analysis shows that much of the perceived “overproduction” at the module
level reflects timing differences rather than structural oversupply.

---

## Key Insight

**A lead time of approximately 6–7 months best aligns domestic PV module supply
with implied domestic demand in China.**

When this timing is taken into account:
- Domestic module production closely tracks installations plus exports
- Residuals fluctuate around zero, with no persistent excess stock
- The inferred lead time is consistent with real-world PV project cycles
  (procurement → construction → grid connection)

---

## What the Analysis Does

1. **Domestic available supply**  
   Annual PV module production minus exports.

2. **Implied domestic demand**  
   Domestic installations are distributed monthly using observed quarterly
   patterns and shifted backward by 1–12 months to reflect procurement timing.

3. **Lead-time grid search**  
   Model fit is evaluated using MAE and RMSE to identify the lead time that best
   aligns supply and demand over 2015–2023.

---

## Data Sources

All input data are publicly available and provided in machine-readable CSV format
under `data/raw/`, with direct links to original sources.

- **PV module production:** China Photovoltaic Industry Association (CPIA)  
- **PV module exports:** CPIA and CCCME, compiled from Eastmoney securities
  research and official trade statistics  
- **Domestic PV installations:** National Energy Administration (NEA), including
  annual and quarterly cumulative data

---

## Repository Structure

├── PV_lead_alignment.py
├── README.md
├── data/
│   ├── raw/        # Input CSV files with source references
│   ├── output/     # Auto-generated results (ignored by Git)
│   └── sources.md  # Detailed data source documentation
└── .gitignore

---

## How to Run

From the repository root:

```bash
pip install pandas numpy matplotlib
python PV_lead_alignment.py
