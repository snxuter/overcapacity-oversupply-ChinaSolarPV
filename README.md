# PV Module Supply–Demand Alignment (China)

**Why does China’s solar PV manufacturing look “overbuilt” in annual statistics, even when installations remain strong?**  
This repository shows that much of the apparent mismatch reflects **timing**, not structural overproduction.

By accounting for a realistic **project procurement and construction lead time**, China’s PV module production closely aligns with **domestic installations plus exports**.

---

## Key Insight

📌 **A 6–7 month lead time best explains the alignment between supply and demand.**

Once this timing is considered:
- Apparent overproduction at the **module** level largely disappears
- Residuals fluctuate around zero, with no evidence of persistent excess stock
- The inferred lead time matches real-world PV project cycles in China

---

## What This Repo Does

**Data → Timing → Insight**

1. **Domestic supply**  
   PV module production − exports

2. **Implied demand**  
   Installations distributed monthly → shifted backward by 1–12 months

3. **Grid search**  
   Identify the lead time that minimizes MAE & RMSE (2015–2023)

📊 Outputs include:
- Lead-time error curves (MAE & RMSE)
- Supply vs. implied demand (Lead = 6 & 7 months)
- Residuals as a share of supply

---

## Data Sources

- **Production:** China Photovoltaic Industry Association (CPIA)  
- **Exports:** CPIA, CCCME, Eastmoney securities research  
- **Installations:** National Energy Administration (NEA)

All data are publicly available and expressed in GW.

---

## Quick Start

```bash
pip install pandas numpy matplotlib
python PV_lead_alignment.py
