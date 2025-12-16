#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PV Module Production vs (Domestic Installation + Exports) with Lead-Time Alignment
===============================================================================

This script reproduces the full analysis we discussed:

1) Inputs (annual):
   - China PV module production (GW): P_y
   - China PV module exports (GW):     E_y
   - China PV PV installation (GW):    I_y  (newly grid-connected capacity)

2) We treat exports as "immediately absorbed" in the same calendar year, so the
   annual domestic-available module supply is:
        Supply_y = P_y - E_y

3) Installation does NOT consume modules instantaneously from the same-year supply.
   Projects place module orders in advance. We model a fixed lead-time (L months)
   between "module production/availability" and "project grid-connection".

   Operationally:
   - Build a monthly installation series I_t by distributing each year's installation
     across 12 months using quarterly weights inferred from cumulative quarterly data.
   - Shift the monthly installation series backward by L months to represent the
     implied monthly demand needed for those installations.
   - Aggregate to annual implied demand:
        ImpliedProd_y(L) = sum_{t in year y} I_{t+L}   (implemented via shift(-L))

4) Grid search:
   - Evaluate L = 1..12 months
   - Evaluate model fit over years 2015–2023 (or 2016–2023 if you want to exclude 2015)
   - Metrics: MAE, RMSE between Supply_y and ImpliedProd_y(L)

5) Outputs:
   - Intermediate tables (CSV):
       * input_annual_data.csv
       * quarterly_cumulative_installation.csv
       * quarterly_increments_and_shares.csv
       * monthly_installation_series.csv
       * grid_search_metrics_2015_2023.csv
       * best_lead_timeseries_2015_2023.csv
   - Figures (PNG):
       * mae_grid_2015_2023.png
       * rmse_grid_2015_2023.png
       * supply_vs_implied_bestLead_2015_2023.png
       * residual_share_bestLead_2015_2023.png

Run:
    python pv_lead_alignment_full.py

Dependencies:
    pandas, numpy, matplotlib
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 0) Configuration
# -----------------------------
OUTDIR = Path("./pv_lead_alignment_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

EVAL_START_YEAR = 2015   # set to 2016 if you want to exclude the "left-edge" year
EVAL_END_YEAR   = 2023   # do NOT include 2024 in error evaluation (needs 2025 installs)
LEAD_MIN = 1
LEAD_MAX = 12


# -----------------------------
# 1) Raw Inputs (FINAL DATASET)
# -----------------------------
# Production (Module) - GW
PRODUCTION_GW = {
    2015: 45.8,
    2016: 53.7,
    2017: 75.0,
    2018: 85.7,
    2019: 98.6,
    2020: 124.6,
    2021: 182.0,
    2022: 288.7,
    2023: 499.0,
    2024: 588.0,
}

# Exports (Module) - GW
EXPORT_GW = {
    2015: 24.0,
    2016: 21.3,
    2017: 31.5,
    2018: 41.6,
    2019: 66.6,
    2020: 78.8,
    2021: 98.5,
    2022: 153.0,
    2023: 211.0,
    2024: 238.8,
}

# Domestic installations (GW)
INSTALL_GW = {
    2015: 15.13,
    2016: 34.54,
    2017: 53.06,
    2018: 44.26,
    2019: 30.10,
    2020: 48.20,
    2021: 54.88,
    2022: 87.408,
    2023: 216.3,
    2024: 277.57,
}

# Quarterly cumulative installation (GW): Q1 cum, H1 cum, Q3 cum, Full-year
# These are the same values used earlier to infer intra-year seasonality.
QUARTERLY_CUM_INSTALL_GW = {
    2015: {"Q1_cum": 5.04,   "H1_cum": 7.73,    "Q3_cum": 9.90,    "Year_cum": 15.13},
    2016: {"Q1_cum": 7.14,   "H1_cum": 13.01,   "Q3_cum": 26.00,   "Year_cum": 34.54},
    2017: {"Q1_cum": 7.21,   "H1_cum": 24.40,   "Q3_cum": 42.00,   "Year_cum": 53.06},
    2018: {"Q1_cum": 9.52,   "H1_cum": 24.306,  "Q3_cum": 34.544,  "Year_cum": 44.26},
    2019: {"Q1_cum": 5.20,   "H1_cum": 11.40,   "Q3_cum": 15.99,   "Year_cum": 30.10},
    2020: {"Q1_cum": 3.95,   "H1_cum": 11.52,   "Q3_cum": 18.70,   "Year_cum": 48.20},
    2021: {"Q1_cum": 5.33,   "H1_cum": 13.01,   "Q3_cum": 25.556,  "Year_cum": 54.88},
    2022: {"Q1_cum": 13.21,  "H1_cum": 30.878,  "Q3_cum": 52.602,  "Year_cum": 87.408},
    2023: {"Q1_cum": 33.656, "H1_cum": 78.423,  "Q3_cum": 128.93,  "Year_cum": 216.30},
    2024: {"Q1_cum": 45.74,  "H1_cum": 102.48,  "Q3_cum": 160.88,  "Year_cum": 277.57},
}


# -----------------------------
# 2) Utility: Build annual input table
# -----------------------------
def build_annual_inputs() -> pd.DataFrame:
    yrs = sorted(set(PRODUCTION_GW) & set(EXPORT_GW) & set(INSTALL_GW))
    df = pd.DataFrame({
        "year": yrs,
        "production_gw": [PRODUCTION_GW[y] for y in yrs],
        "export_gw":     [EXPORT_GW[y]     for y in yrs],
        "install_gw":    [INSTALL_GW[y]    for y in yrs],
    })
    df["supply_domestic_gw"] = df["production_gw"] - df["export_gw"]  # P - E
    return df


# -----------------------------
# 3) Utility: Quarterly -> monthly installation series
# -----------------------------
def build_quarterly_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      qcum_df: quarterly cumulative values as provided
      qinc_df: quarterly increments and quarterly shares
    """
    yrs = sorted(QUARTERLY_CUM_INSTALL_GW.keys())
    qcum_df = pd.DataFrame([{"year": y, **QUARTERLY_CUM_INSTALL_GW[y]} for y in yrs])

    # Increments
    qinc_df = qcum_df.copy()
    qinc_df["Q1"] = qinc_df["Q1_cum"]
    qinc_df["Q2"] = qinc_df["H1_cum"] - qinc_df["Q1_cum"]
    qinc_df["Q3"] = qinc_df["Q3_cum"] - qinc_df["H1_cum"]
    qinc_df["Q4"] = qinc_df["Year_cum"] - qinc_df["Q3_cum"]

    # Shares
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        qinc_df[f"{q}_share"] = qinc_df[q] / qinc_df["Year_cum"]

    return qcum_df, qinc_df


def build_monthly_installation(qinc_df: pd.DataFrame) -> pd.Series:
    """
    Construct monthly installation series from quarterly shares.

    Assumption:
      - Within each quarter, installation is evenly split across 3 months.
      - For each year y:
          Jan-Mar share = Q1_share / 3 each month
          Apr-Jun share = Q2_share / 3 each month
          Jul-Sep share = Q3_share / 3 each month
          Oct-Dec share = Q4_share / 3 each month

    Output:
      pd.Series indexed by month-start timestamps, values in GW.
    """
    start = f"{qinc_df['year'].min()}-01-01"
    end   = f"{qinc_df['year'].max()}-12-01"
    months = pd.date_range(start, end, freq="MS")

    inst = pd.Series(index=months, dtype=float)

    for _, row in qinc_df.iterrows():
        y = int(row["year"])
        weights = np.array(
            [row["Q1_share"]/3]*3 +
            [row["Q2_share"]/3]*3 +
            [row["Q3_share"]/3]*3 +
            [row["Q4_share"]/3]*3
        )
        year_total = float(row["Year_cum"])
        idx = pd.date_range(f"{y}-01-01", f"{y}-12-01", freq="MS")
        inst.loc[idx] = year_total * weights

    return inst


# -----------------------------
# 4) Lead alignment: implied demand from shifted installation
# -----------------------------
def implied_demand_annual(inst_monthly: pd.Series, lead_months: int) -> pd.Series:
    """
    Shift monthly installation backward by lead_months to represent implied demand.

    Implementation note:
      - pandas shift(k) moves values forward in time if k > 0.
      - We want ImpliedProd at time t to correspond to installations at t+lead.
        Therefore we use shift(-lead).

    Edge handling:
      - If a calendar year contains ANY NaNs after shifting (because we ran out of future months),
        that year's implied demand is set to NaN and dropped.

    Returns:
      pd.Series indexed by year with implied demand(GW).
    """
    shifted = inst_monthly.shift(-lead_months)

    def year_sum_strict(x: pd.Series) -> float:
        # if any month is missing, year cannot be fully computed (avoids right-edge bias)
        return np.nan if x.isna().any() else float(x.sum())

    annual = shifted.groupby(shifted.index.year).agg(year_sum_strict).dropna()
    annual.name = f"implied_prod_lead_{lead_months}"
    return annual


# -----------------------------
# 5) Fit metrics
# -----------------------------
@dataclass
class FitResult:
    lead_months: int
    mae_gw: float
    rmse_gw: float
    n_years: int


def compute_metrics_for_lead(annual_df: pd.DataFrame, inst_monthly: pd.Series, lead: int,
                             start_year: int, end_year: int) -> tuple[FitResult, pd.DataFrame]:
    """
    Compare Supply_y = (P-E) with ImpliedProd_y(lead) over [start_year, end_year].

    Returns:
      FitResult and a detailed dataframe containing year-by-year values.
    """
    implied = implied_demand_annual(inst_monthly, lead)

    df = annual_df.merge(implied.rename("implied_demand_gw"), left_on="year", right_index=True, how="left")
    df_eval = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(subset=["implied_demand_gw"]).copy()

    df_eval["residual_gw"] = df_eval["supply_domestic_gw"] - df_eval["implied_demand_gw"]
    df_eval["abs_residual_gw"] = df_eval["residual_gw"].abs()

    mae = float(df_eval["abs_residual_gw"].mean())
    rmse = float(math.sqrt(np.mean(df_eval["residual_gw"] ** 2)))
    n = int(len(df_eval))

    return FitResult(lead, mae, rmse, n), df_eval


def grid_search(annual_df: pd.DataFrame, inst_monthly: pd.Series,
                lead_min: int, lead_max: int,
                start_year: int, end_year: int) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Run lead = lead_min..lead_max, collect MAE/RMSE and store year-by-year tables.
    """
    metrics = []
    detail_tables: dict[int, pd.DataFrame] = {}

    for lead in range(lead_min, lead_max + 1):
        fit, detail = compute_metrics_for_lead(annual_df, inst_monthly, lead, start_year, end_year)
        metrics.append({
            "lead_months": fit.lead_months,
            "mae_gw": fit.mae_gw,
            "rmse_gw": fit.rmse_gw,
            "n_years": fit.n_years,
        })
        detail_tables[lead] = detail

    metrics_df = pd.DataFrame(metrics).sort_values("lead_months").reset_index(drop=True)
    return metrics_df, detail_tables


# -----------------------------
# 6) Plotting helpers
# -----------------------------
def plot_supply_vs_implied_lead6_7(
        detail_6: pd.DataFrame,
        detail_7: pd.DataFrame,
        outfile: Path,
        title: str
):
    """
    Plot domestic supply (P − E) and implied demand under Lead = 6 and Lead = 7
    on the same figure.
    """
    plt.figure(figsize=(10, 5.625), dpi=256)
    # Domestic available supply (same for both leads)
    plt.plot(
        detail_6["year"],
        detail_6["supply_domestic_gw"],
        color="gray",
        linewidth=2,
        marker="o",
        label="Domestic supply (P − E)"
    )

    # Implied demand: Lead = 6
    plt.plot(
        detail_6["year"],
        detail_6["implied_demand_gw"],
        linestyle="--",
        marker="s",
        label="Implied demand (Lead = 6 months)"
    )

    # Implied demand: Lead = 7
    plt.plot(
        detail_7["year"],
        detail_7["implied_demand_gw"],
        linestyle=":",
        marker="^",
        label="Implied demand (Lead = 7 months)"
    )

    plt.xlabel("Year")
    plt.ylabel("GW")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile.with_suffix(".png"), dpi=256)
    plt.savefig(outfile.with_suffix(".pdf"))
    plt.close()



def plot_residual_share(detail_df: pd.DataFrame, lead: int, outfile: Path, title_suffix: str):
    # residual share = residual / (P-E)
    plt.figure(figsize=(10, 5.625), dpi=256)
    plt.rcParams["pdf.fonttype"] = 42   # TrueType
    plt.rcParams["ps.fonttype"] = 42

    rs = detail_df["residual_gw"] / detail_df["supply_domestic_gw"]
    plt.plot(detail_df["year"], rs, marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Residual / (P − E)")
    plt.title(f"Residual Share {title_suffix}")
    plt.tight_layout()
    plt.savefig(outfile.with_suffix(".png"), dpi=256)
    plt.savefig(outfile.with_suffix(".pdf"))
    plt.close()

def plot_mae_rmse_together(metrics_df: pd.DataFrame, outfile: Path, title: str):
    """
    Plot MAE and RMSE curves on the same figure.
    """
    plt.figure(figsize=(10, 5.625), dpi=256)
    plt.rcParams["pdf.fonttype"] = 42   # TrueType
    plt.rcParams["ps.fonttype"] = 42

    # --- 1. 画 MAE ---
    plt.plot(
        metrics_df["lead_months"],
        metrics_df["mae_gw"],
        marker="o",
        label="MAE (GW)"
    )

    # --- 2. 画 RMSE ---
    plt.plot(
        metrics_df["lead_months"],
        metrics_df["rmse_gw"],
        marker="s",
        label="RMSE (GW)"
    )

    # === Option A：就在这里加 ===
    plt.axvline(6, linestyle="--", linewidth=1, alpha=0.6, label="Lead = 6 months")
    plt.axvline(7, linestyle="--", linewidth=1, alpha=0.6, label="Lead = 7 months")

    # --- 3. 坐标轴 & 标题 ---
    plt.xlabel("Lead (months)")
    plt.ylabel("Error (GW)")
    plt.title(title)

    # --- 4. 图例 ---
    plt.legend()

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile.with_suffix(".png"), dpi=256)
    plt.savefig(outfile.with_suffix(".pdf"))
    plt.close()

# -----------------------------
# 7) Main pipeline
# -----------------------------
def main():
    # 7.1 Build annual input table
    annual_df = build_annual_inputs()
    annual_df.to_csv(OUTDIR / "input_annual_data.csv", index=False)

    # 7.2 Quarterly tables (cumulative + increments/shares)
    qcum_df, qinc_df = build_quarterly_tables()
    qcum_df.to_csv(OUTDIR / "quarterly_cumulative_installation.csv", index=False)
    qinc_df.to_csv(OUTDIR / "quarterly_increments_and_shares.csv", index=False)

    # 7.3 Monthly installation series
    inst_monthly = build_monthly_installation(qinc_df)
    inst_monthly.to_frame("monthly_installation_gw").to_csv(OUTDIR / "monthly_installation_series.csv")

    # 7.4 Grid search over lead
    metrics_df, detail_tables = grid_search(
        annual_df=annual_df,
        inst_monthly=inst_monthly,
        lead_min=LEAD_MIN,
        lead_max=LEAD_MAX,
        start_year=EVAL_START_YEAR,
        end_year=EVAL_END_YEAR
    )
    metrics_df.to_csv(OUTDIR / f"grid_search_metrics_{EVAL_START_YEAR}_{EVAL_END_YEAR}.csv", index=False)

    # 7.5 Identify best lead (by RMSE)
    best_row = metrics_df.loc[metrics_df["rmse_gw"].idxmin()]
    best_lead = int(best_row["lead_months"])
    best_detail = detail_tables[best_lead].copy()
    best_detail.to_csv(OUTDIR / f"best_lead_timeseries_{EVAL_START_YEAR}_{EVAL_END_YEAR}.csv", index=False)


    # 7.6 Plot MAE & RMSE together
    plot_mae_rmse_together(
        metrics_df,
        outfile=OUTDIR / f"mae_rmse_grid_{EVAL_START_YEAR}_{EVAL_END_YEAR}.png",
        title=f"Lead Grid Search ({EVAL_START_YEAR}–{EVAL_END_YEAR}): MAE & RMSE"
    )

    # 7.7 Plot best-lead fits
    detail_6 = detail_tables[6]
    detail_7 = detail_tables[7]

    plot_supply_vs_implied_lead6_7(
        detail_6=detail_6,
        detail_7=detail_7,
        outfile=OUTDIR / "supply_vs_implied_lead6_lead7_2015_2023",
        title="China PV Module Supply vs Implied Domestic Demand under Different Lead Times"
    )

    plot_residual_share(
        best_detail, best_lead,
        outfile=OUTDIR / f"residual_share_bestLead_{EVAL_START_YEAR}_{EVAL_END_YEAR}.png",
        title_suffix=f"(Best RMSE: Lead={best_lead}, {EVAL_START_YEAR}–{EVAL_END_YEAR})"
    )

    # 7.8 Print a short console summary
    print("=== Grid Search Summary ===")
    print(metrics_df.to_string(index=False))
    print("\nBest lead by RMSE:", best_lead)
    print("Outputs saved to:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
