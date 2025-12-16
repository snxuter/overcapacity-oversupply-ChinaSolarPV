#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PV Module Supply–Demand Alignment with Lead-Time Adjustment
===========================================================

This script reproduces the lead-time alignment analysis using machine-readable
CSV inputs (stored under `data/raw/`). It compares:

  Domestic available module supply  = Production − Exports   (annual, GW)

against

  Implied domestic module demand    = Installations shifted backward by L months
                                      (annualized from monthly series, GW)

where L is a procurement & construction lead time (1–12 months).

Why lead time?
--------------
PV projects typically procure modules months before grid connection. Annual
installation statistics therefore reflect demand with a delay relative to
production and shipments. Introducing a lead time helps distinguish timing
mismatches from structural overproduction at the module level.

Inputs (CSV)
------------
Expected files under `data/raw/` (filenames can be changed via CLI flags):

- china_pv_module_production.csv
    columns: year, production_gw, [source_url], [source_note]

- china_pv_module_exports.csv
    columns: year, export_gw, [source_url], [source_note]

- china_pv_module_installation.csv
    columns: year, installation_gw, [source_url], [source_note]

- china_pv_installation_quarterly_cumulative.csv
    columns: year, Q1_cum_gw, H1_cum_gw, Q3_cum_gw, Year_cum_gw, [source]

Quarterly cumulative installation is used to infer intra-year seasonality. We
assume installations are evenly distributed across months within each quarter.

Outputs
-------
By default, outputs are written to:
    data/output/<run_id>/

where <run_id> is a timestamp (e.g., 2025-12-16_141530). This avoids accidental
overwrites and makes results auditable.

Exported artifacts include:
- Intermediate tables (CSV):
    * annual_inputs.csv
    * quarterly_cumulative.csv
    * quarterly_increments_and_shares.csv
    * monthly_installations.csv
    * grid_search_metrics_<start>_<end>.csv
    * best_lead_timeseries_<start>_<end>.csv
- Figures (PNG + PDF):
    * mae_rmse_grid_<start>_<end>.{png,pdf}
    * supply_vs_implied_demand_lead6_7_<start>_<end>.{png,pdf}
    * supply_vs_implied_demand_bestLead_<start>_<end>.{png,pdf}
    * residual_share_bestLead_<start>_<end>.{png,pdf}

Usage
-----
Basic run (from repo root):
    python PV_lead_alignment.py

Common options:
    python PV_lead_alignment.py --eval-start 2015 --eval-end 2023
    python PV_lead_alignment.py --output-dir data/output --run-id my_test_run

Dependencies
------------
pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Plot defaults (publication-friendly)
# -----------------------------
FIGSIZE_2K = (10, 5.625)  # 2560x1440 when dpi=256
DPI_2K = 256

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# -----------------------------
# Data loading
# -----------------------------
def load_inputs(raw_dir: Path,
                production_file: str,
                exports_file: str,
                installation_file: str,
                quarterly_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load annual and quarterly-cumulative inputs from CSV files.

    Returns:
      annual_df: columns
          [year, production_gw, export_gw, installation_gw, supply_domestic_gw]
      qcum_df: quarterly cumulative installation dataframe
    """

    # Read CSVs
    prod = pd.read_csv(raw_dir / production_file)
    exp = pd.read_csv(raw_dir / exports_file)
    inst = pd.read_csv(raw_dir / installation_file)
    qcum = pd.read_csv(raw_dir / quarterly_file)

    # Standardize column names (defensive)
    prod.columns = prod.columns.str.strip()
    exp.columns = exp.columns.str.strip()
    inst.columns = inst.columns.str.strip()
    qcum.columns = qcum.columns.str.strip()

    # ---- Column validation (explicit & readable) ----
    required_prod = {"year", "production_gw"}
    required_exp = {"year", "export_gw"}
    required_inst = {"year", "installation_gw"}
    required_q = {"year", "Q1_cum_gw", "H1_cum_gw", "Q3_cum_gw", "Year_cum_gw"}

    if not required_prod.issubset(prod.columns):
        raise ValueError(
            f"{production_file} missing columns: "
            f"{sorted(required_prod - set(prod.columns))}"
        )

    if not required_exp.issubset(exp.columns):
        raise ValueError(
            f"{exports_file} missing columns: "
            f"{sorted(required_exp - set(exp.columns))}"
        )

    if not required_inst.issubset(inst.columns):
        raise ValueError(
            f"{installation_file} missing columns: "
            f"{sorted(required_inst - set(inst.columns))}"
        )

    if not required_q.issubset(qcum.columns):
        raise ValueError(
            f"{quarterly_file} missing columns: "
            f"{sorted(required_q - set(qcum.columns))}"
        )

    # ---- Merge annual data ----
    annual = (
        prod[["year", "production_gw"]]
        .merge(exp[["year", "export_gw"]], on="year", how="inner")
        .merge(inst[["year", "installation_gw"]], on="year", how="inner")
        .sort_values("year")
        .reset_index(drop=True)
    )

    annual["supply_domestic_gw"] = annual["production_gw"] - annual["export_gw"]

    # Keep only needed columns for quarterly cumulative
    qcum = (
        qcum[["year", "Q1_cum_gw", "H1_cum_gw", "Q3_cum_gw", "Year_cum_gw"]]
        .sort_values("year")
        .reset_index(drop=True)
    )

    return annual, qcum


# -----------------------------
# Quarterly → monthly installation series
# -----------------------------
def quarterly_increments_and_shares(qcum_df: pd.DataFrame) -> pd.DataFrame:
    df = qcum_df.copy()
    df["Q1"] = df["Q1_cum_gw"]
    df["Q2"] = df["H1_cum_gw"] - df["Q1_cum_gw"]
    df["Q3"] = df["Q3_cum_gw"] - df["H1_cum_gw"]
    df["Q4"] = df["Year_cum_gw"] - df["Q3_cum_gw"]

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df[f"{q}_share"] = df[q] / df["Year_cum_gw"]
    return df


def build_monthly_installations(qinc_df: pd.DataFrame) -> pd.Series:
    start = f"{int(qinc_df['year'].min())}-01-01"
    end = f"{int(qinc_df['year'].max())}-12-01"
    months = pd.date_range(start, end, freq="MS")
    inst = pd.Series(index=months, dtype=float)

    for _, row in qinc_df.iterrows():
        y = int(row["year"])
        weights = np.array(
            [row["Q1_share"] / 3] * 3 +
            [row["Q2_share"] / 3] * 3 +
            [row["Q3_share"] / 3] * 3 +
            [row["Q4_share"] / 3] * 3
        )
        idx = pd.date_range(f"{y}-01-01", f"{y}-12-01", freq="MS")
        inst.loc[idx] = float(row["Year_cum_gw"]) * weights

    return inst


# -----------------------------
# Lead alignment: implied annual demand
# -----------------------------
def implied_demand_annual(inst_monthly: pd.Series, lead_months: int) -> pd.Series:
    """
    Shift monthly installations backward by lead_months to represent implied demand timing.

    We use shift(-lead) such that implied demand in month t corresponds to installations at t+lead.
    Annual aggregation is strict: if any shifted month in a year is missing (NaN), that year is dropped.
    """
    shifted = inst_monthly.shift(-lead_months)

    def year_sum_strict(x: pd.Series) -> float:
        return np.nan if x.isna().any() else float(x.sum())

    annual = shifted.groupby(shifted.index.year).agg(year_sum_strict).dropna()
    annual.name = f"implied_demand_lead_{lead_months}_gw"
    return annual


@dataclass
class FitResult:
    lead_months: int
    mae_gw: float
    rmse_gw: float
    n_years: int


def compute_metrics_for_lead(annual_df: pd.DataFrame,
                             inst_monthly: pd.Series,
                             lead: int,
                             start_year: int,
                             end_year: int) -> Tuple[FitResult, pd.DataFrame]:
    implied = implied_demand_annual(inst_monthly, lead)

    df = annual_df.merge(implied.rename("implied_demand_gw"), left_on="year", right_index=True, how="left")
    df_eval = df[(df["year"] >= start_year) & (df["year"] <= end_year)].dropna(subset=["implied_demand_gw"]).copy()

    df_eval["residual_gw"] = df_eval["supply_domestic_gw"] - df_eval["implied_demand_gw"]
    df_eval["abs_residual_gw"] = df_eval["residual_gw"].abs()

    mae = float(df_eval["abs_residual_gw"].mean())
    rmse = float(math.sqrt(np.mean(df_eval["residual_gw"] ** 2)))
    n = int(len(df_eval))

    return FitResult(lead, mae, rmse, n), df_eval


def grid_search(annual_df: pd.DataFrame,
                inst_monthly: pd.Series,
                lead_min: int,
                lead_max: int,
                start_year: int,
                end_year: int) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    metrics = []
    details: Dict[int, pd.DataFrame] = {}

    for lead in range(lead_min, lead_max + 1):
        fit, detail = compute_metrics_for_lead(annual_df, inst_monthly, lead, start_year, end_year)
        metrics.append({
            "lead_months": fit.lead_months,
            "mae_gw": fit.mae_gw,
            "rmse_gw": fit.rmse_gw,
            "n_years": fit.n_years,
        })
        details[lead] = detail

    metrics_df = pd.DataFrame(metrics).sort_values("lead_months").reset_index(drop=True)
    return metrics_df, details


# -----------------------------
# Plotting helpers
# -----------------------------
def save_png_pdf(basepath: Path):
    """Helper to return .png and .pdf paths from a base path without suffix."""
    return basepath.with_suffix(".png"), basepath.with_suffix(".pdf")


def plot_mae_rmse_together(metrics_df: pd.DataFrame, outbase: Path, title: str, mark_leads=(6, 7)):
    plt.figure(figsize=FIGSIZE_2K, dpi=DPI_2K)
    plt.plot(metrics_df["lead_months"], metrics_df["mae_gw"], marker="o", label="MAE (GW)")
    plt.plot(metrics_df["lead_months"], metrics_df["rmse_gw"], marker="s", label="RMSE (GW)")

    # Optional vertical reference lines
    for L in mark_leads:
        plt.axvline(L, linestyle="--", linewidth=1, alpha=0.6, label=f"Lead = {L} months")

    plt.xlabel("Lead (months)")
    plt.ylabel("Error (GW)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png, pdf = save_png_pdf(outbase)
    plt.savefig(png, dpi=DPI_2K)
    plt.savefig(pdf)
    plt.close()


def plot_supply_vs_implied(detail_df: pd.DataFrame, lead: int, outbase: Path, title: str):
    plt.figure(figsize=FIGSIZE_2K, dpi=DPI_2K)
    plt.plot(detail_df["year"], detail_df["supply_domestic_gw"], marker="o", linewidth=2, color="black",
             label="Domestic available supply (P − E)")
    plt.plot(detail_df["year"], detail_df["implied_demand_gw"], marker="s",
             label=f"Implied module demand (Lead = {lead} months)")

    plt.xlabel("Year")
    plt.ylabel("GW")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png, pdf = save_png_pdf(outbase)
    plt.savefig(png, dpi=DPI_2K)
    plt.savefig(pdf)
    plt.close()


def plot_supply_vs_implied_lead6_7(detail_6: pd.DataFrame, detail_7: pd.DataFrame, outbase: Path, title: str):
    plt.figure(figsize=FIGSIZE_2K, dpi=DPI_2K)
    # Supply line (same years)
    plt.plot(detail_6["year"], detail_6["supply_domestic_gw"], marker="o", linewidth=2, color="black",
             label="Domestic available supply (P − E)")
    # Lead 6 and lead 7 implied demand
    plt.plot(detail_6["year"], detail_6["implied_demand_gw"], linestyle="--", marker="s",
             label="Implied module demand (Lead = 6 months)")
    plt.plot(detail_7["year"], detail_7["implied_demand_gw"], linestyle=":", marker="^",
             label="Implied module demand (Lead = 7 months)")

    plt.xlabel("Year")
    plt.ylabel("GW")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png, pdf = save_png_pdf(outbase)
    plt.savefig(png, dpi=DPI_2K)
    plt.savefig(pdf)
    plt.close()


def plot_residual_share(detail_df: pd.DataFrame, outbase: Path, title: str):
    rs = detail_df["residual_gw"] / detail_df["supply_domestic_gw"]
    plt.figure(figsize=FIGSIZE_2K, dpi=DPI_2K)
    plt.plot(detail_df["year"], rs, marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Residual / (P − E)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    png, pdf = save_png_pdf(outbase)
    plt.savefig(png, dpi=DPI_2K)
    plt.savefig(pdf)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PV module supply–demand lead-time alignment (CSV-driven).")
    p.add_argument("--raw-dir", default="data/raw", help="Directory containing input CSV files (default: data/raw).")
    p.add_argument("--output-dir", default="data/output", help="Base output directory (default: data/output).")
    p.add_argument("--run-id", default=None, help="Optional run ID subfolder name. Default: timestamp.")
    p.add_argument("--eval-start", type=int, default=2015, help="Evaluation start year (default: 2015).")
    p.add_argument("--eval-end", type=int, default=2023, help="Evaluation end year (default: 2023).")
    p.add_argument("--lead-min", type=int, default=1, help="Minimum lead time in months (default: 1).")
    p.add_argument("--lead-max", type=int, default=12, help="Maximum lead time in months (default: 12).")

    p.add_argument("--production-file", default="china_pv_module_production.csv")
    p.add_argument("--exports-file", default="china_pv_module_exports.csv")
    p.add_argument("--installation-file", default="china_pv_module_installation.csv")
    p.add_argument("--quarterly-file", default="china_pv_installation_quarterly_cumulative.csv")
    return p.parse_args()


def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir.resolve()}")

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = Path(args.output_dir) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    annual_df, qcum_df = load_inputs(
        raw_dir=raw_dir,
        production_file=args.production_file,
        exports_file=args.exports_file,
        installation_file=args.installation_file,
        quarterly_file=args.quarterly_file,
    )

    # Save annual inputs for transparency
    annual_df.to_csv(outdir / "annual_inputs.csv", index=False)
    qcum_df.to_csv(outdir / "quarterly_cumulative.csv", index=False)

    # 2) Quarterly → increments & shares
    qinc_df = quarterly_increments_and_shares(qcum_df)
    qinc_df.to_csv(outdir / "quarterly_increments_and_shares.csv", index=False)

    # 3) Monthly installations
    inst_monthly = build_monthly_installations(qinc_df)
    inst_monthly.to_frame("monthly_installation_gw").to_csv(outdir / "monthly_installations.csv")

    # 4) Grid search
    metrics_df, detail_tables = grid_search(
        annual_df=annual_df,
        inst_monthly=inst_monthly,
        lead_min=args.lead_min,
        lead_max=args.lead_max,
        start_year=args.eval_start,
        end_year=args.eval_end,
    )
    metrics_df.to_csv(outdir / f"grid_search_metrics_{args.eval_start}_{args.eval_end}.csv", index=False)

    # 5) Best lead (by RMSE)
    best_row = metrics_df.loc[metrics_df["rmse_gw"].idxmin()]
    best_lead = int(best_row["lead_months"])
    best_detail = detail_tables[best_lead].copy()
    best_detail.to_csv(outdir / f"best_lead_timeseries_{args.eval_start}_{args.eval_end}.csv", index=False)

    # 6) Plots
    plot_mae_rmse_together(
        metrics_df,
        outbase=outdir / f"mae_rmse_grid_{args.eval_start}_{args.eval_end}",
        title=f"Lead Grid Search ({args.eval_start}–{args.eval_end}): MAE & RMSE",
        mark_leads=(6, 7),
    )

    # Lead 6 & 7 overlay (if available)
    if 6 in detail_tables and 7 in detail_tables:
        plot_supply_vs_implied_lead6_7(
            detail_tables[6],
            detail_tables[7],
            outbase=outdir / f"supply_vs_implied_demand_lead6_7_{args.eval_start}_{args.eval_end}",
            title="Domestic Supply (P − E) vs Implied Module Demand (Lead = 6 and 7 months)"
        )

    # Best-lead plot and residual share
    plot_supply_vs_implied(
        best_detail,
        best_lead,
        outbase=outdir / f"supply_vs_implied_demand_bestLead_{args.eval_start}_{args.eval_end}",
        title=f"Domestic Supply (P − E) vs Implied Module Demand (Best RMSE: Lead = {best_lead} months)"
    )

    plot_residual_share(
        best_detail,
        outbase=outdir / f"residual_share_bestLead_{args.eval_start}_{args.eval_end}",
        title=f"Residual Share under Best Lead (Lead = {best_lead} months, {args.eval_start}–{args.eval_end})"
    )

    # Console summary
    print("=== Grid Search Metrics ===")
    print(metrics_df.to_string(index=False))
    print(f"\nBest lead by RMSE: {best_lead} months")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
