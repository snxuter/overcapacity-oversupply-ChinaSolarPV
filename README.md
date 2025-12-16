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
