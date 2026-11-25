# Verdant Leaf — Marketing Mix Model (MMM) Refinement & Q1 2026 Budget Plan

Reproducible Python codebase for the Honasa Consumer Ltd Marketing Mix Modeling assignment.

## Project Structure
- `data_cleaning.py` — Automated raw export audit and remediation (Q4 2023 Meta Lakhs unit fix, YouTube CPM imputation, negative TV credit-note clamp, exact duplicate removal).
- `mmm_model.py` — Baseline v0 OLS replication, `organic_traffic_index` bad-control demonstration, Stage-1 `brand_search_index` residualization, Bayesian MAP Concave Adstock-Hill MMM with Meta geo-holdout calibration, and circular block-bootstrap uncertainty quantification.
- `budget_optimizer.py` — Constrained nonlinear Q1 2026 (₹16 Cr) budget optimizer with 50%–150% channel guardrails, realistic burst flighting for TV & Influencer, and bootstrap payoff/stability evaluation.
- `run_analysis.py` — End-to-end runner that reproduces all metrics, saves `outputs/analysis_summary.json`, and generates all 5 report figures in `outputs/figures/`.
- `generate_report.py` — Builds the 6-page executive PDF report (`Verdant_Leaf_MMM_Report.pdf`) using ReportLab.

## How to Run
```bash
pip install -r requirements.txt
python run_analysis.py
python generate_report.py
```
Note: Place `mmm_weekly_data.csv` and `lift_test_results.csv` in `../pack/` (or update `DATA_CSV` in `run_analysis.py`).
