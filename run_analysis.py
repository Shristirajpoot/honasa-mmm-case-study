"""
run_analysis.py — Master reproducible pipeline for Verdant Leaf MMM assignment.
Runs data audit, baseline v0, ablations, calibrated MMM, OOS validation,
block-bootstrap uncertainty, Q1 2026 budget optimization, and generates all charts + summary JSON.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_cleaning import load_and_clean_data, CHANNELS, CHANNEL_LABELS
from mmm_model import (
    run_baseline_ols,
    run_bad_control_demo,
    fit_mmm,
    bootstrap_mmm,
    transform_channel,
)
from budget_optimizer import (
    optimize_q1_budget,
    evaluate_bootstrap_uncertainty,
    simulate_quarter_media_revenue,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_csv() -> str:
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(sys.argv[1])
    candidates.extend([
        os.path.join(BASE_DIR, "mmm_weekly_data.csv"),
        os.path.join(os.path.dirname(BASE_DIR), "mmm_weekly_data.csv"),
        os.path.join(os.path.dirname(BASE_DIR), "pack", "mmm_weekly_data.csv"),
        os.path.join(os.getcwd(), "mmm_weekly_data.csv"),
    ])
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not locate mmm_weekly_data.csv. Checked: {candidates}")


DATA_CSV = _resolve_data_csv()
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Clean visual palette (Honasa / Verdant Leaf botanical emerald + warm slate/amber accents)
C_PRIMARY = "#1B5E4A"   # Deep Verdant Green
C_ACCENT = "#2E8B57"    # Sea Green
C_TEAL = "#0F766E"
C_AMBER = "#D97706"
C_CORAL = "#C2410C"
C_RED = "#B91C1C"
C_SLATE = "#475569"
C_LIGHT = "#E2E8F0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def make_figures(raw_df, df, baseline, fit_cal, fit_oos, oos_metrics, opt, unc):
    # ------------------------------------------------------------------
    # FIG 1: Data Quality Audit (Meta Q4 2023 Lakhs bug + CPM check)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), dpi=220)
    raw_u = raw_df.drop_duplicates("week_start").copy()
    raw_u["week_start"] = pd.to_datetime(raw_u["week_start"])
    raw_u = raw_u.sort_values("week_start")

    ax = axes[0]
    ax.plot(raw_u["week_start"], raw_u["meta_spend"] / 1e5, color=C_RED, lw=1.6, alpha=0.85, label="Raw export (Q4 2023 drops to ~0)")
    ax.plot(df["week_start"], df["meta_spend"] / 1e5, color=C_PRIMARY, lw=1.6, linestyle="--", label="Cleaned (×100,000 Lakhs fix)")
    ax.axvspan(pd.Timestamp("2023-10-02"), pd.Timestamp("2023-12-25"), color="#FEE2E2", alpha=0.65, label="13-week unit error window")
    ax.set_title("A. Meta Spend Unit-Scaling Bug (Q4 2023)")
    ax.set_ylabel("Weekly Meta Spend (₹ Lakhs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=8, frameon=True, loc="upper left")

    ax = axes[1]
    cpm_fixed = df["meta_spend"] / df["meta_impressions"] * 1000.0
    q4_mask = (df["week_start"] >= "2023-10-02") & (df["week_start"] <= "2023-12-25")
    ax.scatter(df.loc[~q4_mask, "week_start"], cpm_fixed[~q4_mask], color=C_SLATE, s=16, alpha=0.65, label="Other 143 weeks (median ₹138)")
    ax.scatter(df.loc[q4_mask, "week_start"], cpm_fixed[q4_mask], color=C_PRIMARY, s=34, zorder=4, label="Fixed Q4 2023 (median ₹137)")
    ax.axhline(cpm_fixed[~q4_mask].median(), color=C_AMBER, lw=1.2, ls=":")
    ax.set_ylim(95, 180)
    ax.set_title("B. Verification via Meta CPM (Spend / 1K Impr.)")
    ax.set_ylabel("Effective Meta CPM (₹)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=8, frameon=True, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_data_audit.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIG 2: Actual vs Predicted (with OOS H2 2025) + v0 vs v1 ROI
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), dpi=220, gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    ax.plot(df["week_start"], df["revenue_inr"] / 1e7, color="#1E293B", lw=1.5, label="Actual Weekly Revenue")
    ax.plot(df["week_start"][:130], fit_oos["yhat"][:130] / 1e7, color=C_PRIMARY, lw=1.4, alpha=0.9, label=f"MMM Fit (Train R²={fit_oos['r2_train']:.3f})")
    ax.plot(df["week_start"][130:], fit_oos["yhat"][130:] / 1e7, color=C_AMBER, lw=1.8, ls="--", label=f"Out-of-Sample H2 '25 (R²={oos_metrics['r2_test']:.3f}, MAPE={oos_metrics['mape_test']:.1f}%)")
    ax.axvspan(df["week_start"].iloc[130], df["week_start"].iloc[-1], color="#FEF3C7", alpha=0.45)
    ax.set_title("A. Weekly Revenue: Actual vs. Calibrated MMM")
    ax.set_ylabel("Weekly Revenue (₹ Cr)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=7.8, frameon=True, loc="upper left")

    ax = axes[1]
    labels = [CHANNEL_LABELS[c] for c in CHANNELS]
    y_pos = np.arange(len(CHANNELS))
    v0_vals = [baseline["coefs"][c] for c in CHANNELS]
    v1_vals = [fit_cal["rois_2025"][c] for c in CHANNELS]
    v1_lo = [max(0, v1_vals[i] - unc["roi_ci80"][c][0]) for i, c in enumerate(CHANNELS)]
    v1_hi = [max(0, unc["roi_ci80"][c][2] - v1_vals[i]) for i, c in enumerate(CHANNELS)]

    # Cap visual bar for Google Search v0 (13.72) at 5.2 with an annotation so other channels stay legible!
    v0_plot = [min(v, 5.2) for v in v0_vals]
    h = 0.36
    ax.barh(y_pos + h / 2, v0_plot, height=h, color="#94A3B8", alpha=0.85, label="Baseline v0 (Uncleaned OLS)")
    ax.barh(y_pos - h / 2, v1_vals, height=h, color=C_PRIMARY, alpha=0.92, xerr=[v1_lo, v1_hi],
            error_kw={"ecolor": "#0F172A", "elinewidth": 1.1, "capsize": 2.5}, label="Calibrated v1 (2025 ROI ± 80% CI)")

    # Plot Meta geo-holdout benchmark
    ax.errorbar(1.39, 0 - h / 2, xerr=[[1.39 - 0.94], [1.84 - 1.39]], fmt="D", color=C_AMBER,
                markersize=5.5, capsize=3, lw=1.4, zorder=6, label="Meta Lift Test (1.39 [0.94–1.84])")

    ax.axvline(0, color="#334155", lw=0.8)
    ax.axvline(1.0, color=C_CORAL, lw=0.9, ls=":", alpha=0.8)
    ax.text(5.15, 1 + h / 2, "13.72 →", va="center", ha="right", fontsize=8, fontweight="bold", color="#FFFFFF")
    for i, v in enumerate(v1_vals):
        ax.text(v + v1_hi[i] + 0.08, i - h / 2, f"₹{v:.2f}", va="center", ha="left", fontsize=7.8, fontweight="bold", color=C_PRIMARY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-1.2, 5.6)
    ax.set_xlabel("Incremental Revenue per ₹1 Spent (ROAS)")
    ax.set_title("B. Baseline v0 vs. Calibrated MMM ROI")
    ax.grid(True, axis="x", alpha=0.6)
    ax.legend(fontsize=7.3, frameon=True, loc="lower right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_model_fit_and_roi.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIG 3: Hill Saturation Response Curves (2x3 Grid)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.2), dpi=220)
    axes = axes.ravel()
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        sc = fit_cal["scales"][ch]
        al = fit_cal["alphas"][ch]
        K_rel = fit_cal["Ks"][ch]
        bt = fit_cal["betas_cr"][ch]
        # Weekly spend grid from 0 to 2.2x active mean
        q25_wk = opt["status_quo_cr"][ch] * 1e7 / 13.0
        opt_wk = opt["optimal_cr"][ch] * 1e7 / 13.0
        # For TV & Influencer (bursty), show active-flight weekly spend equivalent
        max_x = max(sc * 2.1, opt_wk * 1.5, q25_wk * 1.5)
        x_grid = np.linspace(0, max_x, 120)
        K_inr = K_rel * sc
        y_grid_lakhs = (x_grid / (x_grid + K_inr)) * bt * 100.0  # Cr -> Lakhs

        ax.plot(x_grid / 1e5, y_grid_lakhs, color=C_PRIMARY, lw=2.0)
        # Mark current avg quarterly weekly spend and optimal weekly spend
        y_sq = (q25_wk / (q25_wk + K_inr)) * bt * 100.0
        y_op = (opt_wk / (opt_wk + K_inr)) * bt * 100.0
        ax.scatter([q25_wk / 1e5], [y_sq], color=C_SLATE, s=42, zorder=5, label="2025 Avg")
        ax.scatter([opt_wk / 1e5], [y_op], color=C_AMBER, s=55, marker="*", zorder=6, label="Q1 '26 Opt")
        ax.set_title(f"{CHANNEL_LABELS[ch]}  (α={al:.2f}, mROI=₹{fit_cal['mrois_2025'][ch]:.2f})", fontsize=9.5)
        ax.set_xlabel("Steady-State Weekly Spend (₹ L)", fontsize=8.2)
        ax.set_ylabel("Weekly Contrib. (₹ L)", fontsize=8.2)
        ax.grid(True, alpha=0.55)
        if i == 0:
            ax.legend(fontsize=7.5, loc="lower right")

    plt.suptitle("Channel Diminishing-Returns (Concave Hill) Curves & Q1 2026 Reallocation Shift", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_response_curves.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIG 4: 2025 Revenue Decomposition (Base vs Paid Media)
    # ------------------------------------------------------------------
    mask25 = (df["year"] == 2025).values
    rev25 = df.loc[mask25, "revenue_inr"].sum() / 1e7
    base25 = fit_cal["base_contrib"][mask25].sum() / 1e7
    ch25 = {CHANNEL_LABELS[c]: fit_cal["contrib_2025_cr"][c] for c in CHANNELS}
    sorted_ch = sorted(ch25.items(), key=lambda kv: kv[1], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), dpi=220, gridspec_kw={"width_ratios": [1.0, 1.35]})
    # Left: Base vs Paid pie/donut
    ax = axes[0]
    paid_tot = sum(ch25.values())
    sizes = [base25, paid_tot]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"Organic Base, Trend,\nSeasonality & Promos\n(₹{base25:.1f} Cr)", f"Paid Media\n(₹{paid_tot:.1f} Cr)"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#CBD5E1", C_PRIMARY],
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 8.5},
    )
    autotexts[1].set_color("white")
    autotexts[1].set_fontweight("bold")
    autotexts[0].set_fontweight("bold")
    ax.set_title(f"A. 2025 Total Revenue Split (₹{rev25:.1f} Cr)")

    # Right: Paid media breakdown
    ax = axes[1]
    ch_names = [k for k, _ in sorted_ch]
    ch_vals = [v for _, v in sorted_ch]
    ch_spends = [fit_cal["spend_2025_cr"][[c for c in CHANNELS if CHANNEL_LABELS[c] == k][0]] for k in ch_names]
    x = np.arange(len(ch_names))
    w = 0.36
    ax.bar(x - w / 2, ch_spends, width=w, color="#94A3B8", label="2025 Spend (₹ Cr)")
    ax.bar(x + w / 2, ch_vals, width=w, color=C_PRIMARY, label="Attributed Revenue (₹ Cr)")
    for i, (sp, rv) in enumerate(zip(ch_spends, ch_vals)):
        ax.text(i + w / 2, rv + 0.6, f"{rv/sp:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold", color=C_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, fontsize=8.5)
    ax.set_ylabel("₹ Crores (2025 Full Year)")
    ax.set_ylim(0, max(ch_vals) * 1.18)
    ax.set_title("B. 2025 Paid Channel Spend vs. Attributed Revenue (ROAS)")
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_revenue_decomposition.png"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # FIG 5: Q1 2026 Budget Reallocation & Bootstrap Gain Distribution
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=220, gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    labels = [CHANNEL_LABELS[c] for c in CHANNELS]
    x = np.arange(len(CHANNELS))
    sq = [opt["status_quo_cr"][c] for c in CHANNELS]
    op = [opt["optimal_cr"][c] for c in CHANNELS]
    lo_b = [opt["bounds_cr"][c][0] for c in CHANNELS]
    hi_b = [opt["bounds_cr"][c][1] for c in CHANNELS]

    w = 0.36
    ax.bar(x - w / 2, sq, width=w, color="#94A3B8", label="2025 Avg Quarterly Split (₹16.0 Cr)")
    ax.bar(x + w / 2, op, width=w, color=C_PRIMARY, label="Recommended Q1 2026 Split (₹16.0 Cr)")
    # Plot 50-150% guardrail range as thin vertical caps
    for i in range(len(CHANNELS)):
        ax.plot([i + w / 2 - 0.14, i + w / 2 + 0.14], [lo_b[i], lo_b[i]], color=C_CORAL, lw=1.2, ls="--")
        ax.plot([i + w / 2 - 0.14, i + w / 2 + 0.14], [hi_b[i], hi_b[i]], color=C_CORAL, lw=1.2, ls="--")
        diff = op[i] - sq[i]
        sign = "+" if diff >= 0 else ""
        col = C_ACCENT if diff >= 0 else C_RED
        ax.text(i + w / 2, op[i] + 0.12, f"{sign}{diff:.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=col)

    ax.plot([], [], color=C_CORAL, lw=1.2, ls="--", label="50%–150% Guardrails")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Quarterly Budget (₹ Crores)")
    ax.set_ylim(0, max(max(hi_b), max(op)) * 1.12)
    ax.set_title("A. Q1 2026 Budget Reallocation (₹16.0 Cr Total)")
    ax.grid(True, axis="y", alpha=0.6)
    ax.legend(fontsize=7.6, loc="upper right")

    ax = axes[1]
    gains = np.array(unc["gain_draws"])
    ax.hist(gains, bins=12, color=C_PRIMARY, alpha=0.8, edgecolor="white")
    ax.axvline(opt["incremental_gain_cr"], color=C_AMBER, lw=2.2, label=f"Point Est: +₹{opt['incremental_gain_cr']:.2f} Cr")
    ax.axvline(unc["gain_ci80"][0], color=C_SLATE, lw=1.2, ls="--", label=f"80% CI: [₹{unc['gain_ci80'][0]:.2f}, ₹{unc['gain_ci80'][2]:.2f}] Cr")
    ax.axvline(unc["gain_ci80"][2], color=C_SLATE, lw=1.2, ls="--")
    ax.axvline(0, color=C_RED, lw=1.0, ls=":")
    ax.set_title(f"B. Bootstrap Expected Lift ({unc['prob_positive_gain']:.0f}% Prob. > ₹0)")
    ax.set_xlabel("Incremental Q1 Revenue Gain vs. Status Quo (₹ Cr)")
    ax.set_ylabel("Bootstrap Replicates")
    ax.grid(True, alpha=0.55)
    ax.legend(fontsize=7.8, loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_budget_optimization.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("1. Loading and cleaning data...")
    raw_df = pd.read_csv(DATA_CSV)
    df, audit = load_and_clean_data(DATA_CSV)

    print("2. Replicating baseline v0 OLS & bad-control demo...")
    baseline = run_baseline_ols(DATA_CSV)
    bad_ctrl = run_bad_control_demo(df)

    print("3. Fitting uncalibrated & Meta-lift-calibrated MMM...")
    fit_uncal = fit_mmm(df, use_bs_resid=True, calibrate_meta=False)
    fit_cal = fit_mmm(df, use_bs_resid=True, calibrate_meta=True)

    print("4. Running Out-of-Sample validation (Train 130 wks -> Test H2 2025 26 wks)...")
    tr_idx = np.arange(130)
    te_idx = np.arange(130, 156)
    fit_oos = fit_mmm(df, use_bs_resid=True, calibrate_meta=True, train_idx=tr_idx)
    y = df["revenue_inr"].values
    r2_test = float(1.0 - np.sum((y[te_idx] - fit_oos["yhat"][te_idx]) ** 2) / np.sum((y[te_idx] - y[te_idx].mean()) ** 2))
    mape_test = float(np.mean(np.abs((y[te_idx] - fit_oos["yhat"][te_idx]) / y[te_idx])) * 100.0)
    oos_metrics = {
        "r2_train": fit_oos["r2_train"],
        "mape_train": fit_oos["mape_train"],
        "r2_test": r2_test,
        "mape_test": mape_test,
        "meta_pred_iroas": fit_oos["meta_pred_iroas"],
    }

    print("5. Optimizing Q1 2026 budget (Rs 16 Cr, 50-150% guardrails)...")
    opt = optimize_q1_budget(fit_cal, df, total_budget_cr=16.0, min_mult=0.50, max_mult=1.50)

    print("6. Running 35 block-bootstrap replicates for uncertainty intervals...")
    boots = bootstrap_mmm(df, fit_cal, n_boot=35, seed=42)
    unc = evaluate_bootstrap_uncertainty(boots, opt, df)

    print("7. Generating high-DPI figures...")
    make_figures(raw_df, df, baseline, fit_cal, fit_oos, oos_metrics, opt, unc)

    # Save structured summary JSON for report generation & verification
    def strip_arrays(d):
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    summary = {
        "audit": audit,
        "baseline": strip_arrays(baseline),
        "bad_control_demo": bad_ctrl,
        "fit_uncal": strip_arrays(fit_uncal),
        "fit_cal": strip_arrays(fit_cal),
        "oos_metrics": oos_metrics,
        "optimization": opt,
        "uncertainty": unc,
    }
    summary_path = os.path.join(OUT_DIR, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY OF KEY RESULTS ===")
    print(f"Calibrated MMM: R2={fit_cal['r2_train']:.4f}, MAPE={fit_cal['mape_train']:.2f}%, DW={fit_cal['dw_train']:.2f}, Meta Holdout iROAS={fit_cal['meta_pred_iroas']:.2f}")
    print(f"OOS H2 2025:    R2={r2_test:.4f}, MAPE={mape_test:.2f}%")
    print(f"Q1 2026 Opt:    Status Quo Media Rev = Rs {opt['sq_media_rev_cr']:.2f} Cr -> Optimal = Rs {opt['opt_media_rev_cr']:.2f} Cr (+Rs {opt['incremental_gain_cr']:.2f} Cr / +{opt['incremental_gain_pct']:.1f}%)")
    print(f"Bootstrap 80% CI on Gain: [Rs {unc['gain_ci80'][0]:.2f}, Rs {unc['gain_ci80'][2]:.2f}] Cr (Prob > 0: {unc['prob_positive_gain']:.1f}%)")
    print("\nChannel Table (2025 ROI, 80% CI, 2025 Avg Qtr -> Q1 2026 Opt):")
    for ch in CHANNELS:
        lo, med, hi = unc["roi_ci80"][ch]
        print(
            f"  {CHANNEL_LABELS[ch]:<14} ROI={fit_cal['rois_2025'][ch]:.2f} [{lo:.2f}-{hi:.2f}]  "
            f"mROI={fit_cal['mrois_2025'][ch]:.2f}  "
            f"Qtr: {opt['status_quo_cr'][ch]:.2f} -> {opt['optimal_cr'][ch]:.2f} Cr"
        )
    print(f"\nSaved summary to {summary_path} and figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
