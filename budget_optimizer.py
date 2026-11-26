"""
budget_optimizer.py — Constrained nonlinear Q1 2026 (Jan–Mar 2026, Rs 16 Cr)
budget allocation optimizer with 50–150% channel guardrails and bootstrap uncertainty.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_cleaning import CHANNELS, CHANNEL_LABELS
from mmm_model import transform_channel


def compute_2025_avg_quarterly_spend(df: pd.DataFrame) -> dict:
    """2025 average quarterly spend per channel (in Cr INR)."""
    df25 = df[df["year"] == 2025]
    return {ch: float(df25[ch].sum() / 4.0 / 1e7) for ch in CHANNELS}


def simulate_quarter_media_revenue(
    q_spend_cr: np.ndarray,
    fit: dict,
    df: pd.DataFrame,
    n_weeks: int = 13,
    carryover_tail_weeks: int = 4,
) -> tuple:
    """
    Simulate Q1 2026 (13 weeks) incremental media revenue for a given quarterly
    channel budget vector (in Cr INR).
    Uses realistic weekly flighting profiles:
      - Always-on channels (Meta, Google Search, YouTube, Amazon Ads): steady weekly pacing.
      - Bursty channels (TV, Influencer): 4-week burst flighting (matching historical TV/Influencer
        deployment patterns, e.g., late-Q1 March launch burst for TV; two 3-week bursts for Influencer)
        so Hill saturation reflects realistic flighting intensity rather than artificially smoothed dust.
    Includes initial carryover from late Dec 2025 and 4 post-quarter tail weeks for Q1 spend carryover
    (or steady-state 13-week equivalent). We evaluate steady-state 13-week revenue attributable to Q1 spend
    by simulating Q1 spend from zero pre-carryover through 13 + 4 tail weeks so full adstock return is credited.
    """
    total_weeks = n_weeks + carryover_tail_weeks
    ch_rev_cr = np.zeros(len(CHANNELS))

    # Realistic weekly pacing weights summing to 1.0 over 13 weeks
    pacing = {
        "meta_spend": np.ones(n_weeks) / n_weeks,
        "google_search_spend": np.ones(n_weeks) / n_weeks,
        "youtube_spend": np.ones(n_weeks) / n_weeks,
        "amazon_ads_spend": np.ones(n_weeks) / n_weeks,
        # Influencer: 6 active burst weeks (2 bursts of 3 wks) + 7 maintenance weeks (~10% of budget)
        "influencer_spend": np.array([0.015, 0.015, 0.145, 0.145, 0.145, 0.015, 0.015, 0.015, 0.145, 0.145, 0.145, 0.030, 0.035]),
        # TV: concentrated 4-week late-Q1 flight (Mar), matching historical Q1/Q2 TV flighting
        "tv_spend": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]),
    }

    for i, ch in enumerate(CHANNELS):
        w = pacing[ch]
        w = w / w.sum()
        sp_vec = np.zeros(total_weeks)
        sp_vec[:n_weeks] = q_spend_cr[i] * 1e7 * w
        # Adstock + Hill
        contrib_cr = transform_channel(
            sp_vec, fit["alphas"][ch], fit["Ks"][ch], fit["scales"][ch]
        ) * fit["betas_cr"][ch]
        ch_rev_cr[i] = float(contrib_cr.sum())

    return float(ch_rev_cr.sum()), ch_rev_cr


def optimize_q1_budget(
    fit: dict,
    df: pd.DataFrame,
    total_budget_cr: float = 16.0,
    min_mult: float = 0.50,
    max_mult: float = 1.50,
) -> dict:
    """
    Maximize total Q1 2026 incremental media revenue subject to:
      sum(budget_c) == 16.0 Cr
      0.50 * avg_q2025_c <= budget_c <= 1.50 * avg_q2025_c
    """
    q25 = compute_2025_avg_quarterly_spend(df)
    base_vec = np.array([q25[ch] for ch in CHANNELS])
    # Scale baseline to exactly 16.0 Cr for apples-to-apples comparison
    status_quo_vec = base_vec * (total_budget_cr / base_vec.sum())

    bounds = [(min_mult * q25[ch], max_mult * q25[ch]) for ch in CHANNELS]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget_cr}]

    def neg_rev(x):
        tot, _ = simulate_quarter_media_revenue(x, fit, df)
        return -tot

    res = minimize(
        neg_rev,
        status_quo_vec,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    opt_vec = res.x

    sq_tot, sq_ch = simulate_quarter_media_revenue(status_quo_vec, fit, df)
    opt_tot, opt_ch = simulate_quarter_media_revenue(opt_vec, fit, df)

    # Marginal ROI at optimal allocation (+1% budget perturbation per channel)
    opt_mroi = {}
    sq_mroi = {}
    for i, ch in enumerate(CHANNELS):
        eps = 0.05  # Rs 5 Lakh = 0.05 Cr
        v_sq = status_quo_vec.copy(); v_sq[i] += eps
        sq_mroi[ch] = (simulate_quarter_media_revenue(v_sq, fit, df)[0] - sq_tot) / eps
        v_op = opt_vec.copy(); v_op[i] += eps
        opt_mroi[ch] = (simulate_quarter_media_revenue(v_op, fit, df)[0] - opt_tot) / eps

    return {
        "success": bool(res.success),
        "total_budget_cr": total_budget_cr,
        "q2025_avg_cr": q25,
        "status_quo_cr": {ch: float(v) for ch, v in zip(CHANNELS, status_quo_vec)},
        "optimal_cr": {ch: float(v) for ch, v in zip(CHANNELS, opt_vec)},
        "bounds_cr": {ch: (float(b[0]), float(b[1])) for ch, b in zip(CHANNELS, bounds)},
        "sq_media_rev_cr": float(sq_tot),
        "opt_media_rev_cr": float(opt_tot),
        "incremental_gain_cr": float(opt_tot - sq_tot),
        "incremental_gain_pct": float((opt_tot - sq_tot) / sq_tot * 100.0),
        "sq_ch_rev_cr": {ch: float(v) for ch, v in zip(CHANNELS, sq_ch)},
        "opt_ch_rev_cr": {ch: float(v) for ch, v in zip(CHANNELS, opt_ch)},
        "sq_mroi": sq_mroi,
        "opt_mroi": opt_mroi,
    }


def evaluate_bootstrap_uncertainty(boots: list, point_opt: dict, df: pd.DataFrame) -> dict:
    """
    Evaluate:
      1) Bootstrap 80% CI on each channel's 2025 ROI and mROI.
      2) Distribution of incremental revenue gain if we deploy the recommended split
         across all bootstrap parameter draws (honest uncertainty on expected payoff!).
      3) Distribution of each bootstrap draw's own optimal split (stability of recommendation!).
    """
    opt_vec = np.array([point_opt["optimal_cr"][ch] for ch in CHANNELS])
    sq_vec = np.array([point_opt["status_quo_cr"][ch] for ch in CHANNELS])

    roi_draws = {ch: [] for ch in CHANNELS}
    mroi_draws = {ch: [] for ch in CHANNELS}
    meta_iroas_draws = []
    gain_draws = []
    opt_rev_draws = []
    sq_rev_draws = []
    boot_opt_splits = {ch: [] for ch in CHANNELS}

    for fb in boots:
        meta_iroas_draws.append(fb["meta_pred_iroas"])
        for ch in CHANNELS:
            roi_draws[ch].append(fb["rois_2025"][ch])
            mroi_draws[ch].append(fb["mrois_2025"][ch])

        # Evaluate our fixed recommended split vs status quo under bootstrap draw fb
        rev_sq, _ = simulate_quarter_media_revenue(sq_vec, fb, df)
        rev_op, _ = simulate_quarter_media_revenue(opt_vec, fb, df)
        sq_rev_draws.append(rev_sq)
        opt_rev_draws.append(rev_op)
        gain_draws.append(rev_op - rev_sq)

        # Also re-optimize under fb to measure allocation stability
        bo = optimize_q1_budget(fb, df)
        for ch in CHANNELS:
            boot_opt_splits[ch].append(bo["optimal_cr"][ch])

    def ci80(arr):
        a = np.asarray(arr)
        return (float(np.percentile(a, 10)), float(np.percentile(a, 50)), float(np.percentile(a, 90)))

    return {
        "n_boot": len(boots),
        "roi_ci80": {ch: ci80(roi_draws[ch]) for ch in CHANNELS},
        "mroi_ci80": {ch: ci80(mroi_draws[ch]) for ch in CHANNELS},
        "meta_iroas_ci80": ci80(meta_iroas_draws),
        "sq_rev_ci80": ci80(sq_rev_draws),
        "opt_rev_ci80": ci80(opt_rev_draws),
        "gain_ci80": ci80(gain_draws),
        "prob_positive_gain": float(np.mean(np.array(gain_draws) > 0) * 100.0),
        "boot_opt_split_ci80": {ch: ci80(boot_opt_splits[ch]) for ch in CHANNELS},
        "gain_draws": [float(x) for x in gain_draws],
    }
