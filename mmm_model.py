"""
mmm_model.py — Bayesian MAP Concave Adstock-Hill Marketing Mix Model
with Stage-1 brand-search residualization, Meta geo-holdout calibration,
out-of-sample validation, and block-bootstrap uncertainty quantification.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_cleaning import CHANNELS, CHANNEL_LABELS


CTRL_COLS = [
    "avg_discount_pct",
    "price_index",
    "festive_flag",
    "marketplace_sale_flag",
    "competitor_promo_flag",
    "new_launch_flag",
]


def adstock_geometric(x: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric carryover: A_t = x_t + alpha * A_{t-1}."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + alpha * out[i - 1]
    return out


def transform_channel(spend: np.ndarray, alpha: float, K_rel: float, scale: float) -> np.ndarray:
    """
    Normalized geometric adstock + strictly concave Michaelis-Menten Hill saturation (S=1):
      A_norm = (1 - alpha) * adstock(spend, alpha)
      f(A_norm) = A_norm / (A_norm + K_rel * scale)
    Strictly concave everywhere so marginal ROI < average ROI at all positive spend levels.
    """
    ad_norm = adstock_geometric(spend, alpha) * (1.0 - alpha)
    K = K_rel * scale
    return ad_norm / (ad_norm + K + 1e-12)


def build_base_controls(df: pd.DataFrame) -> np.ndarray:
    """Linear organic trend + 2 annual Fourier harmonics + exogenous business controls."""
    t = df["t"].values
    t_norm = t / 156.0
    sin1 = np.sin(2.0 * np.pi * t / 52.0)
    cos1 = np.cos(2.0 * np.pi * t / 52.0)
    sin2 = np.sin(4.0 * np.pi * t / 52.0)
    cos2 = np.cos(4.0 * np.pi * t / 52.0)
    return np.column_stack([
        np.ones(len(df)),
        t_norm,
        sin1,
        cos1,
        sin2,
        cos2,
        df[CTRL_COLS].values,
    ])


def compute_brand_search_residual(df: pd.DataFrame, X_ctrl_base: np.ndarray) -> tuple:
    """
    Stage-1 orthogonalization of brand_search_index against exogenous controls and
    adstocked upper-funnel media (Meta, YouTube, Influencer, TV).
    The residual captures unmeasured organic demand shocks that drive Google Search
    endogenously without absorbing upper-funnel media credit.
    """
    ad_meta = adstock_geometric(df["meta_spend"].values, 0.35) / 1e6
    ad_yt = adstock_geometric(df["youtube_spend"].values, 0.60) / 1e6
    ad_inf = adstock_geometric(df["influencer_spend"].values, 0.45) / 1e6
    ad_tv = adstock_geometric(df["tv_spend"].values, 0.65) / 1e6
    X_bs = np.column_stack([X_ctrl_base, ad_meta, ad_yt, ad_inf, ad_tv])
    y_bs = df["brand_search_index"].values
    coef, *_ = np.linalg.lstsq(X_bs, y_bs, rcond=None)
    resid = y_bs - X_bs @ coef
    r2 = float(1.0 - np.sum(resid ** 2) / np.sum((y_bs - y_bs.mean()) ** 2))
    return resid, r2, coef[-4:]


def run_baseline_ols(raw_csv_path: str) -> dict:
    """Replicate baseline_mmm.py (v0) on raw uncleaned data."""
    raw = pd.read_csv(raw_csv_path).dropna().copy()
    cols = CHANNELS + ["festive_flag", "avg_discount_pct"]
    X = np.column_stack([np.ones(len(raw)), raw[cols].values])
    y = raw["revenue_inr"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    r2 = float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))
    mape = float(np.mean(np.abs(resid / y)) * 100.0)
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
    n, k = X.shape
    s2 = np.sum(resid ** 2) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    names = ["const"] + cols
    return {
        "n": int(n),
        "r2": r2,
        "mape": mape,
        "dw": dw,
        "coefs": {nm: float(b) for nm, b in zip(names, beta)},
        "se": {nm: float(s) for nm, s in zip(names, se)},
        "yhat": yhat,
    }


def run_bad_control_demo(df: pd.DataFrame) -> dict:
    """Demonstrate mediator/bad-control bias when organic_traffic_index is included."""
    X_ctrl = build_base_controls(df)
    X_no_org = np.column_stack([X_ctrl, df[CHANNELS].values])
    X_with_org = np.column_stack([X_ctrl, df["organic_traffic_index"].values, df[CHANNELS].values])
    y = df["revenue_inr"].values
    b_no, *_ = np.linalg.lstsq(X_no_org, y, rcond=None)
    b_with, *_ = np.linalg.lstsq(X_with_org, y, rcond=None)
    return {
        "without_organic": {ch: float(b_no[X_ctrl.shape[1] + i]) for i, ch in enumerate(CHANNELS)},
        "with_organic": {ch: float(b_with[X_ctrl.shape[1] + 1 + i]) for i, ch in enumerate(CHANNELS)},
        "organic_coef": float(b_with[X_ctrl.shape[1]]),
    }


def fit_mmm(
    df: pd.DataFrame,
    lift_test: dict = None,
    use_bs_resid: bool = True,
    calibrate_meta: bool = True,
    cal_weight: float = 1.0,
    train_idx: np.ndarray = None,
    y_override: np.ndarray = None,
    x0_override: np.ndarray = None,
    maxiter: int = 1200,
    ftol: float = 1e-9,
) -> dict:
    """
    Fit Bayesian MAP Concave Adstock-Hill MMM with weakly-informative Meridian-style
    LogNormal ROI priors and optional Meta geo-holdout lift-test calibration.
    """
    n = len(df)
    if train_idx is None:
        train_idx = np.arange(n)

    X_ctrl_base = build_base_controls(df)
    bs_resid, bs_r2, bs_upper_coefs = compute_brand_search_residual(df, X_ctrl_base)

    ctrl_names = ["const", "t_norm", "sin1", "cos1", "sin2", "cos2"] + CTRL_COLS
    if use_bs_resid:
        Xc_full = np.column_stack([X_ctrl_base, bs_resid])
        ctrl_names = ctrl_names + ["brand_search_resid"]
    else:
        Xc_full = X_ctrl_base

    Xc_mean = Xc_full[train_idx, 1:].mean(axis=0)
    Xc_std = Xc_full[train_idx, 1:].std(axis=0) + 1e-8
    Xc_s = np.column_stack([np.ones(n), (Xc_full[:, 1:] - Xc_mean) / Xc_std])

    scales = np.array([df.loc[df[ch] > 0, ch].mean() for ch in CHANNELS])
    spends = [df[ch].values.astype(float) for ch in CHANNELS]

    y = df["revenue_inr"].values.astype(float) if y_override is None else y_override.astype(float)
    y_scale = 1e7  # 1 Cr INR
    y_cr = y / y_scale

    if lift_test is None:
        lift_test = {
            "test_start": "2025-04-07",
            "test_end": "2025-05-12",
            "post_end": "2025-06-09",
            "holdout_share": 0.20,
            "measured_iroas": 1.39,
            "ci90_low": 0.94,
            "ci90_high": 1.84,
        }
    test_mask = ((df["week_start"] >= lift_test["test_start"]) & (df["week_start"] <= lift_test["test_end"])).values
    eval_mask = ((df["week_start"] >= lift_test["test_start"]) & (df["week_start"] <= lift_test["post_end"])).values
    spend_withheld = lift_test["holdout_share"] * spends[0][test_mask].sum()
    meta_cf_spend = spends[0].copy()
    meta_cf_spend[test_mask] *= (1.0 - lift_test["holdout_share"])
    se_test = (lift_test["ci90_high"] - lift_test["ci90_low"]) / (2.0 * 1.64485)

    # Order: Meta, Google Search, YouTube, Influencer, TV, Amazon Ads
    alpha_bounds = [(0.10, 0.70), (0.00, 0.25), (0.20, 0.85), (0.15, 0.75), (0.35, 0.85), (0.00, 0.25)]
    alpha_prior = np.array([0.35, 0.05, 0.60, 0.40, 0.65, 0.10])
    K_bounds = [(0.50, 3.00)] * 6
    beta_bounds = [(0.05, 6.00)] * 6
    ctrl_bounds = [(-20.0, 20.0)] * Xc_s.shape[1]

    # Smart initialization via conditional OLS on controls (or warm start)
    if x0_override is not None:
        x0 = list(x0_override)
    else:
        al0 = list(alpha_prior)
        K0 = [1.2] * 6
        bt0 = [0.9, 0.4, 0.3, 0.4, 0.6, 0.7]
        M0 = np.column_stack([transform_channel(spends[i], al0[i], K0[i], scales[i]) for i in range(6)])
        c0, *_ = np.linalg.lstsq(Xc_s[train_idx], y_cr[train_idx] - M0[train_idx] @ np.array(bt0), rcond=None)
        x0 = al0 + K0 + bt0 + list(c0)
    bounds = alpha_bounds + K_bounds + beta_bounds + ctrl_bounds

    sp_tot_cr = np.array([spends[i][train_idx].sum() / y_scale for i in range(6)])

    def objective(params):
        al = params[0:6]
        Ks = params[6:12]
        bt = params[12:18]
        cc = params[18:]

        M = np.column_stack([transform_channel(spends[i], al[i], Ks[i], scales[i]) for i in range(6)])
        yhat_cr = M @ bt + Xc_s @ cc
        resid = y_cr[train_idx] - yhat_cr[train_idx]
        nll = 0.5 * np.sum(resid ** 2) / (0.15 ** 2)

        # Weakly informative Meridian-style LogNormal ROI prior: log(ROI) ~ N(log(1.5), 0.70^2)
        rois = (M[train_idx].sum(axis=0) * bt) / sp_tot_cr
        roi_prior = 0.5 * np.sum(((np.log(np.maximum(rois, 1e-4)) - np.log(1.5)) / 0.70) ** 2)

        hyper_prior = (
            0.5 * np.sum(((al - alpha_prior) / 0.20) ** 2)
            + 0.5 * np.sum((np.log(Ks / 1.2) / 0.45) ** 2)
            + 0.5 * np.sum((cc[1:] / 5.0) ** 2)
        )

        cal = 0.0
        if calibrate_meta:
            mb = transform_channel(spends[0], al[0], Ks[0], scales[0]) * bt[0] * y_scale
            mc = transform_channel(meta_cf_spend, al[0], Ks[0], scales[0]) * bt[0] * y_scale
            pred_iroas = np.sum(mb[eval_mask] - mc[eval_mask]) / spend_withheld
            cal = cal_weight * 0.5 * ((pred_iroas - lift_test["measured_iroas"]) / se_test) ** 2

        return nll + roi_prior + hyper_prior + cal

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": ftol})
    al, Ks, bt, cc = res.x[0:6], res.x[6:12], res.x[12:18], res.x[18:]

    M = np.column_stack([transform_channel(spends[i], al[i], Ks[i], scales[i]) for i in range(6)])
    media_contribs = M * bt * y_scale
    base_contrib = (Xc_s @ cc) * y_scale
    yhat = media_contribs.sum(axis=1) + base_contrib
    resid = y - yhat

    r2_tr = float(1.0 - np.sum(resid[train_idx] ** 2) / np.sum((y[train_idx] - y[train_idx].mean()) ** 2))
    mape_tr = float(np.mean(np.abs(resid[train_idx] / y[train_idx])) * 100.0)
    rmse_tr = float(np.sqrt(np.mean(resid[train_idx] ** 2)))
    dw_tr = float(np.sum(np.diff(resid[train_idx]) ** 2) / np.sum(resid[train_idx] ** 2))

    mb = transform_channel(spends[0], al[0], Ks[0], scales[0]) * bt[0] * y_scale
    mc = transform_channel(meta_cf_spend, al[0], Ks[0], scales[0]) * bt[0] * y_scale
    meta_pred_iroas = float(np.sum(mb[eval_mask] - mc[eval_mask]) / spend_withheld)

    mask25 = (df["year"] == 2025).values
    rois_2025, mrois_2025, contrib_2025_cr, spend_2025_cr = {}, {}, {}, {}
    for i, ch in enumerate(CHANNELS):
        c25 = float(media_contribs[mask25, i].sum())
        s25 = float(spends[i][mask25].sum())
        rois_2025[ch] = c25 / s25
        contrib_2025_cr[ch] = c25 / y_scale
        spend_2025_cr[ch] = s25 / y_scale
        sp_p = spends[i].copy()
        sp_p[mask25] *= 1.05
        cp = float((transform_channel(sp_p, al[i], Ks[i], scales[i])[mask25] * bt[i] * y_scale).sum())
        mrois_2025[ch] = (cp - c25) / (0.05 * s25)

    c_raw = np.empty_like(cc)
    c_raw[1:] = (cc[1:] / Xc_std) * y_scale
    c_raw[0] = (cc[0] - np.sum(cc[1:] * Xc_mean / Xc_std)) * y_scale

    return {
        "success": bool(res.success),
        "r2_train": r2_tr,
        "mape_train": mape_tr,
        "rmse_train": rmse_tr,
        "dw_train": dw_tr,
        "meta_pred_iroas": meta_pred_iroas,
        "alphas": {ch: float(a) for ch, a in zip(CHANNELS, al)},
        "Ks": {ch: float(k) for ch, k in zip(CHANNELS, Ks)},
        "betas_cr": {ch: float(b) for ch, b in zip(CHANNELS, bt)},
        "scales": {ch: float(sc) for ch, sc in zip(CHANNELS, scales)},
        "ctrl_coefs_inr": {nm: float(v) for nm, v in zip(ctrl_names, c_raw)},
        "rois_2025": rois_2025,
        "mrois_2025": mrois_2025,
        "contrib_2025_cr": contrib_2025_cr,
        "spend_2025_cr": spend_2025_cr,
        "media_contribs": media_contribs,
        "base_contrib": base_contrib,
        "yhat": yhat,
        "resid": resid,
        "bs_stage1_r2": bs_r2,
        "bs_upper_coefs": [float(v) for v in bs_upper_coefs],
        "opt_params": res.x,
    }


def bootstrap_mmm(df: pd.DataFrame, base_fit: dict, n_boot: int = 30, seed: int = 42) -> list:
    """
    Circular block bootstrap (4-week blocks) + Meta lift-test uncertainty perturbation
    to produce honest uncertainty intervals on ROIs, mROIs, and optimal budgets.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    yhat = base_fit["yhat"]
    resid = base_fit["resid"]
    x0_warm = base_fit["opt_params"]
    block_len = 4
    n_blocks = int(np.ceil(n / block_len))
    se_test = (1.84 - 0.94) / (2.0 * 1.64485)

    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]
        y_boot = yhat + resid[idx]
        lt_boot = {
            "test_start": "2025-04-07",
            "test_end": "2025-05-12",
            "post_end": "2025-06-09",
            "holdout_share": 0.20,
            "measured_iroas": float(rng.normal(1.39, se_test)),
            "ci90_low": 0.94,
            "ci90_high": 1.84,
        }
        boots.append(
            fit_mmm(
                df,
                lift_test=lt_boot,
                use_bs_resid=True,
                calibrate_meta=True,
                cal_weight=1.0,
                y_override=y_boot,
                x0_override=x0_warm,
                maxiter=300,
                ftol=1e-7,
            )
        )
    return boots
