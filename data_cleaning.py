"""
data_cleaning.py — Data quality audit and cleaning pipeline for Verdant Leaf MMM.
"""
import numpy as np
import pandas as pd


CHANNELS = [
    "meta_spend",
    "google_search_spend",
    "youtube_spend",
    "influencer_spend",
    "tv_spend",
    "amazon_ads_spend",
]

CHANNEL_LABELS = {
    "meta_spend": "Meta",
    "google_search_spend": "Google Search",
    "youtube_spend": "YouTube",
    "influencer_spend": "Influencer",
    "tv_spend": "TV",
    "amazon_ads_spend": "Amazon Ads",
}


def load_and_clean_data(csv_path: str):
    """Load raw MMM weekly export, audit anomalies, and return (clean_df, audit_dict)."""
    raw = pd.read_csv(csv_path)
    audit = {
        "raw_rows": int(len(raw)),
        "raw_cols": int(raw.shape[1]),
    }

    # 1. Duplicate week check
    dup_mask = raw.duplicated(subset=["week_start"], keep=False)
    dup_dates = sorted(raw.loc[dup_mask, "week_start"].unique().tolist())
    audit["duplicate_weeks"] = dup_dates
    audit["duplicate_rows_removed"] = int(raw.duplicated(subset=["week_start"], keep="first").sum())

    df = raw.drop_duplicates(subset=["week_start"], keep="first").copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)

    # Verify exact 7-day continuity across 156 weeks
    diffs = df["week_start"].diff().dropna()
    assert (diffs == pd.Timedelta(days=7)).all(), "Non-7-day gap detected after deduplication!"
    audit["clean_weeks"] = int(len(df))
    audit["date_range"] = (str(df["week_start"].min().date()), str(df["week_start"].max().date()))

    # 2. Meta Q4 2023 unit-scaling error (Lakhs instead of INR)
    meta_cpm_raw = df["meta_spend"] / df["meta_impressions"] * 1000.0
    lakh_mask = meta_cpm_raw < 1.0  # normal CPM is ~120-165 INR; Lakhs rows are ~0.0012-0.0016
    audit["meta_lakh_rows"] = int(lakh_mask.sum())
    audit["meta_lakh_window"] = (
        str(df.loc[lakh_mask, "week_start"].min().date()),
        str(df.loc[lakh_mask, "week_start"].max().date()),
    )
    audit["meta_cpm_before_lakh_fix"] = float(meta_cpm_raw[lakh_mask].median())
    df.loc[lakh_mask, "meta_spend"] = df.loc[lakh_mask, "meta_spend"] * 1e5
    meta_cpm_fixed = df["meta_spend"] / df["meta_impressions"] * 1000.0
    audit["meta_cpm_after_lakh_fix"] = float(meta_cpm_fixed[lakh_mask].median())
    audit["meta_cpm_other_weeks_median"] = float(meta_cpm_fixed[~lakh_mask].median())

    # 3. Negative TV spend (credit note / billing adjustment)
    neg_tv_mask = df["tv_spend"] < 0
    audit["neg_tv_rows"] = int(neg_tv_mask.sum())
    audit["neg_tv_details"] = [
        {"week_start": str(r["week_start"].date()), "tv_spend": float(r["tv_spend"])}
        for _, r in df.loc[neg_tv_mask].iterrows()
    ]
    # Clamping to 0 preserves gross media exposure (the preceding Diwali flight still aired)
    df.loc[neg_tv_mask, "tv_spend"] = 0.0

    # 4. Missing YouTube spend (impressions are present!) -> impute using median YouTube CPM
    yt_na_mask = df["youtube_spend"].isna()
    audit["yt_missing_rows"] = int(yt_na_mask.sum())
    audit["yt_missing_weeks"] = [str(d.date()) for d in df.loc[yt_na_mask, "week_start"]]
    valid_yt = (~yt_na_mask) & (df["youtube_impressions"] > 0)
    yt_cpm_med = float((df.loc[valid_yt, "youtube_spend"] / df.loc[valid_yt, "youtube_impressions"] * 1000.0).median())
    audit["yt_median_cpm"] = yt_cpm_med
    df.loc[yt_na_mask, "youtube_spend"] = (
        df.loc[yt_na_mask, "youtube_impressions"] * yt_cpm_med / 1000.0
    )

    # Confirm zero remaining NaNs or negatives
    assert df.isna().sum().sum() == 0, "Remaining NaNs in cleaned dataframe!"
    for ch in CHANNELS:
        assert (df[ch] >= 0).all(), f"Negative spend remaining in {ch}!"

    # Add calendar helpers
    df["t"] = np.arange(len(df), dtype=float)
    df["year"] = df["week_start"].dt.year
    df["quarter"] = df["week_start"].dt.quarter
    df["month"] = df["week_start"].dt.month
    df["aov"] = df["revenue_inr"] / df["orders"]

    return df, audit


if __name__ == "__main__":
    df, audit = load_and_clean_data(r"c:\Users\hp\Desktop\DE_ASSIGN\pack\mmm_weekly_data.csv")
    print("Audit summary:")
    for k, v in audit.items():
        print(f"  {k}: {v}")
