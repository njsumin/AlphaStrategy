"""
Support/Resistance Level Generator (sr_levels.py)
===================================================
Collects multi-period Volume Profile levels (POC/VAH/VAL), clusters nearby
levels into meaningful S/R zones, and finds the nearest support and resistance
for each bar.

Usage:
------
    from sr_levels import add_sr_levels

    df = add_sr_levels(df, vp_windows=[7, 14, 30])
"""

import numpy as np
import pandas as pd


def _cluster_levels(levels: np.ndarray, close: float,
                    min_gap_pct: float = 0.01) -> np.ndarray:
    """
    Cluster nearby S/R levels to reduce noise.

    Sorts levels, then merges adjacent levels whose gap is less than
    min_gap_pct * close.  Merged levels use the arithmetic mean.

    Args:
        levels: Array of valid (non-NaN) S/R levels.
        close: Current bar close price (used for gap threshold).
        min_gap_pct: Minimum gap as fraction of close.  Levels closer
                     than this are merged.

    Returns:
        Array of clustered levels (typically 2-5).
    """
    if len(levels) == 0:
        return levels

    sorted_lvls = np.sort(levels)
    min_gap = min_gap_pct * close

    clusters = [[sorted_lvls[0]]]
    for lvl in sorted_lvls[1:]:
        if lvl - np.mean(clusters[-1]) < min_gap:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])

    return np.array([np.mean(c) for c in clusters])


def add_sr_levels(df: pd.DataFrame,
                  vp_windows: list = None,
                  cluster_gap_pct: float = 0.01) -> pd.DataFrame:
    """
    Add nearest support/resistance levels derived from multi-period VP.

    For each bar, collects all VP POC/VAH/VAL values across windows, clusters
    nearby levels, then finds the nearest level above (resistance) and below
    (support) current close.

    Args:
        df: DataFrame with VP columns (vp_poc_Xd, vp_vah_Xd, vp_val_Xd).
        vp_windows: List of VP window days to use. Default=[7, 14, 30].
        cluster_gap_pct: Min gap (fraction of close) for level clustering.

    Adds columns:
        nearest_resistance: closest S/R level above current close
        nearest_support: closest S/R level below current close
        dist_to_resistance: (nearest_resistance - close) / close (positive)
        dist_to_support: (close - nearest_support) / close (positive)
        sr_count: number of clustered S/R levels for this bar
    """
    if vp_windows is None:
        vp_windows = [7, 14, 30]

    # Gather all VP column names that exist
    sr_cols = []
    for w in vp_windows:
        for prefix in ["vp_poc", "vp_vah", "vp_val"]:
            col = f"{prefix}_{w}d"
            if col in df.columns:
                sr_cols.append(col)

    if not sr_cols:
        df["nearest_resistance"] = np.nan
        df["nearest_support"] = np.nan
        df["dist_to_resistance"] = np.nan
        df["dist_to_support"] = np.nan
        df["sr_count"] = 0
        return df

    closes = df["close"].values
    sr_matrix = df[sr_cols].values  # shape: (n_bars, n_levels)
    n = len(df)

    nearest_res = np.full(n, np.nan)
    nearest_sup = np.full(n, np.nan)
    sr_counts = np.zeros(n, dtype=int)

    for i in range(n):
        c = closes[i]
        if np.isnan(c) or c <= 0:
            continue

        levels = sr_matrix[i]
        valid = levels[~np.isnan(levels)]
        if len(valid) == 0:
            continue

        # Cluster nearby levels
        clustered = _cluster_levels(valid, c, min_gap_pct=cluster_gap_pct)
        sr_counts[i] = len(clustered)

        above = clustered[clustered > c]
        below = clustered[clustered < c]

        if len(above) > 0:
            nearest_res[i] = above.min()
        if len(below) > 0:
            nearest_sup[i] = below.max()

    df["nearest_resistance"] = nearest_res
    df["nearest_support"] = nearest_sup
    df["dist_to_resistance"] = np.where(
        ~np.isnan(nearest_res) & (closes > 0),
        (nearest_res - closes) / closes,
        np.nan
    )
    df["dist_to_support"] = np.where(
        ~np.isnan(nearest_sup) & (closes > 0),
        (closes - nearest_sup) / closes,
        np.nan
    )
    df["sr_count"] = sr_counts

    return df
