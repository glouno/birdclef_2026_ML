import numpy as np
import pandas as pd


def build_soundscape_context_features(
        soundscapes: pd.DataFrame,
        filenames: np.ndarray | list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build site one-hot and cyclic datetime features for soundscape rows."""
    if "filename" not in soundscapes or "site" not in soundscapes or "datetime" not in soundscapes:
        raise ValueError("soundscapes must include filename, site, and datetime columns")

    base = soundscapes[["filename", "site", "datetime"]].drop_duplicates("filename")
    base = base.set_index("filename")
    filenames_arr = np.asarray(filenames)
    merged = base.reindex(filenames_arr)

    if merged["site"].isna().any() or merged["datetime"].isna().any():
        raise ValueError("filenames contain entries missing from soundscapes metadata")

    sites = np.sort(base["site"].dropna().unique())
    site_cat = pd.Categorical(merged["site"], categories=sites, ordered=True)
    site_one_hot = pd.get_dummies(site_cat, prefix="site", dtype=float)

    dt = pd.to_datetime(merged["datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("datetime contains invalid or missing values")

    hour = dt.dt.hour.astype(float)
    # dayofyear = dt.dt.dayofyear.astype(float)
    month = dt.dt.month.astype(float)
    hour_rad = 2.0 * np.pi * hour / 24.0
    # day_rad = 2.0 * np.pi * dayofyear / 365.0
    month_rad = 2.0 * np.pi * month / 12.0

    cyc_features = np.column_stack(
        [
            np.sin(hour_rad),
            np.cos(hour_rad),
            # np.sin(day_rad),
            # np.cos(day_rad),
            np.sin(month_rad),
            np.cos(month_rad),
        ]
    )

    feature_matrix = np.hstack([site_one_hot.to_numpy(dtype=float), cyc_features])
    feature_names = [
        *site_one_hot.columns.tolist(),
        "hour_sin",
        "hour_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]
    return feature_matrix, feature_names
