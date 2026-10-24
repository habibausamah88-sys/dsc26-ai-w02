
"""Data-quality audit utilities for Rawaj Pay."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _missing_mask(series: pd.Series) -> pd.Series:
    blank = series.map(
        lambda value: isinstance(value, str) and value.strip() == ""
    )
    return series.isna() | blank


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)

    n_rows = len(df)
    masks = {col: _missing_mask(df[col]) for col in df.columns}

    missing = {
        col: float(mask.mean()) if n_rows else 0.0
        for col, mask in masks.items()
    }

    ids = df.loc[~masks[id_col], id_col]
    duplicate_ids = ids[ids.duplicated(keep=False)].drop_duplicates().tolist()
    try:
        duplicate_ids = sorted(duplicate_ids)
    except TypeError:
        duplicate_ids = sorted(duplicate_ids, key=str)

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        values = df[col].loc[~masks[col]]
        if values.nunique(dropna=True) <= 1:
            constant_columns.append(col)

    target_missing = masks[target]
    labelled = df.loc[~target_missing]
    y = labelled[target]

    unlabelled_rows = int(target_missing.sum())
    positive_rate = float((y == 1).mean()) if len(y) else 0.0

    raw_time_missing = masks[time_col]
    parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    bad_timestamps = int((~raw_time_missing & parsed.isna()).sum())
    valid_times = parsed.dropna()

    if valid_times.empty:
        time_range = [None, None]
    else:
        time_range = [
            valid_times.min().isoformat(),
            valid_times.max().isoformat(),
        ]

    leakage_suspects = []

    if len(labelled) >= 20:
        y_binary = (y == 1).astype(int).to_numpy()

        if np.unique(y_binary).size == 2:
            for col in df.columns:
                if col in (target, id_col, time_col):
                    continue

                feature = labelled[col]
                missing_feature = masks[col].loc[labelled.index]
                nonmissing_values = feature.loc[~missing_feature]

                distinct = nonmissing_values.nunique(dropna=True)
                has_missing = bool(missing_feature.any())

                # Ignore columns with only one distinct value, counting
                # missingness as a possible value.
                if distinct + int(has_missing) <= 1:
                    continue

                suspect = False

                if (
                    pd.api.types.is_numeric_dtype(feature.dtype)
                    or pd.api.types.is_bool_dtype(feature.dtype)
                ):
                    usable = ~missing_feature
                    x = feature.loc[usable]
                    y_usable = y_binary[usable.to_numpy()]

                    if len(x) >= 20 and np.unique(y_usable).size == 2:
                        try:
                            auc = roc_auc_score(y_usable, x.to_numpy())
                            if auc >= 0.98 or auc <= 0.02:
                                suspect = True
                        except (TypeError, ValueError):
                            pass

                try:
                    missing_auc = roc_auc_score(
                        y_binary, missing_feature.astype(int).to_numpy()
                    )
                    if missing_auc >= 0.98 or missing_auc <= 0.02:
                        suspect = True
                except ValueError:
                    pass

                if suspect:
                    leakage_suspects.append(col)

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": sorted(constant_columns),
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": float(positive_rate),
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": sorted(leakage_suspects),
    }
