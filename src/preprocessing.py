"""
Leakage-free, date-aligned multi-scale sequence builder.

For each company:
  1. Split raw daily frame chronologically (train_ratio).
  2. Fit scalers ONLY on the train portion.
  3. Build daily/weekly/monthly views.
  4. For each daily target row, grab the most recent weekly/monthly windows
     whose date <= target date (true alignment).
  5. Return sequences + end-date so global chronological split is preserved.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .arima_features import add_arima_features
from .multiscale import to_weekly, to_monthly
from .feature_engineering import PRICE_FEATURES




# def _build_company_sequences(df_company: pd.DataFrame,
#                              features: list,
#                              daily_steps: int,
#                              weekly_steps: int,
#                              monthly_steps: int,
#                              train_ratio: float,
#                              arima_order=(2, 1, 2),
#                              use_garch=True,
#                              use_arima=True):          # ← new flag
#     """Returns dict with X_d, X_w, X_m, y, dates, scalers, split_date."""
#     df_company = df_company.sort_values("Date").set_index("Date")

#     n = len(df_company)
#     split_idx = int(n * train_ratio)
#     split_date = df_company.index[split_idx]

#     # ── ARIMA features (fit on train, extend to full) ──
#     if use_arima:
#         arima_feats = add_arima_features(
#             df_company["Daily_Return"].reset_index(drop=True),
#             train_size=split_idx,
#             order=arima_order,
#             use_garch=use_garch,
#         )
#         arima_feats.index = df_company.index
#         df_full = pd.concat([df_company, arima_feats], axis=1)
#         extra_cols = ["arima_fitted", "arima_resid"] + \
#                      (["garch_var"] if use_garch else [])
#     else:
#         df_full = df_company.copy()
#         extra_cols = []

#     all_features = features + extra_cols

#     # ── Resample to weekly / monthly ──
#     weekly_df = to_weekly(df_full)
#     monthly_df = to_monthly(df_full)

#     # ── Fit scalers on TRAIN ONLY ──
#     train_df = df_full.iloc[:split_idx]

#     scaler_daily = MinMaxScaler().fit(train_df[all_features])

#     weekly_train = weekly_df[weekly_df.index <= split_date]
#     scaler_weekly = MinMaxScaler().fit(
#         weekly_train[all_features] if len(weekly_train) > 0 else weekly_df[all_features]
#     )

#     monthly_train = monthly_df[monthly_df.index <= split_date]
#     scaler_monthly = MinMaxScaler().fit(
#         monthly_train[all_features] if len(monthly_train) > 0 else monthly_df[all_features]
#     )

#     # Targets scaler
#     target_cols = ["Daily_Return", "Weekly_Return", "Monthly_Return"]
#     scaler_target = MinMaxScaler().fit(train_df[target_cols])

#     # ── Transform full series ──
#     daily_scaled   = scaler_daily.transform(df_full[all_features])
#     weekly_scaled  = scaler_weekly.transform(weekly_df[all_features])
#     monthly_scaled = scaler_monthly.transform(monthly_df[all_features])
#     # targets_scaled = scaler_target.transform(df_full[target_cols])
#     targets_scaled = df_full[target_cols].values.astype(np.float32)

#     # ── Build aligned sequences ──
#     daily_dates   = df_full.index
#     weekly_dates  = weekly_df.index
#     monthly_dates = monthly_df.index

#     X_d, X_w, X_m, y, seq_dates = [], [], [], [], []

#     for i in range(daily_steps, len(daily_scaled)):
#         d_date = daily_dates[i]

#         X_d.append(daily_scaled[i - daily_steps:i])

#         w_valid = weekly_dates <= d_date
#         if w_valid.sum() >= weekly_steps:
#             w_idx = np.where(w_valid)[0][-weekly_steps:]
#             X_w.append(weekly_scaled[w_idx])
#         else:
#             continue

#         m_valid = monthly_dates <= d_date
#         if m_valid.sum() >= monthly_steps:
#             m_idx = np.where(m_valid)[0][-monthly_steps:]
#             X_m.append(monthly_scaled[m_idx])
#         else:
#             continue

#         y.append(targets_scaled[i])
#         seq_dates.append(d_date)

#     return {
#         "X_daily":   np.asarray(X_d, dtype=np.float32),
#         "X_weekly":  np.asarray(X_w, dtype=np.float32),
#         "X_monthly": np.asarray(X_m, dtype=np.float32),
#         "y":         np.asarray(y,   dtype=np.float32),
#         "dates":     np.asarray(seq_dates),
#         "scalers": {
#             "daily": scaler_daily, "weekly": scaler_weekly,
#             "monthly": scaler_monthly, "target": scaler_target,
#         },
#         "split_date": split_date,
#         "features": all_features,
#     }


# def prepare_dataset(df: pd.DataFrame, config: dict,
#                     max_sequences: int = None):        # ← subsample option
#     """Loop over companies, pool sequences, split globally by date."""
#     features = PRICE_FEATURES
#     daily_steps   = config["features"]["daily_steps"]
#     weekly_steps  = config["features"]["weekly_steps"]
#     monthly_steps = config["features"]["monthly_steps"]
#     train_ratio   = config["split"]["train_ratio"]

#     arima_cfg  = config.get("arima", {})
#     use_arima  = arima_cfg.get("enabled", False)       # ← respect config
#     arima_order = tuple(arima_cfg.get("order", (2, 1, 2)))
#     use_garch  = arima_cfg.get("use_garch", False) if use_arima else False

#     all_batches = []
#     companies = sorted(df["Scrip"].unique())
#     print(f"Processing {len(companies)} companies...")
#     print(f"ARIMA features: {'ON' if use_arima else 'OFF'}")

#     for k, comp in enumerate(companies, 1):
#         sub = df[df["Scrip"] == comp].copy()
#         try:
#             res = _build_company_sequences(
#                 sub, features,
#                 daily_steps, weekly_steps, monthly_steps,
#                 train_ratio,
#                 arima_order=arima_order,
#                 use_garch=use_garch,
#                 use_arima=use_arima,
#             )
#             if len(res["y"]) == 0:
#                 continue
#             all_batches.append(res)
#         except Exception as e:
#             print(f"  [{comp}] skipped: {e}")
#             continue

#         if k % 20 == 0:
#             print(f"  processed {k}/{len(companies)}")

#     if not all_batches:
#         raise ValueError(
#             "No companies produced sequences. Check that use_arima matches "
#             "the feature columns expected (arima_fitted/resid/garch_var)."
#         )

#     # Pool
#     X_d = np.concatenate([b["X_daily"]   for b in all_batches], axis=0)
#     X_w = np.concatenate([b["X_weekly"]  for b in all_batches], axis=0)
#     X_m = np.concatenate([b["X_monthly"] for b in all_batches], axis=0)
#     y   = np.concatenate([b["y"]         for b in all_batches], axis=0)
#     dates = np.concatenate([b["dates"]   for b in all_batches], axis=0)

#     # Global chronological split
#     order = np.argsort(dates)
#     X_d, X_w, X_m, y, dates = X_d[order], X_w[order], X_m[order], y[order], dates[order]

#     # Global target scaler fit on pooled training targets
#     y[:, 0] = np.clip(y[:, 0], -15, 15)    # daily returns
#     y[:, 1] = np.clip(y[:, 1], -30, 30)    # weekly returns
#     y[:, 2] = np.clip(y[:, 2], -50, 50)    # monthly returns

#     cut_tmp = int(len(y) * train_ratio)
#     global_target_scaler = MinMaxScaler().fit(y[:cut_tmp])
#     y = global_target_scaler.transform(y).astype(np.float32)

#     # Optional subsample for memory
#     if max_sequences is not None and len(y) > max_sequences:
#         rng = np.random.RandomState(42)
#         keep = rng.choice(len(y), size=max_sequences, replace=False)
#         keep.sort()
#         X_d, X_w, X_m, y, dates = (
#             X_d[keep], X_w[keep], X_m[keep], y[keep], dates[keep]
#         )
#         print(f"Subsampled to {max_sequences} sequences.")

#     n = len(y)
#     cut = int(n * train_ratio)
#     split_date = dates[cut]

#     train = {
#         "X_daily": X_d[:cut], "X_weekly": X_w[:cut],
#         "X_monthly": X_m[:cut], "y": y[:cut],
#         "dates": dates[:cut],
#     }
#     test = {
#         "X_daily": X_d[cut:], "X_weekly": X_w[cut:],
#         "X_monthly": X_m[cut:], "y": y[cut:],
#         "dates": dates[cut:],
#     }

#     print(f"\nSequences — total: {n}, train: {cut}, test: {n - cut}")
#     print(f"Split date: {split_date}")
#     print(f"Features used ({len(all_batches[0]['features'])}): "
#           f"{all_batches[0]['features']}")

#     return {
#         "train": train, "test": test,
#         "scalers": {
#             "daily":   all_batches[0]["scalers"]["daily"],
#             "weekly":  all_batches[0]["scalers"]["weekly"],
#             "monthly": all_batches[0]["scalers"]["monthly"],
#             "target":  global_target_scaler,       # ← the global one
#         },
#         "split_date": split_date,
#         "features": all_batches[0]["features"],
#     }

def _build_company_sequences(df_company: pd.DataFrame,
                             features: list,
                             daily_steps: int,
                             weekly_steps: int,
                             monthly_steps: int,
                             train_ratio: float,
                             arima_order=(2, 1, 2),
                             use_garch=True,
                             use_arima=True):
    """Returns dict with X_d, X_w, X_m, y, y_dir, dates, scalers, split_date."""
    df_company = df_company.sort_values("Date").set_index("Date")

    n = len(df_company)
    split_idx = int(n * train_ratio)
    split_date = df_company.index[split_idx]

    # ── ARIMA features (fit on train, extend to full) ──
    if use_arima:
        arima_feats = add_arima_features(
            df_company["Daily_Return"].reset_index(drop=True),
            train_size=split_idx,
            order=arima_order,
            use_garch=use_garch,
        )
        arima_feats.index = df_company.index
        df_full = pd.concat([df_company, arima_feats], axis=1)
        extra_cols = ["arima_fitted", "arima_resid"] + \
                     (["garch_var"] if use_garch else [])
    else:
        df_full = df_company.copy()
        extra_cols = []

    # ── Binary direction labels (1 if return > 0, else 0) ──
    direction_cols = ["Daily_Direction", "Weekly_Direction", "Monthly_Direction"]
    df_full["Daily_Direction"]   = (df_full["Daily_Return"]   > 0).astype(np.float32)
    df_full["Weekly_Direction"]  = (df_full["Weekly_Return"]  > 0).astype(np.float32)
    df_full["Monthly_Direction"] = (df_full["Monthly_Return"] > 0).astype(np.float32)

    all_features = features + extra_cols

    # ── Resample to weekly / monthly ──
    weekly_df = to_weekly(df_full)
    monthly_df = to_monthly(df_full)

    # ── Fit scalers on TRAIN ONLY ──
    train_df = df_full.iloc[:split_idx]

    scaler_daily = MinMaxScaler().fit(train_df[all_features])

    weekly_train = weekly_df[weekly_df.index <= split_date]
    scaler_weekly = MinMaxScaler().fit(
        weekly_train[all_features] if len(weekly_train) > 0 else weekly_df[all_features]
    )

    monthly_train = monthly_df[monthly_df.index <= split_date]
    scaler_monthly = MinMaxScaler().fit(
        monthly_train[all_features] if len(monthly_train) > 0 else monthly_df[all_features]
    )

    # Targets scaler (only for reference; we use global scaler later)
    target_cols = ["Daily_Return", "Weekly_Return", "Monthly_Return"]
    scaler_target = MinMaxScaler().fit(train_df[target_cols])

    # ── Transform full series ──
    daily_scaled   = scaler_daily.transform(df_full[all_features])
    weekly_scaled  = scaler_weekly.transform(weekly_df[all_features])
    monthly_scaled = scaler_monthly.transform(monthly_df[all_features])
    targets_scaled = df_full[target_cols].values.astype(np.float32)

    # ── Build aligned sequences ──
    daily_dates   = df_full.index
    weekly_dates  = weekly_df.index
    monthly_dates = monthly_df.index

    X_d, X_w, X_m, y, y_dir, seq_dates = [], [], [], [], [], []

    for i in range(daily_steps, len(daily_scaled)):
        d_date = daily_dates[i]

        X_d.append(daily_scaled[i - daily_steps:i])

        w_valid = weekly_dates <= d_date
        if w_valid.sum() >= weekly_steps:
            w_idx = np.where(w_valid)[0][-weekly_steps:]
            X_w.append(weekly_scaled[w_idx])
        else:
            continue

        m_valid = monthly_dates <= d_date
        if m_valid.sum() >= monthly_steps:
            m_idx = np.where(m_valid)[0][-monthly_steps:]
            X_m.append(monthly_scaled[m_idx])
        else:
            continue

        y.append(targets_scaled[i])
        y_dir.append(df_full[direction_cols].values[i])
        seq_dates.append(d_date)

    return {
        "X_daily":   np.asarray(X_d, dtype=np.float32),
        "X_weekly":  np.asarray(X_w, dtype=np.float32),
        "X_monthly": np.asarray(X_m, dtype=np.float32),
        "y":         np.asarray(y,     dtype=np.float32),
        "y_dir":     np.asarray(y_dir, dtype=np.float32),
        "dates":     np.asarray(seq_dates),
        "scalers": {
            "daily": scaler_daily, "weekly": scaler_weekly,
            "monthly": scaler_monthly, "target": scaler_target,
        },
        "split_date": split_date,
        "features": all_features,
    }


def prepare_dataset(df: pd.DataFrame, config: dict,
                    max_sequences: int = None):
    """Loop over companies, pool sequences, split globally by date."""
    features = PRICE_FEATURES
    daily_steps   = config["features"]["daily_steps"]
    weekly_steps  = config["features"]["weekly_steps"]
    monthly_steps = config["features"]["monthly_steps"]
    train_ratio   = config["split"]["train_ratio"]

    arima_cfg  = config.get("arima", {})
    use_arima  = arima_cfg.get("enabled", False)
    arima_order = tuple(arima_cfg.get("order", (2, 1, 2)))
    use_garch  = arima_cfg.get("use_garch", False) if use_arima else False

    all_batches = []
    companies = sorted(df["Scrip"].unique())
    print(f"Processing {len(companies)} companies...")
    print(f"ARIMA features: {'ON' if use_arima else 'OFF'}")

    for k, comp in enumerate(companies, 1):
        sub = df[df["Scrip"] == comp].copy()
        try:
            res = _build_company_sequences(
                sub, features,
                daily_steps, weekly_steps, monthly_steps,
                train_ratio,
                arima_order=arima_order,
                use_garch=use_garch,
                use_arima=use_arima,
            )
            if len(res["y"]) == 0:
                continue
            all_batches.append(res)
        except Exception as e:
            print(f"  [{comp}] skipped: {e}")
            continue

        if k % 20 == 0:
            print(f"  processed {k}/{len(companies)}")

    if not all_batches:
        raise ValueError(
            "No companies produced sequences. Check that use_arima matches "
            "the feature columns expected (arima_fitted/resid/garch_var)."
        )

    # Pool
    X_d    = np.concatenate([b["X_daily"]   for b in all_batches], axis=0)
    X_w    = np.concatenate([b["X_weekly"]  for b in all_batches], axis=0)
    X_m    = np.concatenate([b["X_monthly"] for b in all_batches], axis=0)
    y      = np.concatenate([b["y"]         for b in all_batches], axis=0)
    y_dir  = np.concatenate([b["y_dir"]     for b in all_batches], axis=0)
    dates  = np.concatenate([b["dates"]     for b in all_batches], axis=0)

    # Global chronological sort
    order = np.argsort(dates)
    X_d, X_w, X_m, y, y_dir, dates = (
        X_d[order], X_w[order], X_m[order], y[order], y_dir[order], dates[order]
    )

    # Clip outliers on raw returns, then fit global scaler
    y[:, 0] = np.clip(y[:, 0], -15, 15)
    y[:, 1] = np.clip(y[:, 1], -30, 30)
    y[:, 2] = np.clip(y[:, 2], -50, 50)

    cut_tmp = int(len(y) * train_ratio)
    global_target_scaler = MinMaxScaler().fit(y[:cut_tmp])
    y = global_target_scaler.transform(y).astype(np.float32)

    # Optional subsample
    if max_sequences is not None and len(y) > max_sequences:
        rng = np.random.RandomState(42)
        keep = rng.choice(len(y), size=max_sequences, replace=False)
        keep.sort()
        X_d, X_w, X_m, y, y_dir, dates = (
            X_d[keep], X_w[keep], X_m[keep], y[keep], y_dir[keep], dates[keep]
        )
        print(f"Subsampled to {max_sequences} sequences.")

    n = len(y)
    cut = int(n * train_ratio)
    split_date = dates[cut]

    train = {
        "X_daily": X_d[:cut], "X_weekly": X_w[:cut],
        "X_monthly": X_m[:cut], "y": y[:cut], "y_dir": y_dir[:cut],
        "dates": dates[:cut],
    }
    test = {
        "X_daily": X_d[cut:], "X_weekly": X_w[cut:],
        "X_monthly": X_m[cut:], "y": y[cut:], "y_dir": y_dir[cut:],
        "dates": dates[cut:],
    }

    print(f"\nSequences — total: {n}, train: {cut}, test: {n - cut}")
    print(f"Split date: {split_date}")
    print(f"Features used ({len(all_batches[0]['features'])}): "
          f"{all_batches[0]['features']}")

    return {
        "train": train, "test": test,
        "scalers": {
            "daily":   all_batches[0]["scalers"]["daily"],
            "weekly":  all_batches[0]["scalers"]["weekly"],
            "monthly": all_batches[0]["scalers"]["monthly"],
            "target":  global_target_scaler,
        },
        "split_date": split_date,
        "features": all_batches[0]["features"],
    }