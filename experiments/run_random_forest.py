# """
# Random Forest baseline on DSE.
# Uses the same technical-indicator features as the deep learning model,
# flattened into a single vector per sample (no temporal windowing).
# Predicts daily/weekly/monthly returns.
# """
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, balanced_accuracy_score

# from src.utils import load_config
# from src.data_loader import load_raw_data, filter_valid_companies
# from src.feature_engineering import add_features
# from src.feature_engineering import PRICE_FEATURES


# def sharpe(y_true, y_pred, periods=252):
#     pos = np.sign(y_pred)
#     strat = pos * y_true
#     if strat.std() == 0:
#         return 0.0
#     return (strat.mean() / strat.std()) * np.sqrt(periods)


# def main():
#     cfg = load_config("config/config.yaml")
#     train_ratio = cfg["split"]["train_ratio"]

#     print("Loading data...")
#     df = load_raw_data(cfg["data"]["raw_path"])
#     df = filter_valid_companies(df, cfg["data"]["min_records"])
#     df = add_features(df)

#     # Target columns
#     targets = ["Daily_Return", "Weekly_Return", "Monthly_Return"]
#     feature_cols = PRICE_FEATURES

#     print(f"Features: {len(feature_cols)}")
#     print(f"Targets: {targets}")

#     # Chronological split (global)
#     df = df.sort_values("Date").reset_index(drop=True)

#     # Clip outliers
#     for col, bound in zip(targets, [15, 30, 50]):
#         df[col] = df[col].clip(-bound, bound)

#     n = len(df)
#     cut = int(n * train_ratio)
#     split_date = df["Date"].iloc[cut]

#     train_df = df.iloc[:cut]
#     test_df = df.iloc[cut:]

#     print(f"\nSplit date: {split_date}")
#     print(f"Train: {len(train_df)}, Test: {len(test_df)}")

#     X_train = train_df[feature_cols].values
#     X_test = test_df[feature_cols].values

#     results = {}

#     for target in targets:
#         print(f"\nTraining Random Forest for {target}...")
#         y_train = train_df[target].values
#         y_test = test_df[target].values

#         # Subsample training for speed (RF can't handle 1M rows easily)
#         max_train = 200_000
#         if len(X_train) > max_train:
#             idx = np.random.RandomState(42).choice(len(X_train), size=max_train, replace=False)
#             X_tr = X_train[idx]
#             y_tr = y_train[idx]
#         else:
#             X_tr = X_train
#             y_tr = y_train

#         rf = RandomForestRegressor(
#             n_estimators=100,
#             max_depth=12,
#             n_jobs=-1,
#             random_state=42,
#             verbose=1,
#         )
#         rf.fit(X_tr, y_tr)

#         pred = rf.predict(X_test)

#         # Metrics
#         mse = mean_squared_error(y_test, pred)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
#         dir_acc = balanced_accuracy_score(
#             (y_test > 0).astype(int),
#             (pred > 0).astype(int)
#         ) * 100
#         shp = sharpe(y_test, pred, periods=252 if target == "Daily_Return" else (52 if target == "Weekly_Return" else 12))

#         print(f"  R2={r2:.4f}  DirAcc={dir_acc:.2f}%  Sharpe={shp:.3f}")

#         results[target] = {
#             "MSE": mse, "RMSE": rmse, "MAE": mae,
#             "R2": r2, "DirAcc": dir_acc, "Sharpe": shp
#         }

#     # Save
#     out_dir = os.path.join(cfg["data"]["output_dir"], "metrics")
#     os.makedirs(out_dir, exist_ok=True)
#     df_out = pd.DataFrame(results).T
#     out_path = os.path.join(out_dir, "random_forest_metrics.xlsx")
#     df_out.to_excel(out_path)
#     print(f"\nSaved to {out_path}")
#     print(df_out)


# if __name__ == "__main__":
#     main()

"""
Random Forest baseline on DSE — leak-free version.

Fixes applied:
1. Removed Momentum5, Momentum10 (directly encode target returns)
2. Lagged all features by 1 day (RF sees yesterday's data only)
3. Subsample to 200K for speed
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, balanced_accuracy_score

from src.utils import load_config
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features


# ── Leak-free feature set ──
# Removed: Momentum5, Momentum10 (they encode the target)
FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "MA7", "MA21", "MA50",
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Middle",
    "ATR", "Volume_MA", "Volume_Ratio",
]

TARGETS = ["Daily_Return", "Weekly_Return", "Monthly_Return"]
MAX_TRAIN = 200_000


def sharpe(y_true, y_pred, periods=252):
    pos = np.sign(y_pred)
    strat = pos * y_true
    if strat.std() == 0:
        return 0.0
    return (strat.mean() / strat.std()) * np.sqrt(periods)


def main():
    cfg = load_config("config/config.yaml")
    train_ratio = cfg["split"]["train_ratio"]

    print("Loading data...")
    df = load_raw_data(cfg["data"]["raw_path"])
    df = filter_valid_companies(df, cfg["data"]["min_records"])
    df = add_features(df)

    # ── LAG FEATURES BY 1 DAY (RF sees yesterday's data only) ──
    print("Laging features by 1 day per company...")
    for col in FEATURES:
        df[col] = df.groupby("Scrip")[col].shift(1)

    # Drop rows where lags are NaN (first row per company)
    df = df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    print(f"After lagging: {len(df)} rows")

    # Clip target outliers (same as DL preprocessing)
    for col, bound in zip(TARGETS, [15, 30, 50]):
        df[col] = df[col].clip(-bound, bound)

    # Chronological split
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    cut = int(n * train_ratio)
    split_date = df["Date"].iloc[cut]

    print(f"\nSplit date: {split_date}")
    print(f"Train: {cut}, Test: {n - cut}")

    X_train_full = df[FEATURES].iloc[:cut].values
    X_test       = df[FEATURES].iloc[cut:].values

    results = {}

    for target in TARGETS:
        print(f"\n{'='*50}")
        print(f"Training Random Forest for {target}")
        print(f"{'='*50}")

        y_train_full = df[target].iloc[:cut].values
        y_test       = df[target].iloc[cut:].values

        # Subsample for speed
        if len(X_train_full) > MAX_TRAIN:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_train_full), size=MAX_TRAIN, replace=False)
            X_train = X_train_full[idx]
            y_train = y_train_full[idx]
            print(f"Subsampled training to {MAX_TRAIN}")
        else:
            X_train = X_train_full
            y_train = y_train_full

        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,       # shallower = less memorization
            min_samples_leaf=50,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        rf.fit(X_train, y_train)
        print("Fit complete. Predicting...")

        pred = rf.predict(X_test)

        # Winsorize predictions to match truth
        bound = {"Daily_Return": 15, "Weekly_Return": 30, "Monthly_Return": 50}[target]
        pred = np.clip(pred, -bound, bound)

        mse  = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)
        da   = balanced_accuracy_score(
            (y_test > 0).astype(int), (pred > 0).astype(int)
        ) * 100
        periods = {"Daily_Return": 252, "Weekly_Return": 52, "Monthly_Return": 12}[target]
        shp = sharpe(y_test, pred, periods=periods)

        print(f"  MSE    : {mse:.4f}")
        print(f"  RMSE   : {rmse:.4f}")
        print(f"  MAE    : {mae:.4f}")
        print(f"  R2     : {r2:.4f}")
        print(f"  DirAcc : {da:.2f}%")
        print(f"  Sharpe : {shp:.3f}")

        results[target] = {
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "R2": r2, "DirAcc": da, "Sharpe": shp,
        }

    # Save
    out_dir = os.path.join(cfg["data"]["output_dir"], "metrics")
    os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(results).T
    out_path = os.path.join(out_dir, "random_forest_metrics.xlsx")
    df_out.to_excel(out_path)

    print(f"\n{'='*50}")
    print(f"FINAL RANDOM FOREST RESULTS")
    print(f"{'='*50}")
    print(df_out)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()