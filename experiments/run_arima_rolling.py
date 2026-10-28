"""
Rolling-window ARIMA baseline on DSE — fast version with winsorization.
Fixed 252-day window, refit every 10 steps, 75 companies.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, balanced_accuracy_score

from src.utils import load_config
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features


ARIMA_ORDER = (1, 1, 1)
WINDOW = 252
REFIT_STRIDE = 10
MAX_COMPANIES = 75


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

    companies = sorted(df["Scrip"].unique())[:MAX_COMPANIES]
    print(f"Running ARIMA on {len(companies)} companies")
    print(f"Order={ARIMA_ORDER}  Window={WINDOW}  Refit every {REFIT_STRIDE}")

    all_true, all_pred = [], []
    t0 = time.time()

    for i, comp in enumerate(companies, 1):
        sub = df[df["Scrip"] == comp].sort_values("Date")
        returns = sub["Daily_Return"].values
        n = len(returns)
        if n < WINDOW + 100:
            continue

        split_idx = int(n * train_ratio)
        last_fit = None

        for t in range(split_idx, n):
            if last_fit is None or (t - split_idx) % REFIT_STRIDE == 0:
                try:
                    start = max(0, t - WINDOW)
                    model = ARIMA(returns[start:t], order=ARIMA_ORDER)
                    last_fit = model.fit()
                except Exception:
                    continue
            try:
                pred = last_fit.forecast(steps=1)[0]
                all_true.append(returns[t])
                all_pred.append(pred)
            except Exception:
                continue

        elapsed = time.time() - t0
        if i % 5 == 0 or i == len(companies):
            eta = elapsed / i * (len(companies) - i)
            print(f"  [{i:3d}/{len(companies)}]  preds={len(all_true):6d}  "
                  f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s", flush=True)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # ── WINSORIZE (±15% to match DL preprocessing) ──
    print(f"\nBefore winsorization:")
    print(f"  True: min={all_true.min():.2f}, max={all_true.max():.2f}, std={all_true.std():.4f}")
    print(f"  Pred: min={all_pred.min():.2f}, max={all_pred.max():.2f}, std={all_pred.std():.4f}")

    all_true = np.clip(all_true, -15, 15)
    all_pred = np.clip(all_pred, -15, 15)

    print(f"\nAfter winsorization (±15%):")
    print(f"  True: min={all_true.min():.2f}, max={all_true.max():.2f}, std={all_true.std():.4f}")
    print(f"  Pred: min={all_pred.min():.2f}, max={all_pred.max():.2f}, std={all_pred.std():.4f}")

    print(f"\n{'='*50}")
    print(f"ARIMA Rolling Forecast Results (winsorized)")
    print(f"{'='*50}")
    print(f"Total predictions: {len(all_true)}")

    if len(all_true) > 0:
        mse = mean_squared_error(all_true, all_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_true, all_pred)
        r2 = r2_score(all_true, all_pred)
        dir_acc = balanced_accuracy_score(
            (all_true > 0).astype(int), (all_pred > 0).astype(int)
        ) * 100
        shp = sharpe(all_true, all_pred)

        print(f"MSE   : {mse:.4f}")
        print(f"RMSE  : {rmse:.4f}")
        print(f"MAE   : {mae:.4f}")
        print(f"R2    : {r2:.4f}")
        print(f"DirAcc: {dir_acc:.2f}%")
        print(f"Sharpe: {shp:.3f}")

        out_dir = os.path.join(cfg["data"]["output_dir"], "metrics")
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({
            "MSE": [mse], "RMSE": [rmse], "MAE": [mae],
            "R2": [r2], "DirAcc": [dir_acc], "Sharpe": [shp]
        }, index=["Daily"]).to_excel(os.path.join(out_dir, "arima_metrics.xlsx"))
        print(f"\nSaved to {out_dir}/arima_metrics.xlsx")


if __name__ == "__main__":
    main()

# """
# Rolling-window ARIMA baseline on DSE — fast version.
# Fixed 252-day window, refit every 5 steps, 75 companies.
# """
# import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
# warnings.filterwarnings("ignore")

# import time
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, balanced_accuracy_score

# from src.utils import load_config
# from src.data_loader import load_raw_data, filter_valid_companies
# from src.feature_engineering import add_features


# # ── Fast settings ──
# ARIMA_ORDER = (1, 1, 1)      # simplest, fastest
# WINDOW = 252                 # fixed rolling window
# REFIT_STRIDE = 10            # refit every 10 steps
# MAX_COMPANIES = 75


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

#     companies = sorted(df["Scrip"].unique())[:MAX_COMPANIES]
#     print(f"Running ARIMA on {len(companies)} companies")
#     print(f"Order={ARIMA_ORDER}  Window={WINDOW}  Refit every {REFIT_STRIDE}")

#     all_true, all_pred = [], []
#     t0 = time.time()

#     for i, comp in enumerate(companies, 1):
#         sub = df[df["Scrip"] == comp].sort_values("Date")
#         returns = sub["Daily_Return"].values
#         n = len(returns)
#         if n < WINDOW + 100:
#             continue

#         split_idx = int(n * train_ratio)
#         last_fit = None

#         for t in range(split_idx, n):
#             if last_fit is None or (t - split_idx) % REFIT_STRIDE == 0:
#                 try:
#                     start = max(0, t - WINDOW)
#                     model = ARIMA(returns[start:t], order=ARIMA_ORDER)
#                     last_fit = model.fit()
#                 except Exception:
#                     continue
#             try:
#                 pred = last_fit.forecast(steps=1)[0]
#                 all_true.append(returns[t])
#                 all_pred.append(pred)
#             except Exception:
#                 continue

#         elapsed = time.time() - t0
#         if i % 5 == 0 or i == len(companies):
#             eta = elapsed / i * (len(companies) - i)
#             print(f"  [{i:3d}/{len(companies)}]  preds={len(all_true):6d}  "
#                   f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s", flush=True)

#     all_true = np.array(all_true)
#     all_pred = np.array(all_pred)

#     print(f"\n{'='*50}")
#     print(f"ARIMA Rolling Forecast Results")
#     print(f"{'='*50}")
#     print(f"Total predictions: {len(all_true)}")

#     if len(all_true) > 0:
#         mse = mean_squared_error(all_true, all_pred)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(all_true, all_pred)
#         r2 = r2_score(all_true, all_pred)
#         dir_acc = balanced_accuracy_score(
#             (all_true > 0).astype(int), (all_pred > 0).astype(int)
#         ) * 100
#         shp = sharpe(all_true, all_pred)

#         print(f"MSE   : {mse:.4f}")
#         print(f"RMSE  : {rmse:.4f}")
#         print(f"MAE   : {mae:.4f}")
#         print(f"R2    : {r2:.4f}")
#         print(f"DirAcc: {dir_acc:.2f}%")
#         print(f"Sharpe: {shp:.3f}")

#         out_dir = os.path.join(cfg["data"]["output_dir"], "metrics")
#         os.makedirs(out_dir, exist_ok=True)
#         pd.DataFrame({
#             "MSE": [mse], "RMSE": [rmse], "MAE": [mae],
#             "R2": [r2], "DirAcc": [dir_acc], "Sharpe": [shp]
#         }, index=["Daily"]).to_excel(os.path.join(out_dir, "arima_metrics.xlsx"))
#         print(f"\nSaved to {out_dir}/arima_metrics.xlsx")


# if __name__ == "__main__":
#     main()