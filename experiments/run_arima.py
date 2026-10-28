"""ARIMA-only baseline: fit per company on training returns, walk-forward to test."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath("."))

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils import load_config
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features


def main():
    cfg = load_config("config/config.yaml")
    df = load_raw_data(cfg["data"]["raw_path"])
    df = filter_valid_companies(df, cfg["data"]["min_records"])
    df = add_features(df)

    order = tuple(cfg["arima"]["order"])
    all_true, all_pred = [], []

    for comp, sub in df.groupby("Scrip"):
        sub = sub.sort_values("Date")
        rets = sub["Daily_Return"].values
        n = len(rets)
        cut = int(n * cfg["split"]["train_ratio"])
        try:
            res = ARIMA(rets[:cut], order=order).fit()
            res_ext = res.extend(rets)
            preds = res_ext.fittedvalues[cut:]
            actual = rets[cut:]
            all_true.append(actual)
            all_pred.append(preds)
        except Exception:
            continue

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    print("ARIMA baseline:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"  R2  : {r2_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()