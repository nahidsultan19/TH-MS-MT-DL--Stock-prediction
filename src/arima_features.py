"""
Leak-free ARIMA & GARCH features.

Fit ARIMA on training portion only, then use Kalman filter (`extend`) to
produce one-step-ahead forecasts for the full series. Residuals and
conditional variance (GARCH) are appended as extra columns.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

warnings.filterwarnings("ignore")


# def add_arima_features(series: pd.Series, train_size: int,
#                        order=(2, 1, 2), use_garch=True) -> pd.DataFrame:
#     """
#     series: 1-D pd.Series of returns (percent).
#     train_size: index at which train ends (test begins).
#     Returns DataFrame with columns: arima_fitted, arima_resid, [garch_var].
#     """
#     series = series.astype(float).reset_index(drop=True)
#     out = pd.DataFrame(index=series.index)

#     try:
#         train = series.iloc[:train_size]
#         res = ARIMA(train, order=order).fit()

#         # Extend fitted model to the full series (Kalman filter, no leakage)
#         # res_ext = res.extend(series)
#         # Ensure the series passed to extend has a RangeIndex (or any index)
#         res_ext = res.extend(pd.Series(series, index=range(len(series))))
#         fitted = res_ext.fittedvalues
#         resid = series - fitted

#         out["arima_fitted"] = fitted
#         out["arima_resid"] = resid

#         if use_garch:
#             try:
#                 g = arch_model(resid.dropna() * 1.0, vol="Garch", p=1, q=1,
#                                rescale=False)
#                 gres = g.fit(disp="off")
#                 # Conditional variance aligned to residual index
#                 var = pd.Series(np.nan, index=series.index)
#                 var.loc[resid.dropna().index] = gres.conditional_volatility ** 2
#                 out["garch_var"] = var.ffill().fillna(0.0)
#             except Exception:
#                 out["garch_var"] = 0.0

#     except Exception as e:
#         # Fallback: zeros (still keeps shape consistent)
#         print(f"    ARIMA failed: {e}")
#         out["arima_fitted"] = 0.0
#         out["arima_resid"] = series
#         if use_garch:
#             out["garch_var"] = 0.0

#     return out.fillna(0.0)

def add_arima_features(series, train_size, order=(2, 1, 2), use_garch=True):
    """
    Leak-free ARIMA + GARCH features via Kalman-filter extension.
    - Fit ARIMA on train portion only.
    - Extend to full series to get one-step-ahead predictions (no leakage).
    """
    series = pd.Series(series).astype(float).reset_index(drop=True)
    n = len(series)
    out = pd.DataFrame(index=range(n))

    try:
        # Give the training slice its own clean index 0..cut-1
        train = series.iloc[:train_size].reset_index(drop=True)
        res = ARIMA(train, order=order).fit()

        # Extend using the FULL series but with an index that starts
        # where the model ended — append new index positions beyond train.
        # Simplest robust approach: refit-and-filter using `res.extend`
        # with the tail of the series indexed as train.index[-1]+1, ...
        tail = series.iloc[train_size:].reset_index(drop=True)
        tail.index = range(train_size, n)  # 0-based extension of train's 0..cut-1

        full_indexed = pd.concat([
            train,
            pd.Series(tail.values, index=tail.index)
        ])
        # Sanity check: index must be strictly increasing and start at 0
        assert full_indexed.index.is_monotonic_increasing

        res_ext = res.extend(full_indexed)
        fitted = res_ext.fittedvalues
        resid  = full_indexed - fitted

        out["arima_fitted"] = fitted.values
        out["arima_resid"]  = resid.values

        if use_garch:
            try:
                g = arch_model(resid.dropna() * 1.0,
                               vol="Garch", p=1, q=1, rescale=False)
                gres = g.fit(disp="off")
                var = pd.Series(np.nan, index=range(n))
                var.loc[resid.dropna().index] = gres.conditional_volatility ** 2
                out["garch_var"] = var.ffill().fillna(0.0).values
            except Exception:
                out["garch_var"] = 0.0
    except Exception as e:
        print(f"    ARIMA failed: {e}")
        out["arima_fitted"] = 0.0
        out["arima_resid"]  = series.values
        if use_garch:
            out["garch_var"] = 0.0

    return out.fillna(0.0)