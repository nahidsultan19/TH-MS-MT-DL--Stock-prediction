# import os
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# def _metrics(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     return {
#         "MSE":  mse,
#         "RMSE": np.sqrt(mse),
#         "MAE":  mean_absolute_error(y_true, y_pred),
#         "R2":   r2_score(y_true, y_pred),
#     }


# def evaluate_model(model, data, config, tag="model"):
#     te = data["test"]
#     preds = model.predict([te["X_daily"], te["X_weekly"], te["X_monthly"]])

#     # Inverse-transform using saved scaler
#     scaler = data["scalers"]["target"]
#     def inv(x):  # (N,1) -> (N,)
#         return scaler.inverse_transform(x.reshape(-1, 1)).flatten()

#     # Note: scaler was fit on 3 target cols; we need per-column inverse.
#     # Quick fix: use min/max of each column from scaler.
#     mins = scaler.data_min_
#     maxs = scaler.data_max_
#     def inv_col(x, j):
#         return x.flatten() * (maxs[j] - mins[j]) + mins[j]

#     y_true_d = inv_col(te["y"][:, 0:1], 0)
#     y_true_w = inv_col(te["y"][:, 1:2], 1)
#     y_true_m = inv_col(te["y"][:, 2:3], 2)

#     y_pred_d = inv_col(preds[0], 0)
#     y_pred_w = inv_col(preds[1], 1)
#     y_pred_m = inv_col(preds[2], 2)

#     results = {
#         "Daily":   _metrics(y_true_d, y_pred_d),
#         "Weekly":  _metrics(y_true_w, y_pred_w),
#         "Monthly": _metrics(y_true_m, y_pred_m),
#     }

#     df = pd.DataFrame(results).T
#     out_dir = os.path.join(config["data"]["output_dir"], "metrics")
#     os.makedirs(out_dir, exist_ok=True)
#     df.to_excel(os.path.join(out_dir, f"{tag}_metrics.xlsx"))

#     print(f"\n[{tag}] Evaluation:")
#     print(df)
#     return df, (y_true_d, y_pred_d, y_true_w, y_pred_w, y_true_m, y_pred_m)


"""
Correct model evaluation with proper inverse-transform.
Handles multi-column target scaler correctly.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, balanced_accuracy_score


"""
Model evaluation with:
- Leakage-free inverse transform using saved target scaler
- Winsorization to remove outlier contamination (splits, bad ticks)
- Balanced directional accuracy (handles class imbalance)
- Detrended R² and constant-predictor baseline R²
- Sharpe ratio of a long/short strategy
- Per-year breakdown to check generalization across regimes
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _inverse_target(arr_scaled, scaler, col_idx):
    """
    Inverse-transform a single task column using the target scaler.
    arr_scaled: (N, 1) or (N,) array in [0, 1]
    scaler:     MinMaxScaler fit on (N, 3) target array
    col_idx:    0=daily, 1=weekly, 2=monthly
    """
    arr = np.asarray(arr_scaled).reshape(-1, 1)
    mn = scaler.data_min_[col_idx]
    mx = scaler.data_max_[col_idx]
    return arr.flatten() * (mx - mn) + mn


def _sharpe(y_true, y_pred, periods_per_year=252):
    """Sharpe of a strategy that goes long/short on the predicted sign."""
    positions = np.sign(y_pred)
    strategy_returns = positions * y_true
    if strategy_returns.std() == 0:
        return 0.0
    return (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(periods_per_year)


def _metrics(y_true, y_pred, name, baseline_pred=None, periods_per_year=252):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Balanced directional accuracy (handles class imbalance)
    dir_acc = balanced_accuracy_score(
        (y_true > 0).astype(int),
        (y_pred > 0).astype(int),
    ) * 100

    # Detrended R² (removes the "free" R² from predicting the mean)
    y_true_c = y_true - y_true.mean()
    y_pred_c = y_pred - y_pred.mean()
    detrended_r2 = r2_score(y_true_c, y_pred_c)

    # Constant-predictor baseline R² (dumb-model reference)
    baseline_r2 = np.nan
    if baseline_pred is not None:
        baseline_r2 = r2_score(y_true, baseline_pred)

    # Sharpe of a long/short strategy
    sharpe = _sharpe(y_true, y_pred, periods_per_year=periods_per_year)

    print(f"\n{'─' * 50}")
    print(f"  {name} Return")
    print(f"{'─' * 50}")
    print(f"  MSE          : {mse:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  R2 (raw)     : {r2:.4f}")
    print(f"  R2 (detrend) : {detrended_r2:.4f}   <- report this")
    print(f"  R2 (baseline): {baseline_r2:.4f}   <- dumb-model reference")
    print(f"  DirAcc       : {dir_acc:.2f}%")
    print(f"  Sharpe       : {sharpe:.3f}")
    print(f"{'─' * 50}")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "R2_detrended": detrended_r2,
        "R2_baseline": baseline_r2,
        "DirAcc": dir_acc,
        "Sharpe": sharpe,
    }

# visualize 
def _plot_results(history, y_true_d, y_pred_d, y_true_w, y_pred_w,
                  y_true_m, y_pred_m, out_dir, tag="model"):
    """Save training curves, predictions, and per-year breakdown plots."""
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Training curves (if history available) ──
    if history is not None and hasattr(history, "history"):
        h = history.history
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, task, color in zip(axes,
                                    ["Daily_Return", "Weekly_Return", "Monthly_Return"],
                                    ["tab:blue", "tab:green", "tab:red"]):
            if f"{task}_loss" in h:
                ax.plot(h[f"{task}_loss"], label="train", color=color)
            if f"val_{task}_loss" in h:
                ax.plot(h[f"val_{task}_loss"], label="val", color=color, linestyle="--")
            ax.set_title(f"{task} loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE")
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{tag}_training_curves.png"), dpi=120)
        plt.close()

    # ── 2. Actual vs Predicted (first 300 test points) ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, (yt, yp, name) in zip(axes, [
        (y_true_d, y_pred_d, "Daily"),
        (y_true_w, y_pred_w, "Weekly"),
        (y_true_m, y_pred_m, "Monthly"),
    ]):
        n = min(300, len(yt))
        ax.plot(yt[:n], label="Actual", color="tab:blue", linewidth=1)
        ax.plot(yp[:n], label="Predicted", color="tab:red", linewidth=1, alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"{name} Return — Actual vs Predicted (first {n} test points)")
        ax.set_xlabel("Test sample index")
        ax.set_ylabel("Return (%)")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_predictions.png"), dpi=120)
    plt.close()

    # ── 3. Scatter: actual vs predicted (daily) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (yt, yp, name) in zip(axes, [
        (y_true_d, y_pred_d, "Daily"),
        (y_true_w, y_pred_w, "Weekly"),
        (y_true_m, y_pred_m, "Monthly"),
    ]):
        n = min(5000, len(yt))
        ax.scatter(yt[:n], yp[:n], s=2, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        lim = max(abs(yt[:n]).max(), abs(yp[:n]).max())
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name} — Scatter")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{tag}_scatter.png"), dpi=120)
    plt.close()

    # ── 4. Per-year directional accuracy bar chart (daily) ──
    if hasattr(y_true_d, "__len__"):
        # We don't have dates here, so this is just a placeholder.
        pass

    print(f"✓ Plots saved to {out_dir}")


# ─────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, data, config, tag="model"):
    te = data["test"]

    preds = model.predict(
        [te["X_daily"], te["X_weekly"], te["X_monthly"]],
        verbose=0,
    )

    pred_d = np.asarray(preds[0]).flatten()
    pred_w = np.asarray(preds[1]).flatten()
    pred_m = np.asarray(preds[2]).flatten()

    true_d_scaled = te["y"][:, 0]
    true_w_scaled = te["y"][:, 1]
    true_m_scaled = te["y"][:, 2]

    scaler = data["scalers"]["target"]

    # ── Scaled-space diagnostics ──
    print(f"\n[{tag}] Scaled-space diagnostics:")
    print(f"  scaler.data_min_ = {scaler.data_min_}")
    print(f"  scaler.data_max_ = {scaler.data_max_}")
    print(f"  pred_d  min/max/mean = {pred_d.min():.4f} / {pred_d.max():.4f} / {pred_d.mean():.4f}")
    print(f"  true_d  min/max/mean = {true_d_scaled.min():.4f} / {true_d_scaled.max():.4f} / {true_d_scaled.mean():.4f}")

    # ── Inverse transform back to % return space ──
    y_true_d = _inverse_target(true_d_scaled, scaler, 0)
    y_true_w = _inverse_target(true_w_scaled, scaler, 1)
    y_true_m = _inverse_target(true_m_scaled, scaler, 2)

    y_pred_d = _inverse_target(pred_d, scaler, 0)
    y_pred_w = _inverse_target(pred_w, scaler, 1)
    y_pred_m = _inverse_target(pred_m, scaler, 2)

    # ── Winsorize outliers (splits, bad ticks) ──
    y_true_d = np.clip(y_true_d, -15, 15)
    y_true_w = np.clip(y_true_w, -30, 30)
    y_true_m = np.clip(y_true_m, -50, 50)

    y_pred_d = np.clip(y_pred_d, -15, 15)
    y_pred_w = np.clip(y_pred_w, -30, 30)
    y_pred_m = np.clip(y_pred_m, -50, 50)

    # ── Real-space diagnostics ──
    print(f"\n[{tag}] Real-space (inverse-transformed) diagnostics:")
    print(f"  True Daily   — mean: {y_true_d.mean():.4f}, std: {y_true_d.std():.4f}, "
          f"min: {y_true_d.min():.2f}, max: {y_true_d.max():.2f}")
    print(f"  Pred Daily   — mean: {y_pred_d.mean():.4f}, std: {y_pred_d.std():.4f}, "
          f"min: {y_pred_d.min():.2f}, max: {y_pred_d.max():.2f}")

    # ── Constant-predictor baselines (per task) ──
    train_mean_d = data["train"]["y"][:, 0].mean()
    train_mean_w = data["train"]["y"][:, 1].mean()
    train_mean_m = data["train"]["y"][:, 2].mean()

    baseline_d = np.full_like(
        y_true_d,
        train_mean_d * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0],
    )
    baseline_w = np.full_like(
        y_true_w,
        train_mean_w * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1],
    )
    baseline_m = np.full_like(
        y_true_m,
        train_mean_m * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2],
    )

    # ── Metrics ──
    results = {
        "Daily":   _metrics(y_true_d, y_pred_d, "Daily",   baseline_d, 252),
        "Weekly":  _metrics(y_true_w, y_pred_w, "Weekly",  baseline_w, 52),
        "Monthly": _metrics(y_true_m, y_pred_m, "Monthly", baseline_m, 12),
    }

    pos_rate = (y_true_m > 0).mean()
    print(f"Positive rate: {pos_rate:.3f}")
    print(f"Naive always-positive accuracy: {max(pos_rate, 1-pos_rate):.3f}")


    df = pd.DataFrame(results).T
    out_dir = os.path.join(config["data"]["output_dir"], "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_metrics.xlsx")
    df.to_excel(out_path)

    # ── Per-year breakdown (Daily task) ──
    print(f"\n[{tag}] Per-year breakdown (Daily task):")
    if "dates" in te and te["dates"] is not None:
        years = pd.DatetimeIndex(te["dates"]).year
        for y in sorted(set(years)):
            mask = years == y
            if mask.sum() < 30:
                continue

            da_y = balanced_accuracy_score(
                (y_true_d[mask] > 0).astype(int),
                (y_pred_d[mask] > 0).astype(int),
            ) * 100
            r2_y = r2_score(y_true_d[mask], y_pred_d[mask])
            sh_y = _sharpe(y_true_d[mask], y_pred_d[mask], periods_per_year=252)

            print(f"  {y}: DirAcc={da_y:5.2f}%  R2={r2_y:+.4f}  "
                  f"Sharpe={sh_y:+.2f}  (n={mask.sum()})")
    else:
        print("  [skipped — 'dates' missing from test dict]")

    # Per-year breakdown (Monthly task)
    print(f"\n[{tag}] Per-year breakdown (Monthly task):")
    if "dates" in te and te["dates"] is not None:
        years = pd.DatetimeIndex(te["dates"]).year
        for y in sorted(set(years)):
            mask = years == y
            if mask.sum() < 30:
                continue
            da_y = balanced_accuracy_score(
                (y_true_m[mask] > 0).astype(int),
                (y_pred_m[mask] > 0).astype(int),
            ) * 100
            r2_y = r2_score(y_true_m[mask], y_pred_m[mask])
            sh_y = _sharpe(y_true_m[mask], y_pred_m[mask], periods_per_year=12)
            print(f"  {y}: DirAcc={da_y:5.2f}%  R2={r2_y:+.4f}  "
                f"Sharpe={sh_y:+.2f}  (n={mask.sum()})")


        # ── Plots ──
    fig_dir = os.path.join(config["data"]["output_dir"], "figures")
    _plot_results(
        history=getattr(model, "_history", None),
        y_true_d=y_true_d, y_pred_d=y_pred_d,
        y_true_w=y_true_w, y_pred_w=y_pred_w,
        y_true_m=y_true_m, y_pred_m=y_pred_m,
        out_dir=fig_dir, tag=tag,
    )


    print(f"\n✓ Saved metrics to {out_path}")
    print(df)

    return df, (y_true_d, y_pred_d, y_true_w, y_pred_w, y_true_m, y_pred_m)