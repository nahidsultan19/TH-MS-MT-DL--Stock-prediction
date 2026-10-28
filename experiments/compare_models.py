"""
Evaluate each saved model on the test set.
Rebuilds architecture from scratch (avoids Keras version issues)
and loads weights into the fresh model.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import r2_score, balanced_accuracy_score
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model

from src.utils import load_config
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features
from src.preprocessing import prepare_dataset
from src.models.cnn_lstm import build_branch as cnn_lstm_branch
from src.models.transformer import build_branch as transformer_branch


# ─────────────────────────────────────────────────────
# Rebuild the 3-head architecture (matches the earlier model)
# ─────────────────────────────────────────────────────
def build_3head(cfg, n_features):
    ds = cfg["features"]["daily_steps"]
    ws = cfg["features"]["weekly_steps"]
    ms = cfg["features"]["monthly_steps"]
    mcfg = cfg["models"]
    hybrid = cfg["hybrid"]

    d_in = Input((ds, n_features), name="Daily_Input")
    w_in = Input((ws, n_features), name="Weekly_Input")
    m_in = Input((ms, n_features), name="Monthly_Input")

    def one(arch, x, scale):
        if arch == "cnn_lstm":
            return cnn_lstm_branch(
                x,
                units=mcfg["cnn_lstm"]["lstm_units"],
                dropout=mcfg["cnn_lstm"]["dropout"],
                name=f"{scale}_cnn_lstm",
            )
        elif arch == "transformer":
            return transformer_branch(
                x,
                d_model=mcfg["transformer"]["d_model"],
                n_heads=mcfg["transformer"]["n_heads"],
                num_layers=mcfg["transformer"]["num_layers"],
                dropout=mcfg["transformer"]["dropout"],
                name=f"{scale}_transformer",
            )
        raise ValueError(arch)

    d_enc = one(hybrid["daily_branch"],   d_in, "daily")
    w_enc = one(hybrid["weekly_branch"],  w_in, "weekly")
    m_enc = one(hybrid["monthly_branch"], m_in, "monthly")

    merged = Concatenate(name="Fusion")([d_enc, w_enc, m_enc])
    shared = Dense(128, activation="relu", name="shared1")(merged)
    shared = Dropout(0.2, name="shared_drop")(shared)
    shared = Dense(64, activation="relu", name="shared2")(shared)

    hd = Dense(32, activation="relu", name="head_d")(shared)
    od = Dense(1, name="Daily_Return")(hd)
    hw = Dense(32, activation="relu", name="head_w")(shared)
    ow = Dense(1, name="Weekly_Return")(hw)
    hm = Dense(32, activation="relu", name="head_m")(shared)
    om = Dense(1, name="Monthly_Return")(hm)

    return Model(inputs=[d_in, w_in, m_in], outputs=[od, ow, om])


# ─────────────────────────────────────────────────────
# Load model: rebuild architecture, then load weights
# ─────────────────────────────────────────────────────
def load_any(model_path, cfg, n_features):
    """Try direct load first; fall back to rebuild-and-load-weights."""
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        return m, "direct"
    except Exception:
        m = build_3head(cfg, n_features)
        try:
            m.load_weights(model_path)
            return m, "rebuild+weights"
        except Exception as e:
            raise RuntimeError(f"Could not load {model_path}: {e}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    cfg = load_config("config/config.yaml")

    print("Loading data (may take a few minutes)...")
    df = load_raw_data(cfg["data"]["raw_path"])
    df = filter_valid_companies(df, cfg["data"]["min_records"])
    df = add_features(df)
    data = prepare_dataset(df, cfg, max_sequences=200_000)

    n_features = data["train"]["X_daily"].shape[-1]
    print(f"n_features = {n_features}")

    # Load target scaler
    scaler_path = os.path.join(cfg["data"]["output_dir"], "scalers", "hybrid_target.pkl")
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")

    te = data["test"]

    # Which models to test
    models_to_test = [
        "hybrid_full_backup",
        "hybrid",
        "all_cnn_lstm_full",
        "all_cnn_lstm",
        "hybrid_tcl",
        "hybrid_clt",
        "all_transformer",
    ]

    print(f"\n{'='*70}")
    print(f"  EVALUATING ALL MODELS — MONTHLY TASK")
    print(f"{'='*70}")

    results = []

    for name in models_to_test:
        path = f"results/models/{name}.h5"
        if not os.path.exists(path):
            print(f"{name:25s} → file not found")
            continue

        try:
            model, method = load_any(path, cfg, n_features)
        except Exception as e:
            print(f"{name:25s} → LOAD ERROR: {e}")
            continue

        try:
            preds = model.predict(
                [te["X_daily"], te["X_weekly"], te["X_monthly"]],
                verbose=0,
            )
        except Exception as e:
            print(f"{name:25s} → PREDICT ERROR: {e}")
            continue

        # 3-head vs 6-head — we only care about monthly (index 2) for both
        pred_m = np.asarray(preds[2]).flatten()

        # Inverse-transform monthly
        mn_m = scaler.data_min_[2]
        mx_m = scaler.data_max_[2]

        y_true_m = te["y"][:, 2] * (mx_m - mn_m) + mn_m
        y_pred_m = pred_m        * (mx_m - mn_m) + mn_m

        y_true_m = np.clip(y_true_m, -50, 50)
        y_pred_m = np.clip(y_pred_m, -50, 50)

        r2  = r2_score(y_true_m, y_pred_m)
        da  = balanced_accuracy_score(
            (y_true_m > 0).astype(int),
            (y_pred_m > 0).astype(int),
        ) * 100

        results.append((name, method, r2, da))
        print(f"{name:25s} → R2={r2:+.4f}  DirAcc={da:5.2f}%  [{method}]")

    print(f"\n{'='*70}")
    print(f"  RANKED (by Monthly R²)")
    print(f"{'='*70}")
    for name, method, r2, da in sorted(results, key=lambda x: -x[2]):
        print(f"  {name:25s}  R2={r2:+.4f}  DirAcc={da:5.2f}%")


if __name__ == "__main__":
    main()