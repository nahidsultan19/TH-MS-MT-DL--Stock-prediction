"""Re-evaluate an already-trained model without retraining."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import tensorflow as tf

from src.utils import load_config
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features
from src.preprocessing import prepare_dataset
from src.models.fusion import build_hybrid_model
from src.evaluate import evaluate_model


# def main():
#     cfg = load_config("config/config.yaml")

#     print("Rebuilding data pipeline (same seeds = same test set)...")
#     df = load_raw_data(cfg["data"]["raw_path"])
#     df = filter_valid_companies(df, cfg["data"]["min_records"])
#     df = add_features(df)
#     data = prepare_dataset(df, cfg)

#     # Rebuild architecture, then load weights
#     n_features = data["train"]["X_daily"].shape[-1]
#     model = build_hybrid_model(cfg, n_features, cfg["loss_weights"])

#     model_path = os.path.join(cfg["data"]["output_dir"], "models", "hybrid.h5")
#     print(f"Loading weights from: {model_path}")
#     model.load_weights(model_path)

#     # Load target scaler
#     sc_path = os.path.join(cfg["data"]["output_dir"], "scalers", "hybrid_target.pkl")
#     scaler = joblib.load(sc_path)
#     data["scalers"] = {"target": scaler}

   

#     # Evaluate
#     # evaluate_model(model, data, cfg, tag="hybrid_reeval")
#     _, (y_true_d, y_pred_d, y_true_w, y_pred_w, y_true_m, y_pred_m) = evaluate_model(
#     model, data, cfg, tag="hybrid_reeval"
#     )


# if __name__ == "__main__":
#     main()

def main():
    cfg = load_config("config/config.yaml")

    print("Rebuilding data pipeline (same seeds = same test set)...")
    df = load_raw_data(cfg["data"]["raw_path"])
    df = filter_valid_companies(df, cfg["data"]["min_records"])
    df = add_features(df)
    data = prepare_dataset(df, cfg)

    n_features = data["train"]["X_daily"].shape[-1]
    model = build_hybrid_model(cfg, n_features, cfg["loss_weights"])

    model_path = os.path.join(cfg["data"]["output_dir"], "models", "hybrid.h5")
    print(f"Loading weights from: {model_path}")
    model.load_weights(model_path)

    sc_path = os.path.join(cfg["data"]["output_dir"], "scalers", "hybrid_target.pkl")
    scaler = joblib.load(sc_path)
    data["scalers"] = {"target": scaler}

    evaluate_model(model, data, cfg, tag="hybrid_reeval")


if __name__ == "__main__":
    main()