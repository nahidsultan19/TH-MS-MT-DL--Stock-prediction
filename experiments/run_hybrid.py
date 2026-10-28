"""
Entry point for the hybrid model.

Run:  python -m experiments.run_hybrid
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.utils import load_config, set_seeds, ensure_dirs
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features
from src.preprocessing import prepare_dataset
from src.models.fusion import build_hybrid_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    cfg = load_config("config/config.yaml")
    set_seeds(42)
    ensure_dirs(cfg["data"]["output_dir"], "data/raw")

    print("Loading data...")
    df = load_raw_data(cfg["data"]["raw_path"])
    df = filter_valid_companies(df, cfg["data"]["min_records"])

    print("Adding technical indicators...")
    df = add_features(df)

    print("Building multi-scale, leakage-free sequences...")
    # data = prepare_dataset(df, cfg)
    data = prepare_dataset(df, cfg)

    n_features = data["train"]["X_daily"].shape[-1]
    print(f"n_features = {n_features}")

    model = build_hybrid_model(cfg, n_features, cfg["loss_weights"])
    model.summary()

    print("\nTraining...")
    train_model(model, data, cfg, tag="hybrid")

    print("\nEvaluating...")
    evaluate_model(model, data, cfg, tag="hybrid")


if __name__ == "__main__":
    main()