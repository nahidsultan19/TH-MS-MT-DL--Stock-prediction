"""
Remove one branch at a time. Example: hybrid with daily=cnn_lstm,
weekly=transformer, monthly=mamba vs. single-arch variants.
"""
import sys, os, copy
sys.path.insert(0, os.path.abspath("."))

from src.utils import load_config, set_seeds
from src.data_loader import load_raw_data, filter_valid_companies
from src.feature_engineering import add_features
from src.preprocessing import prepare_dataset
from src.models.fusion import build_hybrid_model
from src.train import train_model
from src.evaluate import evaluate_model


# VARIANTS = {
#     "all_cnn_lstm":  ("cnn_lstm", "cnn_lstm", "cnn_lstm"),
#     "all_transformer": ("transformer", "transformer", "transformer"),
#     "all_mamba":     ("mamba", "mamba", "mamba"),
#     "hybrid_ctm":    ("cnn_lstm", "transformer", "mamba"),
#     "hybrid_mtc":    ("mamba", "transformer", "cnn_lstm"),
# }

VARIANTS = {
    "all_cnn_lstm":     ("cnn_lstm",    "cnn_lstm",    "cnn_lstm"),
    "all_transformer":  ("transformer", "transformer", "transformer"),
    "hybrid_clt":       ("cnn_lstm",    "cnn_lstm",    "transformer"),
    "hybrid_tcl":       ("transformer", "cnn_lstm",    "cnn_lstm"),
    "hybrid_full":      ("cnn_lstm",    "transformer", "transformer"),
}


def main():
    base_cfg = load_config("config/config.yaml")
    set_seeds(42)

    df = load_raw_data(base_cfg["data"]["raw_path"])
    df = filter_valid_companies(df, base_cfg["data"]["min_records"])
    df = add_features(df)
    data = prepare_dataset(df, base_cfg, max_sequences=200_000)
    n_features = data["train"]["X_daily"].shape[-1]

    for name, (d_arch, w_arch, m_arch) in VARIANTS.items():
        print(f"\n{'='*60}\nVariant: {name}  ({d_arch} | {w_arch} | {m_arch})\n{'='*60}")
        cfg = copy.deepcopy(base_cfg)
        cfg["hybrid"]["daily_branch"] = d_arch
        cfg["hybrid"]["weekly_branch"] = w_arch
        cfg["hybrid"]["monthly_branch"] = m_arch

        try:
            model = build_hybrid_model(cfg, n_features, cfg["loss_weights"])
            train_model(model, data, cfg, tag=name)
            evaluate_model(model, data, cfg, tag=name)
        except Exception as e:
            print(f"Variant {name} failed: {e}")
            continue


if __name__ == "__main__":
    main()