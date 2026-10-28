import tensorflow as tf
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_config
from src.models.fusion import build_hybrid_model
from src.models.cnn_lstm import build_branch as cnn_lstm_branch
from src.models.transformer import build_branch as transformer_branch
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def build_3head_hybrid(cfg, n_features):
    """Rebuild the 3-head hybrid model from earlier code."""
    ds = cfg["features"]["daily_steps"]
    ws = cfg["features"]["weekly_steps"]
    ms = cfg["features"]["monthly_steps"]
    mcfg = cfg["models"]
    hybrid = cfg["hybrid"]

    d_in = Input((ds, n_features), name="Daily_Input")
    w_in = Input((ws, n_features), name="Weekly_Input")
    m_in = Input((ms, n_features), name="Monthly_Input")

    def build_one(arch, x, scale):
        if arch == "cnn_lstm":
            return cnn_lstm_branch(
                x,
                units=mcfg["cnn_lstm"]["lstm_units"],
                dropout=mcfg["cnn_lstm"]["dropout"],
                name=f"{scale}_cnn_lstm",
            )
        if arch == "transformer":
            return transformer_branch(
                x,
                d_model=mcfg["transformer"]["d_model"],
                n_heads=mcfg["transformer"]["n_heads"],
                num_layers=mcfg["transformer"]["num_layers"],
                dropout=mcfg["transformer"]["dropout"],
                name=f"{scale}_transformer",
            )
        raise ValueError(arch)

    d_enc = build_one(hybrid["daily_branch"],   d_in, "daily")
    w_enc = build_one(hybrid["weekly_branch"],  w_in, "weekly")
    m_enc = build_one(hybrid["monthly_branch"], m_in, "monthly")

    from tensorflow.keras.layers import Concatenate, Dense, Dropout
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

    return Model(inputs=[d_in, w_in, m_in], outputs=[od, ow, om], name="hybrid_3head")


def main():
    cfg = load_config("config/config.yaml")
    n_features = 20  # your feature count

    # For each model file, rebuild with correct architecture then load weights
    for name, arch in [
        ("hybrid_full_backup", "3head"),
        ("hybrid",             "6head"),
        ("hybrid_full",        "6head"),
        ("all_cnn_lstm_full",  "3head_loaded_ok"),
    ]:
        path = f"results/models/{name}.h5"
        try:
            if arch == "3head":
                m = build_3head_hybrid(cfg, n_features)
            elif arch == "6head":
                m = build_hybrid_model(cfg, n_features, cfg["loss_weights"])
            else:
                m = tf.keras.models.load_model(path, compile=False)

            if arch != "3head_loaded_ok":
                m.load_weights(path)

            outputs = [o.name.split("/")[0] for o in m.outputs]
            params = m.count_params()
            print(f"{name:25s} → {len(outputs)} outputs, {params:,} params")
            for o in outputs:
                print(f"    {o}")
        except Exception as e:
            print(f"{name:25s} → ERROR: {e}")


if __name__ == "__main__":
    main()