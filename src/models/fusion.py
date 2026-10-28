import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate)
from tensorflow.keras.models import Model

from .cnn_lstm import build_branch as cnn_lstm_branch
from .transformer import build_branch as transformer_branch
# from .mamba_branch import build_branch as mamba_branch


# BRANCHES = {
#     "cnn_lstm": cnn_lstm_branch,
#     "transformer": transformer_branch,
#     # "mamba": mamba_branch,
# }


# def _build_one(arch_name, x, cfg, scale_name):
#     if arch_name == "cnn_lstm":
#         return BRANCHES["cnn_lstm"](x,
#             units=cfg["cnn_lstm"]["lstm_units"],
#             dropout=cfg["cnn_lstm"]["dropout"],
#             name=f"{scale_name}_cnn_lstm")
#     if arch_name == "transformer":
#         return BRANCHES["transformer"](x,
#             d_model=cfg["transformer"]["d_model"],
#             n_heads=cfg["transformer"]["n_heads"],
#             num_layers=cfg["transformer"]["num_layers"],
#             dropout=cfg["transformer"]["dropout"],
#             name=f"{scale_name}_transformer")
#     # if arch_name == "mamba":
#     #     return BRANCHES["mamba"](x,
#     #         d_state=cfg["mamba"]["d_state"],
#     #         d_conv=cfg["mamba"]["d_conv"],
#     #         expand=cfg["mamba"]["expand"],
#     #         d_model=64,
#     #         name=f"{scale_name}_mamba")
#     raise ValueError(f"Unknown arch: {arch_name}")


# def build_hybrid_model(config, n_features, loss_weights):
#     cfg = config["models"]
#     hybrid = config["hybrid"]
#     ds = config["features"]["daily_steps"]
#     ws = config["features"]["weekly_steps"]
#     ms = config["features"]["monthly_steps"]

#     d_in = Input((ds, n_features), name="Daily_Input")
#     w_in = Input((ws, n_features), name="Weekly_Input")
#     m_in = Input((ms, n_features), name="Monthly_Input")

#     d_enc = _build_one(hybrid["daily_branch"],   d_in, cfg, "daily")
#     w_enc = _build_one(hybrid["weekly_branch"],  w_in, cfg, "weekly")
#     m_enc = _build_one(hybrid["monthly_branch"], m_in, cfg, "monthly")

#     merged = Concatenate(name="Fusion")([d_enc, w_enc, m_enc])
#     shared = Dense(128, activation="relu", name="shared1")(merged)
#     shared = Dropout(0.2, name="shared_drop")(shared)
#     shared = Dense(64, activation="relu", name="shared2")(shared)

#     # Multi-task heads
#     head_d = Dense(32, activation="relu", name="head_d")(shared)
#     out_d  = Dense(1, name="Daily_Return")(head_d)

#     head_w = Dense(32, activation="relu", name="head_w")(shared)
#     out_w  = Dense(1, name="Weekly_Return")(head_w)

#     head_m = Dense(32, activation="relu", name="head_m")(shared)
#     out_m  = Dense(1, name="Monthly_Return")(head_m)

#     model = Model(inputs=[d_in, w_in, m_in], outputs=[out_d, out_w, out_m])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
#         loss={"Daily_Return": "mse",
#               "Weekly_Return": "mse",
#               "Monthly_Return": "mse"},
#         loss_weights={"Daily_Return":   loss_weights["daily"],
#                       "Weekly_Return":  loss_weights["weekly"],
#                       "Monthly_Return": loss_weights["monthly"]},
#         metrics={"Daily_Return": ["mae"],
#                  "Weekly_Return": ["mae"],
#                  "Monthly_Return": ["mae"]},
#     )
#     return model

BRANCHES = {
    "cnn_lstm": cnn_lstm_branch,
    "transformer": transformer_branch,
    # "mamba": mamba_branch,
}


def _build_one(arch_name, x, cfg, scale_name):
    if arch_name == "cnn_lstm":
        return BRANCHES["cnn_lstm"](x,
            units=cfg["cnn_lstm"]["lstm_units"],
            dropout=cfg["cnn_lstm"]["dropout"],
            name=f"{scale_name}_cnn_lstm")
    if arch_name == "transformer":
        return BRANCHES["transformer"](x,
            d_model=cfg["transformer"]["d_model"],
            n_heads=cfg["transformer"]["n_heads"],
            num_layers=cfg["transformer"]["num_layers"],
            dropout=cfg["transformer"]["dropout"],
            name=f"{scale_name}_transformer")
    # if arch_name == "mamba":
    #     return BRANCHES["mamba"](x,
    #         d_state=cfg["mamba"]["d_state"],
    #         d_conv=cfg["mamba"]["d_conv"],
    #         expand=cfg["mamba"]["expand"],
    #         d_model=64,
    #         name=f"{scale_name}_mamba")
    raise ValueError(f"Unknown arch: {arch_name}")


def build_hybrid_model(config, n_features, loss_weights):
    cfg = config["models"]
    hybrid = config["hybrid"]
    ds = config["features"]["daily_steps"]
    ws = config["features"]["weekly_steps"]
    ms = config["features"]["monthly_steps"]

    # ── Inputs ──
    d_in = Input((ds, n_features), name="Daily_Input")
    w_in = Input((ws, n_features), name="Weekly_Input")
    m_in = Input((ms, n_features), name="Monthly_Input")

    # ── Scale-specific encoders ──
    d_enc = _build_one(hybrid["daily_branch"],   d_in, cfg, "daily")
    w_enc = _build_one(hybrid["weekly_branch"],  w_in, cfg, "weekly")
    m_enc = _build_one(hybrid["monthly_branch"], m_in, cfg, "monthly")

    # ── Fusion + shared representation ──
    merged = Concatenate(name="Fusion")([d_enc, w_enc, m_enc])
    shared = Dense(128, activation="relu", name="shared1")(merged)
    shared = Dropout(0.2, name="shared_drop")(shared)
    shared = Dense(64, activation="relu", name="shared2")(shared)

    # ── Regression heads (3) ──
    head_d_reg = Dense(32, activation="relu", name="head_d_reg")(shared)
    out_d_reg  = Dense(1, name="Daily_Return")(head_d_reg)

    head_w_reg = Dense(32, activation="relu", name="head_w_reg")(shared)
    out_w_reg  = Dense(1, name="Weekly_Return")(head_w_reg)

    head_m_reg = Dense(32, activation="relu", name="head_m_reg")(shared)
    out_m_reg  = Dense(1, name="Monthly_Return")(head_m_reg)

    # ── Classification heads (3) — sigmoid outputs ──
    head_d_cls = Dense(32, activation="relu", name="head_d_cls")(shared)
    out_d_cls  = Dense(1, activation="sigmoid", name="Daily_Direction")(head_d_cls)

    head_w_cls = Dense(32, activation="relu", name="head_w_cls")(shared)
    out_w_cls  = Dense(1, activation="sigmoid", name="Weekly_Direction")(head_w_cls)

    head_m_cls = Dense(32, activation="relu", name="head_m_cls")(shared)
    out_m_cls  = Dense(1, activation="sigmoid", name="Monthly_Direction")(head_m_cls)

    # ── Assemble model ──
    model = Model(
        inputs=[d_in, w_in, m_in],
        outputs=[
            out_d_reg, out_w_reg, out_m_reg,
            out_d_cls, out_w_cls, out_m_cls,
        ],
    )

    # ── Loss weights ──
    reg_w = {
        "Daily_Return":   loss_weights["daily"],
        "Weekly_Return":  loss_weights["weekly"],
        "Monthly_Return": loss_weights["monthly"],
    }

    cls_total = config.get("training", {}).get("cls_loss_weight", 0.3)
    cls_each  = cls_total / 3.0

    combined_weights = {
        "Daily_Return":      reg_w["Daily_Return"],
        "Weekly_Return":     reg_w["Weekly_Return"],
        "Monthly_Return":    reg_w["Monthly_Return"],
        "Daily_Direction":   cls_each,
        "Weekly_Direction":  cls_each,
        "Monthly_Direction": cls_each,
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["training"]["learning_rate"]),
        loss={
            "Daily_Return":      "mse",
            "Weekly_Return":     "mse",
            "Monthly_Return":    "mse",
            "Daily_Direction":   "binary_crossentropy",
            "Weekly_Direction":  "binary_crossentropy",
            "Monthly_Direction": "binary_crossentropy",
        },
        loss_weights=combined_weights,
        metrics={
            "Daily_Return":      ["mae"],
            "Weekly_Return":     ["mae"],
            "Monthly_Return":    ["mae"],
            "Daily_Direction":   ["accuracy"],
            "Weekly_Direction":  ["accuracy"],
            "Monthly_Direction": ["accuracy"],
        },
    )
    return model