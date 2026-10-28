import os
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# def train_model(model, data, config, tag="model"):
#     tr, te = data["train"], data["test"]

#     y_tr = {"Daily_Return":   tr["y"][:, 0],
#             "Weekly_Return":  tr["y"][:, 1],
#             "Monthly_Return": tr["y"][:, 2]}

#     cb = [
#         EarlyStopping(monitor="val_loss", patience=config["training"]["patience"],
#                       restore_best_weights=True, verbose=1),
#         ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
#                           min_lr=1e-6, verbose=1),
#     ]

#     history = model.fit(
#         [tr["X_daily"], tr["X_weekly"], tr["X_monthly"]],
#         y_tr,
#         epochs=config["training"]["epochs"],
#         batch_size=config["training"]["batch_size"],
#         validation_split=config["training"]["val_split"],
#         callbacks=cb,
#         verbose=1,
#     )

#     out_dir = os.path.join(config["data"]["output_dir"], "models")
#     os.makedirs(out_dir, exist_ok=True)
#     model.save(os.path.join(out_dir, f"{tag}.h5"))

#     # Save scalers
#     sc_dir = os.path.join(config["data"]["output_dir"], "scalers")
#     os.makedirs(sc_dir, exist_ok=True)
#     if data["scalers"] is not None:
#         for k, v in data["scalers"].items():
#             joblib.dump(v, os.path.join(sc_dir, f"{tag}_{k}.pkl"))

#     return history

def train_model(model, data, config, tag="model"):
    tr, te = data["train"], data["test"]

    # ── Regression targets ──
    y_reg = {
        "Daily_Return":   tr["y"][:, 0],
        "Weekly_Return":  tr["y"][:, 1],
        "Monthly_Return": tr["y"][:, 2],
    }

    # ── Classification targets (direction labels) ──
    y_cls = {
        "Daily_Direction":   tr["y_dir"][:, 0],
        "Weekly_Direction":  tr["y_dir"][:, 1],
        "Monthly_Direction": tr["y_dir"][:, 2],
    }

    # ── Combine into one dict for fit() ──
    y_all = {**y_reg, **y_cls}

    # ── Manual validation split to keep X and y aligned ──
    val_split = config["training"]["val_split"]
    n = len(tr["X_daily"])
    n_val = int(n * val_split)
    n_train = n - n_val

    X_train = [tr["X_daily"][:n_train], tr["X_weekly"][:n_train], tr["X_monthly"][:n_train]]
    X_val   = [tr["X_daily"][n_train:], tr["X_weekly"][n_train:], tr["X_monthly"][n_train:]]

    y_train = {k: v[:n_train] for k, v in y_all.items()}
    y_val   = {k: v[n_train:] for k, v in y_all.items()}

    # ── Callbacks ──
    cb = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    # ── Train ──
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        callbacks=cb,
        verbose=1,
    )

    # ── Save model ──
    out_dir = os.path.join(config["data"]["output_dir"], "models")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{tag}.h5"))

    # ── Save scalers ──
    sc_dir = os.path.join(config["data"]["output_dir"], "scalers")
    os.makedirs(sc_dir, exist_ok=True)
    if data["scalers"] is not None:
        for k, v in data["scalers"].items():
            joblib.dump(v, os.path.join(sc_dir, f"{tag}_{k}.pkl"))

    # Attach history to model so evaluate_model can plot it
    model._history = history

    return history