import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D,
                                     LSTM, Dense, Dropout, Flatten)


def build_branch(x, units=128, dropout=0.2, name="cnn_lstm"):
    """Returns the encoded vector for one scale."""
    x = Conv1D(units, 3, activation="relu", padding="same", name=f"{name}_conv1")(x)
    x = Conv1D(units // 2, 3, activation="relu", padding="same", name=f"{name}_conv2")(x)
    x = MaxPooling1D(2, name=f"{name}_pool")(x)
    x = LSTM(units, return_sequences=True, name=f"{name}_lstm1")(x)
    x = Dropout(dropout, name=f"{name}_drop1")(x)
    x = LSTM(units // 2, name=f"{name}_lstm2")(x)
    x = Dropout(dropout, name=f"{name}_drop2")(x)
    return x