import tensorflow as tf
from tensorflow.keras.layers import (Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)


def build_branch(x, d_model=64, n_heads=4, num_layers=2,
                 dropout=0.1, name="transformer"):
    """Transformer encoder branch. x: (batch, steps, feats)."""
    # Project to d_model
    x = Dense(d_model, name=f"{name}_proj")(x)

    for i in range(num_layers):
        # Self-attention
        attn = MultiHeadAttention(num_heads=n_heads,
                                  key_dim=d_model // n_heads,
                                  name=f"{name}_mha{i}")(x, x)
        x = LayerNormalization(name=f"{name}_ln1_{i}")(x + attn)
        # FFN
        ffn = Dense(d_model * 2, activation="relu", name=f"{name}_ffn1_{i}")(x)
        ffn = Dense(d_model, name=f"{name}_ffn2_{i}")(ffn)
        ffn = Dropout(dropout, name=f"{name}_drop_{i}")(ffn)
        x = LayerNormalization(name=f"{name}_ln2_{i}")(x + ffn)

    # Aggregate over time
    x = GlobalAveragePooling1D(name=f"{name}_gap")(x)
    return x