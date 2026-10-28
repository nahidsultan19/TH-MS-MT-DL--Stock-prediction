"""
Mamba branch. Requires: pip install mamba-ssm causal-conv1d
Falls back with a clear error if not installed.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


def build_branch(x, d_state=16, d_conv=4, expand=2, d_model=64, name="mamba"):
    if not _MAMBA_AVAILABLE:
        raise ImportError(
            "Mamba not installed. Run: pip install mamba-ssm causal-conv1d\n"
            "Or disable the mamba branch in config.yaml."
        )
    # Project to d_model
    x = Dense(d_model, name=f"{name}_proj")(x)

    # Mamba expects a torch tensor — use a Lambda wrapper with TF<->torch bridge
    # (Simplest: replace this branch with a compatible implementation if you
    #  prefer pure TF. Here we call mamba_ssm via tf.numpy_function.)
    def _mamba_call(inputs):
        import torch
        model = Mamba(d_model=d_model, d_state=d_state,
                      d_conv=d_conv, expand=expand).cuda()
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(inputs).cuda().float()
            out = model(t).cpu().numpy()
        return out

    x = tf.numpy_function(_mamba_call, [x], tf.float32)
    x.set_shape([None, None, d_model])  # (batch, steps, d_model)
    x = GlobalAveragePooling1D(name=f"{name}_gap")(x)
    return x