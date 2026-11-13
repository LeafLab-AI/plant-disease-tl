"""ConvNeXtV2 for Keras 3, loaded with the official timm / FAIR ImageNet-1k weights.

Neither keras.applications nor KerasHub ships ConvNeXtV2, so the architecture is rebuilt
here layer-for-layer from timm's ``convnextv2_*`` models and the pretrained safetensors
checkpoint is converted on the fly (numpy only, no PyTorch dependency).

Design choices that keep it a drop-in replacement for keras.applications backbones:

* The backbone takes raw 0-255 RGB (ImageNet mean/std normalisation is its first layer),
  like EfficientNetV2, because the data pipeline feeds un-normalised pixels.
* It returns the last-stage feature map (N, 7, 7, C), so ModelBuilder's GAP head and
  GradCAMExplainer work unchanged.
* Every op is its own functional layer, so StrategyManager's leaf-layer partial
  fine-tuning sees the same granularity as for keras.applications models.
* timm's post-pooling LayerNorm (``head.norm``) is not part of the feature map; it is
  exposed through :func:`build_head_norm` and placed after GAP by ModelBuilder.

Only pure ``keras.ops`` are used, so the port runs on the TensorFlow, JAX and PyTorch
backends (it is numerically checked against timm in tests/test_convnextv2.py).

Weights licence: the FAIR ConvNeXtV2 checkpoints are CC-BY-NC-4.0 (non-commercial use).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import keras
import numpy as np
from keras import layers, ops

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LN_EPS = 1e-6


@dataclass(frozen=True)
class Variant:
    depths: tuple[int, int, int, int]
    dims: tuple[int, int, int, int]
    hf_repo: str
    sha256: str


VARIANTS: dict[str, Variant] = {
    # ImageNet-1k only (FCMAE pre-train -> IN-1k fine-tune), matching the IN-1k weights
    # used for the other backbones. 28.6M params, 83.0% IN-1k top-1.
    "tiny": Variant(
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        hf_repo="timm/convnextv2_tiny.fcmae_ft_in1k",
        sha256="fdfb4edfa5abb3b4ea29ab17e6fae5c117dc6cc32ecfb1d0726a156f40a4c1a5",
    ),
}


# --------------------------------------------------------------------------- layers
@keras.saving.register_keras_serializable(package="convnextv2")
class ImageNetNormalization(layers.Layer):
    """Maps raw 0-255 RGB to ImageNet mean/std normalised input. Weight-free."""

    def call(self, x):
        x = ops.cast(x, self.compute_dtype)
        mean = ops.convert_to_tensor([m * 255.0 for m in IMAGENET_MEAN], dtype=self.compute_dtype)
        std = ops.convert_to_tensor([s * 255.0 for s in IMAGENET_STD], dtype=self.compute_dtype)
        return (x - mean) / std


@keras.saving.register_keras_serializable(package="convnextv2")
class GlobalResponseNorm(layers.Layer):
    """GRN from ConvNeXtV2 (channels-last): x + beta + gamma * x * N(x).

    N(x) is the per-channel spatial L2 norm divided by its mean over channels. It is
    computed in float32: a sum of squares over 56x56 positions overflows float16.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.gamma = self.add_weight(name="gamma", shape=(dim,), initializer="zeros")
        self.beta = self.add_weight(name="beta", shape=(dim,), initializer="zeros")

    def call(self, x):
        x32 = ops.cast(x, "float32")
        gx = ops.sqrt(ops.sum(ops.square(x32), axis=(1, 2), keepdims=True))
        nx = gx / (ops.mean(gx, axis=-1, keepdims=True) + self.epsilon)
        gamma = ops.cast(self.gamma, "float32")
        beta = ops.cast(self.beta, "float32")
        return ops.cast(x32 + beta + gamma * (x32 * nx), self.compute_dtype)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


@keras.saving.register_keras_serializable(package="convnextv2")
class DropPath(layers.Layer):
    """Stochastic depth on the residual branch (per-sample), active only when training."""

    def __init__(self, rate: float = 0.0, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if not training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        mask = ops.floor(keep + keras.random.uniform(shape, seed=self.seed_generator, dtype=x.dtype))
        return x / keep * mask

    def get_config(self):
        return {**super().get_config(), "rate": self.rate, "seed": self.seed}


# --------------------------------------------------------------------------- model
def _block(x, dim: int, drop_path: float, prefix: str):
    shortcut = x
    x = layers.DepthwiseConv2D(7, padding="same", name=f"{prefix}_conv_dw")(x)
    x = layers.LayerNormalization(epsilon=LN_EPS, name=f"{prefix}_norm")(x)
    x = layers.Dense(4 * dim, name=f"{prefix}_fc1")(x)
    x = layers.Activation("gelu", name=f"{prefix}_act")(x)  # exact (erf) GELU, as timm
    x = GlobalResponseNorm(name=f"{prefix}_grn")(x)
    x = layers.Dense(dim, name=f"{prefix}_fc2")(x)
    if drop_path > 0.0:
        x = DropPath(drop_path, name=f"{prefix}_drop_path")(x)
    return layers.Add(name=f"{prefix}_add")([shortcut, x])


def build_backbone(
    input_shape: tuple[int, int, int],
    variant: str = "tiny",
    weights: str | None = "imagenet",
    drop_path_rate: float = 0.2,
) -> keras.Model:
    """ConvNeXtV2 feature extractor: raw 0-255 RGB -> last-stage map (N, H/32, W/32, C).

    drop_path_rate follows timm (linearly increasing over blocks); 0.2 is the official
    ConvNeXtV2-T fine-tuning value and matches EfficientNetV2-B0's Keras default.
    """
    cfg = VARIANTS[variant]
    rates = np.linspace(0.0, drop_path_rate, sum(cfg.depths)).tolist()

    inputs = keras.Input(shape=input_shape, name="image")
    x = ImageNetNormalization(name="imagenet_norm")(inputs)
    x = layers.Conv2D(cfg.dims[0], 4, strides=4, name="stem_conv")(x)
    x = layers.LayerNormalization(epsilon=LN_EPS, name="stem_norm")(x)

    k = 0
    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        if i > 0:
            x = layers.LayerNormalization(epsilon=LN_EPS, name=f"stages_{i}_downsample_norm")(x)
            x = layers.Conv2D(dim, 2, strides=2, name=f"stages_{i}_downsample_conv")(x)
        for j in range(depth):
            x = _block(x, dim, rates[k], prefix=f"stages_{i}_blocks_{j}")
            k += 1

    model = keras.Model(inputs, x, name=f"convnextv2_{variant}")
    if weights == "imagenet":
        _load_backbone_weights(model, cfg, _state_dict(variant))
    elif weights is not None:
        model.load_weights(weights)
    return model


def build_head_norm(
    variant: str = "tiny", weights: str | None = "imagenet", name: str = "head_norm"
) -> layers.LayerNormalization:
    """timm's post-GAP LayerNorm (``head.norm``), pretrained. Applied to pooled features."""
    dim = VARIANTS[variant].dims[-1]
    norm = layers.LayerNormalization(epsilon=LN_EPS, name=name)
    norm.build((None, dim))
    if weights == "imagenet":
        sd = _state_dict(variant)
        norm.set_weights([sd["head.norm.weight"], sd["head.norm.bias"]])
    return norm


# --------------------------------------------------------------------------- weights
@functools.lru_cache(maxsize=2)
def _state_dict(variant: str) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    cfg = VARIANTS[variant]
    path = keras.utils.get_file(
        fname=f"{cfg.hf_repo.split('/')[-1]}.safetensors",
        origin=f"https://huggingface.co/{cfg.hf_repo}/resolve/main/model.safetensors",
        file_hash=cfg.sha256,
        cache_subdir="models",
    )
    return load_file(path)


def _load_backbone_weights(model: keras.Model, cfg: Variant, sd: dict[str, np.ndarray]) -> None:
    used: set[str] = set()

    def take(key: str) -> np.ndarray:
        used.add(key)
        return sd[key]

    def conv(key: str) -> np.ndarray:  # torch (O, I, kh, kw) -> keras (kh, kw, I, O)
        return take(key).transpose(2, 3, 1, 0)

    def set_(name: str, *arrays: np.ndarray) -> None:
        model.get_layer(name).set_weights(list(arrays))

    set_("stem_conv", conv("stem.0.weight"), take("stem.0.bias"))
    set_("stem_norm", take("stem.1.weight"), take("stem.1.bias"))

    for i, depth in enumerate(cfg.depths):
        if i > 0:
            t = f"stages.{i}.downsample"
            set_(f"stages_{i}_downsample_norm", take(f"{t}.0.weight"), take(f"{t}.0.bias"))
            set_(f"stages_{i}_downsample_conv", conv(f"{t}.1.weight"), take(f"{t}.1.bias"))
        for j in range(depth):
            t, p = f"stages.{i}.blocks.{j}", f"stages_{i}_blocks_{j}"
            # depthwise: torch (C, 1, kh, kw) -> keras (kh, kw, C, 1)
            set_(f"{p}_conv_dw", take(f"{t}.conv_dw.weight").transpose(2, 3, 0, 1),
                 take(f"{t}.conv_dw.bias"))
            set_(f"{p}_norm", take(f"{t}.norm.weight"), take(f"{t}.norm.bias"))
            set_(f"{p}_fc1", take(f"{t}.mlp.fc1.weight").T, take(f"{t}.mlp.fc1.bias"))
            set_(f"{p}_grn", take(f"{t}.mlp.grn.weight"), take(f"{t}.mlp.grn.bias"))
            set_(f"{p}_fc2", take(f"{t}.mlp.fc2.weight").T, take(f"{t}.mlp.fc2.bias"))

    leftover = {k for k in sd if not k.startswith("head.")} - used
    if leftover:
        raise ValueError(f"Unmapped ConvNeXtV2 checkpoint tensors: {sorted(leftover)[:5]} ...")
