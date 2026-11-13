"""ConvNeXtV2 Keras port: structure, serialisation, strategies and parity with timm.

The parity test needs PyTorch + timm and downloads the pretrained checkpoint, so it is
skipped unless both are installed. Run it once on any backend, e.g.:
    KERAS_BACKEND=torch pytest tests/test_convnextv2.py
"""
from __future__ import annotations

import keras
import numpy as np
import pytest

from src import convnextv2
from src.strategies import StrategyManager

SHAPE = (224, 224, 3)


@pytest.fixture(scope="module")
def random_backbone() -> keras.Model:
    return convnextv2.build_backbone(SHAPE, weights=None)


def test_output_is_spatial_feature_map(random_backbone):
    x = np.random.uniform(0, 255, (2, *SHAPE)).astype("float32")
    y = keras.ops.convert_to_numpy(random_backbone(x, training=False))
    assert y.shape == (2, 7, 7, 768)


def test_param_count_matches_timm_tiny(random_backbone):
    # timm convnextv2_tiny: 28,635,496 params, of which the 1000-way head fc is
    # 768 * 1000 + 1000 and the head LayerNorm 2 * 768.
    assert random_backbone.count_params() == 28_635_496 - 769_000 - 1_536


def test_keras_roundtrip(random_backbone, tmp_path):
    path = tmp_path / "m.keras"
    random_backbone.save(path)
    restored = keras.models.load_model(path, compile=False)
    x = np.random.uniform(0, 255, (1, *SHAPE)).astype("float32")
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(random_backbone(x, training=False)),
        keras.ops.convert_to_numpy(restored(x, training=False)),
        atol=1e-5,
    )


def test_drop_path_only_when_training():
    layer = convnextv2.DropPath(0.5, seed=0)
    x = np.ones((64, 4, 4, 8), "float32")
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer(x, training=False)), x)
    y = keras.ops.convert_to_numpy(layer(x, training=True))
    per_sample = np.unique(y.reshape(64, -1), axis=1)
    assert per_sample.shape[1] == 1  # whole sample kept or dropped
    assert set(np.unique(y)) <= {0.0, 2.0}


def test_partial_ft_unfreezes_tail(random_backbone):
    StrategyManager.apply_strategy(random_backbone, "partial_ft", unfreeze_fraction=0.25)
    trainable = {w.path.split("/")[0] for w in random_backbone.trainable_weights}
    assert trainable and all(name.startswith("stages_3_") for name in trainable)
    frac = StrategyManager.summarize_trainable(random_backbone)["trainable_fraction"]
    assert 0.25 <= frac < 0.35
    random_backbone.trainable = False


def test_matches_timm_reference():
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")

    ref = timm.create_model("convnextv2_tiny.fcmae_ft_in1k", pretrained=True).eval()
    backbone = convnextv2.build_backbone(SHAPE, weights="imagenet")
    head_norm = convnextv2.build_head_norm()

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (2, *SHAPE)).astype("float32")
    x_ref = (x / 255.0 - np.array(convnextv2.IMAGENET_MEAN)) / np.array(convnextv2.IMAGENET_STD)

    with torch.no_grad():
        t_in = torch.from_numpy(x_ref.transpose(0, 3, 1, 2).astype("float32"))
        t_feat = ref.forward_features(t_in).numpy().transpose(0, 2, 3, 1)
        t_logits = ref(t_in).numpy()

    k_feat = keras.ops.convert_to_numpy(backbone(x, training=False))
    pooled = keras.ops.convert_to_numpy(head_norm(k_feat.mean(axis=(1, 2))))
    fc_w = ref.head.fc.weight.detach().numpy()
    fc_b = ref.head.fc.bias.detach().numpy()
    k_logits = pooled @ fc_w.T + fc_b

    np.testing.assert_allclose(k_feat, t_feat, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(k_logits, t_logits, atol=1e-3, rtol=1e-3)
    assert (k_logits.argmax(1) == t_logits.argmax(1)).all()
