from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import keras

from src import convnextv2


@dataclass(frozen=True)
class BackboneSpec:
    """How to build a pretrained backbone, plus an optional pretrained post-GAP norm.

    ``factory(input_shape)`` must return a model that takes raw 0-255 RGB (preprocessing
    built in) and outputs a 4D feature map, because the data pipeline feeds
    un-normalised pixels and Grad-CAM needs the spatial map.
    """

    factory: Callable[[tuple[int, int, int]], keras.Model]
    head_norm: Callable[[], keras.layers.Layer] | None = None


def _efficientnetv2(input_shape: tuple[int, int, int]) -> keras.Model:
    return keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )


def _convnextv2(input_shape: tuple[int, int, int]) -> keras.Model:
    return convnextv2.build_backbone(input_shape, variant="tiny", weights="imagenet")


class ModelBuilder:
    BACKBONES: dict[str, BackboneSpec] = {
        "efficientnetv2": BackboneSpec(_efficientnetv2),
        # ConvNeXtV2-Tiny, official IN-1k weights (see src/convnextv2.py). Its pretrained
        # head LayerNorm is kept after GAP, as in the original architecture.
        "convnextv2": BackboneSpec(
            _convnextv2, head_norm=lambda: convnextv2.build_head_norm("tiny", "imagenet")
        ),
    }

    def __init__(self, name: str, num_classes: int, img_size: int, dropout: float = 0.2):
        if name not in self.BACKBONES:
            raise NotImplementedError(
                f"Model '{name}' is not supported yet. Available: {sorted(self.BACKBONES)}"
            )
        self.name = name
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout = dropout

    def build(self, strategy: str = "feature_extraction") -> tuple[keras.Model, keras.Model]:
        """Builds a transfer-learning classifier and returns (full_model, backbone).

        The training flag is handled per strategy:
        * feature_extraction: backbone is *always* called with training=False, so BatchNorm
          statistics and stochastic depth never change while it is frozen.
        * partial_ft / full_ft: the flag is inherited from fit(), so unfrozen blocks keep
          their stochastic depth. BatchNorm layers are put in inference mode by
          StrategyManager (trainable=False), not by a hard-coded flag.
        """
        input_shape = (self.img_size, self.img_size, 3)
        spec = self.BACKBONES[self.name]

        backbone = spec.factory(input_shape)
        backbone.trainable = False  # StrategyManager decides the final policy

        inputs = keras.Input(shape=input_shape, name="image")
        if strategy == "feature_extraction":
            x = backbone(inputs, training=False)
        else:
            x = backbone(inputs)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        if spec.head_norm is not None:
            x = spec.head_norm()(x)
        x = keras.layers.Dropout(self.dropout, name="head_dropout")(x)
        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", dtype="float32", name="probabilities"
        )(x)

        model = keras.Model(inputs, outputs, name=f"{self.name}_classifier")
        return model, backbone
