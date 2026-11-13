from __future__ import annotations

import keras


class ModelBuilder:
    # Only backbones that take raw 0-255 RGB input (preprocessing built in) belong here,
    # because the data pipeline feeds un-normalised pixels.
    BACKBONES = {
        "efficientnetv2": keras.applications.EfficientNetV2B0,
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

        backbone = self.BACKBONES[self.name](
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        backbone.trainable = False  # StrategyManager decides the final policy

        inputs = keras.Input(shape=input_shape, name="image")
        if strategy == "feature_extraction":
            x = backbone(inputs, training=False)
        else:
            x = backbone(inputs)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = keras.layers.Dropout(self.dropout, name="head_dropout")(x)
        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", dtype="float32", name="probabilities"
        )(x)

        model = keras.Model(inputs, outputs, name=f"{self.name}_classifier")
        return model, backbone