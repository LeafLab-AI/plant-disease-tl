import keras

class ModelBuilder:
    def __init__(self, name: str, num_classes: int, img_size: int, dropout: float = 0.2):
        self.name = name
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout = dropout

    def build(self) -> tuple[keras.Model, keras.Model]:
        """Builds a transfer learning classifier and returns (full_model, backbone)."""
        if self.name != "efficientnetv2":
            raise NotImplementedError(f"Model '{self.name}' is not supported yet.")

        input_shape = (self.img_size, self.img_size, 3)

        backbone = keras.applications.EfficientNetV2B0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        backbone.trainable = False

        inputs = keras.Input(shape=input_shape, name="image")
        x = backbone(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = keras.layers.Dropout(self.dropout, name="head_dropout")(x)
        outputs = keras.layers.Dense(
            self.num_classes, activation="softmax", dtype="float32", name="probabilities"
        )(x)

        model = keras.Model(inputs, outputs, name=f"{self.name}_classifier")
        return model, backbone