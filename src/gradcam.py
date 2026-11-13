from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


class GradCAMExplainer:
    """Grad-CAM for classifiers built as  input -> backbone(sub-model) -> head layers.

    For a nested backbone the target feature map is the backbone's final output (exactly what
    the GAP layer consumes), which is architecture-agnostic. Flat models fall back to the
    last Conv2D layer.
    """

    def __init__(self, model: keras.Model):
        self.model = model
        self._forward, self.target_name = self._build_forward()

    def _build_forward(self):
        nested = next((l for l in self.model.layers if isinstance(l, keras.Model)), None)

        if nested is not None:
            idx = self.model.layers.index(nested)
            head = list(self.model.layers[idx + 1:])

            def forward(x):
                feats = nested(x, training=False)
                z = feats
                for layer in head:
                    z = layer(z)
                return feats, z

            return forward, nested.name

        conv = next(
            (l for l in reversed(self.model.layers) if isinstance(l, keras.layers.Conv2D)), None
        )
        if conv is None:
            raise RuntimeError("No backbone sub-model or Conv2D layer found for Grad-CAM.")

        # Calling the model directly keeps one connected graph (nested layer outputs would not).
        grad_model = keras.Model(self.model.input, [conv.output, self.model.output])

        def forward(x):
            feats, preds = grad_model(x, training=False)
            return feats, preds

        return forward, conv.name

    def generate_heatmap(self, img: np.ndarray, class_idx: int | None = None) -> np.ndarray:
        """Grad-CAM heatmap in [0, 1] with the same HxW as `img` (float RGB, 0-255).

        class_idx=None explains the predicted class.
        """
        x = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Watching the input records the whole forward pass even when every backbone
            # weight is frozen (feature extraction), where nothing would be tracked otherwise.
            tape.watch(x)
            feats_raw, preds = self._forward(x)
            idx = tf.argmax(preds[0]) if class_idx is None else tf.constant(class_idx, dtype=tf.int64)
            score = tf.gather(tf.cast(preds[0], tf.float32), idx)

        # Gradient must be taken w.r.t. the tensor that fed the head (before any dtype cast).
        grads = tf.cast(tape.gradient(score, feats_raw), tf.float32)
        feats = tf.cast(feats_raw, tf.float32)
        weights = tf.reduce_mean(grads, axis=(1, 2))

        cam = tf.reduce_sum(feats * weights[:, None, None, :], axis=-1)[0].numpy()
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]

    @staticmethod
    def overlay(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blends a heatmap onto an RGB image (0-255). Returns uint8 HxWx3."""
        from matplotlib import colormaps

        colored = colormaps["jet"](heatmap)[..., :3] * 255.0
        blended = (1 - alpha) * img.astype(np.float32) + alpha * colored
        return np.clip(blended, 0, 255).astype(np.uint8)